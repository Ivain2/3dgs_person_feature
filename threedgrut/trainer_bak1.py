# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.utils.data
from addict import Dict
from omegaconf import DictConfig, OmegaConf
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import threedgrut.datasets as datasets
from threedgrut.datasets.protocols import BoundedMultiViewDataset
from threedgrut.datasets.utils import DEFAULT_DEVICE, MultiEpochsDataLoader
from threedgrut.export.ingp_exporter import INGPExporter
from threedgrut.export.ply_exporter import PLYExporter
from threedgrut.export.usdz_exporter import USDZExporter
from threedgrut.model.losses import ssim
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.optimizers import SelectiveAdam
from threedgrut.render import Renderer
from threedgrut.strategy.base import BaseStrategy
from threedgrut.utils.gui import GUI
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import check_step_condition, create_summary_writer, jet_map
from threedgrut.utils.timer import CudaTimer


class Trainer3DGRUT:
    model: MixtureOfGaussians
    train_dataset: BoundedMultiViewDataset
    val_dataset: BoundedMultiViewDataset
    train_dataloader: torch.utils.data.DataLoader
    val_dataloader: torch.utils.data.DataLoader
    scene_extent: float = 1.0
    scene_bbox: tuple[torch.Tensor, torch.Tensor]
    strategy: BaseStrategy
    gui = None
    criterions: Dict
    tracking: Dict

    @staticmethod
    def create_from_checkpoint(resume: str, conf: DictConfig):
        conf.resume = resume
        conf.import_ingp.enabled = False
        conf.import_ply.enabled = False
        return Trainer3DGRUT(conf)

    @staticmethod
    def create_from_ingp(ply_path: str, conf: DictConfig):
        conf.resume = ""
        conf.import_ingp.enabled = True
        conf.import_ingp.path = ply_path
        conf.import_ply.enabled = False
        return Trainer3DGRUT(conf)

    @staticmethod
    def create_from_ply(ply_path: str, conf: DictConfig):
        conf.resume = ""
        conf.import_ingp.enabled = False
        conf.import_ply.enabled = True
        conf.import_ply.path = ply_path
        return Trainer3DGRUT(conf)

    @torch.cuda.nvtx.range("setup-trainer")
    def __init__(self, conf: DictConfig, device=None):
        self.conf = conf
        self.device = device if device is not None else DEFAULT_DEVICE
        self.global_step = 0
        self.n_iterations = conf.n_iterations
        self.n_epochs = 0
        self.val_frequency = conf.val_frequency

        logger.log_rule("Load Datasets")
        self.init_dataloaders(conf)
        self.init_scene_extents(self.train_dataset)
        logger.log_rule("Initialize Model")
        self.init_model(conf, self.scene_extent)
        self.init_densification_and_pruning_strategy(conf)
        logger.log_rule("Setup Model Weights & Training")
        self.init_metrics()
        self.setup_training(conf, self.model, self.train_dataset)
        self.init_experiments_tracking(conf)
        self.init_gui(conf, self.model, self.train_dataset, self.val_dataset, self.scene_bbox)

    def init_dataloaders(self, conf: DictConfig):
        from threedgrut.datasets.utils import configure_dataloader_for_platform

        train_dataset, val_dataset = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)
        train_dataloader_kwargs = configure_dataloader_for_platform(
            {
                "num_workers": conf.num_workers,
                "batch_size": 1,
                "shuffle": True,
                "pin_memory": True,
                "persistent_workers": True if conf.num_workers > 0 else False,
            }
        )

        val_dataloader_kwargs = configure_dataloader_for_platform(
            {
                "num_workers": conf.num_workers,
                "batch_size": 1,
                "shuffle": False,
                "pin_memory": True,
                "persistent_workers": True if conf.num_workers > 0 else False,
            }
        )

        train_dataloader = MultiEpochsDataLoader(train_dataset, **train_dataloader_kwargs)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, **val_dataloader_kwargs)

        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.val_dataset = val_dataset
        self.val_dataloader = val_dataloader

    def teardown_dataloaders(self):
        if self.train_dataloader is not None:
            del self.train_dataloader
        if self.val_dataloader is not None:
            del self.val_dataloader
        if self.train_dataset is not None:
            del self.train_dataset
        if self.val_dataset is not None:
            del self.val_dataset

    def init_scene_extents(self, train_dataset: BoundedMultiViewDataset) -> None:
        scene_bbox: tuple[torch.Tensor, torch.Tensor]
        scene_extent = train_dataset.get_scene_extent()
        scene_bbox = train_dataset.get_scene_bbox()
        self.scene_extent = scene_extent
        self.scene_bbox = scene_bbox

    def init_model(self, conf: DictConfig, scene_extent=None) -> None:
        self.model = MixtureOfGaussians(conf, scene_extent=scene_extent)

    def init_densification_and_pruning_strategy(self, conf: DictConfig) -> None:
        assert self.model is not None
        match self.conf.strategy.method:
            case "GSStrategy":
                from threedgrut.strategy.gs import GSStrategy
                self.strategy = GSStrategy(conf, self.model)
                logger.info("🔆 Using GS strategy")
            case "MCMCStrategy":
                from threedgrut.strategy.mcmc import MCMCStrategy
                self.strategy = MCMCStrategy(conf, self.model)
                logger.info("🔆 Using MCMC strategy")
            case _:
                raise ValueError(f"unrecognized model.strategy {conf.strategy.method}")

    def setup_training(self, conf: DictConfig, model: MixtureOfGaussians, train_dataset: BoundedMultiViewDataset):
        if conf.resume:
            logger.info(f"🤸 Loading a pretrained checkpoint from {conf.resume}!")
            checkpoint = torch.load(conf.resume)
            model.init_from_checkpoint(checkpoint)
            self.strategy.init_densification_buffer(checkpoint)
            global_step = checkpoint["global_step"]
        elif conf.import_ingp.enabled:
            ingp_path = (
                conf.import_ingp.path
                if conf.import_ingp.path
                else f"{conf.out_dir}/{conf.experiment_name}/export_last.inpg"
            )
            logger.info(f"Loading a pretrained ingp model from {ingp_path}!")
            model.init_from_ingp(ingp_path)
            self.strategy.init_densification_buffer()
            model.build_acc()
            global_step = conf.import_ingp.init_global_step
        elif conf.import_ply.enabled:
            ply_path = (
                conf.import_ply.path
                if conf.import_ply.path
                else f"{conf.out_dir}/{conf.experiment_name}/export_last.ply"
            )
            logger.info(f"Loading a ply model from {ply_path}!")
            model.init_from_ply(ply_path)
            self.strategy.init_densification_buffer()
            model.build_acc()
            global_step = conf.import_ply.init_global_step
        else:
            logger.info(f"🤸 Initiating new 3dgrut training..")
            match conf.initialization.method:
                case "random":
                    model.init_from_random_point_cloud(
                        num_gaussians=conf.initialization.num_gaussians,
                        xyz_max=conf.initialization.xyz_max,
                        xyz_min=conf.initialization.xyz_min,
                    )
                case "colmap":
                    observer_points = torch.tensor(
                        train_dataset.get_observer_points(), dtype=torch.float32, device=self.device
                    )
                    model.init_from_colmap(conf.path, observer_points)
                case "fused_point_cloud":
                    observer_points = torch.tensor(
                        train_dataset.get_observer_points(), dtype=torch.float32, device=self.device
                    )
                    ply_path = conf.initialization.fused_point_cloud_path
                    logger.info(f"Initializing from accumulated point cloud: {ply_path}")
                    model.init_from_fused_point_cloud(ply_path, observer_points)
                case "point_cloud":
                    try:
                        ply_path = os.path.join(conf.path, "point_cloud.ply")
                        model.init_from_pretrained_point_cloud(ply_path)
                    except FileNotFoundError as e:
                        logger.error(e)
                        raise e
                case "checkpoint":
                    checkpoint = torch.load(conf.initialization.path)
                    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
                case _:
                    raise ValueError(
                        f"unrecognized initialization.method {conf.initialization.method}, choose from [colmap, point_cloud, random, checkpoint]"
                    )

            self.strategy.init_densification_buffer()
            model.build_acc()
            model.setup_optimizer()
            global_step = 0

        self.global_step = global_step
        self.n_epochs = int((conf.n_iterations + len(train_dataset) - 1) / len(train_dataset))

    def init_gui(
        self,
        conf: DictConfig,
        model: MixtureOfGaussians,
        train_dataset: BoundedMultiViewDataset,
        val_dataset: BoundedMultiViewDataset,
        scene_bbox,
    ):
        gui = None
        if conf.with_gui:
            gui = GUI(conf, model, train_dataset, val_dataset, scene_bbox)
        self.gui = gui

    def init_metrics(self):
        self.criterions = Dict(
            psnr=PeakSignalNoiseRatio(data_range=1).to(self.device),
            ssim=StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device),
            lpips=LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(self.device),
        )

    def init_experiments_tracking(self, conf: DictConfig):
        object_name = Path(conf.path).stem
        writer, out_dir, run_name = create_summary_writer(
            conf, object_name, conf.out_dir, conf.experiment_name, conf.use_wandb
        )
        logger.info(f"📊 Training logs & will be saved to: {out_dir}")
        with open(os.path.join(out_dir, "parsed.yaml"), "w") as fp:
            OmegaConf.save(config=conf, f=fp)
        self.tracking = Dict(writer=writer, run_name=run_name, object_name=object_name, output_dir=out_dir)

    @torch.cuda.nvtx.range("get_metrics")
    def get_metrics(
        self,
        gpu_batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        losses: dict[str, torch.Tensor],
        profilers: dict[str, CudaTimer],
        split: str = "training",
        iteration: Optional[int] = None,
    ) -> dict[str, Union[int, float]]:
        metrics = dict()
        step = self.global_step

        # 这里直接取，假设维度已经是 [1, H, W, 3]
        rgb_pred = outputs["pred_rgb"]
        rgb_gt = gpu_batch.rgb_gt
        
        # 仅用于 torchmetrics 计算的转换 (HWC -> CHW)
        rgb_gt_chw = rgb_gt.permute(0, 3, 1, 2)
        rgb_pred_chw = rgb_pred.permute(0, 3, 1, 2)

        psnr = self.criterions["psnr"]
        ssim = self.criterions["ssim"]
        lpips = self.criterions["lpips"]

        metrics["losses"] = {k: v.detach().item() for k, v in losses.items()}

        is_compute_train_hit_metrics = (split == "training") and (step % self.conf.writer.hit_stat_frequency == 0)
        is_compute_validation_metrics = split == "validation"

        if is_compute_train_hit_metrics or is_compute_validation_metrics:
            metrics["hits_mean"] = outputs["hits_count"].mean().item()
            metrics["hits_std"] = outputs["hits_count"].std().item()
            metrics["hits_min"] = outputs["hits_count"].min().item()
            metrics["hits_max"] = outputs["hits_count"].max().item()

        if is_compute_validation_metrics:
            with torch.cuda.nvtx.range(f"criterions_psnr"):
                metrics["psnr"] = psnr(rgb_pred_chw, rgb_gt_chw).item()

            pred_rgb_full_clipped = rgb_pred_chw.clip(0, 1)

            with torch.cuda.nvtx.range(f"criterions_ssim"):
                metrics["ssim"] = ssim(rgb_pred_chw, rgb_gt_chw).item()
            with torch.cuda.nvtx.range(f"criterions_lpips"):
                metrics["lpips"] = lpips(pred_rgb_full_clipped, rgb_gt_chw).item()

            if iteration in self.conf.writer.log_image_views:
                metrics["img_hit_counts"] = jet_map(outputs["hits_count"][-1], self.conf.writer.max_num_hits)
                metrics["img_gt"] = gpu_batch.rgb_gt[-1].clip(0, 1.0)
                metrics["img_pred"] = outputs["pred_rgb"][-1].clip(0, 1.0)
                metrics["img_pred_dist"] = jet_map(outputs["pred_dist"][-1], 100)
                if "pred_opacity" in outputs:
                    metrics["img_pred_opacity"] = jet_map(outputs["pred_opacity"][-1], 1)

        if profilers:
            timings = {}
            for key, timer in profilers.items():
                if timer.enabled:
                    timings[key] = timer.timing()
            if timings:
                metrics["timings"] = timings

        return metrics

    @torch.cuda.nvtx.range("get_losses")
    def get_losses(
        self, gpu_batch: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Computes dictionary of losses for current batch."""
        
        # 架构合约：
        # 1. gpu_batch.rgb_gt 必须是 [B, H, W, C]
        # 2. outputs["pred_rgb"] 必须是 [B, H, W, C]
        # 3. 两者 H, W 必须完全相等
        
        rgb_gt = gpu_batch.rgb_gt
        rgb_pred = outputs["pred_rgb"]
        mask = gpu_batch.mask

        # Mask out the invalid pixels if the mask is provided
        if mask is not None:
            rgb_gt = rgb_gt * mask
            rgb_pred = rgb_pred * mask

        # L1 loss
        loss_l1 = torch.zeros(1, device=self.device)
        lambda_l1 = 0.0
        if self.conf.loss.use_l1:
            with torch.cuda.nvtx.range(f"loss-l1"):
                loss_l1 = torch.abs(rgb_pred - rgb_gt).mean()
                lambda_l1 = self.conf.loss.lambda_l1

        # L2 loss
        loss_l2 = torch.zeros(1, device=self.device)
        lambda_l2 = 0.0
        if self.conf.loss.use_l2:
            with torch.cuda.nvtx.range(f"loss-l2"):
                loss_l2 = torch.nn.functional.mse_loss(rgb_pred, rgb_gt)
                lambda_l2 = self.conf.loss.lambda_l2

        # DSSIM loss
        loss_ssim = torch.zeros(1, device=self.device)
        lambda_ssim = 0.0
        if self.conf.loss.use_ssim:
            with torch.cuda.nvtx.range(f"loss-ssim"):
                # SSIM 需要 (B, C, H, W)
                rgb_gt_chw = rgb_gt.permute(0, 3, 1, 2)
                rgb_pred_chw = rgb_pred.permute(0, 3, 1, 2)
                loss_ssim = 1.0 - ssim(rgb_pred_chw, rgb_gt_chw)
                lambda_ssim = self.conf.loss.lambda_ssim

        # Opacity regularization
        loss_opacity = torch.zeros(1, device=self.device)
        lambda_opacity = 0.0
        if self.conf.loss.use_opacity:
            with torch.cuda.nvtx.range(f"loss-opacity"):
                if "pred_opacity" in outputs:
                    val = outputs["pred_opacity"]
                else:
                    val = self.model.get_density()
                loss_opacity = torch.abs(val).mean()
                lambda_opacity = self.conf.loss.lambda_opacity

        # Scale regularization
        loss_scale = torch.zeros(1, device=self.device)
        lambda_scale = 0.0
        if self.conf.loss.use_scale:
            with torch.cuda.nvtx.range(f"loss-scale"):
                loss_scale = torch.abs(self.model.get_scale()).mean()
                lambda_scale = self.conf.loss.lambda_scale

        # Total loss
        loss = lambda_l1 * loss_l1 + lambda_ssim * loss_ssim + lambda_opacity * loss_opacity + lambda_scale * loss_scale
        return dict(
            total_loss=loss,
            l1_loss=lambda_l1 * loss_l1,
            l2_loss=lambda_l2 * loss_l2,
            ssim_loss=lambda_ssim * loss_ssim,
            opacity_loss=lambda_opacity * loss_opacity,
            scale_loss=lambda_scale * loss_scale,
        )

    # ... (余下的 log_validation_iter, log_validation_pass, log_training_iter,
    # log_training_pass, on_training_end, save_checkpoint, render_gui, 
    # run_train_pass, run_validation_pass, run_training, _flatten_list_of_dicts 
    # 等方法请直接保留原版，无需修改) ...


    @torch.cuda.nvtx.range("log_validation_iter")
    def log_validation_iter(
        self,
        gpu_batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        batch_metrics: dict[str, Any],
        iteration: Optional[int] = None,
    ) -> None:
        logger.log_progress(
            task_name="Validation",
            advance=1,
            iteration=f"{str(iteration)}",
            psnr=batch_metrics["psnr"],
            loss=batch_metrics["losses"]["total_loss"],
        )

    @torch.cuda.nvtx.range("log_validation_pass")
    def log_validation_pass(self, metrics: dict[str, Any]) -> None:
        writer = self.tracking.writer
        global_step = self.global_step

        if "img_pred" in metrics:
            writer.add_images("image/pred/val", torch.stack(metrics["img_pred"]), global_step, dataformats="NHWC")
        if "img_gt" in metrics:
            writer.add_images("image/gt", torch.stack(metrics["img_gt"]), global_step, dataformats="NHWC")
        if "img_hit_counts" in metrics:
            writer.add_images(
                "image/hit_counts/val", torch.stack(metrics["img_hit_counts"]), global_step, dataformats="NHWC"
            )
        if "img_pred_dist" in metrics:
            writer.add_images("image/dist/val", torch.stack(metrics["img_pred_dist"]), global_step, dataformats="NHWC")
        if "img_pred_opacity" in metrics:
            writer.add_images(
                "image/opacity/val", torch.stack(metrics["img_pred_opacity"]), global_step, dataformats="NHWC"
            )

        mean_timings = {}
        if "timings" in metrics:
            for time_key in metrics["timings"]:
                mean_timings[time_key] = np.mean(metrics["timings"][time_key])
                writer.add_scalar("time/" + time_key + "/val", mean_timings[time_key], global_step)

        writer.add_scalar("num_particles/val", self.model.num_gaussians, self.global_step)

        mean_psnr = np.mean(metrics["psnr"])
        writer.add_scalar("psnr/val", mean_psnr, global_step)
        writer.add_scalar("ssim/val", np.mean(metrics["ssim"]), global_step)
        writer.add_scalar("lpips/val", np.mean(metrics["lpips"]), global_step)
        writer.add_scalar("hits/min/val", np.mean(metrics["hits_min"]), global_step)
        writer.add_scalar("hits/max/val", np.mean(metrics["hits_max"]), global_step)
        writer.add_scalar("hits/mean/val", np.mean(metrics["hits_mean"]), global_step)

        loss = np.mean(metrics["losses"]["total_loss"])
        writer.add_scalar("loss/total/val", loss, global_step)
        if self.conf.loss.use_l1:
            l1_loss = np.mean(metrics["losses"]["l1_loss"])
            writer.add_scalar("loss/l1/val", l1_loss, global_step)
        if self.conf.loss.use_l2:
            l2_loss = np.mean(metrics["losses"]["l2_loss"])
            writer.add_scalar("loss/l2/val", l2_loss, global_step)
        if self.conf.loss.use_ssim:
            ssim_loss = np.mean(metrics["losses"]["ssim_loss"])
            writer.add_scalar("loss/ssim/val", ssim_loss, global_step)

        table = {k: np.mean(v) for k, v in metrics.items() if k in ("psnr", "ssim", "lpips")}
        for time_key in mean_timings:
            table[time_key] = f"{'{:.2f}'.format(mean_timings[time_key])}" + " ms/it"
        logger.log_table(f"📊 Validation Metrics - Step {global_step}", record=table)

    @torch.cuda.nvtx.range(f"log_training_iter")
    def log_training_iter(
        self,
        gpu_batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        batch_metrics: dict[str, Any],
        iteration: Optional[int] = None,
    ) -> None:
        writer = self.tracking.writer
        global_step = self.global_step

        if self.conf.enable_writer and global_step > 0 and global_step % self.conf.log_frequency == 0:
            loss = np.mean(batch_metrics["losses"]["total_loss"])
            writer.add_scalar("loss/total/train", loss, global_step)
            if self.conf.loss.use_l1:
                l1_loss = np.mean(batch_metrics["losses"]["l1_loss"])
                writer.add_scalar("loss/l1/train", l1_loss, global_step)
            if self.conf.loss.use_l2:
                l2_loss = np.mean(batch_metrics["losses"]["l2_loss"])
                writer.add_scalar("loss/l2/train", l2_loss, global_step)
            if self.conf.loss.use_ssim:
                ssim_loss = np.mean(batch_metrics["losses"]["ssim_loss"])
                writer.add_scalar("loss/ssim/train", ssim_loss, global_step)
            if self.conf.loss.use_opacity:
                opacity_loss = np.mean(batch_metrics["losses"]["opacity_loss"])
                writer.add_scalar("loss/opacity/train", opacity_loss, global_step)
            if self.conf.loss.use_scale:
                scale_loss = np.mean(batch_metrics["losses"]["scale_loss"])
                writer.add_scalar("loss/scale/train", scale_loss, global_step)
            if "psnr" in batch_metrics:
                writer.add_scalar("psnr/train", batch_metrics["psnr"], self.global_step)
            if "ssim" in batch_metrics:
                writer.add_scalar("ssim/train", batch_metrics["ssim"], self.global_step)
            if "lpips" in batch_metrics:
                writer.add_scalar("lpips/train", batch_metrics["lpips"], self.global_step)
            if "hits_mean" in batch_metrics:
                writer.add_scalar("hits/mean/train", batch_metrics["hits_mean"], self.global_step)
            if "hits_std" in batch_metrics:
                writer.add_scalar("hits/std/train", batch_metrics["hits_std"], self.global_step)
            if "hits_min" in batch_metrics:
                writer.add_scalar("hits/min/train", batch_metrics["hits_min"], self.global_step)
            if "hits_max" in batch_metrics:
                writer.add_scalar("hits/max/train", batch_metrics["hits_max"], self.global_step)

            if "timings" in batch_metrics:
                for time_key in batch_metrics["timings"]:
                    writer.add_scalar(
                        "time/" + time_key + "/train", batch_metrics["timings"][time_key], self.global_step
                    )

            writer.add_scalar("num_particles/train", self.model.num_gaussians, self.global_step)
            writer.add_scalar("train/num_GS", self.model.num_gaussians, self.global_step)

        logger.log_progress(
            task_name="Training",
            advance=1,
            step=f"{str(self.global_step)}",
            loss=batch_metrics["losses"]["total_loss"],
        )

    @torch.cuda.nvtx.range(f"log_training_pass")
    def log_training_pass(self, metrics):
        pass

    @torch.cuda.nvtx.range(f"on_training_end")
    def on_training_end(self):
        conf = self.conf
        out_dir = self.tracking.output_dir

        logger.log_rule("Exporting Models")
        if conf.export_ingp.enabled:
            ingp_path = conf.export_ingp.path if conf.export_ingp.path else os.path.join(out_dir, "export_last.ingp")
            exporter = INGPExporter()
            exporter.export(
                self.model,
                Path(ingp_path),
                dataset=self.train_dataset,
                conf=conf,
                force_half=conf.export_ingp.force_half,
            )
        if conf.export_ply.enabled:
            ply_path = conf.export_ply.path if conf.export_ply.path else os.path.join(out_dir, "export_last.ply")
            exporter = PLYExporter()
            exporter.export(self.model, Path(ply_path), dataset=self.train_dataset, conf=conf)
        if conf.export_usdz.enabled:
            usdz_path = conf.export_usdz.path if conf.export_usdz.path else os.path.join(out_dir, "export_last.usdz")
            exporter = USDZExporter()
            exporter.export(self.model, Path(usdz_path), dataset=self.train_dataset, conf=conf)

        if conf.test_last:
            logger.log_rule("Evaluation on Test Set")

            self.teardown_dataloaders()
            self.save_checkpoint(last_checkpoint=True)

            renderer = Renderer.from_preloaded_model(
                model=self.model,
                out_dir=out_dir,
                path=conf.path,
                save_gt=False,
                writer=self.tracking.writer,
                global_step=self.global_step,
                compute_extra_metrics=conf.compute_extra_metrics,
            )
            renderer.render_all()

    @torch.cuda.nvtx.range(f"save_checkpoint")
    def save_checkpoint(self, last_checkpoint: bool = False):
        global_step = self.global_step
        out_dir = self.tracking.output_dir
        parameters = self.model.get_model_parameters()
        parameters |= {"global_step": self.global_step, "epoch": self.n_epochs - 1}

        strategy_parameters = self.strategy.get_strategy_parameters()
        parameters = {**parameters, **strategy_parameters}

        os.makedirs(os.path.join(out_dir, f"ours_{int(global_step)}"), exist_ok=True)
        if not last_checkpoint:
            ckpt_path = os.path.join(out_dir, f"ours_{int(global_step)}", f"ckpt_{global_step}.pt")
        else:
            ckpt_path = os.path.join(out_dir, "ckpt_last.pt")
        torch.save(parameters, ckpt_path)
        logger.info(f'💾 Saved checkpoint to: "{os.path.abspath(ckpt_path)}"')

    def render_gui(self, scene_updated):
        gui = self.gui
        if gui is not None:
            import polyscope as ps

            if gui.live_update:
                if scene_updated or self.model.positions.requires_grad:
                    gui.update_cloud_viz()
                gui.update_render_view_viz()

            ps.frame_tick()
            while not gui.viz_do_train:
                ps.frame_tick()

            if ps.window_requests_close():
                logger.warning(
                    "Terminating training from GUI window is not supported. Please terminate it from the terminal."
                )

    @torch.cuda.nvtx.range(f"run_train_pass")
    def run_train_pass(self, conf: DictConfig):
        global_step = self.global_step
        model = self.model

        metrics = []
        profilers = {
            "inference": CudaTimer(enabled=self.conf.enable_frame_timings),
            "backward": CudaTimer(enabled=self.conf.enable_frame_timings),
            "build_as": CudaTimer(enabled=self.conf.enable_frame_timings),
        }

        for iter, batch in enumerate(self.train_dataloader):

            if self.global_step >= conf.n_iterations:
                return

            gpu_batch = self.train_dataset.get_gpu_batch_with_intrinsics(batch)

            is_time_to_validate = (global_step > 0 or conf.validate_first) and (global_step % self.val_frequency == 0)
            if is_time_to_validate:
                self.run_validation_pass(conf)

            with torch.cuda.nvtx.range(f"train_{global_step}_fwd"):
                profilers["inference"].start()
                outputs = model(gpu_batch, train=True, frame_id=global_step)
                profilers["inference"].end()

            with torch.cuda.nvtx.range(f"train_{global_step}_loss"):
                # outputs 已经在 tracer.py 里被切片还原了，
                # get_losses 已经在上方修复了维度对齐，
                # 所以这里直接传，不需要任何裁剪逻辑
                batch_losses = self.get_losses(gpu_batch, outputs)

            with torch.cuda.nvtx.range(f"train_{global_step}_pre_bwd"):
                self.strategy.pre_backward(
                    step=global_step,
                    scene_extent=self.scene_extent,
                    train_dataset=self.train_dataset,
                    batch=gpu_batch,
                    writer=self.tracking.writer,
                )

            with torch.cuda.nvtx.range(f"train_{global_step}_bwd"):
                profilers["backward"].start()
                batch_losses["total_loss"].backward()
                profilers["backward"].end()

            with torch.cuda.nvtx.range(f"train_{global_step}_post_bwd"):
                scene_updated = self.strategy.post_backward(
                    step=global_step,
                    scene_extent=self.scene_extent,
                    train_dataset=self.train_dataset,
                    batch=gpu_batch,
                    writer=self.tracking.writer,
                )

            with torch.cuda.nvtx.range(f"train_{global_step}_backprop"):
                if isinstance(model.optimizer, SelectiveAdam):
                    assert (
                        outputs["mog_visibility"].shape == model.density.shape
                    ), f"Visibility shape {outputs['mog_visibility'].shape} does not match density shape {model.density.shape}"
                    model.optimizer.step(outputs["mog_visibility"])
                else:
                    model.optimizer.step()
                model.optimizer.zero_grad()

            with torch.cuda.nvtx.range(f"train_{global_step}_scheduler"):
                model.scheduler_step(global_step)

            with torch.cuda.nvtx.range(f"train_{global_step}_post_opt_step"):
                scene_updated = self.strategy.post_optimizer_step(
                    step=global_step,
                    scene_extent=self.scene_extent,
                    train_dataset=self.train_dataset,
                    batch=gpu_batch,
                    writer=self.tracking.writer,
                )

            if self.model.progressive_training and check_step_condition(
                global_step, 0, 1e6, self.model.feature_dim_increase_interval
            ):
                self.model.increase_num_active_features()

            if scene_updated or (
                conf.model.bvh_update_frequency > 0 and global_step % conf.model.bvh_update_frequency == 0
            ):
                with torch.cuda.nvtx.range(f"train_{global_step}_bvh"):
                    profilers["build_as"].start()
                    model.build_acc(rebuild=True)
                    profilers["build_as"].end()

            self.global_step += 1
            global_step = self.global_step

            batch_metrics = self.get_metrics(
                gpu_batch, outputs, batch_losses, profilers, split="training", iteration=iter
            )

            if "forward_render" in model.renderer.timings:
                batch_metrics["timings"]["forward_render_cuda"] = model.renderer.timings["forward_render"]
            if "backward_render" in model.renderer.timings:
                batch_metrics["timings"]["backward_render_cuda"] = model.renderer.timings["backward_render"]
            metrics.append(batch_metrics)

            with torch.cuda.nvtx.range(f"train_{global_step-1}_log_iter"):
                self.log_training_iter(gpu_batch, outputs, batch_metrics, iter)

            with torch.cuda.nvtx.range(f"train_{global_step-1}_save_ckpt"):
                if global_step in conf.checkpoint.iterations:
                    self.save_checkpoint()

            with torch.cuda.nvtx.range(f"train_{global_step-1}_update_gui"):
                self.render_gui(scene_updated)

        self.log_training_pass(metrics)

    @torch.cuda.nvtx.range(f"run_validation_pass")
    @torch.no_grad()
    def run_validation_pass(self, conf: DictConfig) -> dict[str, Any]:
        profilers = {
            "inference": CudaTimer(),
        }
        metrics = []
        logger.info(f"Step {self.global_step} -- Running validation..")
        logger.start_progress(task_name="Validation", total_steps=len(self.val_dataloader), color="medium_purple3")

        for val_iteration, batch_idx in enumerate(self.val_dataloader):

            gpu_batch = self.val_dataset.get_gpu_batch_with_intrinsics(batch_idx)

            with torch.cuda.nvtx.range(f"train.validation_step_{self.global_step}"):
                profilers["inference"].start()
                outputs = self.model(gpu_batch, train=False)
                profilers["inference"].end()
                batch_losses = self.get_losses(gpu_batch, outputs)
                batch_metrics = self.get_metrics(
                    gpu_batch, outputs, batch_losses, profilers, split="validation", iteration=val_iteration
                )

                self.log_validation_iter(gpu_batch, outputs, batch_metrics, iteration=val_iteration)
                metrics.append(batch_metrics)

        logger.end_progress(task_name="Validation")

        metrics = self._flatten_list_of_dicts(metrics)
        self.log_validation_pass(metrics)
        return metrics

    @staticmethod
    def _flatten_list_of_dicts(list_of_dicts):
        flat_dict = defaultdict(list)
        for d in list_of_dicts:
            for k, v in d.items():
                if isinstance(v, dict):
                    flat_dict[k] = defaultdict(list) if k not in flat_dict else flat_dict[k]
                    for inner_k, inner_v in v.items():
                        flat_dict[k][inner_k].append(inner_v)
                else:
                    flat_dict[k].append(v)
        return flat_dict

    def run_training(self):
        assert self.model.optimizer is not None, "Optimizer needs to be initialized before the training can start!"
        conf = self.conf

        logger.log_rule(f"Training {conf.render.method.upper()}")

        logger.start_progress(task_name="Training", total_steps=conf.n_iterations, color="spring_green1")

        for epoch_idx in range(self.n_epochs):
            self.run_train_pass(conf)

        logger.end_progress(task_name="Training")

        stats = logger.finished_tasks["Training"]
        table = dict(
            n_steps=f"{self.global_step}",
            n_epochs=f"{self.n_epochs}",
            training_time=f"{stats['elapsed']:.2f} s",
            iteration_speed=f"{self.global_step / stats['elapsed']:.2f} it/s",
        )
        logger.log_table(f"🎊 Training Statistics", record=table)

        self.on_training_end()
        logger.info(f"🥳 Training Complete.")

        if self.gui is not None:
            self.gui.training_done = True
            logger.info(f"🎨 GUI Blocking... Terminate GUI to Stop.")
            self.gui.block_in_rendering_loop(fps=60)
