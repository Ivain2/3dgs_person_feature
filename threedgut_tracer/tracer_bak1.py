# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.utils.cpp_extension
from omegaconf import OmegaConf

from threedgrut.datasets.protocols import Batch
from threedgrut.datasets.camera_models import ShutterType

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# 全局插件加载逻辑

_3dgut_plugin = None

def load_3dgut_plugin(conf):
    global _3dgut_plugin
    if _3dgut_plugin is None:
        try:
            from . import lib3dgut_cc as tdgut  # type: ignore
        except ImportError:
            from .setup_3dgut import setup_3dgut
            setup_3dgut(conf)
            import lib3dgut_cc as tdgut  # type: ignore
        _3dgut_plugin = tdgut


# ----------------------------------------------------------------------------
# 辅助类

@dataclass
class SensorPose3D:
    T_world_sensors: list
    timestamps_us: list


class SensorPose3DModel:
    def __init__(self, R, T, trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        super(SensorPose3DModel, self).__init__()
        self.R = R
        self.T = T
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale
        self.world_view_transform = (
            torch.tensor(SensorPose3DModel.__getWorld2View2(R, T, trans, scale)).transpose(0, 1).cpu()
        )

    @staticmethod
    def __getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0
        C2W = np.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        Rt = np.linalg.inv(C2W)
        return np.float32(Rt)

    @staticmethod
    def __so3_matrix_to_quat(R: torch.Tensor | np.ndarray, unbatch: bool = True) -> torch.Tensor:
        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R)
        R = R.reshape((-1, 3, 3))
        num_rotations, D1, D2 = R.shape
        decision_matrix = torch.empty((num_rotations, 4), dtype=R.dtype, device=R.device)
        quat = torch.empty((num_rotations, 4), dtype=R.dtype, device=R.device)
        decision_matrix[:, :3] = R.diagonal(dim1=1, dim2=2)
        decision_matrix[:, -1] = decision_matrix[:, :3].sum(dim=1)
        choices = decision_matrix.argmax(dim=1)
        ind = torch.nonzero(choices != 3, as_tuple=True)[0]
        i = choices[ind]
        j = (i + 1) % 3
        k = (j + 1) % 3
        quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * R[ind, i, i]
        quat[ind, j] = R[ind, j, i] + R[ind, i, j]
        quat[ind, k] = R[ind, k, i] + R[ind, i, k]
        quat[ind, 3] = R[ind, k, j] - R[ind, j, k]
        ind = torch.nonzero(choices == 3, as_tuple=True)[0]
        quat[ind, 0] = R[ind, 2, 1] - R[ind, 1, 2]
        quat[ind, 1] = R[ind, 0, 2] - R[ind, 2, 0]
        quat[ind, 2] = R[ind, 1, 0] - R[ind, 0, 1]
        quat[ind, 3] = 1 + decision_matrix[ind, -1]
        quat = quat / torch.norm(quat, dim=1)[:, None]
        if unbatch:
            quat = quat.squeeze()
        return quat

    def get_sensor_pose(self):
        T_world_sensor_t = self.world_view_transform[3, :3]
        T_world_sensor_R = self.world_view_transform[:3, :3].transpose(0, 1)
        T_world_sensor_quat = SensorPose3DModel.__so3_matrix_to_quat(T_world_sensor_R)
        T_world_sensor_tquat = torch.hstack([T_world_sensor_t.cpu(), T_world_sensor_quat.cpu()])
        return SensorPose3D(
            T_world_sensors=[T_world_sensor_tquat, T_world_sensor_tquat],
            timestamps_us=[0, 1],
        )


# ----------------------------------------------------------------------------
# Tracer 类

class Tracer:
    class _Autograd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tracer_wrapper, frame_id, n_active_features, ray_ori, ray_dir, 
                    mog_pos, mog_rot, mog_scl, mog_dns, mog_sph, sensor_params, sensor_poses):
            
            # ========== 【调试】打印输入形状 ==========
            if frame_id == 0:
                import sys
                print("\n" + "="*80, file=sys.stderr)
                print("📥 INPUT TO C++ Forward", file=sys.stderr)
                print("="*80, file=sys.stderr)
                print(f"ray_ori.shape = {ray_ori.shape}", file=sys.stderr)
                print(f"ray_dir.shape = {ray_dir.shape}", file=sys.stderr)
                print("="*80 + "\n", file=sys.stderr)
                sys.stderr.flush()
            # ==========================================
            
            particle_density = torch.concat(
                [mog_pos, mog_dns, mog_rot, mog_scl, torch.zeros_like(mog_dns)], dim=1
            ).contiguous()
            particle_radiance = mog_sph.contiguous()

            # 【关键】保持原始逻辑：硬编码 4D 假设
            ray_time = (
                torch.ones(
                    (ray_ori.shape[0], ray_ori.shape[1], ray_ori.shape[2], 1), 
                    device=ray_ori.device, dtype=torch.long
                ) * sensor_poses.timestamps_us[0]
            )

            # ========== 【调试】打印实际传入 C++ 的参数 ==========
            if frame_id == 0:
                import sys
                print("\n" + "="*80, file=sys.stderr)
                print("📥 CALLING tracer_wrapper.trace()", file=sys.stderr)
                print("="*80, file=sys.stderr)
                print(f"ray_ori.shape = {ray_ori.shape}", file=sys.stderr)
                print(f"ray_dir.shape = {ray_dir.shape}", file=sys.stderr)
                print(f"ray_time.shape = {ray_time.shape}", file=sys.stderr)
                print(f"particle_density.shape = {particle_density.shape}", file=sys.stderr)
                print(f"particle_radiance.shape = {particle_radiance.shape}", file=sys.stderr)
                print("="*80 + "\n", file=sys.stderr)
                sys.stderr.flush()
            # =====================================================

            ray_radiance_density, ray_hit_distance, ray_hit_count, mog_visibility = tracer_wrapper.trace(
                frame_id, n_active_features, particle_density, particle_radiance,
                ray_ori.contiguous(), ray_dir.contiguous(), ray_time.contiguous(),
                sensor_params,
                sensor_poses.timestamps_us[0], sensor_poses.timestamps_us[1],
                sensor_poses.T_world_sensors[0], sensor_poses.T_world_sensors[1],
            )

            # ========== 【调试】打印 C++ 返回值 ==========
            if frame_id == 0:
                import sys
                print("\n" + "="*80, file=sys.stderr)
                print("🔧 C++ RETURNED:", file=sys.stderr)
                print("="*80, file=sys.stderr)
                print(f"ray_radiance_density.shape = {ray_radiance_density.shape}", file=sys.stderr)
                print(f"ray_hit_distance.shape = {ray_hit_distance.shape}", file=sys.stderr)
                print(f"ray_hit_count.shape = {ray_hit_count.shape}", file=sys.stderr)
                print(f"mog_visibility.shape = {mog_visibility.shape}", file=sys.stderr)
                print("="*80 + "\n", file=sys.stderr)
                sys.stderr.flush()
            # ============================================

            ctx.save_for_backward(
                ray_ori, ray_dir, ray_time, ray_radiance_density, 
                ray_hit_distance, particle_density, particle_radiance
            )
            ctx.tracer_wrapper = tracer_wrapper
            ctx.sensor_params = sensor_params
            ctx.sensor_poses = sensor_poses
            ctx.frame_id = frame_id
            ctx.n_active_features = n_active_features

            return ray_radiance_density, ray_hit_distance, ray_hit_count, mog_visibility

        @staticmethod
        def backward(ctx, ray_radiance_density_grd, ray_hit_distance_grd, 
                     ray_hit_count_grd_UNUSED, mog_visibility_grd_UNUSED):
            
            (ray_ori, ray_dir, ray_time, ray_radiance_density, ray_hit_distance, 
             particle_density, particle_radiance) = ctx.saved_variables

            # ========== 【调试】打印 Backward 输入 ==========
            if ctx.frame_id == 0:
                import sys
                print("\n" + "="*80, file=sys.stderr)
                print("🔙 INPUT TO C++ Backward", file=sys.stderr)
                print("="*80, file=sys.stderr)
                print(f"ray_ori.shape = {ray_ori.shape}", file=sys.stderr)
                print(f"ray_dir.shape = {ray_dir.shape}", file=sys.stderr)
                print(f"ray_time.shape = {ray_time.shape}", file=sys.stderr)
                print(f"ray_radiance_density.shape = {ray_radiance_density.shape}", file=sys.stderr)
                if ray_radiance_density_grd is not None:
                    print(f"ray_radiance_density_grd.shape = {ray_radiance_density_grd.shape}", file=sys.stderr)
                print(f"ray_hit_distance.shape = {ray_hit_distance.shape}", file=sys.stderr)
                if ray_hit_distance_grd is not None:
                    print(f"ray_hit_distance_grd.shape = {ray_hit_distance_grd.shape}", file=sys.stderr)
                print("="*80 + "\n", file=sys.stderr)
                sys.stderr.flush()
            # ===============================================

            particle_density_grd, particle_radiance_grd = ctx.tracer_wrapper.trace_bwd(
                ctx.frame_id, ctx.n_active_features, 
                particle_density, particle_radiance,
                ray_ori, ray_dir, ray_time, 
                ctx.sensor_params,
                ctx.sensor_poses.timestamps_us[0], ctx.sensor_poses.timestamps_us[1],
                ctx.sensor_poses.T_world_sensors[0], ctx.sensor_poses.T_world_sensors[1],
                ray_radiance_density, ray_radiance_density_grd,
                ray_hit_distance, ray_hit_distance_grd,
            )

            mog_pos_grd, mog_dns_grd, mog_rot_grd, mog_scl_grd, _ = torch.split(
                particle_density_grd, [3, 1, 4, 3, 1], dim=1
            )

            return (
                None,  # tracer_wrapper
                None,  # frame_id
                None,  # n_active_features
                None,  # ray_ori
                None,  # ray_dir
                mog_pos_grd.contiguous(),
                mog_rot_grd.contiguous(),
                mog_scl_grd.contiguous(),
                mog_dns_grd.contiguous(),
                particle_radiance_grd.contiguous(),
                None,  # sensor_params
                None,  # sensor_poses
            )

    def __init__(self, conf):
        self.device = "cuda"
        self.conf = conf
        torch.zeros(1, device=self.device)
        load_3dgut_plugin(conf)
        self.tracer_wrapper = _3dgut_plugin.SplatRaster(OmegaConf.to_container(conf))

    @property
    def timings(self):
        return self.tracer_wrapper.collect_times()

    def build_acc(self, gaussians, rebuild=True):
        pass

    @staticmethod
    def __fov2focal(fov_radians: float, pixels: int):
        return pixels / (2 * math.tan(fov_radians / 2))

    @staticmethod
    def __focal2fov(focal: float, pixels: int):
        return 2 * math.atan(pixels / (2 * focal))

    def render(self, gaussians, gpu_batch: Batch, train=False, frame_id=0):
        # 【关键】保持 4D 输入，不做 squeeze
        rays_o = gpu_batch.rays_ori  # [1, 1088, 1920, 3]
        rays_d = gpu_batch.rays_dir  # [1, 1088, 1920, 3]

        # ========== 【调试】打印从 Dataset 收到的形状 ==========
        if frame_id == 0:
            import sys
            print("\n" + "="*80, file=sys.stderr)
            print("📦 INPUT FROM DATASET", file=sys.stderr)
            print("="*80, file=sys.stderr)
            print(f"rays_o.shape: {rays_o.shape}", file=sys.stderr)
            print(f"rays_d.shape: {rays_d.shape}", file=sys.stderr)
            if hasattr(gpu_batch, 'rgb_gt') and gpu_batch.rgb_gt is not None:
                print(f"rgb_gt.shape: {gpu_batch.rgb_gt.shape}", file=sys.stderr)
            print("="*80 + "\n", file=sys.stderr)
            sys.stderr.flush()
        # ========================================================

        sensor_params, sensor_poses = Tracer.__create_camera_parameters(gpu_batch)

        num_gaussians = gaussians.num_gaussians
        with torch.cuda.nvtx.range(f"model.forward({num_gaussians} gaussians)"):
            pred_rgba, pred_dist, hits_count, mog_visibility = Tracer._Autograd.apply(
                self.tracer_wrapper,
                frame_id,
                gaussians.n_active_features,
                rays_o.contiguous(),  # 保持 4D ✅
                rays_d.contiguous(),  # 保持 4D ✅
                gaussians.positions.contiguous(),
                gaussians.get_rotation().contiguous(),
                gaussians.get_scale().contiguous(),
                gaussians.get_density().contiguous(),
                gaussians.get_features().contiguous(),
                sensor_params,
                sensor_poses,
            )

            # C++ 返回 3D，手动添加 Batch 维度
            pred_rgb = pred_rgba[..., :3].unsqueeze(0).contiguous()
            pred_opacity = pred_rgba[..., 3:].unsqueeze(0).contiguous()
            pred_dist = pred_dist.unsqueeze(0).contiguous()
            hits_count = hits_count.unsqueeze(0).contiguous()

            # ========== 【调试】打印最终返回的形状 ==========
            if frame_id == 0:
                import sys
                print("\n" + "="*80, file=sys.stderr)
                print("📤 OUTPUT FROM Tracer.render", file=sys.stderr)
                print("="*80, file=sys.stderr)
                print(f"pred_rgb.shape: {pred_rgb.shape}", file=sys.stderr)
                print(f"pred_opacity.shape: {pred_opacity.shape}", file=sys.stderr)
                print(f"pred_dist.shape: {pred_dist.shape}", file=sys.stderr)
                print("="*80 + "\n", file=sys.stderr)
                sys.stderr.flush()
            # =================================================

            pred_rgb, pred_opacity = gaussians.background(
                gpu_batch.T_to_world.contiguous(), rays_d, pred_rgb, pred_opacity, train
            )

            timings = self.tracer_wrapper.collect_times()

        return {
            "pred_rgb": pred_rgb,
            "pred_opacity": pred_opacity,
            "pred_dist": pred_dist,
            "pred_normals": torch.nn.functional.normalize(torch.ones_like(pred_rgb), dim=3),
            "hits_count": hits_count,
            "frame_time_ms": timings["forward_render"] if "forward_render" in timings else 0.0,
            "mog_visibility": mog_visibility,
        }

    @staticmethod
    def __create_camera_parameters(gpu_batch):
        SHUTTER_TYPE_MAP = {
            ShutterType.ROLLING_TOP_TO_BOTTOM: _3dgut_plugin.ShutterType.ROLLING_TOP_TO_BOTTOM,
            ShutterType.ROLLING_LEFT_TO_RIGHT: _3dgut_plugin.ShutterType.ROLLING_LEFT_TO_RIGHT,
            ShutterType.ROLLING_BOTTOM_TO_TOP: _3dgut_plugin.ShutterType.ROLLING_BOTTOM_TO_TOP,
            ShutterType.ROLLING_RIGHT_TO_LEFT: _3dgut_plugin.ShutterType.ROLLING_RIGHT_TO_LEFT,
            ShutterType.GLOBAL: _3dgut_plugin.ShutterType.GLOBAL,
        }

        # Extrinsics
        pose = gpu_batch.T_to_world.squeeze()
        if pose.ndim == 3:
            pose = pose[0]
        if pose.shape == (3, 4):
            C2W = np.concatenate((pose[:3, :4].cpu().detach().numpy(), np.zeros((1, 4))))
            C2W[3, 3] = 1.0
        else:
            C2W = np.eye(4, dtype=np.float32)
            C2W[:3, :4] = pose[:3, :4].cpu().detach().numpy()

        W2C = np.linalg.inv(C2W)
        R = np.transpose(W2C[:3, :3])
        T = W2C[:3, 3]
        pose_model = SensorPose3DModel(R=R, T=T)

        # Intrinsics
        if (K := gpu_batch.intrinsics) is not None:
            focalx, focaly, cx, cy = K[0], K[1], K[2], K[3]
            orig_w = int(2 * cx)
            orig_h = int(2 * cy)
            FovX = Tracer.__focal2fov(focalx, orig_w)
            FovY = Tracer.__focal2fov(focaly, orig_h)
            
            camera_model_parameters = _3dgut_plugin.fromOpenCVPinholeCameraModelParameters(
                resolution=np.array([orig_w, orig_h], dtype=np.uint64),
                shutter_type=_3dgut_plugin.ShutterType.GLOBAL,
                principal_point=np.array([orig_w, orig_h], dtype=np.float32) / 2,
                focal_length=np.array(
                    [orig_w / (2.0 * math.tan(FovX * 0.5)), orig_h / (2.0 * math.tan(FovY * 0.5))], 
                    dtype=np.float32
                ),
                radial_coeffs=np.zeros((6,), dtype=np.float32),
                tangential_coeffs=np.zeros((2,), dtype=np.float32),
                thin_prism_coeffs=np.zeros((4,), dtype=np.float32),
            )
            return camera_model_parameters, pose_model.get_sensor_pose()

        elif (K := gpu_batch.intrinsics_OpenCVPinholeCameraModelParameters) is not None:
            camera_model_parameters = _3dgut_plugin.fromOpenCVPinholeCameraModelParameters(
                resolution=K["resolution"],
                shutter_type=SHUTTER_TYPE_MAP[K["shutter_type"]],
                principal_point=K["principal_point"],
                focal_length=K["focal_length"],
                radial_coeffs=K["radial_coeffs"],
                tangential_coeffs=K["tangential_coeffs"],
                thin_prism_coeffs=K["thin_prism_coeffs"],
            )
            return camera_model_parameters, pose_model.get_sensor_pose()

        elif (K := gpu_batch.intrinsics_OpenCVFisheyeCameraModelParameters) is not None:
            camera_model_parameters = _3dgut_plugin.fromOpenCVFisheyeCameraModelParameters(
                resolution=K["resolution"],
                shutter_type=SHUTTER_TYPE_MAP[K["shutter_type"]],
                principal_point=K["principal_point"],
                focal_length=K["focal_length"],
                radial_coeffs=K["radial_coeffs"],
                max_angle=K["max_angle"],
            )
            return camera_model_parameters, pose_model.get_sensor_pose()

        raise ValueError(
            f"Camera intrinsics unavailable or unsupported, input keys are [{', '.join(gpu_batch.keys())}]"
        )
