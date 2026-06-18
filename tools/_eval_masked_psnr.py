import os
import re
import numpy as np
import torch
import cv2
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

RENDER_DIR = "/data02/zhangrunxiang/3dgrut/runs/Wildtrack-2905_235547/ours_30000/renders"
DATA_PATH = "/data02/zhangrunxiang/data/Wildtrack"
DOWNSAMPLE = 4
ORIGINAL_H = 1080
PADDED_H = 1088

def main():
    effective_h = ORIGINAL_H // DOWNSAMPLE
    padded_h = PADDED_H // DOWNSAMPLE
    effective_w = 1920 // DOWNSAMPLE
    print(f"Effective image: {effective_w}x{effective_h}, Padded: {effective_w}x{padded_h}")

    camera_ids = [f"C{i}" for i in range(1, 8)]
    image_paths = {}
    for cam_id in camera_ids:
        cam_dir = os.path.join(DATA_PATH, "Image_subsets", cam_id)
        paths = sorted([
            os.path.join(cam_dir, f) for f in os.listdir(cam_dir)
            if f.endswith(".png")
        ], key=lambda x: int(re.search(r'([0-9]+)\.png', os.path.basename(x)).group(1)))
        image_paths[cam_id] = paths

    test_split_interval = 5
    num_frames = len(image_paths[camera_ids[0]])
    available_frames = list(range(0, num_frames * 5, 5))
    val_frame_ids = [f for f in available_frames if f % (test_split_interval * 5) == 0]

    val_indices = []
    for frame_id in val_frame_ids:
        for cam_id in camera_ids:
            val_indices.append((cam_id, frame_id))

    print(f"Val samples: {len(val_indices)}")
    print(f"Rendered images: {len(os.listdir(RENDER_DIR))}")

    psnr_fn = PeakSignalNoiseRatio(data_range=1).to("cuda")
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to("cuda")

    full_psnr_list = []
    masked_psnr_list = []
    full_ssim_list = []
    masked_ssim_list = []
    full_lpips_list = []
    masked_lpips_list = []

    for idx in range(min(len(val_indices), len(os.listdir(RENDER_DIR)))):
        render_path = os.path.join(RENDER_DIR, f"{idx:05d}.png")
        if not os.path.exists(render_path):
            print(f"Missing render: {render_path}")
            continue

        cam_id, frame_id = val_indices[idx]
        image_index = frame_id // 5
        gt_path = image_paths[cam_id][image_index]

        pred = cv2.imread(render_path)
        gt_full = cv2.imread(gt_path)

        if pred is None or gt_full is None:
            print(f"Failed to read image at idx {idx}")
            continue

        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        gt_full = cv2.cvtColor(gt_full, cv2.COLOR_BGR2RGB)

        h, w = gt_full.shape[:2]
        h_padded = ((h + 15) // 16) * 16
        w_padded = ((w + 15) // 16) * 16

        if h != h_padded or w != w_padded:
            gt_full = cv2.copyMakeBorder(
                gt_full, 0, h_padded - h, 0, w_padded - w,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

        if DOWNSAMPLE > 1:
            h_p, w_p = gt_full.shape[:2]
            new_h, new_w = h_p // DOWNSAMPLE, w_p // DOWNSAMPLE
            gt_full = cv2.resize(gt_full, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pred_tensor = torch.from_numpy(pred.astype(np.float32) / 255.0).unsqueeze(0).cuda()
        gt_tensor = torch.from_numpy(gt_full.astype(np.float32) / 255.0).unsqueeze(0).cuda()

        full_psnr = psnr_fn(pred_tensor, gt_tensor).item()
        full_psnr_list.append(full_psnr)

        pred_masked = pred_tensor[:, :effective_h, :, :]
        gt_masked = gt_tensor[:, :effective_h, :, :]
        masked_psnr = psnr_fn(pred_masked, gt_masked).item()
        masked_psnr_list.append(masked_psnr)

        pred_chw = pred_tensor.permute(0, 3, 1, 2)
        gt_chw = gt_tensor.permute(0, 3, 1, 2)
        full_ssim = ssim_fn(pred_chw, gt_chw).item()
        full_ssim_list.append(full_ssim)

        pred_masked_chw = pred_masked.permute(0, 3, 1, 2)
        gt_masked_chw = gt_masked.permute(0, 3, 1, 2)
        masked_ssim = ssim_fn(pred_masked_chw, gt_masked_chw).item()
        masked_ssim_list.append(masked_ssim)

        full_lpips = lpips_fn(pred_tensor.clip(0, 1).permute(0, 3, 1, 2), gt_tensor.permute(0, 3, 1, 2)).item()
        full_lpips_list.append(full_lpips)

        masked_lpips = lpips_fn(pred_masked.clip(0, 1).permute(0, 3, 1, 2), gt_masked.permute(0, 3, 1, 2)).item()
        masked_lpips_list.append(masked_lpips)

        if idx % 50 == 0:
            print(f"  [{idx}/{len(val_indices)}] cam={cam_id} frame={frame_id}: full_psnr={full_psnr:.2f}, masked_psnr={masked_psnr:.2f}")

    print(f"\n{'='*60}")
    print(f"  VAL (TEST) SPLIT RESULTS ({len(full_psnr_list)} images)")
    print(f"{'='*60}")
    print(f"  Full image  PSNR: {np.mean(full_psnr_list):.2f} +/- {np.std(full_psnr_list):.2f}")
    print(f"  Masked      PSNR: {np.mean(masked_psnr_list):.2f} +/- {np.std(masked_psnr_list):.2f}")
    print(f"  Full image  SSIM: {np.mean(full_ssim_list):.4f}")
    print(f"  Masked      SSIM: {np.mean(masked_ssim_list):.4f}")
    print(f"  Full image  LPIPS: {np.mean(full_lpips_list):.4f}")
    print(f"  Masked      LPIPS: {np.mean(masked_lpips_list):.4f}")
    print(f"  PSNR delta (masked - full): {np.mean(masked_psnr_list) - np.mean(full_psnr_list):.2f}")

if __name__ == "__main__":
    main()
