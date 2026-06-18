#!/usr/bin/env python3
"""
Leave-One-Out 实验共享工具模块

包含 eval_leave_one_out.py 和 visualize_loo.py 共用的函数：
- PSNR/SSIM 计算
- bbox mask 构建
- 精确分割 mask 加载
- GT 图像预处理
- instance_table 加载
- 相机外参加载与角度差计算
- 渲染文件查找
"""

import csv
import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image


# ---------------------------------------------------------------------------
# PSNR / SSIM
# ---------------------------------------------------------------------------

def compute_psnr_masked(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                        min_pixels: int = 100) -> float:
    """
    计算mask区域内PSNR，只对mask内有效像素求MSE

    Args:
        pred: [H, W, 3], range [0, 1]
        gt:   [H, W, 3], range [0, 1]
        mask: [H, W], bool
        min_pixels: mask内最少像素数

    Returns:
        PSNR (dB), 或 NaN
    """
    assert pred.shape == gt.shape, f"Shape mismatch: {pred.shape} vs {gt.shape}"
    assert pred.shape[:2] == mask.shape, f"Mask shape mismatch"
    assert np.isfinite(pred).all(), "pred contains NaN/Inf"
    assert np.isfinite(gt).all(), "gt contains NaN/Inf"

    num_pixels = int(mask.sum())
    if num_pixels < min_pixels:
        return np.nan

    # 检查mask内像素：pred和gt都全黑时无意义
    pred_masked = pred[mask]
    gt_masked = gt[mask]
    if pred_masked.max() < 1e-6 and gt_masked.max() < 1e-6:
        return np.nan  # 双方全黑，无法区分

    diff = (pred - gt) ** 2
    mse = diff[mask].mean()

    if mse < 1e-10:
        return 100.0

    return -10.0 * np.log10(mse)


# ---------------------------------------------------------------------------
# Instance table
# ---------------------------------------------------------------------------

def load_instance_table(instance_table_path: str) -> list:
    """加载 instance_table.csv"""
    inst = []
    with open(instance_table_path) as f:
        for r in csv.DictReader(f):
            r['frame_id'] = int(r['frame_id'])
            r['person_id'] = int(r['person_id'])
            r['xmin'] = int(r['xmin'])
            r['ymin'] = int(r['ymin'])
            r['xmax'] = int(r['xmax'])
            r['ymax'] = int(r['ymax'])
            r['bbox_valid'] = r['bbox_valid'] == 'True'
            inst.append(r)
    return inst


# ---------------------------------------------------------------------------
# Mask 构建
# ---------------------------------------------------------------------------

def build_bbox_mask(cam_id: str, frame_id: int, instance_table: list,
                    effective_h: int, render_w: int,
                    downsample: int) -> np.ndarray:
    """构建bbox矩形mask [effective_h, render_w]"""
    mask = np.zeros((effective_h, render_w), dtype=bool)
    for r in instance_table:
        if r['camera_id'] != cam_id or r['frame_id'] != frame_id:
            continue
        if not r.get('bbox_valid', False):
            continue
        xmin = max(int(r['xmin'] / downsample), 0)
        ymin = max(int(r['ymin'] / downsample), 0)
        xmax = min(int(r['xmax'] / downsample), render_w)
        ymax = min(int(r['ymax'] / downsample), effective_h)
        if ymax > ymin and xmax > xmin:
            mask[ymin:ymax, xmin:xmax] = True
    return mask


def load_segmentation_mask(seg_path: Path, effective_h: int,
                          render_w: int) -> np.ndarray:
    """加载精确分割mask（如果存在），返回 None 表示不可用"""
    if not seg_path.exists():
        return None
    try:
        mask_img = Image.open(seg_path)
        mask = np.array(mask_img)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0] > 128
        if mask.shape != (effective_h, render_w):
            mask_img = mask_img.resize((render_w, effective_h), Image.NEAREST)
            mask = np.array(mask_img) > 128
        return mask.astype(bool)
    except Exception as e:
        print(f"  警告: 无法加载mask {seg_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# GT 预处理
# ---------------------------------------------------------------------------

def preprocess_gt(image_bgr: np.ndarray, downsample: int) -> np.ndarray:
    """预处理GT图像，与训练时一致。返回 [H, W, 3] float32 [0,1]"""
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    h_padded = ((h + 15) // 16) * 16
    w_padded = ((w + 15) // 16) * 16
    if h != h_padded or w != w_padded:
        image = cv2.copyMakeBorder(
            image, 0, h_padded - h, 0, w_padded - w,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    if downsample > 1:
        h2, w2 = image.shape[:2]
        new_h, new_w = h2 // downsample, w2 // downsample
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# 渲染文件查找
# ---------------------------------------------------------------------------

def find_render_files(exp_dir: Path) -> list:
    """
    在实验目录下查找渲染文件（只取最新一次运行）。
    目录结构: exp_dir/run/Wildtrack-*/ours_*/renders/*.png

    Returns:
        排序后的渲染文件路径列表
    """
    render_dirs = sorted(exp_dir.glob("run/Wildtrack-*/ours_*/renders"))
    if not render_dirs:
        return []
    # Only take the latest run directory
    return sorted(render_dirs[-1].glob("*.png"))


def find_gt_path(dataset_path: str, cam_id: str, frame_id: int) -> Path:
    """构建GT图像路径"""
    return Path(dataset_path) / "Image_subsets" / cam_id / f"{frame_id:08d}.png"


# ---------------------------------------------------------------------------
# 相机外参与角度差
# ---------------------------------------------------------------------------

CAMERA_IDS = [f"C{i}" for i in range(1, 8)]


def load_extrinsics(dataset_path: str) -> dict:
    """
    从 WildTrack calibrations 目录加载外参 (C2W 矩阵)

    Returns:
        {cam_id: 4x4 C2W matrix}
    """
    extrinsics = {}
    extrinsic_dir = os.path.join(dataset_path, "calibrations", "extrinsic")
    for cam_id in CAMERA_IDS:
        xml_path = os.path.join(extrinsic_dir, f"extr_{cam_id}.xml")
        if not os.path.exists(xml_path):
            continue
        fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
        R = fs.getNode("R").mat()
        T = fs.getNode("T").mat().flatten()
        fs.release()
        W2C = np.eye(4, dtype=np.float32)
        W2C[:3, :3] = R
        W2C[:3, 3] = T
        C2W = np.linalg.inv(W2C).astype(np.float32)
        extrinsics[cam_id] = C2W
    return extrinsics


def get_camera_direction(c2w: np.ndarray) -> np.ndarray:
    """从C2W矩阵提取相机朝向向量（-z轴方向，即光轴朝向）"""
    direction = -c2w[:3, 2]  # -z column
    return direction / np.linalg.norm(direction)


def compute_camera_angle_diff(c2w_a: np.ndarray, c2w_b: np.ndarray) -> float:
    """
    计算两个相机之间的视角差（度）

    Args:
        c2w_a, c2w_b: 4x4 C2W 矩阵

    Returns:
        角度差（度），0~180
    """
    dir_a = get_camera_direction(c2w_a)
    dir_b = get_camera_direction(c2w_b)
    cos_angle = np.clip(np.dot(dir_a, dir_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def find_nearest_train_angle_diff(held_cam: str, extrinsics: dict) -> float:
    """
    找到 held-out 相机与最近的训练相机之间的角度差

    Args:
        held_cam: held-out 相机 ID (如 "C3")
        extrinsics: {cam_id: C2W matrix}

    Returns:
        最小角度差（度）
    """
    held_c2w = extrinsics[held_cam]
    train_cams = [c for c in CAMERA_IDS if c != held_cam]
    min_diff = 180.0
    for cam in train_cams:
        if cam not in extrinsics:
            continue
        diff = compute_camera_angle_diff(held_c2w, extrinsics[cam])
        min_diff = min(min_diff, diff)
    return min_diff
