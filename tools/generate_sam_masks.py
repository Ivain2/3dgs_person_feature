#!/usr/bin/env python3
"""
SAM 人体分割 mask 生成脚本

用途：
为 WildTrack 数据集生成精确的人体分割 mask，用于 leave-one-out 实验的精确前景评估。

运行方式：
    python tools/generate_sam_masks.py \
        --dataset_path /data02/zhangrunxiang/data/Wildtrack_small_sample \
        --output_dir data/wildtrack/segmentation_masks \
        --frames 0 100 200 300

输出：
    data/wildtrack/segmentation_masks/
    ├── C1/
    │   ├── 0000.png
    │   ├── 0100.png
    │   └── ...
    └── C7/

依赖：
    pip install segment-anything opencv-python
    SAM checkpoint: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

annotations_remapped 格式说明：
    每条记录: {"bbox": {"xmin":..., "ymin":..., "xmax":..., "ymax":...},
               "camera_id": 0, "new_id": 94, "frame_id": 100, "raw_id": 122, ...}
    camera_id 是 0-based 整数 (0~6)，对应 "C1"~"C7"（映射: cam_str = f"C{camera_id+1}"）
    bbox 是嵌套字典，不是顶层字段
    没有 bbox_valid 字段，用 bbox 坐标是否在图像范围内判断有效性
"""

import argparse
import cv2
import json
import numpy as np
import sys
import torch
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from loo_utils import CAMERA_IDS


# camera_id (0-based int) → cam_id (1-based string)
CAM_ID_MAP = {i: f"C{i+1}" for i in range(7)}


def load_annotations_remapped(dataset_path: str) -> dict:
    """
    加载 annotations_remapped JSON 文件

    Returns:
        {frame_id: [record, ...]}
        每条 record: {"bbox": {...}, "camera_id": int, "new_id": int, ...}
    """
    ann_dir = Path(dataset_path) / "annotations_remapped"
    annotations = {}
    for json_file in sorted(ann_dir.glob("*.json")):
        frame_id = int(json_file.stem)
        with open(json_file) as f:
            annotations[frame_id] = json.load(f)
    return annotations


def is_bbox_valid(bbox: dict, img_w: int = 1920, img_h: int = 1080) -> bool:
    """检查 bbox 是否有效（坐标在合理范围内）"""
    xmin = bbox.get('xmin', -1)
    ymin = bbox.get('ymin', -1)
    xmax = bbox.get('xmax', -1)
    ymax = bbox.get('ymax', -1)
    # 至少有一部分在图像内
    if xmax <= 0 or ymax <= 0 or xmin >= img_w or ymin >= img_h:
        return False
    if xmax <= xmin or ymax <= ymin:
        return False
    return True


def initialize_sam(checkpoint_path: str):
    """初始化 SAM 模型"""
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        print("错误: 未安装 segment-anything")
        print("请运行: pip install segment-anything")
        sys.exit(1)

    if not Path(checkpoint_path).exists():
        print(f"错误: checkpoint 不存在 {checkpoint_path}")
        sys.exit(1)

    print("加载 SAM 模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        min_mask_region_area=100,
    )
    return mask_generator


def generate_person_masks(image: np.ndarray, bbox_list: list,
                         mask_generator) -> np.ndarray:
    """
    为图像生成人体分割 mask

    Args:
        image: 输入图像 [H, W, 3] (BGR)
        bbox_list: 该帧该相机的 bbox 列表 [{"xmin":..., ...}, ...]
        mask_generator: SAM mask generator

    Returns:
        二值 mask [H, W]
    """
    masks = mask_generator.generate(image)
    person_mask = np.zeros(image.shape[:2], dtype=bool)

    for bbox in bbox_list:
        xmin, ymin = int(bbox['xmin']), int(bbox['ymin'])
        xmax, ymax = int(bbox['xmax']), int(bbox['ymax'])

        # 裁剪到图像范围
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(image.shape[1], xmax)
        ymax = min(image.shape[0], ymax)

        if xmax <= xmin or ymax <= ymin:
            continue

        bbox_area = (ymax - ymin) * (xmax - xmin)

        # 在 bbox 区域内寻找最佳匹配的 SAM mask (using proper IoU)
        best_mask = None
        best_iou = 0

        for m in masks:
            seg = m['segmentation']
            seg_in_bbox = seg[ymin:ymax, xmin:xmax]
            intersection = seg_in_bbox.sum()
            union = seg_in_bbox.sum() + bbox_area - intersection
            iou = intersection / union if union > 0 else 0

            if iou > best_iou and iou > 0.2:
                best_iou = iou
                best_mask = seg

        if best_mask is not None:
            # Crop mask to bbox to avoid mask spilling outside bbox
            best_mask_cropped = best_mask.copy()
            best_mask_cropped[:ymin, :] = False
            best_mask_cropped[ymax:, :] = False
            best_mask_cropped[:, :xmin] = False
            best_mask_cropped[:, xmax:] = False
            person_mask |= best_mask_cropped

    return person_mask


def main():
    parser = argparse.ArgumentParser(description="生成 SAM 人体分割 mask")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--frames", nargs="+", type=int, default=None)
    parser.add_argument("--cameras", nargs="+", type=str, default=CAMERA_IDS)
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/sam_vit_h_4b8939.pth")

    args = parser.parse_args()

    # 初始化 SAM
    mask_generator = initialize_sam(args.checkpoint)

    # 加载 annotations
    print("加载 annotations_remapped...")
    annotations = load_annotations_remapped(args.dataset_path)
    print(f"  {len(annotations)} 帧")

    frames = args.frames or sorted(annotations.keys())
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # 构建 cam_id → camera_id (int) 的反向映射
    cam_str_to_int = {v: k for k, v in CAM_ID_MAP.items()}

    # Pre-compute valid combinations for accurate progress counting
    valid_combos = []
    for frame_id in frames:
        if frame_id not in annotations:
            continue
        frame_annots = annotations[frame_id]
        for cam_str in args.cameras:
            camera_id_int = cam_str_to_int.get(cam_str)
            if camera_id_int is None:
                continue
            cam_annots = [a for a in frame_annots if a.get('camera_id') == camera_id_int]
            bbox_list = [a.get('bbox', {}) for a in cam_annots if is_bbox_valid(a.get('bbox', {}))]
            if not bbox_list:
                continue
            img_path = Path(args.dataset_path) / "Image_subsets" / cam_str / f"{frame_id:08d}.png"
            if not img_path.exists():
                continue
            out_path = output_root / cam_str / f"{frame_id:04d}.png"
            if out_path.exists():
                continue
            valid_combos.append((frame_id, cam_str, camera_id_int))

    total = len(valid_combos)
    processed = 0

    for frame_id, cam_str, camera_id_int in valid_combos:
        processed += 1
        frame_annots = annotations[frame_id]
        cam_annots = [a for a in frame_annots if a.get('camera_id') == camera_id_int]
        bbox_list = [a.get('bbox', {}) for a in cam_annots if is_bbox_valid(a.get('bbox', {}))]

        # 加载图像
        img_path = Path(args.dataset_path) / "Image_subsets" / cam_str / f"{frame_id:08d}.png"
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # 输出路径
        out_path = output_root / cam_str / f"{frame_id:04d}.png"

        print(f"[{processed}/{total}] {cam_str}/{frame_id:04d} ({len(bbox_list)} bboxes)...")

        try:
            person_mask = generate_person_masks(image, bbox_list, mask_generator)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            mask_uint8 = (person_mask * 255).astype(np.uint8)
            Image.fromarray(mask_uint8).save(out_path)
        except Exception as e:
            print(f"  错误: {e}")

    print(f"\n完成，输出: {output_root}")


if __name__ == "__main__":
    main()
