#!/usr/bin/env python3
"""
Phase 14: Preflight Data/Camera/Bbox Check

Purpose: Validate dataset, cameras, bboxes, and configuration before starting geometry training.

Output: outputs/phase14_clean_geometry/preflight/
"""

import json
import os
import sys
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from threedgrut.datasets import make as make_dataset


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 14 Preflight Check")
    parser.add_argument("--config", type=str, default="configs/apps/wildtrack_full_3dgut.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/phase14_clean_geometry/preflight")
    parser.add_argument("--num_samples_first50", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for cam_id in range(1, 8):
        (overlay_dir / f"C{cam_id}").mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "preflight.log"
    log_file = open(log_path, "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log("=" * 80)
    log("Phase 14: Preflight Data/Camera/Bbox Check")
    log("=" * 80)

    # Load config
    log(f"\nLoading config: {args.config}")
    base_config = OmegaConf.load(args.config)
    log(f"  downsample_factor: {base_config.dataset.downsample_factor}")
    log(f"  initialization.num_gaussians: {base_config.initialization.num_gaussians}")
    log(f"  path: {base_config.get('path', 'N/A')}")

    # Save config snapshot
    OmegaConf.save(base_config, output_dir / "config_snapshot.yaml")
    log(f"  Config snapshot saved")

    # Load dataset
    dataset_path = base_config.get('path', '')
    log(f"\nLoading dataset from: {dataset_path}")
    try:
        train_dataset, val_dataset = make_dataset(
            "wildtrack",
            base_config,
            ray_jitter=None,
        )
        log(f"  Dataset loaded successfully")
        log(f"  Train samples: {len(train_dataset)}")
        log(f"  Val samples: {len(val_dataset)}")
    except Exception as e:
        import traceback
        log(f"  ERROR: Failed to load dataset: {e}")
        log(traceback.format_exc())
        log("\nPREFLIGHT FAIL: Cannot load dataset. Stopping.")
        generate_fail_report(output_dir, f"Dataset load failed: {e}")
        log_file.close()
        return

    dataset = train_dataset

    # Check camera coverage
    log(f"\n{'=' * 60}")
    log("1. Camera Coverage Check")
    log(f"{'=' * 60}")

    cam_counts = defaultdict(int)
    cam_ids_found = set()
    expected_cams = {f"C{i}" for i in range(1, 8)}

    for idx in range(len(dataset)):
        sample = dataset[idx]
        cam_id = sample.get("camera_id", "UNKNOWN")
        cam_counts[cam_id] += 1
        cam_ids_found.add(cam_id)

    log(f"  Total samples: {len(dataset)}")
    log(f"  Cameras found: {sorted(cam_ids_found)}")

    missing_cams = expected_cams - cam_ids_found
    extra_cams = cam_ids_found - expected_cams

    if missing_cams:
        log(f"  WARNING: Missing expected cameras: {missing_cams}")
    if extra_cams:
        log(f"  WARNING: Unexpected cameras: {extra_cams}")

    for cam in sorted(cam_counts.keys()):
        log(f"    {cam}: {cam_counts[cam]} samples")

    # Camera intrinsics/extrinsics summary
    log(f"\n{'=' * 60}")
    log("2. Camera Intrinsics/Extrinsics Check")
    log(f"{'=' * 60}")

    intrinsics_path = output_dir / "camera_intrinsics_summary.csv"
    extrinsics_path = output_dir / "camera_extrinsics_summary.csv"

    with open(intrinsics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["camera_id", "image_w_original", "image_h_original", 
                        "image_w_padded", "image_h_padded", 
                        "render_w", "render_h",
                        "fx", "fy", "cx", "cy",
                        "downsample_factor"])

        for cam_id in sorted(cam_ids_found):
            sample = None
            for idx in range(len(dataset)):
                s = dataset[idx]
                if s.get("camera_id") == cam_id:
                    sample = s
                    break

            if sample is None:
                log(f"  WARNING: No sample for {cam_id}")
                continue

            intr = sample.get("intrinsics", {})
            img_w_orig = sample.get("img_width_original", 1920)
            img_h_orig = sample.get("img_height_original", 1088)
            
            downsample = base_config.dataset.downsample_factor
            render_w = img_w_orig // downsample
            render_h = img_h_orig // downsample

            focal = intr.get("focal_length", [0, 0])
            pp = intr.get("principal_point", [0, 0])
            if isinstance(focal, list):
                fx, fy = focal[0], focal[1]
            else:
                fx = float(focal) if hasattr(focal, '__float__') else 0
                fy = fx
            if isinstance(pp, list):
                cx, cy = pp[0], pp[1]
            else:
                cx = float(pp) if hasattr(pp, '__float__') else 0
                cy = cx

            writer.writerow([cam_id, img_w_orig, img_h_orig, img_w_orig, img_h_orig,
                           render_w, render_h, fx, fy, cx, cy, downsample])

            log(f"  {cam_id}: orig={img_w_orig}x{img_h_orig}, render={render_w}x{render_h}, "
                f"fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    log(f"  Intrinsics summary saved to: {intrinsics_path}")

    with open(extrinsics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["camera_id", "c2w_r00", "c2w_r01", "c2w_r02", "c2w_t0",
                        "c2w_r10", "c2w_r11", "c2w_r12", "c2w_t1",
                        "c2w_r20", "c2w_r21", "c2w_r22", "c2w_t2"])

        for cam_id in sorted(cam_ids_found):
            c2w = dataset.extrinsics.get(cam_id, None)
            if c2w is not None:
                c2w_np = np.array(c2w)
                writer.writerow([cam_id] + c2w_np[:3, :].flatten().tolist())

    log(f"  Extrinsics summary saved to: {extrinsics_path}")

    # Bbox analysis
    log(f"\n{'=' * 60}")
    log("3. Bbox Coordinate Check")
    log(f"{'=' * 60}")

    bbox_scale_first50_path = output_dir / "bbox_scale_first50.jsonl"
    suspicious_count = 0
    total_bbox_count = 0
    samples_with_bboxes = 0

    with open(bbox_scale_first50_path, "w") as f:
        sample_counter = 0
        for idx in range(len(dataset)):
            if sample_counter >= args.num_samples_first50:
                break

            sample = dataset[idx]
            cam_id = sample.get("camera_id")
            frame_idx = sample.get("frame_idx")
            instances = sample.get("instances", [])

            img_w_orig = sample.get("img_width_original", 1920)
            img_h_orig = sample.get("img_height_original", 1088)
            downsample = base_config.dataset.downsample_factor
            render_w = img_w_orig // downsample
            render_h = img_h_orig // downsample

            if not instances:
                continue

            has_any_bbox = False
            for inst in instances:
                if inst.get("bbox_xyxy_original") is not None or inst.get("bbox_xyxy") is not None:
                    has_any_bbox = True
                    break

            if not has_any_bbox:
                continue

            for inst in instances:
                if sample_counter >= args.num_samples_first50:
                    break

                bbox_original = inst.get("bbox_xyxy_original")
                bbox_downsampled = inst.get("bbox_xyxy")
                train_id = inst.get("train_id")
                person_id = inst.get("person_id")

                record = {
                    "sample_idx": idx,
                    "cam_id": cam_id,
                    "frame_id": frame_idx,
                    "person_id": person_id,
                    "train_id": train_id,
                    "image_size_original": f"{img_w_orig}x{img_h_orig}",
                    "render_size": f"{render_w}x{render_h}",
                }

                suspicious = False
                suspicious_reasons = []

                if bbox_original is not None:
                    if isinstance(bbox_original, (list, tuple)):
                        bbox_orig_arr = list(bbox_original)
                    else:
                        bbox_orig_arr = bbox_original.tolist() if hasattr(bbox_original, 'tolist') else list(bbox_original)

                    record["bbox_xyxy_original"] = bbox_orig_arr

                    xmax_orig = bbox_orig_arr[2]
                    ymax_orig = bbox_orig_arr[3]

                    if xmax_orig > img_w_orig * 1.1 or ymax_orig > img_h_orig * 1.1:
                        suspicious = True
                        suspicious_reasons.append("bbox_original exceeds image size")

                    bbox_render_scaled = [x / downsample for x in bbox_orig_arr]
                    record["bbox_render_scaled_from_original"] = bbox_render_scaled

                else:
                    suspicious = True
                    suspicious_reasons.append("missing_bbox_original")

                if bbox_downsampled is not None:
                    if isinstance(bbox_downsampled, (list, tuple)):
                        bbox_ds_arr = list(bbox_downsampled)
                    else:
                        bbox_ds_arr = bbox_downsampled.tolist() if hasattr(bbox_downsampled, 'tolist') else list(bbox_downsampled)

                    record["bbox_xyxy"] = bbox_ds_arr

                    if len(bbox_ds_arr) >= 4:
                        xmin_c = max(0, min(bbox_ds_arr[0], render_w - 1))
                        ymin_c = max(0, min(bbox_ds_arr[1], render_h - 1))
                        xmax_c = max(xmin_c + 1, min(bbox_ds_arr[2], render_w))
                        ymax_c = max(ymin_c + 1, min(bbox_ds_arr[3], render_h))
                        bbox_clamped = [xmin_c, ymin_c, xmax_c, ymax_c]
                        record["bbox_clamped"] = bbox_clamped
                        record["bbox_width"] = xmax_c - xmin_c
                        record["bbox_height"] = ymax_c - ymin_c

                        bw = xmax_c - xmin_c
                        bh = ymax_c - ymin_c
                        if bw <= 2 or bh <= 2:
                            suspicious = True
                            suspicious_reasons.append(f"tiny_bbox_after_clamp: {bw}x{bh}")

                        if xmin_c == 0 or ymin_c == 0 or xmax_c == render_w or ymax_c == render_h:
                            suspicious_reasons.append("bbox_touches_edge")

                        total_bbox_count += 1

                else:
                    suspicious = True
                    suspicious_reasons.append("missing_bbox_xyxy")

                record["scale_mode"] = "downsample_division"
                record["suspicious_reason"] = "; ".join(suspicious_reasons) if suspicious_reasons else "none"

                if suspicious:
                    suspicious_count += 1

                f.write(json.dumps(record) + "\n")
                sample_counter += 1

            samples_with_bboxes += 1

    log(f"  First {args.num_samples_first50} samples with bboxes processed")
    log(f"  Total bboxes checked: {total_bbox_count}")
    log(f"  Suspicious bboxes: {suspicious_count} ({suspicious_count / max(1, total_bbox_count) * 100:.1f}%)")
    log(f"  Saved to: {bbox_scale_first50_path}")

    # Generate overlays
    log(f"\n{'=' * 60}")
    log("4. Bbox Overlay Visualization")
    log(f"{'=' * 60}")

    overlay_generated = 0

    try:
        import cv2

        for cam_id_num in range(1, 8):
            cam_id = f"C{cam_id_num}"
            overlay_cam_dir = overlay_dir / cam_id
            samples_for_cam = 0

            for idx in range(len(dataset)):
                if samples_for_cam >= 3:
                    break

                sample = dataset[idx]
                if sample.get("camera_id") != cam_id:
                    continue

                img = sample.get("rgb")
                if img is None:
                    continue

                if hasattr(img, 'numpy'):
                    img_np = img.numpy()
                else:
                    img_np = np.array(img)

                if img_np.ndim == 3:
                    if img_np.shape[0] == 3:
                        img_np = np.transpose(img_np, (1, 2, 0))
                    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                else:
                    continue

                img_overlay = img_np.copy()
                instances = sample.get("instances", [])
                img_h, img_w = img_overlay.shape[:2]

                for inst in instances:
                    bbox_xyxy = inst.get("bbox_xyxy")
                    if bbox_xyxy is None:
                        continue

                    if isinstance(bbox_xyxy, (list, tuple)):
                        xmin, ymin, xmax, ymax = bbox_xyxy
                    else:
                        xmin, ymin, xmax, ymax = bbox_xyxy.tolist()

                    xmin, ymin = int(xmin), int(ymin)
                    xmax, ymax = int(xmax), int(ymax)

                    xmin = max(0, min(xmin, img_w - 1))
                    ymin = max(0, min(ymin, img_h - 1))
                    xmax = max(xmin + 1, min(xmax, img_w))
                    ymax = max(ymin + 1, min(ymax, img_h))

                    color = (0, 255, 0)
                    cv2.rectangle(img_overlay, (xmin, ymin), (xmax, ymax), color, 2)

                    train_id = inst.get("train_id", "?")
                    cv2.putText(img_overlay, str(train_id), 
                               (xmin, min(ymin - 5, img_h - 10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                frame_idx = sample.get("frame_idx", 0)
                overlay_path = str(overlay_cam_dir / f"frame{frame_idx:06d}_overlay.png")
                cv2.imwrite(overlay_path, cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR))
                overlay_generated += 1
                samples_for_cam += 1

            log(f"  {cam_id}: {samples_for_cam} overlays generated")

        log(f"  Total overlays: {overlay_generated}")
        log(f"  Overlays saved to: {overlay_dir}")

    except ImportError:
        log("  WARNING: OpenCV not available, skipping overlay generation")
    except Exception as e:
        import traceback
        log(f"  WARNING: Overlay generation failed: {e}")
        log(traceback.format_exc())

    # Teacher embedding availability
    log(f"\n{'=' * 60}")
    log("5. Teacher Embedding Availability")
    log(f"{'=' * 60}")

    total_instances = 0
    valid_teacher_count = 0
    missing_teacher_count = 0

    for idx in range(min(200, len(dataset))):
        sample = dataset[idx]
        instances = sample.get("instances", [])
        for inst in instances:
            total_instances += 1
            teacher_emb = inst.get("teacher_embedding")
            if teacher_emb is not None:
                valid_teacher_count += 1
            else:
                missing_teacher_count += 1

    teacher_ratio = valid_teacher_count / max(1, total_instances)
    log(f"  Checked: {total_instances} instances (first 200 samples)")
    log(f"  Valid teacher embeddings: {valid_teacher_count} ({teacher_ratio * 100:.1f}%)")
    log(f"  Missing teacher embeddings: {missing_teacher_count}")

    # Cam/person/frame consistency
    log(f"\n{'=' * 60}")
    log("6. Cam/Frame/Person Consistency Check")
    log(f"{'=' * 60}")

    cam_frame_person_map = defaultdict(set)
    mismatches = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        cam_id = sample.get("camera_id")
        frame_idx = sample.get("frame_idx")
        instances = sample.get("instances", [])

        for inst in instances:
            person_id = inst.get("person_id")
            train_id = inst.get("train_id")

            if person_id is not None and train_id is not None:
                key = (cam_id, frame_idx)
                cam_frame_person_map[key].add((person_id, train_id))

                if person_id != train_id:
                    mismatches += 1

    log(f"  Checked: {len(cam_frame_person_map)} unique (cam, frame) pairs")
    log(f"  person_id != train_id mismatches: {mismatches}")

    # Final summary
    log(f"\n{'=' * 60}")
    log("PREFLIGHT SUMMARY")
    log(f"{'=' * 60}")
    log(f"  Dataset: OK ({len(dataset)} samples)")
    log(f"  Cameras: {sorted(cam_ids_found)}")
    log(f"  Missing cameras: {missing_cams if missing_cams else 'None'}")
    log(f"  Total bboxes: {total_bbox_count}")
    suspicious_ratio = suspicious_count / max(1, total_bbox_count) * 100
    log(f"  Suspicious bboxes: {suspicious_count} ({suspicious_ratio:.1f}%)")
    log(f"  Teacher embedding ratio: {teacher_ratio * 100:.1f}%")
    log(f"  Overlays generated: {overlay_generated}")

    stop_conditions = []

    if missing_cams:
        stop_conditions.append(f"Missing cameras: {missing_cams}")

    if suspicious_ratio > 50:
        stop_conditions.append(f"Too many suspicious bboxes: {suspicious_ratio:.1f}%")

    if teacher_ratio < 0.5:
        stop_conditions.append(f"Low teacher embedding ratio: {teacher_ratio * 100:.1f}%")

    if stop_conditions:
        log(f"\nPREFLIGHT FAIL - Stop conditions:")
        for cond in stop_conditions:
            log(f"  - {cond}")
        generate_fail_report(output_dir, "; ".join(stop_conditions))
    else:
        log(f"\nPREFLIGHT PASS - Ready to proceed to smoke run")
        generate_pass_report(output_dir, base_config, len(dataset), sorted(cam_ids_found), 
                           total_bbox_count, suspicious_count, teacher_ratio)

    log_file.close()


def generate_fail_report(output_dir: Path, reason: str):
    report_path = output_dir / "preflight_report.md"
    with open(report_path, "w") as f:
        f.write("# Phase 14 Preflight Report\n\n")
        f.write("## Status: FAIL\n\n")
        f.write(f"## Reason\n\n{reason}\n\n")
        f.write("## Recommendation\n\n")
        f.write("Fix the identified issues before proceeding to smoke run.\n")


def generate_pass_report(output_dir: Path, config, num_samples: int, 
                        cam_ids: list, total_bbox: int, suspicious: int,
                        teacher_ratio: float):
    report_path = output_dir / "preflight_report.md"
    suspicious_ratio = suspicious / max(1, total_bbox) * 100
    with open(report_path, "w") as f:
        f.write("# Phase 14 Preflight Report\n\n")
        f.write("## Status: PASS\n\n")

        f.write("## Dataset Summary\n\n")
        f.write(f"- Total samples: {num_samples}\n")
        f.write(f"- Cameras: {', '.join(cam_ids)}\n")
        f.write(f"- Downsample factor: {config.dataset.downsample_factor}\n")
        f.write(f"- Initialization num_gaussians: {config.initialization.num_gaussians}\n\n")

        f.write("## Bbox Summary\n\n")
        f.write(f"- Total bboxes checked: {total_bbox}\n")
        f.write(f"- Suspicious bboxes: {suspicious} ({suspicious_ratio:.1f}%)\n")
        f.write(f"- Teacher embedding ratio: {teacher_ratio * 100:.1f}%\n\n")

        f.write("## Output Files\n\n")
        f.write("- `bbox_scale_first50.jsonl`: First 50 sample bbox records\n")
        f.write("- `camera_intrinsics_summary.csv`: Camera intrinsics summary\n")
        f.write("- `camera_extrinsics_summary.csv`: Camera extrinsics summary\n")
        f.write("- `overlays/C1-C7/`: Bbox overlay visualizations\n")
        f.write("- `config_snapshot.yaml`: Config snapshot\n\n")

        f.write("## Next Step\n\n")
        f.write("Proceed to Phase 14 Geometry Smoke Run.\n")


if __name__ == "__main__":
    main()
