#!/usr/bin/env python3
"""X1 Task 1: Build instance_table.csv mapping features to annotations.

Each row = one single-view person observation (frame_id + person_id + camera_id).
Feature index comes from detections.csv row order (matches features.npz row order).
"""

import argparse
import csv
import json
import os
from collections import Counter

import numpy as np

CAMERA_NAMES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
VIEWNUM_TO_CAM = {i: c for i, c in enumerate(CAMERA_NAMES)}
CAM_TO_VIEWNUM = {c: i for i, c in enumerate(CAMERA_NAMES)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/data02/zhangrunxiang/data/Wildtrack")
    p.add_argument("--baseline-root", default="outputs/v2_2d_reid_baseline")
    p.add_argument("--output-dir", default="outputs/x1_mv_fusion")
    return p.parse_args()


def load_detections(baseline_root):
    path = os.path.join(baseline_root, "detections.csv")
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"[LOAD] detections.csv: {len(rows)} rows")
    print(f"[LOAD] Columns: {reader.fieldnames}")
    return rows


def load_features(baseline_root):
    path = os.path.join(baseline_root, "features.npz")
    data = np.load(path)
    feats = data["features"]
    print(f"[LOAD] features.npz: shape={feats.shape}, dtype={feats.dtype}")
    norms = np.linalg.norm(feats, axis=1)
    print(f"[LOAD] L2 norms: mean={norms.mean():.6f}, std={norms.std():.6f}, "
          f"min={norms.min():.6f}, max={norms.max():.6f}")
    is_l2 = abs(norms.mean() - 1.0) < 0.01
    print(f"[LOAD] Is L2 normalized: {is_l2}")
    if not is_l2:
        print("[ERROR] Features are NOT L2 normalized!")
    return feats, norms


def load_annotations(data_root):
    ann_dir = os.path.join(data_root, "annotations_positions")
    annotations = {}
    for fname in sorted(os.listdir(ann_dir)):
        if not fname.endswith(".json"):
            continue
        frame_id = int(fname.replace(".json", ""))
        with open(os.path.join(ann_dir, fname)) as f:
            annotations[frame_id] = json.load(f)
    print(f"[LOAD] annotations: {len(annotations)} frames")
    return annotations


def build_instance_table(detections, features, norms, annotations):
    rows = []

    for feat_idx, det in enumerate(detections):
        cam_id = det["camera_id"]
        frame_id = int(det["frame_id"])
        person_id = int(det["person_id"])
        bbox_str = det["bbox_xyxy_original"]
        bbox = eval(bbox_str)
        xmin, ymin, xmax, ymax = bbox

        bbox_valid = not (xmin < 0 and ymin < 0 and xmax < 0 and ymax < 0)

        if bbox_valid:
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin
            bbox_area = bbox_width * bbox_height
            if not (0 <= xmin <= 1920 and 0 <= xmax <= 1920 and
                    0 <= ymin <= 1080 and 0 <= ymax <= 1080):
                print(f"[WARN] bbox out of range: feat_idx={feat_idx} cam={cam_id} "
                      f"frame={frame_id} person={person_id} bbox={bbox}")
        else:
            bbox_width = 0
            bbox_height = 0
            bbox_area = 0

        view_num = CAM_TO_VIEWNUM.get(cam_id, -1)

        position_id = -1
        if frame_id in annotations:
            for person_ann in annotations[frame_id]:
                if person_ann["personID"] == person_id:
                    position_id = person_ann["positionID"]
                    break

        row = {
            "frame_id": frame_id,
            "frame_name": f"{frame_id:08d}",
            "person_id": person_id,
            "camera_id": cam_id,
            "camera_index": view_num,
            "view_num": view_num,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "bbox_width": bbox_width,
            "bbox_height": bbox_height,
            "bbox_area": bbox_area,
            "bbox_valid": bbox_valid,
            "position_id": position_id,
            "feature_index": feat_idx,
            "feature_norm": float(norms[feat_idx]),
        }
        rows.append(row)

    return rows


def print_stats(rows):
    print(f"\n{'='*60}")
    print(f"INSTANCE TABLE STATS")
    print(f"{'='*60}")
    print(f"Total observations: {len(rows)}")

    valid_rows = [r for r in rows if r["bbox_valid"]]
    print(f"Valid bbox observations: {len(valid_rows)}")
    print(f"Invalid bbox observations: {len(rows) - len(valid_rows)}")

    cam_counts = Counter(r["camera_id"] for r in valid_rows)
    print(f"\nPer-camera observation count:")
    for cam in CAMERA_NAMES:
        print(f"  {cam}: {cam_counts.get(cam, 0)}")

    frame_counts = Counter(r["frame_id"] for r in valid_rows)
    print(f"\nPer-frame person count: mean={np.mean(list(frame_counts.values())):.1f}, "
          f"min={min(frame_counts.values())}, max={max(frame_counts.values())}")

    person_counts = Counter(r["person_id"] for r in valid_rows)
    print(f"\nPer-personID observation count: mean={np.mean(list(person_counts.values())):.1f}, "
          f"min={min(person_counts.values())}, max={max(person_counts.values())}")
    print(f"Unique personIDs: {len(person_counts)}")

    multi_view = 0
    for (frame_id, person_id), group in _group_by_frame_person(valid_rows):
        if len(group) >= 2:
            multi_view += 1
    print(f"\nMulti-view (person seen in >=2 cameras in same frame): {multi_view}")

    no_position = sum(1 for r in valid_rows if r["position_id"] == -1)
    print(f"Observations without position_id: {no_position}")


def _group_by_frame_person(rows):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["frame_id"], r["person_id"])].append(r)
    return groups.items()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    detections = load_detections(args.baseline_root)
    features, norms = load_features(args.baseline_root)
    annotations = load_annotations(args.data_root)

    if len(detections) != features.shape[0]:
        print(f"[FATAL] detections ({len(detections)}) != features ({features.shape[0]})")
        return

    rows = build_instance_table(detections, features, norms, annotations)

    print_stats(rows)

    # Random sample of 20 rows
    print(f"\n{'='*60}")
    print("RANDOM SAMPLE (20 rows):")
    print(f"{'='*60}")
    indices = np.random.choice(len(rows), min(20, len(rows)), replace=False)
    indices.sort()
    for idx in indices:
        r = rows[idx]
        print(f"  [{r['feature_index']:4d}] frame={r['frame_id']:4d} cam={r['camera_id']} "
              f"pid={r['person_id']:4d} bbox=[{r['xmin']},{r['ymin']},{r['xmax']},{r['ymax']}] "
              f"area={r['bbox_area']:6.0f} valid={r['bbox_valid']} pos={r['position_id']}")

    # Save CSV
    csv_path = os.path.join(args.output_dir, "instance_table.csv")
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[SAVE] {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
