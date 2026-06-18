#!/usr/bin/env python3
"""X1 Task 2: Multi-view feature fusion with leave-one-view-out.

Methods: naive_average, bbox_area_weighted, bbox_height_weighted.
Strict mode: gallery persons exclude query camera.
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

CAMERA_NAMES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance-table", default="outputs/x1_mv_fusion/instance_table.csv")
    p.add_argument("--features", default="outputs/v2_2d_reid_baseline/features.npz")
    p.add_argument("--output-dir", default="outputs/x1_mv_fusion")
    p.add_argument("--methods", nargs="+",
                   default=["naive_average", "bbox_area_weighted", "bbox_height_weighted"])
    return p.parse_args()


def load_instance_table(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        r["frame_id"] = int(r["frame_id"])
        r["person_id"] = int(r["person_id"])
        r["camera_index"] = int(r["camera_index"])
        r["view_num"] = int(r["view_num"])
        r["xmin"] = int(r["xmin"])
        r["ymin"] = int(r["ymin"])
        r["xmax"] = int(r["xmax"])
        r["ymax"] = int(r["ymax"])
        r["bbox_width"] = int(r["bbox_width"])
        r["bbox_height"] = int(r["bbox_height"])
        r["bbox_area"] = int(r["bbox_area"])
        r["bbox_valid"] = r["bbox_valid"] == "True"
        r["position_id"] = int(r["position_id"])
        r["feature_index"] = int(r["feature_index"])
        r["feature_norm"] = float(r["feature_norm"])
    return rows


def fuse_features_weighted(obs_features, weights):
    """Weighted average fusion + L2 normalize."""
    w = np.array(weights, dtype=np.float32)
    w = w / w.sum()
    fused = np.sum(w[:, None] * obs_features, axis=0)
    norm = np.linalg.norm(fused)
    if norm > 0:
        fused = fused / norm
    return fused


def fuse_features_mean(obs_features):
    """Naive average fusion + L2 normalize."""
    fused = np.mean(obs_features, axis=0)
    norm = np.linalg.norm(fused)
    if norm > 0:
        fused = fused / norm
    return fused


def build_fusion_data(instance_table, features, method):
    """Build query/gallery with leave-one-view-out fusion.

    For each valid observation (frame_id, person_id, camera_id=Cq):
      - query = single-view feature at Cq
      - positive gallery = fused feature from same (frame, person) excluding Cq
      - negative gallery = fused features from same frame, other persons, excluding Cq

    Returns dict with arrays.
    """
    valid_rows = [r for r in instance_table if r["bbox_valid"]]

    frame_person_groups = defaultdict(list)
    for r in valid_rows:
        frame_person_groups[(r["frame_id"], r["person_id"])].append(r)

    frame_persons = defaultdict(set)
    for r in valid_rows:
        frame_persons[r["frame_id"]].add(r["person_id"])

    query_features = []
    query_person_ids = []
    query_frame_ids = []
    query_camera_ids = []
    gallery_features = []
    gallery_person_ids = []
    gallery_frame_ids = []
    gallery_camera_excluded = []
    valid_query_mask = []

    for frame_id in sorted(frame_persons.keys()):
        person_ids_in_frame = sorted(frame_persons[frame_id])

        for person_id in person_ids_in_frame:
            group = frame_person_groups.get((frame_id, person_id), [])
            if len(group) < 2:
                continue

            for query_row in group:
                q_cam = query_row["camera_id"]
                q_feat = features[query_row["feature_index"]]

                other_rows = [r for r in group if r["camera_id"] != q_cam]
                if not other_rows:
                    valid_query_mask.append(False)
                    query_features.append(q_feat)
                    query_person_ids.append(person_id)
                    query_frame_ids.append(frame_id)
                    query_camera_ids.append(q_cam)
                    gallery_features.append(np.zeros_like(q_feat))
                    gallery_person_ids.append(-1)
                    gallery_frame_ids.append(frame_id)
                    gallery_camera_excluded.append(q_cam)
                    continue

                other_feats = np.array([features[r["feature_index"]] for r in other_rows])

                if method == "naive_average":
                    pos_fused = fuse_features_mean(other_feats)
                elif method == "bbox_area_weighted":
                    ws = [float(r["bbox_area"]) for r in other_rows]
                    pos_fused = fuse_features_weighted(other_feats, ws)
                elif method == "bbox_height_weighted":
                    ws = [float(r["bbox_height"]) for r in other_rows]
                    pos_fused = fuse_features_weighted(other_feats, ws)
                else:
                    raise ValueError(f"Unknown method: {method}")

                query_features.append(q_feat)
                query_person_ids.append(person_id)
                query_frame_ids.append(frame_id)
                query_camera_ids.append(q_cam)
                valid_query_mask.append(True)

                gallery_features.append(pos_fused)
                gallery_person_ids.append(person_id)
                gallery_frame_ids.append(frame_id)
                gallery_camera_excluded.append(q_cam)

                for neg_pid in person_ids_in_frame:
                    if neg_pid == person_id:
                        continue
                    neg_group = frame_person_groups.get((frame_id, neg_pid), [])
                    neg_other = [r for r in neg_group if r["camera_id"] != q_cam]
                    if not neg_other:
                        continue

                    neg_feats = np.array([features[r["feature_index"]] for r in neg_other])
                    if method == "naive_average":
                        neg_fused = fuse_features_mean(neg_feats)
                    elif method == "bbox_area_weighted":
                        ws = [float(r["bbox_area"]) for r in neg_other]
                        neg_fused = fuse_features_weighted(neg_feats, ws)
                    elif method == "bbox_height_weighted":
                        ws = [float(r["bbox_height"]) for r in neg_other]
                        neg_fused = fuse_features_weighted(neg_feats, ws)

                    gallery_features.append(neg_fused)
                    gallery_person_ids.append(neg_pid)
                    gallery_frame_ids.append(frame_id)
                    gallery_camera_excluded.append(q_cam)

    result = {
        "query_features": np.array(query_features, dtype=np.float32),
        "query_person_ids": np.array(query_person_ids, dtype=np.int32),
        "query_frame_ids": np.array(query_frame_ids, dtype=np.int32),
        "query_camera_ids": np.array(query_camera_ids),
        "gallery_features": np.array(gallery_features, dtype=np.float32),
        "gallery_person_ids": np.array(gallery_person_ids, dtype=np.int32),
        "gallery_frame_ids": np.array(gallery_frame_ids, dtype=np.int32),
        "gallery_camera_excluded": np.array(gallery_camera_excluded),
        "valid_query_mask": np.array(valid_query_mask, dtype=bool),
    }

    return result


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    instance_table = load_instance_table(args.instance_table)
    features = np.load(args.features)["features"]
    print(f"[LOAD] instance_table: {len(instance_table)} rows")
    print(f"[LOAD] features: {features.shape}")

    for method in args.methods:
        print(f"\n{'='*60}")
        print(f"METHOD: {method}")
        print(f"{'='*60}")

        data = build_fusion_data(instance_table, features, method)

        n_queries = len(data["query_features"])
        n_valid = data["valid_query_mask"].sum()
        n_gallery = len(data["gallery_features"])
        print(f"Total queries: {n_queries}")
        print(f"Valid queries: {n_valid}")
        print(f"Invalid queries (skipped): {n_queries - n_valid}")
        print(f"Gallery entries: {n_gallery}")
        print(f"Avg gallery per query: {n_gallery / max(1, n_valid):.1f}")

        out_path = os.path.join(args.output_dir, f"fused_features_{method}.npz")
        np.savez(out_path, **data)
        print(f"[SAVE] {out_path}")


if __name__ == "__main__":
    main()
