#!/usr/bin/env python3
"""X1 Task 3: Leave-one-view-out ReID evaluation.

Evaluates: single_view, naive_average, bbox_area_weighted, bbox_height_weighted.
All methods computed on-the-fly from instance_table + features.
Metrics: mAP, Rank-1/5/10, per-camera breakdown.
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

CAMERA_NAMES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--instance-table", default="outputs/x1_mv_fusion/instance_table.csv")
    p.add_argument("--features", default="outputs/v2_2d_reid_baseline/features.npz")
    p.add_argument("--output-dir", default="outputs/x1_mv_fusion")
    return p.parse_args()


def load_instance_table(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        r["frame_id"] = int(r["frame_id"])
        r["person_id"] = int(r["person_id"])
        r["feature_index"] = int(r["feature_index"])
        r["bbox_valid"] = r["bbox_valid"] == "True"
        r["bbox_area"] = int(r["bbox_area"])
        r["bbox_height"] = int(r["bbox_height"])
    return rows


def fuse(obs_feats, weights=None):
    if weights is None:
        f = np.mean(obs_feats, axis=0)
    else:
        w = np.array(weights, dtype=np.float32)
        w = w / w.sum()
        f = np.sum(w[:, None] * obs_feats, axis=0)
    norm = np.linalg.norm(f)
    if norm > 0:
        f = f / norm
    return f


def evaluate_method(instance_table, features, method):
    """Evaluate a fusion method with leave-one-view-out protocol.

    Query: single-view feature from camera Cq.
    Gallery: one fused feature per person in same frame, excluding camera Cq.
    Positive: same person_id, fused from cameras != Cq.
    Negative: other person_ids, fused from cameras != Cq.
    """
    valid_rows = [r for r in instance_table if r["bbox_valid"]]

    frame_person_groups = defaultdict(list)
    for r in valid_rows:
        frame_person_groups[(r["frame_id"], r["person_id"])].append(r)

    frame_persons = defaultdict(list)
    for r in valid_rows:
        if r["person_id"] not in [p for p in frame_persons[r["frame_id"]]]:
            frame_persons[r["frame_id"]].append(r["person_id"])

    ranks = []
    aps = []
    skipped = 0
    gallery_sizes = []
    per_cam_ranks = defaultdict(list)
    per_cam_aps = defaultdict(list)

    for frame_id in sorted(frame_persons.keys()):
        person_ids_in_frame = frame_persons[frame_id]

        for person_id in person_ids_in_frame:
            group = frame_person_groups.get((frame_id, person_id), [])
            if len(group) < 2:
                continue

            for query_row in group:
                q_cam = query_row["camera_id"]
                q_feat = features[query_row["feature_index"]]

                other_views = [r for r in group if r["camera_id"] != q_cam]
                if not other_views:
                    skipped += 1
                    continue

                gallery = []
                for pid in person_ids_in_frame:
                    pid_group = frame_person_groups.get((frame_id, pid), [])
                    pid_other = [r for r in pid_group if r["camera_id"] != q_cam]
                    if not pid_other:
                        continue

                    pid_feats = np.array([features[r["feature_index"]] for r in pid_other])

                    if method == "single_view":
                        for r_idx, r in enumerate(pid_other):
                            gallery.append((pid, pid_feats[r_idx]))
                    elif method == "naive_average":
                        gallery.append((pid, fuse(pid_feats)))
                    elif method == "bbox_area_weighted":
                        ws = [float(r["bbox_area"]) for r in pid_other]
                        gallery.append((pid, fuse(pid_feats, ws)))
                    elif method == "bbox_height_weighted":
                        ws = [float(r["bbox_height"]) for r in pid_other]
                        gallery.append((pid, fuse(pid_feats, ws)))

                pos_in_gallery = [g for g in gallery if g[0] == person_id]
                if not pos_in_gallery:
                    skipped += 1
                    continue

                gallery_sizes.append(len(gallery))
                gallery_feats = np.array([g[1] for g in gallery])
                gallery_pids = [g[0] for g in gallery]

                sims = gallery_feats @ q_feat
                sorted_idx = np.argsort(-sims)
                sorted_pids = [gallery_pids[j] for j in sorted_idx]

                rank = None
                for r_idx, pid in enumerate(sorted_pids):
                    if pid == person_id:
                        rank = r_idx + 1
                        break

                if rank is not None:
                    ranks.append(rank)
                    per_cam_ranks[q_cam].append(rank)

                hits = 0
                precision_sum = 0.0
                for r_idx, j in enumerate(sorted_idx):
                    if gallery_pids[j] == person_id:
                        hits += 1
                        precision_sum += hits / (r_idx + 1)
                n_pos = len(pos_in_gallery)
                ap = precision_sum / max(1, n_pos)
                aps.append(ap)
                per_cam_aps[q_cam].append(ap)

    return {
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "Rank-1": float(np.mean([1 if r <= 1 else 0 for r in ranks])) if ranks else 0.0,
        "Rank-5": float(np.mean([1 if r <= 5 else 0 for r in ranks])) if ranks else 0.0,
        "Rank-10": float(np.mean([1 if r <= 10 else 0 for r in ranks])) if ranks else 0.0,
        "num_queries": len(ranks),
        "skipped_queries": skipped,
        "avg_gallery_size": float(np.mean(gallery_sizes)) if gallery_sizes else 0.0,
        "per_camera": {
            cam: {
                "mAP": float(np.mean(per_cam_aps[cam])) if per_cam_aps[cam] else 0.0,
                "Rank-1": float(np.mean([1 if r <= 1 else 0 for r in per_cam_ranks[cam]])) if per_cam_ranks[cam] else 0.0,
            }
            for cam in CAMERA_NAMES
        },
    }


def write_readme(results, output_dir):
    lines = []
    lines.append("# X1 Multi-View Fusion Baseline")
    lines.append("")
    lines.append("## 1. Experiment Protocol")
    lines.append("")
    lines.append("Leave-one-view-out evaluation on WildTrack dataset (7 cameras, 89 frames, 92 person IDs).")
    lines.append("")
    lines.append("## 2. Query / Gallery Construction")
    lines.append("")
    lines.append("- **Query**: single-view person feature from camera Cq")
    lines.append("- **Gallery**: for same frame, one feature per person (excluding camera Cq)")
    lines.append("- **Positive**: same person_id, feature(s) from cameras != Cq")
    lines.append("- **Negative**: other person_ids, feature(s) from cameras != Cq")
    lines.append("- If a person has no other camera views, the query is skipped")
    lines.append("- Similarity: cosine similarity (dot product of L2-normalized features)")
    lines.append("")
    lines.append("## 3. Leave-One-View-Out")
    lines.append("")
    lines.append("Yes. Gallery features for all persons exclude the query camera Cq.")
    lines.append("This prevents information leakage from the query view.")
    lines.append("")
    lines.append("## 4. Baseline Methods")
    lines.append("")
    lines.append("| Method | Gallery Feature | Description |")
    lines.append("|--------|----------------|-------------|")
    lines.append("| single_view | Individual view features | Each person-camera pair is a separate gallery entry |")
    lines.append("| naive_average | mean(f_c), L2-norm | Average all views, no weighting |")
    lines.append("| bbox_area_weighted | sum(area_c * f_c)/sum(area_c), L2-norm | Weight by bbox area |")
    lines.append("| bbox_height_weighted | sum(height_c * f_c)/sum(height_c), L2-norm | Weight by bbox height |")
    lines.append("")
    lines.append("## 5. Results")
    lines.append("")
    lines.append("| Method | mAP | Rank-1 | Rank-5 | Rank-10 | #Queries | Avg Gallery |")
    lines.append("|--------|-----|--------|--------|---------|----------|-------------|")

    for method in ["single_view", "naive_average", "bbox_area_weighted", "bbox_height_weighted"]:
        r = results.get(method, {})
        lines.append(f"| {method} | {r.get('mAP', 0):.4f} | {r.get('Rank-1', 0):.4f} | "
                     f"{r.get('Rank-5', 0):.4f} | {r.get('Rank-10', 0):.4f} | "
                     f"{r.get('num_queries', 0)} | {r.get('avg_gallery_size', 0):.1f} |")

    lines.append("")
    lines.append("### Per-Camera Results")
    lines.append("")

    for metric_name in ["mAP", "Rank-1"]:
        lines.append(f"#### {metric_name}")
        lines.append("")
        header = "| Method | " + " | ".join(CAMERA_NAMES) + " |"
        sep = "|--------|" + "|".join(["------" for _ in CAMERA_NAMES]) + "|"
        lines.append(header)
        lines.append(sep)
        for method in ["single_view", "naive_average", "bbox_area_weighted", "bbox_height_weighted"]:
            r = results.get(method, {})
            per_cam = r.get("per_camera", {})
            vals = [f"{per_cam.get(cam, {}).get(metric_name, 0):.4f}" for cam in CAMERA_NAMES]
            lines.append(f"| {method} | " + " | ".join(vals) + " |")
        lines.append("")

    lines.append("## 6. Preliminary Conclusions")
    lines.append("")

    sv = results.get("single_view", {})
    na = results.get("naive_average", {})
    ba = results.get("bbox_area_weighted", {})
    bh = results.get("bbox_height_weighted", {})

    na_beats_sv = na.get("mAP", 0) > sv.get("mAP", 0)
    ba_beats_na = ba.get("mAP", 0) > na.get("mAP", 0)
    bh_beats_na = bh.get("mAP", 0) > na.get("mAP", 0)

    if na_beats_sv:
        delta = (na['mAP'] - sv['mAP']) * 100
        lines.append(f"- naive_average (mAP={na['mAP']:.4f}) > single_view (mAP={sv['mAP']:.4f}): "
                     f"multi-view fusion helps (+{delta:.2f}% mAP)")
    else:
        delta = (sv['mAP'] - na['mAP']) * 100
        lines.append(f"- naive_average (mAP={na['mAP']:.4f}) <= single_view (mAP={sv['mAP']:.4f}): "
                     f"multi-view fusion does NOT help (-{delta:.2f}% mAP)")

    if ba_beats_na:
        lines.append(f"- bbox_area_weighted (mAP={ba['mAP']:.4f}) > naive_average (mAP={na['mAP']:.4f}): "
                     f"bbox area weighting helps")
    else:
        lines.append(f"- bbox_area_weighted (mAP={ba['mAP']:.4f}) <= naive_average (mAP={na['mAP']:.4f}): "
                     f"bbox area weighting does NOT help")

    if bh_beats_na:
        lines.append(f"- bbox_height_weighted (mAP={bh['mAP']:.4f}) > naive_average (mAP={na['mAP']:.4f}): "
                     f"bbox height weighting helps")
    else:
        lines.append(f"- bbox_height_weighted (mAP={bh['mAP']:.4f}) <= naive_average (mAP={na['mAP']:.4f}): "
                     f"bbox height weighting does NOT help")

    lines.append("")
    lines.append("## 7. Analysis (if naive_average < single_view)")
    lines.append("")
    if not na_beats_sv:
        lines.append("Possible reasons:")
        lines.append("1. Averaging dilutes discriminative features from the best view")
        lines.append("2. Low-quality views (occluded, small bbox) add noise")
        lines.append("3. L2 normalization after averaging changes the feature distribution")
        lines.append("4. Gallery size differs: single_view has multiple entries per person, "
                     "fused has one")
    else:
        lines.append("N/A - naive_average beats single_view")

    path = os.path.join(output_dir, "README_X1_BASELINE.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[SAVE] {path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    instance_table = load_instance_table(args.instance_table)
    features = np.load(args.features)["features"]
    print(f"[LOAD] instance_table: {len(instance_table)} rows")
    print(f"[LOAD] features: {features.shape}")

    all_results = {}

    for method in ["single_view", "naive_average", "bbox_area_weighted", "bbox_height_weighted"]:
        print(f"\n{'='*60}")
        print(f"EVALUATING: {method}")
        print(f"{'='*60}")
        metrics = evaluate_method(instance_table, features, method)
        all_results[method] = metrics
        print(f"  mAP={metrics['mAP']:.4f}  Rank-1={metrics['Rank-1']:.4f}  "
              f"Rank-5={metrics['Rank-5']:.4f}  Rank-10={metrics['Rank-10']:.4f}  "
              f"queries={metrics['num_queries']}  skipped={metrics['skipped_queries']}  "
              f"avg_gallery={metrics['avg_gallery_size']:.1f}")

    csv_path = os.path.join(args.output_dir, "eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "mAP", "Rank-1", "Rank-5", "Rank-10",
                                                "num_queries", "skipped_queries", "avg_gallery_size"])
        writer.writeheader()
        for method, m in all_results.items():
            writer.writerow({
                "method": method,
                "mAP": f"{m['mAP']:.6f}",
                "Rank-1": f"{m['Rank-1']:.6f}",
                "Rank-5": f"{m['Rank-5']:.6f}",
                "Rank-10": f"{m['Rank-10']:.6f}",
                "num_queries": m["num_queries"],
                "skipped_queries": m["skipped_queries"],
                "avg_gallery_size": f"{m['avg_gallery_size']:.2f}",
            })
    print(f"\n[SAVE] {csv_path}")

    json_path = os.path.join(args.output_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[SAVE] {json_path}")

    write_readme(all_results, args.output_dir)


if __name__ == "__main__":
    main()
