#!/usr/bin/env python3
"""X5: ReID Feature Cross-Frame Stability Analysis.

Step0: Multi-view aggregation headroom pre-check.
Step1: Cross-frame stability with F1/F2/F3 feature types.

Results saved to outputs/x5_stability/.
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

# ── Paths ──────────────────────────────────────────────────────────────
FEATURES_PATH = "outputs/v2_2d_reid_baseline/features.npz"
INSTANCE_TABLE = "outputs/x1_mv_fusion/instance_table.csv"
VISIBILITY_PATH = "outputs/x2_3dgs_visibility_fusion_bugfix/visibility_scores.csv"
OUTPUT_DIR = "outputs/x5_stability"

# ── Constants ──────────────────────────────────────────────────────────
CAMERA_IDS = [f"C{i}" for i in range(1, 8)]
FRAME_STEP = 5  # WildTrack: frame_id step = 5, so frame_id 0,5,10,...
FPS = 2.0       # WildTrack: 2 FPS
DELTA_IDS = [1, 2, 5, 10, 20]  # in frame_id units
# Actual frame gaps: delta_id * FRAME_STEP / FPS seconds


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════

def load_features():
    """Load L2-normalized ReID features. Shape: (N, 512)."""
    data = np.load(FEATURES_PATH)
    features = data["features"].astype(np.float32)
    # Re-normalize to be safe
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    features = features / norms
    print(f"[load] Features: {features.shape}, L2 norms: min={norms.min():.6f}, max={norms.max():.6f}")
    return features


def load_instance_table():
    """Load instance_table.csv. Returns list of dicts."""
    rows = []
    with open(INSTANCE_TABLE) as f:
        for i, r in enumerate(csv.DictReader(f)):
            r["feature_index"] = int(r["feature_index"])
            r["frame_id"] = int(r["frame_id"])
            r["person_id"] = int(r["person_id"])
            r["bbox_valid"] = r["bbox_valid"] == "True"
            r["xmin"] = int(r["xmin"])
            r["ymin"] = int(r["ymin"])
            r["xmax"] = int(r["xmax"])
            r["ymax"] = int(r["ymax"])
            r["bbox_area"] = int(r["bbox_area"])
            r["bbox_height"] = int(r["bbox_height"])
            # Sanity: feature_index == row index
            assert r["feature_index"] == i, f"feature_index mismatch at row {i}: {r['feature_index']} != {i}"
            rows.append(r)
    print(f"[load] Instance table: {len(rows)} rows")
    return rows


def load_visibility_scores(instance_table):
    """Load visibility_scores.csv. Returns dict keyed by (frame_id, person_id, camera_id)."""
    scores = {}
    if not os.path.exists(VISIBILITY_PATH):
        print(f"[load] Visibility scores not found: {VISIBILITY_PATH}")
        return scores
    with open(VISIBILITY_PATH) as f:
        for r in csv.DictReader(f):
            key = (int(r["frame_id"]), int(r["person_id"]), r["camera_id"])
            entry = {
                "v_bbox_occlusion": float(r.get("v_bbox_occlusion", 0)),
                "bbox_area": float(r.get("bbox_area", 0)),
                "bbox_height": float(r.get("bbox_height", 0)),
            }
            # Parse additional fields if present
            for k in ["v_depth_ratio_torso", "v_weighted_opacity_torso",
                       "world_x", "world_y"]:
                if k in r and r[k]:
                    try:
                        entry[k] = float(r[k])
                    except ValueError:
                        pass
            scores[key] = entry
    print(f"[load] Visibility scores: {len(scores)} entries")

    # Coverage check: how many valid instance_table rows have visibility data?
    valid_rows = [r for r in instance_table if r["bbox_valid"]]
    covered = sum(1 for r in valid_rows
                  if (r["frame_id"], r["person_id"], r["camera_id"]) in scores)
    total = len(valid_rows)
    print(f"[load] Visibility coverage: {covered}/{total} ({covered/max(total,1)*100:.1f}%)")
    if covered < total:
        print(f"[load] WARNING: {total - covered} rows missing visibility data. "
              f"F3 weights will default to 1.0 (F3 degrades to F2 for missing entries).")

    return scores


def build_index(instance_table):
    """Build lookup indices from instance_table."""
    # Key: (frame_id, person_id, camera_id) -> row
    by_fpc = {}
    # Key: person_id -> list of rows
    by_person = defaultdict(list)
    # Key: frame_id -> list of rows
    by_frame = defaultdict(list)
    # Key: (person_id, frame_id) -> list of rows (all cameras)
    by_pf = defaultdict(list)

    for r in instance_table:
        if not r["bbox_valid"]:
            continue
        key = (r["frame_id"], r["person_id"], r["camera_id"])
        by_fpc[key] = r
        by_person[r["person_id"]].append(r)
        by_frame[r["frame_id"]].append(r)
        by_pf[(r["person_id"], r["frame_id"])].append(r)

    return by_fpc, by_person, by_frame, by_pf


# ═══════════════════════════════════════════════════════════════════════
# Feature Type Constructors
# ═══════════════════════════════════════════════════════════════════════

def make_f1(features, rows):
    """F1: single-view bbox feature. Returns dict: (frame, person, cam) -> feature."""
    result = {}
    for r in rows:
        if not r["bbox_valid"]:
            continue
        key = (r["frame_id"], r["person_id"], r["camera_id"])
        result[key] = features[r["feature_index"]].copy()
    return result


def make_f2(features, by_pf):
    """F2: same-frame multi-view average. Returns dict: (frame, person) -> feature."""
    result = {}
    for (pid, fid), rows in by_pf.items():
        feats = [features[r["feature_index"]] for r in rows]
        avg = np.mean(feats, axis=0)
        avg = avg / max(np.linalg.norm(avg), 1e-8)
        result[(fid, pid)] = avg
    return result


def make_f3(features, by_pf, visibility_scores):
    """F3: same-frame quality-weighted average (1 - occlusion as weight).
    Returns dict: (frame, person) -> feature."""
    result = {}
    for (pid, fid), rows in by_pf.items():
        feats = []
        weights = []
        for r in rows:
            feat = features[r["feature_index"]]
            key = (fid, pid, r["camera_id"])
            vs = visibility_scores.get(key, {})
            occ = vs.get("v_bbox_occlusion", 0.0)
            w = max(1.0 - occ, 1e-8)
            feats.append(feat)
            weights.append(w)
        feats = np.array(feats)
        weights = np.array(weights)
        weights = weights / weights.sum()
        avg = (weights[:, None] * feats).sum(axis=0)
        avg = avg / max(np.linalg.norm(avg), 1e-8)
        result[(fid, pid)] = avg
    return result


# ═══════════════════════════════════════════════════════════════════════
# Step0: Multi-View Aggregation Headroom
# ═══════════════════════════════════════════════════════════════════════

def step0_headroom(features, by_pf, visibility_scores, by_fpc):
    """Quantify multi-view feature dispersion and headroom for aggregation."""
    print("\n" + "=" * 70)
    print("STEP 0: MULTI-VIEW AGGREGATION HEADROOM")
    print("=" * 70)

    dispersions = []  # per (person, frame)
    per_entry = []    # detailed records

    for (pid, fid), rows in by_pf.items():
        if len(rows) < 2:
            continue
        feats = np.array([features[r["feature_index"]] for r in rows])
        # Pairwise cosine distance = 1 - cosine_similarity
        # Since features are L2-normalized, dot = cosine_sim
        sim_matrix = feats @ feats.T
        # Upper triangle (exclude diagonal)
        n = len(feats)
        pairwise_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                pairwise_sims.append(sim_matrix[i, j])
        pairwise_dists = [1.0 - s for s in pairwise_sims]

        # Distance to mean
        mean_feat = feats.mean(axis=0)
        mean_feat = mean_feat / max(np.linalg.norm(mean_feat), 1e-8)
        dists_to_mean = [1.0 - float(f @ mean_feat) for f in feats]

        mean_pair_dist = np.mean(pairwise_dists)
        max_pair_dist = np.max(pairwise_dists)
        var_to_mean = np.var(dists_to_mean)

        dispersions.append({
            "person_id": pid, "frame_id": fid,
            "n_views": n,
            "mean_pair_dist": mean_pair_dist,
            "max_pair_dist": max_pair_dist,
            "var_to_mean": var_to_mean,
            "dists_to_mean": dists_to_mean,
        })
        per_entry.append((pid, fid, mean_pair_dist, max_pair_dist, n))

    print(f"  Total (person, frame) groups with >=2 views: {len(dispersions)}")

    mean_dists = [d["mean_pair_dist"] for d in dispersions]
    max_dists = [d["max_pair_dist"] for d in dispersions]

    print(f"\n  Pairwise cosine distance distribution:")
    for q in [0, 10, 25, 50, 75, 90, 95, 100]:
        print(f"    P{q}: mean_dist={np.percentile(mean_dists, q):.4f}, "
              f"max_dist={np.percentile(max_dists, q):.4f}")

    # Absolute dispersion assessment
    p90 = np.percentile(mean_dists, 90)
    p95 = np.percentile(mean_dists, 95)
    print(f"\n  Absolute dispersion: P90={p90:.4f}, P95={p95:.4f}")
    if p90 < 0.1:
        print(f"  → P90 < 0.1: views are highly consistent in 90%+ cases → aggregation headroom is SMALL")
    else:
        print(f"  → P90 >= 0.1: non-trivial dispersion exists → aggregation may have headroom")

    # Split into low/high dispersion for recall comparison
    median_dist = np.median(mean_dists)
    low_disp = [d for d in dispersions if d["mean_pair_dist"] <= median_dist]
    high_disp = [d for d in dispersions if d["mean_pair_dist"] > median_dist]
    print(f"\n  Median mean_pair_dist: {median_dist:.4f}")
    print(f"  Low dispersion (<=median): {len(low_disp)} groups")
    print(f"  High dispersion (>median): {len(high_disp)} groups")

    # Recall@1 on same-frame retrieval for each subset
    low_recall, low_n = _same_frame_retrieval(features, by_pf, low_disp)
    high_recall, high_n = _same_frame_retrieval(features, by_pf, high_disp)
    print(f"  Low dispersion Recall@1:  {low_recall:.4f} ({low_n} queries)")
    print(f"  High dispersion Recall@1: {high_recall:.4f} ({high_n} queries)")
    recall_drop = low_recall - high_recall
    print(f"  Recall drop (low - high): {recall_drop:.4f}")
    if recall_drop < 0.02:
        print(f"  → Drop < 2%: even high-dispersion cases are easy → aggregation headroom is SMALL")
    else:
        print(f"  → Drop >= 2%: high-dispersion cases are harder → aggregation has headroom on hard cases")

    # Plot dispersion histogram
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(mean_dists, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(median_dist, color="red", linestyle="--", label=f"median={median_dist:.4f}")
    axes[0].set_xlabel("Mean pairwise cosine distance")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Step0: Multi-view dispersion distribution")
    axes[0].legend()

    axes[1].hist(max_dists, bins=50, edgecolor="black", alpha=0.7, color="orange")
    axes[1].set_xlabel("Max pairwise cosine distance")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Step0: Max pairwise dispersion")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step0_dispersion_hist.png"), dpi=150)
    plt.close()
    print(f"  Saved: step0_dispersion_hist.png")

    # Save dispersion data
    with open(os.path.join(OUTPUT_DIR, "step0_dispersion.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["person_id", "frame_id", "n_views", "mean_pair_dist", "max_pair_dist"])
        for pid, fid, md, mxd, nv in per_entry:
            writer.writerow([pid, fid, nv, f"{md:.6f}", f"{mxd:.6f}"])

    return dispersions, median_dist


def _same_frame_retrieval(features, by_pf, subset):
    """Same-frame ID retrieval: query one view, gallery = same frame other persons."""
    hits = 0
    total = 0
    subset_set = set((d["person_id"], d["frame_id"]) for d in subset)

    # Pre-build frame_persons cache: fid -> {pid: [feature_indices]}
    frame_persons_cache = defaultdict(lambda: defaultdict(list))
    for (pid, fid), rows in by_pf.items():
        for r in rows:
            frame_persons_cache[fid][pid].append(r["feature_index"])

    for (pid, fid), rows in by_pf.items():
        if (pid, fid) not in subset_set:
            continue
        if len(rows) < 2:
            continue

        for r in rows:
            query_feat = features[r["feature_index"]]
            query_cam = r["camera_id"]
            # Gallery: all other persons' average feature (excluding query camera for same person)
            best_sim = -1.0
            best_pid = -1
            for p2, feat_indices in frame_persons_cache[fid].items():
                if p2 == pid:
                    # Exclude query camera
                    other_indices = [rr["feature_index"] for rr in by_pf[(pid, fid)]
                                   if rr["camera_id"] != query_cam]
                    if not other_indices:
                        continue
                    gal_feat = np.mean(features[other_indices], axis=0)
                else:
                    gal_feat = np.mean(features[feat_indices], axis=0)
                gal_feat = gal_feat / max(np.linalg.norm(gal_feat), 1e-8)
                sim = float(query_feat @ gal_feat)
                if sim > best_sim:
                    best_sim = sim
                    best_pid = p2

            if best_pid == pid:
                hits += 1
            total += 1

    return hits / max(total, 1), total


# ═══════════════════════════════════════════════════════════════════════
# Step1: Cross-Frame Stability
# ═══════════════════════════════════════════════════════════════════════

def sample_pairs(by_pf, by_frame, by_person, instance_table, visibility_scores):
    """Sample positive and negative pairs for cross-frame analysis."""
    # Get all frame_ids sorted
    all_frame_ids = sorted(set(r["frame_id"] for r in instance_table if r["bbox_valid"]))
    frame_id_set = set(all_frame_ids)

    # Positive pairs: same person_id, different frames
    pos_pairs = []  # (pid, fid_t, fid_s, delta_id)
    for pid, rows in by_person.items():
        person_frames = sorted(set(r["frame_id"] for r in rows))
        for fid_t in person_frames:
            for delta in DELTA_IDS:
                fid_s = fid_t + delta * FRAME_STEP
                if fid_s in frame_id_set and (pid, fid_s) in by_pf:
                    pos_pairs.append((pid, fid_t, fid_s, delta))

    # Negative pairs: different person_id
    # Include hard negatives: spatial proximity, same camera bbox overlap, same time
    neg_pairs_random = []
    neg_pairs_hard = []

    # Build per-frame person list with world-space positions
    # Use world_x/world_y from visibility_scores (POM grid decoded position)
    frame_person_world_pos = {}  # fid -> {pid: (world_x, world_y)}
    for fid, rows in by_frame.items():
        person_pos = {}
        for r in rows:
            pid = r["person_id"]
            if pid in person_pos:
                continue
            key = (fid, pid, r["camera_id"])
            vs = visibility_scores.get(key, {})
            wx = vs.get("world_x")
            wy = vs.get("world_y")
            if wx is not None and wy is not None:
                person_pos[pid] = (wx, wy)
        frame_person_world_pos[fid] = person_pos

    # For each positive pair, sample negatives
    rng = np.random.RandomState(42)
    all_pids = sorted(by_person.keys())

    for pid, fid_t, fid_s, delta in pos_pairs:
        # Random negatives: different person in fid_s
        other_pids_s = [p for p in by_pf if p[1] == fid_s and p[0] != pid]
        if other_pids_s:
            # Sample 1 random negative
            neg_key = other_pids_s[rng.randint(len(other_pids_s))]
            neg_pairs_random.append((pid, neg_key[0], fid_t, fid_s, delta))

        # Hard negatives: spatial proximity in world coordinates
        # Use pid's position in fid_s if available, otherwise fid_t
        pos_ref = frame_person_world_pos.get(fid_s, {}).get(pid)
        if pos_ref is None:
            pos_ref = frame_person_world_pos.get(fid_t, {}).get(pid)
        if pos_ref is None:
            continue
        # Find persons in fid_s closest to pid's reference position (world coords, cm)
        dists = []
        for p2 in frame_person_world_pos.get(fid_s, {}):
            if p2 == pid:
                continue
            pos_p2 = frame_person_world_pos[fid_s][p2]
            d = np.sqrt((pos_ref[0] - pos_p2[0])**2 + (pos_ref[1] - pos_p2[1])**2)
            dists.append((d, p2))
        if dists:
            dists.sort()
            # Top-1 closest as hard negative
            hard_pid = dists[0][1]
            neg_pairs_hard.append((pid, hard_pid, fid_t, fid_s, delta, dists[0][0]))

    print(f"\n  Positive pairs: {len(pos_pairs)}")
    print(f"  Random negative pairs: {len(neg_pairs_random)}")
    print(f"  Hard negative pairs (spatial): {len(neg_pairs_hard)}")

    # World coordinate coverage statistics
    total_pf = len(by_pf)  # total (person, frame) groups
    pf_with_world = 0
    for fid, person_pos in frame_person_world_pos.items():
        pf_with_world += len(person_pos)
    world_cov = pf_with_world / max(total_pf, 1) * 100
    print(f"\n  World coordinate coverage: {pf_with_world}/{total_pf} ({world_cov:.1f}%)")
    if world_cov < 90:
        print(f"  WARNING: world coordinate coverage < 90%. "
              f"Hard negatives may be under-sampled due to missing positions, "
              f"not because of genuinely few spatial neighbors.")
    print(f"  Hard/Positive ratio: {len(neg_pairs_hard)}/{len(pos_pairs)} "
          f"= {len(neg_pairs_hard)/max(len(pos_pairs),1):.2f}")

    for delta in DELTA_IDS:
        n_pos = sum(1 for p in pos_pairs if p[3] == delta)
        print(f"    Δ={delta} (frame_id gap={delta*FRAME_STEP}, "
              f"time={delta*FRAME_STEP/FPS:.1f}s): {n_pos} positive pairs")

    return pos_pairs, neg_pairs_random, neg_pairs_hard


def compute_pair_cosine(feat_dict, key_t, key_s):
    """Compute cosine similarity between two features from a dict."""
    f_t = feat_dict.get(key_t)
    f_s = feat_dict.get(key_s)
    if f_t is None or f_s is None:
        return None
    return float(f_t @ f_s)


def step1_stability(features, by_pf, by_person, by_frame, instance_table,
                    visibility_scores):
    """Cross-frame stability analysis with F1/F2/F3."""
    print("\n" + "=" * 70)
    print("STEP 1: CROSS-FRAME STABILITY")
    print("=" * 70)

    # Build feature dicts
    f1_dict = make_f1(features, instance_table)
    f2_dict = make_f2(features, by_pf)
    f3_dict = make_f3(features, by_pf, visibility_scores)

    # Sample pairs
    pos_pairs, neg_random, neg_hard = sample_pairs(by_pf, by_frame, by_person, instance_table, visibility_scores)

    # ── Compute cosine similarities ────────────────────────────────────
    results = {"F1": {}, "F2": {}, "F3": {}}

    for ftype, fdict, key_fn in [
        ("F1", f1_dict, lambda pid, fid, cam: (fid, pid, cam)),
        ("F2", f2_dict, lambda pid, fid, cam: (fid, pid)),
        ("F3", f3_dict, lambda pid, fid, cam: (fid, pid)),
    ]:
        print(f"\n  Computing {ftype}...")

        # Positive pairs
        pos_sims_by_delta = defaultdict(list)
        pos_sims_all = []
        pos_details = []

        for pid, fid_t, fid_s, delta in pos_pairs:
            if ftype == "F1":
                # F1 = per-camera cross-frame pairs
                # Each common camera produces an independent sample.
                # NO averaging within a pair.
                cams_t = {r["camera_id"] for r in by_pf[(pid, fid_t)]}
                cams_s = {r["camera_id"] for r in by_pf[(pid, fid_s)]}
                common_cams = cams_t & cams_s
                for cam in common_cams:
                    sim = compute_pair_cosine(fdict, key_fn(pid, fid_t, cam),
                                              key_fn(pid, fid_s, cam))
                    if sim is not None:
                        pos_sims_by_delta[delta].append(sim)
                        pos_sims_all.append(sim)
                        pos_details.append((pid, fid_t, fid_s, delta, sim))
            else:
                sim = compute_pair_cosine(fdict, key_fn(pid, fid_t, None),
                                          key_fn(pid, fid_s, None))
                if sim is not None:
                    pos_sims_by_delta[delta].append(sim)
                    pos_sims_all.append(sim)
                    pos_details.append((pid, fid_t, fid_s, delta, sim))

        # Random negative pairs
        neg_rand_sims = []
        for pid, neg_pid, fid_t, fid_s, delta in neg_random:
            if ftype == "F1":
                # Per-camera: c must be in both query person's and neg person's cameras
                cams_query = {r["camera_id"] for r in by_pf[(pid, fid_t)]}
                cams_neg = {r["camera_id"] for r in by_pf.get((neg_pid, fid_s), [])}
                common_cams = cams_query & cams_neg
                for cam in common_cams:
                    sim = compute_pair_cosine(fdict, key_fn(pid, fid_t, cam),
                                              key_fn(neg_pid, fid_s, cam))
                    if sim is not None:
                        neg_rand_sims.append(sim)
            else:
                sim = compute_pair_cosine(fdict, key_fn(pid, fid_t, None),
                                          key_fn(neg_pid, fid_s, None))
                if sim is not None:
                    neg_rand_sims.append(sim)

        # Hard negative pairs
        neg_hard_sims = []
        for pid, neg_pid, fid_t, fid_s, delta, _dist in neg_hard:
            if ftype == "F1":
                # Per-camera: c must be in both query person's and neg person's cameras
                cams_query = {r["camera_id"] for r in by_pf[(pid, fid_t)]}
                cams_neg = {r["camera_id"] for r in by_pf.get((neg_pid, fid_s), [])}
                common_cams = cams_query & cams_neg
                for cam in common_cams:
                    sim = compute_pair_cosine(fdict, key_fn(pid, fid_t, cam),
                                              key_fn(neg_pid, fid_s, cam))
                    if sim is not None:
                        neg_hard_sims.append(sim)
            else:
                sim = compute_pair_cosine(fdict, key_fn(pid, fid_t, None),
                                          key_fn(neg_pid, fid_s, None))
                if sim is not None:
                    neg_hard_sims.append(sim)

        results[ftype] = {
            "pos_sims_all": pos_sims_all,
            "pos_sims_by_delta": dict(pos_sims_by_delta),
            "neg_rand_sims": neg_rand_sims,
            "neg_hard_sims": neg_hard_sims,
            "pos_details": pos_details,
        }

        # Print summary
        if pos_sims_all:
            print(f"    Pos cosine: mean={np.mean(pos_sims_all):.4f}, "
                  f"std={np.std(pos_sims_all):.4f}, "
                  f"P5={np.percentile(pos_sims_all, 5):.4f}")
        if neg_rand_sims:
            print(f"    Neg(rand) cosine: mean={np.mean(neg_rand_sims):.4f}, "
                  f"std={np.std(neg_rand_sims):.4f}")
        if neg_hard_sims:
            print(f"    Neg(hard) cosine: mean={np.mean(neg_hard_sims):.4f}, "
                  f"std={np.std(neg_hard_sims):.4f}")

        # Margin & AUC
        if pos_sims_all and neg_rand_sims:
            margin = np.mean(pos_sims_all) - np.mean(neg_rand_sims)
            labels = [1] * len(pos_sims_all) + [0] * len(neg_rand_sims)
            scores = pos_sims_all + neg_rand_sims
            try:
                auc = roc_auc_score(labels, scores)
            except ValueError:
                auc = float("nan")
            print(f"    Margin(pos-neg_rand): {margin:.4f}, AUC: {auc:.4f}")

        if pos_sims_all and neg_hard_sims:
            margin_hard = np.mean(pos_sims_all) - np.mean(neg_hard_sims)
            labels = [1] * len(pos_sims_all) + [0] * len(neg_hard_sims)
            scores = pos_sims_all + neg_hard_sims
            try:
                auc_hard = roc_auc_score(labels, scores)
            except ValueError:
                auc_hard = float("nan")
            print(f"    Margin(pos-neg_hard): {margin_hard:.4f}, AUC(hard): {auc_hard:.4f}")

    # ── Plots ──────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Pos/Neg cosine distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, ftype in enumerate(["F1", "F2", "F3"]):
        ax = axes[idx]
        r = results[ftype]
        if r["pos_sims_all"]:
            ax.hist(r["pos_sims_all"], bins=50, alpha=0.6, label="Positive", density=True, color="blue")
        if r["neg_rand_sims"]:
            ax.hist(r["neg_rand_sims"], bins=50, alpha=0.6, label="Neg(random)", density=True, color="red")
        if r["neg_hard_sims"]:
            ax.hist(r["neg_hard_sims"], bins=50, alpha=0.4, label="Neg(hard)", density=True, color="orange")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Density")
        ax.set_title(f"{ftype}: Pos vs Neg distribution")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step1_pos_neg_dist.png"), dpi=150)
    plt.close()

    # 2. Cosine vs Δ decay curve (x-axis = frame_id gap)
    fig, ax = plt.subplots(figsize=(8, 5))
    for ftype in ["F1", "F2", "F3"]:
        r = results[ftype]
        deltas = sorted(r["pos_sims_by_delta"].keys())
        means = [np.mean(r["pos_sims_by_delta"][d]) for d in deltas]
        stds = [np.std(r["pos_sims_by_delta"][d]) for d in deltas]
        delta_frame_gaps = [d * FRAME_STEP for d in deltas]
        ax.errorbar(delta_frame_gaps, means, yerr=stds, marker="o", label=ftype, capsize=3)
    ax.set_xlabel("Frame ID gap (actual time interval uncertain, ~2.5-50s)")
    ax.set_ylabel("Same-ID cosine similarity")
    ax.set_title("Step1: Cross-frame feature decay")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step1_decay_curve.png"), dpi=150)
    plt.close()

    # ── Recall@1 cross-frame retrieval ─────────────────────────────────
    # Gallery always uses F2 (multi-view average) features for fairness.
    # Iterate over pos_pairs (unique person-frame pairs), NOT pos_details
    # (which is per-camera expanded for F1 — using it would double-count).
    # F1: per-camera single-view query (each camera at fid_t is a separate trial).
    # F2/F3: aggregated feature at fid_t as query.
    print("\n  Cross-frame Recall@1 (gallery=F2 aggregated):")
    for ftype, fdict, key_fn in [
        ("F1", f1_dict, lambda pid, fid, cam: (fid, pid, cam)),
        ("F2", f2_dict, lambda pid, fid, cam: (fid, pid)),
        ("F3", f3_dict, lambda pid, fid, cam: (fid, pid)),
    ]:
        for delta in DELTA_IDS:
            hits = 0
            total = 0
            for pid, fid_t, fid_s, d in pos_pairs:
                if d != delta:
                    continue
                if ftype == "F1":
                    # Per-camera: each camera at fid_t is a separate query
                    rows_t = by_pf.get((pid, fid_t), [])
                    for r_t in rows_t:
                        query_feat = fdict.get(key_fn(pid, fid_t, r_t["camera_id"]))
                        if query_feat is None:
                            continue
                        best_sim = -1.0
                        best_pid = -1
                        for (p2, f2) in by_pf:
                            if f2 != fid_s:
                                continue
                            gal_feat = f2_dict.get((fid_s, p2))
                            if gal_feat is not None:
                                sim = float(query_feat @ gal_feat)
                                if sim > best_sim:
                                    best_sim = sim
                                    best_pid = p2
                        if best_pid == pid:
                            hits += 1
                        total += 1
                else:
                    query_key = key_fn(pid, fid_t, None)
                    query_feat = fdict.get(query_key)
                    if query_feat is None:
                        continue
                    best_sim = -1.0
                    best_pid = -1
                    for (p2, f2) in by_pf:
                        if f2 != fid_s:
                            continue
                        gal_feat = f2_dict.get((fid_s, p2))
                        if gal_feat is not None:
                            sim = float(query_feat @ gal_feat)
                            if sim > best_sim:
                                best_sim = sim
                                best_pid = p2
                    if best_pid == pid:
                        hits += 1
                    total += 1

            recall = hits / max(total, 1)
            print(f"    {ftype} Δ={delta} (frame_gap={delta*FRAME_STEP}): "
                  f"Recall@1={recall:.4f} ({total} queries)")

    # ── Stratified analysis ────────────────────────────────────────────
    print("\n  Stratified analysis (F2):")
    r = results["F2"]

    # By occlusion
    occ_bins = [(0, 0.1, "low_occ"), (0.1, 0.5, "mid_occ"), (0.5, 1.01, "high_occ")]
    for lo, hi, label in occ_bins:
        sims = []
        for pid, fid_t, fid_s, delta, sim in r["pos_details"]:
            # Get occlusion for pid at fid_t
            key_t = (fid_t, pid)
            rows_t = by_pf.get((pid, fid_t), [])
            occ_vals = []
            for rr in rows_t:
                vs = visibility_scores.get((fid_t, pid, rr["camera_id"]), {})
                occ = vs.get("v_bbox_occlusion", 0.0)
                occ_vals.append(occ)
            avg_occ = np.mean(occ_vals) if occ_vals else 0.0
            if lo <= avg_occ < hi:
                sims.append(sim)
        if sims:
            print(f"    {label} (occ [{lo:.1f},{hi:.1f})): mean={np.mean(sims):.4f}, n={len(sims)}")

    # By bbox area
    area_vals = []
    for pid, fid_t, fid_s, delta, sim in r["pos_details"]:
        rows_t = by_pf.get((pid, fid_t), [])
        areas = [rr["bbox_area"] for rr in rows_t]
        area_vals.append((np.mean(areas) if areas else 0, sim))
    if area_vals:
        areas_arr = np.array([a for a, _ in area_vals])
        p33, p66 = np.percentile(areas_arr, [33, 66])
        for lo, hi, label in [(0, p33, "small"), (p33, p66, "medium"), (p66, 1e9, "large")]:
            sims = [s for a, s in area_vals if lo <= a < hi]
            if sims:
                print(f"    {label} (area [{lo:.0f},{hi:.0f})): mean={np.mean(sims):.4f}, n={len(sims)}")

    # By view count
    for nv in [2, 3, 4, 5, 6, 7]:
        sims = []
        for pid, fid_t, fid_s, delta, sim in r["pos_details"]:
            n = len(by_pf.get((pid, fid_t), []))
            if n == nv:
                sims.append(sim)
        if sims:
            print(f"    {nv} views: mean={np.mean(sims):.4f}, n={len(sims)}")

    # ── Save results ───────────────────────────────────────────────────
    # Summary table
    summary = {}
    for ftype in ["F1", "F2", "F3"]:
        r = results[ftype]
        pos = r["pos_sims_all"]
        neg_r = r["neg_rand_sims"]
        neg_h = r["neg_hard_sims"]
        entry = {
            "pos_mean": float(np.mean(pos)) if pos else None,
            "pos_std": float(np.std(pos)) if pos else None,
            "neg_rand_mean": float(np.mean(neg_r)) if neg_r else None,
            "neg_hard_mean": float(np.mean(neg_h)) if neg_h else None,
        }
        if pos and neg_r:
            entry["margin_rand"] = float(np.mean(pos) - np.mean(neg_r))
            labels = [1] * len(pos) + [0] * len(neg_r)
            scores = pos + neg_r
            try:
                entry["auc_rand"] = float(roc_auc_score(labels, scores))
            except ValueError:
                entry["auc_rand"] = None
        if pos and neg_h:
            entry["margin_hard"] = float(np.mean(pos) - np.mean(neg_h))
            labels = [1] * len(pos) + [0] * len(neg_h)
            scores = pos + neg_h
            try:
                entry["auc_hard"] = float(roc_auc_score(labels, scores))
            except ValueError:
                entry["auc_hard"] = None
        # Per-delta
        entry["per_delta"] = {}
        for delta in DELTA_IDS:
            ds = r["pos_sims_by_delta"].get(delta, [])
            if ds:
                n_key = "n_percamera" if ftype == "F1" else "n_pairs"
                entry["per_delta"][str(delta)] = {
                    "mean": float(np.mean(ds)),
                    "std": float(np.std(ds)),
                    n_key: len(ds),
                    "frame_id_gap": delta * FRAME_STEP,
                }
        if ftype == "F1":
            entry["caveat"] = (
                "F1 n_percamera counts per-camera samples. "
                "Multiple cameras from the same person-frame-pair are correlated, "
                "which inflates effective N and narrows confidence intervals. "
                "Point estimates (mean, margin, AUC) are unbiased. "
                "Gate experiment only uses separation and point estimates, which is acceptable."
            )
        summary[ftype] = entry

    with open(os.path.join(OUTPUT_DIR, "step1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    features = load_features()
    instance_table = load_instance_table()
    visibility_scores = load_visibility_scores(instance_table)
    by_fpc, by_person, by_frame, by_pf = build_index(instance_table)

    # Print basic stats
    n_persons = len(by_person)
    n_frames = len(by_frame)
    print(f"\n  Unique persons: {n_persons}")
    print(f"  Unique frames: {n_frames}")
    print(f"  Frame range: {min(by_frame.keys())} - {max(by_frame.keys())}")
    print(f"  (person, frame) groups: {len(by_pf)}")

    # Step0
    dispersions, median_dist = step0_headroom(features, by_pf, visibility_scores, by_fpc)

    # Step1
    results = step1_stability(features, by_pf, by_person, by_frame,
                              instance_table, visibility_scores)

    print("\n" + "=" * 70)
    print("DONE. Results saved to:", OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
