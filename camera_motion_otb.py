#!/usr/bin/env python3
"""
camera_motion_otb.py — relate per-sequence *camera motion* to tracking quality.

`camera_motion.py` estimates camera motion from a video file. OTB sequences are
image folders, not videos, so this driver reuses that engine
(`CameraMotionEstimator` + `decompose_affine`) over each sequence's frames and
asks a single question:

    Does more camera motion make the CSRT tracker do worse (lower mean IoU)?

For every OTB sequence that has a results/<seq>/predictions.csv, it:

  1. Streams the frames in ds/.../<seq>/img/*.jpg through the sparse
     Lucas-Kanade + RANSAC background-motion estimator, giving a per-frame
     (tx, ty, rotation, zoom) for the *background* (camera) after moving
     foreground is rejected.
  2. Aggregates per sequence:
        trans_px_mean / trans_px_p95   translation magnitude (px/frame)
        rot_deg_mean                   |rotation|            (deg/frame)
        zoom_abs_mean                  |log scale|           (per frame)
        jerk_px_mean                   frame-to-frame change in translation
                                       vector magnitude (shake / instability)
        inlier_frac_mean               RANSAC inliers / tracked features
  3. Joins with each sequence's mean tracking IoU (from predictions.csv).
  4. Writes per-frame and per-sequence CSVs and a scatter grid reporting the
     Spearman rho between each camera-motion metric and mean IoU.

Units are per *frame* (OTB image folders have no real frame rate), which is
fine for a rank correlation.

Usage
-----
    python camera_motion_otb.py
    python camera_motion_otb.py --sample-every 2 --seqs Shaking Basketball
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy import stats

from camera_motion import CameraMotionEstimator, decompose_affine

# ---------------------------------------------------------------------------
# Metrics we correlate against mean IoU (higher = more camera motion, except
# inlier_frac_mean which is a data-quality diagnostic).
# ---------------------------------------------------------------------------
METRIC_NAMES = [
    "trans_px_mean",
    "trans_px_p95",
    "rot_deg_mean",
    "zoom_abs_mean",
    "jerk_px_mean",
    "inlier_frac_mean",
]
METRIC_LABELS = {
    "trans_px_mean":    "Translation\n(mean px/frame)",
    "trans_px_p95":     "Translation burst\n(95th pct px/frame)",
    "rot_deg_mean":     "Rotation\n(mean deg/frame)",
    "zoom_abs_mean":    "Zoom rate\n(mean |log scale|/frame)",
    "jerk_px_mean":     "Shake / jerk\n(mean |Δtranslation| px)",
    "inlier_frac_mean": "RANSAC inlier fraction\n(background stability)",
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_predictions(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def mean_iou_of(preds: list[dict]) -> float:
    ious = [float(r.get("iou") or 0.0) for r in preds]
    return float(np.mean(ious)) if ious else 0.0


# ---------------------------------------------------------------------------
# Per-sequence camera-motion pass
# ---------------------------------------------------------------------------
def measure_sequence(
    img_dir: Path, sample_every: int, max_frames: Optional[int]
) -> list[dict]:
    """Run the camera-motion estimator over an image folder.

    Returns a list of per-frame dicts: frame, trans_px, rot_deg, zoom_abs,
    inliers, features. Frame 1 has no predecessor so the series starts at 2.
    """
    frames = sorted(img_dir.glob("*.jpg"))
    if max_frames:
        frames = frames[:max_frames]
    if len(frames) < 2:
        return []

    est = CameraMotionEstimator()
    prev = cv2.imread(str(frames[0]))
    if prev is None:
        return []
    prev_gray = est._grayscale(prev)
    prev_pts = est._detect(prev_gray)

    out: list[dict] = []
    step = max(1, sample_every)
    for idx in range(1, len(frames), step):
        curr = cv2.imread(str(frames[idx]))
        if curr is None:
            continue
        curr_gray = est._grayscale(curr)
        affine, inlier_mask, prev_pts = est.step(prev_gray, curr_gray, prev_pts)
        if affine is None or getattr(affine, "size", 0) == 0:
            tx = ty = 0.0
            rot_deg = 0.0
            zoom_abs = 0.0
            inliers = 0
        else:
            tx, ty, ang_rad, scale = decompose_affine(affine)
            rot_deg = abs(math.degrees(ang_rad))
            zoom_abs = abs(math.log(scale)) if scale > 0 else 0.0
            inliers = int(inlier_mask.sum())
        out.append({
            "frame": idx + 1,               # 1-based index of the current frame
            "trans_px": math.hypot(tx, ty),
            "tx": tx,
            "ty": ty,
            "rot_deg": rot_deg,
            "zoom_abs": zoom_abs,
            "inliers": inliers,
            "features": int(prev_pts.shape[0]) if prev_pts is not None else 0,
        })
        prev_gray = curr_gray
    return out


def aggregate(per_frame: list[dict]) -> dict:
    """Collapse per-frame motion into per-sequence summary metrics."""
    if not per_frame:
        return {k: float("nan") for k in METRIC_NAMES}
    trans = np.array([r["trans_px"] for r in per_frame], dtype=float)
    rot = np.array([r["rot_deg"] for r in per_frame], dtype=float)
    zoom = np.array([r["zoom_abs"] for r in per_frame], dtype=float)
    inl = np.array([r["inliers"] for r in per_frame], dtype=float)
    feat = np.array([max(1, r["features"]) for r in per_frame], dtype=float)

    # Jerk = frame-to-frame change of the translation *vector* (shake proxy).
    txy = np.array([[r["tx"], r["ty"]] for r in per_frame], dtype=float)
    if len(txy) >= 2:
        jerk = np.linalg.norm(np.diff(txy, axis=0), axis=1)
        jerk_mean = float(np.mean(jerk))
    else:
        jerk_mean = float("nan")

    return {
        "trans_px_mean":    float(np.mean(trans)),
        "trans_px_p95":     float(np.percentile(trans, 95)),
        "rot_deg_mean":     float(np.mean(rot)),
        "zoom_abs_mean":    float(np.mean(zoom)),
        "jerk_px_mean":     jerk_mean,
        "inlier_frac_mean": float(np.mean(inl / feat)),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_csv(path: Path, header: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def spearman_xy(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p), n


def plot_scatter(seq_summary: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")
        return
    if len(seq_summary) < 2:
        print("  (need >=2 sequences for a scatter; skipping plot)")
        return

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, m in zip(axes.flatten(), METRIC_NAMES):
        xs = np.array([d[m] for d in seq_summary], dtype=float)
        ys = np.array([d["mean_iou"] for d in seq_summary], dtype=float)
        rho, p, n_used = spearman_xy(xs, ys)
        for d in seq_summary:
            v = d[m]
            if not np.isfinite(v):
                continue
            ax.scatter(v, d["mean_iou"], s=70, c="tab:red",
                       edgecolor="k", alpha=0.8)
            ax.annotate(d["sequence"], (v, d["mean_iou"]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        if np.isfinite(rho):
            title = (f"{METRIC_LABELS[m]}\n"
                     f"Spearman ρ = {rho:+.2f}  (p={p:.3f}, n={n_used})")
        else:
            title = f"{METRIC_LABELS[m]}\n(n<3 usable points — no ρ)"
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("mean IoU")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Camera motion vs. CSRT mean IoU (per OTB sequence)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Relate per-sequence camera motion to CSRT tracking IoU."
    )
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--seqs-dir", default="ds/OTB-dataset/OTB_downloads")
    ap.add_argument("--out-dir", default="camera_motion_out")
    ap.add_argument("--sample-every", type=int, default=1,
                    help="Process every K-th frame (default 1 = all frames).")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Cap frames per sequence (default: no cap).")
    ap.add_argument("--seqs", nargs="*", default=None,
                    help="Optional whitelist of sequence names.")
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    seqs_dir = Path(args.seqs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"Error: results dir '{results_dir}' does not exist "
              f"(run run_ds_tracker.py first).", file=sys.stderr)
        return 2
    if not seqs_dir.exists():
        print(f"Error: sequences dir '{seqs_dir}' does not exist.", file=sys.stderr)
        return 2

    candidates = sorted(
        d for d in seqs_dir.iterdir()
        if d.is_dir() and (results_dir / d.name / "predictions.csv").exists()
    )
    if args.seqs:
        wanted = set(args.seqs)
        candidates = [d for d in candidates if d.name in wanted]
    if not candidates:
        print("Error: no sequences with both frames and predictions.csv found.",
              file=sys.stderr)
        return 2

    print(f"Processing {len(candidates)} sequences "
          f"(sample_every={args.sample_every})")

    all_per_frame_rows: list[list] = []
    seq_summary: list[dict] = []

    for seq_dir in candidates:
        name = seq_dir.name
        print(f"  {name:<14s} ...", end="", flush=True)
        per_frame = measure_sequence(
            seq_dir / "img", args.sample_every, args.max_frames
        )
        if not per_frame:
            print(" (skipped: <2 frames)")
            continue
        agg = aggregate(per_frame)
        preds = load_predictions(results_dir / name / "predictions.csv")
        m_iou = mean_iou_of(preds)

        seq_summary.append({
            "sequence": name,
            "frames": len(preds),
            "motion_frames": len(per_frame),
            "mean_iou": m_iou,
            **agg,
        })
        for r in per_frame:
            all_per_frame_rows.append([
                name, r["frame"], f"{r['trans_px']:.3f}", f"{r['rot_deg']:.4f}",
                f"{r['zoom_abs']:.5f}", r["inliers"], r["features"],
            ])
        print(f" {len(per_frame):>4d} frames | mean_iou={m_iou:.3f} | "
              f"trans={agg['trans_px_mean']:.2f}px | "
              f"jerk={agg['jerk_px_mean']:.2f}px | "
              f"inlier={agg['inlier_frac_mean']:.2f}")

    # --- write CSVs ---
    write_csv(
        out_dir / "per_frame_camera_motion.csv",
        ["sequence", "frame", "trans_px", "rot_deg", "zoom_abs",
         "inliers", "features"],
        all_per_frame_rows,
    )
    if seq_summary:
        header = ["sequence", "frames", "motion_frames", "mean_iou"] + METRIC_NAMES

        def fmt(v):
            if isinstance(v, float):
                return "NaN" if math.isnan(v) else f"{v:.4f}"
            return str(v)

        write_csv(
            out_dir / "per_sequence_camera_motion.csv",
            header,
            [[fmt(d.get(k, "")) for k in header] for d in seq_summary],
        )
    plot_scatter(seq_summary, out_dir / "scatter_camera_motion_vs_iou.png")

    # --- console correlation summary ---
    if len(seq_summary) >= 3:
        print("\nCross-sequence Spearman correlation with mean IoU:")
        print(f"  n = {len(seq_summary)} sequences")
        for m in METRIC_NAMES:
            xs = np.array([d[m] for d in seq_summary], dtype=float)
            ys = np.array([d["mean_iou"] for d in seq_summary], dtype=float)
            rho, p, n_used = spearman_xy(xs, ys)
            rho_s = f"{rho:+.3f}" if np.isfinite(rho) else "  n/a"
            p_s = f"{p:.3f}" if np.isfinite(p) else "  n/a"
            print(f"  {m:<20s}  ρ = {rho_s}   p = {p_s}   n_used = {n_used}")
    else:
        print(f"\n  (n={len(seq_summary)} sequences — need >=3 for a Spearman)")

    print(f"\nWrote outputs to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
