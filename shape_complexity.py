#!/usr/bin/env python3
"""
shape_complexity.py — measure how the *visual complexity* of a tracked target
relates to per-frame tracking quality (IoU).

For each OTB sequence with an existing run_ds_tracker.py output, this script:

  1. Loads the first-frame GT box and the per-frame predicted boxes from
     results/<seq>/predictions.csv.
  2. For frame 1 (GT box) and every K-th frame (predicted box), it runs
     cv2.grabCut to obtain a foreground mask, then computes 6 metrics:

        silhouette_complexity  perimeter^2 / area of the FG mask
        convex_hull_ratio      area(mask) / area(convexHull(mask))
        texture_entropy        Shannon entropy of Sobel gradient magnitudes
        color_entropy          Shannon entropy of HSV histogram
        edge_density           Canny-edge pixel fraction inside the box
        fg_bg_contrast         Bhattacharyya distance between in-mask and
                               out-of-mask H-S histograms

  3. Joins the metrics with per-frame IoU from the predictions CSV.
  4. Writes per-frame and per-sequence CSVs plus a 2x3 scatter grid that
     reports the Spearman rho between each metric and mean IoU across
     sequences.

Usage
-----
    python shape_complexity.py
    python shape_complexity.py --sample-every 10 --grabcut-iters 3
    python shape_complexity.py --seqs Basketball Biker

Outputs (under --out-dir, default "complexity_out")
----------------------------------------------------
    complexity_per_frame.csv     long form, one row per sampled frame
    per_sequence_complexity.csv  long form, one row per sequence (frame-1 GT)
    complexity_per_sequence.csv  wide form, frame-1 metrics + mean IoU
    scatter_complexity_vs_iou.png  2x3 panel scatter
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METRIC_NAMES = [
    "silhouette_complexity",
    "convex_hull_ratio",
    "texture_entropy",
    "color_entropy",
    "edge_density",
    "fg_bg_contrast",
]
METRIC_LABELS = {
    "silhouette_complexity": "Silhouette complexity\n(perimeter²/area)",
    "convex_hull_ratio":     "Convex-hull ratio\n(mask / hull, 1=convex)",
    "texture_entropy":       "Texture entropy\n(gradient histogram, bits)",
    "color_entropy":         "Color entropy\n(HSV histogram, bits)",
    "edge_density":          "Edge density\n(Canny fraction)",
    "fg_bg_contrast":        "FG / BG contrast\n(Bhattacharyya distance)",
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_gt(seq_dir: Path) -> list[tuple[int, int, int, int]]:
    """Read OTB groundtruth_rect.txt. 4 ints per line: x, y, w, h."""
    out: list[tuple[int, int, int, int]] = []
    p = seq_dir / "groundtruth_rect.txt"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [s for s in line.replace(",", " ").split() if s]
        if len(parts) < 4:
            continue
        try:
            x, y, w, h = (int(float(s)) for s in parts[:4])
        except ValueError:
            continue
        out.append((x, y, w, h))
    return out


def load_predictions(path: Path) -> list[dict]:
    """Read results/<seq>/predictions.csv; skip the header row."""
    with path.open() as f:
        return list(csv.DictReader(f))


def img_path(seq_dir: Path, frame_1_indexed: int) -> Path:
    """Map a 1-based frame index to the OTB image path."""
    return seq_dir / "img" / f"{frame_1_indexed:04d}.jpg"


def safe_crop(img: np.ndarray, bbox: tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Clip bbox to image bounds and return the crop, or None if empty."""
    H, W = img.shape[:2]
    x, y, w, h = bbox
    if w <= 1 or h <= 1:
        return None
    x0 = max(0, int(x)); y0 = max(0, int(y))
    x1 = min(W, int(x + w)); y1 = min(H, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1].copy()


# ---------------------------------------------------------------------------
# Foreground segmentation
# ---------------------------------------------------------------------------
def grabcut_mask(crop_bgr: np.ndarray, iters: int = 5) -> np.ndarray:
    """
    Run cv2.grabCut on the whole crop with a small margin as the
    "definitely-background" hint. Returns a uint8 mask of {0,1}
    (1 = foreground).

    For very small crops where the margin would consume the whole image,
    we instead fall back to the *whole crop* being foreground — that way
    the silhouette metrics degrade gracefully on tiny targets instead of
    returning all-zero masks.
    """
    h, w = crop_bgr.shape[:2]
    if h < 8 or w < 8:
        # Too small for GrabCut's margin; treat as fully FG.
        return np.ones((h, w), dtype=np.uint8)

    # 1-pixel absolute margin, with a 2% target if there's room.
    mx = max(1, int(round(0.02 * w)))
    my = max(1, int(round(0.02 * h)))
    rect_w = w - 2 * mx
    rect_h = h - 2 * my
    if rect_w < 4 or rect_h < 4:
        return np.ones((h, w), dtype=np.uint8)
    rect = (mx, my, rect_w, rect_h)

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop_bgr, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.ones((h, w), dtype=np.uint8)
    return ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _shannon_entropy(hist: np.ndarray) -> float:
    """Histogram → entropy in bits (treats zero bins as 0 contribution)."""
    p = hist.astype(np.float64)
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _build_hs_hist(img_bgr: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """2D H-S histogram (18 x 16 bins) normalized to sum 1. Mask may be None."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, _ = cv2.split(hsv)
    hist = cv2.calcHist(
        images=[hsv], channels=[0, 1], mask=mask,
        histSize=[18, 16], ranges=[0, 180, 0, 256],
    )
    hist = hist.flatten().astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _bhattacharyya(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    bc = float(np.sum(np.sqrt(p * q)))
    return float(np.sqrt(max(0.0, 1.0 - bc)))


def compute_metrics(
    crop_bgr: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    """
    Compute the six complexity metrics on a crop + its foreground mask.

    Texture / color / edge statistics are computed on the *whole crop*
    (the bounding-box region). The mask is only used for the silhouette
    shape metrics (perimeter / convex hull) and the FG/BG contrast
    statistic. This way texture complexity is well-defined even on
    heavily-blurred targets where GrabCut fails to find a clean FG mask.

    Silhouette / hull / contrast metrics that depend on a usable FG mask
    return ``float('nan')`` when GrabCut produces an empty or near-empty
    foreground mask. The downstream Spearman correlation then ignores
    those pairs rather than treating them as "zero complexity".
    """
    h, w = crop_bgr.shape[:2]
    box_area = float(h * w)
    fg_area = float(mask.sum())
    fg_frac = fg_area / box_area if box_area > 0 else 0.0

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # ----- silhouette_complexity: perimeter^2 / area of FG mask -----
    silhouette_complexity = float("nan")
    if fg_area > 4:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            perim = cv2.arcLength(largest, True)
            silhouette_complexity = (perim * perim) / max(fg_area, 1.0)

    # ----- convex_hull_ratio: FG area / convex-hull area (1 = convex) -----
    convex_hull_ratio = float("nan")
    if fg_area > 4:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest)
            hull_area = float(cv2.contourArea(hull))
            if hull_area > 0:
                convex_hull_ratio = float(fg_area / hull_area)

    # ----- texture_entropy: Shannon entropy of gradient-magnitude hist -----
    # Always defined, regardless of mask quality.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    hi = float(mag.max()) if mag.max() > 0 else 1.0
    bins = np.linspace(0, hi, 33)
    hist, _ = np.histogram(mag.flatten(), bins=bins)
    texture_entropy = _shannon_entropy(hist)

    # ----- color_entropy: entropy of HSV histogram (full crop) -----
    color_hist = _build_hs_hist(crop_bgr, None)
    color_entropy = _entropy_from_normalized(color_hist)

    # ----- edge_density: fraction of Canny-edge pixels in the box -----
    edges = cv2.Canny(gray, 80, 200)
    edge_density = float(edges.sum()) / (255.0 * box_area)

    # ----- fg_bg_contrast: Bhattacharyya distance between in/out H-S hist -----
    if 0.05 <= fg_frac <= 0.95:
        in_mask = (mask > 0).astype(np.uint8) * 255
        out_mask = ((mask == 0)).astype(np.uint8) * 255
        h_in = _build_hs_hist(crop_bgr, in_mask)
        h_out = _build_hs_hist(crop_bgr, out_mask)
        fg_bg_contrast = _bhattacharyya(h_in, h_out)
    else:
        fg_bg_contrast = float("nan")

    return {
        "silhouette_complexity": float(silhouette_complexity),
        "convex_hull_ratio":     float(convex_hull_ratio),
        "texture_entropy":       float(texture_entropy),
        "color_entropy":         float(color_entropy),
        "edge_density":          float(edge_density),
        "fg_bg_contrast":        float(fg_bg_contrast),
        "_fg_frac":              float(fg_frac),  # diagnostic, dropped later
    }


def _entropy_from_normalized(p: np.ndarray) -> float:
    """Entropy in bits of an already-normalized histogram."""
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------
@dataclass
class FrameRow:
    sequence: str
    frame: int
    box_kind: str        # "gt" or "pred"
    iou: float
    silhouette_complexity: float
    convex_hull_ratio: float
    texture_entropy: float
    color_entropy: float
    edge_density: float
    fg_bg_contrast: float

    def as_csv_row(self) -> list:
        def fmt(v: float) -> str:
            return "NaN" if np.isnan(v) else f"{v:.4f}"

        return [
            self.sequence, self.frame, self.box_kind, f"{self.iou:.4f}",
            fmt(self.silhouette_complexity), fmt(self.convex_hull_ratio),
            fmt(self.texture_entropy), fmt(self.color_entropy),
            fmt(self.edge_density), fmt(self.fg_bg_contrast),
        ]


def measure_box(
    frame_img: np.ndarray, bbox: tuple[int, int, int, int], grabcut_iters: int
) -> Optional[dict[str, float]]:
    crop = safe_crop(frame_img, bbox)
    if crop is None:
        return None
    mask = grabcut_mask(crop, iters=grabcut_iters)
    return compute_metrics(crop, mask)


def process_sequence(
    seq_name: str,
    seq_dir: Path,
    results_dir: Path,
    sample_every: int,
    grabcut_iters: int,
) -> tuple[list[FrameRow], dict, Optional[FrameRow]]:
    """
    Returns:
        per_frame_rows : all FrameRow entries (GT pass + sampled pred pass)
        per_seq_long   : dict of {sequence, frame, box_kind, *metrics} for the
                         frame-1 GT pass (used by the long-form per-seq CSV)
        gt_row         : the FrameRow corresponding to frame 1 (used for
                         per-sequence aggregation)
    """
    gt = load_gt(seq_dir)
    preds = load_predictions(results_dir / seq_name / "predictions.csv")
    if not gt or not preds:
        return [], {}, None

    n = min(len(gt), len(preds))
    sample_every = max(1, sample_every)
    per_frame_rows: list[FrameRow] = []
    gt_row: Optional[FrameRow] = None

    # Index set we'll process: frame 1 always; then every K-th pred frame.
    sample_idx = {0}  # 0-based, i.e. frame 1
    sample_idx.update(range(0, n, sample_every))

    for i in sorted(sample_idx):
        if i >= n:
            continue
        row = preds[i]
        iou = float(row.get("iou") or 0.0)
        frame_num = i + 1  # 1-based

        img_p = img_path(seq_dir, frame_num)
        if not img_p.exists():
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue

        if i == 0:
            # Frame-1 GT pass
            m = measure_box(img, gt[0], grabcut_iters)
            if m is None:
                continue
            m.pop("_fg_frac", None)  # diagnostic only, not for the row
            r = FrameRow(
                sequence=seq_name, frame=frame_num, box_kind="gt", iou=iou, **m
            )
            gt_row = r
            per_frame_rows.append(r)
        else:
            # Sampled predicted-box pass
            try:
                pb = (
                    int(float(row["pred_x"])), int(float(row["pred_y"])),
                    int(float(row["pred_w"])), int(float(row["pred_h"])),
                )
            except (KeyError, ValueError):
                continue
            if pb[2] <= 0 or pb[3] <= 0:
                continue
            m = measure_box(img, pb, grabcut_iters)
            if m is None:
                continue
            m.pop("_fg_frac", None)
            r = FrameRow(
                sequence=seq_name, frame=frame_num, box_kind="pred", iou=iou, **m
            )
            per_frame_rows.append(r)

    # Per-sequence long form (frame-1 GT only, with mean_iou for context)
    per_seq_long: dict = {}
    if gt_row is not None:
        per_seq_long = {
            "sequence": seq_name,
            "frame": gt_row.frame,
            "box_kind": "gt",
            "iou": gt_row.iou,
            **{k: getattr(gt_row, k) for k in METRIC_NAMES},
        }
    return per_frame_rows, per_seq_long, gt_row


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_csv(path: Path, header: list[str], rows: Iterable[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_summary_csvs(
    out_dir: Path,
    per_frame_rows: list[FrameRow],
    per_seq_long: list[dict],
    seq_summary: list[dict],
) -> None:
    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "NaN"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    write_csv(
        out_dir / "complexity_per_frame.csv",
        [
            "sequence", "frame", "box_kind", "iou",
            "silhouette_complexity", "convex_hull_ratio", "texture_entropy",
            "color_entropy", "edge_density", "fg_bg_contrast",
        ],
        [r.as_csv_row() for r in per_frame_rows],
    )
    if per_seq_long:
        write_csv(
            out_dir / "per_sequence_complexity.csv",
            [
                "sequence", "frame", "box_kind", "iou",
                "silhouette_complexity", "convex_hull_ratio",
                "texture_entropy", "color_entropy", "edge_density",
                "fg_bg_contrast",
            ],
            [
                [d["sequence"], d["frame"], d["box_kind"],
                 fmt(d["iou"])] +
                [fmt(d[k]) for k in METRIC_NAMES]
                for d in per_seq_long
            ],
        )
    if seq_summary:
        header = list(seq_summary[0].keys())
        rows = [[fmt(d.get(k, "")) for k in header] for d in seq_summary]
        write_csv(out_dir / "complexity_per_sequence.csv", header, rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def spearman_xy(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Spearman ρ ignoring NaNs; returns (rho, p, n_used)."""
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p), n


def plot_scatter(
    seq_summary: list[dict], out_path: Path
) -> None:
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
    axes = axes.flatten()
    for ax, m in zip(axes, METRIC_NAMES):
        xs = np.array([d[m] for d in seq_summary], dtype=float)
        ys = np.array([d["mean_iou"] for d in seq_summary], dtype=float)
        rho, p, n_used = spearman_xy(xs, ys)
        # Only label points that have a finite value for this metric.
        for d in seq_summary:
            v = d[m]
            if not np.isfinite(v):
                continue
            ax.scatter(v, d["mean_iou"], s=70, c="tab:blue",
                       edgecolor="k", alpha=0.8)
            ax.annotate(
                d["sequence"], (v, d["mean_iou"]),
                fontsize=7, xytext=(4, 4), textcoords="offset points",
            )
        if np.isfinite(rho):
            title = (
                f"{METRIC_LABELS[m]}\n"
                f"Spearman ρ = {rho:+.2f}  (p={p:.3f}, n={n_used})"
            )
        else:
            title = (
                f"{METRIC_LABELS[m]}\n(n<3 usable points — no ρ)"
            )
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Target complexity vs. CSRT mean IoU (per OTB sequence)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Measure how target shape complexity affects tracking IoU."
    )
    ap.add_argument(
        "--results-dir", default="results",
        help="Directory containing results/<seq>/predictions.csv (default: results)",
    )
    ap.add_argument(
        "--seqs-dir", default="ds/OTB-dataset/OTB_downloads",
        help="Directory of OTB sequences (default: ds/OTB-dataset/OTB_downloads)",
    )
    ap.add_argument(
        "--out-dir", default="complexity_out",
        help="Output directory (default: complexity_out)",
    )
    ap.add_argument(
        "--sample-every", type=int, default=15,
        help="Sample one predicted-box measurement every K frames (default: 15)",
    )
    ap.add_argument(
        "--grabcut-iters", type=int, default=5,
        help="Iterations of cv2.grabCut (default: 5)",
    )
    ap.add_argument(
        "--seqs", nargs="*", default=None,
        help="Optional whitelist of sequence names to process.",
    )
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    seqs_dir = Path(args.seqs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"Error: results dir '{results_dir}' does not exist.", file=sys.stderr)
        return 2
    if not seqs_dir.exists():
        print(f"Error: sequences dir '{seqs_dir}' does not exist.", file=sys.stderr)
        return 2

    # Sequences with BOTH an OTB image folder and a predictions.csv
    candidates = sorted(
        d for d in seqs_dir.iterdir()
        if d.is_dir() and (results_dir / d.name / "predictions.csv").exists()
    )
    if args.seqs:
        wanted = set(args.seqs)
        candidates = [d for d in candidates if d.name in wanted]
        if not candidates:
            print(
                f"Error: none of the requested sequences were found in "
                f"both {seqs_dir} and {results_dir}.",
                file=sys.stderr,
            )
            return 2

    print(
        f"Processing {len(candidates)} sequences "
        f"(sample_every={args.sample_every}, grabcut_iters={args.grabcut_iters})"
    )

    all_per_frame: list[FrameRow] = []
    all_per_seq_long: list[dict] = []
    seq_summary: list[dict] = []

    for seq_dir in candidates:
        name = seq_dir.name
        print(f"  {name:<14s} ...", end="", flush=True)
        rows, long_row, gt_row = process_sequence(
            seq_name=name, seq_dir=seq_dir,
            results_dir=results_dir, sample_every=args.sample_every,
            grabcut_iters=args.grabcut_iters,
        )
        if not rows:
            print(" (skipped: empty)")
            continue
        all_per_frame.extend(rows)
        if long_row:
            all_per_seq_long.append(long_row)
        # mean IoU from this sequence's predictions
        preds = load_predictions(results_dir / name / "predictions.csv")
        ious = [float(r.get("iou") or 0.0) for r in preds]
        mean_iou = float(np.mean(ious)) if ious else 0.0
        if gt_row is not None:
            seq_summary.append({
                "sequence": name,
                "frames": len(preds),
                "mean_iou": mean_iou,
                "gt_frame": gt_row.frame,
                **{k: getattr(gt_row, k) for k in METRIC_NAMES},
            })
        print(
            f" {len(rows):>4d} rows  |  mean_iou={mean_iou:.3f}  |  "
            f"tex_ent={gt_row.texture_entropy:.2f}  "
            f"edge_d={gt_row.edge_density:.3f}  "
            f"sil_cx={gt_row.silhouette_complexity:.1f}"
            if gt_row else " (no GT row)"
        )

    write_summary_csvs(out_dir, all_per_frame, all_per_seq_long, seq_summary)
    plot_scatter(seq_summary, out_dir / "scatter_complexity_vs_iou.png")

    # Cross-sequence correlation summary
    if len(seq_summary) >= 3:
        print("\nCross-sequence Spearman correlation with mean IoU:")
        print(f"  n = {len(seq_summary)} sequences (NaN-pairs dropped per metric)")
        for m in METRIC_NAMES:
            xs = np.array([d[m] for d in seq_summary], dtype=float)
            ys = np.array([d["mean_iou"] for d in seq_summary], dtype=float)
            rho, p, n_used = spearman_xy(xs, ys)
            rho_s = f"{rho:+.3f}" if np.isfinite(rho) else "  n/a"
            p_s = f"{p:.3f}" if np.isfinite(p) else "  n/a"
            print(f"  {m:<24s}  ρ = {rho_s}   p = {p_s}   n_used = {n_used}")
    else:
        print(
            f"\n  (n={len(seq_summary)} sequences — need >=3 for a Spearman, "
            "so no ρ is printed)"
        )

    print(f"\nWrote outputs to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
