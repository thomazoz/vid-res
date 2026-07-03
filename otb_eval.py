"""Evaluate YOLO tracker against OTB ground-truth bounding boxes.

For each sequence:
  - Run model.track() on every frame (image sequence → video-like stream)
  - Lock onto ONE track ID (the track best overlapping GT on the acquisition
    frame) and follow *that identity* for the rest of the sequence — a missed
    or switched identity scores IoU 0, exactly as a real single-target tracker
    would be penalised.
  - Report, per-sequence and overall:
      * success    — OTB-style AUC of the success plot (mean over IoU
                     thresholds 0…1), plus success@0.5 for reference
      * precision  — fraction of frames with centre error < 20px
      * mean IoU   — of the followed identity
      * ceiling_iou — mean best-IoU over *all* detections (a GT-guided detector
                      upper bound, NOT a tracking score; shown for context)

Usage:
    python3 otb_eval.py                          # runs all sequences in OTB_downloads
    python3 otb_eval.py --seqs Basketball Bolt   # specific sequences only
    python3 otb_eval.py --model yolo11s.pt       # use a larger model
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

OTB_DIR = Path("ds/OTB-dataset/OTB_downloads")
MODEL    = "yolo11n.pt"
CONF     = 0.20   # lower than default so we don't miss the target object
OUT_DIR  = Path("otb_eval_out")

# Inference device. Set in main() from --device; "cpu" until then. Passed
# explicitly to model.track() because Ultralytics does not auto-select MPS.
DEVICE = "cpu"


def auto_device(pref: str | None = None) -> str:
    """Resolve the inference device: honour an explicit choice, else prefer
    Apple MPS when available, else fall back to CPU."""
    if pref and pref != "auto":
        return pref
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ── helpers ────────────────────────────────────────────────────────────────────

def load_gt(seq_dir: Path) -> list[tuple[int,int,int,int]]:
    """Return list of (x, y, w, h) ground-truth boxes, one per frame."""
    gt = []
    with open(seq_dir / "groundtruth_rect.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Some OTB files use tabs or spaces, others commas
            parts = line.replace("\t", ",").replace(" ", ",").split(",")
            parts = [p for p in parts if p]
            x, y, w, h = int(float(parts[0])), int(float(parts[1])), \
                         int(float(parts[2])), int(float(parts[3]))
            gt.append((x, y, w, h))
    return gt


def xywh_to_xyxy(x, y, w, h):
    return x, y, x + w, y + h


def iou_xyxy(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw  = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def centre_error(pred_xyxy, gt_xywh):
    gx, gy, gw, gh = gt_xywh
    gc = np.array([gx + gw/2, gy + gh/2])
    pc = np.array([(pred_xyxy[0]+pred_xyxy[2])/2, (pred_xyxy[1]+pred_xyxy[3])/2])
    return float(np.linalg.norm(gc - pc))


def detector_ceiling(results, gt_xyxy):
    """Best IoU over ALL detections vs the GT box — a GT-guided detector upper
    bound, NOT a tracking score (it may pick a different object each frame)."""
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return 0.0
    best_iou = 0.0
    for b in r.boxes:
        xy = b.xyxy[0].cpu().numpy().astype(float)
        best_iou = max(best_iou, iou_xyxy(xy, gt_xyxy))
    return best_iou


def acquire_id(results, gt_xyxy):
    """Track ID of the box best overlapping the GT box (IoU > 0), else None."""
    r = results[0]
    if r.boxes is None:
        return None
    best_id, best_iou = None, 0.0
    for b in r.boxes:
        if b.id is None:
            continue
        i = iou_xyxy(b.xyxy[0].cpu().numpy().astype(float), gt_xyxy)
        if i > best_iou:
            best_iou, best_id = i, int(b.id)
    return best_id


def box_for_id(results, target_id):
    """xyxy box of the track with ``target_id`` in this frame, or None."""
    r = results[0]
    if r.boxes is None or target_id is None:
        return None
    for b in r.boxes:
        if b.id is not None and int(b.id) == target_id:
            return b.xyxy[0].cpu().numpy().astype(float)
    return None


def success_auc(ious, n_thresholds=21):
    """OTB success-plot AUC: mean success rate over IoU thresholds in [0, 1]."""
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    return float(np.mean([np.mean([v >= t for v in ious]) for t in thresholds]))


# ── per-sequence evaluation ────────────────────────────────────────────────────

def eval_sequence(model, seq_dir: Path, out_dir: Path, conf: float = CONF) -> dict:
    gt_list = load_gt(seq_dir)
    img_files = sorted((seq_dir / "img").iterdir())
    n = min(len(gt_list), len(img_files))

    ious, cerrs, successes, ceiling = [], [], [], []
    target_id = None
    # Reset tracker state so IDs don't carry over from the previous sequence:
    # nulling the predictor forces model.track() to rebuild fresh ByteTrack
    # trackers (restarting IDs) on the next call.
    if hasattr(model, "predictor") and model.predictor is not None:
        model.predictor = None

    for i in range(n):
        frame = cv2.imread(str(img_files[i]))
        if frame is None:
            continue
        gt = gt_list[i]
        gt_xy = xywh_to_xyxy(*gt)

        results = model.track(frame, persist=True, conf=conf, verbose=False,
                              device=DEVICE)

        # Lock onto one identity as soon as a track overlaps the GT, then follow
        # only that ID — this measures tracking, not a per-frame detector oracle.
        if target_id is None:
            target_id = acquire_id(results, gt_xy)
        pred_xy = box_for_id(results, target_id)

        iou_val = iou_xyxy(pred_xy, gt_xy) if pred_xy is not None else 0.0
        ious.append(iou_val)
        successes.append(1 if iou_val >= 0.5 else 0)
        cerrs.append(centre_error(pred_xy, gt) if pred_xy is not None else 999.0)
        ceiling.append(detector_ceiling(results, gt_xy))

    hit_cerrs     = [e for e in cerrs if e < 900]
    mean_iou      = float(np.mean(ious)) if ious else 0.0
    success_rate  = success_auc(ious) if ious else 0.0       # OTB success-plot AUC
    success_at_50 = float(np.mean(successes)) if successes else 0.0
    precision     = float(np.mean([e < 20 for e in cerrs])) if cerrs else 0.0
    mean_cerr     = float(np.mean(hit_cerrs)) if hit_cerrs else float("nan")
    ceiling_iou   = float(np.mean(ceiling)) if ceiling else 0.0

    print(f"  {seq_dir.name:<18} frames={n:4d}  "
          f"IoU={mean_iou:.3f}  successAUC={success_rate:.3f}  "
          f"s@0.5={success_at_50:.3f}  prec={precision:.3f}  "
          f"cerr={mean_cerr:.1f}px  ceil={ceiling_iou:.3f}")

    return {
        "sequence":      seq_dir.name,
        "frames":        n,
        "mean_iou":      mean_iou,
        "success_rate":  success_rate,
        "success_at_50": success_at_50,
        "precision":     precision,
        "mean_cerr":     mean_cerr,
        "ceiling_iou":   ceiling_iou,
    }


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_results(rows: list[dict], out_path: Path):
    seqs        = [r["sequence"] for r in rows]
    ious        = [r["mean_iou"] for r in rows]
    success     = [r["success_rate"] for r in rows]
    precision   = [r["precision"] for r in rows]

    x = np.arange(len(seqs))
    fig, axes = plt.subplots(3, 1, figsize=(max(10, len(seqs)*0.6), 11))

    for ax, vals, label, colour in [
        (axes[0], ious,      "Mean IoU",          "steelblue"),
        (axes[1], success,   "Success (AUC over IoU thresholds)", "seagreen"),
        (axes[2], precision, "Precision (centre err<20px)", "darkorange"),
    ]:
        bars = ax.bar(x, vals, color=colour, alpha=0.82)
        ax.axhline(np.mean(vals), color="red", ls="--", lw=1.2,
                   label=f"mean={np.mean(vals):.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels(seqs, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(f"YOLO11n OTB Tracker Evaluation — {len(seqs)} sequences", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved plot -> {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs",  nargs="+", default=None,
                    help="Sequence names to evaluate (default: all)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--conf",  type=float, default=CONF)
    ap.add_argument("--outdir", default=str(OUT_DIR))
    ap.add_argument("--device", default="auto",
                    help="Inference device: auto (MPS>CUDA>CPU), cpu, mps, cuda, 0…")
    args = ap.parse_args()

    global DEVICE
    DEVICE = auto_device(args.device)

    out = Path(args.outdir)
    conf = args.conf
    out.mkdir(exist_ok=True)

    all_seqs = sorted(p for p in OTB_DIR.iterdir() if p.is_dir())
    if args.seqs:
        all_seqs = [OTB_DIR / s for s in args.seqs]

    print(f"Model: {args.model}  |  Sequences: {len(all_seqs)}  |  Conf: {conf}  "
          f"|  Device: {DEVICE}\n")
    model = YOLO(args.model)

    rows = []
    for seq in all_seqs:
        if not (seq / "groundtruth_rect.txt").exists():
            print(f"  {seq.name}: no ground truth, skipping")
            continue
        rows.append(eval_sequence(model, seq, out, conf))

    # summary
    print(f"\n{'─'*60}")
    print(f"OVERALL  sequences={len(rows)}  "
          f"IoU={np.mean([r['mean_iou'] for r in rows]):.3f}  "
          f"success={np.mean([r['success_rate'] for r in rows]):.3f}  "
          f"precision={np.mean([r['precision'] for r in rows]):.3f}")

    # write CSV
    csv_path = out / "otb_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Saved CSV  -> {csv_path}")

    plot_results(rows, out / "otb_results.png")


if __name__ == "__main__":
    main()
