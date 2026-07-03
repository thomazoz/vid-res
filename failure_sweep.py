"""Systematic failure sweep: one condition at a time, varying degrees, 20 conf levels.

For every sequence in the catalog we:
  1. Build pseudo-ground-truth from the clean video (conf=0.25, no corruption)
  2. Apply each condition (motion_blur / gaussian_blur / brightness) at increasing severity
  3. Re-run detection at 20 confidence thresholds (0.05 → 1.00)
  4. Record retention (class-aware and class-agnostic), mean IoU, mean
     confidence, det/frame

Note: the pseudo-GT is the model's own clean detections, so "retention" is a
self-consistency / recall proxy under corruption, not accuracy against human
labels — it is not directly comparable to the OTB success metric in otb_eval.py.

Output: failure_sweep_out/sweep_results.csv  (one row per sequence×condition×severity×conf)

Usage:
    python3 failure_sweep.py                         # all sequences
    python3 failure_sweep.py --seqs Car1 BlurBody    # subset
    python3 failure_sweep.py --stride 3              # sample every 3rd frame (faster)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from corruptions import motion_blur, gaussian_blur, brightness

# ── config ─────────────────────────────────────────────────────────────────────
OTB_DIR   = Path("ds/OTB-dataset/OTB_downloads")
CATALOG   = Path("dataset_catalog.csv")
OUT_DIR   = Path("failure_sweep_out")
MODEL     = "yolo11n.pt"
STRIDE    = 3   # sample every Nth frame (balance speed vs accuracy)

# Inference device. Set in main() from --device; "cpu" until then. Passed
# explicitly to model.predict() because Ultralytics does not auto-select MPS.
DEVICE = "cpu"


def auto_device(pref=None):
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

CONF_REF  = 0.25                              # fixed conf for building pseudo-GT
CONF_LEVELS = [round(i * 0.05, 2) for i in range(1, 21)]  # 0.05 … 1.00  (20 levels)

# Severity ladders — designed to go from "barely noticeable" to "tracker fails"
CONDITIONS = {
    "motion_blur":   [0,  3,  7, 11, 15, 19, 25, 31],   # kernel px
    "gaussian_blur": [0,  1,  2,  3,  5,  7,  9, 12],   # sigma
    "brightness":    [1.0, 0.7, 0.5, 0.35, 0.2,          # darkening
                      1.3, 1.6, 2.0, 2.5, 3.0],          # brightening
}
CLEAN_LEVEL = {"motion_blur": 0, "gaussian_blur": 0, "brightness": 1.0}
CORRUPTION_FN = {
    "motion_blur":   motion_blur,
    "gaussian_blur": gaussian_blur,
    "brightness":    brightness,
}

IOU_THR = 0.5   # for success/retention


# ── helpers ────────────────────────────────────────────────────────────────────

def load_catalog():
    rows = {}
    with open(CATALOG) as f:
        for r in csv.DictReader(f):
            rows[r["sequence"]] = r
    return rows


def load_frames(seq_dir: Path, stride: int):
    files = sorted((seq_dir / "img").iterdir())
    return [cv2.imread(str(f)) for i, f in enumerate(files) if i % stride == 0]


def detect(model, frame, conf):
    r = model.predict(frame, conf=conf, verbose=False, device=DEVICE)[0]
    out = []
    if r.boxes is not None:
        for b in r.boxes:
            out.append((b.xyxy[0].cpu().numpy().astype(float), int(b.cls), float(b.conf)))
    return out


def iou_boxes(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(ref, det, class_agnostic=False):
    """Greedy IoU matching of detections to the pseudo-GT.

    With ``class_agnostic=False`` a detection must share the reference class to
    match (a corruption-induced class flip counts as a miss). With
    ``class_agnostic=True`` only localization matters, so comparing the two
    retention numbers isolates how much of the loss is class flips vs. genuinely
    missed/mislocated objects.
    """
    used, matched_ious, matched_confs = set(), [], []
    for box, cls, conf in sorted(det, key=lambda d: -d[2]):
        best_j, best_i = -1, IOU_THR
        for j, (rbox, rcls, _) in enumerate(ref):
            if j in used or (not class_agnostic and rcls != cls):
                continue
            i = iou_boxes(box, rbox)
            if i >= best_i:
                best_j, best_i = j, i
        if best_j >= 0:
            used.add(best_j)
            matched_ious.append(best_i)
            matched_confs.append(conf)
    return len(matched_ious), matched_ious, matched_confs


def sweep_sequence(model, frames, ref_dets, condition, levels, clean_level):
    """Measure each severity × conf against a pre-built pseudo-GT (``ref_dets``,
    the clean-frame detections at ``CONF_REF``)."""
    rows = []
    for severity in levels:
        is_clean = (severity == clean_level)
        corrupted = [
            f if is_clean else CORRUPTION_FN[condition](f, severity)
            for f in frames
        ]
        for conf_thr in CONF_LEVELS:
            tot_match = tot_match_any = tot_ref = tot_det = 0
            all_iou = []
            all_conf = []
            for f, ref in zip(corrupted, ref_dets):
                if not ref:
                    continue
                det = detect(model, f, conf_thr)
                m, ious, confs = match(ref, det)
                m_any, _, _ = match(ref, det, class_agnostic=True)
                tot_match     += m
                tot_match_any += m_any
                tot_ref   += len(ref)
                tot_det   += len(det)
                all_iou   += ious
                all_conf  += confs

            rows.append({
                "condition":    condition,
                "severity":     severity,
                "conf_thr":     conf_thr,
                "retention":    round(tot_match / tot_ref, 4) if tot_ref else 0.0,
                "retention_anyclass": round(tot_match_any / tot_ref, 4) if tot_ref else 0.0,
                "mean_iou":     round(float(np.mean(all_iou)), 4) if all_iou else 0.0,
                "mean_conf":    round(float(np.mean(all_conf)), 4) if all_conf else 0.0,
                "det_per_frame": round(tot_det / len(frames), 3),
                "ref_objects":  tot_ref,
            })
        clean_tag = "CLEAN" if is_clean else ""
        best_ret = max(r["retention"] for r in rows[-len(CONF_LEVELS):])
        print(f"    {condition:<15} sev={str(severity):<5} "
              f"best_retention={best_ret:.3f}  {clean_tag}")
    return rows


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs",   nargs="+", default=None)
    ap.add_argument("--model",  default=MODEL)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--outdir", default=str(OUT_DIR),
                    help="where to write sweep_results.csv (default: failure_sweep_out)")
    ap.add_argument("--device", default="auto",
                    help="Inference device: auto (MPS>CUDA>CPU), cpu, mps, cuda, 0…")
    args = ap.parse_args()

    global DEVICE
    DEVICE = auto_device(args.device)

    out_dir = Path(args.outdir)
    out_dir.mkdir(exist_ok=True)
    catalog = load_catalog()

    seq_dirs = sorted(p for p in OTB_DIR.iterdir() if p.is_dir())
    if args.seqs:
        seq_dirs = [OTB_DIR / s for s in args.seqs]

    print(f"Model: {args.model} | Stride: {args.stride} | "
          f"Conf levels: {len(CONF_LEVELS)} | Sequences: {len(seq_dirs)} | "
          f"Device: {DEVICE}\n")
    model = YOLO(args.model)

    csv_path = out_dir / "sweep_results.csv"
    fieldnames = [
        "sequence", "object_type", "object_category",
        "primary_challenge", "difficulty",
        "condition", "severity", "conf_thr",
        "retention", "retention_anyclass", "mean_iou", "mean_conf",
        "det_per_frame", "ref_objects",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for seq_dir in seq_dirs:
            if not (seq_dir / "img").exists():
                continue
            seq_name = seq_dir.name
            meta = catalog.get(seq_name, {
                "object_type": "unknown", "object_category": "unknown",
                "primary_challenge": "unknown", "difficulty": "unknown",
            })
            print(f"\n[{seq_name}]  {meta.get('object_type','')} "
                  f"| {meta.get('primary_challenge','')} "
                  f"| difficulty={meta.get('difficulty','')}")

            frames = load_frames(seq_dir, args.stride)
            if not frames:
                print("  no frames, skipping")
                continue
            print(f"  {len(frames)} sampled frames")

            # Pseudo-GT = clean-frame detections at the reference conf, built
            # once per sequence and reused for every condition. (This script
            # uses model.predict() only, so there is no tracker state to reset.)
            ref_dets = [detect(model, f, CONF_REF) for f in frames]

            for condition, levels in CONDITIONS.items():
                sweep_rows = sweep_sequence(model, frames, ref_dets, condition,
                                            levels, CLEAN_LEVEL[condition])
                for row in sweep_rows:
                    writer.writerow({
                        "sequence":          seq_name,
                        "object_type":       meta.get("object_type", ""),
                        "object_category":   meta.get("object_category", ""),
                        "primary_challenge": meta.get("primary_challenge", ""),
                        "difficulty":        meta.get("difficulty", ""),
                        **row,
                    })
                f.flush()   # write after each condition so results survive interruption

    print(f"\nDone. Results saved to {csv_path}")
    print(f"Rows: {sum(1 for _ in open(csv_path)) - 1}")


if __name__ == "__main__":
    main()
