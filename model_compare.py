"""Compare detection models trained on DIFFERENT DATASETS on the same video.

The project's numbers are all yolo11n (COCO, 80 classes). This tool asks: does
a model trained on another dataset see things COCO models miss, and how does it
behave differently? Default zoo:

    yolo11n.pt          COCO (80 cls)      — the project baseline
    yolo11s.pt          COCO (80 cls)      — capacity control (same data, bigger)
    yolov8s-oiv7.pt     Open Images V7     — 601 classes, different taxonomy
    yolov8s-worldv2.pt  YOLO-World v2      — open-vocabulary grounding data

Each model runs plain detection (predict, not track) over the same sampled
frames. Reported per model: detections/frame, mean confidence, top classes,
inference fps, and — vs the yolo11n reference — class-agnostic box agreement
(share of reference boxes matched at IoU>=0.5) and extra boxes/frame the
reference missed. A model that fails to load is skipped with a loud warning
recorded in the outputs.

Usage:
    python3 model_compare.py                                   # traffic.mp4, default zoo
    python3 model_compare.py samples/people.mp4 --max-frames 100
    python3 model_compare.py samples/traffic.mp4 --models yolo11n.pt yolov8s-oiv7.pt
    python3 model_compare.py samples/traffic.mp4 --full        # adds rtdetr-l (COCO, transformer)
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET_OF = {
    "yolo11n.pt": "COCO",
    "yolo11s.pt": "COCO",
    "yolo11l.pt": "COCO",
    "yolov8s-oiv7.pt": "Open Images V7",
    "yolov8s-worldv2.pt": "YOLO-World v2 (open-vocab)",
    "rtdetr-l.pt": "COCO (RT-DETR transformer)",
}
DEFAULT_MODELS = ["yolo11n.pt", "yolo11s.pt", "yolov8s-oiv7.pt", "yolov8s-worldv2.pt"]
REFERENCE = "yolo11n.pt"


def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def read_frames(video: Path, max_frames: int, stride: int):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video}")
    frames, i = [], -1
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % stride:
            continue
        frames.append((i, frame))
    cap.release()
    if not frames:
        raise SystemExit(f"no frames read from {video}")
    return frames


def run_model(name: str, frames, conf: float, device: str):
    """Run one model over the frames. Returns per-frame records + metadata."""
    from ultralytics import YOLO
    t_load = time.time()
    model = YOLO(name)  # auto-downloads on first use
    print(f"  loaded {name} in {time.time() - t_load:.1f}s "
          f"({len(model.names)} classes, dataset: {DATASET_OF.get(name, 'unknown')})")

    per_frame = []
    t0 = time.time()
    for idx, frame in frames:
        try:
            r = model.predict(frame, conf=conf, device=device, verbose=False)[0]
        except Exception:
            if device != "cpu":  # MPS fallback
                device = "cpu"
                r = model.predict(frame, conf=conf, device=device, verbose=False)[0]
            else:
                raise
        boxes = r.boxes
        rec = {
            "frame": idx,
            "n": len(boxes),
            "confs": boxes.conf.cpu().numpy().tolist() if len(boxes) else [],
            "classes": [model.names[int(c)] for c in boxes.cls.cpu().numpy()] if len(boxes) else [],
            "xyxy": boxes.xyxy.cpu().numpy().tolist() if len(boxes) else [],
        }
        per_frame.append(rec)
    elapsed = time.time() - t0
    return {"model": model, "per_frame": per_frame,
            "fps": len(frames) / elapsed if elapsed > 0 else 0.0}


def agreement_vs_reference(ref_frames, other_frames):
    """Class-agnostic: share of reference boxes matched at IoU>=0.5, and
    unmatched extra boxes/frame the other model adds."""
    matched = total_ref = extras = 0
    for ref, oth in zip(ref_frames, other_frames):
        used = set()
        for rb in ref["xyxy"]:
            total_ref += 1
            best_j, best = None, 0.5
            for j, ob in enumerate(oth["xyxy"]):
                if j in used:
                    continue
                v = iou_xyxy(rb, ob)
                if v >= best:
                    best, best_j = v, j
            if best_j is not None:
                used.add(best_j)
                matched += 1
        extras += len(oth["xyxy"]) - len(used)
    return (matched / total_ref if total_ref else float("nan"),
            extras / len(ref_frames) if ref_frames else 0.0)


def annotate(frame, rec, title):
    img = frame.copy()
    for (x1, y1, x2, y2), cls, cf in zip(rec["xyxy"], rec["classes"], rec["confs"]):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, (0, 220, 80), 2)
        cv2.putText(img, f"{cls} {cf:.2f}", (p1[0], max(12, p1[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 80), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (25, 25, 25), -1)
    cv2.putText(img, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", nargs="?", default="samples/traffic.mp4")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--full", action="store_true", help="also run rtdetr-l.pt")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--device", default="auto",
                    help="auto = MPS > CUDA > CPU")
    ap.add_argument("--outdir", default="model_compare_out")
    args = ap.parse_args()
    if args.device == "auto":
        try:
            import torch
            args.device = ("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            args.device = "cpu"

    video = Path(args.video)
    out = Path(args.outdir)
    out.mkdir(exist_ok=True)
    models = args.models or list(DEFAULT_MODELS)
    if args.full and "rtdetr-l.pt" not in models:
        models.append("rtdetr-l.pt")

    print(f"Reading frames from {video} (max {args.max_frames}, stride {args.stride})…")
    frames = read_frames(video, args.max_frames, max(1, args.stride))
    print(f"  {len(frames)} frames loaded\n")

    results, skipped = {}, []
    for name in models:
        print(f"Running {name} …")
        try:
            results[name] = run_model(name, frames, args.conf, args.device)
        except Exception as e:
            msg = f"SKIPPED {name}: {type(e).__name__}: {e}"
            print(f"  *** WARNING: {msg} ***")
            skipped.append(msg)
    if not results:
        raise SystemExit("no model could be run — see warnings above")

    ref_name = REFERENCE if REFERENCE in results else next(iter(results))
    ref_pf = results[ref_name]["per_frame"]

    # aggregate
    summary_rows = []
    for name, res in results.items():
        pf = res["per_frame"]
        all_confs = [c for r in pf for c in r["confs"]]
        cls_counts = Counter(c for r in pf for c in r["classes"])
        agree, extras = (1.0, 0.0) if name == ref_name else agreement_vs_reference(ref_pf, pf)
        summary_rows.append({
            "model": name,
            "dataset": DATASET_OF.get(name, "unknown"),
            "n_classes_in_model": len(res["model"].names),
            "det_per_frame": round(float(np.mean([r["n"] for r in pf])), 3),
            "mean_conf": round(float(np.mean(all_confs)), 3) if all_confs else 0.0,
            "fps": round(res["fps"], 1),
            "agreement_vs_ref": round(agree, 3) if agree == agree else "",
            "extra_boxes_per_frame": round(extras, 3),
            "top_classes": "; ".join(f"{c}:{n}" for c, n in cls_counts.most_common(10)),
        })

    with open(out / "comparison_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        w.writeheader()
        w.writerows(summary_rows)
        if skipped:
            f.write("\n# " + "\n# ".join(skipped) + "\n")
    print(f"\n  csv -> {out / 'comparison_summary.csv'}")

    with open(out / "per_frame_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame"] + list(results))
        for i, (idx, _) in enumerate(frames):
            w.writerow([idx] + [results[m]["per_frame"][i]["n"] for m in results])
    print(f"  csv -> {out / 'per_frame_counts.csv'}")

    # bar chart: det/frame, fps, agreement
    names = [r["model"].replace(".pt", "") for r in summary_rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title in zip(
            axes, ["det_per_frame", "fps", "agreement_vs_ref"],
            ["detections / frame", "inference fps", f"box agreement vs {ref_name}"]):
        vals = [r[key] if r[key] != "" else 0 for r in summary_rows]
        ax.bar(names, vals, color="#4c78a8")
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
    fig.suptitle(f"Model comparison on {video.name} ({len(frames)} frames, conf {args.conf})")
    fig.tight_layout()
    fig.savefig(out / "comparison_bars.png", dpi=130)
    plt.close(fig)
    print(f"  png -> {out / 'comparison_bars.png'}")

    # detections over time
    fig, ax = plt.subplots(figsize=(11, 4))
    for m in results:
        ax.plot([r["frame"] for r in results[m]["per_frame"]],
                [r["n"] for r in results[m]["per_frame"]],
                lw=1.3, label=f"{m.replace('.pt', '')} ({DATASET_OF.get(m, '?')})")
    ax.set_xlabel("frame")
    ax.set_ylabel("detections")
    ax.legend(fontsize=8)
    ax.set_title("Detections over time per model")
    fig.tight_layout()
    fig.savefig(out / "detections_over_time.png", dpi=130)
    plt.close(fig)
    print(f"  png -> {out / 'detections_over_time.png'}")

    # side-by-side on the reference model's busiest frame
    busy_i = int(np.argmax([r["n"] for r in ref_pf]))
    busy_frame = frames[busy_i][1]
    tiles = [annotate(busy_frame, results[m]["per_frame"][busy_i],
                      f"{m.replace('.pt', '')} — {DATASET_OF.get(m, '?')} "
                      f"({results[m]['per_frame'][busy_i]['n']} dets)")
             for m in results]
    cols = 2
    rows_n = (len(tiles) + cols - 1) // cols
    h, w_, _ = tiles[0].shape
    grid = np.zeros((rows_n * h, cols * w_, 3), dtype=np.uint8)
    for k, tile in enumerate(tiles):
        r_, c_ = divmod(k, cols)
        grid[r_ * h:(r_ + 1) * h, c_ * w_:(c_ + 1) * w_] = tile
    cv2.imwrite(str(out / "side_by_side.png"), grid)
    print(f"  png -> {out / 'side_by_side.png'} (frame {frames[busy_i][0]})")

    # findings
    print("\nFINDINGS")
    best = max(summary_rows, key=lambda r: r["det_per_frame"])
    print(f"  most objects found : {best['model']} ({best['det_per_frame']}/frame, {best['dataset']})")
    ref_classes = set(c for r in ref_pf for c in r["classes"])
    for row in summary_rows:
        if row["model"] == ref_name:
            continue
        other_classes = set(c for r in results[row['model']]["per_frame"] for c in r["classes"])
        new = other_classes - ref_classes
        if new:
            print(f"  {row['model']} contributed classes the reference never used: "
                  f"{', '.join(sorted(new)[:12])}")
    if skipped:
        print("  WARNING — skipped models:")
        for s in skipped:
            print(f"    {s}")
    print(f"\nall outputs in {out}/")


if __name__ == "__main__":
    main()
