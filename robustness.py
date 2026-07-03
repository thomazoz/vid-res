"""Measure how motion blur, general blur, and brightness affect detection.

Method
------
We take a real, ground-truth-annotated dataset (COCO128: 128 images with COCO
labels) and apply each corruption at increasing severity. For every severity we
run Ultralytics' official validation, which computes detection metrics against
the ground truth:

* mAP@0.5        - standard "detection quality" headline.
* mAP@0.5:0.95   - stricter, localization-sensitive quality.
* precision      - fraction of detections that are correct (false-positive rate).
* recall         - fraction of real objects that were found (missed-object rate).

Because severity 0 of every corruption is the identity transform, the first
point of each sweep is the clean baseline, so each curve shows the *relative*
damage done by that corruption. Results are written to CSV, plotted, and a
qualitative example grid is saved per corruption.

Run:  python3 robustness.py
"""

from __future__ import annotations

import csv
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from corruptions import SWEEPS

# ---------------------------------------------------------------- config
DATA_IMAGES = Path("datasets/coco128/images/train2017")
DATA_LABELS = Path("datasets/coco128/labels/train2017")
OUT = Path("robustness_out")
MODEL = "yolo11n.pt"
IMGSZ = 640
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
EXAMPLE_IMG = "000000000196.jpg"  # busy dining scene, many small objects


def load_images():
    """Preload all dataset images once (BGR uint8) keyed by stem."""
    imgs = {}
    for p in sorted(DATA_IMAGES.glob("*.jpg")):
        imgs[p.stem] = cv2.imread(str(p))
    return imgs


def build_config_dataset(images, fn, severity, cfg_dir, names):
    """Write corrupted images + labels + data.yaml for one severity."""
    img_dir = cfg_dir / "images"
    lbl_dir = cfg_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for stem, im in images.items():
        out = fn(im, severity)
        cv2.imwrite(str(img_dir / f"{stem}.png"), out)
        src_lbl = DATA_LABELS / f"{stem}.txt"
        if src_lbl.exists():
            shutil.copy(src_lbl, lbl_dir / f"{stem}.txt")
    yml = cfg_dir / "data.yaml"
    with open(yml, "w") as f:
        yaml.safe_dump(
            {"path": str(cfg_dir.resolve()), "train": "images", "val": "images",
             "names": {int(k): v for k, v in names.items()}},
            f,
        )
    return yml


def evaluate(model, yml):
    """Run official validation, return the headline metrics."""
    m = model.val(
        data=str(yml), imgsz=IMGSZ, device=DEVICE,
        verbose=False, plots=False, save_json=False,
        project=str(OUT / "_val"), name="tmp", exist_ok=True,
    )
    return {
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
    }


def run_study(model, images, names):
    rows = []
    for corr_name, spec in SWEEPS.items():
        fn = spec["fn"]
        print(f"\n=== {corr_name} ===")
        for i, sev in enumerate(spec["severities"]):
            cfg_dir = OUT / "_data" / f"{corr_name}_{i}"
            t0 = time.time()
            yml = build_config_dataset(images, fn, sev, cfg_dir, names)
            metrics = evaluate(model, yml)
            shutil.rmtree(cfg_dir, ignore_errors=True)  # free disk
            row = {"corruption": corr_name, "severity": sev,
                   "is_clean": sev == spec["clean"], **metrics}
            rows.append(row)
            print(f"  sev={sev:<5} mAP50={metrics['mAP50']:.3f} "
                  f"mAP50-95={metrics['mAP50_95']:.3f} "
                  f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                  f"({time.time()-t0:.1f}s)")
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_results(rows):
    metrics = [("mAP50", "mAP@0.5"), ("recall", "Recall"), ("precision", "Precision")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (corr_name, spec) in zip(axes, SWEEPS.items()):
        sub = [r for r in rows if r["corruption"] == corr_name]
        sub.sort(key=lambda r: r["severity"])
        xs = [r["severity"] for r in sub]
        for key, label in metrics:
            ax.plot(xs, [r[key] for r in sub], marker="o", label=label)
        clean = spec["clean"]
        ax.axvline(clean, color="gray", ls="--", lw=1, alpha=0.7)
        ax.annotate("clean", (clean, ax.get_ylim()[1]), color="gray",
                    fontsize=8, ha="center", va="bottom")
        ax.set_title(corr_name.replace("_", " ").title())
        ax.set_xlabel(spec["xlabel"])
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("score")
    fig.suptitle(f"Detection robustness vs corruption  ({MODEL}, COCO128, n=128)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "robustness_summary.png", dpi=130)
    print(f"\nSaved plot -> {OUT/'robustness_summary.png'}")


def example_grids(model, images):
    """For each corruption, a row of the same image at rising severity with
    detections drawn, so the effect is visible, not just numeric."""
    im = images[Path(EXAMPLE_IMG).stem]
    for corr_name, spec in SWEEPS.items():
        fn = spec["fn"]
        sevs = spec["severities"]
        # pick clean + 4 spread-out severities
        picks = [spec["clean"]] + [s for s in sevs if s != spec["clean"]][1::2][:4]
        tiles = []
        for sev in picks:
            corrupted = fn(im, sev)
            r = model.predict(corrupted, imgsz=IMGSZ, device=DEVICE,
                              conf=0.25, verbose=False)
            plotted = r[0].plot()
            n = 0 if r[0].boxes is None else len(r[0].boxes)
            cv2.putText(plotted, f"sev={sev} | {n} det", (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            tiles.append(plotted)
        h = min(t.shape[0] for t in tiles)
        tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
        grid = cv2.hconcat(tiles)
        out = OUT / f"example_{corr_name}.png"
        cv2.imwrite(str(out), grid)
        print(f"Saved example -> {out}")


def summarize(rows):
    print("\n================ EFFECT SUMMARY ================")
    for corr_name, spec in SWEEPS.items():
        sub = [r for r in rows if r["corruption"] == corr_name]
        clean = next(r for r in sub if r["is_clean"])
        worst = min(sub, key=lambda r: r["mAP50"])
        drop = clean["mAP50"] - worst["mAP50"]
        rel = 100 * drop / clean["mAP50"] if clean["mAP50"] else 0
        print(f"\n{corr_name}:")
        print(f"  clean mAP50            : {clean['mAP50']:.3f}")
        print(f"  worst mAP50            : {worst['mAP50']:.3f} at severity {worst['severity']}")
        print(f"  max degradation        : -{drop:.3f} mAP50 ({rel:.0f}% relative)")
        print(f"  recall clean -> worst  : {clean['recall']:.3f} -> {worst['recall']:.3f}")


def main():
    OUT.mkdir(exist_ok=True)
    print(f"device={DEVICE}  model={MODEL}")
    model = YOLO(MODEL)
    names = model.names
    print("Preloading 128 images...")
    images = load_images()

    rows = run_study(model, images, names)
    write_csv(rows, OUT / "robustness_results.csv")
    print(f"\nSaved CSV -> {OUT/'robustness_results.csv'}")
    plot_results(rows)
    example_grids(model, images)
    summarize(rows)
    shutil.rmtree(OUT / "_data", ignore_errors=True)
    shutil.rmtree(OUT / "_val", ignore_errors=True)


if __name__ == "__main__":
    main()
