"""Measure brightness / exposure of a video, image, or directory of images.

The robustness studies in this project (robustness.py, video_robustness.py,
failure_sweep.py) apply a *known* multiplicative gain via
``corruptions.brightness`` and watch the YOLO11 tracker degrade. This tool is
the inverse instrument: given footage of *unknown* exposure, it estimates how
bright/dark it is, whether highlights or shadows are clipping, and what that
implies for detection quality given what we already measured:

  * Brightness is the mildest of our corruptions (<=30% mAP loss over the
    whole 0.2x-3.0x gain sweep).
  * Over-exposure hurts MORE than under-exposure (3.0x gain -> about -30% mAP
    vs 0.2x -> about -11%): highlight clipping destroys texture irreversibly,
    while a dark frame keeps its gradients and can even be re-normalized.
  * The detector is happiest around 0.5x-1.0x gain, i.e. mid-to-slightly-dark
    exposure, and stays usable down to ~0.3x.

Per frame we compute (all on 8-bit luma, ITU-R BT.709 from BGR:
Y = 0.2126 R + 0.7152 G + 0.0722 B):

  mean_luma / median_luma / p5_luma / p95_luma
  rms_contrast   - std of Y (texture proxy; clipping crushes this)
  clip_low_pct   - % of pixels with Y <= 5   (crushed shadows)
  clip_high_pct  - % of pixels with Y >= 250 (blown highlights - the killer)
  exposure_index - mean_luma / 118

WHY 118? 118 is the empirical mean luma of a well-exposed frame in this
project: the clean ``samples/`` clips average ~110-125 mean luma, and 118/255
is ~46%, i.e. the classic "mid-grey renders just under half of full scale"
target for 8-bit video. Dividing by it makes exposure_index read like the
gain factor of the robustness sweeps: exposure_index ~= 0.3 means the footage
looks like our 0.3x-gain "dark" condition, ~= 1.0 means nominal exposure,
> 2 means it looks like the over-driven end of the sweep.

Per-frame classification bands (on mean luma, 0-255):

  very dark     mean < 40    (< ~0.34 exposure_index; below the safe range)
  dark          40 - 80      (0.34-0.68x; the sweet-spot's dark edge - safe)
  well-exposed  80 - 150     (0.68-1.27x; nominal)
  bright        150 - 200    (1.27-1.7x; heading toward clipping)
  over-exposed  > 200 OR clip_high_pct > 10%

The bands are anchored to the sweep: 40 is roughly 0.3x gain on a nominal
frame (the darkest condition the tracker handled gracefully), 80-150 brackets
the 0.68-1.27x region around the detector's 0.5-1.0x comfort zone, and the
clip_high_pct>10% override catches frames whose *mean* still looks sane but
whose highlights are already destroyed (the failure mode that actually costs
mAP). The override is checked first, so a clipped frame is always flagged.

Outputs (default brightness_out/):
  <stem>_brightness.csv   per-frame metrics
  <stem>_brightness.png   mean-luma timeline over shaded classification
                          bands, clip percentages on a twin axis
  <stem>_summary.txt      overall classification, band shares, worst-case
                          frames, DETECTION-RISK verdict (also printed)
  <stemA>_vs_<stemB>_compare.png  when two (or more) inputs are given

Usage:
    python3 brightness_meter.py samples/people.mp4
    python3 brightness_meter.py video_robustness_out/people_dark.mp4 --stride 2
    python3 brightness_meter.py robustness_out/example_brightness.png
    python3 brightness_meter.py samples/people.mp4 video_robustness_out/people_dark.mp4
    python3 brightness_meter.py frames_dir/ --max-frames 200 --out brightness_out

Importable API (used by other project tools):
    from brightness_meter import measure_brightness
    stats = measure_brightness(img_bgr)   # -> dict of the metrics above
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Empirical mean luma of a well-exposed frame in this project (clean sample
# clips average ~110-125; 118/255 ~ 46% grey). See module docstring.
WELL_EXPOSED_LUMA = 118.0

CLIP_LOW_THR = 5      # Y <= 5   counts as crushed shadow
CLIP_HIGH_THR = 250   # Y >= 250 counts as blown highlight

# (name, luma_lo, luma_hi, shade colour) - see docstring for the rationale.
BANDS = [
    ("very dark", 0, 40, "#31315e"),
    ("dark", 40, 80, "#5a5a8f"),
    ("well-exposed", 80, 150, "#7fbf7f"),
    ("bright", 150, 200, "#e8c96b"),
    ("over-exposed", 200, 256, "#e07b54"),
]
BAND_NAMES = [b[0] for b in BANDS]

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# --------------------------------------------------------------------------
# Core measurement (importable)
# --------------------------------------------------------------------------

def bt709_luma(img_bgr: np.ndarray) -> np.ndarray:
    """ITU-R BT.709 luma from a BGR (OpenCV order!) uint8 image, as float32.

    Y = 0.2126 R + 0.7152 G + 0.0722 B, so channel 2 (R) gets 0.2126 and
    channel 0 (B) gets 0.0722. Grayscale input is returned as-is (float32).
    """
    if img_bgr.ndim == 2:
        return img_bgr.astype(np.float32)
    b = img_bgr[..., 0].astype(np.float32)
    g = img_bgr[..., 1].astype(np.float32)
    r = img_bgr[..., 2].astype(np.float32)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def classify(mean_luma: float, clip_high_pct: float) -> str:
    """Classification bands documented in the module docstring.

    The clipping override is checked first: a frame with >10% blown pixels is
    'over-exposed' no matter what its mean says, because clipped texture is
    what actually costs the detector mAP.
    """
    if mean_luma > 200 or clip_high_pct > 10.0:
        return "over-exposed"
    if mean_luma < 40:
        return "very dark"
    if mean_luma < 80:
        return "dark"
    if mean_luma < 150:
        return "well-exposed"
    return "bright"


def measure_brightness(img_bgr: np.ndarray) -> dict:
    """Measure brightness/exposure statistics of one BGR uint8 frame.

    Returns a dict with keys: mean_luma, median_luma, p5_luma, p95_luma,
    rms_contrast, clip_low_pct, clip_high_pct, exposure_index,
    classification. All lumas are BT.709 on the 0-255 scale;
    exposure_index = mean_luma / 118 so that 1.0 ~ well-exposed and the value
    reads like the gain factor used in the project's robustness sweeps.
    """
    y = bt709_luma(img_bgr)
    mean = float(y.mean())
    p5, med, p95 = (float(v) for v in np.percentile(y, [5, 50, 95]))
    clip_low = float((y <= CLIP_LOW_THR).mean() * 100.0)
    clip_high = float((y >= CLIP_HIGH_THR).mean() * 100.0)
    return {
        "mean_luma": mean,
        "median_luma": med,
        "p5_luma": p5,
        "p95_luma": p95,
        "rms_contrast": float(y.std()),
        "clip_low_pct": clip_low,
        "clip_high_pct": clip_high,
        "exposure_index": mean / WELL_EXPOSED_LUMA,
        "classification": classify(mean, clip_high),
    }


# --------------------------------------------------------------------------
# Input handling: video / image / directory, auto-detected
# --------------------------------------------------------------------------

def _detect_kind(path: Path) -> str:
    if path.is_dir():
        return "dir"
    ext = path.suffix.lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in IMAGE_EXTS:
        return "image"
    # Unknown extension: try image first (cheap), then video.
    if cv2.imread(str(path)) is not None:
        return "image"
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened()
    cap.release()
    if ok:
        return "video"
    raise SystemExit(f"error: cannot read {path} as image, video or directory")


def iter_frames(path: Path, stride: int, max_frames: int | None):
    """Yield (frame_index, time_s_or_None, img_bgr). Auto-detects input kind."""
    kind = _detect_kind(path)
    if kind == "video":
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise SystemExit(f"error: cannot open video {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        idx = taken = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, (idx / fps if fps > 0 else None), frame
                taken += 1
                if max_frames is not None and taken >= max_frames:
                    break
            idx += 1
        cap.release()
    elif kind == "image":
        img = cv2.imread(str(path))
        if img is None:
            raise SystemExit(f"error: cannot read image {path}")
        yield 0, None, img
    else:  # directory of images
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise SystemExit(f"error: no images ({'/'.join(sorted(IMAGE_EXTS))}) in {path}")
        files = files[::stride]
        if max_frames is not None:
            files = files[:max_frames]
        for i, f in enumerate(files):
            img = cv2.imread(str(f))
            if img is None:
                print(f"  warning: skipping unreadable {f.name}")
                continue
            yield i, None, img


def analyze_input(path: Path, stride: int, max_frames: int | None) -> list[dict]:
    """Run measure_brightness over every sampled frame; print progress."""
    print(f"\n== analyzing {path} (stride={stride}, "
          f"max_frames={'all' if max_frames is None else max_frames}) ==")
    rows = []
    for idx, t, frame in iter_frames(path, stride, max_frames):
        rec = {"frame": idx, "time_s": t}
        rec.update(measure_brightness(frame))
        rows.append(rec)
        if len(rows) == 1 or len(rows) % 50 == 0:
            print(f"  frame {idx:5d}  mean_luma={rec['mean_luma']:6.1f}  "
                  f"clip_high={rec['clip_high_pct']:5.2f}%  -> {rec['classification']}")
    if not rows:
        raise SystemExit(f"error: no frames read from {path}")
    print(f"  done: {len(rows)} frames measured")
    return rows


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------

CSV_FIELDS = ["frame", "time_s", "mean_luma", "median_luma", "p5_luma",
              "p95_luma", "rms_contrast", "clip_low_pct", "clip_high_pct",
              "exposure_index", "classification"]


def write_csv(rows: list[dict], out_csv: Path) -> None:
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            r = dict(r)
            for k in CSV_FIELDS:
                if isinstance(r.get(k), float):
                    r[k] = round(r[k], 4)
            w.writerow(r)
    print(f"  wrote {out_csv}")


def _shade_bands(ax) -> None:
    for name, lo, hi, color in BANDS:
        ax.axhspan(lo, min(hi, 255), color=color, alpha=0.18, lw=0)
        ax.text(1.005, (lo + min(hi, 255)) / 2, name, transform=ax.get_yaxis_transform(),
                fontsize=7, va="center", ha="left", color="0.25")


def plot_timeline(rows: list[dict], title: str, out_png: Path) -> None:
    x = [r["frame"] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5))
    _shade_bands(ax)
    ax.plot(x, [r["mean_luma"] for r in rows], color="k", lw=1.5, label="mean luma (BT.709)")
    if len(rows) == 1:  # single image: a line would be invisible
        ax.scatter(x, [rows[0]["mean_luma"]], color="k", zorder=5, s=60)
    ax.set_xlabel("frame")
    ax.set_ylabel("luma (0-255)")
    ax.set_ylim(0, 255)
    ax.set_title(title)

    ax2 = ax.twinx()
    ax2.plot(x, [r["clip_high_pct"] for r in rows], color="crimson", lw=1.0,
             ls="--", label="clip high % (Y>=250)")
    ax2.plot(x, [r["clip_low_pct"] for r in rows], color="royalblue", lw=1.0,
             ls=":", label="clip low % (Y<=5)")
    ax2.set_ylabel("% pixels clipped")
    ax2.set_ylim(bottom=0)
    ax2.spines["right"].set_position(("outward", 45))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_png}")


def plot_compare(all_rows: dict[str, list[dict]], out_png: Path) -> None:
    """Overlay mean-luma timelines (and clip-high %) of several inputs."""
    fig, ax = plt.subplots(figsize=(11, 5))
    _shade_bands(ax)
    ax2 = ax.twinx()
    colors = plt.cm.tab10.colors
    for i, (stem, rows) in enumerate(all_rows.items()):
        c = colors[i % len(colors)]
        x = [r["frame"] for r in rows]
        mean_all = float(np.mean([r["mean_luma"] for r in rows]))
        ax.plot(x, [r["mean_luma"] for r in rows], color=c, lw=1.5,
                label=f"{stem} (mean {mean_all:.0f}, EI {mean_all / WELL_EXPOSED_LUMA:.2f}x)")
        ax2.plot(x, [r["clip_high_pct"] for r in rows], color=c, lw=0.9, ls="--", alpha=0.7)
    ax.set_xlabel("frame")
    ax.set_ylabel("mean luma (0-255)")
    ax.set_ylim(0, 255)
    ax2.set_ylabel("% pixels clipped high (dashed)")
    ax2.set_ylim(bottom=0)
    ax2.spines["right"].set_position(("outward", 45))
    ax.set_title("Brightness comparison (solid = mean luma, dashed = clip-high %)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_png}")


# --------------------------------------------------------------------------
# Summary + detection-risk verdict (wired to the project findings)
# --------------------------------------------------------------------------

def detection_risk_verdict(overall_mean: float, ei: float, band_pct: dict,
                           mean_clip_high: float, mean_clip_low: float) -> str:
    """Map the measurements onto the project's YOLO11 robustness findings."""
    over = band_pct.get("over-exposed", 0.0)
    bright = band_pct.get("bright", 0.0)
    vdark = band_pct.get("very dark", 0.0)
    if over > 20 or mean_clip_high > 10:
        return ("HIGH RISK - over-exposure. This footage sits at the harmful end of "
                "the sweep: in our study a 3.0x gain cost ~30% mAP, and highlight "
                "clipping destroys texture IRREVERSIBLY (no post-hoc fix). "
                f"{over:.0f}% of frames are over-exposed / clipping "
                f"(mean clip-high {mean_clip_high:.1f}%). Expect the largest "
                "brightness-related detection losses seen in this project.")
    if over + bright > 30 or mean_clip_high > 3:
        return ("ELEVATED RISK - trending bright. Over-exposure is the direction that "
                "actually hurts (3.0x gain -> -30% mAP vs only -11% at 0.2x); "
                f"exposure index {ei:.2f}x with {mean_clip_high:.1f}% highlights "
                "already clipping. Watch for texture loss on light surfaces; "
                "consider pulling exposure down toward the 0.5-1.0x comfort zone.")
    if vdark > 50 or ei < 0.25:
        return ("MODERATE RISK - very dark. Darkness is the *mild* direction "
                "(0.2x gain -> only ~-11% mAP) but this footage is below the ~0.3x "
                f"level that stayed comfortable (exposure index {ei:.2f}x). Expect "
                "some missed small/low-contrast objects; gradients are preserved, so "
                "a simple gain-up / normalization should recover most of it.")
    if band_pct.get("dark", 0.0) + vdark > 50:
        return ("LOW RISK - dark but safe. This matches the 0.3-0.7x-gain regime where "
                "the detector barely suffered (<= ~11% mAP loss even at 0.2x), and the "
                "detector is actually happiest around 0.5-1.0x, i.e. mid-to-slightly-"
                f"dark exposure. Exposure index {ei:.2f}x, clip-low "
                f"{mean_clip_low:.1f}%. No action needed for detection.")
    return ("LOW RISK - well-exposed. Mean luma sits in the detector's comfort zone "
            f"(exposure index {ei:.2f}x, ~0.5-1.0x-gain equivalent), with negligible "
            f"clipping (high {mean_clip_high:.1f}%, low {mean_clip_low:.1f}%). "
            "Brightness was the mildest corruption in our sweeps; expect no "
            "meaningful brightness-related detection loss here.")


def summarize(stem: str, src: Path, rows: list[dict], out_txt: Path) -> str:
    n = len(rows)
    means = np.array([r["mean_luma"] for r in rows])
    clips_hi = np.array([r["clip_high_pct"] for r in rows])
    clips_lo = np.array([r["clip_low_pct"] for r in rows])
    overall_mean = float(means.mean())
    overall_clip_hi = float(clips_hi.mean())
    overall_clip_lo = float(clips_lo.mean())
    ei = overall_mean / WELL_EXPOSED_LUMA
    overall_cls = classify(overall_mean, overall_clip_hi)
    band_pct = {b: 100.0 * sum(r["classification"] == b for r in rows) / n
                for b in BAND_NAMES}

    darkest = rows[int(means.argmin())]
    brightest = rows[int(means.argmax())]
    worst_hi = rows[int(clips_hi.argmax())]
    worst_lo = rows[int(clips_lo.argmax())]

    lines = [
        f"BRIGHTNESS SUMMARY - {src}",
        f"frames measured: {n}",
        "",
        f"overall classification : {overall_cls}",
        f"mean luma (BT.709)     : {overall_mean:.1f} / 255",
        f"exposure index         : {ei:.2f}x  (mean_luma/118; ~gain factor vs well-exposed)",
        f"mean RMS contrast      : {np.mean([r['rms_contrast'] for r in rows]):.1f}",
        f"mean clip high / low   : {overall_clip_hi:.2f}% / {overall_clip_lo:.2f}%",
        "",
        "frames per band:",
    ]
    lines += [f"  {b:<13s}: {band_pct[b]:5.1f}%" for b in BAND_NAMES]
    lines += [
        "",
        "worst-case frames:",
        f"  darkest        : frame {darkest['frame']} (mean luma {darkest['mean_luma']:.1f})",
        f"  brightest      : frame {brightest['frame']} (mean luma {brightest['mean_luma']:.1f})",
        f"  most clip-high : frame {worst_hi['frame']} ({worst_hi['clip_high_pct']:.2f}% pixels >=250)",
        f"  most clip-low  : frame {worst_lo['frame']} ({worst_lo['clip_low_pct']:.2f}% pixels <=5)",
        "",
        "DETECTION-RISK verdict:",
        "  " + detection_risk_verdict(overall_mean, ei, band_pct,
                                      overall_clip_hi, overall_clip_lo),
        "",
    ]
    text = "\n".join(lines)
    out_txt.write_text(text)
    print(f"  wrote {out_txt}")
    return text


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Measure brightness/exposure of videos, images or image "
                    "directories and rate the detection risk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path,
                    help="video file(s), image(s) and/or director(ies) of images; "
                         "give two inputs to get an A/B overlay plot")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="max number of sampled frames per input (default: all)")
    ap.add_argument("--stride", type=int, default=1,
                    help="sample every Nth frame")
    ap.add_argument("--out", type=Path, default=Path("brightness_out"),
                    help="output directory")
    args = ap.parse_args(argv)
    if args.stride < 1:
        ap.error("--stride must be >= 1")

    args.out.mkdir(parents=True, exist_ok=True)
    all_rows: dict[str, list[dict]] = {}
    for src in args.inputs:
        if not src.exists():
            raise SystemExit(f"error: {src} does not exist")
        stem = src.stem if not src.is_dir() else src.name
        while stem in all_rows:  # two inputs with the same stem
            stem += "_2"
        rows = analyze_input(src, args.stride, args.max_frames)
        all_rows[stem] = rows
        write_csv(rows, args.out / f"{stem}_brightness.csv")
        plot_timeline(rows, f"Brightness timeline - {src.name}",
                      args.out / f"{stem}_brightness.png")
        print()
        print(summarize(stem, src, rows, args.out / f"{stem}_summary.txt"))

    if len(all_rows) >= 2:
        stems = list(all_rows)
        plot_compare(all_rows, args.out / f"{stems[0]}_vs_{stems[1]}_compare.png")

    print(f"all outputs in {args.out}/")


if __name__ == "__main__":
    main()
