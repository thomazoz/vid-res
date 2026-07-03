"""Case analysis: divide OTB sequences into cases that lead to FAILURE or SUCCESS.

Joins every per-sequence table in the project (YOLO OTB eval, CSRT summary,
hand-labeled catalog, camera-motion metrics, target-complexity metrics,
corruption sweep) and builds a data-driven taxonomy of when tracking works
and when it breaks:

  1. Tier each sequence for BOTH trackers:
         SUCCESS  mean_iou >= 0.5
         PARTIAL  0.3 <= mean_iou < 0.5
         FAILURE  mean_iou < 0.3
     and flag tracker disagreements (one tracker SUCCESS, the other FAILURE).
  2. Factor analysis:
         (a) categorical factors (primary_challenge, object_category,
             difficulty): per-category tier counts + mean IoU
         (b) numeric factors (num_frames, camera-motion, complexity):
             Spearman rho vs mean_iou + Mann-Whitney U SUCCESS vs FAILURE
         (c) depth-2 decision tree SUCCESS-vs-FAILURE (interpretability only)
  3. Corruption-tolerance knees from failure_sweep_out/sweep_results.csv:
     per sequence x condition, the lowest severity where retention (at the
     conf level closest to --conf) drops below --knee-retention.

Outputs (all in --outdir, default case_analysis_out/):
    cases.csv                     full joined per-sequence table + tiers + knees
    factor_ranking.csv            numeric factor stats ranked by |rho|
    knees.csv                     corruption knees per sequence x condition
    sequences_sorted_by_iou.png   per-tracker IoU bars colored by challenge
    factor_boxplots.png           top-6 numeric factors, SUCCESS vs FAILURE
    decision_tree.txt             exported depth-2 tree rules (both trackers)
    CASES.md                      readable report: cases, factors, knees, rules

Usage:
    python3 case_analysis.py
    python3 case_analysis.py --outdir case_analysis_out
    python3 case_analysis.py --conf 0.25 --knee-retention 0.5
    python3 case_analysis.py --success-thr 0.5 --failure-thr 0.3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent

# ── input locations ────────────────────────────────────────────────────────────
YOLO_CSV = ROOT / "otb_eval_out" / "otb_results.csv"
CSRT_CSV = ROOT / "results" / "summary.csv"
CATALOG_CSV = ROOT / "dataset_catalog.csv"
CAMERA_CSV = ROOT / "camera_motion_out" / "per_sequence_camera_motion.csv"
COMPLEXITY_CSVS = [
    ROOT / "complexity_out" / "complexity_per_sequence.csv",
    ROOT / "complexity_out" / "per_sequence_complexity.csv",
]
SWEEP_CSV = ROOT / "failure_sweep_out" / "sweep_results.csv"
PRED_DIR = ROOT / "results"

CAMERA_FACTORS = ["trans_px_mean", "trans_px_p95", "rot_deg_mean",
                  "zoom_abs_mean", "jerk_px_mean", "inlier_frac_mean"]
COMPLEXITY_FACTORS = ["silhouette_complexity", "convex_hull_ratio",
                      "texture_entropy", "color_entropy", "edge_density",
                      "fg_bg_contrast"]
CATEGORICAL_FACTORS = ["primary_challenge", "object_category", "difficulty"]

SMALL_N_CAVEAT = ("NOTE: with only ~31 sequences, per-category groups are small "
                  "(often n<=5); treat every per-category statistic as "
                  "descriptive, not inferential.")


# ── loading ────────────────────────────────────────────────────────────────────

def read_csv_safe(path: Path, label: str) -> pd.DataFrame | None:
    """Read a CSV, returning None (with a printed warning) if missing/empty."""
    if not path.exists():
        print(f"  [warn] {label}: {path} not found — skipping")
        return None
    if path.stat().st_size == 0:
        print(f"  [warn] {label}: {path} is empty (0 bytes) — skipping")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # malformed / partial file
        print(f"  [warn] {label}: could not parse {path} ({exc}) — skipping")
        return None
    if df.empty:
        print(f"  [warn] {label}: {path} has a header but no rows — skipping")
        return None
    print(f"  [ok]   {label}: {len(df)} rows, {len(df.columns)} cols")
    return df


def tier_of(iou: float, success_thr: float, failure_thr: float) -> str:
    if pd.isna(iou):
        return "NO_DATA"
    if iou >= success_thr:
        return "SUCCESS"
    if iou >= failure_thr:
        return "PARTIAL"
    return "FAILURE"


def csrt_first_loss(seq: str, thr: float = 0.1) -> tuple[float, float]:
    """First frame where CSRT per-frame IoU drops below thr, and the fraction
    of the sequence tracked before that. Returns (nan, nan) if unavailable."""
    p = PRED_DIR / seq / "predictions.csv"
    if not p.exists():
        return (np.nan, np.nan)
    try:
        d = pd.read_csv(p, usecols=["frame", "iou"])
    except Exception:
        return (np.nan, np.nan)
    if d.empty:
        return (np.nan, np.nan)
    lost = d[d["iou"] < thr]
    if lost.empty:
        return (np.nan, 1.0)
    first = int(lost["frame"].iloc[0])
    return (first, first / len(d))


def build_cases(args) -> tuple[pd.DataFrame, dict]:
    """Outer-join all per-sequence tables on the YOLO results as spine."""
    print("\n=== [1/6] Loading & joining per-sequence tables ===")
    coverage: dict[str, list[str]] = {}

    yolo = read_csv_safe(YOLO_CSV, "YOLO otb_results")
    if yolo is None:
        raise SystemExit("FATAL: YOLO results (the join spine) are required.")
    yolo = yolo.rename(columns={
        "mean_iou": "mean_iou_yolo", "success_rate": "success_auc_yolo",
        "success_at_50": "success_at_50_yolo", "precision": "precision_yolo",
        "mean_cerr": "mean_cerr_yolo", "ceiling_iou": "ceiling_iou_yolo",
    })
    cases = yolo.copy()

    csrt = read_csv_safe(CSRT_CSV, "CSRT summary")
    if csrt is not None:
        keep = csrt[["sequence", "mean_iou", "success_rate@0.5",
                     "throughput_fps"]].rename(columns={
            "mean_iou": "mean_iou_csrt",
            "success_rate@0.5": "success_at_50_csrt",
            "throughput_fps": "csrt_fps"})
        cases = cases.merge(keep, on="sequence", how="outer")

    catalog = read_csv_safe(CATALOG_CSV, "dataset catalog")
    if catalog is not None:
        cases = cases.merge(catalog, on="sequence", how="outer")

    cam = read_csv_safe(CAMERA_CSV, "camera motion")
    if cam is not None:
        cases = cases.merge(cam[["sequence"] + CAMERA_FACTORS],
                            on="sequence", how="outer")

    cx = None
    for p in COMPLEXITY_CSVS:
        cx = read_csv_safe(p, f"complexity ({p.name})")
        if cx is not None:
            break
    if cx is not None:
        cxcols = [c for c in COMPLEXITY_FACTORS if c in cx.columns]
        cases = cases.merge(cx[["sequence"] + cxcols],
                            on="sequence", how="outer")

    # unified frame count (spine 'frames', fall back to catalog num_frames)
    if "num_frames" in cases.columns:
        cases["num_frames"] = cases["num_frames"].fillna(cases.get("frames"))
    else:
        cases["num_frames"] = cases.get("frames")

    # per-sequence data coverage report
    blocks = {
        "YOLO": ["mean_iou_yolo"],
        "CSRT": ["mean_iou_csrt"] if "mean_iou_csrt" in cases else [],
        "catalog": ["primary_challenge"] if "primary_challenge" in cases else [],
        "camera_motion": [c for c in CAMERA_FACTORS if c in cases][:1],
        "complexity": [c for c in COMPLEXITY_FACTORS if c in cases][:1],
    }
    print("\n  Data coverage (sequences missing a block):")
    for block, cols in blocks.items():
        if not cols:
            print(f"    {block:<14} table entirely missing")
            coverage[block] = ["<table missing>"]
            continue
        missing = cases.loc[cases[cols[0]].isna(), "sequence"].tolist()
        coverage[block] = missing
        print(f"    {block:<14} missing for: {missing if missing else 'none'}")

    # tiers
    print("\n=== [2/6] Tiering sequences (both trackers) ===")
    cases["tier_yolo"] = cases["mean_iou_yolo"].apply(
        tier_of, args=(args.success_thr, args.failure_thr))
    if "mean_iou_csrt" in cases:
        cases["tier_csrt"] = cases["mean_iou_csrt"].apply(
            tier_of, args=(args.success_thr, args.failure_thr))
    else:
        cases["tier_csrt"] = "NO_DATA"

    def disagreement(r):
        if r["tier_yolo"] == "SUCCESS" and r["tier_csrt"] == "FAILURE":
            return "YOLO_succeeds_CSRT_fails"
        if r["tier_csrt"] == "SUCCESS" and r["tier_yolo"] == "FAILURE":
            return "CSRT_succeeds_YOLO_fails"
        return ""
    cases["disagreement"] = cases.apply(disagreement, axis=1)

    # optional CSRT failure onset from per-frame predictions
    onset = cases["sequence"].apply(lambda s: pd.Series(
        csrt_first_loss(s), index=["csrt_first_loss_frame",
                                   "csrt_frac_before_loss"]))
    cases = pd.concat([cases, onset], axis=1)

    for trk in ["yolo", "csrt"]:
        counts = cases[f"tier_{trk}"].value_counts().to_dict()
        total = sum(counts.values())
        print(f"  {trk.upper():<5} tiers: {counts}  (sum={total}, "
              f"rows={len(cases)})")
    n_dis = (cases["disagreement"] != "").sum()
    print(f"  Disagreements (one SUCCESS, other FAILURE): {n_dis}")
    return cases, coverage


# ── factor analysis ────────────────────────────────────────────────────────────

def categorical_analysis(cases: pd.DataFrame) -> dict[str, pd.DataFrame]:
    print("\n=== [3/6] Categorical factor analysis ===")
    print(f"  {SMALL_N_CAVEAT}")
    tables = {}
    ev = cases[cases["tier_yolo"] != "NO_DATA"]  # evaluated sequences only
    for fac in CATEGORICAL_FACTORS:
        if fac not in ev.columns:
            print(f"  [warn] categorical factor {fac} unavailable")
            continue
        rows = []
        for val, g in ev.groupby(fac):
            row = {fac: val, "n": len(g)}
            for trk in ["yolo", "csrt"]:
                tc = g[f"tier_{trk}"].value_counts()
                row[f"{trk}_SUCCESS"] = int(tc.get("SUCCESS", 0))
                row[f"{trk}_PARTIAL"] = int(tc.get("PARTIAL", 0))
                row[f"{trk}_FAILURE"] = int(tc.get("FAILURE", 0))
                row[f"{trk}_mean_iou"] = round(
                    g[f"mean_iou_{trk}"].mean(), 3)
            rows.append(row)
        t = pd.DataFrame(rows).sort_values("csrt_mean_iou", ascending=False)
        tables[fac] = t
        print(f"\n  -- {fac} --")
        print(t.to_string(index=False))
    return tables


def numeric_analysis(cases: pd.DataFrame, numeric_factors: list[str]
                     ) -> pd.DataFrame:
    print("\n=== [4/6] Numeric factor analysis (Spearman + Mann-Whitney) ===")
    rows = []
    for trk in ["yolo", "csrt"]:
        iou_col, tier_col = f"mean_iou_{trk}", f"tier_{trk}"
        ev = cases[cases[tier_col] != "NO_DATA"]
        for fac in numeric_factors:
            if fac not in ev.columns:
                continue
            sub = ev[[fac, iou_col, tier_col]].dropna(subset=[fac, iou_col])
            if len(sub) < 5:
                continue
            rho, p = stats.spearmanr(sub[fac], sub[iou_col])
            s = sub.loc[sub[tier_col] == "SUCCESS", fac]
            f = sub.loc[sub[tier_col] == "FAILURE", fac]
            if len(s) >= 2 and len(f) >= 2:
                u, up = stats.mannwhitneyu(s, f, alternative="two-sided")
            else:
                u, up = np.nan, np.nan
            rows.append({
                "tracker": trk.upper(), "factor": fac, "n": len(sub),
                "spearman_rho": round(rho, 3), "spearman_p": round(p, 4),
                "mwu_U": u if pd.isna(u) else round(float(u), 1),
                "mwu_p": up if pd.isna(up) else round(float(up), 4),
                "n_success": len(s), "n_failure": len(f),
                "median_success": round(s.median(), 3) if len(s) else np.nan,
                "median_failure": round(f.median(), 3) if len(f) else np.nan,
            })
    rank = pd.DataFrame(rows)
    rank["abs_rho"] = rank["spearman_rho"].abs()
    rank = rank.sort_values(["tracker", "abs_rho"],
                            ascending=[True, False]).drop(columns="abs_rho")
    for trk in ["CSRT", "YOLO"]:
        top = rank[rank["tracker"] == trk].head(6)
        print(f"\n  Top factors for {trk} (by |rho| vs mean_iou):")
        print(top.to_string(index=False))
    return rank


def fit_trees(cases: pd.DataFrame, numeric_factors: list[str],
              out_path: Path) -> dict[str, str]:
    """Depth-2 decision trees SUCCESS-vs-FAILURE per tracker (interpretability
    only — n is small, so these are descriptive rules, not a validated model)."""
    from sklearn.tree import DecisionTreeClassifier, export_text
    print("\n=== [5/6] Depth-2 decision trees (SUCCESS vs FAILURE) ===")
    texts = {}
    lines = ["Depth-2 DecisionTreeClassifier rules, SUCCESS vs FAILURE.",
             "Fit on ALL rows (no held-out set): read these as descriptive",
             f"rules, not a validated classifier — n is small. {SMALL_N_CAVEAT}",
             ""]
    for trk in ["csrt", "yolo"]:
        sub = cases[cases[f"tier_{trk}"].isin(["SUCCESS", "FAILURE"])].copy()
        if len(sub) < 8 or sub[f"tier_{trk}"].nunique() < 2:
            msg = f"[{trk.upper()}] not enough data to fit a tree"
            print(f"  {msg}")
            lines += [msg, ""]
            continue
        num = [f for f in numeric_factors if f in sub.columns]
        X = sub[num].copy()
        cats = [c for c in CATEGORICAL_FACTORS if c in sub.columns]
        if cats:
            X = pd.concat([X, pd.get_dummies(sub[cats], prefix=cats,
                                             dtype=float)], axis=1)
        y = sub[f"tier_{trk}"]
        clf = DecisionTreeClassifier(max_depth=2, class_weight="balanced",
                                     random_state=0)
        try:
            clf.fit(X, y)  # sklearn >=1.3 trees accept NaN natively
        except ValueError:
            X = X.fillna(X.median(numeric_only=True))
            clf.fit(X, y)
        acc = clf.score(X, y)
        txt = export_text(clf, feature_names=list(X.columns),
                          class_names=list(clf.classes_))
        header = (f"[{trk.upper()}]  n={len(y)} "
                  f"({(y == 'SUCCESS').sum()} SUCCESS / "
                  f"{(y == 'FAILURE').sum()} FAILURE), "
                  f"training accuracy={acc:.2f} (descriptive only)")
        print(f"  {header}")
        for ln in txt.rstrip().splitlines():
            print(f"    {ln}")
        lines += [header, txt, ""]
        texts[trk] = txt
    out_path.write_text("\n".join(lines))
    print(f"  wrote {out_path}")
    return texts


# ── corruption knees ───────────────────────────────────────────────────────────

def severity_ladder(cond: str, severities: list[float]) -> list[list[float]]:
    """Order severities from mild to harsh. Brightness is bidirectional
    (darken <1.0 and brighten >1.0), so it yields two ladders."""
    sev = sorted(set(severities))
    if cond == "brightness":
        dark = sorted([s for s in sev if s < 1.0], reverse=True)   # 0.7 → 0.2
        bright = sorted([s for s in sev if s > 1.0])               # 1.3 → 3.0
        return [dark, bright] if dark or bright else [sev]
    return [sorted(s for s in sev if s > 0)]  # 0 = clean for blurs


def compute_knees(args) -> tuple[pd.DataFrame | None, str]:
    print("\n=== [6/6] Corruption-tolerance knees from failure sweep ===")
    sweep = read_csv_safe(SWEEP_CSV, "failure sweep")
    if sweep is None:
        note = (f"Corruption sweep data unavailable ({SWEEP_CSV.name} missing "
                "or empty — failure_sweep.py may still be running). "
                "Re-run case_analysis.py once the sweep finishes.")
        print(f"  [warn] {note}")
        return None, note
    needed = {"sequence", "condition", "severity", "conf_thr"}
    if not needed.issubset(sweep.columns):
        note = f"sweep file lacks required columns {needed}; skipping knees"
        print(f"  [warn] {note}")
        return None, note
    ret_col = "retention" if "retention" in sweep.columns else \
        "retention_anyclass"
    confs = sweep["conf_thr"].unique()
    conf_used = float(confs[np.argmin(np.abs(confs - args.conf))])
    sw = sweep[np.isclose(sweep["conf_thr"], conf_used)]
    print(f"  using conf_thr={conf_used} (closest to {args.conf}), "
          f"retention column='{ret_col}', {len(sw)} rows")

    rows = []
    for (seq, cond), g in sw.groupby(["sequence", "condition"]):
        for i, ladder in enumerate(severity_ladder(
                cond, g["severity"].tolist())):
            if not ladder:
                continue
            cond_name = cond
            if cond == "brightness":
                cond_name = "brightness_dark" if ladder[0] < 1.0 \
                    else "brightness_bright"
            ret = {s: g.loc[g["severity"] == s, ret_col].mean()
                   for s in ladder}
            knee, knee_pos = np.nan, np.nan
            for pos, s in enumerate(ladder, start=1):
                if ret[s] < args.knee_retention:
                    knee, knee_pos = s, pos
                    break
            half = int(np.ceil(len(ladder) / 2))
            if pd.isna(knee):
                frag = "tolerant"
            elif knee_pos <= half:
                frag = "fragile"
            else:
                frag = "moderate"
            rows.append({
                "sequence": seq, "condition": cond_name,
                "conf_thr": conf_used,
                "knee_severity": knee, "knee_position": knee_pos,
                "ladder_len": len(ladder),
                "min_retention": round(min(ret.values()), 3),
                "fragility": frag,
                "label": f"{cond_name}-{frag}",
            })
    knees = pd.DataFrame(rows)
    if knees.empty:
        return None, "sweep parsed but produced no knee rows"
    print("\n  Fragility counts per condition:")
    tab = knees.groupby(["condition", "fragility"]).size().unstack(fill_value=0)
    print(tab.to_string())
    note = (f"knees computed at conf_thr={conf_used}, retention<"
            f"{args.knee_retention} threshold, column '{ret_col}'")
    return knees, note


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_sorted_bars(cases: pd.DataFrame, out: Path, s_thr: float,
                     f_thr: float):
    ev = cases[cases["tier_yolo"] != "NO_DATA"].copy()
    challenges = sorted(ev["primary_challenge"].fillna("unknown").unique())
    cmap = plt.get_cmap("tab10")
    colors = {c: cmap(i % 10) for i, c in enumerate(challenges)}
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    for ax, trk in zip(axes, ["yolo", "csrt"]):
        col = f"mean_iou_{trk}"
        d = ev.dropna(subset=[col]).sort_values(col, ascending=False)
        bar_colors = [colors[c] for c in
                      d["primary_challenge"].fillna("unknown")]
        ax.bar(range(len(d)), d[col], color=bar_colors)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(d["sequence"], rotation=75, ha="right", fontsize=8)
        ax.axhline(s_thr, color="green", ls="--", lw=1,
                   label=f"SUCCESS >= {s_thr}")
        ax.axhline(f_thr, color="red", ls="--", lw=1,
                   label=f"FAILURE < {f_thr}")
        ax.set_ylabel("mean IoU")
        ax.set_title(f"{trk.upper()} — sequences sorted by mean IoU "
                     "(color = primary challenge)")
        ax.set_ylim(0, 1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c])
               for c in challenges]
    axes[0].legend(handles + axes[0].get_lines(),
                   challenges + [l.get_label() for l in axes[0].get_lines()],
                   fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_factor_boxplots(cases: pd.DataFrame, rank: pd.DataFrame, out: Path):
    top = (rank.assign(abs_rho=rank["spearman_rho"].abs())
           .sort_values("abs_rho", ascending=False)
           .drop_duplicates("factor").head(6)["factor"].tolist())
    if not top:
        print("  [warn] no numeric factors ranked; skipping boxplots")
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, fac in zip(axes.flat, top):
        data, labels, colors = [], [], []
        for trk, col in [("YOLO", "tier_yolo"), ("CSRT", "tier_csrt")]:
            for tier, c in [("SUCCESS", "tab:green"), ("FAILURE", "tab:red")]:
                v = cases.loc[cases[col] == tier, fac].dropna()
                data.append(v)
                labels.append(f"{trk}\n{tier[:4]}\n(n={len(v)})")
                colors.append(c)
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                        showfliers=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
        rho = rank.loc[rank["factor"] == fac]
        sub = ", ".join(f"{r.tracker} rho={r.spearman_rho:+.2f}"
                        for r in rho.itertuples())
        ax.set_title(f"{fac}\n{sub}", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
    for ax in axes.flat[len(top):]:
        ax.axis("off")
    fig.suptitle("Top numeric factors — SUCCESS vs FAILURE distributions",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# ── report ─────────────────────────────────────────────────────────────────────

def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def shared_conditions(g: pd.DataFrame) -> str:
    """Summarize what a group of sequences has in common."""
    bits = []
    for fac in ["primary_challenge", "object_category", "difficulty"]:
        if fac in g.columns and g[fac].notna().any():
            vc = g[fac].value_counts()
            top = ", ".join(f"{k} ({v}/{len(g)})"
                            for k, v in vc.head(3).items())
            bits.append(f"{fac}: {top}")
    for fac, fmt in [("trans_px_mean", "{:.2f}px"),
                     ("jerk_px_mean", "{:.2f}px"),
                     ("num_frames", "{:.0f}")]:
        if fac in g.columns and g[fac].notna().any():
            bits.append(f"median {fac}: {fmt.format(g[fac].median())}")
    return "; ".join(bits)


def build_rules(cases, rank, cat_tables, knees, tree_texts, args) -> list[str]:
    """Generate IF/THEN rules strictly from the computed statistics."""
    rules = []
    ev = cases[cases["tier_yolo"] != "NO_DATA"]

    # rule from top CSRT numeric factor (median split)
    csrt_rank = rank[rank["tracker"] == "CSRT"]
    if not csrt_rank.empty:
        r = csrt_rank.iloc[0]
        med = ev[r["factor"]].median()
        hi = ev[ev[r["factor"]] > med]
        lo = ev[ev[r["factor"]] <= med]
        hs = (hi["tier_csrt"] == "SUCCESS").mean()
        ls = (lo["tier_csrt"] == "SUCCESS").mean()
        direction = ">" if hs < ls else "<="
        bad, good = (hs, ls) if hs < ls else (ls, hs)
        rules.append(
            f"IF `{r['factor']}` {direction} {med:.3g} (dataset median) THEN "
            f"CSRT success rate drops to {bad:.0%} vs {good:.0%} on the other "
            f"side (Spearman rho={r['spearman_rho']:+.2f}, "
            f"p={r['spearman_p']:.3f}, n={r['n']}).")

    # rule from worst / best primary_challenge for CSRT
    if "primary_challenge" in cat_tables:
        t = cat_tables["primary_challenge"]
        worst, best = t.iloc[-1], t.iloc[0]
        rules.append(
            f"IF primary challenge is **{worst['primary_challenge']}** THEN "
            f"expect CSRT trouble: mean IoU {worst['csrt_mean_iou']:.2f} with "
            f"{worst['csrt_FAILURE']}/{worst['n']} outright failures "
            f"(small n={worst['n']}).")
        rules.append(
            f"IF primary challenge is **{best['primary_challenge']}** THEN "
            f"expect CSRT success: mean IoU {best['csrt_mean_iou']:.2f}, "
            f"{best['csrt_SUCCESS']}/{best['n']} SUCCESS "
            f"(small n={best['n']}).")

    # rule about YOLO and non-COCO objects
    if "object_category" in cat_tables:
        t = cat_tables["object_category"].sort_values("yolo_mean_iou")
        worst, best = t.iloc[0], t.iloc[-1]
        rules.append(
            f"IF the target's category is **{worst['object_category']}** "
            f"THEN YOLO tracking collapses: mean IoU "
            f"{worst['yolo_mean_iou']:.2f}, {worst['yolo_FAILURE']}/"
            f"{worst['n']} FAILURE — vs **{best['object_category']}** at mean "
            f"IoU {best['yolo_mean_iou']:.2f} "
            f"({best['yolo_SUCCESS']}/{best['n']} SUCCESS).")

    # rule from tracker disagreement pattern
    dis = ev[ev["disagreement"] == "CSRT_succeeds_YOLO_fails"]
    if len(dis):
        rules.append(
            f"IF CSRT succeeds on a sequence, do NOT assume YOLO will: on "
            f"{len(dis)}/{len(ev)} sequences CSRT is SUCCESS while YOLO is "
            f"FAILURE (e.g., {', '.join(dis['sequence'].head(5))}). The "
            f"reverse ({'YOLO_succeeds_CSRT_fails'}) happens on "
            f"{(ev['disagreement'] == 'YOLO_succeeds_CSRT_fails').sum()} "
            f"sequences.")

    # rule from second CSRT factor if meaningful
    if len(csrt_rank) > 1:
        r = csrt_rank.iloc[1]
        rules.append(
            f"IF `{r['factor']}` is high, expect "
            f"{'worse' if r['spearman_rho'] < 0 else 'better'} CSRT tracking "
            f"(rho={r['spearman_rho']:+.2f}, p={r['spearman_p']:.3f}; median "
            f"in SUCCESS={r['median_success']}, "
            f"FAILURE={r['median_failure']}).")

    # rule from knees
    if knees is not None and not knees.empty:
        for cond in knees["condition"].unique():
            k = knees[knees["condition"] == cond]
            frag = k[k["fragility"] == "fragile"]
            if len(frag):
                rules.append(
                    f"IF the video suffers **{cond}** corruption THEN "
                    f"{len(frag)}/{len(k)} sequences lose >50% detection "
                    f"retention within the first half of the severity ladder "
                    f"(fragile: {', '.join(frag['sequence'].head(6))}).")
                break

    return rules


def write_cases_md(out_dir: Path, cases, coverage, cat_tables, rank,
                   knees, knee_note, tree_texts, rules, args):
    ev = cases[cases["tier_yolo"] != "NO_DATA"]
    L = []
    L.append("# CASES — when tracking succeeds and when it fails\n")
    L.append(f"Auto-generated by `case_analysis.py` on {pd.Timestamp.now():%Y-%m-%d %H:%M}. "
             f"Tiers: SUCCESS mean IoU >= {args.success_thr}, PARTIAL "
             f">= {args.failure_thr}, FAILURE < {args.failure_thr}.\n")

    L.append("## Data coverage\n")
    L.append(f"- {len(cases)} sequences in the joined table; "
             f"{len(ev)} have tracker evaluations.")
    for block, missing in coverage.items():
        L.append(f"- **{block}**: missing for "
                 f"{', '.join(missing) if missing else 'none'}")
    L.append("")

    L.append("## Outcome tiers\n")
    for trk in ["yolo", "csrt"]:
        vc = cases[f"tier_{trk}"].value_counts()
        L.append(f"**{trk.upper()}**: " + ", ".join(
            f"{k}={v}" for k, v in vc.items()))
    L.append("")

    def tier_block(title, trk, tier):
        g = ev[ev[f"tier_{trk}"] == tier].sort_values(
            f"mean_iou_{trk}", ascending=(tier == "SUCCESS"))
        if g.empty:
            return [f"### {title}\n", "_none_\n"]
        names = ", ".join(
            f"{r.sequence} ({getattr(r, f'mean_iou_{trk}'):.2f})"
            for r in g.itertuples())
        return [f"### {title} (n={len(g)})\n", names + "\n",
                f"Shared conditions — {shared_conditions(g)}\n"]

    L.append("## The success cases\n")
    L += tier_block("CSRT SUCCESS", "csrt", "SUCCESS")
    L += tier_block("YOLO SUCCESS", "yolo", "SUCCESS")
    L.append("## The failure cases\n")
    L += tier_block("CSRT FAILURE", "csrt", "FAILURE")
    L += tier_block("YOLO FAILURE", "yolo", "FAILURE")
    part = ev[(ev["tier_yolo"] == "PARTIAL") | (ev["tier_csrt"] == "PARTIAL")]
    if len(part):
        L.append("### PARTIAL cases\n")
        L.append(", ".join(
            f"{r.sequence} (YOLO {r.mean_iou_yolo:.2f} / CSRT "
            f"{r.mean_iou_csrt:.2f})" for r in part.itertuples()) + "\n")

    L.append("## Tracker disagreements (most interesting cases)\n")
    dis = ev[ev["disagreement"] != ""]
    if dis.empty:
        L.append("_No sequence where one tracker succeeds and the other "
                 "fails._\n")
    else:
        cols = ["sequence", "disagreement", "mean_iou_yolo", "mean_iou_csrt",
                "primary_challenge", "object_category", "notes"]
        cols = [c for c in cols if c in dis.columns]
        L.append(md_table(dis[cols].round(3)) + "\n")

    L.append("## Categorical factors\n")
    L.append(f"_{SMALL_N_CAVEAT}_\n")
    for fac, t in cat_tables.items():
        L.append(f"### {fac}\n")
        L.append(md_table(t) + "\n")

    L.append("## Ranked numeric factors (Spearman rho vs mean IoU; "
             "Mann-Whitney U SUCCESS vs FAILURE)\n")
    L.append(md_table(rank) + "\n")

    L.append("## Decision-tree rules (depth 2, descriptive)\n")
    for trk, txt in tree_texts.items():
        L.append(f"**{trk.upper()}**\n\n```\n{txt}\n```\n")

    L.append("## Corruption tolerance (knees)\n")
    L.append(f"_{knee_note}_\n")
    if knees is not None and not knees.empty:
        tab = knees.groupby(["condition", "fragility"]).size().unstack(
            fill_value=0).reset_index()
        L.append(md_table(tab) + "\n")
        frag = knees[knees["fragility"] == "fragile"]
        if len(frag):
            L.append("Fragile cases: " + "; ".join(
                f"{r.sequence} ({r.condition} knee={r.knee_severity})"
                for r in frag.itertuples()) + "\n")

    L.append("## Rules: IF condition THEN expected outcome\n")
    for i, r in enumerate(rules, 1):
        L.append(f"{i}. {r}")
    L.append("")
    L.append("_Every number above is computed from the joined project data; "
             "see cases.csv / factor_ranking.csv / knees.csv for the raw "
             "values._")

    path = out_dir / "CASES.md"
    path.write_text("\n".join(L))
    print(f"  wrote {path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="case_analysis_out",
                    help="output directory (default: case_analysis_out)")
    ap.add_argument("--success-thr", type=float, default=0.5,
                    help="mean IoU >= this => SUCCESS (default 0.5)")
    ap.add_argument("--failure-thr", type=float, default=0.3,
                    help="mean IoU < this => FAILURE (default 0.3)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="sweep conf level to use for knees (default 0.25)")
    ap.add_argument("--knee-retention", type=float, default=0.5,
                    help="retention threshold defining the knee (default 0.5)")
    args = ap.parse_args()

    out_dir = ROOT / args.outdir
    out_dir.mkdir(exist_ok=True)

    cases, coverage = build_cases(args)
    cat_tables = categorical_analysis(cases)

    numeric_factors = (["num_frames"] +
                       [f for f in CAMERA_FACTORS if f in cases.columns] +
                       [f for f in COMPLEXITY_FACTORS if f in cases.columns])
    rank = numeric_analysis(cases, numeric_factors)
    tree_texts = fit_trees(cases, numeric_factors,
                           out_dir / "decision_tree.txt")
    knees, knee_note = compute_knees(args)

    # merge knee labels into cases table
    if knees is not None and not knees.empty:
        wide = knees.pivot_table(index="sequence", columns="condition",
                                 values="knee_severity", aggfunc="first")
        wide.columns = [f"knee_{c}" for c in wide.columns]
        cases = cases.merge(wide.reset_index(), on="sequence", how="left")
        lab = (knees.groupby("sequence")["label"]
               .apply(lambda s: "; ".join(s)).rename("fragility_labels"))
        cases = cases.merge(lab.reset_index(), on="sequence", how="left")

    print("\n=== Writing outputs ===")
    cases.to_csv(out_dir / "cases.csv", index=False)
    print(f"  wrote {out_dir / 'cases.csv'} ({len(cases)} rows)")
    rank.to_csv(out_dir / "factor_ranking.csv", index=False)
    print(f"  wrote {out_dir / 'factor_ranking.csv'} ({len(rank)} rows)")
    if knees is not None:
        knees.to_csv(out_dir / "knees.csv", index=False)
        print(f"  wrote {out_dir / 'knees.csv'} ({len(knees)} rows)")
    else:
        pd.DataFrame(columns=["sequence", "condition", "conf_thr",
                              "knee_severity", "knee_position", "ladder_len",
                              "min_retention", "fragility", "label"]
                     ).to_csv(out_dir / "knees.csv", index=False)
        print(f"  wrote {out_dir / 'knees.csv'} (empty — sweep unavailable)")

    plot_sorted_bars(cases, out_dir / "sequences_sorted_by_iou.png",
                     args.success_thr, args.failure_thr)
    plot_factor_boxplots(cases, rank, out_dir / "factor_boxplots.png")

    rules = build_rules(cases, rank, cat_tables, knees, tree_texts, args)
    write_cases_md(out_dir, cases, coverage, cat_tables, rank, knees,
                   knee_note, tree_texts, rules, args)

    # sanity check
    for trk in ["yolo", "csrt"]:
        vc = cases[f"tier_{trk}"].value_counts()
        assert vc.sum() == len(cases), "tier counts must sum to row count"
    print("\nDone. Sanity check passed: tier counts sum to "
          f"{len(cases)} sequences for both trackers.")


if __name__ == "__main__":
    main()
