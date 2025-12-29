#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_demo_full_addons.py
==========================
ADD-ON analyses that were commonly present in older Compare_Plot / Advanced flows
but are not in the baseline `compare_demo_full_pro.py` (v2).

This file intentionally DOES NOT re-implement the core comparisons (RSCE compare,
metrics_aggregated compare, compute_cost compare, fold-level RSCE paired tests, regression).
It only adds:
  1) ablation_summary.csv  (Demo vs Full) : numeric deltas + barh plot
  2) ablation_rank_agreement.csv : 1-row delta table
  3) paired_tests.csv : delta table (align by model/metric when possible)
  4) metrics_per_fold.csv : per-metric per-model paired tests + Holm correction + summary
  5) reliability_curve_points.csv : curve distance metrics + overlay plots (top-k shifts)

Assumptions:
- Your Code.py outputs are in a single folder `--base`.
- demo files are prefixed with `demo_`, full files with `full_`.

Run (Colab):
  !python compare_demo_full_addons.py --base /content/results --outdir /content/compare_out_addons

"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel, wilcoxon


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def holm_correction(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty_like(p)
    for i, idx in enumerate(order):
        adj[idx] = min(1.0, (m - i) * p[idx])
    for i in range(1, m):
        adj[order[i]] = max(adj[order[i]], adj[order[i - 1]])
    return adj

def bootstrap_mean_ci_fast(values: np.ndarray, n_boot: int = 3500, alpha: float = 0.05, seed: int = 7) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    n = v.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = v[idx].mean(axis=1)
    lo = np.quantile(boot, alpha/2)
    hi = np.quantile(boot, 1-alpha/2)
    return float(v.mean()), float(lo), float(hi)

def infer_model_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if c.lower() == "model":
            return c
    from pandas.api.types import is_string_dtype, is_object_dtype, is_categorical_dtype
    candidates = [c for c in df.columns if is_string_dtype(df[c]) or is_object_dtype(df[c]) or is_categorical_dtype(df[c])]
    if not candidates:
        return None
    # prefer one with many unique values
    scored = sorted([(df[c].nunique(dropna=True), c) for c in candidates], reverse=True)
    return scored[0][1]

def normalize_model_col(df: pd.DataFrame) -> pd.DataFrame:
    m = infer_model_col(df)
    if m is None or m == "model":
        return df
    return df.rename(columns={m: "model"})

def find_pair(base: Path, demo_prefix: str, full_prefix: str, suffix: str) -> Optional[Tuple[Path, Path]]:
    d = base / f"{demo_prefix}{suffix}"
    f = base / f"{full_prefix}{suffix}"
    if d.exists() and f.exists():
        return d, f
    return None


# -----------------------------
# 1) ablation_summary compare
# -----------------------------
def compare_ablation_summary(demo_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    D = demo_df.copy()
    F = full_df.copy()
    # pick key
    key = None
    for cand in ["component", "ablation", "name", "setting"]:
        if cand in D.columns and cand in F.columns:
            key = cand
            break
        if cand.capitalize() in D.columns and cand.capitalize() in F.columns:
            key = cand.capitalize()
            break
    if key is None:
        # fallback: first shared non-numeric col
        nn = [c for c in D.columns if c in F.columns and (not pd.api.types.is_numeric_dtype(D[c]))]
        key = nn[0] if nn else None
    if key is None:
        D["_idx"] = np.arange(len(D)); F["_idx"] = np.arange(len(F)); key = "_idx"

    M = F.merge(D, on=key, suffixes=("_full", "_demo"), how="inner")
    for c in F.columns:
        if c == key:
            continue
        if c in D.columns and pd.api.types.is_numeric_dtype(F[c]) and pd.api.types.is_numeric_dtype(D[c]):
            M[f"{c}_delta"] = M[f"{c}_full"] - M[f"{c}_demo"]
    return M

def plot_ablation_summary(ab_cmp: pd.DataFrame, outpath: Path) -> None:
    delta_cols = [c for c in ab_cmp.columns if c.endswith("_delta") and pd.api.types.is_numeric_dtype(ab_cmp[c])]
    if not delta_cols:
        return
    chosen = None
    for tok in ["rsce", "RSCE"]:
        for c in delta_cols:
            if tok.lower() in c.lower():
                chosen = c
                break
        if chosen:
            break
    chosen = chosen or delta_cols[0]
    # key col
    keycol = None
    for c in ab_cmp.columns:
        if c.endswith("_full") or c.endswith("_demo") or c.endswith("_delta"):
            continue
        if not pd.api.types.is_numeric_dtype(ab_cmp[c]):
            keycol = c
            break
    keycol = keycol or ab_cmp.columns[0]

    df = ab_cmp[[keycol, chosen]].copy().sort_values(chosen)
    plt.figure(figsize=(10, 0.45*len(df)+3))
    y = np.arange(len(df))
    plt.barh(y, df[chosen].astype(float).values)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.yticks(y, df[keycol].astype(str).tolist())
    plt.xlabel(chosen)
    plt.title(f"Ablation summary delta ({chosen})")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# -----------------------------
# 2) ablation_rank_agreement compare
# -----------------------------
def compare_one_row_numeric_deltas(demo_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    D = demo_df.copy()
    F = full_df.copy()
    common = [c for c in D.columns if c in F.columns]
    row = {}
    for c in common:
        if pd.api.types.is_numeric_dtype(D[c]) and pd.api.types.is_numeric_dtype(F[c]):
            row[c+"_demo"] = float(D[c].iloc[0])
            row[c+"_full"] = float(F[c].iloc[0])
            row[c+"_delta"] = float(F[c].iloc[0] - D[c].iloc[0])
        else:
            row[c] = str(F[c].iloc[0]) if len(F) else ""
    return pd.DataFrame([row])


# -----------------------------
# 3) paired_tests.csv compare
# -----------------------------
def compare_paired_tests(demo_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    D = normalize_model_col(demo_df.copy())
    F = normalize_model_col(full_df.copy())

    key = []
    if "model" in D.columns and "model" in F.columns:
        key.append("model")
    for extra in ["metric", "world", "setting", "group", "test"]:
        if extra in D.columns and extra in F.columns:
            key.append(extra)

    if not key:
        # fallback
        nn = [c for c in D.columns if c in F.columns and (not pd.api.types.is_numeric_dtype(D[c]))]
        if nn:
            key = [nn[0]]
        else:
            D["_idx"] = np.arange(len(D)); F["_idx"] = np.arange(len(F)); key = ["_idx"]

    M = F.merge(D, on=key, suffixes=("_full", "_demo"), how="inner")
    for c in F.columns:
        if c in key:
            continue
        if c in D.columns and pd.api.types.is_numeric_dtype(F[c]) and pd.api.types.is_numeric_dtype(D[c]):
            M[f"{c}_delta"] = M[f"{c}_full"] - M[f"{c}_demo"]
    return M


# -----------------------------
# 4) metrics_per_fold paired tests
# -----------------------------
def _cohen_dz(delta: np.ndarray) -> float:
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return np.nan
    sd = d.std(ddof=1)
    return float(d.mean()/sd) if sd > 0 else np.nan

def metrics_per_fold_tests(demo_df: pd.DataFrame, full_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    D = normalize_model_col(demo_df.copy())
    F = normalize_model_col(full_df.copy())
    # keys
    keys = ["model"]
    if "fold" in D.columns and "fold" in F.columns:
        keys.append("fold")
    if "world" in D.columns and "world" in F.columns:
        keys.append("world")

    metric_cols = [c for c in F.columns if c in D.columns and c not in keys
                   and pd.api.types.is_numeric_dtype(F[c]) and pd.api.types.is_numeric_dtype(D[c])]
    if not metric_cols:
        raise ValueError("No common numeric metric columns found in metrics_per_fold.")

    M = F[keys+metric_cols].merge(D[keys+metric_cols], on=keys, suffixes=("_full","_demo"), how="inner")

    rows = []
    for metric in metric_cols:
        for model in sorted(M["model"].unique()):
            mm = M[M["model"] == model]
            x_full = mm[f"{metric}_full"].astype(float).values
            x_demo = mm[f"{metric}_demo"].astype(float).values
            keep = np.isfinite(x_full) & np.isfinite(x_demo)
            x_full, x_demo = x_full[keep], x_demo[keep]
            if len(x_full) < 2:
                continue
            delta = x_full - x_demo

            t_stat, t_p = ttest_rel(x_full, x_demo, nan_policy="omit")
            try:
                w_res = wilcoxon(x_full, x_demo, zero_method="wilcox", alternative="two-sided", method="auto")
                w_p = float(w_res.pvalue)
                w_stat = float(w_res.statistic)
            except Exception:
                w_p, w_stat = np.nan, np.nan

            dz = _cohen_dz(delta)
            _, ci_lo, ci_hi = bootstrap_mean_ci_fast(delta)

            rows.append({
                "metric": metric,
                "model": model,
                "n_pairs": int(len(x_full)),
                "mean_demo": float(np.mean(x_demo)),
                "mean_full": float(np.mean(x_full)),
                "mean_delta_full_minus_demo": float(np.mean(delta)),
                "delta_ci95_lo": float(ci_lo),
                "delta_ci95_hi": float(ci_hi),
                "cohen_dz": float(dz),
                "ttest_p": float(t_p),
                "ttest_stat": float(t_stat),
                "wilcoxon_p": w_p,
                "wilcoxon_stat": w_stat,
            })

    per = pd.DataFrame(rows)
    if len(per):
        per["ttest_p_holm"] = np.nan
        per["wilcoxon_p_holm"] = np.nan
        for metric in per["metric"].unique():
            idx = per["metric"] == metric
            per.loc[idx, "ttest_p_holm"] = holm_correction(per.loc[idx, "ttest_p"].fillna(1.0).values)
            per.loc[idx, "wilcoxon_p_holm"] = holm_correction(per.loc[idx, "wilcoxon_p"].fillna(1.0).values)
        per = per.sort_values(["metric","mean_delta_full_minus_demo"], ascending=[True, False]).reset_index(drop=True)

    summary_rows = []
    if len(per):
        for metric in per["metric"].unique():
            pm = per[per["metric"] == metric]
            summary_rows.append({
                "metric": metric,
                "n_models": int(pm["model"].nunique()),
                "mean_delta_avg_over_models": float(pm["mean_delta_full_minus_demo"].mean()),
                "median_delta": float(pm["mean_delta_full_minus_demo"].median()),
                "n_models_sig_wilcoxon_holm_0p05": int(np.sum((pm["wilcoxon_p_holm"]<=0.05) & np.isfinite(pm["wilcoxon_p_holm"]))),
                "n_models_sig_ttest_holm_0p05": int(np.sum((pm["ttest_p_holm"]<=0.05) & np.isfinite(pm["ttest_p_holm"]))),
            })
    summary = pd.DataFrame(summary_rows)
    return per, summary


# -----------------------------
# 5) reliability curve overlay + distances
# -----------------------------
def _infer_curve_cols(df: pd.DataFrame) -> Dict[str,str]:
    cols = {c.lower(): c for c in df.columns}
    def pick(opts):
        for o in opts:
            if o in cols:
                return cols[o]
        return None
    x = pick(["p_mean","p","pred_mean","prob_mean","pbar"])
    y = pick(["y_mean","y","obs_mean","rate","empirical"])
    b = pick(["bin","bin_id","bucket","decile"])
    n = pick(["count","n","freq","num"])
    missing = [k for k,v in [("x",x),("y",y),("bin",b),("count",n)] if v is None]
    if missing:
        raise ValueError(f"Cannot infer curve columns (missing {missing}). Available={list(df.columns)}")
    return {"x":x,"y":y,"bin":b,"count":n}

def reliability_curve_distance(demo_df: pd.DataFrame, full_df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    D = normalize_model_col(demo_df.copy())
    F = normalize_model_col(full_df.copy())
    cd, cf = _infer_curve_cols(D), _infer_curve_cols(F)

    D2 = D.rename(columns={cd["x"]:"x", cd["y"]:"y", cd["bin"]:"bin", cd["count"]:"count"})
    F2 = F.rename(columns={cf["x"]:"x", cf["y"]:"y", cf["bin"]:"bin", cf["count"]:"count"})

    M = F2[by+["bin","x","y","count"]].merge(D2[by+["bin","x","y","count"]], on=by+["bin"], suffixes=("_full","_demo"), how="inner")
    M["abs_y_diff"] = np.abs(M["y_full"].astype(float) - M["y_demo"].astype(float))
    w = (M["count_full"].astype(float) + M["count_demo"].astype(float))/2.0
    M["w"] = w

    rows = []
    for grp, gdf in M.groupby(by):
        if not isinstance(grp, tuple):
            grp = (grp,)
        total = float(gdf["w"].sum())
        wl1 = float((gdf["w"]*gdf["abs_y_diff"]).sum()/total) if total>0 else np.nan
        maxabs = float(gdf["abs_y_diff"].max())
        row = {k:v for k,v in zip(by, grp)}
        row.update({"weighted_L1": wl1, "max_abs": maxabs, "n_bins": int(len(gdf))})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("weighted_L1", ascending=False).reset_index(drop=True)

def plot_reliability_overlays(demo_df: pd.DataFrame, full_df: pd.DataFrame, outdir: Path, by_world: bool, topk: int) -> pd.DataFrame:
    ensure_dir(outdir / "reliability_plots")
    D = normalize_model_col(demo_df.copy())
    F = normalize_model_col(full_df.copy())
    cd, cf = _infer_curve_cols(D), _infer_curve_cols(F)

    group_cols = ["model"] + (["world"] if (by_world and "world" in D.columns and "world" in F.columns) else [])
    dist = reliability_curve_distance(D, F, by=group_cols)
    dist.to_csv(outdir / "reliability_curve_distance.csv", index=False)

    # plot topk
    top = dist.head(int(topk))
    for _, row in top.iterrows():
        # filter
        dsub = D.copy()
        fsub = F.copy()
        title_parts = []
        for gc in group_cols:
            dsub = dsub[dsub[gc].astype(str) == str(row[gc])]
            fsub = fsub[fsub[gc].astype(str) == str(row[gc])]
            title_parts.append(f"{gc}={row[gc]}")
        if dsub.empty or fsub.empty:
            continue

        dsub = dsub.rename(columns={cd["x"]:"x", cd["y"]:"y", cd["bin"]:"bin", cd["count"]:"count"})
        fsub = fsub.rename(columns={cf["x"]:"x", cf["y"]:"y", cf["bin"]:"bin", cf["count"]:"count"})

        # aggregate across folds if present
        agg_keys = group_cols + ["bin"]
        def agg_curve(df):
            if "fold" in df.columns:
                g = df.groupby(agg_keys, as_index=False).apply(
                    lambda t: pd.Series({
                        "x": np.average(t["x"].astype(float), weights=t["count"].astype(float)),
                        "y": np.average(t["y"].astype(float), weights=t["count"].astype(float)),
                        "count": t["count"].astype(float).sum()
                    })
                ).reset_index(drop=True)
                return g
            g = df.groupby(agg_keys, as_index=False).apply(
                lambda t: pd.Series({
                    "x": np.average(t["x"].astype(float), weights=t["count"].astype(float)),
                    "y": np.average(t["y"].astype(float), weights=t["count"].astype(float)),
                    "count": t["count"].astype(float).sum()
                })
            ).reset_index(drop=True)
            return g

        dc = agg_curve(dsub).sort_values("x")
        fc = agg_curve(fsub).sort_values("x")

        plt.figure(figsize=(6.5,6))
        plt.plot(dc["x"].astype(float).values, dc["y"].astype(float).values, marker="o", label="Demo")
        plt.plot(fc["x"].astype(float).values, fc["y"].astype(float).values, marker="o", label="Full")
        mn = min(dc["x"].min(), fc["x"].min())
        mx = max(dc["x"].max(), fc["x"].max())
        plt.plot([mn,mx],[mn,mx], linestyle="--", linewidth=1, label="Perfect")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Empirical event rate")
        plt.title("Reliability curve\n" + ", ".join(title_parts) + f"\nweighted_L1={row['weighted_L1']:.4f}, max_abs={row['max_abs']:.4f}")
        plt.legend()
        plt.tight_layout()
        fname = "reliability_" + "_".join([str(row[c]).replace('/','_') for c in group_cols]) + ".png"
        plt.savefig(outdir / "reliability_plots" / fname, dpi=300)
        plt.close()

    return dist


# -----------------------------
# Main
# -----------------------------
def parse_args(argv: Optional[List[str]] = None):
    """Parse CLI args.

    Notes
    -----
    - In notebooks (ipykernel), `sys.argv` contains Jupyter runtime flags (e.g., -f),
      which can break argparse. If `argv` is None and we're in ipykernel, we parse
      an empty list by default.
    - `--base` is no longer required; default is current directory. If the expected
      demo_/full_ CSVs are not found under base, the script will simply skip those
      add-ons and write a note file.
    """
    if argv is None:
        try:
            import sys as _sys
            if "ipykernel" in _sys.modules:
                argv = []
        except Exception:
            argv = None

    p = argparse.ArgumentParser()
    p.add_argument("--base", default=".", type=str,
                   help="Folder containing Code.py outputs (demo_/full_ CSVs).")
    p.add_argument("--outdir", default="compare_out_addons", type=str)
    p.add_argument("--demo_prefix", default="demo_", type=str)
    p.add_argument("--full_prefix", default="full_", type=str)
    p.add_argument("--reliability_topk", default=6, type=int)
    p.add_argument("--reliability_by_world", default=0, type=int)
    return p.parse_args(args=argv)

def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    base = Path(args.base).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    ensure_dir(outdir)

    notes = []

    # --- ablation_summary ---
    pair = find_pair(base, args.demo_prefix, args.full_prefix, "ablation_summary.csv")
    if pair:
        d, f = pair
        ab = compare_ablation_summary(read_csv(d), read_csv(f))
        ab.to_csv(outdir / "ablation_summary_demo_vs_full.csv", index=False)
        plot_ablation_summary(ab, outdir / "fig_ablation_summary_deltas.png")
    else:
        notes.append("Missing demo_ablation_summary.csv / full_ablation_summary.csv")

    # --- ablation_rank_agreement ---
    pair = find_pair(base, args.demo_prefix, args.full_prefix, "ablation_rank_agreement.csv")
    if pair:
        d, f = pair
        ra = compare_one_row_numeric_deltas(read_csv(d), read_csv(f))
        ra.to_csv(outdir / "ablation_rank_agreement_demo_vs_full.csv", index=False)
    else:
        notes.append("Missing demo_ablation_rank_agreement.csv / full_ablation_rank_agreement.csv")

    # --- paired_tests.csv ---
    pair = find_pair(base, args.demo_prefix, args.full_prefix, "paired_tests.csv")
    if pair:
        d, f = pair
        pt = compare_paired_tests(read_csv(d), read_csv(f))
        pt.to_csv(outdir / "paired_tests_file_demo_vs_full.csv", index=False)
    else:
        notes.append("Missing demo_paired_tests.csv / full_paired_tests.csv")

    # --- metrics_per_fold tests ---
    pair = find_pair(base, args.demo_prefix, args.full_prefix, "metrics_per_fold.csv")
    if pair:
        d, f = pair
        per, summ = metrics_per_fold_tests(read_csv(d), read_csv(f))
        per.to_csv(outdir / "metrics_per_fold_tests_demo_vs_full.csv", index=False)
        summ.to_csv(outdir / "metrics_per_fold_summary_demo_vs_full.csv", index=False)
    else:
        notes.append("Missing demo_metrics_per_fold.csv / full_metrics_per_fold.csv")

    # --- reliability_curve_points ---
    pair = find_pair(base, args.demo_prefix, args.full_prefix, "reliability_curve_points.csv")
    if pair:
        d, f = pair
        dist = plot_reliability_overlays(read_csv(d), read_csv(f), outdir, by_world=bool(args.reliability_by_world), topk=int(args.reliability_topk))
        dist.to_csv(outdir / "reliability_curve_distance.csv", index=False)
    else:
        notes.append("Missing demo_reliability_curve_points.csv / full_reliability_curve_points.csv")

    # notes
    if notes:
        (outdir / "ADDON_NOTES.txt").write_text("\n".join(notes), encoding="utf-8")

    print("[Done] Add-on comparisons written to:", outdir)
    if notes:
        print("Some items were skipped; see ADDON_NOTES.txt")

if __name__ == "__main__":
    main()
