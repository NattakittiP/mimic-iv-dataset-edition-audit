#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_addons.py
=================
ADD-ON Demo-vs-Full analyses that complement compare_pro.py.

This file intentionally DOES NOT re-implement the core comparisons (RSCE compare,
metrics_aggregated compare, compute_cost compare, fold-level RSCE paired tests, regression).
It only adds:
  1) ablation_summary.csv  (Demo vs Full) : component-mean deltas + grouped bar plot
  2) ablation_rank_agreement.csv : joined on `variant` (Full vs Demo agreement per variant)
  3) paired_tests.csv : Demo vs Full table joined on the model pair (m1, m2)
  4) metrics_per_fold.csv : per (metric, model, world) UNPAIRED Full-vs-Demo tests
     (Welch, Mann-Whitney, bootstrap CI) + Holm within metric -- descriptive
  5) reliability_curve_points.csv : fold-averaged curves, distance per (model, world),
     overlay plots of the top-k shifts

Assumptions:
- run_rsce.py outputs are in a single folder `--base` (made by make_compare_base.py).
- demo files are prefixed with `demo_`, full files with `full_`.

Run:
  python compare_addons.py --base Results/compare/hosp/_base --outdir Results/compare/hosp

"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: these scripts only ever plt.savefig()/plt.close(),
                        # never plt.show() -- forcing Agg avoids a real crash seen on Windows where the
                        # auto-selected interactive TkAgg backend hit a Tcl/Tk "wrong thread" assertion
                        # (Tcl_AsyncDelete) and killed the whole process (exit code 0x80000003) partway
                        # through a run_rsce.py SHAP-plotting pass.
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, mannwhitneyu


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
    from pandas.api.types import is_string_dtype, is_object_dtype
    candidates = [c for c in df.columns if is_string_dtype(df[c]) or is_object_dtype(df[c])
                  or isinstance(df[c].dtype, pd.CategoricalDtype)]
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
    """Grouped bars of Full - Demo change in each component mean, per model."""
    comps = [c for c in ["R_mean", "S_ratio_mean", "S_drop_mean", "C_linear_mean", "C_exp_mean", "E_mix_mean"]
             if f"{c}_delta" in ab_cmp.columns]
    keycol = "model" if "model" in ab_cmp.columns else ab_cmp.columns[0]
    if not comps:
        return
    df = ab_cmp[[keycol] + [f"{c}_delta" for c in comps]].copy()
    y = np.arange(len(df))
    h = 0.8 / len(comps)
    plt.figure(figsize=(10, 0.6 * len(df) * len(comps) / 2 + 3))
    for k, c in enumerate(comps):
        plt.barh(y + k * h, df[f"{c}_delta"].astype(float).values, height=h, label=c.replace("_mean", ""))
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.yticks(y + 0.4 - h / 2, df[keycol].astype(str).tolist())
    plt.xlabel("Full - Demo (component mean)")
    plt.title("Change in RSCE components, Full - Demo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# -----------------------------
# 2) ablation_rank_agreement compare
# -----------------------------
def compare_rank_agreement(demo_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    """Join the two ablation_rank_agreement tables on `variant` (each file is sorted
    independently, so comparing row 0 with row 0 compared different variants)."""
    if "variant" not in demo_df.columns or "variant" not in full_df.columns:
        raise ValueError("ablation_rank_agreement.csv must have a 'variant' column.")
    M = full_df.merge(demo_df, on="variant", suffixes=("_full", "_demo"), how="outer", indicator=True)
    for c in ["spearman_rank_vs_ref", "kendall_rank_vs_ref"]:
        if f"{c}_full" in M.columns and f"{c}_demo" in M.columns:
            M[f"{c}_delta"] = M[f"{c}_full"] - M[f"{c}_demo"]
    return M


# -----------------------------
# 3) paired_tests.csv compare
# -----------------------------
def compare_paired_tests(demo_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    # paired_tests.csv (run_rsce.py) has one row per model PAIR (m1, m2) and no
    # "model" column; join on the pair (the old code guessed m2 as the model
    # column and matched unrelated pairs: 91 rows instead of 21).
    D, F = demo_df.copy(), full_df.copy()
    key = ["m1", "m2"]
    for name, df in (("demo", D), ("full", F)):
        if not set(key).issubset(df.columns):
            raise ValueError(f"{name} paired_tests.csv has no m1/m2 columns.")
    M = F.merge(D, on=key, suffixes=("_full", "_demo"), how="outer", indicator=True)
    for c in F.columns:
        if c in key:
            continue
        if c in D.columns and pd.api.types.is_numeric_dtype(F[c]) and pd.api.types.is_numeric_dtype(D[c]):
            M[f"{c}_delta"] = M[f"{c}_full"] - M[f"{c}_demo"]
    return M


# -----------------------------
# 4) metrics_per_fold paired tests
# -----------------------------
METRIC_COLS = ["AUROC", "Brier", "LogLoss", "ECE", "aECE", "Brier_REL", "Brier_RES", "Brier_UNC"]


def metrics_per_fold_tests(demo_df: pd.DataFrame, full_df: pd.DataFrame, n_boot: int = 2000, seed: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full vs Demo, per (metric, model, world), on the 15 per-fold values of each
    side. UNPAIRED (Welch t, Mann-Whitney U, independent-bootstrap CI of the
    difference in means): Demo fold k and Full fold k are unrelated. Tested per
    world -- the previous version pooled 15 folds x 10 worlds into 150 "pairs",
    although the 10 worlds of a fold share one fitted model, and it also tested
    n_eval and severity as if they were metrics. Holm adjustment within each
    metric. Repeated-CV folds are correlated: treat p-values as descriptive.
    """
    D = normalize_model_col(demo_df.copy())
    F = normalize_model_col(full_df.copy())
    metrics = [m for m in METRIC_COLS if m in D.columns and m in F.columns]
    if not metrics:
        raise ValueError("No common metric columns found in metrics_per_fold.")
    rng = np.random.default_rng(seed)
    rows = []
    for metric in metrics:
        for (model, world), fsub in F.groupby(["model", "world"]):
            dsub = D[(D["model"] == model) & (D["world"] == world)]
            xf = fsub[metric].astype(float).to_numpy(); xf = xf[np.isfinite(xf)]
            xd = dsub[metric].astype(float).to_numpy(); xd = xd[np.isfinite(xd)]
            if len(xf) < 2 or len(xd) < 2:
                continue
            bd = np.array([rng.choice(xf, len(xf)).mean() - rng.choice(xd, len(xd)).mean() for _ in range(n_boot)])
            try:
                p_mwu = float(mannwhitneyu(xf, xd, alternative="two-sided").pvalue)
            except ValueError:
                p_mwu = np.nan
            p_w = ttest_ind(xf, xd, equal_var=False).pvalue
            rows.append({
                "metric": metric, "model": model, "world": world,
                "n_full": int(len(xf)), "n_demo": int(len(xd)),
                "mean_full": float(xf.mean()), "mean_demo": float(xd.mean()),
                "mean_delta_full_minus_demo": float(xf.mean() - xd.mean()),
                "delta_ci95_lo": float(np.quantile(bd, 0.025)), "delta_ci95_hi": float(np.quantile(bd, 0.975)),
                "welch_p": float(p_w) if np.isfinite(p_w) else np.nan,
                "mannwhitney_p": p_mwu,
            })
    per = pd.DataFrame(rows)
    if len(per):
        per["welch_p_holm"] = np.nan
        per["mannwhitney_p_holm"] = np.nan
        for metric in per["metric"].unique():
            idx = per["metric"] == metric
            per.loc[idx, "welch_p_holm"] = holm_correction(per.loc[idx, "welch_p"].fillna(1.0).values)
            per.loc[idx, "mannwhitney_p_holm"] = holm_correction(per.loc[idx, "mannwhitney_p"].fillna(1.0).values)
        per = per.sort_values(["metric", "model", "world"]).reset_index(drop=True)

    summary_rows = []
    for metric in (per["metric"].unique() if len(per) else []):
        pm = per[per["metric"] == metric]
        summary_rows.append({
            "metric": metric,
            "n_model_world_cells": int(len(pm)),
            "mean_delta_avg": float(pm["mean_delta_full_minus_demo"].mean()),
            "median_delta": float(pm["mean_delta_full_minus_demo"].median()),
            "n_cells_welch_holm_le_0p05": int(np.sum(pm["welch_p_holm"] <= 0.05)),
            "n_cells_mwu_holm_le_0p05": int(np.sum(pm["mannwhitney_p_holm"] <= 0.05)),
        })
    return per, pd.DataFrame(summary_rows)


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

def _agg_curve(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """Count-weighted average of each bin over folds (per keys + bin)."""
    d = df.copy()
    d["count"] = d["count"].astype(float)
    d["xw"] = d["x"].astype(float) * d["count"]
    d["yw"] = d["y"].astype(float) * d["count"]
    g = d.groupby(keys + ["bin"], as_index=False)[["xw", "yw", "count"]].sum()
    g["x"] = g["xw"] / g["count"]
    g["y"] = g["yw"] / g["count"]
    return g[keys + ["bin", "x", "y", "count"]]


def _std_curve_cols(df: pd.DataFrame) -> pd.DataFrame:
    c = _infer_curve_cols(df)
    return normalize_model_col(df.copy()).rename(columns={c["x"]: "x", c["y"]: "y", c["bin"]: "bin", c["count"]: "count"})


def reliability_curve_distance(demo_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Distance between the Demo and Full reliability curves per (model, world).
    Each side is first averaged over folds (count-weighted, per bin); the old
    code joined raw rows on model+bin only, i.e. every fold/world of Demo with
    every fold/world of Full (a run compared with itself gave L1 = 0.15, not 0).
    """
    D, F = _std_curve_cols(demo_df), _std_curve_cols(full_df)
    keys = ["model"] + (["world"] if "world" in D.columns and "world" in F.columns else [])
    Da, Fa = _agg_curve(D, keys), _agg_curve(F, keys)
    M = Fa.merge(Da, on=keys + ["bin"], suffixes=("_full", "_demo"), how="inner")
    M["abs_y_diff"] = (M["y_full"] - M["y_demo"]).abs()
    M["w"] = (M["count_full"] + M["count_demo"]) / 2.0
    rows = []
    for grp, gdf in M.groupby(keys):
        grp = grp if isinstance(grp, tuple) else (grp,)
        tot = float(gdf["w"].sum())
        row = dict(zip(keys, grp))
        row.update({"weighted_L1": float((gdf["w"] * gdf["abs_y_diff"]).sum() / tot) if tot > 0 else np.nan,
                    "max_abs": float(gdf["abs_y_diff"].max()), "n_bins_shared": int(len(gdf))})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("weighted_L1", ascending=False).reset_index(drop=True)


def plot_reliability_overlays(demo_df: pd.DataFrame, full_df: pd.DataFrame, outdir: Path, by_world: bool, topk: int) -> pd.DataFrame:
    ensure_dir(outdir / "reliability_plots")
    dist = reliability_curve_distance(demo_df, full_df)
    D, F = _std_curve_cols(demo_df), _std_curve_cols(full_df)
    keys = ["model"] + (["world"] if "world" in dist.columns else [])
    Da, Fa = _agg_curve(D, keys), _agg_curve(F, keys)
    if not by_world and "world" in dist.columns:
        per_model = dist.groupby("model", as_index=False)[["weighted_L1", "max_abs"]].mean()
        per_model.to_csv(outdir / "reliability_curve_distance_per_model_mean_over_worlds.csv", index=False)
    for _, row in dist.head(int(topk)).iterrows():
        sel_d = np.logical_and.reduce([Da[k].astype(str) == str(row[k]) for k in keys])
        sel_f = np.logical_and.reduce([Fa[k].astype(str) == str(row[k]) for k in keys])
        dc, fc = Da[sel_d].sort_values("x"), Fa[sel_f].sort_values("x")
        if dc.empty or fc.empty:
            continue
        title = ", ".join(f"{k}={row[k]}" for k in keys)
        plt.figure(figsize=(6.5, 6))
        plt.plot(dc["x"], dc["y"], marker="o", label="Demo")
        plt.plot(fc["x"], fc["y"], marker="o", label="Full")
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Empirical event rate")
        plt.title(f"Reliability curve (folds averaged)\n{title}\nweighted_L1={row['weighted_L1']:.4f}, max_abs={row['max_abs']:.4f}")
        plt.legend()
        plt.tight_layout()
        fname = "reliability_" + "_".join(str(row[k]).replace("/", "_") for k in keys) + ".png"
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
                   help="Folder containing run_rsce.py outputs (demo_/full_ CSVs).")
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
        ra = compare_rank_agreement(read_csv(d), read_csv(f))
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
