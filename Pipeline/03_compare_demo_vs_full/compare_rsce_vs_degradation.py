#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_rsce_vs_degradation.py
==============================
Question: does the RSCE score actually predict how much a model degrades under
realistic perturbations? For each model, this computes "real-world degradation"
features (mean/worst-case AUROC drop, and LogLoss/Brier/ECE drift, relative to
the clean WA_clean world) and correlates them against RSCE_full (Spearman +
Kendall, with bootstrap CIs and permutation p-values), producing:

  - RSCE_vs_realworld_degradation_correlation.csv
  - scatter_RSCE_vs_<target>.png  (one per degradation target)
  - worst_worlds_by_metric.csv    (which world is the worst case per model/metric)

Schema note: run_rsce.py writes `metrics_aggregated.csv` (columns like
AUROC_mean, ECE_mean, ... per model/world) and `rsce_scores.csv` (model,
RSCE_full). This script reads those files directly.

Usage:
  python compare_rsce_vs_degradation.py \
      --rsce_scores results/hosp_full/rsce_scores.csv \
      --metrics_aggregated results/hosp_full/metrics_aggregated.csv \
      --outdir results/hosp_full/degradation_analysis \
      --dataset_tag hosp_full
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: these scripts only ever plt.savefig()/plt.close(),
                        # never plt.show() -- forcing Agg avoids a real crash seen on Windows where the
                        # auto-selected interactive TkAgg backend hit a Tcl/Tk "wrong thread" assertion
                        # (Tcl_AsyncDelete) and killed the whole process (exit code 0x80000003) partway
                        # through a run_rsce.py SHAP-plotting pass.
import matplotlib.pyplot as plt


def log(msg: str) -> None:
    print(f"[compare_rsce_vs_degradation] {msg}")


def choose_baseline_world(world_series: pd.Series) -> str:
    ws = world_series.astype(str).unique().tolist()
    for pref in ["WA_clean", "WA_real", "WA", "WorldA", "worldA"]:
        if pref in ws:
            return pref
    for w in ws:
        if "WA" in w:
            return w
    return world_series.astype(str).value_counts().index[0]


def compute_degradation_table(df: pd.DataFrame, base_world: str, metric_cols: dict) -> pd.DataFrame:
    """metric_cols: {"AUROC": colname, "LogLoss": colname, "Brier": colname, "ECE": colname} (values may be None)"""
    out_rows = []
    models = sorted(df["model"].unique())
    worlds = sorted(df["world"].astype(str).unique())
    non_base_worlds = [w for w in worlds if w != base_world]
    if len(non_base_worlds) == 0:
        raise ValueError("Only one world present; cannot compute degradation.")

    def _agg(series: pd.Series) -> Tuple[float, float]:
        vals = series.dropna().values
        if len(vals) == 0:
            return np.nan, np.nan
        return float(np.mean(vals)), float(np.max(vals))

    for m in models:
        sub = df[df["model"] == m].set_index("world")
        if base_world not in sub.index:
            continue
        row = {"model": m}

        auroc_c, logloss_c, brier_c, ece_c = (metric_cols.get(k) for k in ["AUROC", "LogLoss", "Brier", "ECE"])

        if auroc_c is not None:
            row["AUROC_base"] = sub.loc[base_world, auroc_c]
            drops = pd.Series([sub.loc[base_world, auroc_c] - sub.loc[w, auroc_c] for w in non_base_worlds if w in sub.index], dtype=float)
            row["AUROC_drop_mean"], row["AUROC_drop_worst"] = _agg(drops)

        if logloss_c is not None:
            row["LogLoss_base"] = sub.loc[base_world, logloss_c]
            drifts = pd.Series([sub.loc[w, logloss_c] - sub.loc[base_world, logloss_c] for w in non_base_worlds if w in sub.index], dtype=float)
            row["LogLoss_drift_mean"], row["LogLoss_drift_worst"] = _agg(drifts)

        if brier_c is not None:
            row["Brier_base"] = sub.loc[base_world, brier_c]
            drifts = pd.Series([sub.loc[w, brier_c] - sub.loc[base_world, brier_c] for w in non_base_worlds if w in sub.index], dtype=float)
            row["Brier_drift_mean"], row["Brier_drift_worst"] = _agg(drifts)

        if ece_c is not None:
            row["ECE_base"] = sub.loc[base_world, ece_c]
            drifts = pd.Series([sub.loc[w, ece_c] - sub.loc[base_world, ece_c] for w in non_base_worlds if w in sub.index], dtype=float)
            row["ECE_drift_mean"], row["ECE_drift_worst"] = _agg(drifts)

        out_rows.append(row)
    return pd.DataFrame(out_rows)


def bootstrap_corr(x: np.ndarray, y: np.ndarray, corr_fn: Callable, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4:
        return np.nan, (np.nan, np.nan)
    r0 = corr_fn(x, y)
    boots = np.array([corr_fn(x[idx], y[idx]) for idx in (rng.integers(0, n, size=n) for _ in range(n_boot))], dtype=float)
    finite = boots[np.isfinite(boots)]
    if len(finite) == 0:
        return float(r0), (np.nan, np.nan)
    lo, hi = np.quantile(finite, [0.025, 0.975])
    return float(r0), (float(lo), float(hi))


def perm_test_corr(x: np.ndarray, y: np.ndarray, corr_fn: Callable, n_perm: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4:
        return np.nan
    r0 = corr_fn(x, y)
    cnt = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        rp = corr_fn(x, yp)
        if np.isfinite(rp) and abs(rp) >= abs(r0):
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def spearman_only(x, y):
    r, _ = spearmanr(x, y)
    return r


def kendall_only(x, y):
    r, _ = kendalltau(x, y)
    return r


def worst_worlds_by_metric(df: pd.DataFrame, base_world: str, metric_cols: dict) -> pd.DataFrame:
    rows = []
    for m in sorted(df["model"].unique()):
        sub = df[df["model"] == m].set_index("world")
        if base_world not in sub.index:
            continue
        others = [w for w in sub.index.astype(str).tolist() if w != base_world]
        if not others:
            continue
        row = {"model": m}
        auroc_c, logloss_c, brier_c, ece_c = (metric_cols.get(k) for k in ["AUROC", "LogLoss", "Brier", "ECE"])

        if auroc_c is not None:
            drops = {w: float(sub.loc[base_world, auroc_c] - sub.loc[w, auroc_c]) for w in others if pd.notna(sub.loc[w, auroc_c])}
            if drops:
                ww = max(drops, key=drops.get)
                row["worst_world_AUROC_drop"], row["AUROC_drop_worst"] = ww, drops[ww]

        for name, col in [("ECE", ece_c), ("LogLoss", logloss_c), ("Brier", brier_c)]:
            if col is not None:
                drifts = {w: float(sub.loc[w, col] - sub.loc[base_world, col]) for w in others if pd.notna(sub.loc[w, col])}
                if drifts:
                    ww = max(drifts, key=drifts.get)
                    row[f"worst_world_{name}_drift"], row[f"{name}_drift_worst"] = ww, drifts[ww]
        rows.append(row)
    return pd.DataFrame(rows)


def scatter_with_labels(df: pd.DataFrame, xcol: str, ycol: str, title: str, fname: Path) -> None:
    d = df[[xcol, ycol, "model"]].dropna()
    plt.figure(figsize=(7, 5))
    plt.scatter(d[xcol], d[ycol])
    for _, r in d.iterrows():
        plt.annotate(r["model"], (r[xcol], r[ycol]), fontsize=9, xytext=(4, 4), textcoords="offset points")
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correlate RSCE against real-world degradation (AUROC drop / calibration drift) across worlds.")
    p.add_argument("--rsce_scores", type=str, required=True, help="Path to rsce_scores.csv (from run_rsce.py).")
    p.add_argument("--metrics_aggregated", type=str, required=True, help="Path to metrics_aggregated.csv (from run_rsce.py).")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--dataset_tag", type=str, default="dataset")
    p.add_argument("--n_boot", type=int, default=5000)
    p.add_argument("--n_perm", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rsce = pd.read_csv(args.rsce_scores)
    world = pd.read_csv(args.metrics_aggregated)

    rsce_col = next((c for c in ["RSCE_full", "RSCE", "rsce_full", "rsce"] if c in rsce.columns), None)
    if rsce_col is None:
        raise ValueError(f"Cannot find an RSCE column in {args.rsce_scores}; columns={list(rsce.columns)}")
    rsce = rsce.rename(columns={rsce_col: "RSCE"})
    rsce["RSCE"] = pd.to_numeric(rsce["RSCE"], errors="coerce")
    rsce = rsce.dropna(subset=["RSCE"])

    metric_cols = {}
    for key, candidates in [
        ("AUROC", ["AUROC_mean", "auroc_mean", "AUROC"]),
        ("LogLoss", ["LogLoss_mean", "logloss_mean", "LogLoss"]),
        ("Brier", ["Brier_mean", "brier_mean", "Brier"]),
        ("ECE", ["ECE_mean", "ece_mean", "ECE"]),
    ]:
        metric_cols[key] = next((c for c in candidates if c in world.columns), None)
    log(f"Detected metric columns: {metric_cols}")

    base_world = choose_baseline_world(world["world"])
    log(f"Baseline world: {base_world}")

    deg = compute_degradation_table(world, base_world, metric_cols)
    merged = deg.merge(rsce[["model", "RSCE"]], on="model", how="inner")

    targets = [c for c in [
        "AUROC_drop_mean", "AUROC_drop_worst",
        "LogLoss_drift_mean", "LogLoss_drift_worst",
        "Brier_drift_mean", "Brier_drift_worst",
        "ECE_drift_mean", "ECE_drift_worst",
    ] if c in merged.columns]

    results = []
    for t in targets:
        x = merged["RSCE"].values
        y = merged[t].values
        sp, sp_ci = bootstrap_corr(x, y, spearman_only, n_boot=args.n_boot, seed=args.seed)
        sp_p = perm_test_corr(x, y, spearman_only, n_perm=args.n_perm, seed=args.seed + 1)
        kd, kd_ci = bootstrap_corr(x, y, kendall_only, n_boot=args.n_boot, seed=args.seed)
        kd_p = perm_test_corr(x, y, kendall_only, n_perm=args.n_perm, seed=args.seed + 2)
        results.append({
            "target": t, "spearman_r": sp,
            "spearman_ci95_lo": float(sp_ci[0]) if sp_ci is not None else np.nan,
            "spearman_ci95_hi": float(sp_ci[1]) if sp_ci is not None else np.nan,
            "spearman_perm_p": sp_p,
            "kendall_tau": kd,
            "kendall_ci95_lo": float(kd_ci[0]) if kd_ci is not None else np.nan,
            "kendall_ci95_hi": float(kd_ci[1]) if kd_ci is not None else np.nan,
            "kendall_perm_p": kd_p,
            "n_models": int(np.sum(np.isfinite(x) & np.isfinite(y))),
        })
        scatter_with_labels(
            merged, xcol="RSCE", ycol=t,
            title=f"RSCE vs {t} (baseline={base_world}, {args.dataset_tag})",
            fname=outdir / f"scatter_RSCE_vs_{t}.png",
        )

    corr_df = pd.DataFrame(results).sort_values(by="spearman_r")
    # Holm across the targets tested (8 targets x 2 statistics are not independent
    # tests of 8 hypotheses, so adjust within each statistic).
    for col in ("spearman_perm_p", "kendall_perm_p"):
        pv = corr_df[col].fillna(1.0).to_numpy(dtype=float)
        order = np.argsort(pv); m = len(pv); adj = np.empty(m); run = 0.0
        for rank, i in enumerate(order):
            run = max(run, min(1.0, (m - rank) * pv[i])); adj[i] = run
        corr_df[col + "_holm"] = adj
    # CAVEATS for interpretation (also printed): with 7 models Spearman has little
    # power (rho=-0.5 detected ~16% of the time); RSCE's S and C components are
    # computed from the same per-world AUROC/ECE values the targets summarise, so
    # a correlation is partly by construction; bootstrap CIs over 7 models are wide.
    log("NOTE: 7 models -> low power; S/C are built from the same AUROC/ECE values (partly circular). Descriptive only.")
    corr_df.to_csv(outdir / "RSCE_vs_realworld_degradation_correlation.csv", index=False)
    log(f"Saved: {outdir / 'RSCE_vs_realworld_degradation_correlation.csv'}")

    worst_tbl = worst_worlds_by_metric(world, base_world, metric_cols)
    worst_tbl = worst_tbl.merge(rsce[["model", "RSCE"]], on="model", how="left").sort_values("RSCE", ascending=False)
    worst_tbl.to_csv(outdir / "worst_worlds_by_metric.csv", index=False)
    log(f"Saved: {outdir / 'worst_worlds_by_metric.csv'}")

    log("Done.")


if __name__ == "__main__":
    main()
