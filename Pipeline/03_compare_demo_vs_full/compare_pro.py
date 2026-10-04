#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_pro.py
==============
Single-folder demo/full comparator for run_rsce.py outputs (the folder is
assembled by make_compare_base.py).

File convention (same folder):
    demo_<suffix>.csv / .json
    full_<suffix>.csv / .json

This version addresses:
1) Robust RSCE file detection (no mistaken "full" token in suffix).
2) Correct model-column inference for object/string/category dtypes.
3) Stable sign test using scipy.stats.binomtest.
4) Fast (vectorized) bootstrap for mean CIs.
5) (revised) Exact decomposition of dRSCE into weighted component changes
   (replaces an OLS "regression attribution" that never ran and would have had
   2-3 residual degrees of freedom with 7 models).
6) (revised) Full vs Demo per-fold RSCE compared with UNPAIRED tests (Welch,
   Mann-Whitney, Hedges g, independent bootstrap) -- Demo fold k and Full fold
   k are unrelated, so paired tests were invalid.
7) Heatmap uses diverging cmap centered at 0 (vmin/vmax symmetric).
8) RSCE column must be RSCE_full on both sides (fails loudly otherwise); flag
   columns such as has_E are not differenced.
9) Missing core inputs (rsce_scores, metrics_aggregated, rsce_per_fold,
   ablation_summary, schema.json scoring block) raise an error instead of being
   skipped with a note.

Outputs:
- paired_files_manifest.csv
- rsce_comparison_demo_vs_full.csv (+ plots)
- sign_test_summary.csv
- rank_agreement.csv
- delta_rsce_decomposition.csv
- rsce_full_vs_demo_unpaired_tests.csv
- metrics_aggregated_deltas_demo_vs_full.csv
- bootstrap_CI_per_model.csv / bootstrap_CI_per_world.csv
- compute_cost_compare.csv (+ plot)
- comparison_summary.md

Run:
  python compare_pro.py --base Results/compare/hosp/_base --outdir Results/compare/hosp --dataset_tag hosp

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: these scripts only ever plt.savefig()/plt.close(),
                        # never plt.show() -- forcing Agg avoids a real crash seen on Windows where the
                        # auto-selected interactive TkAgg backend hit a Tcl/Tk "wrong thread" assertion
                        # (Tcl_AsyncDelete) and killed the whole process (exit code 0x80000003) partway
                        # through a run_rsce.py SHAP-plotting pass.
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, kendalltau, ttest_ind, mannwhitneyu, binomtest

# Optional but recommended
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    _HAS_SM = True
except Exception:
    sm = None
    variance_inflation_factor = None
    _HAS_SM = False


# -----------------------------
# Plot style (matplotlib only)
# -----------------------------
def _set_plot_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 11,
        "figure.autolayout": True,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

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

def bootstrap_mean_ci_fast(values: np.ndarray, n_boot: int = 4000, alpha: float = 0.05, seed: int = 7) -> Tuple[float, float, float]:
    """
    Vectorized bootstrap CI for the mean.
    - values: 1D array (finite values only considered)
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    n = v.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = v[idx].mean(axis=1)
    lo = np.quantile(boot_means, alpha / 2)
    hi = np.quantile(boot_means, 1 - alpha / 2)
    return float(v.mean()), float(lo), float(hi)

def safe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or len(df) == 0:
        return "_(empty)_"
    if len(df) > max_rows:
        return df.head(max_rows).to_markdown(index=False) + f"\n\n... ({len(df)-max_rows} more rows)"
    return df.to_markdown(index=False)


# -----------------------------
# Pair discovery (CSV + JSON)
# -----------------------------
@dataclass
class Pair:
    suffix: str
    demo_path: Path
    full_path: Path

def discover_pairs(base: Path, demo_prefix: str, full_prefix: str, exts: Tuple[str, ...] = (".csv",)) -> List[Pair]:
    pairs: List[Pair] = []
    for ext in exts:
        demo_files = sorted(base.glob(f"{demo_prefix}*{ext}"))
        full_files = sorted(base.glob(f"{full_prefix}*{ext}"))
        demo_map = {f.name[len(demo_prefix):]: f for f in demo_files}
        full_map = {f.name[len(full_prefix):]: f for f in full_files}
        for sfx in sorted(set(demo_map).intersection(full_map)):
            pairs.append(Pair(sfx, demo_map[sfx], full_map[sfx]))
    # de-dup by suffix (prefer csv if both)
    seen = set()
    uniq = []
    for p in pairs:
        if p.suffix in seen:
            continue
        seen.add(p.suffix)
        uniq.append(p)
    return uniq


# -----------------------------
# Heuristic column inference (FIXED)
# -----------------------------
def infer_model_col(df: pd.DataFrame) -> str:
    # prefer explicit
    for c in df.columns:
        if c.lower() in ("model", "model_name", "name", "estimator", "clf"):
            return c

    # then any string-like / categorical
    from pandas.api.types import is_string_dtype, is_object_dtype, is_categorical_dtype
    candidates = []
    for c in df.columns:
        if is_string_dtype(df[c]) or is_object_dtype(df[c]) or is_categorical_dtype(df[c]):
            candidates.append(c)
    if candidates:
        # pick the one with highest cardinality but not insane
        scored = []
        for c in candidates:
            nunq = df[c].nunique(dropna=True)
            scored.append((nunq, c))
        scored.sort(reverse=True)
        return scored[0][1]

    raise ValueError("Cannot infer model column (no obvious string/object/category columns).")

def infer_world_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if "world" in c.lower():
            return c
    return None

def infer_fold_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if c.lower() in ("fold", "cv_fold", "split", "kfold"):
            return c
    return None

def normalize_std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    m = infer_model_col(df)
    if m != "model":
        df = df.rename(columns={m: "model"})
    w = infer_world_col(df)
    if w and w != "world":
        df = df.rename(columns={w: "world"})
    f = infer_fold_col(df)
    if f and f != "fold":
        df = df.rename(columns={f: "fold"})
    return df

def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# -----------------------------
# Artifact identification (updated for your actual filenames)
# -----------------------------
def is_rsce_scores_suffix(sfx: str) -> bool:
    s = sfx.lower()
    # e.g., rsce_scores.csv
    return ("rsce" in s) and ("scores" in s)

def is_metrics_aggregated_suffix(sfx: str) -> bool:
    s = sfx.lower()
    return "metrics_aggregated" in s

def is_metrics_per_fold_suffix(sfx: str) -> bool:
    s = sfx.lower()
    return "metrics_per_fold" in s

def is_compute_cost_suffix(sfx: str) -> bool:
    s = sfx.lower()
    return "compute_cost" in s

def is_components_per_fold_suffix(sfx: str) -> bool:
    s = sfx.lower()
    return "ablation_components_per_fold" in s

def is_e_ablation_per_fold_suffix(sfx: str) -> bool:
    s = sfx.lower()
    return "e_ablation_per_fold" in s

def is_paired_tests_suffix(sfx: str) -> bool:
    s = sfx.lower()
    return "paired_tests" in s

def is_reliability_curve_suffix(sfx: str) -> bool:
    s = sfx.lower()
    return "reliability_curve_points" in s


# -----------------------------
# Robust RSCE column selection
# -----------------------------
def select_rsce_column(df: pd.DataFrame, side: str, strict: bool = False) -> str:
    """
    Prefer:
      1) RSCE_full
      2) RSCE (exact)
      3) Any column that startswith 'RSCE' and does NOT look like std/ci/fold
    """
    cols = list(df.columns)
    # exact preferred names
    for name in ["RSCE_full", "rsce_full", "RSCE", "rsce"]:
        if name in cols:
            return name

    rsce_like = [c for c in cols if "rsce" in c.lower()]
    # filter out likely non-score columns
    bad_tokens = ["std", "stderr", "se", "ci", "fold", "pval", "p_value"]
    cand = []
    for c in rsce_like:
        lc = c.lower()
        if any(t in lc for t in bad_tokens):
            continue
        cand.append(c)

    if len(cand) == 1:
        return cand[0]
    if len(cand) > 1:
        # choose the one with largest variance (most informative)
        variances = []
        for c in cand:
            if pd.api.types.is_numeric_dtype(df[c]):
                variances.append((np.nanvar(df[c].astype(float).values), c))
        variances.sort(reverse=True)
        if variances:
            return variances[0][1]

    if strict:
        raise ValueError(f"Cannot unambiguously select RSCE column for {side}. Candidates: {rsce_like}")
    # fallback: last resort numeric column
    num = [c for c in cols if c != "model" and pd.api.types.is_numeric_dtype(df[c])]
    if not num:
        raise ValueError("No numeric columns to use as RSCE.")
    return num[0]


# -----------------------------
# RSCE compare
# -----------------------------
def compare_rsce_scores(demo_df: pd.DataFrame, full_df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    D = normalize_std_cols(demo_df)
    F = normalize_std_cols(full_df)

    rsce_demo_col = select_rsce_column(D, "demo", strict=True)
    rsce_full_col = select_rsce_column(F, "full", strict=True)
    if rsce_demo_col != rsce_full_col:
        raise ValueError(f"Demo and Full rsce_scores use different score columns ({rsce_demo_col} vs {rsce_full_col}).")

    # merge
    M = F.merge(D, on="model", suffixes=("_full", "_demo"), how="inner")

    # numeric overlaps: compute deltas
    numeric_common = []
    for c in D.columns:
        if c in F.columns and c not in ("model", "has_E"):
            if (pd.api.types.is_numeric_dtype(D[c]) and pd.api.types.is_numeric_dtype(F[c])
                    and not pd.api.types.is_bool_dtype(D[c]) and not pd.api.types.is_bool_dtype(F[c])):
                numeric_common.append(c)

    for c in numeric_common:
        M[f"{c}_delta"] = M[f"{c}_full"] - M[f"{c}_demo"]

    # canonicalize chosen RSCE cols into stable names
    M = M.rename(columns={
        f"{rsce_full_col}_full": "RSCE_full_full",
        f"{rsce_demo_col}_demo": "RSCE_full_demo",
    })
    M["RSCE_full_delta"] = M["RSCE_full_full"].astype(float) - M["RSCE_full_demo"].astype(float)

    M["rank_full"] = (-M["RSCE_full_full"].astype(float)).rank(method="min")
    M["rank_demo"] = (-M["RSCE_full_demo"].astype(float)).rank(method="min")
    M["rank_change_full_minus_demo"] = M["rank_full"] - M["rank_demo"]

    return M.sort_values("RSCE_full_delta", ascending=False).reset_index(drop=True), rsce_demo_col, rsce_full_col


def rank_agreement_from_rsce(rsce_cmp: pd.DataFrame) -> pd.DataFrame:
    rho, _ = spearmanr(rsce_cmp["rank_demo"].values, rsce_cmp["rank_full"].values)
    tau, _ = kendalltau(rsce_cmp["rank_demo"].values, rsce_cmp["rank_full"].values)
    return pd.DataFrame([{"n_models": int(len(rsce_cmp)), "spearman_rank": float(rho), "kendall_rank": float(tau)}])


def sign_test_rsce_delta(rsce_cmp: pd.DataFrame) -> pd.DataFrame:
    deltas = rsce_cmp["RSCE_full_delta"].astype(float).values
    pos = int(np.sum(deltas > 0))
    neg = int(np.sum(deltas < 0))
    zero = int(np.sum(deltas == 0))
    n_eff = pos + neg
    p = float(binomtest(pos, n_eff, p=0.5, alternative="two-sided").pvalue) if n_eff > 0 else float("nan")
    return pd.DataFrame([{
        "positive": pos, "negative": neg, "zero": zero,
        "effective_n": n_eff,
        "p_value_two_sided_binomtest": p,
    }])


# -----------------------------
# Metrics aggregated compare (world-level)
# -----------------------------
def compare_metrics_aggregated(demo_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    D = normalize_std_cols(demo_df)
    F = normalize_std_cols(full_df)
    key = ["model"] + (["world"] if "world" in D.columns and "world" in F.columns else [])
    M = F.merge(D, on=key, suffixes=("_full", "_demo"), how="inner")
    for c in [c for c in F.columns if c not in key]:
        if not c.endswith("_mean"):
            continue  # skip severity (constant) and CI bounds -- deltas of those are not quantities of interest
        if c in D.columns and pd.api.types.is_numeric_dtype(F[c]) and pd.api.types.is_numeric_dtype(D[c]):
            M[f"{c}_delta"] = M[f"{c}_full"] - M[f"{c}_demo"]
    return M


def bootstrap_ci_tables(metric_cmp: pd.DataFrame, outdir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "world" not in metric_cmp.columns:
        raise ValueError("metrics_aggregated table must have 'world' to compute per-world CIs.")
    delta_cols = [c for c in metric_cmp.columns if c.endswith("_delta") and c not in ("model_delta", "world_delta")]
    delta_cols = [c for c in delta_cols if pd.api.types.is_numeric_dtype(metric_cmp[c])]
    per_model_rows = []
    per_world_rows = []
    for col in delta_cols:
        metric = col.replace("_delta", "")
        # per model
        for m in sorted(metric_cmp["model"].unique()):
            v = metric_cmp.loc[metric_cmp["model"] == m, col].values
            mean, lo, hi = bootstrap_mean_ci_fast(v, n_boot=3500, alpha=0.05)
            per_model_rows.append({"metric": metric, "model": m, "mean_delta": mean, "ci_lower": lo, "ci_upper": hi, "n_worlds": int(np.isfinite(v).sum())})
        # per world
        for w in sorted(metric_cmp["world"].unique()):
            v = metric_cmp.loc[metric_cmp["world"] == w, col].values
            mean, lo, hi = bootstrap_mean_ci_fast(v, n_boot=3500, alpha=0.05)
            per_world_rows.append({"metric": metric, "world": w, "mean_delta": mean, "ci_lower": lo, "ci_upper": hi, "n_models": int(np.isfinite(v).sum())})
    per_model = pd.DataFrame(per_model_rows)
    per_world = pd.DataFrame(per_world_rows)
    per_model.to_csv(outdir / "bootstrap_CI_per_model.csv", index=False)
    per_world.to_csv(outdir / "bootstrap_CI_per_world.csv", index=False)
    return per_model, per_world


# -----------------------------
# Exact decomposition of delta-RSCE into component contributions
# -----------------------------
# RSCE_full is a weighted sum of per-model component means, so
#   dRSCE = w_R*dR + w_S*dS + w_C*dC + w_E*dE   (exactly; weights renormalized
# over R,S,C for models without E). The previous regression of dRSCE on
# dR/dS/dC across 7 models never ran (it looked for columns rsce_scores.csv
# does not have) and would have had 2-3 residual degrees of freedom; the exact
# decomposition answers the same question without any fitting.
def decompose_delta_rsce(ab_demo: pd.DataFrame, ab_full: pd.DataFrame, scoring: dict) -> pd.DataFrame:
    W = scoring["weights_normalized"]
    Scol, Ccol = scoring["primary_S"] + "_mean", scoring["primary_C"] + "_mean"
    e_missing = scoring.get("E_missing", "renormalize")
    cols = ["model", "R_mean", Scol, Ccol] + (["E_mix_mean"] if "E_mix_mean" in ab_full.columns else [])
    M = ab_full[cols].merge(ab_demo[cols], on="model", suffixes=("_full", "_demo"), how="inner")
    rows = []
    for _, r in M.iterrows():
        has_e = ("E_mix_mean_full" in M.columns and np.isfinite(r["E_mix_mean_full"]) and np.isfinite(r["E_mix_mean_demo"]))
        if has_e or e_missing == "zero":
            wr, ws_, wc, we = W["R"], W["S"], W["C"], W["E"]
        else:
            tot = W["R"] + W["S"] + W["C"]
            wr, ws_, wc, we = W["R"] / tot, W["S"] / tot, W["C"] / tot, 0.0
        dR = r["R_mean_full"] - r["R_mean_demo"]
        dS = r[f"{Scol}_full"] - r[f"{Scol}_demo"]
        dC = r[f"{Ccol}_full"] - r[f"{Ccol}_demo"]
        dE = (r["E_mix_mean_full"] - r["E_mix_mean_demo"]) if has_e else 0.0
        rows.append({"model": r["model"], "has_E": int(has_e),
                     "contrib_R": wr * dR, "contrib_S": ws_ * dS, "contrib_C": wc * dC, "contrib_E": we * dE,
                     "dR": dR, "dS": dS, "dC": dC, "dE": dE if has_e else np.nan,
                     "sum_of_contributions": wr * dR + ws_ * dS + wc * dC + we * dE})
    return pd.DataFrame(rows)


# -----------------------------
# Paired tests across folds (effect sizes + CI)
# -----------------------------
def _cohen_dz(delta: np.ndarray) -> float:
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return np.nan
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else np.nan

def _rank_biserial_from_wilcoxon(x_full: np.ndarray, x_demo: np.ndarray) -> float:
    # rank-biserial = (W_plus - W_minus) / (W_plus + W_minus)
    d = np.asarray(x_full - x_demo, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if d.size == 0:
        return np.nan
    ranks = pd.Series(np.abs(d)).rank(method="average").values
    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    denom = (w_plus + w_minus)
    return float((w_plus - w_minus) / denom) if denom > 0 else np.nan

def _bootstrap_ci_delta_mean(deltas: np.ndarray, n_boot: int = 4000, alpha: float = 0.05, seed: int = 7) -> Tuple[float, float]:
    mean, lo, hi = bootstrap_mean_ci_fast(deltas, n_boot=n_boot, alpha=alpha, seed=seed)
    return lo, hi

def unpaired_tests_rsce_per_fold(demo_df: pd.DataFrame, full_df: pd.DataFrame, n_boot: int = 4000, seed: int = 7) -> pd.DataFrame:
    """
    Per-model Full vs Demo comparison of the per-fold RSCE scores written by
    run_rsce.py (rsce_per_fold.csv, column RSCE_fold -- same definition as
    RSCE_full). Demo and Full are different datasets with different CV splits,
    so fold k of one has nothing to do with fold k of the other: the previous
    PAIRED t-test/Wilcoxon on (model, fold) were invalid (p-values changed under
    an equally valid renumbering of folds). Unpaired Welch t and Mann-Whitney U
    are reported, with an independent-bootstrap CI of the difference in means.
    Folds of a repeated CV are correlated, so these are DESCRIPTIVE.
    """
    D = normalize_std_cols(demo_df)
    F = normalize_std_cols(full_df)
    for name, df in (("demo", D), ("full", F)):
        if "RSCE_fold" not in df.columns:
            raise ValueError(f"{name} rsce_per_fold.csv has no RSCE_fold column (re-run run_rsce.py).")
    rng = np.random.default_rng(seed)
    rows = []
    for m in sorted(set(D["model"]) & set(F["model"])):
        xf = F.loc[F["model"] == m, "RSCE_fold"].astype(float).to_numpy()
        xd = D.loc[D["model"] == m, "RSCE_fold"].astype(float).to_numpy()
        xf, xd = xf[np.isfinite(xf)], xd[np.isfinite(xd)]
        if len(xf) < 2 or len(xd) < 2:
            continue
        bd = np.array([rng.choice(xf, len(xf)).mean() - rng.choice(xd, len(xd)).mean() for _ in range(n_boot)])
        sp = np.sqrt(((len(xf) - 1) * xf.var(ddof=1) + (len(xd) - 1) * xd.var(ddof=1)) / (len(xf) + len(xd) - 2))
        g = (1 - 3 / (4 * (len(xf) + len(xd)) - 9)) * (xf.mean() - xd.mean()) / sp if sp > 0 else np.nan
        try:
            p_mwu = float(mannwhitneyu(xf, xd, alternative="two-sided").pvalue)
        except ValueError:
            p_mwu = np.nan
        rows.append({
            "model": m, "n_folds_full": int(len(xf)), "n_folds_demo": int(len(xd)),
            "mean_full": float(xf.mean()), "mean_demo": float(xd.mean()),
            "mean_delta_full_minus_demo": float(xf.mean() - xd.mean()),
            "delta_ci95_lo": float(np.quantile(bd, 0.025)), "delta_ci95_hi": float(np.quantile(bd, 0.975)),
            "hedges_g": float(g),
            "welch_p_descriptive": float(ttest_ind(xf, xd, equal_var=False).pvalue),
            "mannwhitney_p_descriptive": p_mwu,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["welch_p_holm"] = holm_correction(out["welch_p_descriptive"].fillna(1.0).values)
        out["mannwhitney_p_holm"] = holm_correction(out["mannwhitney_p_descriptive"].fillna(1.0).values)
        out = out.sort_values("mean_delta_full_minus_demo", ascending=False).reset_index(drop=True)
    return out


# -----------------------------
# Plots
# -----------------------------
def _save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_rsce_bar(rsce_cmp: pd.DataFrame, outpath: Path, top_n: Optional[int] = None) -> None:
    df = rsce_cmp.copy()
    if top_n is not None:
        df = df.sort_values("RSCE_full_full", ascending=False).head(int(top_n))
    models = df["model"].astype(str).tolist()
    x = np.arange(len(models))
    width = 0.42
    plt.figure(figsize=(10, 4.8 + 0.22*len(models)))
    plt.bar(x - width/2, df["RSCE_full_demo"].astype(float).values, width=width, label="Demo")
    plt.bar(x + width/2, df["RSCE_full_full"].astype(float).values, width=width, label="Full")
    plt.xticks(x, models, rotation=45, ha="right")
    plt.ylabel("RSCE")
    plt.title("RSCE: Demo vs Full")
    plt.legend()
    _save_fig(outpath)

def plot_rsce_delta_lollipop(rsce_cmp: pd.DataFrame, outpath: Path) -> None:
    df = rsce_cmp.sort_values("RSCE_full_delta", ascending=True).copy()
    y = np.arange(len(df))
    plt.figure(figsize=(10, 4.8 + 0.22*len(df)))
    plt.hlines(y=y, xmin=0, xmax=df["RSCE_full_delta"].astype(float).values, linewidth=2)
    plt.scatter(df["RSCE_full_delta"].astype(float).values, y, s=40)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.yticks(y, df["model"].astype(str).tolist())
    plt.xlabel("ΔRSCE (Full - Demo)")
    plt.title("Per-model ΔRSCE (Full - Demo)")
    _save_fig(outpath)

def plot_rank_scatter(rsce_cmp: pd.DataFrame, outpath: Path) -> None:
    df = rsce_cmp.copy()
    plt.figure(figsize=(6, 6))
    plt.scatter(df["rank_demo"], df["rank_full"])
    mx = max(df["rank_demo"].max(), df["rank_full"].max()) + 0.5
    plt.plot([0.5, mx], [0.5, mx], linestyle="--", linewidth=1)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.xlabel("Rank (Demo) [1=best]")
    plt.ylabel("Rank (Full) [1=best]")
    plt.title("Rank shift: Demo vs Full")
    _save_fig(outpath)

def plot_heatmap_center0(pivot: pd.DataFrame, title: str, outpath: Path, cbar_label: str) -> None:
    arr = pivot.values.astype(float)
    vmax = np.nanmax(np.abs(arr))
    vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1.0
    vmin = -vmax
    plt.figure(figsize=(1.1*pivot.shape[1] + 3, 0.45*pivot.shape[0] + 3))
    plt.imshow(arr, aspect="auto", vmin=vmin, vmax=vmax, cmap="coolwarm")
    plt.colorbar(label=cbar_label)
    plt.xticks(np.arange(pivot.shape[1]), pivot.columns.tolist(), rotation=45, ha="right")
    plt.yticks(np.arange(pivot.shape[0]), pivot.index.tolist())
    plt.title(title)
    _save_fig(outpath)

def plot_metrics_aggregated_heatmap(metric_cmp: pd.DataFrame, metric: str, outpath: Path) -> None:
    col = f"{metric}_delta"
    if col not in metric_cmp.columns:
        return
    if "world" not in metric_cmp.columns:
        return
    pivot = metric_cmp.pivot_table(index="model", columns="world", values=col, aggfunc="mean")
    pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))
    plot_heatmap_center0(pivot, f"Δ{metric} heatmap (Full - Demo)", outpath, cbar_label=f"Δ{metric}")

def plot_compute_cost_compare(demo_df: pd.DataFrame, full_df: pd.DataFrame, outpath: Path) -> None:
    D = normalize_std_cols(demo_df)
    F = normalize_std_cols(full_df)
    # find common time columns
    time_cols = [c for c in F.columns if c in D.columns and "time" in c.lower() and pd.api.types.is_numeric_dtype(F[c]) and pd.api.types.is_numeric_dtype(D[c])]
    if not time_cols:
        return
    # prefer "total" if present
    chosen = None
    for tok in ["total", "pred", "fit", "shap"]:
        for c in time_cols:
            if tok in c.lower():
                chosen = c
                break
        if chosen:
            break
    chosen = chosen or time_cols[0]
    M = F[["model", chosen]].merge(D[["model", chosen]], on="model", suffixes=("_full", "_demo"), how="inner").sort_values(f"{chosen}_full", ascending=False)
    x = np.arange(len(M))
    width = 0.42
    plt.figure(figsize=(10, 4.8 + 0.22*len(M)))
    plt.bar(x - width/2, M[f"{chosen}_demo"].astype(float).values, width=width, label="Demo")
    plt.bar(x + width/2, M[f"{chosen}_full"].astype(float).values, width=width, label="Full")
    plt.xticks(x, M["model"].astype(str).tolist(), rotation=45, ha="right")
    plt.ylabel(f"{chosen} (s)")
    plt.title(f"Compute cost: {chosen} (Demo vs Full)")
    plt.legend()
    _save_fig(outpath)


# -----------------------------
# Markdown summary
# -----------------------------
def write_summary(outdir: Path, dataset_tag: str, rsce_cmp: Optional[pd.DataFrame], sign_df: Optional[pd.DataFrame],
                        agree_df: Optional[pd.DataFrame], paired_df: Optional[pd.DataFrame], reg_df: Optional[pd.DataFrame],
                        ci_model: Optional[pd.DataFrame], ci_world: Optional[pd.DataFrame], notes: List[str]) -> None:
    md: List[str] = []
    md.append(f"# Demo vs Full comparison summary ({dataset_tag})\n")
    md.append("Generated by `compare_demo_full_pro.py`.\n")

    if notes:
        md.append("## Notes / warnings\n")
        for n in notes:
            md.append(f"- {n}")
        md.append("")

    if rsce_cmp is not None:
        md.append("## RSCE (model-level)\n")
        md.append(f"- Models compared: **{len(rsce_cmp)}**")
        md.append(f"- Mean ΔRSCE (Full - Demo): **{rsce_cmp['RSCE_full_delta'].mean():.6f}**")
        md.append(f"- Median ΔRSCE (Full - Demo): **{rsce_cmp['RSCE_full_delta'].median():.6f}**\n")
        md.append("All models, sorted by ΔRSCE:\n")
        md.append(rsce_cmp[["model", "RSCE_full_demo", "RSCE_full_full", "RSCE_full_delta", "rank_demo", "rank_full"]].to_markdown(index=False))
        md.append("")

    if agree_df is not None:
        md.append("## Rank agreement\n")
        md.append(agree_df.to_markdown(index=False))
        md.append("")

    if sign_df is not None:
        md.append("## Sign test on ΔRSCE\n")
        md.append(sign_df.to_markdown(index=False))
        md.append("")

    if paired_df is not None and len(paired_df):
        md.append("## Full vs Demo per-fold RSCE (unpaired, descriptive)\n")
        md.append("Welch t / Mann-Whitney U with Holm adjustment, Hedges g, independent-bootstrap 95% CI of the difference. "
                  "Demo and Full folds are not paired; repeated-CV folds are correlated, so p-values are descriptive.\n")
        md.append(paired_df.to_markdown(index=False))
        md.append("")

    if reg_df is not None:
        md.append("## Exact decomposition of ΔRSCE into weighted component changes\n")
        md.append(safe_to_markdown(reg_df, max_rows=40))
        md.append("")

    if ci_model is not None and len(ci_model):
        md.append("## Bootstrap CIs (per model) — preview\n")
        md.append(safe_to_markdown(ci_model, max_rows=30))
        md.append("")

    if ci_world is not None and len(ci_world):
        md.append("## Bootstrap CIs (per world) — preview\n")
        md.append(safe_to_markdown(ci_world, max_rows=30))
        md.append("")

    (outdir / "comparison_summary.md").write_text("\n".join(md), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare demo_ vs full_ outputs from run_rsce.py (single-folder convention).")
    p.add_argument("--base", type=str, required=True, help="Folder containing demo_* and full_* outputs.")
    p.add_argument("--outdir", type=str, default="compare_out", help="Output folder.")
    p.add_argument("--demo_prefix", type=str, default="demo_", help="Prefix for demo outputs.")
    p.add_argument("--full_prefix", type=str, default="full_", help="Prefix for full outputs.")
    p.add_argument("--dataset_tag", type=str, default="dataset", help="Label for summary.")
    p.add_argument("--top_n", type=int, default=None, help="Limit RSCE bar plot to top N models (by Full).")
    p.add_argument("--heatmap_metric", type=str, default="AUROC_mean", help="Metric column base in metrics_aggregated (e.g., AUROC_mean, ECE_mean, aECE_mean).")
    return p.parse_args()


def main() -> None:
    _set_plot_style()
    args = parse_args()
    base = Path(args.base).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    ensure_dir(outdir)

    notes: List[str] = []

    # Pairs: CSV + schema JSON
    csv_pairs = discover_pairs(base, args.demo_prefix, args.full_prefix, exts=(".csv",))
    json_pairs = discover_pairs(base, args.demo_prefix, args.full_prefix, exts=(".json",))

    if not csv_pairs:
        raise FileNotFoundError(f"No paired CSVs found in {base}. Expected demo_*.csv and full_*.csv.")

    manifest = pd.DataFrame([{"suffix": p.suffix, "demo_file": p.demo_path.name, "full_file": p.full_path.name} for p in csv_pairs])
    manifest.to_csv(outdir / "paired_files_manifest.csv", index=False)

    # schema
    schema_demo = schema_full = None
    schema_pair = next((p for p in json_pairs if "schema" in p.suffix.lower()), None)
    if schema_pair is not None:
        schema_demo = read_json(schema_pair.demo_path)
        schema_full = read_json(schema_pair.full_path)

    # Find required pairs by your file list
    rsce_pair = next((p for p in csv_pairs if is_rsce_scores_suffix(p.suffix)), None)
    metrics_agg_pair = next((p for p in csv_pairs if is_metrics_aggregated_suffix(p.suffix)), None)
    cost_pair = next((p for p in csv_pairs if is_compute_cost_suffix(p.suffix)), None)
    comp_fold_pair = next((p for p in csv_pairs if p.suffix == "rsce_per_fold.csv"), None)
    ab_sum_pair = next((p for p in csv_pairs if p.suffix == "ablation_summary.csv"), None)
    missing_core = [n for n, pr in (("rsce_scores.csv", rsce_pair), ("metrics_aggregated.csv", metrics_agg_pair),
                                    ("rsce_per_fold.csv", comp_fold_pair), ("ablation_summary.csv", ab_sum_pair))
                    if pr is None]
    if missing_core:
        raise FileNotFoundError(f"Missing demo_/full_ pairs in {base}: {missing_core}. Re-run make_compare_base.py "
                                f"on two complete run_rsce.py output folders (current version).")

    # Core artifacts
    rsce_cmp = sign_df = agree_df = paired_df = reg_df = None
    metric_cmp = None
    ci_model = ci_world = None

    # RSCE compare
    if rsce_pair is None:
        notes.append("rsce_scores pair not found (need demo_rsce_scores.csv and full_rsce_scores.csv).")
    else:
        demo_rsce = read_csv(rsce_pair.demo_path)
        full_rsce = read_csv(rsce_pair.full_path)
        rsce_cmp, rsce_demo_col, rsce_full_col = compare_rsce_scores(demo_rsce, full_rsce)
        rsce_cmp.to_csv(outdir / "rsce_comparison_demo_vs_full.csv", index=False)
        agree_df = rank_agreement_from_rsce(rsce_cmp)
        agree_df.to_csv(outdir / "rank_agreement.csv", index=False)
        sign_df = sign_test_rsce_delta(rsce_cmp)
        sign_df.to_csv(outdir / "sign_test_summary.csv", index=False)

        notes.append(f"RSCE column chosen: demo='{rsce_demo_col}', full='{rsce_full_col}'.")

        plot_rsce_bar(rsce_cmp, outdir / "fig_rsce_bar.png", top_n=args.top_n)
        plot_rsce_delta_lollipop(rsce_cmp, outdir / "fig_rsce_delta_lollipop.png")
        plot_rank_scatter(rsce_cmp, outdir / "fig_rank_scatter.png")

    # metrics_aggregated compare + CIs + heatmap
    if metrics_agg_pair is None:
        notes.append("metrics_aggregated pair not found (need demo_metrics_aggregated.csv and full_metrics_aggregated.csv).")
    else:
        demo_m = read_csv(metrics_agg_pair.demo_path)
        full_m = read_csv(metrics_agg_pair.full_path)
        metric_cmp = compare_metrics_aggregated(demo_m, full_m)
        metric_cmp.to_csv(outdir / "metrics_aggregated_deltas_demo_vs_full.csv", index=False)
        try:
            ci_model, ci_world = bootstrap_ci_tables(metric_cmp, outdir)
            # heatmap centered at 0
            plot_metrics_aggregated_heatmap(metric_cmp, metric=args.heatmap_metric, outpath=outdir / f"fig_metrics_aggregated_heatmap_{args.heatmap_metric}.png")
        except Exception as e:
            notes.append(f"Bootstrap/heatmap skipped: {e}")

    # Full vs Demo per-fold RSCE (unpaired)
    paired_df = unpaired_tests_rsce_per_fold(read_csv(comp_fold_pair.demo_path), read_csv(comp_fold_pair.full_path))
    paired_df.to_csv(outdir / "rsce_full_vs_demo_unpaired_tests.csv", index=False)

    # Exact decomposition of dRSCE (needs the scoring definition written by run_rsce.py into schema.json)
    scoring_d = (schema_demo or {}).get("scoring")
    scoring_f = (schema_full or {}).get("scoring")
    if not scoring_d or not scoring_f:
        raise ValueError("schema.json has no 'scoring' block -- re-run run_rsce.py (current version) for both sides.")
    if scoring_d != scoring_f:
        raise ValueError(f"Demo and Full were scored differently: {scoring_d} vs {scoring_f}")
    reg_df = decompose_delta_rsce(read_csv(ab_sum_pair.demo_path), read_csv(ab_sum_pair.full_path), scoring_f)
    chk = rsce_cmp[["model", "RSCE_full_delta"]].merge(reg_df[["model", "sum_of_contributions"]], on="model")
    max_err = float((chk["RSCE_full_delta"] - chk["sum_of_contributions"]).abs().max()) if len(chk) else np.nan
    reg_df["max_abs_check_vs_RSCE_full_delta"] = max_err
    if np.isfinite(max_err) and max_err > 1e-9:
        notes.append(f"WARNING: decomposition does not reproduce ΔRSCE (max abs error {max_err:.2e}).")
    reg_df.to_csv(outdir / "delta_rsce_decomposition.csv", index=False)

    # Compute cost compare
    if cost_pair is not None:
        try:
            demo_cost = read_csv(cost_pair.demo_path)
            full_cost = read_csv(cost_pair.full_path)
            plot_compute_cost_compare(demo_cost, full_cost, outdir / "fig_compute_cost.png")
            Dc = normalize_std_cols(demo_cost)
            Fc = normalize_std_cols(full_cost)
            Mc = Fc.merge(Dc, on="model", suffixes=("_full", "_demo"), how="inner")
            Mc.to_csv(outdir / "compute_cost_compare.csv", index=False)
        except Exception as e:
            notes.append(f"Compute cost compare skipped: {e}")
    else:
        notes.append("compute_cost pair not found; compute-cost comparison skipped.")

    # Write summary
    write_summary(
        outdir=outdir,
        dataset_tag=args.dataset_tag,
        rsce_cmp=rsce_cmp,
        sign_df=sign_df,
        agree_df=agree_df,
        paired_df=paired_df,
        reg_df=reg_df,
        ci_model=ci_model,
        ci_world=ci_world,
        notes=notes,
    )

    print("\n[Done] Comparison artifacts written to:", outdir)
    print("Key outputs:")
    print(" - paired_files_manifest.csv")
    if rsce_cmp is not None:
        print(" - rsce_comparison_demo_vs_full.csv")
        print(" - rank_agreement.csv")
        print(" - sign_test_summary.csv")
        print(" - fig_rsce_bar.png / fig_rsce_delta_lollipop.png / fig_rank_scatter.png")
    if metric_cmp is not None:
        print(" - metrics_aggregated_deltas_demo_vs_full.csv")
        print(" - bootstrap_CI_per_model.csv / bootstrap_CI_per_world.csv (if created)")
    if paired_df is not None:
        print(" - rsce_full_vs_demo_unpaired_tests.csv")
    if cost_pair is not None:
        print(" - compute_cost_compare.csv / fig_compute_cost.png")
    print(" - delta_rsce_decomposition.csv")
    print(" - comparison_summary.md")

if __name__ == "__main__":
    main()
