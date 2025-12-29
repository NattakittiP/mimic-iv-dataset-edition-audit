"""

This version addresses:
1) Robust RSCE file detection (no mistaken "full" token in suffix).
2) Correct model-column inference for object/string/category dtypes.
3) Stable sign test using scipy.stats.binomtest.
4) Fast (vectorized) bootstrap for mean CIs.
5) Regression attribution with statsmodels (SE/CI/p-values) + VIF (if available),
   and clear reporting of sample sizes.
6) Paired tests: adds effect sizes (Cohen's dz + rank-biserial) and bootstrap CI.
7) Heatmap uses diverging cmap centered at 0 (vmin/vmax symmetric).
8) Robust RSCE column selection (prefers RSCE_full; warns on ambiguity).

Outputs:
- paired_files_manifest.csv
- rsce_comparison_demo_vs_full.csv (+ plots)
- sign_test_summary.csv
- rank_agreement.csv
- regression_coefficients.csv (statsmodels summary)
- paired_tests_between_runs.csv (if per-fold components exist)
- metrics_aggregated_deltas_demo_vs_full.csv
- compute_cost_compare.csv (+ plot)
- paper_summary.md

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, kendalltau, ttest_rel, wilcoxon, binomtest

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

    rsce_demo_col = select_rsce_column(D, "demo")
    rsce_full_col = select_rsce_column(F, "full")

    # merge
    M = F.merge(D, on="model", suffixes=("_full", "_demo"), how="inner")

    # numeric overlaps: compute deltas
    numeric_common = []
    for c in D.columns:
        if c in F.columns and c != "model":
            if pd.api.types.is_numeric_dtype(D[c]) and pd.api.types.is_numeric_dtype(F[c]):
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
# Regression attribution (statsmodels)
# -----------------------------
def regression_delta_rsce(rsce_cmp: pd.DataFrame, e_ablation_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    ΔRSCE_full ~ ΔR + ΔS + ΔC (+ΔE)
    Uses statsmodels OLS if available (reports SE/CI/p-values).
    Also reports VIF for collinearity diagnostics.
    """
    df = rsce_cmp.copy()

    def find_delta(prefix: str) -> Optional[str]:
        # exact first
        for c in [f"{prefix}_delta", f"{prefix.lower()}_delta", f"{prefix}_raw_delta", f"{prefix.lower()}_raw_delta"]:
            if c in df.columns:
                return c
        # any startswith prefix and endswith _delta
        cands = [c for c in df.columns if c.lower().startswith(prefix.lower()) and c.lower().endswith("_delta")]
        return cands[0] if cands else None

    r = find_delta("R")
    s = find_delta("S")
    c = find_delta("C")
    if r is None or s is None or c is None:
        return pd.DataFrame([{"error": "Missing ΔR/ΔS/ΔC columns in rsce_scores; regression skipped."}])

    features = [r, s, c]

    # Optional E delta from e_ablation_per_fold or other E table (model-level)
    if e_ablation_df is not None:
        E = normalize_std_cols(e_ablation_df)
        # Try to compute E_mean per model then delta if both sides present, but here caller provides full-vs-demo merged already.
        # So we expect a column named E_delta (or similar) in e_ablation_df if user also paired it.
        # If not, skip.
        cand = pick_first_existing(E, ["E_delta", "E_norm_delta", "E_raw_delta"])
        if cand is not None:
            df = df.merge(E[["model", cand]].rename(columns={cand: "E_delta_used"}), on="model", how="left")
            features += ["E_delta_used"]

    y = df["RSCE_full_delta"].astype(float)
    X = df[features].astype(float)

    # report n retained per feature
    n_before = len(df)
    keep = np.isfinite(y.values)
    for col in features:
        keep &= np.isfinite(X[col].values)
    reg = df.loc[keep, ["model", "RSCE_full_delta"] + features].copy()
    n = len(reg)
    if n < 3:
        return pd.DataFrame([{"error": f"Too few complete cases for regression (n={n}, before={n_before})."}])

    if _HAS_SM:
        X_sm = sm.add_constant(reg[features].astype(float), has_constant="add")
        model = sm.OLS(reg["RSCE_full_delta"].astype(float), X_sm).fit()
        # build output table
        out = pd.DataFrame({
            "term": model.params.index,
            "coef": model.params.values,
            "std_err": model.bse.values,
            "t": model.tvalues.values,
            "p": model.pvalues.values,
            "ci_lower": model.conf_int().iloc[:, 0].values,
            "ci_upper": model.conf_int().iloc[:, 1].values,
        })
        out["R2"] = float(model.rsquared)
        out["adj_R2"] = float(model.rsquared_adj)
        out["n_models_used"] = int(n)

        # VIF
        try:
            X_vif = reg[features].astype(float).values
            vif_vals = [variance_inflation_factor(X_vif, i) for i in range(X_vif.shape[1])]
            vif_df = pd.DataFrame({"term": features, "VIF": vif_vals})
            out = out.merge(vif_df, on="term", how="left")
        except Exception:
            out["VIF"] = np.nan

        return out.sort_values("term").reset_index(drop=True)

    # Fallback (no statsmodels): coefficients + R2 only
    Xn = reg[features].astype(float).values
    Xd = np.column_stack([np.ones(n), Xn])
    beta, *_ = np.linalg.lstsq(Xd, reg["RSCE_full_delta"].astype(float).values, rcond=None)
    yhat = Xd @ beta
    ss_res = float(np.sum((reg["RSCE_full_delta"].values - yhat) ** 2))
    ss_tot = float(np.sum((reg["RSCE_full_delta"].values - reg["RSCE_full_delta"].mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rows = [{"term": "const", "coef": float(beta[0]), "R2": r2, "n_models_used": int(n)}]
    for t, b in zip(features, beta[1:]):
        rows.append({"term": t, "coef": float(b), "R2": r2, "n_models_used": int(n)})
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

def paired_tests_from_components_per_fold(demo_df: pd.DataFrame, full_df: pd.DataFrame, schema_demo: Optional[dict], schema_full: Optional[dict]) -> pd.DataFrame:
    D = normalize_std_cols(demo_df)
    F = normalize_std_cols(full_df)
    if "fold" not in D.columns or "fold" not in F.columns:
        raise ValueError("ablation_components_per_fold must contain fold.")

    # Ensure RSCE_fold exists; construct using weights from schema if available
    def ensure_rsce_fold(df: pd.DataFrame, schema: Optional[dict]) -> pd.DataFrame:
        df = df.copy()
        if "RSCE_fold" in df.columns and pd.api.types.is_numeric_dtype(df["RSCE_fold"]):
            return df
        have = set(df.columns)
        if not {"R", "S_ratio", "C_exp"}.issubset(have):
            raise ValueError("Cannot build RSCE_fold: missing R, S_ratio, C_exp.")
        E = df["E_mix"] if "E_mix" in df.columns else 0.0
        # weights from schema.json if present; else default
        wR, wS, wC, wE = 0.4, 0.3, 0.2, 0.1
        if schema:
            for k in ["wR", "wS", "wC", "wE"]:
                if k not in schema:
                    break
            else:
                wR, wS, wC, wE = float(schema["wR"]), float(schema["wS"]), float(schema["wC"]), float(schema["wE"])
        ws = wR + wS + wC + wE
        wR, wS, wC, wE = wR/ws, wS/ws, wC/ws, wE/ws
        df["RSCE_fold"] = wR*df["R"].astype(float) + wS*df["S_ratio"].astype(float) + wC*df["C_exp"].astype(float) + wE*np.asarray(E, dtype=float)
        return df

    D = ensure_rsce_fold(D, schema_demo)
    F = ensure_rsce_fold(F, schema_full)

    M = F[["model", "fold", "RSCE_fold"]].merge(D[["model", "fold", "RSCE_fold"]], on=["model", "fold"], suffixes=("_full", "_demo"), how="inner")

    rows = []
    for m in sorted(M["model"].unique()):
        mm = M[M["model"] == m].sort_values("fold")
        if len(mm) < 2:
            continue
        x_full = mm["RSCE_fold_full"].astype(float).values
        x_demo = mm["RSCE_fold_demo"].astype(float).values
        delta = x_full - x_demo

        t_stat, t_p = ttest_rel(x_full, x_demo, nan_policy="omit")
        try:
            w_res = wilcoxon(x_full, x_demo, zero_method="wilcox", alternative="two-sided", method="auto")
            w_p = float(w_res.pvalue)
            w_stat = float(w_res.statistic)
        except Exception:
            w_p, w_stat = np.nan, np.nan

        dz = _cohen_dz(delta)
        rbc = _rank_biserial_from_wilcoxon(x_full, x_demo)
        ci_lo, ci_hi = _bootstrap_ci_delta_mean(delta, n_boot=3500, alpha=0.05)

        rows.append({
            "model": m,
            "n_folds_aligned": int(len(mm)),
            "mean_demo": float(np.mean(x_demo)),
            "mean_full": float(np.mean(x_full)),
            "mean_delta_full_minus_demo": float(np.mean(delta)),
            "delta_ci95_lo": float(ci_lo),
            "delta_ci95_hi": float(ci_hi),
            "cohen_dz": float(dz),
            "rank_biserial": float(rbc),
            "ttest_p": float(t_p),
            "ttest_stat": float(t_stat),
            "wilcoxon_p": w_p,
            "wilcoxon_stat": w_stat,
        })

    out = pd.DataFrame(rows)
    if len(out):
        out["ttest_p_holm"] = holm_correction(out["ttest_p"].fillna(1.0).values)
        out["wilcoxon_p_holm"] = holm_correction(out["wilcoxon_p"].fillna(1.0).values)
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
# Paper summary
# -----------------------------
def write_paper_summary(outdir: Path, dataset_tag: str, rsce_cmp: Optional[pd.DataFrame], sign_df: Optional[pd.DataFrame],
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
        md.append("Top +ΔRSCE:\n")
        md.append(rsce_cmp[["model", "RSCE_full_delta", "rank_demo", "rank_full"]].head(10).to_markdown(index=False))
        md.append("\nTop -ΔRSCE:\n")
        md.append(rsce_cmp[["model", "RSCE_full_delta", "rank_demo", "rank_full"]].tail(10).to_markdown(index=False))
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
        md.append("## Paired tests across folds (per model)\n")
        md.append("Includes effect sizes (Cohen’s dz, rank-biserial) and bootstrap 95% CI for mean ΔRSCE_fold.\n")
        md.append(paired_df.head(20).to_markdown(index=False))
        md.append("\n_(NaN p-values occur when Wilcoxon cannot be computed due to ties/zeros; those are kept as NaN and not treated as significant.)_\n")

    if reg_df is not None:
        md.append("## Regression attribution (ΔRSCE)\n")
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

    (outdir / "paper_summary.md").write_text("\n".join(md), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare demo_ vs full_ outputs from Code.py (single-folder convention).")
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
    if not _HAS_SM:
        notes.append("statsmodels is not available; regression output will be coefficients-only without SE/CI/p-values.")

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
    comp_fold_pair = next((p for p in csv_pairs if is_components_per_fold_suffix(p.suffix)), None)
    e_fold_pair = next((p for p in csv_pairs if is_e_ablation_per_fold_suffix(p.suffix)), None)

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

    # Paired tests across folds from ablation_components_per_fold
    if comp_fold_pair is not None:
        try:
            demo_cf = read_csv(comp_fold_pair.demo_path)
            full_cf = read_csv(comp_fold_pair.full_path)
            paired_df = paired_tests_from_components_per_fold(demo_cf, full_cf, schema_demo=schema_demo, schema_full=schema_full)
            paired_df.to_csv(outdir / "paired_tests_between_runs.csv", index=False)
        except Exception as e:
            notes.append(f"Paired fold tests skipped: {e}")
    else:
        notes.append("ablation_components_per_fold pair not found; fold-level paired tests skipped.")

    # Regression attribution: if E per-fold exists, we can precompute model-level E delta and pass it
    e_delta_df = None
    if e_fold_pair is not None:
        try:
            demo_e = normalize_std_cols(read_csv(e_fold_pair.demo_path))
            full_e = normalize_std_cols(read_csv(e_fold_pair.full_path))
            # compute mean E per model
            # infer E column
            e_col_demo = pick_first_existing(demo_e, ["E", "E_norm", "E_raw", "E_mix"])
            e_col_full = pick_first_existing(full_e, ["E", "E_norm", "E_raw", "E_mix"])
            if e_col_demo and e_col_full:
                Ed = demo_e.groupby("model")[e_col_demo].mean().reset_index().rename(columns={e_col_demo: "E_demo"})
                Ef = full_e.groupby("model")[e_col_full].mean().reset_index().rename(columns={e_col_full: "E_full"})
                e_delta_df = Ef.merge(Ed, on="model", how="inner")
                e_delta_df["E_delta"] = e_delta_df["E_full"].astype(float) - e_delta_df["E_demo"].astype(float)
                e_delta_df.to_csv(outdir / "E_modellevel_demo_vs_full.csv", index=False)
            else:
                notes.append("E_ablation_per_fold found but cannot infer E column; regression will omit ΔE.")
        except Exception as e:
            notes.append(f"E_ablation processing failed: {e}")

    if rsce_cmp is not None:
        reg_df = regression_delta_rsce(rsce_cmp, e_delta_df)
        reg_df.to_csv(outdir / "regression_coefficients.csv", index=False)
    else:
        reg_df = pd.DataFrame([{"error": "RSCE comparison missing; regression skipped."}])

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
    write_paper_summary(
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
        print(" - paired_tests_between_runs.csv")
    if cost_pair is not None:
        print(" - compute_cost_compare.csv / fig_compute_cost.png")
    print(" - regression_coefficients.csv")
    print(" - paper_summary.md")

if __name__ == "__main__":
    main()
