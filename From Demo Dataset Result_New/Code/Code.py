"""
This version implements:
1) A full RSCE benchmarking pipeline with multi-world perturbations, repeated stratified CV,
   calibration-aware evaluation, and comprehensive statistical testing.
2) Deterministic experiment control via global seeding, hashed world-specific RNGs,
   and full environment / package version logging.
3) Robust preprocessing with numeric and categorical pipelines
   (median / most-frequent imputation, scaling, one-hot encoding, dense fallback when required).
4) A diverse model zoo (linear, tree-based, kernel, neural, probabilistic),
   optionally wrapped with probability calibration (sigmoid / isotonic).
5) A rich set of realism-inspired “worlds” simulating noise, missingness, distribution shift,
   surrogate corruption, nonlinearity, subgroup shift, prevalence shift, concept drift,
   and label noise, each with explicit severity levels.
6) Extensive evaluation metrics per fold, model, and world:
   AUROC, Brier, LogLoss, ECE, adaptive ECE, and full Brier decomposition (REL / RES / UNC).
7) Bootstrap confidence intervals for aggregated metrics at the world level.
8) RSCE component ablations:
   - R: clean-world discrimination (AUROC).
   - S: robustness to covariate shifts (drop- and ratio-based formulations).
   - C: calibration stability across worlds (exponential and linear forms).
   - E: explainability stability using SHAP (cosine, rank, Jaccard, and mixed scores).
9) SHAP-based explainability analysis:
   - Cross-world feature attribution stability.
   - Summary, bar, and dependence plots (saved for selected folds/models).
10) Aggregation of RSCE scores across models with configurable weights,
    plus sensitivity analysis over weight choices and perturbation severity.
11) Ranking stability analysis across alternative RSCE formulations
    using Spearman and Kendall rank correlations.
12) Paired statistical testing between models on per-fold RSCE
    (Wilcoxon signed-rank and paired t-test with Holm correction).
13) Reliability diagram aggregation and visualization across folds and worlds.
14) Detailed compute-cost accounting (fit, prediction, SHAP, total time) with aggregation.
15) Progress tracking with nested tqdm bars and ETA for long-running experiments.

Outputs:
- schema.json
- metrics_per_fold.csv
- metrics_aggregated.csv
- ablation_components_per_fold.csv
- ablation_summary.csv
- ablation_rank_agreement.csv
- rsce_scores.csv
- paired_tests.csv
- compute_cost_per_fold.csv
- compute_cost.csv
- reliability_curve_points.csv
- reliability_diagram.png
- rsce_bar.png
- sensitivity_weights.png
- sensitivity_severity.png
- shap_plots
"""
from __future__ import annotations

import argparse
import json
import random
import hashlib
import time
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, kendalltau, wilcoxon, ttest_rel

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.calibration import CalibratedClassifierCV

# ✅ Progress bar (tqdm)
from tqdm.auto import tqdm

# Optional: SHAP for explainability stability E
try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Version metadata
try:
    import importlib.metadata as importlib_metadata
except Exception:
    import importlib_metadata  # type: ignore


# -----------------------------
# Utilities
# -----------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_dense_if_sparse(X):
    try:
        import scipy.sparse as sp  # type: ignore
        if sp.issparse(X):
            return X.toarray()
    except Exception:
        pass
    return X


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(np.mean(values[idx]))
    alpha = (1 - ci) / 2
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1 - alpha))
    return lo, hi


def deterministic_int_hash(s: str, mod: int = 2**31 - 1) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def holm_correction(pvals: pd.Series) -> pd.Series:
    """Holm step-down adjustment."""
    p = pvals.values.astype(float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    for i, idx in enumerate(order):
        adj[idx] = min(1.0, (m - i) * p[idx])
    # enforce monotonicity
    for i in range(1, m):
        adj[order[i]] = max(adj[order[i]], adj[order[i - 1]])
    return pd.Series(adj, index=pvals.index)


def get_versions(pkgs: List[str]) -> Dict[str, str]:
    out = {}
    for p in pkgs:
        try:
            out[p] = importlib_metadata.version(p)
        except Exception:
            out[p] = "NA"
    return out


# -----------------------------
# Calibration metrics
# -----------------------------
def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob = np.clip(y_prob, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    N = len(y_prob)
    for i in range(n_bins):
        mask = bin_ids == i
        if not np.any(mask):
            continue
        w = np.mean(mask)
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += w * abs(acc - conf)
    return float(ece)


def adaptive_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Quantile-binned ECE (equal-mass bins)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob = np.clip(y_prob, 0.0, 1.0)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.quantile(y_prob, qs)
    bins[0] = 0.0
    bins[-1] = 1.0
    bin_ids = np.digitize(y_prob, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    N = len(y_prob)
    for i in range(n_bins):
        mask = bin_ids == i
        if not np.any(mask):
            continue
        w = np.sum(mask) / N
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += w * abs(acc - conf)
    return float(ece)


def reliability_curve_points(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15, adaptive: bool = False) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob = np.clip(y_prob, 0.0, 1.0)

    if not adaptive:
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(y_prob, bins) - 1
    else:
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        bins = np.quantile(y_prob, qs)
        bins[0], bins[-1] = 0.0, 1.0
        bin_ids = np.digitize(y_prob, bins, right=True) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    rows = []
    for i in range(n_bins):
        mask = bin_ids == i
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": i,
                "p_mean": float(np.mean(y_prob[mask])),
                "y_mean": float(np.mean(y_true[mask])),
                "count": int(np.sum(mask)),
            }
        )
    return pd.DataFrame(rows)


def brier_decomposition(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15, adaptive: bool = True) -> Dict[str, float]:
    """
    Murphy (1973) decomposition:
      BS = REL - RES + UNC
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob = np.clip(y_prob, 0.0, 1.0)
    p = float(np.mean(y_true))
    UNC = p * (1.0 - p)

    curve = reliability_curve_points(y_true, y_prob, n_bins=n_bins, adaptive=adaptive)
    if len(curve) == 0:
        return {"REL": np.nan, "RES": np.nan, "UNC": UNC}

    N = float(len(y_true))
    RES = 0.0
    REL = 0.0
    for _, r in curve.iterrows():
        nk = float(r["count"])
        ok = float(r["y_mean"])
        fk = float(r["p_mean"])
        RES += (nk / N) * (ok - p) ** 2
        REL += (nk / N) * (ok - fk) ** 2

    return {"REL": float(REL), "RES": float(RES), "UNC": float(UNC)}


# -----------------------------
# Worlds (perturbations)
# -----------------------------
@dataclass(frozen=True)
class WorldSpec:
    name: str
    kind: str
    params: Dict[str, Any]
    severity: int


def add_gaussian_noise(X_df: pd.DataFrame, num_cols: List[str], std_fraction: float, rng: np.random.Generator) -> pd.DataFrame:
    Xn = X_df.copy()
    for col in num_cols:
        std = float(np.nanstd(X_df[col].values))
        scale = std_fraction * (std if std > 0 else 1.0)
        Xn[col] = X_df[col].astype(float).values + rng.normal(0.0, scale, size=len(X_df))
    return Xn


def add_outliers(X_df: pd.DataFrame, num_cols: List[str], frac: float, rng: np.random.Generator) -> pd.DataFrame:
    Xo = X_df.copy()
    n = len(Xo)
    m = max(1, int(frac * n))
    for col in num_cols:
        idx = rng.choice(n, size=m, replace=False)
        std = float(np.nanstd(Xo[col].values))
        if std == 0 or not np.isfinite(std):
            std = 1.0
        noise = rng.standard_t(df=3, size=len(idx)) * (3 * std)
        arr = Xo[col].astype(float).values
        arr[idx] = arr[idx] + noise
        Xo[col] = arr
    return Xo


def induce_missingness_mcar_mar(X_df: pd.DataFrame, num_cols: List[str], base_p: float, extra_p: float, rng: np.random.Generator) -> pd.DataFrame:
    Xm = X_df.copy()
    if len(num_cols) == 0:
        return Xm
    mat = Xm[num_cols].astype(float).values
    mat[rng.random(mat.shape) < base_p] = np.nan
    Xm[num_cols] = mat
    anchor = num_cols[0]
    anchor_vals = Xm[anchor].astype(float).values
    q75 = np.nanquantile(anchor_vals, 0.75) if np.isfinite(anchor_vals).any() else np.nan
    if np.isfinite(q75):
        high = anchor_vals >= q75
        for col in num_cols:
            m2 = (rng.random(len(Xm)) < extra_p) & high
            tmp = Xm[col].astype(float).values
            tmp[m2] = np.nan
            Xm[col] = tmp
    return Xm


def distribution_shift_additive(X_df: pd.DataFrame, num_cols: List[str], shift_scale: float) -> pd.DataFrame:
    Xs = X_df.copy()
    for col in num_cols:
        std = float(np.nanstd(Xs[col].astype(float).values))
        if std == 0 or not np.isfinite(std):
            std = 1.0
        Xs[col] = Xs[col].astype(float).values + shift_scale * std
    return Xs


def corrupt_surrogates(X_df: pd.DataFrame, num_cols: List[str], gamma: float, rng: np.random.Generator, k: int = 3) -> pd.DataFrame:
    Xc = X_df.copy()
    if len(num_cols) == 0:
        return Xc
    kk = min(k, len(num_cols))
    sur_cols = rng.choice(num_cols, size=kk, replace=False)
    for col in sur_cols:
        sur = Xc[col].astype(float).values
        mu = float(np.nanmean(sur)) if np.isfinite(sur).any() else 0.0
        sd = float(np.nanstd(sur)) if np.isfinite(sur).any() else 1.0
        delta_sys = 0.1 * mu
        noise = rng.normal(0.0, 0.2 * (sd if sd > 0 else 1.0), size=len(sur))
        Xc[col] = (1 - gamma) * sur + gamma * (sur + delta_sys + noise)
    return Xc


def nonlinear_distortion(X_df: pd.DataFrame, num_cols: List[str], alpha: float) -> pd.DataFrame:
    Xn = X_df.copy()
    for col in num_cols:
        x = Xn[col].astype(float).values
        sd = float(np.nanstd(x))
        if not np.isfinite(sd) or sd == 0:
            continue
        if np.nanmin(x) >= 0:
            g = np.log1p(np.maximum(x, 0.0))
        else:
            g = (x ** 2) / (1.0 + np.abs(x))
        Xn[col] = (1 - alpha) * x + alpha * g
    return Xn


def flip_labels(y: np.ndarray, eta: float, rng: np.random.Generator) -> np.ndarray:
    y2 = y.copy()
    mask = rng.random(len(y2)) < eta
    y2[mask] = 1 - y2[mask]
    return y2


def subgroup_shift(X_df: pd.DataFrame, num_cols: List[str], mask: np.ndarray, shift_scale: float) -> pd.DataFrame:
    Xs = X_df.copy()
    if len(num_cols) == 0:
        return Xs
    for col in num_cols:
        std = float(np.nanstd(Xs[col].astype(float).values))
        if std == 0 or not np.isfinite(std):
            std = 1.0
        arr = Xs[col].astype(float).values
        arr[mask] = arr[mask] + shift_scale * std
        Xs[col] = arr
    return Xs


def prevalence_shift_resample(X_df: pd.DataFrame, y: np.ndarray, target_prev: float, rng: np.random.Generator) -> Tuple[pd.DataFrame, np.ndarray]:
    y = np.asarray(y, dtype=int)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X_df.copy(), y.copy()

    n = len(y)
    n_pos = int(round(target_prev * n))
    n_neg = n - n_pos

    pos_s = rng.choice(pos_idx, size=n_pos, replace=(n_pos > len(pos_idx)))
    neg_s = rng.choice(neg_idx, size=n_neg, replace=(n_neg > len(neg_idx)))

    idx = np.concatenate([pos_s, neg_s])
    rng.shuffle(idx)
    return X_df.iloc[idx].reset_index(drop=True), y[idx].copy()


def concept_drift_label_conditional(
    X_df: pd.DataFrame,
    y: np.ndarray,
    num_cols: List[str],
    k: int,
    shift_scale: float,
    rng: np.random.Generator
) -> pd.DataFrame:
    Xc = X_df.copy()
    if len(num_cols) == 0:
        return Xc
    kk = min(k, len(num_cols))
    cols = rng.choice(num_cols, size=kk, replace=False)
    pos = (np.asarray(y, dtype=int) == 1)
    for col in cols:
        std = float(np.nanstd(Xc[col].astype(float).values))
        if std == 0 or not np.isfinite(std):
            std = 1.0
        arr = Xc[col].astype(float).values
        arr[pos] = arr[pos] + shift_scale * std
        Xc[col] = arr
    return Xc


def build_world_specs() -> List[WorldSpec]:
    return [
        WorldSpec("WA_clean", "clean", {}, 0),
        WorldSpec("WB_noise_outliers", "noise_outliers", {"std_fraction": 0.20, "outlier_frac": 0.02}, 1),
        WorldSpec("WC_missingness", "missingness", {"base_p": 0.10, "extra_p": 0.15}, 2),
        WorldSpec("WD_shift", "shift", {"shift_scale": 0.30}, 3),
        WorldSpec("WE_surrogate_corrupt", "surrogate", {"gamma": 0.50}, 4),
        WorldSpec("WF_nonlinear", "nonlinear", {"alpha": 0.60}, 5),
        WorldSpec("WH_subgroup_shift", "subgroup_shift", {"shift_scale": 0.35}, 6),
        WorldSpec("WI_prevalence_shift", "prevalence_shift", {"target_prev": 0.35}, 7),
        WorldSpec("WJ_concept_drift", "concept_drift", {"k": 3, "shift_scale": 0.35}, 8),
        WorldSpec("WG_label_noise", "label_noise", {"eta": 0.10}, 9),
    ]


def apply_world(spec: WorldSpec, X: pd.DataFrame, y: np.ndarray, num_cols: List[str], cat_cols: List[str], seed: int) -> Tuple[pd.DataFrame, np.ndarray]:
    h = deterministic_int_hash(spec.name)
    rng = np.random.default_rng(int(seed + 1000 * spec.severity + h))

    if spec.kind == "clean":
        return X.copy(), y.copy()
    if spec.kind == "noise_outliers":
        Xw = add_gaussian_noise(X, num_cols, float(spec.params["std_fraction"]), rng)
        Xw = add_outliers(Xw, num_cols, float(spec.params["outlier_frac"]), rng)
        return Xw, y.copy()
    if spec.kind == "missingness":
        return induce_missingness_mcar_mar(X, num_cols, float(spec.params["base_p"]), float(spec.params["extra_p"]), rng), y.copy()
    if spec.kind == "shift":
        return distribution_shift_additive(X, num_cols, float(spec.params["shift_scale"])), y.copy()
    if spec.kind == "surrogate":
        return corrupt_surrogates(X, num_cols, float(spec.params["gamma"]), rng), y.copy()
    if spec.kind == "nonlinear":
        return nonlinear_distortion(X, num_cols, float(spec.params["alpha"])), y.copy()

    if spec.kind == "subgroup_shift":
        if len(cat_cols) > 0:
            c = cat_cols[0]
            vals = X[c].astype(str).values
            uniq, cnt = np.unique(vals, return_counts=True)
            sg = uniq[int(np.argmax(cnt))]
            mask = (vals == sg)
        elif len(num_cols) > 0:
            c = num_cols[0]
            v = X[c].astype(float).values
            q75 = np.nanquantile(v, 0.75)
            mask = v >= q75
        else:
            mask = np.zeros(len(X), dtype=bool)
        return subgroup_shift(X, num_cols, mask, float(spec.params["shift_scale"])), y.copy()

    if spec.kind == "prevalence_shift":
        Xr, yr = prevalence_shift_resample(X, y, float(spec.params["target_prev"]), rng)
        return Xr, yr

    if spec.kind == "concept_drift":
        Xc = concept_drift_label_conditional(X, y, num_cols, int(spec.params["k"]), float(spec.params["shift_scale"]), rng)
        return Xc, y.copy()

    if spec.kind == "label_noise":
        return X.copy(), flip_labels(y, float(spec.params["eta"]), rng)

    raise ValueError(f"Unknown world kind: {spec.kind}")


# -----------------------------
# Pipelines
# -----------------------------
def make_preprocessor(num_cols: List[str], cat_cols: List[str], force_dense: bool) -> ColumnTransformer:
    num_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # sklearn compatibility: sparse_output (new) vs sparse (old)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=not force_dense)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=not force_dense)

    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])

    return ColumnTransformer(
        transformers=[("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_model_zoo(seed: int) -> Dict[str, Any]:
    return {
        "Logistic_L2": LogisticRegression(max_iter=5000, random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=1200, n_jobs=-1, random_state=seed, class_weight="balanced_subsample"),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=1200, n_jobs=-1, random_state=seed, class_weight="balanced_subsample"),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=700, learning_rate=0.03, random_state=seed),
        "SVC_RBF": SVC(kernel="rbf", probability=True, C=3.0, gamma="scale", random_state=seed),
        "MLP": MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=2500, random_state=seed),
        "GaussianNB": GaussianNB(),
    }


def make_pipeline(name: str, clf: Any, num_cols: List[str], cat_cols: List[str], calibrate: Optional[str]) -> Pipeline:
    need_dense = (name == "GaussianNB")
    pre = make_preprocessor(num_cols, cat_cols, force_dense=need_dense)
    steps = [("preprocess", pre)]
    if need_dense:
        steps.append(("to_dense", FunctionTransformer(to_dense_if_sparse, accept_sparse=True)))

    est = clf
    if calibrate is not None:
        est = CalibratedClassifierCV(estimator=clf, method=calibrate, cv=3)

    steps.append(("clf", est))
    return Pipeline(steps)


def predict_proba_safe(pipe: Pipeline, Xw: pd.DataFrame) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        return np.asarray(pipe.predict_proba(Xw)[:, 1], dtype=float)
    if hasattr(pipe, "decision_function"):
        z = pipe.decision_function(Xw)
        return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=float)
    raise RuntimeError("Model does not support probability prediction.")


def eval_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1.0 - 1e-7)
    out = {
        "AUROC": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "LogLoss": float(log_loss(y_true, y_prob)),
        "ECE": float(expected_calibration_error(y_true, y_prob, n_bins=15)),
        "aECE": float(adaptive_ece(y_true, y_prob, n_bins=15)),
    }
    dec = brier_decomposition(y_true, y_prob, n_bins=15, adaptive=True)
    out.update({f"Brier_{k}": float(v) for k, v in dec.items()})
    return out


# -----------------------------
# SHAP-based explainability stability E + ablations
# -----------------------------
def _select_shap_indices(n: int, max_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=min(max_samples, n), replace=False)


def compute_shap_matrix_on_index(pipe: Pipeline, X: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
    if not SHAP_AVAILABLE:
        raise RuntimeError("SHAP is not available. Please install shap.")

    X_sub = X.iloc[idx].copy()
    Xt = pipe.named_steps["preprocess"].transform(X_sub)
    if "to_dense" in pipe.named_steps:
        Xt = pipe.named_steps["to_dense"].transform(Xt)

    clf = pipe.named_steps["clf"]
    base_model = clf.estimator if isinstance(clf, CalibratedClassifierCV) else clf

    tree_like = (RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier)
    if isinstance(base_model, tree_like):
        explainer = shap.TreeExplainer(base_model)
        sv = explainer.shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv, dtype=float)
        if sv.ndim > 2:
            sv = sv.reshape(sv.shape[0], -1)
        return sv

    if isinstance(base_model, LogisticRegression):
        explainer = shap.LinearExplainer(base_model, Xt, feature_perturbation="interventional")
        sv = np.asarray(explainer.shap_values(Xt), dtype=float)
        if sv.ndim > 2:
            sv = sv.reshape(sv.shape[0], -1)
        return sv

    masker = shap.maskers.Independent(Xt)
    explainer = shap.Explainer(base_model, masker)
    sv = np.asarray(explainer(Xt).values, dtype=float)
    if sv.ndim > 2:
        sv = sv.reshape(sv.shape[0], -1)
    return sv


def topk_jaccard(a: np.ndarray, b: np.ndarray, k: int = 20) -> float:
    ia = set(np.argsort(-np.abs(a))[:k].tolist())
    ib = set(np.argsort(-np.abs(b))[:k].tolist())
    inter = len(ia & ib)
    union = len(ia | ib)
    return float(inter / union) if union > 0 else np.nan


def explainability_stability_ablation(
    pipe: Pipeline,
    X_clean: pd.DataFrame,
    y_clean: np.ndarray,
    world_specs: List[WorldSpec],
    num_cols: List[str],
    cat_cols: List[str],
    seed: int,
    shap_samples: int = 250,
    topk: int = 20
) -> Dict[str, float]:
    n = len(X_clean)
    idx = _select_shap_indices(n=n, max_samples=shap_samples, seed=seed)

    shapA = compute_shap_matrix_on_index(pipe, X_clean, idx)
    gA = np.mean(np.abs(shapA), axis=0)

    cos_scores, rho_scores, jac_scores = [], [], []
    for spec in world_specs:
        if spec.kind in ("clean", "label_noise"):
            continue

        X_sub = X_clean.iloc[idx].copy()
        y_sub = y_clean[idx].copy()
        Xw, yw = apply_world(spec, X_sub, y_sub, num_cols=num_cols, cat_cols=cat_cols, seed=seed)
        shapW = compute_shap_matrix_on_index(pipe, Xw, np.arange(len(idx)))
        gW = np.mean(np.abs(shapW), axis=0)

        eps = 1e-10
        num = np.sum(shapA * shapW, axis=1)
        den = (np.linalg.norm(shapA, axis=1) + eps) * (np.linalg.norm(shapW, axis=1) + eps)
        cos = float(np.mean(num / den))
        cos_scores.append((cos + 1.0) / 2.0)

        rho, _ = spearmanr(gA, gW)
        rho = float(np.nan_to_num(rho))
        rho_scores.append((rho + 1.0) / 2.0)

        jac_scores.append(topk_jaccard(gA, gW, k=topk))

    E_cos = float(np.mean(cos_scores)) if len(cos_scores) else np.nan
    E_rank = float(np.mean(rho_scores)) if len(rho_scores) else np.nan
    E_jaccard = float(np.mean(jac_scores)) if len(jac_scores) else np.nan
    E_mix = float(np.nanmean([E_cos, E_rank, E_jaccard]))
    return {"E_cos": E_cos, "E_rank": E_rank, "E_jaccard": E_jaccard, "E_mix": E_mix}


# -----------------------------
# SHAP plots (summary + dependence)
# -----------------------------
def _get_transformed_X_and_feature_names(pipe: Pipeline, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    Xt = pipe.named_steps["preprocess"].transform(X)
    if "to_dense" in pipe.named_steps:
        Xt = pipe.named_steps["to_dense"].transform(Xt)
    Xt = np.asarray(Xt)

    pre = pipe.named_steps["preprocess"]
    try:
        fn = pre.get_feature_names_out()
        feature_names = [str(s) for s in fn]
    except Exception:
        feature_names = [f"f{i}" for i in range(Xt.shape[1])]
    return Xt, feature_names


def _compute_shap_values_for_Xt(pipe: Pipeline, Xt: np.ndarray) -> np.ndarray:
    if not SHAP_AVAILABLE:
        raise RuntimeError("SHAP is not available. Please install shap.")

    clf = pipe.named_steps["clf"]
    base_model = clf.estimator if isinstance(clf, CalibratedClassifierCV) else clf

    tree_like = (RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier)
    if isinstance(base_model, tree_like):
        explainer = shap.TreeExplainer(base_model)
        sv = explainer.shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv, dtype=float)
        if sv.ndim > 2:
            sv = sv.reshape(sv.shape[0], -1)
        return sv

    if isinstance(base_model, LogisticRegression):
        explainer = shap.LinearExplainer(base_model, Xt, feature_perturbation="interventional")
        sv = np.asarray(explainer.shap_values(Xt), dtype=float)
        if sv.ndim > 2:
            sv = sv.reshape(sv.shape[0], -1)
        return sv

    masker = shap.maskers.Independent(Xt)
    explainer = shap.Explainer(base_model, masker)
    sv = np.asarray(explainer(Xt).values, dtype=float)
    if sv.ndim > 2:
        sv = sv.reshape(sv.shape[0], -1)
    return sv


def plot_shap_suite(pipe: Pipeline, X: pd.DataFrame, outdir: Path, tag: str, max_display: int = 20) -> None:
    if not SHAP_AVAILABLE:
        return

    ensure_dir(outdir)

    Xt, feature_names = _get_transformed_X_and_feature_names(pipe, X)
    sv = _compute_shap_values_for_Xt(pipe, Xt)

    plt.figure(figsize=(8.5, 8.5))
    shap.summary_plot(sv, features=Xt, feature_names=feature_names, plot_type="dot", max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(outdir / f"shap_summary_beeswarm_{tag}.png", dpi=350, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8.5, 8.5))
    shap.summary_plot(sv, features=Xt, feature_names=feature_names, plot_type="bar", max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(outdir / f"shap_summary_bar_{tag}.png", dpi=350, bbox_inches="tight")
    plt.close()

    mean_abs = np.mean(np.abs(sv), axis=0)
    top_idx = int(np.nanargmax(mean_abs))
    top_name = feature_names[top_idx] if top_idx < len(feature_names) else f"f{top_idx}"

    plt.figure(figsize=(8.5, 8.5))
    shap.dependence_plot(top_name, sv, Xt, feature_names=feature_names, show=False, interaction_index="auto")
    plt.tight_layout()
    plt.savefig(outdir / f"shap_dependence_{tag}.png", dpi=350, bbox_inches="tight")
    plt.close()


# -----------------------------
# RSCE ablation formulations (S, C)
# -----------------------------
def compute_S_formulations(auc_clean: float, auc_worlds: List[float], eps: float = 1e-12) -> Dict[str, float]:
    drops = [max(0.0, auc_clean - a) for a in auc_worlds]
    S_drop = clip01(1.0 - float(np.mean(drops))) if len(drops) else np.nan
    ratios = [clip01(a / (auc_clean + eps)) for a in auc_worlds]
    S_ratio = float(np.mean(ratios)) if len(ratios) else np.nan
    return {"S_drop": S_drop, "S_ratio": S_ratio}


def compute_C_formulations(ece_clean: float, ece_worlds: List[float], eps: float = 1e-12) -> Dict[str, float]:
    drifts = [abs(e - ece_clean) for e in ece_worlds]
    C_exp = float(np.mean([float(np.exp(-d / (ece_clean + eps))) for d in drifts])) if len(drifts) else np.nan
    C_linear = clip01(1.0 - float(np.mean([min(1.0, d) for d in drifts]))) if len(drifts) else np.nan
    return {"C_exp": C_exp, "C_linear": C_linear}


# -----------------------------
# Reliability diagram aggregation
# -----------------------------
def plot_reliability_diagram(curves: pd.DataFrame, outpath: Path, title: str) -> None:
    df = curves.copy()
    has_fold = "fold" in df.columns

    def _wavg(x: pd.Series, w: pd.Series) -> float:
        x = x.astype(float).values
        w = w.astype(float).values
        s = np.sum(w)
        return float(np.sum(w * x) / s) if s > 0 else float(np.nanmean(x))

    grp_cols = ["model", "world", "bin"]
    agg_rows = []
    for (m, w, b), sub in df.groupby(grp_cols):
        p_bar = _wavg(sub["p_mean"], sub["count"])
        y_bar = _wavg(sub["y_mean"], sub["count"])
        n_tot = int(sub["count"].sum())

        row = {"model": m, "world": w, "bin": int(b), "p_mean": p_bar, "y_mean": y_bar, "count": n_tot}

        if has_fold:
            fold_means = []
            for f, sf in sub.groupby("fold"):
                fold_means.append(_wavg(sf["y_mean"], sf["count"]))
            row["y_std_fold"] = float(np.nanstd(np.asarray(fold_means, dtype=float)))

        agg_rows.append(row)

    agg = pd.DataFrame(agg_rows).sort_values(["model", "world", "bin"])

    plt.figure(figsize=(8.5, 8.5))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5)

    cmin = max(1, int(agg["count"].min())) if len(agg) else 1
    cmax = max(1, int(agg["count"].max())) if len(agg) else 1

    def _msize(c: int) -> float:
        if cmax == cmin:
            return 6.0
        t = (c - cmin) / (cmax - cmin)
        return 4.0 + 10.0 * float(t)

    for (m, w), sub in agg.groupby(["model", "world"]):
        sub = sub.sort_values("p_mean")
        x = sub["p_mean"].values.astype(float)
        y = sub["y_mean"].values.astype(float)

        plt.plot(x, y, linewidth=2.0, alpha=0.9, label=f"{m} | {w}")
        sizes = [_msize(int(c)) for c in sub["count"].values]
        plt.scatter(x, y, s=np.square(sizes), alpha=0.75)

        if has_fold and "y_std_fold" in sub.columns:
            ystd = sub["y_std_fold"].values.astype(float)
            lo = np.clip(y - ystd, 0.0, 1.0)
            hi = np.clip(y + ystd, 0.0, 1.0)
            plt.fill_between(x, lo, hi, alpha=0.12)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical event rate")
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=350, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main benchmarking loop
# -----------------------------
def run_benchmark(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset.")
    df = df.dropna(subset=[args.target]).copy()
    df[args.target] = df[args.target].astype(int)

    drop_cols = set([args.target])
    for c in args.drop_cols:
        if c in df.columns:
            drop_cols.add(c)

    X_full = df.drop(columns=list(drop_cols), errors="ignore")
    y = df[args.target].values.astype(int)

    num_cols = [c for c in X_full.columns if pd.api.types.is_numeric_dtype(X_full[c])]
    cat_cols = [c for c in X_full.columns if c not in num_cols]

    if args.numeric_cols is not None:
        num_cols = args.numeric_cols
        cat_cols = [c for c in X_full.columns if c not in num_cols]
    if args.categorical_cols is not None:
        cat_cols = args.categorical_cols
        num_cols = [c for c in X_full.columns if c not in cat_cols]

    X = X_full[num_cols + cat_cols].copy()

    world_specs = build_world_specs()
    model_zoo = make_model_zoo(seed=args.seed)
    pipelines: Dict[str, Pipeline] = {
        m: make_pipeline(m, clf, num_cols, cat_cols, args.calibrate) for m, clf in model_zoo.items()
    }

    pkgs = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "shap"]
    versions = get_versions(pkgs)

    schema = {
        "n_samples": int(len(df)),
        "target": args.target,
        "dropped": sorted(list(drop_cols)),
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "seed": args.seed,
        "folds": args.folds,
        "repeats": args.repeats,
        "calibrate": args.calibrate,
        "compute_shap": bool(args.compute_shap),
        "shap_samples": args.shap_samples,
        "worlds": [w.__dict__ for w in world_specs],
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "package_versions": versions,
        "component_ablation": {
            "S": ["S_drop", "S_ratio"],
            "C": ["C_exp", "C_linear"],
            "E": ["E_cos", "E_rank", "E_jaccard", "E_mix"],
        },
    }
    (outdir / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    rskf = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed)

    records = []
    cost_records = []
    shap_records = []
    reliability_records = []

    # ✅ Progress accounting (for one overall progress bar)
    n_folds_total = args.folds * args.repeats
    n_models = len(pipelines)
    n_worlds = len(world_specs)
    n_tasks_total = n_folds_total * n_models * n_worlds  # eval tasks per (fold, model, world)
    # SHAP is optional and not counted strictly; tqdm ETA still works well enough.

    fold_id = 0

    with tqdm(total=n_tasks_total, desc="RSCE Benchmark (eval)", unit="task", dynamic_ncols=True) as pbar:
        for train_idx, test_idx in rskf.split(X, y):
            fold_id += 1
            X_tr, y_tr = X.iloc[train_idx].copy(), y[train_idx].copy()
            X_te0, y_te0 = X.iloc[test_idx].copy(), y[test_idx].copy()

            # Nested bar for models in this fold (visual step indicator)
            with tqdm(total=n_models, desc=f"Fold {fold_id}/{n_folds_total} (models)", unit="model", leave=False, dynamic_ncols=True) as pbar_models:
                for mname, pipe in pipelines.items():
                    pbar_models.set_postfix_str(mname)

                    t0 = time.perf_counter()
                    pipe.fit(X_tr, y_tr)
                    t_fit = time.perf_counter() - t0

                    t_pred_total = 0.0

                    # Nested bar for worlds in this model (optional, leave=False)
                    with tqdm(total=n_worlds, desc=f"{mname} (worlds)", unit="world", leave=False, dynamic_ncols=True) as pbar_worlds:
                        for spec in world_specs:
                            pbar_worlds.set_postfix_str(spec.name)

                            X_te, y_te = apply_world(
                                spec, X_te0, y_te0,
                                num_cols=num_cols, cat_cols=cat_cols,
                                seed=args.seed + fold_id
                            )

                            tp0 = time.perf_counter()
                            y_prob = predict_proba_safe(pipe, X_te)
                            t_pred = time.perf_counter() - tp0
                            t_pred_total += t_pred

                            mets = eval_metrics(y_te, y_prob)
                            mets.update({
                                "fold": fold_id,
                                "model": mname,
                                "world": spec.name,
                                "severity": spec.severity,
                                "n_eval": int(len(y_te)),
                            })
                            records.append(mets)

                            if (mname in args.reliability_models) and (spec.name in args.reliability_worlds):
                                curve = reliability_curve_points(
                                    y_te, y_prob,
                                    n_bins=args.reliability_bins,
                                    adaptive=args.reliability_adaptive
                                )
                                if len(curve):
                                    curve["fold"] = fold_id
                                    curve["model"] = mname
                                    curve["world"] = spec.name
                                    reliability_records.append(curve)

                            # ✅ Update progress
                            pbar_worlds.update(1)
                            pbar.update(1)

                    # ---- SHAP (optional) after world loop ----
                    t_shap = 0.0
                    if args.compute_shap:
                        if not SHAP_AVAILABLE:
                            raise RuntimeError("You requested --compute_shap but shap is not installed.")

                        est = pipe.named_steps["clf"]
                        base = est.estimator if isinstance(est, CalibratedClassifierCV) else est
                        supported = isinstance(
                            base,
                            (RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, LogisticRegression),
                        )

                        if supported or args.shap_allow_generic:
                            # show "stage" in main bar postfix
                            old_postfix = pbar.postfix
                            pbar.set_postfix_str(f"SHAP: {mname} (fold {fold_id})")

                            ts0 = time.perf_counter()
                            eab = explainability_stability_ablation(
                                pipe=pipe,
                                X_clean=X_te0,
                                y_clean=y_te0,
                                world_specs=world_specs,
                                num_cols=num_cols,
                                cat_cols=cat_cols,
                                seed=args.seed + 999 * fold_id + deterministic_int_hash(mname),
                                shap_samples=args.shap_samples,
                                topk=args.E_topk,
                            )
                            t_shap = time.perf_counter() - ts0
                            eab.update({"fold": fold_id, "model": mname})
                            shap_records.append(eab)

                            # SHAP plots (fold 1 only; for reliability_models if specified)
                            rel_models = args.reliability_models
                            plot_for_model = True if not rel_models else (mname in rel_models)
                            plot_for_fold = (fold_id == 1)

                            if plot_for_fold and plot_for_model:
                                shap_dir = outdir / "shap_plots"
                                ensure_dir(shap_dir)

                                n_plot = min(args.shap_samples, len(X_te0))
                                idx_plot = _select_shap_indices(
                                    n=len(X_te0),
                                    max_samples=n_plot,
                                    seed=args.seed + 123 + deterministic_int_hash(mname),
                                )
                                X_plot = X_te0.iloc[idx_plot].copy()

                                try:
                                    plot_shap_suite(
                                        pipe=pipe,
                                        X=X_plot,
                                        outdir=shap_dir,
                                        tag=f"{mname}_fold{fold_id}",
                                        max_display=20,
                                    )
                                    tqdm.write(f"[SHAP] Plots saved to: {shap_dir} (tag={mname}_fold{fold_id})")
                                except Exception as e:
                                    tqdm.write(f"[SHAP] plot_shap_suite failed for {mname}, fold={fold_id}: {e}")

                            # restore postfix (best-effort)
                            try:
                                pbar.postfix = old_postfix
                            except Exception:
                                pass
                        else:
                            tqdm.write(
                                f"[SHAP] Skip SHAP: model base estimator not supported ({type(base).__name__}) "
                                f"and args.shap_allow_generic is False."
                            )

                    # ---- cost per fold/model ----
                    cost_records.append({
                        "fold": fold_id,
                        "model": mname,
                        "time_fit_s": float(t_fit),
                        "time_pred_all_worlds_s": float(t_pred_total),
                        "time_shap_s": float(t_shap),
                        "time_total_s": float(t_fit + t_pred_total + t_shap),
                    })

                    pbar_models.update(1)

    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(outdir / "metrics_per_fold.csv", index=False)

    cost_df = pd.DataFrame(cost_records)
    cost_df.to_csv(outdir / "compute_cost_per_fold.csv", index=False)

    cost_agg = cost_df.groupby("model")[["time_fit_s", "time_pred_all_worlds_s", "time_shap_s", "time_total_s"]].agg(["mean", "std"]).reset_index()
    cost_agg.columns = ["_".join([c for c in col if c]) for col in cost_agg.columns.values]
    cost_agg.to_csv(outdir / "compute_cost.csv", index=False)

    shap_df = pd.DataFrame(shap_records)
    if len(shap_df):
        shap_df.to_csv(outdir / "E_ablation_per_fold.csv", index=False)

    if len(reliability_records):
        rel_df = pd.concat(reliability_records, ignore_index=True)
        rel_df.to_csv(outdir / "reliability_curve_points.csv", index=False)
        plot_reliability_diagram(
            curves=rel_df,
            outpath=outdir / "reliability_diagram.png",
            title=f"Reliability diagram (bins={args.reliability_bins}, adaptive={args.reliability_adaptive})"
        )

    # World-level aggregation
    agg_rows = []
    for (model, world), sub in metrics_df.groupby(["model", "world"]):
        row = {"model": model, "world": world, "severity": int(sub["severity"].iloc[0])}
        for met in ["AUROC", "Brier", "LogLoss", "ECE", "aECE", "Brier_REL", "Brier_RES", "Brier_UNC"]:
            vals = sub[met].values.astype(float)
            row[met + "_mean"] = float(np.nanmean(vals))
            lo, hi = bootstrap_ci(vals, n_boot=args.ci_boot, ci=0.95, seed=args.seed + deterministic_int_hash(model + world + met))
            row[met + "_ci_lo"] = lo
            row[met + "_ci_hi"] = hi
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows).sort_values(["model", "severity"])
    agg_df.to_csv(outdir / "metrics_aggregated.csv", index=False)

    # Component ablations per fold
    ref_world = "WA_clean"
    cov_worlds = [w.name for w in world_specs if w.kind not in ("clean", "label_noise")]
    label_world = next((w.name for w in world_specs if w.kind == "label_noise"), None)

    comp_fold_rows = []
    for (fold, model), sub in metrics_df.groupby(["fold", "model"]):
        auc0 = float(sub.loc[sub["world"] == ref_world, "AUROC"].values[0])
        ece0 = float(sub.loc[sub["world"] == ref_world, "ECE"].values[0])

        auc_ws = [float(sub.loc[sub["world"] == w, "AUROC"].values[0]) for w in cov_worlds]
        ece_ws = [float(sub.loc[sub["world"] == w, "ECE"].values[0]) for w in cov_worlds]

        S_forms = compute_S_formulations(auc0, auc_ws)
        C_forms = compute_C_formulations(ece0, ece_ws)

        Q = np.nan
        if label_world is not None:
            aucL = float(sub.loc[sub["world"] == label_world, "AUROC"].values[0])
            Q = clip01(aucL / (auc0 + 1e-12))

        row = {"fold": int(fold), "model": model, "R": auc0, "Q_label": Q}
        row.update(S_forms)
        row.update(C_forms)
        comp_fold_rows.append(row)

    comp_fold_df = pd.DataFrame(comp_fold_rows)

    if len(shap_df):
        comp_fold_df = comp_fold_df.merge(shap_df, on=["fold", "model"], how="left")
    comp_fold_df.to_csv(outdir / "ablation_components_per_fold.csv", index=False)

    ab_cols = [c for c in comp_fold_df.columns if c not in ("fold", "model")]
    sum_rows = []
    for model, sub in comp_fold_df.groupby("model"):
        row = {"model": model}
        for c in ab_cols:
            vals = sub[c].astype(float).values
            row[c + "_mean"] = float(np.nanmean(vals))
            lo, hi = bootstrap_ci(vals, n_boot=args.ci_boot, ci=0.95, seed=args.seed + deterministic_int_hash(model + c))
            row[c + "_ci_lo"] = lo
            row[c + "_ci_hi"] = hi
        sum_rows.append(row)
    ab_sum = pd.DataFrame(sum_rows)
    ab_sum.to_csv(outdir / "ablation_summary.csv", index=False)

    # Ranking agreement across formulations
    def compute_score(dfm: pd.DataFrame, Scol: str, Ccol: str, Ecol: Optional[str]) -> pd.Series:
        E = dfm[Ecol].fillna(0.0) if (Ecol is not None and Ecol in dfm.columns) else 0.0
        wR, wS, wC, wE = args.wR, args.wS, args.wC, args.wE
        ws = wR + wS + wC + wE
        wR, wS, wC, wE = wR / ws, wS / ws, wC / ws, wE / ws
        return wR * dfm["R"] + wS * dfm[Scol] + wC * dfm[Ccol] + wE * E

    mean_cols = {c: c.replace("_mean", "") for c in ab_sum.columns if c.endswith("_mean")}
    mean_df = pd.DataFrame({"model": ab_sum["model"]})
    for c_mean, c_raw in mean_cols.items():
        mean_df[c_raw] = ab_sum[c_mean].values

    score_variants = {}
    for Scol in ["S_drop", "S_ratio"]:
        for Ccol in ["C_exp", "C_linear"]:
            for Ecol in (["E_mix", "E_cos", "E_rank", "E_jaccard"] if "E_mix" in mean_df.columns else [None]):
                key = f"{Scol}|{Ccol}|{Ecol if Ecol else 'E0'}"
                score_variants[key] = compute_score(mean_df, Scol=Scol, Ccol=Ccol, Ecol=Ecol)

    ref_key = "S_ratio|C_exp|E_mix" if "E_mix" in mean_df.columns else "S_ratio|C_exp|E0"
    ref_scores = score_variants[ref_key]
    ref_rank = ref_scores.rank(ascending=False, method="average")

    agree_rows = []
    for k, sc in score_variants.items():
        r = sc.rank(ascending=False, method="average")
        rho, _ = spearmanr(ref_rank.values, r.values)
        tau, _ = kendalltau(ref_rank.values, r.values)
        agree_rows.append({"variant": k, "spearman_rank_vs_ref": float(rho), "kendall_rank_vs_ref": float(tau)})
    agree_df = pd.DataFrame(agree_rows).sort_values("spearman_rank_vs_ref", ascending=False)
    agree_df.to_csv(outdir / "ablation_rank_agreement.csv", index=False)

    main_variant = ref_key
    main_scores = score_variants[main_variant]
    rsce_out = mean_df[["model"]].copy()
    rsce_out["RSCE_full"] = main_scores.values
    rsce_out = rsce_out.sort_values("RSCE_full", ascending=False)
    rsce_out.to_csv(outdir / "rsce_scores.csv", index=False)

    # Paired tests
    if "E_mix" in comp_fold_df.columns:
        fold_scores = comp_fold_df.copy()
        fold_scores["RSCE_fold"] = compute_score(fold_scores, Scol="S_ratio", Ccol="C_exp", Ecol="E_mix").values
    else:
        fold_scores = comp_fold_df.copy()
        fold_scores["RSCE_fold"] = compute_score(fold_scores, Scol="S_ratio", Ccol="C_exp", Ecol=None).values

    models = sorted(fold_scores["model"].unique().tolist())
    pairs = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            f1 = fold_scores.loc[fold_scores["model"] == m1, ["fold", "RSCE_fold"]].sort_values("fold")
            f2 = fold_scores.loc[fold_scores["model"] == m2, ["fold", "RSCE_fold"]].sort_values("fold")
            s1 = f1["RSCE_fold"].values.astype(float)
            s2 = f2["RSCE_fold"].values.astype(float)

            try:
                w_stat, w_p = wilcoxon(s1, s2, zero_method="wilcox", alternative="two-sided")
            except Exception:
                w_stat, w_p = np.nan, np.nan

            t_stat, t_p = ttest_rel(s1, s2, nan_policy="omit")

            pairs.append({
                "m1": m1, "m2": m2,
                "wilcoxon_p": float(w_p), "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else np.nan,
                "ttest_p": float(t_p), "ttest_stat": float(t_stat),
            })

    pairs_df = pd.DataFrame(pairs)
    if len(pairs_df):
        pairs_df["wilcoxon_p_holm"] = holm_correction(pairs_df["wilcoxon_p"].fillna(1.0))
        pairs_df["ttest_p_holm"] = holm_correction(pairs_df["ttest_p"].fillna(1.0))
    pairs_df.to_csv(outdir / "paired_tests.csv", index=False)

    # Sensitivity plots
    focus_models = args.focus_models or ["RandomForest", "ExtraTrees", "GradientBoosting"]
    wR, wS, wC, wE = args.wR, args.wS, args.wC, args.wE
    ws = wR + wS + wC + wE
    wR, wS, wC, wE = wR / ws, wS / ws, wC / ws, wE / ws
    base = np.array([wR, wS, wC, wE], dtype=float)
    ratio_rest = base[1:] / max(base[1:].sum(), 1e-12)

    wR_values = np.linspace(args.wR_min, args.wR_max, args.wR_steps)
    comp_use = mean_df.set_index("model")
    comp_use = comp_use.loc[[m for m in focus_models if m in comp_use.index]]

    rsce_sweep = {m: [] for m in comp_use.index}
    for wR2 in wR_values:
        rem = 1.0 - float(wR2)
        wS2, wC2, wE2 = rem * ratio_rest
        for m in comp_use.index:
            Rm = float(comp_use.loc[m, "R"])
            Sm = float(comp_use.loc[m, "S_ratio"])
            Cm = float(comp_use.loc[m, "C_exp"])
            Em = float(comp_use.loc[m, "E_mix"]) if "E_mix" in comp_use.columns else 0.0
            rsce_sweep[m].append(float(wR2 * Rm + wS2 * Sm + wC2 * Cm + wE2 * Em))

    plt.figure(figsize=(8.5, 8.5))
    for m, vals in rsce_sweep.items():
        plt.plot(wR_values, vals, marker="o", label=m)
    plt.xlabel("Weight on Reliability (wR)")
    plt.ylabel("RSCE (recomputed)")
    plt.title("Sensitivity of RSCE to weight on Reliability")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "sensitivity_weights.png", dpi=300)
    plt.close()

    # Severity sensitivity plot (plot-only composite)
    ws_df = agg_df.copy()
    ws_df = ws_df[ws_df["model"].isin(focus_models)]
    ws_df["world_score"] = 0.5 * ws_df["AUROC_mean"].astype(float) + 0.5 * np.exp(-ws_df["aECE_mean"].astype(float))

    plt.figure(figsize=(8.5, 8.5))
    for m in focus_models:
        sub = ws_df[ws_df["model"] == m].sort_values("severity")
        if len(sub) == 0:
            continue
        plt.plot(sub["severity"].values, sub["world_score"].values, marker="o", label=m)
    plt.xlabel("Perturbation severity (world index)")
    plt.ylabel("Composite world score (plot-only)")
    plt.title("Sensitivity to perturbation severity (held-out worlds)")
    plt.xticks([w.severity for w in world_specs], [w.name for w in world_specs], rotation=45, ha="right")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "sensitivity_severity.png", dpi=300)
    plt.close()

    # RSCE bar plot
    plt.figure(figsize=(8.5, 8.5))
    top = rsce_out.sort_values("RSCE_full", ascending=False)
    plt.bar(top["model"].values, top["RSCE_full"].values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("RSCE_full")
    plt.title(f"RSCE_full by model (main variant: {main_variant})")
    plt.tight_layout()
    plt.savefig(outdir / "rsce_bar.png", dpi=300)
    plt.close()

    print("\n[Done] RSCE benchmark (v3) completed.")
    print(f"Outputs saved in: {outdir.resolve()}")
    print("Key outputs: rsce_scores.csv, paired_tests.csv, compute_cost.csv, ablation_summary.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RSCE multi-world benchmark (v3: ablations, realism worlds, calibration, tests, cost, env).")

    p.add_argument("--data", type=str, default="analytic_dataset_mortality_all_admissions.csv", help="Path to CSV dataset.")
    p.add_argument("--target", type=str, default="label_mortality", help="Binary target column name (0/1).")
    p.add_argument("--outdir", type=str, default="rsce_results_full_dataset", help="Output directory.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--ci_boot", type=int, default=2000)

    # ✅ แก้ choices: ไม่ใส่ None ใน choices (ให้ default=None แทน)
    p.add_argument("--calibrate", type=str, default=None, choices=["sigmoid", "isotonic"], help="Calibration method (optional).")

    p.add_argument("--compute_shap", action="store_true", default=True)
    p.add_argument("--shap_samples", type=int, default=250)
    p.add_argument("--shap_allow_generic", action="store_true")
    p.add_argument("--E_topk", type=int, default=20, help="Top-k for Jaccard feature stability.")

    p.add_argument("--drop_cols", type=str, nargs="*", default=[
        "hadm_id", "subject_id", "discharge_location", "anchor_year", "anchor_year_group"
    ])

    p.add_argument("--numeric_cols", type=str, nargs="*", default=None)
    p.add_argument("--categorical_cols", type=str, nargs="*", default=None)

    p.add_argument("--wR", type=float, default=0.4)
    p.add_argument("--wS", type=float, default=0.3)
    p.add_argument("--wC", type=float, default=0.2)
    p.add_argument("--wE", type=float, default=0.1)

    p.add_argument("--focus_models", type=str, nargs="*", default=None)
    p.add_argument("--wR_min", type=float, default=0.30)
    p.add_argument("--wR_max", type=float, default=0.50)
    p.add_argument("--wR_steps", type=int, default=7)

    p.add_argument("--reliability_models", type=str, nargs="*", default=["Logistic_L2", "RandomForest"])
    p.add_argument("--reliability_worlds", type=str, nargs="*", default=["WA_clean", "WI_prevalence_shift"])
    p.add_argument("--reliability_bins", type=int, default=15)
    p.add_argument("--reliability_adaptive", action="store_true")

    args, _ = p.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()


