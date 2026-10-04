#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_rsce.py
===========
RSCE = 0.4*R + 0.3*S + 0.2*C + 0.1*E multi-world benchmark. Fully CLI-driven and
dataset-agnostic already -- the same script runs for hosp or ED, demo or full,
by pointing --data/--target/--drop_cols/--outdir at the right analytic dataset.

Typical usage (see the repository README.md for the full command set):

  # Hospital module (target = label_mortality, produced by 01_prepare_data/prepare_hosp.py)
  python run_rsce.py --data demo_analytic_dataset_mortality_all_admissions.csv \
      --target label_mortality --drop_cols hadm_id subject_id discharge_location anchor_year anchor_year_group \
      --outdir results/hosp_demo

  # ED module (target = label_ed_admit, produced by 01_prepare_data/prepare_ed.py)
  python run_rsce.py --data demo_ed_analytic_dataset_admission.csv \
      --target label_ed_admit --drop_cols stay_id \
      --outdir results/ed_demo

Design choices (the reasons are in the comments at each site):
patient-grouped repeated CV (--cv_mode group); signed-log1p nonlinear world;
E computed on stratified, model-independent SHAP rows with a fixed clean SHAP
background, excluding the prevalence-shift world; positive-class SHAP slice
for 3-D SHAP output; C_linear as primary C and missing E renormalised
(--E_missing renormalize), with RSCE_RSC and RSCE_legacy reported alongside;
Nadeau-Bengio corrected paired tests with Holm adjustment; a fresh clone() per
(fold, model) unit; EarlyStoppedMLP in place of a plain MLPClassifier;
per-unit checkpoint/resume with configuration and model-parameter signatures;
parallel TreeSHAP (--shap_n_jobs); SVC training-fold cap (--svc_max_train_n).
"""

from __future__ import annotations

import argparse
import json
import random
import hashlib
import time
import platform
import gc
import os
import pickle
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: these scripts only ever plt.savefig()/plt.close(),
                        # never plt.show() -- forcing Agg avoids a real crash seen on Windows where the
                        # auto-selected interactive TkAgg backend hit a Tcl/Tk "wrong thread" assertion
                        # (Tcl_AsyncDelete) and killed the whole process (exit code 0x80000003) partway
                        # through a run_rsce.py SHAP-plotting pass.
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, kendalltau, wilcoxon, ttest_rel

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold, StratifiedShuffleSplit, ShuffleSplit
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
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from scipy.stats import t as student_t

# Progress bar (tqdm)
from tqdm.auto import tqdm
from joblib import Parallel, delayed

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


# -----------------------------
# Group-aware repeated CV
# -----------------------------
# WHY: MIMIC-IV admissions/ED-stays are not i.i.d. -- the same patient
# (subject_id) can have several admissions / ED stays. Plain row-level
# RepeatedStratifiedKFold can put two admissions from the *same* patient in
# both the train and test side of a fold, letting a model partially memorize
# that patient rather than generalize -- an optimistic bias in AUROC/etc.
# sklearn has StratifiedGroupKFold (group-aware) but no "Repeated" version of
# it, so repeats are implemented by re-instantiating it with a different
# random_state per repeat, mirroring what RepeatedStratifiedKFold does
# internally for the non-grouped case.
def choose_safe_group_folds(y: np.ndarray, groups: Optional[np.ndarray], desired_splits: int) -> int:
    """
    Largest n_splits (<= desired_splits) for which a stratified split is
    possible: bounded by the minority class count, and -- if grouping -- also
    by the number of *distinct groups* within the minority class (a group
    can't be split across folds, so you need at least n_splits distinct
    groups in the smaller class). Returns 0 if no valid split (>=2) exists.
    """
    y = np.asarray(y)
    vals, cnts = np.unique(y, return_counts=True)
    if len(vals) < 2:
        return 0
    min_class = int(cnts.min())
    if groups is None:
        safe = min(desired_splits, min_class)
    else:
        groups = np.asarray(groups)
        min_groups_per_class = min(int(len(np.unique(groups[y == v]))) for v in vals)
        safe = min(desired_splits, min_class, min_groups_per_class)
    return int(safe) if safe >= 2 else 0


def repeated_stratified_group_kfold(X, y: np.ndarray, groups: np.ndarray, n_splits: int, n_repeats: int, seed: int):
    """Yield (train_idx, test_idx) for n_repeats independent StratifiedGroupKFold passes."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    for r in range(n_repeats):
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for train_idx, test_idx in skf.split(X, y, groups):
            yield train_idx, test_idx


def stratified_subsample_for_fit_cap(
    X_tr: pd.DataFrame, y_tr: np.ndarray, cap: int, seed: int
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Stratified (class-preserving) random subsample of an already-safe training
    partition, used only to cap the cost of slow-to-fit models (e.g. SVC with
    probability=True) on Full-scale data. Never touches the test partition, so
    it does not reintroduce leakage -- it only reduces how much of the (already
    correctly split) training data a given model is fit on.
    """
    n = len(y_tr)
    if n <= cap:
        return X_tr, y_tr
    rng = np.random.default_rng(seed)
    y_tr = np.asarray(y_tr)
    vals, cnts = np.unique(y_tr, return_counts=True)
    frac = cap / n
    idx_parts = []
    for v, c in zip(vals, cnts):
        idx_v = np.where(y_tr == v)[0]
        take = max(1, int(round(c * frac)))
        take = min(take, len(idx_v))
        idx_parts.append(rng.choice(idx_v, size=take, replace=False))
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    return X_tr.iloc[idx].copy(), y_tr[idx].copy()


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
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)  # p == 1.0 belongs to the last bin (was silently dropped)
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
        arr = Xo[col].astype(float).to_numpy(copy=True)
        arr[idx] = arr[idx] + noise
        Xo[col] = arr
    return Xo


def induce_missingness_mcar_mar(X_df: pd.DataFrame, num_cols: List[str], base_p: float, extra_p: float, rng: np.random.Generator) -> pd.DataFrame:
    Xm = X_df.copy()
    if len(num_cols) == 0:
        return Xm
    mat = Xm[num_cols].astype(float).to_numpy(copy=True)
    mat[rng.random(mat.shape) < base_p] = np.nan
    Xm[num_cols] = mat
    anchor = num_cols[0]
    anchor_vals = Xm[anchor].astype(float).to_numpy(copy=True)
    q75 = np.nanquantile(anchor_vals, 0.75) if np.isfinite(anchor_vals).any() else np.nan
    if np.isfinite(q75):
        high = anchor_vals >= q75
        for col in num_cols:
            m2 = (rng.random(len(Xm)) < extra_p) & high
            tmp = Xm[col].astype(float).to_numpy(copy=True)
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
        # Signed log1p: identical to log1p(x) for non-negative values and defined for
        # negatives. (The previous version switched a column to a different transform,
        # x^2/(1+|x|), whenever the TEST FOLD happened to contain one negative value --
        # e.g. a single anion gap of -1 in hosp Full -- so the same world meant a
        # different perturbation in different folds.)
        g = np.sign(x) * np.log1p(np.abs(x))
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
        arr = Xs[col].astype(float).to_numpy(copy=True)
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
        arr = Xc[col].astype(float).to_numpy(copy=True)
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
            # fillna before stringifying: on newer pandas (StringDtype / infer_string),
            # `.astype(str)` on an object/string column can leave real missing values as
            # NaN/pd.NA instead of the literal "nan" string, which then makes np.unique's
            # sort crash comparing float NaN to str. Fill first so vals is uniformly str.
            vals = X[c].fillna("__MISSING__").astype(str).to_numpy()
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


# -----------------------------
# MLP with early stopping on the validation LOG-LOSS
# -----------------------------
# A plain MLPClassifier((256,128,64), max_iter=2500) with no early
# stopping overfits (clean AUROC, 5x3 grouped CV, retired checkpoint units:
# Full hosp 0.7240 / Full ED 0.7303, vs Logistic_L2 0.8565 / 0.7862). sklearn's built-in
# early_stopping=True is NOT a usable fix here: it monitors validation ACCURACY, which
# on an imbalanced outcome sits at the majority-class rate from the first epoch, so
# training stops after n_iter_no_change+1 epochs and keeps near-initial weights
# (sandbox diagnostic, 5x3 grouped CV on the landmark Demo files: Demo hosp AUROC
# 0.4389, Demo ED 0.4354). This wrapper keeps the same network/optimizer (adam, relu,
# lr 1e-3, alpha 1e-4, batch 'auto') but stops on the validation log-loss (a proper
# score) with patience n_iter_no_change and restores the best-epoch weights
# (Keras-style restore_best_weights).
# Final RSCE outputs (clean AUROC, 5x3 grouped CV): Full hosp 0.8920 / Full ED 0.8012
# (2nd and 1st of 7 models); Demo hosp 0.5359 / Demo ED 0.6239 (7th and 5th of 7 --
# at Demo size the 10% validation split holds only 1-2 deaths in hosp).
# The validation split is a stratified 10% of the TRAINING fold only (never the test
# fold), so there is no leakage into evaluation.
MODEL_PARAMS_CHANGED_SINCE_LEGACY = {"MLP"}  # checkpoint units without a model_params_sha


class EarlyStoppedMLP(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden_layer_sizes=(256, 128, 64), alpha=1e-4, learning_rate_init=1e-3,
                 batch_size="auto", max_iter=2500, validation_fraction=0.1, n_iter_no_change=10,
                 tol=1e-4, random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        idx = np.arange(len(y))
        try:
            tr, va = next(StratifiedShuffleSplit(n_splits=1, test_size=self.validation_fraction,
                                                 random_state=self.random_state).split(idx, y))
        except ValueError:  # a class with < 2 rows: fall back to an unstratified split
            tr, va = next(ShuffleSplit(n_splits=1, test_size=self.validation_fraction,
                                       random_state=self.random_state).split(idx))
        take = (lambda ix: X.iloc[ix]) if hasattr(X, "iloc") else (lambda ix: X[ix])
        X_tr, y_tr = take(tr), y[tr]
        X_va, y_va = take(va), y[va]
        mlp = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                            learning_rate_init=self.learning_rate_init, batch_size=self.batch_size,
                            tol=self.tol, random_state=self.random_state)
        best_loss, best_state, best_epoch, bad, epoch = np.inf, None, 0, 0, 0
        for epoch in range(1, int(self.max_iter) + 1):
            mlp.partial_fit(X_tr, y_tr, classes=self.classes_)  # one adam epoch
            loss = log_loss(y_va, mlp.predict_proba(X_va), labels=self.classes_)
            if loss < best_loss - self.tol:
                best_loss, best_epoch, bad = loss, epoch, 0
                best_state = ([c.copy() for c in mlp.coefs_], [b.copy() for b in mlp.intercepts_])
            else:
                bad += 1
                if bad >= self.n_iter_no_change:
                    break
        if best_state is not None:
            mlp.coefs_, mlp.intercepts_ = best_state
        self.mlp_ = mlp
        self.n_iter_ = int(best_epoch)          # epoch whose weights are kept
        self.n_epochs_run_ = int(epoch)         # epochs actually trained
        self.best_validation_loss_ = float(best_loss)
        return self

    def predict_proba(self, X):
        return self.mlp_.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def model_params_record(clf: Any) -> Dict[str, Any]:
    """JSON-safe description of an (unfitted) estimator: class + get_params()."""
    params = json.loads(json.dumps(clf.get_params(deep=False), sort_keys=True, default=repr))
    return {"class": type(clf).__name__, "params": params}


def model_params_sha(clf: Any) -> str:
    return hashlib.sha256(json.dumps(model_params_record(clf), sort_keys=True).encode("utf-8")).hexdigest()


def make_model_zoo(seed: int) -> Dict[str, Any]:
    return {
        "Logistic_L2": LogisticRegression(max_iter=5000, random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=1200, n_jobs=-1, random_state=seed, class_weight="balanced_subsample"),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=1200, n_jobs=-1, random_state=seed, class_weight="balanced_subsample"),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=700, learning_rate=0.03, random_state=seed),
        "SVC_RBF": SVC(kernel="rbf", probability=True, C=3.0, gamma="scale", random_state=seed),
        # was MLPClassifier((256,128,64), max_iter=2500) without early stopping -- see EarlyStoppedMLP
        "MLP": EarlyStoppedMLP(hidden_layer_sizes=(256, 128, 64), max_iter=2500, validation_fraction=0.1,
                               n_iter_no_change=10, random_state=seed),
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
SHAP_MIN_POSITIVES = 5
RSCE_COMPUTE_VERSION = "v2-2026-09-24"  # stratified SHAP rows, model-independent E seeds, E excludes prevalence_shift, fixed clean SHAP background, signed-log1p nonlinear world


def _select_shap_indices(n: int, max_samples: int, seed: int, y: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Rows of the test fold to explain. With y given, the draw is stratified and
    guarantees min(SHAP_MIN_POSITIVES, #positives) positive rows: with 2%
    prevalence a plain 50-row draw contains no positive at all ~36% of the time,
    in which case the label-conditional concept-drift world perturbs nothing and
    E is trivially 1 for it. Deterministic in `seed`; identical for every model
    (the caller passes a model-independent seed).
    """
    rng = np.random.default_rng(seed)
    k = min(max_samples, n)
    if y is None or k >= n:
        return np.sort(rng.choice(n, size=k, replace=False))
    y = np.asarray(y).astype(int)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_pos = int(round(k * len(pos) / n))
    n_pos = max(n_pos, min(SHAP_MIN_POSITIVES, len(pos)))
    n_pos = min(n_pos, len(pos), k)
    n_neg = min(k - n_pos, len(neg))
    idx = np.concatenate([rng.choice(pos, size=n_pos, replace=False), rng.choice(neg, size=n_neg, replace=False)])
    return np.sort(idx)


def _select_positive_class_shap(sv) -> np.ndarray:
    """
    Normalize a binary-classification SHAP explainer's output to shape
    (n_samples, n_features). Handles both conventions seen across shap
    versions:
      - a list of per-class arrays (older shap): sv[1] is the positive class.
      - a single stacked array (this environment's shap==0.51.0, observed
        for TreeExplainer on RandomForestClassifier/ExtraTreesClassifier --
        NOT for GradientBoostingClassifier or the Logistic_L2 LinearExplainer
        path, both of which already return 2D): shape
        (n_samples, n_features, n_classes) -- select the positive-class
        slice.

    BUG FIXED HERE: the previous code did `sv.reshape(sv.shape[0], -1)` on a
    3D array instead of indexing the class axis. That silently concatenates
    BOTH classes' SHAP values into one array of double width (n_features*2
    columns) rather than selecting one class's (n_features) columns. Visibly,
    this crashes `shap.summary_plot` with a shape mismatch against the data
    matrix (caught and logged, so it silently drops the SHAP plot for
    RandomForest/ExtraTrees rather than crashing the run). Invisibly, and
    more seriously, the SAME reshape corrupted `compute_shap_matrix_on_index`
    -- used by `explainability_stability_ablation` to compute the E
    (explainability-stability) component of RSCE_full -- for every
    RandomForest/ExtraTrees fold, with NO error raised (mismatched-but-equal
    array lengths across worlds let cosine/Spearman/top-k-Jaccard all run
    "successfully" on a garbled class0+class1-mixed feature axis). Any
    RSCE_full numbers for RandomForest/ExtraTrees computed with
    --compute_shap before this fix should be treated as unreliable and
    re-run.
    """
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv, dtype=float)
    if sv.ndim == 3:
        class_idx = min(1, sv.shape[-1] - 1)
        sv = sv[:, :, class_idx]
    return sv


# -----------------------------
# Parallel TreeSHAP for RandomForest / ExtraTrees
# -----------------------------
# shap's TreeExplainer is single-threaded (it holds the GIL; threads give no
# speed-up -- measured) and TreeSHAP on 1200 fully grown trees dominated the
# whole Full-scale benchmark (~90% of RSCE time, ~1 core busy on a many-core machine).
# A forest's probability is the MEAN of its trees, and TreeExplainer scales
# every tree by 1/n_trees, so SHAP values are exactly linear in the trees:
#     phi(forest) = sum_k (m_k / M) * phi(sub-forest_k)
# We split estimators_ into SHAP_N_JOBS contiguous sub-forests, explain each in
# its own worker process (each worker receives only its 1/k of the trees, so
# total memory stays ~one model), and recombine. Identical to the serial
# result up to floating-point summation order (verified, max |diff| ~1e-17).
SHAP_N_JOBS = 1  # set from --shap_n_jobs in run_benchmark()


def _tree_shap_chunk(sub_model, Xt_list: List[np.ndarray]) -> List[np.ndarray]:
    explainer = shap.TreeExplainer(sub_model)
    return [_select_positive_class_shap(explainer.shap_values(Xt, check_additivity=False)) for Xt in Xt_list]


def _forest_shap_parallel(model, Xt_list: List[np.ndarray], n_jobs: int) -> List[np.ndarray]:
    est = list(model.estimators_)
    M = len(est)
    k = max(1, min(int(n_jobs), M))
    bounds = np.linspace(0, M, k + 1).astype(int)
    subs, weights = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        sub = copy.copy(model)          # shallow: only estimators_ differs
        sub.estimators_ = est[a:b]
        sub.n_estimators = b - a
        subs.append(sub)
        weights.append((b - a) / M)
    parts = Parallel(n_jobs=len(subs))(delayed(_tree_shap_chunk)(sub, Xt_list) for sub in subs)
    out = []
    for i in range(len(Xt_list)):
        acc = None
        for w, part in zip(weights, parts):
            acc = w * part[i] if acc is None else acc + w * part[i]
        out.append(acc)
    return out


def _is_forest(model) -> bool:
    return isinstance(model, (RandomForestClassifier, ExtraTreesClassifier))


def _transform_for_shap(pipe: Pipeline, X_sub: pd.DataFrame) -> np.ndarray:
    Xt = pipe.named_steps["preprocess"].transform(X_sub)
    if "to_dense" in pipe.named_steps:
        Xt = pipe.named_steps["to_dense"].transform(Xt)
    return Xt


def compute_shap_matrix_on_index(
    pipe: Pipeline,
    X: pd.DataFrame,
    idx: np.ndarray,
    explainer_cache: Optional[Dict[str, object]] = None,
) -> np.ndarray:
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
        # Reuse one TreeExplainer per (fold, model) instead of rebuilding it for every
        # caller. shap.TreeExplainer(base_model) does its own (potentially large) internal
        # allocation from the tree structure alone -- e.g. a `thresholds` array of shape
        # (n_estimators, max_nodes) -- which does NOT depend on the data being explained.
        # explainability_stability_ablation() below calls this once for the clean baseline
        # and then once per non-trivial world (often ~7-8x per fold/model for a 1200-tree
        # RandomForest/ExtraTrees), so rebuilding the explainer every time was allocating
        # the same large structure repeatedly instead of once, and on a many-hour run this
        # was observed to eventually exhaust available memory
        # (numpy._core._exceptions._ArrayMemoryError: "Unable to allocate 297. MiB for an
        # array with shape (1200, 32433)") even though 297 MiB alone is modest -- long-running
        # Windows processes fragment memory, so repeated alloc/free of the same large block
        # eventually fails where a single alloc would not have. Caching fixes both the
        # redundant work and the redundant peak memory.
        if explainer_cache is not None and "tree_explainer" in explainer_cache:
            explainer = explainer_cache["tree_explainer"]
        else:
            explainer = shap.TreeExplainer(base_model)
            if explainer_cache is not None:
                explainer_cache["tree_explainer"] = explainer
        # check_additivity=False: TreeExplainer's internal sanity check (sum(phi) == model
        # output) is done in exact floating point and can fail by a tiny margin on large
        # forests (seen in practice on RandomForest/ExtraTrees with n_estimators=1200 and
        # class_weight="balanced_subsample") purely from FP accumulation error -- it is not
        # a sign the SHAP values themselves are wrong. Left enabled, a single borderline
        # sample can raise shap.utils._exceptions.ExplainerError and abort the ENTIRE
        # benchmark run (observed after ~14.6h / 30 of 140 tasks on a Full-scale probe).
        # We only use these values for relative feature-importance ranking (the E-component
        # ablation), which does not depend on exact additivity, so it's safe to skip the check.
        sv = explainer.shap_values(Xt, check_additivity=False)
        return _select_positive_class_shap(sv)

    # Linear / generic explainers use the data they are BUILT with as the
    # background (the "expected value" reference). Build them once, on the first
    # call -- the clean baseline in explainability_stability_ablation() -- and
    # reuse them for every world. (Previously each world's perturbed data became
    # its own background, which partly cancels the shift being measured, e.g. an
    # additive shift moved both the inputs and the reference.)
    if isinstance(base_model, LogisticRegression):
        if explainer_cache is not None and "linear_explainer" in explainer_cache:
            explainer = explainer_cache["linear_explainer"]
        else:
            explainer = shap.LinearExplainer(base_model, Xt, feature_perturbation="interventional")
            if explainer_cache is not None:
                explainer_cache["linear_explainer"] = explainer
        sv = explainer.shap_values(Xt)
        return _select_positive_class_shap(sv)

    if explainer_cache is not None and "generic_explainer" in explainer_cache:
        explainer = explainer_cache["generic_explainer"]
    else:
        masker = shap.maskers.Independent(Xt)
        explainer = shap.Explainer(base_model, masker)
        if explainer_cache is not None:
            explainer_cache["generic_explainer"] = explainer
    sv = explainer(Xt).values
    return _select_positive_class_shap(sv)


def topk_jaccard(a: np.ndarray, b: np.ndarray, k: int = 20) -> float:
    ia = set(np.argsort(-np.abs(a))[:k].tolist())
    ib = set(np.argsort(-np.abs(b))[:k].tolist())
    inter = len(ia & ib)
    union = len(ia | ib)
    return float(inter / union) if union > 0 else np.nan


# World kinds NOT used for E (explanation stability):
#  - clean: it is the reference.
#  - label_noise: changes labels only; features (hence SHAP) are untouched.
#  - prevalence_shift: RESAMPLES rows (with replacement, shuffled) and changes no
#    individual's features. The earlier code compared row i of the clean SHAP
#    matrix with row i of the resampled one -- i.e. two different patients --
#    which made E_cos for this world ~0.55 vs ~0.95 for the others (a pure
#    artifact). Per individual, explanations are unchanged by construction.
E_EXCLUDED_WORLD_KINDS = ("clean", "label_noise", "prevalence_shift")


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
    idx = _select_shap_indices(n=n, max_samples=shap_samples, seed=seed, y=y_clean)

    # Shared across every compute_shap_matrix_on_index() call below so a tree-based
    # explainer (the expensive one to construct) is built ONCE per (fold, model) and
    # reused for the clean baseline and every world, instead of once per call. See the
    # comment in compute_shap_matrix_on_index() for why this matters for memory too.
    explainer_cache: Dict[str, object] = {}

    # Perturbed copies of the SAME SHAP rows for every world used by E (same
    # seeds and order as before), then one SHAP pass over [clean] + worlds.
    X_sub0 = X_clean.iloc[idx].copy()
    world_X: List[pd.DataFrame] = []
    for spec in world_specs:
        if spec.kind in E_EXCLUDED_WORLD_KINDS:
            continue
        X_sub = X_clean.iloc[idx].copy()
        y_sub = y_clean[idx].copy()
        Xw, yw = apply_world(spec, X_sub, y_sub, num_cols=num_cols, cat_cols=cat_cols, seed=seed)
        world_X.append(Xw)

    clf = pipe.named_steps["clf"]
    base_model = clf.estimator if isinstance(clf, CalibratedClassifierCV) else clf
    if SHAP_AVAILABLE and _is_forest(base_model) and SHAP_N_JOBS > 1:
        Xt_list = [_transform_for_shap(pipe, X_sub0)] + [_transform_for_shap(pipe, Xw) for Xw in world_X]
        shap_all = _forest_shap_parallel(base_model, Xt_list, SHAP_N_JOBS)
    else:
        shap_all = [compute_shap_matrix_on_index(pipe, X_clean, idx, explainer_cache=explainer_cache)]
        for Xw in world_X:
            shap_all.append(compute_shap_matrix_on_index(pipe, Xw, np.arange(len(idx)), explainer_cache=explainer_cache))

    shapA = shap_all[0]
    gA = np.mean(np.abs(shapA), axis=0)

    cos_scores, rho_scores, jac_scores = [], [], []
    for shapW in shap_all[1:]:
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

    if _is_forest(base_model) and SHAP_N_JOBS > 1:
        return _forest_shap_parallel(base_model, [Xt], SHAP_N_JOBS)[0]

    tree_like = (RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier)
    if isinstance(base_model, tree_like):
        explainer = shap.TreeExplainer(base_model)
        # See compute_shap_matrix_on_index() above for why check_additivity=False is needed here.
        sv = explainer.shap_values(Xt, check_additivity=False)
        return _select_positive_class_shap(sv)

    if isinstance(base_model, LogisticRegression):
        explainer = shap.LinearExplainer(base_model, Xt, feature_perturbation="interventional")
        sv = explainer.shap_values(Xt)
        return _select_positive_class_shap(sv)

    masker = shap.maskers.Independent(Xt)
    explainer = shap.Explainer(base_model, masker)
    sv = explainer(Xt).values
    return _select_positive_class_shap(sv)


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
    if not np.isfinite(auc_clean):
        return {"S_drop": np.nan, "S_ratio": np.nan}
    auc_worlds = [a for a in auc_worlds if np.isfinite(a)]  # max(0, nan) would silently count as "no drop"
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
# -----------------------------
# Checkpoint / resume (per fold x model unit)
# -----------------------------
# A Full-scale run can take weeks. Without this, any interruption (power cut,
# Windows Update restart, out-of-memory) loses everything. Each (fold, model)
# unit's outputs are written atomically to <outdir>/_checkpoint/ as soon as the
# unit finishes; re-running the SAME command skips finished units and reloads
# their saved outputs. Every unit's computation depends only on the fold split
# (seeded), the model (fixed random_state) and seeds derived from fold_id/model
# name -- never on which units ran before it -- so a resumed run produces the
# same numbers as an uninterrupted one. The run config (data file hash, all
# result-affecting settings, package versions) is stored with the checkpoint;
# a mismatch aborts instead of silently mixing results from different setups.
def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _unit_file(ckpt_dir: Path, fold_id: int, mname: str) -> Path:
    return ckpt_dir / f"fold{fold_id:03d}__{mname}.pkl"


def _ckpt_save(ckpt_dir: Path, fold_id: int, mname: str, payload: Dict[str, Any]) -> None:
    final = _unit_file(ckpt_dir, fold_id, mname)
    tmp = final.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(20):  # atomic: a killed process never leaves a half-written unit
        try:
            os.replace(tmp, final)
            break
        except PermissionError:
            # Windows: antivirus/indexer can briefly hold the target file open
            if attempt == 19:
                raise
            time.sleep(0.5 * (attempt + 1))


def _ckpt_load(ckpt_dir: Path, fold_id: int, mname: str) -> Optional[Dict[str, Any]]:
    fp = _unit_file(ckpt_dir, fold_id, mname)
    if not fp.exists():
        return None
    with open(fp, "rb") as f:
        return pickle.load(f)


def _unit_is_stale(unit: Dict[str, Any], mname: str, current_sha: str) -> bool:
    """A unit is stale when it was fitted with other model settings than the current zoo.
    Units written before 2026-09-29 carry no model_params_sha; of those only the models
    whose settings changed since then (MODEL_PARAMS_CHANGED_SINCE_LEGACY) are stale."""
    saved = unit.get("model_params_sha")
    if saved is None:
        return mname in MODEL_PARAMS_CHANGED_SINCE_LEGACY
    return saved != current_sha


def _ckpt_retire(ckpt_dir: Path, fold_id: int, mname: str) -> None:
    """Move a stale unit into <ckpt>/replaced_units/ (kept for reference, never loaded)."""
    src = _unit_file(ckpt_dir, fold_id, mname)
    dst_dir = ckpt_dir / "replaced_units"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    k = 1
    while dst.exists():
        dst = dst_dir / f"{src.stem}.old{k}{src.suffix}"
        k += 1
    os.replace(src, dst)


def _ckpt_prepare(ckpt_dir: Path, config: Dict[str, Any]) -> None:
    ensure_dir(ckpt_dir)
    cfg_path = ckpt_dir / "checkpoint_config.json"
    code_sha = config.pop("_code_sha256", None)
    if cfg_path.exists():
        saved = json.loads(cfg_path.read_text(encoding="utf-8"))
        saved_code = saved.pop("_code_sha256", None)
        if saved != config:
            diff = sorted(k for k in set(saved) | set(config) if saved.get(k) != config.get(k))
            raise RuntimeError(
                f"[run_rsce] Checkpoint in {ckpt_dir} was made with DIFFERENT settings/data/packages "
                f"(differs in: {diff}). Refusing to mix results. Use a new --outdir, or delete "
                f"{ckpt_dir} to start over."
            )
        if code_sha and saved_code and code_sha != saved_code:
            print("[run_rsce] NOTE: run_rsce.py has changed since this checkpoint was created. "
                  "Resuming (settings/data/packages match). Units of models whose settings changed "
                  "are detected per unit and recomputed; all other units are reused.")
    else:
        to_write = dict(config)
        to_write["_code_sha256"] = code_sha
        cfg_path.write_text(json.dumps(to_write, indent=2), encoding="utf-8")


def run_benchmark(args: argparse.Namespace) -> None:
    global SHAP_N_JOBS
    SHAP_N_JOBS = max(1, int(args.shap_n_jobs)) if args.shap_n_jobs >= 1 else max(1, (os.cpu_count() or 1) + 1 + args.shap_n_jobs)
    print(f"[run_rsce] TreeSHAP worker processes for RandomForest/ExtraTrees: {SHAP_N_JOBS} "
          f"(cpu_count={os.cpu_count()})")
    set_global_seed(args.seed)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset.")
    df = df.dropna(subset=[args.target]).copy()
    df[args.target] = df[args.target].astype(int)

    # ---- group-aware CV setup (see choose_safe_group_folds / repeated_stratified_group_kfold) ----
    groups_all: Optional[np.ndarray] = None
    if args.cv_mode == "group":
        if args.group_col in df.columns:
            groups_all = df[args.group_col].to_numpy()
        else:
            print(
                f"[run_rsce] WARNING: --group_col '{args.group_col}' not found in {args.data}; "
                f"falling back to row-level CV (patient-overlap-across-folds is NOT prevented). "
                f"Pass --cv_mode row explicitly to silence this warning."
            )

    # ---- estimate_only: probe with a minimal (2-fold, 1-repeat) CV, then extrapolate ----
    orig_folds, orig_repeats = args.folds, args.repeats
    if args.estimate_only:
        args.folds, args.repeats = max(2, min(2, orig_folds)), 1
        print(f"[run_rsce] --estimate_only: probing with folds=2, repeats=1 "
              f"(will extrapolate to the requested folds={orig_folds}, repeats={orig_repeats}).")

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

    # ---- resolve safe fold count for the CV mode actually in effect ----
    if groups_all is not None:
        safe_folds = choose_safe_group_folds(y, groups_all, args.folds)
        if safe_folds == 0:
            print("[run_rsce] WARNING: not enough distinct groups per class for group-aware CV "
                  "(too few patients in the minority class); falling back to row-level StratifiedKFold.")
            groups_all = None
        elif safe_folds < args.folds:
            print(f"[run_rsce] WARNING: reducing folds from {args.folds} to {safe_folds} "
                  f"(limited by --group_col '{args.group_col}' group/class counts).")
            args.folds = safe_folds
    if groups_all is None:
        safe_folds = choose_safe_group_folds(y, None, args.folds)
        if safe_folds == 0:
            raise ValueError(f"Not enough samples in the minority class of '{args.target}' to run any CV split.")
        if safe_folds < args.folds:
            print(f"[run_rsce] WARNING: reducing folds from {args.folds} to {safe_folds} (minority class too small).")
            args.folds = safe_folds

    world_specs = build_world_specs()
    model_zoo = make_model_zoo(seed=args.seed)
    if args.exclude_models:
        excluded = set(args.exclude_models)
        model_zoo = {m: clf for m, clf in model_zoo.items() if m not in excluded}
        print(f"[run_rsce] Excluding models: {sorted(excluded)}. Remaining: {sorted(model_zoo.keys())}")
    pipelines: Dict[str, Pipeline] = {
        m: make_pipeline(m, clf, num_cols, cat_cols, args.calibrate) for m, clf in model_zoo.items()
    }

    pkgs = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "shap"]
    versions = get_versions(pkgs)
    model_params = {m: model_params_record(clf) for m, clf in model_zoo.items()}
    model_sha = {m: model_params_sha(clf) for m, clf in model_zoo.items()}

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
        "cv_mode": "group" if groups_all is not None else "row",
        "group_col": args.group_col if groups_all is not None else None,
        "excluded_models": sorted(args.exclude_models) if args.exclude_models else [],
        "model_params": model_params,
        "svc_max_train_n": args.svc_max_train_n,
        "estimate_only": bool(args.estimate_only),
        "requested_folds_repeats": [orig_folds, orig_repeats],
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

    ckpt_dir: Optional[Path] = None
    if args.checkpoint and not args.estimate_only:
        ckpt_dir = outdir / "_checkpoint"
        ckpt_config = {
            "data_sha256": _sha256_file(args.data),
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
            "shap_allow_generic": bool(args.shap_allow_generic),
            "E_topk": args.E_topk,
            "cv_mode": "group" if groups_all is not None else "row",
            "group_col": args.group_col if groups_all is not None else None,
            "models": list(pipelines.keys()),
            "svc_max_train_n": args.svc_max_train_n,
            "reliability_models": list(args.reliability_models),
            "reliability_worlds": list(args.reliability_worlds),
            "reliability_bins": args.reliability_bins,
            "reliability_adaptive": bool(args.reliability_adaptive),
            "worlds": [w.__dict__ for w in world_specs],
            "python": platform.python_version(),
            "package_versions": versions,
            # Bump when a change alters what a finished unit contains, so checkpoints
            # written by older code are refused instead of being mixed in.
            "compute_version": RSCE_COMPUTE_VERSION,
            "_code_sha256": _sha256_file(__file__),
        }
        _ckpt_prepare(ckpt_dir, ckpt_config)
        n_done = len(list(ckpt_dir.glob("fold*__*.pkl")))
        if n_done:
            print(f"[run_rsce] Resuming: {n_done} finished (fold, model) unit(s) found in {ckpt_dir}; "
                  f"they will be loaded instead of recomputed (units made with different model "
                  f"settings are recomputed and the old file is kept in {ckpt_dir / 'replaced_units'}).")

    if groups_all is not None:
        split_iterable_factory = lambda: repeated_stratified_group_kfold(
            X, y, groups_all, n_splits=args.folds, n_repeats=args.repeats, seed=args.seed
        )
    else:
        rskf = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed)
        split_iterable_factory = lambda: rskf.split(X, y)

    records = []
    cost_records = []
    shap_records = []
    reliability_records = []

    # Progress accounting (for one overall progress bar)
    n_folds_total = args.folds * args.repeats
    n_models = len(pipelines)
    n_worlds = len(world_specs)
    n_tasks_total = n_folds_total * n_models * n_worlds  # eval tasks per (fold, model, world)
    # SHAP is optional and not counted strictly; tqdm ETA still works well enough.

    fold_id = 0

    with tqdm(total=n_tasks_total, desc="RSCE Benchmark (eval)", unit="task", dynamic_ncols=True) as pbar:
        for train_idx, test_idx in split_iterable_factory():
            fold_id += 1
            X_tr, y_tr = X.iloc[train_idx].copy(), y[train_idx].copy()
            X_te0, y_te0 = X.iloc[test_idx].copy(), y[test_idx].copy()

            # Nested bar for models in this fold (visual step indicator)
            with tqdm(total=n_models, desc=f"Fold {fold_id}/{n_folds_total} (models)", unit="model", leave=False, dynamic_ncols=True) as pbar_models:
                for mname, pipe_template in pipelines.items():
                    pbar_models.set_postfix_str(mname)

                    if ckpt_dir is not None:
                        saved_unit = _ckpt_load(ckpt_dir, fold_id, mname)
                        if saved_unit is not None and _unit_is_stale(saved_unit, mname, model_sha[mname]):
                            _ckpt_retire(ckpt_dir, fold_id, mname)
                            tqdm.write(f"[run_rsce] fold {fold_id} {mname}: checkpoint unit was made with "
                                       f"different model settings -> recomputing it.")
                            saved_unit = None
                        if saved_unit is not None:
                            records.extend(saved_unit["records"])
                            reliability_records.extend(saved_unit["reliability"])
                            shap_records.extend(saved_unit["shap"])
                            cost_records.extend(saved_unit["cost"])
                            pbar.update(n_worlds)
                            pbar_models.update(1)
                            continue
                    n_rec0, n_rel0 = len(records), len(reliability_records)
                    n_shap0, n_cost0 = len(shap_records), len(cost_records)

                    # Fresh, unfitted copy per (fold, model) unit, deleted at the end of the
                    # unit. (Refitting the shared template kept every model's previous fit
                    # alive in `pipelines` -- on ED Full a fitted 1200-tree ExtraTrees/RF is
                    # ~10-30 GB -- so peak memory held one fitted forest per model at once.)
                    pipe = clone(pipe_template)

                    X_tr_fit, y_tr_fit = X_tr, y_tr
                    if mname == "SVC_RBF" and args.svc_max_train_n and len(y_tr) > args.svc_max_train_n:
                        X_tr_fit, y_tr_fit = stratified_subsample_for_fit_cap(
                            X_tr, y_tr, cap=args.svc_max_train_n, seed=args.seed + fold_id
                        )
                        tqdm.write(
                            f"[run_rsce] SVC_RBF fold {fold_id}: capped training set "
                            f"{len(y_tr):,} -> {len(y_tr_fit):,} rows (--svc_max_train_n={args.svc_max_train_n})."
                        )

                    t0 = time.perf_counter()
                    pipe.fit(X_tr_fit, y_tr_fit)
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

                            # Update progress
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
                                # SAME seed for every model: identical SHAP rows and identical
                                # world perturbations, so E is paired across models (the
                                # metric worlds already use a model-independent seed).
                                seed=args.seed + 999 * fold_id,
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
                                    seed=args.seed + 123,
                                    y=y_te0,
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

                    if ckpt_dir is not None:
                        _ckpt_save(ckpt_dir, fold_id, mname, {
                            "records": records[n_rec0:],
                            "reliability": reliability_records[n_rel0:],
                            "shap": shap_records[n_shap0:],
                            "cost": cost_records[n_cost0:],
                            "model_params_sha": model_sha[mname],
                        })

                    del pipe
                    est = base = None  # drop the last references to the fitted estimator

                    # Proactively release this model/fold's fitted estimator, SHAP
                    # explainer(s), and any lingering transformed-feature arrays before
                    # moving to the next model. On a many-hour run with large forests
                    # (n_estimators=1200) this keeps peak memory from ratcheting up across
                    # models/folds instead of being reclaimed between them -- see the
                    # memory-allocation-failure note in compute_shap_matrix_on_index().
                    gc.collect()

                    pbar_models.update(1)

    metrics_df = pd.DataFrame(records)
    metrics_df.to_csv(outdir / "metrics_per_fold.csv", index=False)

    cost_df = pd.DataFrame(cost_records)
    cost_df.to_csv(outdir / "compute_cost_per_fold.csv", index=False)

    cost_agg = cost_df.groupby("model")[["time_fit_s", "time_pred_all_worlds_s", "time_shap_s", "time_total_s"]].agg(["mean", "std"]).reset_index()
    cost_agg.columns = ["_".join([c for c in col if c]) for col in cost_agg.columns.values]
    cost_agg.to_csv(outdir / "compute_cost.csv", index=False)

    if args.estimate_only:
        target_total_folds = orig_folds * orig_repeats
        per_model_mean_s = cost_df.groupby("model")["time_total_s"].mean()
        est_rows = []
        for m, mean_s in per_model_mean_s.items():
            est_rows.append({
                "model": m,
                "mean_seconds_per_fold_observed": float(mean_s),
                "estimated_total_seconds_for_requested_folds_x_repeats": float(mean_s * target_total_folds),
            })
        est_df = pd.DataFrame(est_rows).sort_values(
            "estimated_total_seconds_for_requested_folds_x_repeats", ascending=False
        )
        est_df["estimated_total_hours"] = est_df["estimated_total_seconds_for_requested_folds_x_repeats"] / 3600.0
        grand_total_s = float(est_df["estimated_total_seconds_for_requested_folds_x_repeats"].sum())
        est_df.to_csv(outdir / "estimate_timing.csv", index=False)
        print("\n[ESTIMATE] Per-model extrapolated runtime for the FULL requested run "
              f"(folds={orig_folds}, repeats={orig_repeats}, {target_total_folds} folds total):")
        print(est_df.to_string(index=False))
        print(f"\n[ESTIMATE] Grand total (sum across models, worlds/SHAP included as probed): "
              f"~{grand_total_s/3600.0:.2f} hours (~{grand_total_s/86400.0:.2f} days).")
        print("[ESTIMATE] This is a rough linear extrapolation from a 2-fold probe -- real Full-scale runtime")
        print("[ESTIMATE] can differ (tree/SVC fit cost is often superlinear in n). Consider --exclude_models,")
        print("[ESTIMATE] --svc_max_train_n, --shap_samples, or --no-compute_shap if this is too slow.")
        print(f"[ESTIMATE] Wrote: {outdir / 'estimate_timing.csv'}")
        print("[ESTIMATE] NOTE: this run's other output files (rsce_scores.csv etc.) reflect only the 2-fold")
        print("[ESTIMATE] probe, NOT the full requested folds/repeats -- re-run without --estimate_only for real results.")
        return

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
            row[c + "_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")  # all-NaN: E of no-SHAP models
            lo, hi = bootstrap_ci(vals, n_boot=args.ci_boot, ci=0.95, seed=args.seed + deterministic_int_hash(model + c))
            row[c + "_ci_lo"] = lo
            row[c + "_ci_hi"] = hi
        sum_rows.append(row)
    ab_sum = pd.DataFrame(sum_rows)
    ab_sum.to_csv(outdir / "ablation_summary.csv", index=False)

    # ------------------------------------------------------------------
    # RSCE scoring
    # ------------------------------------------------------------------
    # E (explanation stability) exists only for models SHAP can explain here
    # (RandomForest, ExtraTrees, GradientBoosting, Logistic_L2). The original
    # RSCE formulation filled a missing E with 0, which silently subtracted
    # about wE*E (~0.09 with wE=0.1, E~0.9) from SVC_RBF, MLP and GaussianNB --
    # enough to decide rankings, for a reason unrelated to robustness.
    # Default now (--E_missing renormalize): a model without E is scored on the
    # components it has, with their weights renormalized to sum to 1, i.e.
    # (wR*R + wS*S + wC*C) / (wR + wS + wC) -- the standard available-case rule
    # for weighted composite indices. Always reported alongside, for every model:
    #   RSCE_RSC    -- R, S, C only with the same formula for all 7 models
    #                  (the strictly like-for-like comparison), and
    #   RSCE_legacy -- the original RSCE formulation (S_ratio, C_exp, missing E -> 0).
    # Primary S/C formulations: --primary_S (default S_ratio), --primary_C
    # (default C_linear; C_exp = exp(-|dECE|/ECE_clean) rewards models that are
    # badly calibrated to begin with and depends on test-set size through the
    # finite-sample bias of ECE_clean, so it is kept only as a sensitivity variant).
    wsum = args.wR + args.wS + args.wC + args.wE
    W = {"R": args.wR / wsum, "S": args.wS / wsum, "C": args.wC / wsum, "E": args.wE / wsum}
    has_E_col = "E_mix" in comp_fold_df.columns

    def compute_score(dfm: pd.DataFrame, Scol: str, Ccol: str, Ecol: Optional[str], e_missing: str) -> pd.Series:
        rsc = W["R"] * dfm["R"].astype(float) + W["S"] * dfm[Scol].astype(float) + W["C"] * dfm[Ccol].astype(float)
        rsc_only = rsc / (W["R"] + W["S"] + W["C"])
        if Ecol is None or Ecol not in dfm.columns:
            return rsc_only
        E = dfm[Ecol].astype(float)
        if e_missing == "zero":
            return rsc + W["E"] * E.fillna(0.0)
        return (rsc + W["E"] * E).where(E.notna(), rsc_only)

    mean_cols = {c: c.replace("_mean", "") for c in ab_sum.columns if c.endswith("_mean")}
    mean_df = pd.DataFrame({"model": ab_sum["model"]})
    for c_mean, c_raw in mean_cols.items():
        mean_df[c_raw] = ab_sum[c_mean].values

    E_list = ["E_mix", "E_cos", "E_rank", "E_jaccard"] if has_E_col else []
    score_variants: Dict[str, pd.Series] = {}
    for Scol in ["S_drop", "S_ratio"]:
        for Ccol in ["C_exp", "C_linear"]:
            for Ecol in E_list:
                score_variants[f"{Scol}|{Ccol}|{Ecol}"] = compute_score(mean_df, Scol, Ccol, Ecol, args.E_missing)
            score_variants[f"{Scol}|{Ccol}|noE"] = compute_score(mean_df, Scol, Ccol, None, args.E_missing)
    legacy_key = "legacy:S_ratio|C_exp|E_mix|missingE=0"
    if has_E_col:
        score_variants[legacy_key] = compute_score(mean_df, "S_ratio", "C_exp", "E_mix", "zero")

    ref_key = f"{args.primary_S}|{args.primary_C}|{'E_mix' if has_E_col else 'noE'}"
    ref_rank = score_variants[ref_key].rank(ascending=False, method="average")

    agree_rows = []
    for k, sc in score_variants.items():
        r = sc.rank(ascending=False, method="average")
        rho, _ = spearmanr(ref_rank.values, r.values)
        tau, _ = kendalltau(ref_rank.values, r.values)
        agree_rows.append({"variant": k, "spearman_rank_vs_ref": float(rho), "kendall_rank_vs_ref": float(tau),
                           "is_reference": bool(k == ref_key)})
    agree_df = pd.DataFrame(agree_rows).sort_values("spearman_rank_vs_ref", ascending=False)
    agree_df.to_csv(outdir / "ablation_rank_agreement.csv", index=False)

    main_variant = ref_key
    rsce_out = mean_df[["model"]].copy()
    rsce_out["RSCE_full"] = score_variants[main_variant].values
    rsce_out["has_E"] = (mean_df["E_mix"].notna().astype(int).values if has_E_col else 0)
    rsce_out["RSCE_RSC"] = score_variants[f"{args.primary_S}|{args.primary_C}|noE"].values
    if has_E_col:
        rsce_out["RSCE_legacy"] = score_variants[legacy_key].values
    rsce_out = rsce_out.sort_values("RSCE_full", ascending=False)
    rsce_out.to_csv(outdir / "rsce_scores.csv", index=False)

    # Per-fold scores (same definition as RSCE_full) -- also consumed by compare_pro.py
    fold_scores = comp_fold_df.copy()
    fold_scores["RSCE_fold"] = compute_score(
        fold_scores, args.primary_S, args.primary_C, "E_mix" if has_E_col else None, args.E_missing).values
    fold_scores["RSCE_RSC_fold"] = compute_score(fold_scores, args.primary_S, args.primary_C, None, args.E_missing).values
    fold_scores.to_csv(outdir / "rsce_per_fold.csv", index=False)

    # Paired model-vs-model tests across the folds x repeats.
    # PRIMARY: Nadeau & Bengio (2003) corrected resampled t-test. The 15 folds of
    # a 5x3 repeated CV share training data, so the plain paired t-test and
    # Wilcoxon (kept for reference) treat correlated folds as independent and are
    # anti-conservative. Corrected variance = (1/J + n_test/n_train) * s^2 with
    # n_test/n_train = 1/(K-1) for K-fold CV, J = folds*repeats, df = J-1.
    ratio_test_train = 1.0 / max(args.folds - 1, 1)
    models = sorted(fold_scores["model"].unique().tolist())
    pairs = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            f1 = fold_scores.loc[fold_scores["model"] == m1, ["fold", "RSCE_fold"]]
            f2 = fold_scores.loc[fold_scores["model"] == m2, ["fold", "RSCE_fold"]]
            mm = f1.merge(f2, on="fold", suffixes=("_1", "_2")).dropna()
            s1 = mm["RSCE_fold_1"].to_numpy(dtype=float)
            s2 = mm["RSCE_fold_2"].to_numpy(dtype=float)
            d = s1 - s2
            J = len(d)
            if J >= 2 and np.std(d, ddof=1) > 0:
                var_corr = (1.0 / J + ratio_test_train) * float(np.var(d, ddof=1))
                nb_t = float(np.mean(d) / np.sqrt(var_corr))
                nb_p = float(2.0 * student_t.sf(abs(nb_t), df=J - 1))
            else:
                nb_t, nb_p = np.nan, np.nan
            try:
                w_stat, w_p = wilcoxon(s1, s2, zero_method="wilcox", alternative="two-sided")
            except Exception:
                w_stat, w_p = np.nan, np.nan
            t_stat, t_p = ttest_rel(s1, s2, nan_policy="omit") if J >= 2 else (np.nan, np.nan)
            pairs.append({
                "m1": m1, "m2": m2, "n_folds": int(J),
                "mean_diff_m1_minus_m2": float(np.mean(d)) if J else np.nan,
                "nb_corrected_t": nb_t, "nb_corrected_p": nb_p,
                "wilcoxon_p": float(w_p), "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else np.nan,
                "ttest_p": float(t_p), "ttest_stat": float(t_stat),
            })

    pairs_df = pd.DataFrame(pairs)
    if len(pairs_df):
        pairs_df["nb_corrected_p_holm"] = holm_correction(pairs_df["nb_corrected_p"].fillna(1.0))
        pairs_df["wilcoxon_p_holm"] = holm_correction(pairs_df["wilcoxon_p"].fillna(1.0))
        pairs_df["ttest_p_holm"] = holm_correction(pairs_df["ttest_p"].fillna(1.0))
    pairs_df.to_csv(outdir / "paired_tests.csv", index=False)

    # record the scoring definition next to the run settings
    try:
        sch = json.loads((outdir / "schema.json").read_text(encoding="utf-8"))
        sch["scoring"] = {
            "weights_normalized": W, "primary_S": args.primary_S, "primary_C": args.primary_C,
            "E_missing": args.E_missing, "reference_variant": main_variant,
            "E_excluded_world_kinds": list(E_EXCLUDED_WORLD_KINDS),
            "paired_test_primary": "Nadeau-Bengio corrected resampled t-test, Holm-adjusted",
        }
        (outdir / "schema.json").write_text(json.dumps(sch, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[run_rsce] WARNING: could not add scoring info to schema.json: {e}")

    # Sensitivity plots
    focus_models = args.focus_models or ["RandomForest", "ExtraTrees", "GradientBoosting"]
    base_w = np.array([W["R"], W["S"], W["C"], W["E"]], dtype=float)
    ratio_rest = base_w[1:] / max(base_w[1:].sum(), 1e-12)

    wR_values = np.linspace(args.wR_min, args.wR_max, args.wR_steps)
    comp_use = mean_df.set_index("model")
    comp_use = comp_use.loc[[m for m in focus_models if m in comp_use.index]]

    rsce_sweep = {m: [] for m in comp_use.index}
    for wR2 in wR_values:
        rem = 1.0 - float(wR2)
        wS2, wC2, wE2 = rem * ratio_rest
        for m in comp_use.index:
            Rm = float(comp_use.loc[m, "R"])
            Sm = float(comp_use.loc[m, args.primary_S])
            Cm = float(comp_use.loc[m, args.primary_C])
            Em = float(comp_use.loc[m, "E_mix"]) if "E_mix" in comp_use.columns else np.nan
            if np.isfinite(Em):
                val = wR2 * Rm + wS2 * Sm + wC2 * Cm + wE2 * Em
            elif args.E_missing == "zero":
                val = wR2 * Rm + wS2 * Sm + wC2 * Cm
            else:
                val = (wR2 * Rm + wS2 * Sm + wC2 * Cm) / max(wR2 + wS2 + wC2, 1e-12)
            rsce_sweep[m].append(float(val))

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

    print("\n[Done] RSCE benchmark completed.")
    print(f"Outputs saved in: {outdir.resolve()}")
    print("Key outputs: rsce_scores.csv, paired_tests.csv, compute_cost.csv, ablation_summary.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RSCE multi-world benchmark (ablations, realism worlds, calibration, tests, cost, env).")

    p.add_argument("--data", type=str, default="analytic_dataset_mortality_all_admissions.csv", help="Path to CSV dataset.")
    p.add_argument("--target", type=str, default="label_mortality", help="Binary target column name (0/1).")
    p.add_argument("--outdir", type=str, default="rsce_results_full_dataset", help="Output directory.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--ci_boot", type=int, default=2000)

    # None is not listed in choices; it is the default (no calibration).
    p.add_argument("--calibrate", type=str, default=None, choices=["sigmoid", "isotonic"], help="Calibration method (optional).")

    # NOTE: originally `action="store_true", default=True`, which meant --compute_shap
    # could never actually be turned off from the CLI (passing or omitting the flag both
    # left it True). Fixed with BooleanOptionalAction so `--no-compute_shap` now works --
    # useful to skip the most expensive step on a Full-scale run.
    p.add_argument("--compute_shap", action=argparse.BooleanOptionalAction, default=True,
                    help="Compute SHAP-based explainability stability (E). Use --no-compute_shap to skip "
                         "the most expensive step of the benchmark on large datasets.")
    p.add_argument("--shap_samples", type=int, default=250)
    p.add_argument("--shap_n_jobs", type=int, default=8,
                    help="Worker processes for TreeSHAP of RandomForest/ExtraTrees (the dominant cost). The forest is "
                         "split into this many sub-forests explained in parallel and recombined exactly. "
                         "1 = serial (old behaviour). Does not change results beyond float rounding.")
    p.add_argument("--shap_allow_generic", action="store_true")
    p.add_argument("--E_topk", type=int, default=20, help="Top-k for Jaccard feature stability.")
    p.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=True,
                    help="Save each finished (fold, model) unit to <outdir>/_checkpoint and skip finished "
                         "units when the same command is re-run (resume after a crash). Default on; "
                         "--no-checkpoint disables it. Ignored with --estimate_only.")

    p.add_argument("--group_col", type=str, default="subject_id",
                    help="Column identifying the patient/subject for group-aware CV (prevents the same "
                         "patient's admissions/stays from appearing in both train and test).")
    p.add_argument("--cv_mode", type=str, default="group", choices=["group", "row"],
                    help="'group' (default): StratifiedGroupKFold keyed on --group_col, so no patient is "
                         "split across train/test. 'row': original row-level RepeatedStratifiedKFold "
                         "(kept for sensitivity-analysis comparison against the group-aware result).")
    p.add_argument("--exclude_models", type=str, nargs="*", default=[],
                    help="Model names to skip entirely (e.g. --exclude_models SVC_RBF MLP) -- useful to cut "
                         "compute cost on Full-scale data.")
    p.add_argument("--svc_max_train_n", type=int, default=None,
                    help="If set and a training fold exceeds this many rows, SVC_RBF is fit on a stratified "
                         "subsample of this size instead (evaluation is still on the full test fold). SVC with "
                         "probability=True scales very poorly (roughly O(n^2)-O(n^3)); this caps its cost on "
                         "Full-scale data without dropping the model entirely. No effect on other models.")
    p.add_argument("--estimate_only", action="store_true",
                    help="Run a minimal 2-fold, 1-repeat probe (with the full requested model zoo / worlds / "
                         "SHAP settings) and print/save an extrapolated total runtime for the folds/repeats "
                         "you actually requested, then exit. Use this before committing to a long Full-scale run.")

    p.add_argument("--drop_cols", type=str, nargs="*", default=[
        "hadm_id", "subject_id", "discharge_location", "anchor_year", "anchor_year_group"
    ])

    p.add_argument("--numeric_cols", type=str, nargs="*", default=None)
    p.add_argument("--categorical_cols", type=str, nargs="*", default=None)

    p.add_argument("--wR", type=float, default=0.4)
    p.add_argument("--wS", type=float, default=0.3)
    p.add_argument("--wC", type=float, default=0.2)
    p.add_argument("--wE", type=float, default=0.1)
    p.add_argument("--primary_S", type=str, default="S_ratio", choices=["S_ratio", "S_drop"],
                    help="S formulation used in RSCE_full (the other is reported as a sensitivity variant).")
    p.add_argument("--primary_C", type=str, default="C_linear", choices=["C_linear", "C_exp"],
                    help="C formulation used in RSCE_full. C_linear = 1 - mean|dECE| (absolute calibration drift). "
                         "C_exp = exp(-mean|dECE|/ECE_clean) is the original RSCE choice; it rewards models "
                         "that are already badly calibrated, so it is kept only as a sensitivity variant.")
    p.add_argument("--E_missing", type=str, default="renormalize", choices=["renormalize", "zero"],
                    help="How RSCE_full treats models without an E component (no SHAP support): 'renormalize' "
                         "(default) scores them on R,S,C with weights renormalized to sum to 1; 'zero' reproduces "
                         "the original RSCE formulation (E = 0). RSCE_legacy always reports the latter.")

    p.add_argument("--focus_models", type=str, nargs="*", default=None)
    p.add_argument("--wR_min", type=float, default=0.30)
    p.add_argument("--wR_max", type=float, default=0.50)
    p.add_argument("--wR_steps", type=int, default=7)

    p.add_argument("--reliability_models", type=str, nargs="*", default=["Logistic_L2", "RandomForest"])
    p.add_argument("--reliability_worlds", type=str, nargs="*", default=["WA_clean", "WI_prevalence_shift"])
    p.add_argument("--reliability_bins", type=int, default=15)
    p.add_argument("--reliability_adaptive", action="store_true")

    args = p.parse_args()  # strict: a mistyped flag (e.g. --shap-samples) must be an error, not silently ignored
    return args


def main() -> None:
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()

