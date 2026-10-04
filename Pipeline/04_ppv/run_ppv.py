# run_ppv.py  (prevalence-standardized PPV benchmark, one dataset per call --
#               see the "CLI entry point" section near the bottom for usage)

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import argparse
import hashlib
import json
import os
import pickle
import platform
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, StratifiedGroupKFold, StratifiedShuffleSplit, ShuffleSplit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# ---------- Optional progress bar ----------
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# -----------------------------
# Utilities: progress + ETA
# -----------------------------
def _fmt_seconds(sec: float) -> str:
    if sec is None or not np.isfinite(sec):
        return "?"
    sec = max(0.0, float(sec))
    h = int(sec // 3600); sec -= 3600 * h
    m = int(sec // 60);   sec -= 60 * m
    s = int(sec)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def _now() -> float:
    return time.perf_counter()

class RunningETA:
    """Simple running average ETA based on completed steps."""
    def __init__(self, total_steps: int):
        self.total = int(total_steps)
        self.done = 0
        self.t0 = _now()
        self._last = self.t0

    def step(self, n: int = 1) -> None:
        self.done += int(n)
        self._last = _now()

    def elapsed(self) -> float:
        return _now() - self.t0

    def rate(self) -> float:
        # steps per second
        e = self.elapsed()
        return (self.done / e) if e > 1e-9 else np.nan

    def eta(self) -> float:
        r = self.rate()
        if not np.isfinite(r) or r <= 0:
            return np.nan
        remaining = max(0, self.total - self.done)
        return remaining / r

def read_csv_with_progress(path: str, *, usecols=None, chunksize: int = 200_000, desc: str = "Reading CSV") -> pd.DataFrame:
    """
    Read CSV with a progress bar that behaves like 'download progress'.
    Shows estimated remaining time based on bytes read.
    """
    file_size = None
    try:
        file_size = os.path.getsize(path)
    except Exception:
        file_size = None

    # If tqdm isn't available, fallback to normal read
    if tqdm is None or file_size is None or file_size <= 0:
        return pd.read_csv(path, usecols=usecols)

    bytes_read = 0
    t0 = _now()

    pbar = tqdm(total=file_size, unit="B", unit_scale=True, desc=desc, leave=True)
    chunks = []
    # pandas iterator
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunks.append(chunk)
        # rough bytes estimate: memory usage of chunk (not exact file bytes, but tracks progress smoothly)
        est = int(chunk.memory_usage(deep=True).sum())
        bytes_read += est
        # cap at file_size so bar reaches 100%
        pbar.update(min(est, max(0, file_size - pbar.n)))

        # set postfix with ETA computed from pbar
        elapsed = _now() - t0
        speed = (pbar.n / elapsed) if elapsed > 1e-9 else np.nan
        eta = (file_size - pbar.n) / speed if np.isfinite(speed) and speed > 0 else np.nan
        pbar.set_postfix_str(f"ETA {_fmt_seconds(eta)}")

    pbar.close()
    df = pd.concat(chunks, ignore_index=True)
    return df


# -----------------------------
# Helpers: target detection + alias
# -----------------------------
def find_target_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "label_mortality",
        "hospital_expire_flag",
        "in_hospital_mortality",
        "mortality", "Mortality", "death", "Death", "outcome", "Outcome",
        "label", "Label", "target", "Target", "y", "Y",
        "HOSPITAL_EXPIRE_FLAG", "IN_HOSPITAL_MORTALITY",
    ]
    for c in candidates:
        if c in df.columns:
            vals = df[c].dropna().unique()
            if len(vals) <= 2:
                return c

    # fallback: any non-object with <=2 uniques
    for c in df.columns:
        if df[c].dtype == "O":
            continue
        vals = df[c].dropna().unique()
        if len(vals) <= 2:
            return c
    return None


def resolve_target(df: pd.DataFrame, target: str = "") -> str:
    """
    Resolve target name robustly:
      - exact match
      - case-insensitive match
      - alias mapping (hospital_expire_flag -> label_mortality)
      - auto-detect if blank
    """
    t = (target or "").strip()

    # If blank -> autodetect
    if not t:
        t2 = find_target_column(df)
        if not t2:
            raise ValueError("Target column not found (auto-detect failed).")
        return t2

    # Exact match
    if t in df.columns:
        return t

    # Case-insensitive match
    lower_map = {c.lower(): c for c in df.columns}
    if t.lower() in lower_map:
        return lower_map[t.lower()]

    # Alias mapping commonly used in MIMIC pipelines
    alias = {
        "hospital_expire_flag": "label_mortality",
        "HOSPITAL_EXPIRE_FLAG": "label_mortality",
    }
    if t in alias and alias[t] in df.columns:
        return alias[t]

    raise ValueError(
        f"Target column not found. Got target='{t}'. "
        f"Example columns: {list(df.columns[:20])} ... "
        f"(Try target='label_mortality')"
    )


def coerce_binary_y(y: pd.Series | np.ndarray) -> np.ndarray:
    y_ser = pd.Series(y).dropna()
    uniq = list(pd.unique(y_ser))
    if len(uniq) != 2:
        raise ValueError(f"Target must be binary-like. Found unique={uniq[:10]} (n={len(uniq)})")

    # stable mapping: smaller->0, larger->1 when sortable
    try:
        uniq_sorted = sorted(uniq)
        mapping = {uniq_sorted[0]: 0, uniq_sorted[1]: 1}
        y01 = pd.Series(y).map(mapping).to_numpy()
    except Exception:
        uniq_sorted = sorted([str(u) for u in uniq])
        mapping = {uniq_sorted[0]: 0, uniq_sorted[1]: 1}
        y01 = pd.Series(y).astype(str).map(mapping).to_numpy()

    y01 = np.asarray(y01)
    if set(np.unique(y01[~pd.isna(y01)])) - {0, 1}:
        raise ValueError("Failed to coerce y into {0,1}.")
    return y01.astype(int)


# -----------------------------
# Group-aware repeated CV (see run_rsce.py for the identical rationale/impl --
# kept duplicated here rather than shared across the two script folders so
# each script still runs standalone with no cross-directory import).
# -----------------------------
def choose_safe_group_folds(y: np.ndarray, groups: Optional[np.ndarray], desired_splits: int) -> int:
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
    y = np.asarray(y)
    groups = np.asarray(groups)
    for r in range(n_repeats):
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for train_idx, test_idx in skf.split(X, y, groups):
            yield train_idx, test_idx


def stratified_subsample_for_fit_cap(X_tr: pd.DataFrame, y_tr: np.ndarray, cap: int, seed: int) -> Tuple[pd.DataFrame, np.ndarray]:
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


# -----------------------------
# Preprocess & models
# -----------------------------
def to_dense_if_sparse(X):
    try:
        import scipy.sparse as sp  # type: ignore
        if sp.issparse(X):
            return X.toarray()
    except Exception:
        pass
    return X


def make_preprocessor(num_cols: List[str], cat_cols: List[str], force_dense: bool) -> ColumnTransformer:
    num_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
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


def make_model_zoo(seed: int, fast: bool=False) -> Dict[str, Any]:
    # fast=True uses fewer trees/iterations (quicker smoke tests); the reported runs use fast=False
    rf_n = 300 if fast else 1200
    et_n = 300 if fast else 1200
    gb_n = 250 if fast else 700
    mlp_iter = 800 if fast else 2500

    return {
        "Logistic_L2": LogisticRegression(max_iter=5000, random_state=seed),
        "RandomForest": RandomForestClassifier(
            n_estimators=rf_n, n_jobs=-1, random_state=seed, class_weight="balanced_subsample"
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=et_n, n_jobs=-1, random_state=seed, class_weight="balanced_subsample"
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=gb_n, learning_rate=0.03, random_state=seed
        ),
        # probability=False: PPV only needs a RANKING score. Without calibration the
        # pipeline scores with decision_function (monotone); with CalibratedClassifierCV
        # the calibrator also uses decision_function. probability=True only added an
        # internal 5-fold Platt fit (~4x SVC cost) whose output was never used.
        "SVC_RBF": SVC(kernel="rbf", probability=False, C=3.0, gamma="scale", random_state=seed),
        # was MLPClassifier((256,128,64), max_iter=mlp_iter) without early stopping -- see EarlyStoppedMLP
        "MLP": EarlyStoppedMLP(hidden_layer_sizes=(256, 128, 64), max_iter=mlp_iter, validation_fraction=0.1,
                               n_iter_no_change=10, random_state=seed),
        "GaussianNB": GaussianNB(),
    }


def make_pipeline(model_name: str, clf: Any, num_cols: List[str], cat_cols: List[str], calibrate: Optional[str]) -> Pipeline:
    need_dense = (model_name == "GaussianNB")
    pre = make_preprocessor(num_cols, cat_cols, force_dense=need_dense)
    steps = [("preprocess", pre)]
    if need_dense:
        steps.append(("to_dense", FunctionTransformer(to_dense_if_sparse, accept_sparse=True)))

    est = clf
    if calibrate is not None:
        est = CalibratedClassifierCV(estimator=clf, method=calibrate, cv=3)

    steps.append(("clf", est))
    return Pipeline(steps)


def predict_proba_safe(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        return np.asarray(pipe.predict_proba(X)[:, 1], dtype=float)
    if hasattr(pipe, "decision_function"):
        z = pipe.decision_function(X)
        return np.asarray(1.0 / (1.0 + np.exp(-z)), dtype=float)
    raise RuntimeError("Model does not support probability prediction.")


# -----------------------------
# Option 2B core
# -----------------------------
def threshold_for_fixed_sensitivity(y_true: np.ndarray, p: np.ndarray, target_sens: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)

    pos = (y_true == 1)
    if pos.sum() == 0:
        return float("inf")

    # "inverted_cdf" returns an actual observed score such that AT LEAST target_sens
    # of the positives have score >= threshold. The default (linear interpolation)
    # can land between two positives and undershoot: with the 12 deaths of a Demo
    # hosp training fold it gave 0.75 instead of 0.80, so Demo and Full were
    # evaluated at different operating points. Scores are not clipped to [0, 1]
    # (they may be decision-function values; only their ranking matters).
    q = np.quantile(p[pos], 1.0 - float(target_sens), method="inverted_cdf")
    return float(q)


def inner_oof_predictions_for_threshold(
    pipe_template: Pipeline,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    inner_splits: int,
    seed: int,
    *,
    groups_tr: Optional[np.ndarray] = None,
    svc_max_train_n: Optional[int] = None,
    model_name: str = "",
    show_progress: bool = True,
    desc: str = "innerCV",
) -> np.ndarray:
    # Group-aware inner CV when groups_tr is given (same patient can't be split
    # across the inner train/validation used just to pick the decision threshold).
    safe_inner = choose_safe_group_folds(y_tr, groups_tr, inner_splits)
    if safe_inner == 0:
        print(f"[run_ppv] WARNING ({desc}): group-aware inner CV impossible (too few patients in a class); "
              f"falling back to ROW-level inner CV for threshold selection.")
        safe_inner = choose_safe_group_folds(y_tr, None, inner_splits)
        groups_tr = None
    inner_splits = max(safe_inner, 2)

    if groups_tr is not None:
        split_fn = lambda: repeated_stratified_group_kfold(X_tr, y_tr, groups_tr, n_splits=inner_splits, n_repeats=1, seed=seed)
    else:
        skf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
        split_fn = lambda: skf.split(X_tr, y_tr)

    p_oof = np.full(len(y_tr), np.nan, dtype=float)

    from sklearn.base import clone

    it = list(enumerate(split_fn(), start=1))
    if tqdm is not None and show_progress:
        it = tqdm(it, total=inner_splits, desc=desc, leave=False)

    for k, (i_tr, i_va) in it:
        pipe = clone(pipe_template)
        Xi_tr, yi_tr = X_tr.iloc[i_tr], y_tr[i_tr]
        if model_name == "SVC_RBF" and svc_max_train_n and len(yi_tr) > svc_max_train_n:
            Xi_tr, yi_tr = stratified_subsample_for_fit_cap(Xi_tr, yi_tr, cap=svc_max_train_n, seed=seed + k)
        pipe.fit(Xi_tr, yi_tr)
        p_oof[i_va] = predict_proba_safe(pipe, X_tr.iloc[i_va])

    if np.any(~np.isfinite(p_oof)):
        print(f"[run_ppv] WARNING ({desc}): {int(np.sum(~np.isfinite(p_oof)))} training rows got no inner "
              f"out-of-fold prediction; filling them with IN-SAMPLE predictions of a model fit on the whole "
              f"training fold (optimistic for those rows).")
        pipe = clone(pipe_template)
        Xf_tr, yf_tr = X_tr, y_tr
        if model_name == "SVC_RBF" and svc_max_train_n and len(y_tr) > svc_max_train_n:
            Xf_tr, yf_tr = stratified_subsample_for_fit_cap(X_tr, y_tr, cap=svc_max_train_n, seed=seed)
        pipe.fit(Xf_tr, yf_tr)
        bad = ~np.isfinite(p_oof)
        p_oof[bad] = predict_proba_safe(pipe, X_tr.iloc[bad])

    return p_oof


def confusion_from_threshold(y_true: np.ndarray, p: np.ndarray, thr: float) -> Tuple[int, int, int, int]:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(p).astype(float) >= thr).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    return tp, fp, tn, fn


def sens_spec_from_conf(tp: int, fp: int, tn: int, fn: int) -> Tuple[float, float]:
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return float(sens), float(spec)


def ppv_from_conf(tp: int, fp: int) -> float:
    return float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan


def ppv_standardized(sens: float, spec: float, pi_ref: float) -> float:
    if not np.isfinite(sens) or not np.isfinite(spec):
        return np.nan
    pi = float(pi_ref)
    denom = sens * pi + (1.0 - spec) * (1.0 - pi)
    if denom <= 0:
        return np.nan
    return float((sens * pi) / denom)


# -----------------------------
# Columns that must never be model features
# -----------------------------
# POST-OUTCOME columns: known only AFTER the admission ends and definitionally
# tied to the label. `discharge_location == "DIED"` covers ~97% of in-hospital
# deaths in MIMIC-IV hosp (sens 0.970, PPV 0.981 as a one-line rule on Full;
# a perfect 15/15, 0 FP separator on Demo). Earlier PPV runs that did not drop
# it produced PPV ~0.98 (Full) / 1.00 (Demo) purely from this leak. These are
# ALWAYS dropped, whatever --drop_cols says (defense in depth).
ALWAYS_DROP_POST_OUTCOME = ("discharge_location",)

# Default --drop_cols: IDs + the same non-feature columns run_rsce.py's hosp
# commands drop (anchor_year / anchor_year_group are MIMIC's per-patient
# date-shift bookkeeping, dropped in RSCE for consistency). Names not present
# in a given dataset (e.g. hadm_id in ED data) are silently ignored.
DEFAULT_DROP_COLS = ["hadm_id", "stay_id", "subject_id",
                     "discharge_location", "anchor_year", "anchor_year_group"]


# -----------------------------
# Checkpoint / resume (per fold x model unit)
# -----------------------------
# Each finished (fold, model) unit is one row of ppv_std_per_fold.csv and is
# saved atomically to <outdir>/_checkpoint. Re-running the SAME command skips
# finished units. Every unit depends only on (seed, fold_id, model) -- the outer
# splits are regenerated deterministically and each unit fits a fresh clone --
# so a resumed run gives the same results as an uninterrupted one. The config
# (data SHA-256, all result-affecting settings, package versions) must match,
# otherwise the run is refused so results are never mixed.
PPV_COMPUTE_VERSION = "ppv-v2-2026-09-25"


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _ckpt_unit_file(ckpt_dir: Path, fold_id: int, model_name: str) -> Path:
    return ckpt_dir / f"fold{fold_id:03d}__{model_name}.pkl"


def _atomic_replace(tmp: Path, final: Path) -> None:
    for attempt in range(20):
        try:
            os.replace(tmp, final)
            return
        except PermissionError:  # Windows antivirus/indexer can briefly lock the target
            if attempt == 19:
                raise
            time.sleep(0.5 * (attempt + 1))


def _ckpt_save(ckpt_dir: Path, fold_id: int, model_name: str, row: Dict[str, Any], params_sha: str) -> None:
    final = _ckpt_unit_file(ckpt_dir, fold_id, model_name)
    tmp = final.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"row": row, "model_params_sha": params_sha}, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    _atomic_replace(tmp, final)


def _ckpt_load(ckpt_dir: Path, fold_id: int, model_name: str, params_sha: str) -> Optional[Dict[str, Any]]:
    """Return the saved result row, or None when the unit is missing or was made with other
    model settings (then the old file is moved to <ckpt>/replaced_units/ and recomputed).
    Units written by an earlier version are the bare row dict without a model_params_sha; of
    those only models in MODEL_PARAMS_CHANGED_SINCE_LEGACY are treated as stale."""
    fp = _ckpt_unit_file(ckpt_dir, fold_id, model_name)
    if not fp.exists():
        return None
    with open(fp, "rb") as f:
        unit = pickle.load(f)
    if isinstance(unit, dict) and "row" in unit and "model_params_sha" in unit:
        row, stale = unit["row"], unit["model_params_sha"] != params_sha
    else:  # legacy format
        row, stale = unit, model_name in MODEL_PARAMS_CHANGED_SINCE_LEGACY
    if not stale:
        return row
    dst_dir = ckpt_dir / "replaced_units"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst, k = dst_dir / fp.name, 1
    while dst.exists():
        dst, k = dst_dir / f"{fp.stem}.old{k}{fp.suffix}", k + 1
    os.replace(fp, dst)
    print(f"[run_ppv] fold {fold_id} {model_name}: checkpoint unit was made with different model "
          f"settings -> recomputing it (old file kept in {dst_dir}).")
    return None


def _ckpt_prepare(ckpt_dir: Path, config: Dict[str, Any]) -> int:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(json.dumps(config, sort_keys=True, default=str))
    cfg_path = ckpt_dir / "checkpoint_config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        if old != cfg:
            diff = sorted(k for k in set(old) | set(cfg) if old.get(k) != cfg.get(k))
            raise RuntimeError(
                f"[run_ppv] Checkpoint in {ckpt_dir} was made with different settings/data/packages ({diff}). "
                f"Refusing to mix results. Delete that _checkpoint folder (or use another --outdir) to start over.")
    else:
        tmp = cfg_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
        _atomic_replace(tmp, cfg_path)
    return len(list(ckpt_dir.glob("fold*__*.pkl")))


# -----------------------------
# Runner (with progress)
# -----------------------------
def run_ppv_std(
    data_path: str,
    outdir: str,
    pi_ref: Optional[float] = None,
    target: str = "label_mortality",  # <-- default for your files
    drop_cols: Optional[List[str]] = None,
    seed: int = 42,
    folds: int = 5,
    repeats: int = 3,
    calibrate: str = "none",      # "none" (default) | "sigmoid" | "isotonic"
    inner_splits: int = 3,
    target_sens: float = 0.80,
    fast: bool = False,           # True = fewer trees/iterations (quicker)
    show_progress: bool = True,    # <--- NEW
    group_col: str = "subject_id",
    cv_mode: str = "group",
    exclude_models: Optional[List[str]] = None,
    svc_max_train_n: Optional[int] = None,
    checkpoint: bool = True,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Read CSV with download-like progress ----
    df = read_csv_with_progress(data_path, desc=f"Loading {Path(data_path).name}")
    tcol = resolve_target(df, target)
    df = df.dropna(subset=[tcol]).copy()
    y = coerce_binary_y(df[tcol].values)

    if pi_ref is None:
        pi_ref = float(np.mean(y))

    # ---- group-aware CV setup (prevents the same patient's admissions/stays
    # from appearing in both train and test -- see run_rsce.py for full rationale) ----
    groups_all: Optional[np.ndarray] = None
    if cv_mode == "group":
        if group_col in df.columns:
            groups_all = df[group_col].to_numpy()
        else:
            print(f"[run_ppv] WARNING: --group_col '{group_col}' not found; falling back to row-level CV.")

    if drop_cols is None:
        drop_cols = list(DEFAULT_DROP_COLS)
    drop_set = set([tcol])
    for c in drop_cols:
        if c in df.columns:
            drop_set.add(c)
    for c in ALWAYS_DROP_POST_OUTCOME:
        if c in df.columns and c not in drop_set:
            print(f"[run_ppv] WARNING: '{c}' is a post-outcome column (label leakage) and was not in "
                  f"--drop_cols; dropping it anyway.")
            drop_set.add(c)
    print(f"[run_ppv] Dropped from features: {sorted(drop_set - {tcol})}")
    X_full = df.drop(columns=list(drop_set), errors="ignore")

    num_cols = [c for c in X_full.columns if pd.api.types.is_numeric_dtype(X_full[c])]
    cat_cols = [c for c in X_full.columns if c not in num_cols]
    X = X_full[num_cols + cat_cols].copy()

    if groups_all is not None:
        safe_folds = choose_safe_group_folds(y, groups_all, folds)
        if safe_folds == 0:
            print("[run_ppv] WARNING: not enough distinct groups per class for group-aware CV; falling back to row-level.")
            groups_all = None
        elif safe_folds < folds:
            print(f"[run_ppv] WARNING: reducing folds from {folds} to {safe_folds} (group/class counts).")
            folds = safe_folds
    if groups_all is None:
        safe_folds = choose_safe_group_folds(y, None, folds)
        if safe_folds == 0:
            raise ValueError(f"Not enough samples in the minority class of '{tcol}' to run any CV split.")
        if safe_folds < folds:
            print(f"[run_ppv] WARNING: reducing folds from {folds} to {safe_folds} (minority class too small).")
            folds = safe_folds

    cal = None if calibrate == "none" else calibrate
    if cal is not None:
        print(f"[run_ppv] NOTE: --calibrate {cal}: CalibratedClassifierCV uses a row-level inner split "
              f"(patients can appear on both sides). Use --calibrate none for the main analysis.")

    model_zoo = make_model_zoo(seed=seed, fast=fast)
    if exclude_models:
        excl = set(exclude_models)
        model_zoo = {k: v for k, v in model_zoo.items() if k not in excl}
        print(f"[run_ppv] Excluding models: {sorted(excl)}. Remaining: {sorted(model_zoo.keys())}")
    pipelines = {
        name: make_pipeline(name, est, num_cols=num_cols, cat_cols=cat_cols, calibrate=cal)
        for name, est in model_zoo.items()
    }
    model_sha = {name: model_params_sha(est) for name, est in model_zoo.items()}
    import sklearn as _skl
    run_info = {
        "data": str(data_path), "target": tcol, "n_samples": int(len(y)), "pi_ref": float(pi_ref),
        "seed": seed, "folds": folds, "repeats": repeats, "calibrate": calibrate,
        "inner_splits": inner_splits, "target_sens": target_sens, "fast": bool(fast),
        "cv_mode": "group" if groups_all is not None else "row", "svc_max_train_n": svc_max_train_n,
        "model_params": {name: model_params_record(est) for name, est in model_zoo.items()},
        "python": platform.python_version(), "sklearn": _skl.__version__,
    }
    (outdir / "ppv_run_info.json").write_text(json.dumps(run_info, indent=2, default=str), encoding="utf-8")

    if groups_all is not None:
        outer_split_factory = lambda: repeated_stratified_group_kfold(X, y, groups_all, n_splits=folds, n_repeats=repeats, seed=seed)
    else:
        rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
        outer_split_factory = lambda: rskf.split(X, y)

    rows = []
    fold_id = 0
    n_total_folds = folds * repeats
    n_models = len(pipelines)
    total_steps = n_total_folds * n_models

    ckpt_dir: Optional[Path] = None
    n_done = 0
    if checkpoint:
        import sklearn
        ckpt_dir = outdir / "_checkpoint"
        n_done = _ckpt_prepare(ckpt_dir, {
            "compute_version": PPV_COMPUTE_VERSION,
            "data_sha256": _sha256_file(data_path),
            "n_samples": int(len(y)),
            "target": tcol,
            "dropped": sorted(drop_set),
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols,
            "pi_ref": repr(float(pi_ref)),
            "seed": seed, "folds": folds, "repeats": repeats,
            "calibrate": calibrate, "inner_splits": inner_splits, "target_sens": target_sens,
            "fast": bool(fast), "cv_mode": "group" if groups_all is not None else "row",
            "group_col": group_col if groups_all is not None else None,
            "models": list(pipelines.keys()), "svc_max_train_n": svc_max_train_n,
            "python": platform.python_version(), "sklearn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
        })
        n_stale = 0
        for fp in ckpt_dir.glob("fold*__*.pkl"):
            mname_fp = fp.stem.split("__", 1)[1]
            try:
                with open(fp, "rb") as f:
                    u = pickle.load(f)
            except Exception:
                continue
            if isinstance(u, dict) and "row" in u and "model_params_sha" in u:
                n_stale += int(u["model_params_sha"] != model_sha.get(mname_fp, u["model_params_sha"]))
            else:
                n_stale += int(mname_fp in MODEL_PARAMS_CHANGED_SINCE_LEGACY)
        n_done -= n_stale
        if n_done or n_stale:
            print(f"[run_ppv] Resuming: {n_done} finished (fold, model) unit(s) found in {ckpt_dir}; "
                  f"they are reloaded, not recomputed."
                  + (f" {n_stale} unit(s) were made with older model settings and will be recomputed." if n_stale else ""))

    overall_eta = RunningETA(total_steps=max(total_steps - n_done, 0))

    from sklearn.base import clone

    # Outer progress bar
    outer_iter = outer_split_factory()
    if tqdm is not None and show_progress:
        outer_iter = tqdm(outer_iter, total=n_total_folds, desc=f"CV folds ({n_total_folds})", leave=True)

    for tr_idx, te_idx in outer_iter:
        fold_id += 1
        X_tr, y_tr = X.iloc[tr_idx].copy(), y[tr_idx].copy()
        X_te, y_te = X.iloc[te_idx].copy(), y[te_idx].copy()
        groups_tr = groups_all[tr_idx] if groups_all is not None else None
        pi_test = float(np.mean(y_te)) if len(y_te) else np.nan

        # Per-fold models progress
        model_iter = pipelines.items()
        if tqdm is not None and show_progress:
            model_iter = tqdm(list(model_iter), total=n_models, desc=f"Models (fold {fold_id}/{n_total_folds})", leave=False)
            # rebuild iterator after list() for safety
            model_iter = tqdm(pipelines.items(), total=n_models, desc=f"Models (fold {fold_id}/{n_total_folds})", leave=False)

        for model_name, pipe_template in model_iter:
            if ckpt_dir is not None:
                saved = _ckpt_load(ckpt_dir, fold_id, model_name, model_sha[model_name])
                if saved is not None:
                    rows.append(saved)
                    continue
            t_step0 = _now()

            # Inner OOF for threshold
            p_tr_oof = inner_oof_predictions_for_threshold(
                pipe_template=pipe_template,
                X_tr=X_tr, y_tr=y_tr,
                inner_splits=inner_splits,
                seed=seed + 10_000 * fold_id,
                groups_tr=groups_tr,
                svc_max_train_n=svc_max_train_n,
                model_name=model_name,
                show_progress=show_progress,
                desc=f"innerCV ({model_name})",
            )
            thr = threshold_for_fixed_sensitivity(y_tr, p_tr_oof, target_sens=target_sens)

            # Fit full train, eval test
            pipe = clone(pipe_template)
            X_tr_fit, y_tr_fit = X_tr, y_tr
            if model_name == "SVC_RBF" and svc_max_train_n and len(y_tr) > svc_max_train_n:
                X_tr_fit, y_tr_fit = stratified_subsample_for_fit_cap(X_tr, y_tr, cap=svc_max_train_n, seed=seed + fold_id)
            pipe.fit(X_tr_fit, y_tr_fit)
            p_te = predict_proba_safe(pipe, X_te)

            tp, fp, tn, fn = confusion_from_threshold(y_te, p_te, thr)
            sens, spec = sens_spec_from_conf(tp, fp, tn, fn)
            ppv_obs = ppv_from_conf(tp, fp)
            ppv_std = ppv_standardized(sens, spec, pi_ref=pi_ref)

            # ---- Update overall ETA ----
            overall_eta.step(1)
            t_step = _now() - t_step0

            row = {
                "fold": fold_id,
                "repeat": int((fold_id - 1) // folds + 1),
                "model": model_name,
                "n_test": int(len(y_te)),
                "pi_test": pi_test,
                "pi_ref": float(pi_ref),
                "target_sens_train": float(target_sens),
                "thr_from_train_inner_oof": float(thr),
                "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                "sens_test": sens,
                "spec_test": spec,
                "ppv_observed_test": ppv_obs,
                "ppv_standardized_pi_ref": ppv_std,
                "time_total_s": float(t_step),
            }
            rows.append(row)
            if ckpt_dir is not None:
                _ckpt_save(ckpt_dir, fold_id, model_name, row, model_sha[model_name])

            # Print / postfix with overall ETA
            if tqdm is not None and show_progress:
                # attach to current model progress bar if exists
                try:
                    model_iter.set_postfix_str(
                        f"last={_fmt_seconds(t_step)} overallETA={_fmt_seconds(overall_eta.eta())}"
                    )
                except Exception:
                    pass
            else:
                # fallback prints
                done = overall_eta.done
                tot = overall_eta.total
                print(
                    f"[{done}/{tot}] fold {fold_id}/{n_total_folds} | {model_name} "
                    f"| last={_fmt_seconds(t_step)} | overall ETA={_fmt_seconds(overall_eta.eta())}"
                )

        # update outer fold bar with overall ETA too
        if tqdm is not None and show_progress:
            try:
                outer_iter.set_postfix_str(f"overallETA={_fmt_seconds(overall_eta.eta())}")
            except Exception:
                pass

    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(outdir / "ppv_std_per_fold.csv", index=False)

    agg = (
        per_fold
        .groupby("model", as_index=False)
        .agg(
            n_folds=("ppv_standardized_pi_ref", "count"),   # non-NaN folds only
            ppv_std_mean=("ppv_standardized_pi_ref", "mean"),
            ppv_std_std=("ppv_standardized_pi_ref", "std"),
            ppv_obs_mean=("ppv_observed_test", "mean"),
            sens_mean=("sens_test", "mean"),
            spec_mean=("spec_test", "mean"),
        )
    )
    agg["ppv_std_sem_naive"] = agg["ppv_std_std"] / np.sqrt(np.maximum(agg["n_folds"], 1))

    # PRIMARY estimate: pool the confusion counts over the folds of each repeat
    # (every patient is tested exactly once per repeat), compute sens/spec/PPV_std
    # from the pooled counts, then average over repeats. Averaging per-fold PPV is
    # biased when test folds contain few positives (Demo hosp: 3 deaths per fold, so
    # fold sensitivity can only be 0, 1/3, 2/3 or 1 and PPV_std is nonlinear in it).
    rep_rows = []
    for (model_name, rep_id), sub in per_fold.groupby(["model", "repeat"]):
        TP, FP, TN, FN = (int(sub[c].sum()) for c in ("TP", "FP", "TN", "FN"))
        se, sp = sens_spec_from_conf(TP, FP, TN, FN)
        rep_rows.append({"model": model_name, "repeat": rep_id, "TP": TP, "FP": FP, "TN": TN, "FN": FN,
                         "sens_pooled": se, "spec_pooled": sp,
                         "ppv_std_pooled": ppv_standardized(se, sp, pi_ref=pi_ref)})
    per_repeat = pd.DataFrame(rep_rows)
    per_repeat.to_csv(outdir / "ppv_std_per_repeat.csv", index=False)
    pooled = (per_repeat.groupby("model", as_index=False)
              .agg(n_repeats=("ppv_std_pooled", "count"),
                   ppv_std_pooled_mean=("ppv_std_pooled", "mean"),
                   ppv_std_pooled_sd_over_repeats=("ppv_std_pooled", "std"),
                   sens_pooled_mean=("sens_pooled", "mean"),
                   spec_pooled_mean=("spec_pooled", "mean")))
    agg = pooled.merge(agg, on="model", how="outer")
    agg["pi_ref"] = float(pi_ref)
    agg["target_sens"] = float(target_sens)
    agg.to_csv(outdir / "ppv_std_aggregated.csv", index=False)

    print(f"\n[OK] {data_path}")
    print(f"  Target = {tcol}")
    print(f"  pi_ref = {pi_ref!r}")
    print(f"  Total time = {_fmt_seconds(overall_eta.elapsed())}")
    print(f"  Wrote:\n    - {outdir/'ppv_std_per_fold.csv'}\n    - {outdir/'ppv_std_per_repeat.csv'}\n    - {outdir/'ppv_std_aggregated.csv'}")
    return per_fold, agg


# -----------------------------
# CLI entry point
# -----------------------------
# One dataset per invocation, so it composes cleanly with the rest of the
# pipeline (and works for hosp or ED). The key idea -- standardize both Full
# and Demo PPV to the SAME reference prevalence, taken from Full -- is
# implemented via --pi_ref (or --pi_ref_from):
#
#   # 1) Full run: prints pi_ref (Full's own prevalence) to the console
#   python run_ppv.py --data full_analytic_dataset_mortality_all_admissions.csv \
#       --target label_mortality --outdir results/hosp_full/ppv
#
#   # 2) Demo run: pass the SAME pi_ref printed above, so both are standardized
#   #    to Full's prevalence and are directly comparable
#   python run_ppv.py --data demo_analytic_dataset_mortality_all_admissions.csv \
#       --target label_mortality --outdir results/hosp_demo/ppv --pi_ref <value from step 1>
#
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prevalence-standardized PPV benchmark for one dataset.")
    p.add_argument("--data", type=str, required=True, help="Path to the analytic dataset CSV.")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--target", type=str, default="label_mortality", help="Target column (e.g. label_mortality or label_ed_admit).")
    p.add_argument("--pi_ref", type=float, default=None,
                    help="Reference prevalence to standardize PPV to. If omitted, uses this dataset's own prevalence "
                         "(appropriate for a standalone Full run); pass the Full run's pi_ref explicitly when running Demo, "
                         "so both sides are standardized to the same reference.")
    p.add_argument("--pi_ref_from", type=str, default=None,
                    help="Alternative to --pi_ref: path of the FULL analytic CSV; pi_ref is computed from its target "
                         "column exactly as the Full run computes its own pi_ref (use this for the Demo side so the "
                         "value never has to be copied by hand).")
    p.add_argument("--drop_cols", type=str, nargs="*", default=list(DEFAULT_DROP_COLS),
                    help="Columns to drop from the FEATURE set (subject_id is kept in the loaded dataframe "
                         "for --group_col grouping, then dropped here from X like any other ID column).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--calibrate", type=str, default="none", choices=["none", "sigmoid", "isotonic"],
                    help="Default 'none'. PPV at a threshold chosen for a fixed sensitivity depends only on how "
                         "the model RANKS patients, so calibration cannot improve it; CalibratedClassifierCV "
                         "(3 ensemble members, row-level unshuffled inner split -> the same patient on both sides) "
                         "only added noise and 3x the fits, and at Demo scale produced inverted calibrators and a "
                         "collapsed MLP (spec = 0 in 10/15 hosp folds). 'sigmoid'/'isotonic' kept for sensitivity only.")
    p.add_argument("--inner_splits", type=int, default=3)
    p.add_argument("--target_sens", type=float, default=0.80)
    p.add_argument("--fast", action="store_true", help="Use smaller model configs (fewer trees/iterations) for a quicker run.")
    p.add_argument("--no_progress", action="store_true")

    p.add_argument("--group_col", type=str, default="subject_id",
                    help="Column identifying the patient for group-aware CV (prevents the same patient's "
                         "admissions/stays from appearing in both train and test).")
    p.add_argument("--cv_mode", type=str, default="group", choices=["group", "row"],
                    help="'group' (default): patient-level StratifiedGroupKFold. 'row': original row-level "
                         "RepeatedStratifiedKFold (kept for sensitivity-analysis comparison).")
    p.add_argument("--exclude_models", type=str, nargs="*", default=[],
                    help="Model names to skip entirely, e.g. --exclude_models SVC_RBF MLP.")
    p.add_argument("--svc_max_train_n", type=int, default=None,
                    help="Cap SVC_RBF's training-set size (stratified subsample) on large folds; see run_rsce.py "
                         "for the same option and rationale.")
    p.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=True,
                    help="Save each finished (fold, model) unit to <outdir>/_checkpoint and skip finished units "
                         "when the same command is re-run (resume after a crash / power cut). Default on; "
                         "--no-checkpoint disables it. Ignored with --estimate_only.")
    p.add_argument("--estimate_only", action="store_true",
                    help="Run a minimal 2-fold, 1-repeat probe and print/save an extrapolated total runtime for "
                         "the folds/repeats you actually requested, then exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.pi_ref is not None and args.pi_ref_from:
        raise SystemExit("[run_ppv] give either --pi_ref or --pi_ref_from, not both.")
    pi_ref = args.pi_ref
    if args.pi_ref_from:
        tmp = read_csv_with_progress(args.pi_ref_from, desc=f"Loading {Path(args.pi_ref_from).name} for pi_ref")
        tcol_ref = resolve_target(tmp, args.target)
        pi_ref = float(coerce_binary_y(tmp[tcol_ref].dropna().values).mean())
        del tmp
        print(f"[run_ppv] pi_ref from {args.pi_ref_from}: {pi_ref!r}")
    if pi_ref is None:
        tmp = read_csv_with_progress(args.data, desc=f"Loading {Path(args.data).name} for pi_ref")
        tcol_ref = resolve_target(tmp, args.target)
        pi_ref = float(coerce_binary_y(tmp[tcol_ref].dropna().values).mean())
        print(f"[run_ppv] --pi_ref not given; using this dataset's own prevalence: pi_ref={pi_ref!r}")
        print("[run_ppv] If this is the FULL side of a Demo-vs-Full comparison, run the DEMO side with "
              "--pi_ref_from <this Full CSV> (or --pi_ref with ALL digits printed above).")

    orig_folds, orig_repeats = args.folds, args.repeats
    run_folds, run_repeats = args.folds, args.repeats
    if args.estimate_only:
        run_folds, run_repeats = max(2, min(2, orig_folds)), 1
        print(f"[run_ppv] --estimate_only: probing with folds=2, repeats=1 "
              f"(will extrapolate to the requested folds={orig_folds}, repeats={orig_repeats}).")

    per_fold, agg = run_ppv_std(
        data_path=args.data,
        outdir=str(outdir),
        pi_ref=pi_ref,
        target=args.target,
        drop_cols=args.drop_cols,
        seed=args.seed,
        folds=run_folds,
        repeats=run_repeats,
        calibrate=args.calibrate,
        inner_splits=args.inner_splits,
        target_sens=args.target_sens,
        fast=args.fast,
        show_progress=not args.no_progress,
        group_col=args.group_col,
        cv_mode=args.cv_mode,
        exclude_models=args.exclude_models,
        svc_max_train_n=args.svc_max_train_n,
        checkpoint=bool(args.checkpoint) and not args.estimate_only,
    )

    if args.estimate_only:
        target_total_folds = orig_folds * orig_repeats
        per_model_mean_s = per_fold.groupby("model")["time_total_s"].mean()
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
        print(f"\n[ESTIMATE] Per-model extrapolated runtime for folds={orig_folds}, repeats={orig_repeats} "
              f"({target_total_folds} folds total):")
        print(est_df.to_string(index=False))
        print(f"\n[ESTIMATE] Grand total: ~{grand_total_s/3600.0:.2f} hours (~{grand_total_s/86400.0:.2f} days).")
        print(f"[ESTIMATE] Wrote: {outdir / 'estimate_timing.csv'}. Re-run without --estimate_only for real results.")
        return

    print("\n=== ppv_std_aggregated (sorted) ===")
    print(agg.sort_values("ppv_std_pooled_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
