"""
This version implements:
1) A prevalence-standardized PPV evaluation framework designed for fair comparison
   across datasets with different outcome prevalences (FULL vs DEMO).
2) Robust CSV loading with download-style progress bars and ETA,
   suitable for large clinical datasets and Colab environments.
3) Automatic target-column resolution with alias handling and fallback auto-detection,
   plus strict coercion of labels to binary {0,1}.
4) A unified preprocessing and modeling pipeline supporting mixed numeric/categorical data,
   optional probability calibration (sigmoid / isotonic / none),
   and dense-matrix fallback for models that require it.
5) A diverse model zoo (linear, tree-based, kernel, neural, and probabilistic),
   with an optional fast mode for quicker exploratory runs.
6) Repeated stratified cross-validation with explicit fold tracking
   to ensure comparability across runs.
7) Inner cross-validation with out-of-fold predictions to determine
   decision thresholds that achieve a fixed target sensitivity on training data.
8) Test-set evaluation using fixed-sensitivity thresholds, producing:
   sensitivity, specificity, observed PPV, and prevalence-standardized PPV.
9) Explicit handling and reporting of test-set prevalence (π_test)
   and a shared reference prevalence (π_ref) for standardization.
10) End-to-end progress tracking with nested progress bars,
    per-step timing, and overall ETA across folds and models.
11) Per-fold result export enabling downstream paired statistical comparisons.
12) Aggregation of results at the model level with means, standard deviations,
    and standard errors for standardized PPV and related metrics.
13) Reproducible experiment control via fixed seeds and deterministic CV splits.
14) Side-by-side execution on FULL and DEMO datasets using the same π_ref,
    enabling apples-to-apples PPV comparisons.

Outputs:
- ppv_std_per_fold.csv
- ppv_std_aggregated.csv
- Console summaries reporting target resolution, π_ref, and total runtime
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import os
import math
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
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


def make_model_zoo(seed: int, fast: bool=False) -> Dict[str, Any]:
    # fast=True จะลดจำนวน trees/iters ให้รันไวขึ้นใน Colab (ถ้าอยากได้เต็ม ๆ ปิด fast)
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
        "SVC_RBF": SVC(kernel="rbf", probability=True, C=3.0, gamma="scale", random_state=seed),
        "MLP": MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=mlp_iter, random_state=seed),
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
        return 1.0

    q = np.quantile(p[pos], 1.0 - float(target_sens))
    return float(np.clip(q, 0.0, 1.0))


def inner_oof_predictions_for_threshold(
    pipe_template: Pipeline,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    inner_splits: int,
    seed: int,
    *,
    show_progress: bool = True,
    desc: str = "innerCV",
) -> np.ndarray:
    skf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    p_oof = np.full(len(y_tr), np.nan, dtype=float)

    from sklearn.base import clone

    it = enumerate(skf.split(X_tr, y_tr), start=1)
    if tqdm is not None and show_progress:
        it = tqdm(list(it), total=inner_splits, desc=desc, leave=False)
        # NOTE: list(it) consumes iterator; so we rebuild safely:
        it = enumerate(skf.split(X_tr, y_tr), start=1)
        it = tqdm(it, total=inner_splits, desc=desc, leave=False)

    for k, (i_tr, i_va) in it:
        pipe = clone(pipe_template)
        pipe.fit(X_tr.iloc[i_tr], y_tr[i_tr])
        p_oof[i_va] = predict_proba_safe(pipe, X_tr.iloc[i_va])

    if np.any(~np.isfinite(p_oof)):
        pipe = clone(pipe_template)
        pipe.fit(X_tr, y_tr)
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
# Runner for Colab (with progress)
# -----------------------------
def run_ppv_std(
    data_path: str,
    outdir: str,
    pi_ref: Optional[float] = None,
    target: str = "label_mortality",  # <-- default for your files
    drop_cols: List[str] = ["hadm_id"],
    seed: int = 42,
    folds: int = 5,
    repeats: int = 3,
    calibrate: str = "sigmoid",   # "sigmoid" | "isotonic" | "none"
    inner_splits: int = 3,
    target_sens: float = 0.80,
    fast: bool = False,           # ตั้ง True ถ้าอยากให้รันไวขึ้น
    show_progress: bool = True,    # <--- NEW
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

    drop_set = set([tcol])
    for c in drop_cols:
        if c in df.columns:
            drop_set.add(c)
    X_full = df.drop(columns=list(drop_set), errors="ignore")

    num_cols = [c for c in X_full.columns if pd.api.types.is_numeric_dtype(X_full[c])]
    cat_cols = [c for c in X_full.columns if c not in num_cols]
    X = X_full[num_cols + cat_cols].copy()

    cal = None if calibrate == "none" else calibrate

    model_zoo = make_model_zoo(seed=seed, fast=fast)
    pipelines = {
        name: make_pipeline(name, est, num_cols=num_cols, cat_cols=cat_cols, calibrate=cal)
        for name, est in model_zoo.items()
    }

    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)

    rows = []
    fold_id = 0
    n_total_folds = folds * repeats
    n_models = len(pipelines)
    total_steps = n_total_folds * n_models

    overall_eta = RunningETA(total_steps=total_steps)

    from sklearn.base import clone

    # Outer progress bar
    outer_iter = rskf.split(X, y)
    if tqdm is not None and show_progress:
        outer_iter = tqdm(outer_iter, total=n_total_folds, desc=f"CV folds ({n_total_folds})", leave=True)

    for tr_idx, te_idx in outer_iter:
        fold_id += 1
        X_tr, y_tr = X.iloc[tr_idx].copy(), y[tr_idx].copy()
        X_te, y_te = X.iloc[te_idx].copy(), y[te_idx].copy()
        pi_test = float(np.mean(y_te)) if len(y_te) else np.nan

        # Per-fold models progress
        model_iter = pipelines.items()
        if tqdm is not None and show_progress:
            model_iter = tqdm(list(model_iter), total=n_models, desc=f"Models (fold {fold_id}/{n_total_folds})", leave=False)
            # rebuild iterator after list() for safety
            model_iter = tqdm(pipelines.items(), total=n_models, desc=f"Models (fold {fold_id}/{n_total_folds})", leave=False)

        for model_name, pipe_template in model_iter:
            t_step0 = _now()

            # Inner OOF for threshold
            p_tr_oof = inner_oof_predictions_for_threshold(
                pipe_template=pipe_template,
                X_tr=X_tr, y_tr=y_tr,
                inner_splits=inner_splits,
                seed=seed + 10_000 * fold_id,
                show_progress=show_progress,
                desc=f"innerCV ({model_name})",
            )
            thr = threshold_for_fixed_sensitivity(y_tr, p_tr_oof, target_sens=target_sens)

            # Fit full train, eval test
            pipe = clone(pipe_template)
            pipe.fit(X_tr, y_tr)
            p_te = predict_proba_safe(pipe, X_te)

            tp, fp, tn, fn = confusion_from_threshold(y_te, p_te, thr)
            sens, spec = sens_spec_from_conf(tp, fp, tn, fn)
            ppv_obs = ppv_from_conf(tp, fp)
            ppv_std = ppv_standardized(sens, spec, pi_ref=pi_ref)

            rows.append({
                "fold": fold_id,
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
            })

            # ---- Update overall ETA ----
            overall_eta.step(1)
            t_step = _now() - t_step0

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
            n_folds=("fold", "count"),
            ppv_std_mean=("ppv_standardized_pi_ref", "mean"),
            ppv_std_std=("ppv_standardized_pi_ref", "std"),
            ppv_obs_mean=("ppv_observed_test", "mean"),
            sens_mean=("sens_test", "mean"),
            spec_mean=("spec_test", "mean"),
            thr_mean=("thr_from_train_inner_oof", "mean"),
        )
    )
    agg["ppv_std_sem"] = agg["ppv_std_std"] / np.sqrt(np.maximum(agg["n_folds"], 1))
    agg.to_csv(outdir / "ppv_std_aggregated.csv", index=False)

    print(f"\n[OK] {data_path}")
    print(f"  Target = {tcol}")
    print(f"  pi_ref = {pi_ref:.8f}")
    print(f"  Total time = {_fmt_seconds(overall_eta.elapsed())}")
    print(f"  Wrote:\n    - {outdir/'ppv_std_per_fold.csv'}\n    - {outdir/'ppv_std_aggregated.csv'}")
    return per_fold, agg


# -----------------------------
# RUN on your two uploaded files
# -----------------------------
FULL_PATH = "full_analytic_dataset_mortality_all_admissions.csv"
DEMO_PATH = "demo_analytic_dataset_mortality_all_admissions.csv"

# โหลดเฉพาะคอลัมน์ label_mortality เพื่อหา pi_ref ให้ไว + มี progress แบบ download
tmp_full = read_csv_with_progress(FULL_PATH, usecols=["label_mortality"], desc="Loading FULL for pi_ref")
pi_ref_full = float(tmp_full["label_mortality"].mean())
del tmp_full

# รัน full
per_fold_full, agg_full = run_ppv_std(
    data_path=FULL_PATH,
    outdir="outputs/out_full_ppvstd",
    pi_ref=pi_ref_full,
    target="label_mortality",
    target_sens=0.80,
    fast=False,          # ถ้าช้าไป ให้เปลี่ยนเป็น True
    show_progress=True,  # <--- progress + ETA
)

# รัน demo (ใช้ pi_ref เดียวกับ full เพื่อ compare)
per_fold_demo, agg_demo = run_ppv_std(
    data_path=DEMO_PATH,
    outdir="outputs/out_demo_ppvstd",
    pi_ref=pi_ref_full,
    target="label_mortality",
    target_sens=0.80,
    fast=False,          # ถ้าช้าไป ให้เปลี่ยนเป็น True
    show_progress=True,  # <--- progress + ETA
)

display(agg_full.sort_values("ppv_std_mean", ascending=False))
display(agg_demo.sort_values("ppv_std_mean", ascending=False))


