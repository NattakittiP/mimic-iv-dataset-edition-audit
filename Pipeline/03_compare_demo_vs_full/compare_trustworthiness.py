# compare_trustworthiness.py
#
# QUESTION ANSWERED
#   "Is the real MIMIC-IV Demo a TYPICAL Demo-sized draw from Full?"  For each
#   metric / ranking / best-model decision / feature-importance profile we
#   compare the value obtained on the real Demo with a NULL distribution built
#   from many Demo-sized subsamples of Full, each evaluated with exactly the
#   same procedure (same models, same CV, same calibration) as Demo itself.
#
# Usage (subsample size defaults to the Demo row count -- leave it unset):
#   python compare_trustworthiness.py \
#       --full_path full_analytic_dataset_mortality_all_admissions.csv \
#       --demo_path demo_analytic_dataset_mortality_all_admissions.csv \
#       --target_col label_mortality --svc_max_train_n 20000 --n_jobs 4 \
#       --outdir results/compare/hosp/trustworthiness
#
# TWO NULL DISTRIBUTIONS (--null_modes, default both)
#   random             : sample whole PATIENTS of Full at random until >= n rows
#                        (n = Demo rows). NOTE: this samples ALL Full patients;
#                        every official Demo patient has an ICU stay (vs ~17% of
#                        Full patients), so this null does not
#                        reproduce the Demo's patient selection. Draws with fewer than
#                        min_per_class rows of either class cannot be evaluated
#                        with 5-fold CV and are redrawn; the acceptance rate is
#                        reported (run_info.json, `attempts` column) because
#                        the null is conditional on that.
#   prevalence_matched : patients with >=1 positive row are sampled until the
#                        subsample holds at least as many positive rows as the
#                        Demo, then negative-only patients until >= n rows.
#                        Separates "Demo differs because of its prevalence"
#                        from "Demo differs beyond its prevalence" (hosp: Demo
#                        mortality 5.5% vs Full ~2.0%, so a purely random
#                        Demo-sized draw holds ~3x fewer deaths than the Demo).
#                        Whole patients are kept, so when positive patients
#                        also carry negative rows (ED: repeat visits) the draw
#                        can exceed n rows and fall short of Demo's prevalence;
#                        the achieved rows / positives / prevalence of every
#                        draw are recorded (run_info.json, per-run columns).
#                        Note for ED: 62 of the 64 ED-Demo patients have >= 1
#                        admitted stay (vs ~52% of Full ED patients; ~96% among
#                        Full ED patients with an ICU stay). This null matches
#                        the positive-row count only, not the ICU-patient case mix.
#   Outputs of the random null have no suffix; outputs of the
#   matched null carry the suffix `_prevmatched`.
#
# NULL POOL (--null_pool)
#   all (default)      : both nulls sample from ALL Full patients. Behaviour,
#                        outputs and the checkpoint signature are the same as
#                        when the option is not given.
#   icu                : both nulls sample ONLY Full patients that have >= 1 ICU
#                        stay (--icu_subjects_path: a CSV with a subject_id column,
#                        made by 01_prepare_data/make_icu_subject_list.py from the
#                        official icu/icustays.csv.gz). Every official Demo patient
#                        has an ICU stay, so this null mirrors how the Demo was
#                        selected. Whole patients, all of their rows (incl. their
#                        non-ICU admissions / ED stays), exactly like the Demo.
#                        Additionally computed in this mode ("Full-ICU reference"):
#                        the same 7-model evaluation and the permutation importance
#                        on the WHOLE ICU-patient part of Full, saved as
#                        full_icu_reference_metrics.csv / importance_fullicu_vs_demo.csv,
#                        and exp2/exp3/exp4 against that reference
#                        (*_vs_fullicu*.csv). exp1/exp2/exp3/exp4 against the Full
#                        reference keep their usual file names. The ICU list only
#                        selects which patients may be sampled; it is never a
#                        feature (split_X_y is unchanged). Use a separate --outdir.
#
# EXPERIMENTS
#   exp1  per model x metric: Demo value vs null quantiles; mid-rank percentile
#         (in the "better" direction) and a two-sided empirical p-value.
#   exp2  Spearman correlation of the model ordering (primary metric) with the
#         Full reference ordering: Demo value vs null; top-k overlap.
#   exp3  does the best model on Demo / on each null draw equal the best model
#         on Full? Null probability with a Wilson 95% CI.
#   exp4  feature-importance agreement with Full: cross-validated permutation
#         importance of a logistic regression (drop in AUROC of the POOLED
#         held-out predictions of grouped CV folds when a feature is permuted);
#         Full reference computed once on Full; observed = Spearman(Demo, Full),
#         null = Spearman(subsample, Full) for every null draw.
#
# DESIGN NOTES (statistical safeguards):
#   * prevalence-matched null added; subsample size defaults to Demo rows.
#   * calibration: CalibratedClassifierCV(ensemble=False) with GROUP-AWARE
#     inner splits computed on each training fold (before: row-level inner
#     folds that split a patient across calibration folds). SVC is fitted with
#     probability=False when it is wrapped by the calibrator (no double Platt
#     scaling) and probability=True only when it is not.
#   * no silent in-sample fallback: a dataset that cannot be cross-validated
#     yields NaN metrics (and a warning) instead of optimistic in-sample ones;
#     no silent row-level fallback subsample after failed retries (error).
#   * exp4 rebuilt: before it correlated IN-SAMPLE importances of each
#     subsample with Demo's, which does not measure whether Demo is typical.
#   * exp3 null probability uses a Wilson interval (quantiles of 0/1 hits are
#     meaningless); ranks are float and NaN-safe.
#   * fresh estimator per fold (sklearn.clone); the subsample of draw r is
#     drawn with a generator seeded by (seed, null mode, r) and its models/CV
#     use seed + r, so every draw is independent of order, chunking, n_jobs
#     and resume (workers run single-threaded BLAS for the same reason).
#   * parallel over null draws (--n_jobs; each worker only receives its small
#     subsample) and checkpoint/resume (<outdir>/_checkpoint): re-running the
#     same command continues where it stopped; a changed setting, data file or
#     package version is refused so results are never mixed.
#   * Demo and Full must have identical feature columns.

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # never open windows (TkAgg crashed unattended Windows runs before)
import matplotlib.pyplot as plt

import sklearn
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, StratifiedShuffleSplit, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

SCRIPT_VERSION = "trust-v2.2-2026-09-23"
NULL_MODES = ("random", "prevalence_matched")
MODE_CODE = {"random": 1, "prevalence_matched": 2}
MODE_SUFFIX = {"random": "", "prevalence_matched": "_prevmatched"}
METRICS = ["AUROC", "AUPRC", "LogLoss", "Brier", "ECE"]
HIGHER_BETTER = {"AUROC", "AUPRC"}


# -----------------------------
# CONFIG
# -----------------------------
@dataclass
class Config:
    full_path: str = "full_analytic_dataset_mortality_all_admissions.csv"
    demo_path: str = "demo_analytic_dataset_mortality_all_admissions.csv"

    outdir: Path = Path("outputs_demo_trust")
    plots_dir: Path = Path("outputs_demo_trust") / "plots"

    n_subsamples: int = 1000
    subsample_n: Optional[int] = None  # None -> number of Demo rows (recommended)
    n_splits: int = 5
    seed: int = 42

    target_col: Optional[str] = None

    # "sigmoid", "isotonic", or None
    calibration: Optional[str] = "sigmoid"
    calibration_cv: int = 3

    primary_metric: str = "AUROC"  # AUROC, AUPRC, LogLoss, Brier, ECE
    topk_list: Tuple[int, ...] = (1, 2, 3)

    run_importance_stability: bool = True
    perm_repeats: int = 10
    importance_test_cap: int = 20000  # rows of each held-out fold used for Full's permutation importance

    progress_every: int = 10

    min_per_class_in_subsample: Optional[int] = None
    subsample_max_retries: int = 200

    group_col: str = "subject_id"
    cv_mode: str = "group"  # "group" or "row"

    # Cap SVC_RBF's training-fold size (stratified row subsample of the
    # TRAINING fold only). Same rule as run_rsce.py / run_ppv.py; no effect on
    # Demo-sized data.
    svc_max_train_n: Optional[int] = None

    null_modes: Tuple[str, ...] = NULL_MODES
    n_jobs: int = 4

    # "all": null draws from all Full patients (default, original analysis);
    # "icu": only Full patients with >= 1 ICU stay (see header, NULL POOL).
    null_pool: str = "all"
    icu_subjects_path: Optional[str] = None


CFG = Config()


# -----------------------------
# Utilities
# -----------------------------
def ensure_outdirs(cfg: Config) -> None:
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)


def find_target_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "label_mortality", "label_ed_admit",
        "mortality", "Mortality", "death", "Death", "outcome", "Outcome",
        "label", "Label", "target", "Target", "y", "Y",
        "hospital_expire_flag", "HOSPITAL_EXPIRE_FLAG",
    ]
    for c in candidates:
        if c in df.columns and df[c].dropna().nunique() == 2:
            return c
    return None


def coerce_binary_y(y: pd.Series | np.ndarray) -> np.ndarray:
    """Make y strictly {0,1}; the smaller sorted value maps to 0."""
    y_ser = pd.Series(y).reset_index(drop=True)
    if y_ser.isna().any():
        raise ValueError(f"Target has {int(y_ser.isna().sum())} missing values.")
    uniq = list(pd.unique(y_ser))
    if len(uniq) != 2:
        raise ValueError(f"Target must be binary. Found unique={uniq[:10]} (n={len(uniq)})")
    try:
        uniq_sorted = sorted(uniq)
    except TypeError:
        y_ser = y_ser.astype(str)
        uniq_sorted = sorted(pd.unique(y_ser))
    mapping = {uniq_sorted[0]: 0, uniq_sorted[1]: 1}
    return y_ser.map(mapping).to_numpy().astype(int)


# ID-like columns never used as features (needed only for grouping).
ID_LIKE_COLS = ("subject_id", "hadm_id", "stay_id")
# Non-ID columns never used as features, matching run_rsce.py's hosp
# --drop_cols: discharge_location is post-outcome label leakage; anchor_year /
# anchor_year_group are date-shift bookkeeping. Ignored if absent (ED).
NON_FEATURE_COLS = ("discharge_location", "anchor_year", "anchor_year_group")


def split_X_y(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
    y = coerce_binary_y(df[target_col].values)
    drop_cols = [target_col] + [c for c in ID_LIKE_COLS + NON_FEATURE_COLS if c in df.columns]
    X = df.drop(columns=drop_cols).reset_index(drop=True)
    return X, y


def get_groups(df: pd.DataFrame, cfg: Config) -> Optional[np.ndarray]:
    if cfg.cv_mode != "group":
        return None
    if cfg.group_col not in df.columns:
        raise ValueError(f"--group_col '{cfg.group_col}' not found in the data (use --cv_mode row to "
                         f"run row-level on purpose).")
    return df[cfg.group_col].to_numpy()


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(transformers=[("num", numeric, num_cols), ("cat", categorical, cat_cols)],
                             remainder="drop")


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def choose_safe_group_folds(y: np.ndarray, groups: Optional[np.ndarray], desired: int) -> int:
    """Largest k <= desired such that every class has >= k rows (and >= k patients)."""
    y = np.asarray(y)
    vals, cnts = np.unique(y, return_counts=True)
    if len(vals) < 2:
        return 0
    safe = min(desired, int(cnts.min()))
    if groups is not None:
        groups = np.asarray(groups)
        safe = min(safe, min(int(len(np.unique(groups[y == v]))) for v in vals))
    return int(safe) if safe >= 2 else 0


def _split(X, y, groups, n_splits, seed):
    if groups is not None:
        return list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y, groups))
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y))


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows antivirus/indexer can briefly lock the target
            if attempt == 19:
                raise
            time.sleep(0.5 * (attempt + 1))


# -----------------------------
# Progress helpers
# -----------------------------
def _fmt_seconds(sec: float) -> str:
    if sec is None or not np.isfinite(sec) or sec < 0:
        return "?:??"
    sec = int(round(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:d}:{s:02d}"


def _progress_line(label: str, done: int, total: int, t0: float, done_before: int = 0) -> str:
    elapsed = time.time() - t0
    new = max(done - done_before, 1)
    remaining = (total - done) * elapsed / new
    pct = (done / total) * 100.0 if total > 0 else 0.0
    bar_len = 24
    filled = min(max(int(round(bar_len * done / total)) if total > 0 else 0, 0), bar_len)
    bar = "#" * filled + "." * (bar_len - filled)
    return (f"[{label}] {pct:6.2f}% |{bar}| {done}/{total} "
            f"Elapsed {_fmt_seconds(elapsed)} ETA {_fmt_seconds(remaining)}")


# -----------------------------
# Checkpoint
# -----------------------------
class Checkpoint:
    """Tiny file-based checkpoint under <outdir>/_checkpoint, guarded by a config signature."""

    def __init__(self, root: Path, signature: Dict[str, Any]):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        sig = json.loads(json.dumps(signature, sort_keys=True, default=str))
        sp = self.root / "checkpoint_config.json"
        if sp.exists():
            old = json.loads(sp.read_text(encoding="utf-8"))
            if old != sig:
                diff = sorted(k for k in set(old) | set(sig) if old.get(k) != sig.get(k))
                raise RuntimeError(
                    f"Checkpoint in {self.root} was made with different settings/data/code ({diff}). "
                    f"Refusing to mix results. Delete that _checkpoint folder (or use another --outdir) "
                    f"to start over.")
            print(f"[checkpoint] resuming from {self.root}")
        else:
            _atomic_write_text(sp, json.dumps(sig, indent=2, sort_keys=True))

    def load_df(self, name: str) -> Optional[pd.DataFrame]:
        # round_trip: floats read back bit-identical to what was written, so a
        # resumed run gives exactly the same percentiles/ties as an uninterrupted one
        p = self.root / f"{name}.csv"
        return pd.read_csv(p, float_precision="round_trip") if p.exists() else None

    def save_df(self, name: str, df: pd.DataFrame) -> None:
        _atomic_write_text(self.root / f"{name}.csv", df.to_csv(index=False))

    def load_json(self, name: str) -> Optional[Any]:
        p = self.root / f"{name}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def save_json(self, name: str, obj: Any) -> None:
        _atomic_write_text(self.root / f"{name}.json", json.dumps(obj, indent=2))


# -----------------------------
# MLP: early stopping on the validation LOG-LOSS
# -----------------------------
# A plain MLPClassifier((128,64), max_iter=400,
# early_stopping=True). sklearn's early stopping monitors validation ACCURACY,
# which on an imbalanced outcome is flat at the majority rate from epoch 1, so on
# Demo-sized data the MLP stopped almost untrained (Demo AUROC 0.47 hosp / 0.52 ED)
# while on Full it was the ED Full-best model -> exp3 for ED was driven by that
# artefact. The MLP is now the SAME EarlyStoppedMLP as in run_rsce.py / run_ppv.py
# (256-128-64, adam, alpha 1e-4, stratified 10% validation split of the training
# fold, patience 10, best epoch restored). Checkpointed rows of a model whose
# settings changed are retired (see _retire_stale_model_rows) and only that model
# is recomputed; every other row is reused unchanged.
MODEL_PARAMS_CHANGED_SINCE_LEGACY = {"MLP"}  # checkpoints written before model_params.json existed


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


def zoo_params_sha(seed: int) -> Dict[str, str]:
    """sha256 of (class, get_params) per model of the zoo (n_jobs excluded: speed-only)."""
    out = {}
    for ms in get_model_zoo(seed=seed, n_jobs=1):
        prm = {k: v for k, v in ms.estimator.get_params(deep=False).items() if k != "n_jobs"}
        rec = json.dumps({"class": type(ms.estimator).__name__, "params": prm}, sort_keys=True, default=repr)
        out[ms.name] = hashlib.sha256(rec.encode("utf-8")).hexdigest()
    return out


# -----------------------------
# Model zoo
# -----------------------------
@dataclass
class ModelSpec:
    name: str
    estimator: object


def get_model_zoo(seed: int = 42, n_jobs: int = 1) -> List[ModelSpec]:
    # SVC is created with probability=False; _fit_one switches it to
    # probability=True only when it is NOT wrapped by CalibratedClassifierCV.
    return [
        ModelSpec("Logistic_L2", LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced")),
        ModelSpec("RandomForest", RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=n_jobs,
                                                         class_weight="balanced_subsample", min_samples_leaf=2)),
        ModelSpec("ExtraTrees", ExtraTreesClassifier(n_estimators=600, random_state=seed, n_jobs=n_jobs,
                                                     class_weight="balanced", min_samples_leaf=2)),
        ModelSpec("GradientBoosting", GradientBoostingClassifier(random_state=seed)),
        ModelSpec("GaussianNB", GaussianNB()),
        ModelSpec("MLP", EarlyStoppedMLP(hidden_layer_sizes=(256, 128, 64), max_iter=2500, validation_fraction=0.1,
                                         n_iter_no_change=10, random_state=seed)),
        ModelSpec("SVC_RBF", SVC(C=2.0, kernel="rbf", probability=False, class_weight="balanced",
                                 random_state=seed)),
    ]


# -----------------------------
# Null-distribution sampler (patient level)
# -----------------------------
class GroupSampler:
    """Draws Demo-sized subsamples of Full by whole patients (or rows if groups is None)."""

    def __init__(self, y: np.ndarray, groups: Optional[np.ndarray]):
        y = np.asarray(y).astype(int)
        self.y = y
        if groups is None:
            codes = np.arange(len(y))
        else:
            codes, _ = pd.factorize(pd.Series(groups), sort=False)
        self.order = np.argsort(codes, kind="stable")
        self.counts = np.bincount(codes)
        self.starts = np.concatenate([[0], np.cumsum(self.counts)[:-1]])
        self.pos_count = np.bincount(codes, weights=y).astype(int)
        self.n_groups = len(self.counts)
        self.pos_groups = np.flatnonzero(self.pos_count > 0)
        self.neg_groups = np.flatnonzero(self.pos_count == 0)

    def _rows(self, gids: np.ndarray) -> np.ndarray:
        return np.sort(np.concatenate([self.order[self.starts[g]:self.starts[g] + self.counts[g]] for g in gids]))

    @staticmethod
    def _take_until(perm: np.ndarray, sizes: np.ndarray, need: int) -> np.ndarray:
        if need <= 0:
            return perm[:0]
        cum = np.cumsum(sizes[perm])
        k = int(np.searchsorted(cum, need, side="left")) + 1
        return perm[:min(k, len(perm))]

    def draw_random(self, n: int, rng: np.random.Generator) -> np.ndarray:
        perm = rng.permutation(self.n_groups)
        return self._rows(self._take_until(perm, self.counts, n))

    def draw_prevalence_matched(self, n: int, target_pos: int, rng: np.random.Generator) -> np.ndarray:
        pp = rng.permutation(self.pos_groups)
        chosen_pos = self._take_until(pp, self.pos_count, target_pos)
        rows_pos = int(self.counts[chosen_pos].sum())
        nn = rng.permutation(self.neg_groups)
        chosen_neg = self._take_until(nn, self.counts, n - rows_pos)
        return self._rows(np.concatenate([chosen_pos, chosen_neg]))

    def draw(self, mode: str, n: int, target_pos: int, min_per_class: int, rng: np.random.Generator,
             max_retries: int) -> Tuple[np.ndarray, int]:
        for attempt in range(1, max_retries + 1):
            idx = self.draw_random(n, rng) if mode == "random" else self.draw_prevalence_matched(n, target_pos, rng)
            yy = self.y[idx]
            n1 = int(yy.sum())
            if n1 >= min_per_class and (len(yy) - n1) >= min_per_class:
                return idx, attempt
        raise RuntimeError(
            f"[{mode}] could not draw a subsample with >= {min_per_class} rows of each class in "
            f"{max_retries} attempts (n={n}). The outcome is too rare for a Demo-sized CV at this n.")


# -----------------------------
# Fit + OOF prediction
# -----------------------------
def _stratified_fit_cap_idx(y_tr: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Class-preserving random subsample (positions) of a TRAINING fold."""
    rng = np.random.default_rng(seed)
    y_tr = np.asarray(y_tr)
    frac = cap / len(y_tr)
    parts = []
    for v in np.unique(y_tr):
        idx_v = np.flatnonzero(y_tr == v)
        take = min(len(idx_v), max(1, int(round(len(idx_v) * frac))))
        parts.append(rng.choice(idx_v, size=take, replace=False))
    return np.sort(np.concatenate(parts))


def _fit_one(X_tr: pd.DataFrame, y_tr: np.ndarray, g_tr: Optional[np.ndarray], model: ModelSpec,
             calibration: Optional[str], calibration_cv_desired: int, seed: int):
    """Fit a FRESH pipeline on one training fold. Returns (fitted, calibrated_bool)."""
    k_cal = 0
    if calibration in ("sigmoid", "isotonic"):
        k_cal = choose_safe_group_folds(y_tr, g_tr, calibration_cv_desired)
    use_cal = k_cal >= 2
    est = clone(model.estimator)
    if isinstance(est, SVC):
        est.set_params(probability=not use_cal)
    pipe = Pipeline(steps=[("preprocess", build_preprocessor(X_tr)), ("clf", est)])
    if use_cal:
        # Group-aware inner splits computed on THIS training fold; ensemble=False:
        # the calibrator is fitted on out-of-fold predictions of the whole
        # training fold and the final model is refitted on the whole fold.
        splits = _split(X_tr, y_tr, g_tr, k_cal, seed)
        clf = CalibratedClassifierCV(pipe, method=calibration, cv=splits, ensemble=False)
    else:
        clf = pipe
    clf.fit(X_tr, y_tr)
    return clf, use_cal


def fit_predict_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    model: ModelSpec,
    seed: int,
    n_splits_desired: int,
    calibration: Optional[str],
    calibration_cv_desired: int,
    groups: Optional[np.ndarray] = None,
    svc_max_train_n: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Out-of-fold probabilities. Rows of folds that could not be fitted stay NaN."""
    y = np.asarray(y).astype(int)
    info = {"n_splits": 0, "folds_uncalibrated": 0, "folds_failed": 0}
    oof = np.full(len(y), np.nan)
    n_splits = choose_safe_group_folds(y, groups, n_splits_desired)
    if n_splits < 2:
        print(f"[WARNING] {model.name}: too few rows/patients per class for CV -> metrics NaN "
              f"(no in-sample fallback).")
        return oof, info
    info["n_splits"] = n_splits

    for fold_k, (tr_idx, te_idx) in enumerate(_split(X, y, groups, n_splits, seed)):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        g_tr = groups[tr_idx] if groups is not None else None
        if len(np.unique(y_tr)) < 2:
            info["folds_failed"] += 1
            continue
        if model.name == "SVC_RBF" and svc_max_train_n and len(y_tr) > svc_max_train_n:
            keep = _stratified_fit_cap_idx(y_tr, svc_max_train_n, seed + fold_k)
            X_tr, y_tr = X_tr.iloc[keep], y_tr[keep]
            g_tr = g_tr[keep] if g_tr is not None else None
        try:
            clf, calibrated = _fit_one(X_tr, y_tr, g_tr, model, calibration, calibration_cv_desired,
                                       seed + 1000 + fold_k)
        except Exception as e:  # rare degenerate folds on tiny data; never silently optimistic
            print(f"[WARNING] {model.name} fold {fold_k}: fit failed ({type(e).__name__}: {e}) -> NaN")
            info["folds_failed"] += 1
            continue
        if calibration in ("sigmoid", "isotonic") and not calibrated:
            info["folds_uncalibrated"] += 1
        oof[te_idx] = clf.predict_proba(X.iloc[te_idx])[:, 1]
        del clf
    return oof, info


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if (not np.all(np.isfinite(y_prob))) or len(np.unique(y_true)) < 2:
        return {m: float("nan") for m in METRICS}
    p = np.clip(y_prob, 1e-15, 1 - 1e-15)
    return {
        "AUROC": float(roc_auc_score(y_true, y_prob)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "LogLoss": float(log_loss(y_true, p, labels=[0, 1])),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "ECE": float(expected_calibration_error(y_true, y_prob, n_bins=10)),
    }


def eval_dataset_once(X: pd.DataFrame, y: np.ndarray, groups: Optional[np.ndarray], cfg: Config, seed: int,
                      n_jobs: int, label: str, ckpt: Optional[Checkpoint] = None) -> pd.DataFrame:
    done = ckpt.load_df(f"{label}_metrics_partial") if ckpt is not None else None
    rows = done.to_dict("records") if done is not None else []
    have = {r["model"] for r in rows}
    for ms in get_model_zoo(seed=seed, n_jobs=n_jobs):
        if ms.name in have:
            continue
        t0 = time.time()
        oof, info = fit_predict_oof(X, y, ms, seed=seed, n_splits_desired=cfg.n_splits,
                                    calibration=cfg.calibration, calibration_cv_desired=cfg.calibration_cv,
                                    groups=groups, svc_max_train_n=cfg.svc_max_train_n)
        rows.append({"model": ms.name, **compute_metrics(y, oof), **info})
        print(f"[{label}] {ms.name:<17s} AUROC={rows[-1]['AUROC']:.4f}  ({_fmt_seconds(time.time() - t0)})",
              flush=True)
        if ckpt is not None:
            ckpt.save_df(f"{label}_metrics_partial", pd.DataFrame(rows))
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


# -----------------------------
# Importance (held-out permutation importance of a logistic regression)
# -----------------------------
def fit_importance_cv(X: pd.DataFrame, y: np.ndarray, groups: Optional[np.ndarray], cfg: Config, seed: int,
                      test_cap: Optional[int] = None) -> Dict[str, float]:
    """Cross-validated permutation importance of a logistic regression, on POOLED held-out predictions.

    One model per grouped CV fold predicts its own held-out rows; importance of a
    feature = AUROC(pooled out-of-fold predictions) - AUROC(pooled out-of-fold
    predictions after permuting that feature within each held-out fold),
    averaged over perm_repeats. Pooling uses every positive of the dataset in
    one AUROC (a Demo-sized fold alone holds only ~1-3 deaths). test_cap limits
    the held-out rows per fold (Full only; class-preserving subsample).
    """
    y = np.asarray(y).astype(int)
    k = choose_safe_group_folds(y, groups, cfg.n_splits)
    if k < 2:
        return {}
    fitted = []  # (model, held-out X, held-out y) per fold
    for fold_k, (tr, te) in enumerate(_split(X, y, groups, k, seed)):
        if len(np.unique(y[tr])) < 2:
            return {}
        pipe = Pipeline([("preprocess", build_preprocessor(X)),
                         ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs"))])
        pipe.fit(X.iloc[tr], y[tr])
        if test_cap and len(te) > test_cap:
            te = te[_stratified_fit_cap_idx(y[te], test_cap, seed + 77 + fold_k)]
        fitted.append((pipe, X.iloc[te].reset_index(drop=True), y[te]))
    y_all = np.concatenate([f[2] for f in fitted])
    if len(np.unique(y_all)) < 2:
        return {}
    base = roc_auc_score(y_all, np.concatenate([m.predict_proba(Xt)[:, 1] for m, Xt, _ in fitted]))
    rng = np.random.default_rng(seed)
    out: Dict[str, float] = {}
    for col in X.columns:
        drops = []
        for _ in range(cfg.perm_repeats):
            preds = []
            for m, Xt, _ in fitted:
                Xp = Xt.copy()
                Xp[col] = rng.permutation(Xp[col].to_numpy())
                preds.append(m.predict_proba(Xp)[:, 1])
            drops.append(base - roc_auc_score(y_all, np.concatenate(preds)))
        out[col] = float(np.mean(drops))
    return out


def spearman_corr_dict(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        return float("nan")
    va = pd.Series([a[k] for k in keys], dtype=float)
    vb = pd.Series([b[k] for k in keys], dtype=float)
    ok = va.notna() & vb.notna()
    if ok.sum() < 2 or va[ok].nunique() < 2 or vb[ok].nunique() < 2:
        return float("nan")
    return float(va[ok].corr(vb[ok], method="spearman"))


# -----------------------------
# One null draw (runs in a worker process)
# -----------------------------
def _eval_subsample_run(run_id: int, X: pd.DataFrame, y: np.ndarray, g: Optional[np.ndarray], cfg: Config,
                        full_imp: Optional[Dict[str, float]], diag: Dict[str, Any],
                        models: Optional[List[str]] = None, imp_known: Optional[float] = None,
                        icu_imp: Optional[Dict[str, float]] = None, imp_icu_known: Optional[float] = None
                        ) -> List[Dict[str, Any]]:
    warnings.filterwarnings("ignore")
    # single-threaded BLAS/OpenMP inside the worker: results then do not depend on --n_jobs
    with threadpool_limits(limits=1):
        return _eval_subsample_run_inner(run_id, X, y, g, cfg, full_imp, diag, models, imp_known,
                                         icu_imp, imp_icu_known)


def _eval_subsample_run_inner(run_id, X, y, g, cfg, full_imp, diag, models=None, imp_known=None,
                              icu_imp=None, imp_icu_known=None):
    """models=None: the whole zoo + importance. models=[...]: only those models (a draw whose other
    models are already checkpointed); the draw's importance correlation imp_known is reused.
    Every model uses seed + run_id and the same draw, so its rows do not depend on which other
    models are computed in the same call."""
    seed = cfg.seed + run_id
    rows = []
    for ms in get_model_zoo(seed=seed, n_jobs=1):
        if models is not None and ms.name not in models:
            continue
        oof, info = fit_predict_oof(X, y, ms, seed=seed, n_splits_desired=cfg.n_splits,
                                    calibration=cfg.calibration, calibration_cv_desired=cfg.calibration_cv,
                                    groups=g, svc_max_train_n=cfg.svc_max_train_n)
        rows.append({"run_id": run_id, "model": ms.name, **compute_metrics(y, oof), **info})
    imp_corr = float("nan")
    imp_icu_corr = float("nan")
    if models is not None:
        imp_corr = float("nan") if imp_known is None else imp_known
        imp_icu_corr = float("nan") if imp_icu_known is None else imp_icu_known
    elif full_imp:
        # ONE importance vector per draw, correlated with each reference (no second fit)
        imp = fit_importance_cv(X, y, g, cfg, seed=seed + 999)
        imp_corr = spearman_corr_dict(imp, full_imp) if imp else float("nan")
        if icu_imp:
            imp_icu_corr = spearman_corr_dict(imp, icu_imp) if imp else float("nan")
    for r in rows:
        r.update(diag)
        r["imp_spearman_vs_full"] = imp_corr
        if icu_imp is not None:  # column exists only in --null_pool icu runs
            r["imp_spearman_vs_full_icu"] = imp_icu_corr
    return rows


def _retire_stale_model_rows(ckpt: Checkpoint, current: Dict[str, str]) -> None:
    """Drop checkpointed rows of models whose settings changed, so only those are recomputed.

    <ckpt>/model_params.json records the zoo (sha per model) that wrote the partial CSVs.
    Checkpoints written before it existed are treated as made with the earlier zoo,
    which differs only in MODEL_PARAMS_CHANGED_SINCE_LEGACY. The original CSVs are kept in
    <ckpt>/replaced_units/ before they are rewritten."""
    saved = ckpt.load_json("model_params")
    partials = sorted(ckpt.root.glob("*_partial.csv"))
    if saved is None:
        stale = set(MODEL_PARAMS_CHANGED_SINCE_LEGACY) if partials else set()
    else:
        stale = {m for m, sha in current.items() if saved.get(m) != sha}
    if stale:
        bak = ckpt.root / "replaced_units"
        bak.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for pth in partials:
            df = pd.read_csv(pth, float_precision="round_trip")
            if "model" not in df.columns or not df["model"].isin(stale).any():
                continue
            _atomic_write_text(bak / f"{pth.stem}.{stamp}.csv", pth.read_text(encoding="utf-8"))
            kept = df[~df["model"].isin(stale)]
            ckpt.save_df(pth.stem, kept)
            print(f"[checkpoint] {pth.name}: {len(df) - len(kept)} row(s) of {sorted(stale)} made with other "
                  f"model settings -> will be recomputed (old file kept in {bak})")
    ckpt.save_json("model_params", current)


def run_subsample_experiments(X_full: pd.DataFrame, y_full: np.ndarray, groups_full: Optional[np.ndarray],
                              sampler: GroupSampler, cfg: Config, mode: str, n: int, target_pos: int,
                              min_per_class: int, full_imp: Optional[Dict[str, float]],
                              ckpt: Checkpoint, icu_imp: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    name = f"subsample_{mode}_partial"
    prev = ckpt.load_df(name)
    rows: List[Dict[str, Any]] = prev.to_dict("records") if prev is not None else []
    zoo_names = [ms.name for ms in get_model_zoo(seed=cfg.seed, n_jobs=1)]
    have: Dict[int, set] = {}
    imp_of: Dict[int, float] = {}
    imp_icu_of: Dict[int, float] = {}
    for r in rows:
        have.setdefault(int(r["run_id"]), set()).add(r["model"])
        imp_of[int(r["run_id"])] = r.get("imp_spearman_vs_full", float("nan"))
        imp_icu_of[int(r["run_id"])] = r.get("imp_spearman_vs_full_icu", float("nan"))
    missing = {r: [m for m in zoo_names if m not in have.get(r, set())] for r in range(cfg.n_subsamples)}
    todo = [r for r in range(cfg.n_subsamples) if missing[r]]
    total = cfg.n_subsamples
    done_before = total - len(todo)
    n_partial = sum(1 for r in todo if r in have)
    if done_before or n_partial:
        print(f"[{mode}] {done_before} null draws already done (checkpoint); {len(todo)} to go"
              + (f" ({n_partial} of them only for model(s) "
                 f"{sorted(set(m for r in todo if r in have for m in missing[r]))})" if n_partial else ""))
    chunk = max(1, cfg.n_jobs) * 4
    t0 = time.time()
    last_print = done_before
    with Parallel(n_jobs=cfg.n_jobs) as par:
        for start in range(0, len(todo), chunk):
            ids = todo[start:start + chunk]
            tasks = []
            for run_id in ids:
                rng = np.random.default_rng([cfg.seed, MODE_CODE[mode], run_id])
                idx, attempts = sampler.draw(mode, n, target_pos, min_per_class, rng, cfg.subsample_max_retries)
                y_s = y_full[idx]
                g_s = groups_full[idx] if groups_full is not None else None
                diag = {"null_mode": mode, "n_rows": int(len(idx)), "n_pos": int(y_s.sum()),
                        "n_patients": int(len(np.unique(g_s))) if g_s is not None else int(len(idx)),
                        "attempts": int(attempts)}
                partial = run_id in have
                tasks.append(delayed(_eval_subsample_run)(run_id, X_full.iloc[idx].reset_index(drop=True),
                                                          y_s, g_s, cfg, full_imp, diag,
                                                          missing[run_id] if partial else None,
                                                          imp_of.get(run_id) if partial else None,
                                                          icu_imp,
                                                          imp_icu_of.get(run_id) if partial else None))
            for res in par(tasks):
                rows.extend(res)
            ckpt.save_df(name, pd.DataFrame(rows))
            done = done_before + start + len(ids)
            if done - last_print >= cfg.progress_every or done == total:
                print(_progress_line(f"Null:{mode}", done, total, t0, done_before), flush=True)
                last_print = done
    df = pd.DataFrame(rows)
    df = df[df["run_id"] < cfg.n_subsamples]
    return df.sort_values(["run_id", "model"]).reset_index(drop=True)


# -----------------------------
# Statistics helpers
# -----------------------------
def _empirical_position(null: np.ndarray, obs: float, higher_better: bool) -> Tuple[float, float]:
    """(mid-rank percentile of obs in the 'better' direction, two-sided empirical p)."""
    v = np.asarray(null, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0 or not np.isfinite(obs):
        return float("nan"), float("nan")
    below, above, ties = (v < obs).mean(), (v > obs).mean(), (v == obs).mean()
    pct = 100.0 * ((below if higher_better else above) + 0.5 * ties)
    n = len(v)
    p_lo = (1 + np.sum(v <= obs)) / (n + 1)
    p_hi = (1 + np.sum(v >= obs)) / (n + 1)
    return float(pct), float(min(1.0, 2 * min(p_lo, p_hi)))


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _q(a, q):
    a = np.asarray(a, dtype=float)
    return float(np.nanquantile(a, q)) if np.isfinite(a).any() else float("nan")


def topk_set(df_metrics: pd.DataFrame, metric: str, k: int) -> set:
    df = df_metrics.dropna(subset=[metric]).sort_values(metric, ascending=metric not in HIGHER_BETTER)
    return set(df["model"].head(k).tolist())


def best_model(df_metrics: pd.DataFrame, metric: str) -> Optional[str]:
    top = topk_set(df_metrics, metric, 1)
    return next(iter(top)) if top else None


def _spearman_models(a: pd.DataFrame, b: pd.DataFrame, metric: str) -> float:
    m = a[["model", metric]].merge(b[["model", metric]], on="model", suffixes=("_a", "_b")).dropna()
    if len(m) < 3 or m[f"{metric}_a"].nunique() < 2 or m[f"{metric}_b"].nunique() < 2:
        return float("nan")
    return float(m[f"{metric}_a"].corr(m[f"{metric}_b"], method="spearman"))


# -----------------------------
# Experiments
# -----------------------------
def exp1_demo_percentile(demo_metrics: pd.DataFrame, subsample_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in demo_metrics["model"].unique():
        demo_row = demo_metrics.loc[demo_metrics["model"] == model].iloc[0]
        dist = subsample_long[subsample_long["model"] == model]
        for metric in METRICS:
            vals = dist[metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            demo_val = float(demo_row[metric])
            if len(vals) == 0 or not np.isfinite(demo_val):
                continue
            pct, p2 = _empirical_position(vals, demo_val, metric in HIGHER_BETTER)
            rows.append({
                "model": model, "metric": metric, "demo_value": demo_val,
                "n_null_valid": int(len(vals)), "n_null_total": int(len(dist)),
                "subsample_mean": float(np.mean(vals)),
                "subsample_q025": _q(vals, 0.025), "subsample_q50": _q(vals, 0.5), "subsample_q975": _q(vals, 0.975),
                "demo_percentile_(better_direction)": pct,
                "p_two_sided_empirical": p2,
            })
    return pd.DataFrame(rows)


def exp2_rank_stability(full_metrics: pd.DataFrame, demo_metrics: pd.DataFrame, subsample_long: pd.DataFrame,
                        cfg: Config) -> pd.DataFrame:
    pm = cfg.primary_metric
    full_top = {k: topk_set(full_metrics, pm, k) for k in cfg.topk_list}
    demo_sp = _spearman_models(full_metrics, demo_metrics, pm)

    spearmans, overlaps = [], {k: [] for k in cfg.topk_list}
    for _, g in subsample_long.groupby("run_id"):
        spearmans.append(_spearman_models(full_metrics, g, pm))
        for k in cfg.topk_list:
            sub_top = topk_set(g, pm, k)
            overlaps[k].append(len(full_top[k] & sub_top) / k if len(sub_top) == k else np.nan)
    spearmans = np.asarray(spearmans, dtype=float)
    pct, p2 = _empirical_position(spearmans, demo_sp, higher_better=True)

    row_demo = {"summary": "demo_vs_full", "primary_metric": pm, "spearman_rank_corr": demo_sp,
                "demo_spearman_percentile_in_null": pct, "p_two_sided_empirical": p2}
    for k in cfg.topk_list:
        row_demo[f"top{k}_overlap"] = len(full_top[k] & topk_set(demo_metrics, pm, k)) / k
    row_sub = {"summary": "subsample_vs_full_distribution", "primary_metric": pm,
               "n_null_valid": int(np.isfinite(spearmans).sum()),
               "spearman_mean": float(np.nanmean(spearmans)) if np.isfinite(spearmans).any() else np.nan,
               "spearman_q025": _q(spearmans, 0.025), "spearman_q50": _q(spearmans, 0.5),
               "spearman_q975": _q(spearmans, 0.975)}
    for k in cfg.topk_list:
        arr = np.asarray(overlaps[k], dtype=float)
        row_sub[f"top{k}_overlap_mean"] = float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan
        row_sub[f"top{k}_overlap_q025"] = _q(arr, 0.025)
        row_sub[f"top{k}_overlap_q50"] = _q(arr, 0.5)
        row_sub[f"top{k}_overlap_q975"] = _q(arr, 0.975)
    return pd.DataFrame([row_demo, row_sub])


def exp3_decision_stability(full_metrics: pd.DataFrame, demo_metrics: pd.DataFrame, subsample_long: pd.DataFrame,
                            cfg: Config) -> pd.DataFrame:
    pm = cfg.primary_metric
    full_best = best_model(full_metrics, pm)
    demo_best = best_model(demo_metrics, pm)
    hits, cover = [], {k: [] for k in cfg.topk_list}
    for _, g in subsample_long.groupby("run_id"):
        sb = best_model(g, pm)
        if sb is None:
            continue
        hits.append(int(sb == full_best))
        for k in cfg.topk_list:
            cover[k].append(int(full_best in topk_set(g, pm, k)))
    n = len(hits)
    k_hit = int(np.sum(hits))
    lo, hi = _wilson(k_hit, n)
    out = {
        "primary_metric": pm, "full_best_model": full_best, "demo_best_model": demo_best,
        "demo_best_matches_full": int(demo_best == full_best) if demo_best is not None else np.nan,
        "n_null_valid": n,
        "subsample_P(best_matches_full)": k_hit / n if n else np.nan,
        "subsample_P_wilson95_low": lo, "subsample_P_wilson95_high": hi,
    }
    for k in cfg.topk_list:
        kk = int(np.sum(cover[k]))
        out[f"subsample_P(full_best_in_top{k})"] = kk / n if n else np.nan
        l2, h2 = _wilson(kk, n)
        out[f"subsample_P(full_best_in_top{k})_wilson95_low"] = l2
        out[f"subsample_P(full_best_in_top{k})_wilson95_high"] = h2
        out[f"demo_full_best_in_top{k}"] = int(full_best in topk_set(demo_metrics, pm, k))
    return pd.DataFrame([out])


def exp4_summary(demo_corr: float, per_run: pd.DataFrame, col: str = "imp_spearman_vs_full") -> pd.DataFrame:
    v = per_run[col].to_numpy(dtype=float)
    pct, p2 = _empirical_position(v, demo_corr, higher_better=True)
    return pd.DataFrame([{
        "demo_spearman_vs_full": demo_corr,
        "n_null_valid": int(np.isfinite(v).sum()),
        "null_mean": float(np.nanmean(v)) if np.isfinite(v).any() else np.nan,
        "null_q025": _q(v, 0.025), "null_q50": _q(v, 0.5), "null_q975": _q(v, 0.975),
        "demo_percentile_in_null": pct, "p_two_sided_empirical": p2,
    }])


# -----------------------------
# Plots
# -----------------------------
def plot_auroc_hist_per_model(subsample_long: pd.DataFrame, demo_metrics: pd.DataFrame, cfg: Config,
                              n: int, mode: str) -> None:
    sfx = MODE_SUFFIX[mode]
    for model in sorted(subsample_long["model"].unique()):
        vals = subsample_long.loc[subsample_long["model"] == model, "AUROC"].dropna().to_numpy(dtype=float)
        demo_row = demo_metrics.loc[demo_metrics["model"] == model, "AUROC"]
        demo_val = float(demo_row.iloc[0]) if len(demo_row) else float("nan")
        if len(vals) == 0 or not np.isfinite(demo_val):
            continue
        plt.figure(figsize=(7, 4))
        plt.hist(vals, bins=30, color="#8aa9c9")
        plt.axvline(demo_val, linewidth=2, color="#c0392b", label="Demo")
        plt.title(f"AUROC null ({mode}, n~{n}) - {model}")
        plt.xlabel("AUROC")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(cfg.plots_dir / f"auroc_hist_{model}{sfx}.png")
        plt.close()


def plot_importance_stability(per_run: pd.DataFrame, demo_corr: float, cfg: Config, mode: str) -> None:
    vals = per_run["imp_spearman_vs_full"].dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return
    plt.figure(figsize=(7, 4))
    plt.hist(vals, bins=30, color="#8aa9c9")
    if np.isfinite(demo_corr):
        plt.axvline(demo_corr, linewidth=2, color="#c0392b", label="Demo")
        plt.legend()
    plt.title(f"Importance agreement with Full (Spearman), null={mode}")
    plt.xlabel("Spearman correlation with Full importance")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(cfg.plots_dir / f"exp4_importance_stability_hist{MODE_SUFFIX[mode]}.png")
    plt.close()


# -----------------------------
# main
# -----------------------------
def main(cfg: Config) -> None:
    ensure_outdirs(cfg)
    for p in (cfg.full_path, cfg.demo_path):
        if not Path(p).exists():
            raise FileNotFoundError(p)
    for m in cfg.null_modes:
        if m not in NULL_MODES:
            raise ValueError(f"unknown null mode {m!r}; choose from {NULL_MODES}")

    full_df = pd.read_csv(cfg.full_path, low_memory=False)
    demo_df = pd.read_csv(cfg.demo_path, low_memory=False)

    target = cfg.target_col or find_target_column(full_df)
    if target is None or target not in full_df.columns:
        raise ValueError("Target column auto-detection failed; pass --target_col.")
    if target not in demo_df.columns:
        raise ValueError(f"Demo file does not contain target column '{target}'.")

    groups_full = get_groups(full_df, cfg)
    groups_demo = get_groups(demo_df, cfg)
    subj_full = subj_demo = None
    if cfg.null_pool == "icu":
        if cfg.cv_mode != "group" or groups_full is None:
            raise ValueError("--null_pool icu needs patient-grouped CV (--cv_mode group) and the subject_id column.")
        subj_full = pd.to_numeric(full_df[cfg.group_col], errors="raise").astype("int64").to_numpy()
        subj_demo = pd.to_numeric(demo_df[cfg.group_col], errors="raise").astype("int64").to_numpy()
    X_full, y_full = split_X_y(full_df, target)
    X_demo, y_demo = split_X_y(demo_df, target)
    del full_df, demo_df
    if set(X_full.columns) != set(X_demo.columns):
        raise ValueError(f"Demo and Full feature columns differ. Only Full: "
                         f"{sorted(set(X_full.columns) - set(X_demo.columns))}; only Demo: "
                         f"{sorted(set(X_demo.columns) - set(X_full.columns))}. Regenerate the Demo "
                         f"with the current prepare scripts.")
    X_demo = X_demo[list(X_full.columns)]

    n_demo, pos_demo = int(len(y_demo)), int(y_demo.sum())
    n = int(cfg.subsample_n or n_demo)
    if cfg.subsample_n and cfg.subsample_n != n_demo:
        print(f"[WARNING] --subsample_n {cfg.subsample_n} differs from the Demo row count {n_demo}; the "
              f"null then does not describe Demo-sized draws. Leave --subsample_n unset normally.")
    need_for_cal = cfg.calibration_cv if cfg.calibration in ("sigmoid", "isotonic") else 2
    min_per_class = int(cfg.min_per_class_in_subsample or max(cfg.n_splits, need_for_cal, 2))

    print(f"[OK] target={target} | Full n={len(y_full):,} (pos {int(y_full.sum()):,}, "
          f"{y_full.mean():.4f}) | Demo n={n_demo} (pos {pos_demo}, {pos_demo / n_demo:.4f})")
    print(f"[OK] null size n={n}, min_per_class={min_per_class}, modes={list(cfg.null_modes)}, "
          f"n_subsamples={cfg.n_subsamples}, n_jobs={cfg.n_jobs}")
    if pos_demo < min_per_class and "prevalence_matched" in cfg.null_modes:
        raise ValueError(f"Demo has only {pos_demo} positives (< min_per_class={min_per_class}).")

    pool_idx: Optional[np.ndarray] = None
    pool_info: Dict[str, Any] = {}
    if cfg.null_pool == "icu":
        if not cfg.icu_subjects_path or not Path(cfg.icu_subjects_path).exists():
            raise FileNotFoundError(f"--null_pool icu needs --icu_subjects_path (got {cfg.icu_subjects_path!r}).")
        icu_tab = pd.read_csv(cfg.icu_subjects_path, usecols=["subject_id"])
        icu_ids = np.unique(pd.to_numeric(icu_tab["subject_id"], errors="raise").astype("int64").to_numpy())
        pool_idx = np.flatnonzero(np.isin(subj_full, icu_ids))
        if len(pool_idx) == 0:
            raise ValueError("No Full row belongs to a patient of --icu_subjects_path (wrong file?).")
        y_pool_chk = y_full[pool_idx]
        pool_info = {
            "icu_subjects_path": str(cfg.icu_subjects_path),
            "icu_subjects_sha256": _sha256_file(cfg.icu_subjects_path),
            "n_icu_subjects_in_list": int(len(icu_ids)),
            "pool_rows": int(len(pool_idx)), "pool_patients": int(len(np.unique(subj_full[pool_idx]))),
            "pool_pos": int(y_pool_chk.sum()), "pool_prevalence": float(y_pool_chk.mean()),
            "pool_fraction_of_full_rows": float(len(pool_idx) / len(y_full)),
            "demo_rows_in_pool_fraction": float(np.isin(subj_demo, icu_ids).mean()),
        }
        print(f"[OK] null pool = ICU patients: {pool_info['pool_rows']:,} rows / {pool_info['pool_patients']:,} "
              f"patients (prevalence {pool_info['pool_prevalence']:.4f}); Demo rows whose patient is in the "
              f"pool: {pool_info['demo_rows_in_pool_fraction']:.4f}")
        if pool_info["demo_rows_in_pool_fraction"] < 1.0:
            print("[WARNING] some Demo patients are NOT in the ICU list -- check --icu_subjects_path.")
        if int(y_pool_chk.sum()) < pos_demo:
            raise ValueError("The ICU pool has fewer positive rows than the Demo; cannot draw matched nulls.")

    print("[..] hashing input files for the checkpoint signature")
    sig = {
        "script_version": SCRIPT_VERSION,
        "full_sha256": _sha256_file(cfg.full_path), "demo_sha256": _sha256_file(cfg.demo_path),
        "target": target, "subsample_n": n, "n_splits": cfg.n_splits, "seed": cfg.seed,
        "calibration": cfg.calibration, "calibration_cv": cfg.calibration_cv,
        "perm_repeats": cfg.perm_repeats, "importance_test_cap": cfg.importance_test_cap,
        "run_importance_stability": cfg.run_importance_stability,
        "min_per_class": min_per_class, "subsample_max_retries": cfg.subsample_max_retries,
        "group_col": cfg.group_col, "cv_mode": cfg.cv_mode, "svc_max_train_n": cfg.svc_max_train_n,
        "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
    }
    if cfg.null_pool != "all":  # keys added only for non-default pools: old checkpoints stay valid
        sig["null_pool"] = cfg.null_pool
        sig["icu_subjects_sha256"] = pool_info.get("icu_subjects_sha256")
    ckpt = Checkpoint(cfg.outdir / "_checkpoint", sig)
    _retire_stale_model_rows(ckpt, zoo_params_sha(cfg.seed))

    # References
    full_metrics = eval_dataset_once(X_full, y_full, groups_full, cfg, seed=cfg.seed, n_jobs=cfg.n_jobs,
                                     label="full_reference", ckpt=ckpt)
    demo_metrics = eval_dataset_once(X_demo, y_demo, groups_demo, cfg, seed=cfg.seed, n_jobs=cfg.n_jobs,
                                     label="demo")
    full_metrics.to_csv(cfg.outdir / "full_reference_metrics.csv", index=False)
    demo_metrics.to_csv(cfg.outdir / "demo_metrics.csv", index=False)
    print("[SAVED] full_reference_metrics.csv, demo_metrics.csv")

    # Null pool (all Full rows, or the ICU patients' rows) and, for the ICU pool, the Full-ICU reference
    if pool_idx is None:
        X_pool, y_pool, g_pool = X_full, y_full, groups_full
    else:
        X_pool = X_full.iloc[pool_idx].reset_index(drop=True)
        y_pool = y_full[pool_idx]
        g_pool = groups_full[pool_idx]
    icu_metrics: Optional[pd.DataFrame] = None
    if pool_idx is not None:
        icu_metrics = eval_dataset_once(X_pool, y_pool, g_pool, cfg, seed=cfg.seed, n_jobs=cfg.n_jobs,
                                        label="full_icu_reference", ckpt=ckpt)
        icu_metrics.to_csv(cfg.outdir / "full_icu_reference_metrics.csv", index=False)
        print("[SAVED] full_icu_reference_metrics.csv")

    full_imp = None
    demo_corr = float("nan")
    if cfg.run_importance_stability:
        full_imp = ckpt.load_json("full_importance")
        if full_imp is None:
            print("[..] Full reference importance (held-out permutation importance)")
            full_imp = fit_importance_cv(X_full, y_full, groups_full, cfg, seed=cfg.seed + 999,
                                         test_cap=cfg.importance_test_cap)
            ckpt.save_json("full_importance", full_imp)
        demo_imp = fit_importance_cv(X_demo, y_demo, groups_demo, cfg, seed=cfg.seed + 999)
        demo_corr = spearman_corr_dict(demo_imp, full_imp) if demo_imp else float("nan")
        imp_df = pd.DataFrame({"feature": list(full_imp), "importance_full": list(full_imp.values())})
        imp_df["importance_demo"] = imp_df["feature"].map(demo_imp)
        imp_df.sort_values("importance_full", ascending=False).to_csv(
            cfg.outdir / "importance_full_vs_demo.csv", index=False)
        print(f"[OK] Spearman(Demo importance, Full importance) = {demo_corr:.3f}")

    icu_imp = None
    demo_corr_icu = float("nan")
    if cfg.run_importance_stability and pool_idx is not None:
        icu_imp = ckpt.load_json("full_icu_importance")
        if icu_imp is None:
            print("[..] Full-ICU reference importance (held-out permutation importance)")
            icu_imp = fit_importance_cv(X_pool, y_pool, g_pool, cfg, seed=cfg.seed + 999,
                                        test_cap=cfg.importance_test_cap)
            ckpt.save_json("full_icu_importance", icu_imp)
        demo_corr_icu = spearman_corr_dict(demo_imp, icu_imp) if demo_imp else float("nan")
        imp2 = pd.DataFrame({"feature": list(icu_imp), "importance_full_icu": list(icu_imp.values())})
        imp2["importance_full"] = imp2["feature"].map(full_imp)
        imp2["importance_demo"] = imp2["feature"].map(demo_imp)
        imp2.sort_values("importance_full_icu", ascending=False).to_csv(
            cfg.outdir / "importance_fullicu_vs_demo.csv", index=False)
        print(f"[OK] Spearman(Demo importance, Full-ICU importance) = {demo_corr_icu:.3f}")

    sampler = GroupSampler(y_pool, g_pool)
    run_info = {"script_version": SCRIPT_VERSION, "target": target, "full_n": int(len(y_full)),
                "full_pos": int(y_full.sum()), "demo_n": n_demo, "demo_pos": pos_demo,
                "subsample_n": n, "min_per_class": min_per_class, "n_subsamples": cfg.n_subsamples,
                "demo_importance_spearman_vs_full": demo_corr,
                "null_pool": cfg.null_pool, "pool": pool_info,
                "demo_importance_spearman_vs_full_icu": (demo_corr_icu if pool_idx is not None else None),
                "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
                "null": {}}
    results = {}
    for mode in cfg.null_modes:
        sfx = MODE_SUFFIX[mode]
        long = run_subsample_experiments(X_pool, y_pool, g_pool, sampler, cfg, mode, n, pos_demo,
                                         min_per_class, full_imp, ckpt, icu_imp=icu_imp)
        long.to_csv(cfg.outdir / f"subsample_metrics_long{sfx}.csv", index=False)
        per_run = long.drop_duplicates("run_id")
        nan_runs = long.groupby("run_id")[METRICS].apply(lambda d: bool(d.isna().any().any()))
        run_info["null"][mode] = {
            "n_runs": int(len(per_run)),
            "acceptance_rate": float(len(per_run) / per_run["attempts"].sum()),
            "n_rows_mean": float(per_run["n_rows"].mean()),
            "n_pos_mean": float(per_run["n_pos"].mean()),
            "n_pos_min": int(per_run["n_pos"].min()), "n_pos_max": int(per_run["n_pos"].max()),
            "prevalence_mean": float((per_run["n_pos"] / per_run["n_rows"]).mean()),
            "demo_prevalence": pos_demo / n_demo,
            "n_patients_mean": float(per_run["n_patients"].mean()),
            "runs_with_nan_metric": int(nan_runs.sum()),
            "model_folds_uncalibrated": int(long["folds_uncalibrated"].sum()),
            "model_folds_failed": int(long["folds_failed"].sum()),
        }
        e1 = exp1_demo_percentile(demo_metrics, long)
        e2 = exp2_rank_stability(full_metrics, demo_metrics, long, cfg)
        e3 = exp3_decision_stability(full_metrics, demo_metrics, long, cfg)
        e1.to_csv(cfg.outdir / f"exp1_demo_percentiles{sfx}.csv", index=False)
        e2.to_csv(cfg.outdir / f"exp2_rank_stability{sfx}.csv", index=False)
        e3.to_csv(cfg.outdir / f"exp3_decision_stability{sfx}.csv", index=False)
        e4 = None
        if cfg.run_importance_stability:
            imp_cols = ["run_id", "null_mode", "n_rows", "n_pos", "imp_spearman_vs_full"]
            if icu_imp is not None:
                imp_cols.append("imp_spearman_vs_full_icu")
            per_run[imp_cols].to_csv(cfg.outdir / f"exp4_importance_stability{sfx}.csv", index=False)
            e4 = exp4_summary(demo_corr, per_run)
            e4.to_csv(cfg.outdir / f"exp4_importance_summary{sfx}.csv", index=False)
            if icu_imp is not None:
                exp4_summary(demo_corr_icu, per_run, col="imp_spearman_vs_full_icu").to_csv(
                    cfg.outdir / f"exp4_importance_summary_vs_fullicu{sfx}.csv", index=False)
            plot_importance_stability(per_run, demo_corr, cfg, mode)
        plot_auroc_hist_per_model(long, demo_metrics, cfg, n, mode)
        if icu_metrics is not None:  # same null draws, judged against the Full-ICU reference
            exp2_rank_stability(icu_metrics, demo_metrics, long, cfg).to_csv(
                cfg.outdir / f"exp2_rank_stability_vs_fullicu{sfx}.csv", index=False)
            exp3_decision_stability(icu_metrics, demo_metrics, long, cfg).to_csv(
                cfg.outdir / f"exp3_decision_stability_vs_fullicu{sfx}.csv", index=False)
        results[mode] = (e1, e2, e3, e4)
        print(f"[SAVED] {mode} null outputs (suffix '{sfx}')")

    _atomic_write_text(cfg.outdir / "run_info.json", json.dumps(run_info, indent=2, default=str))

    # Console summary
    pm = cfg.primary_metric
    asc = pm not in HIGHER_BETTER
    print("\n================ SUMMARY ================")
    print("PRIMARY_METRIC:", pm)
    print("\n[Full reference metrics]")
    print(full_metrics.sort_values(pm, ascending=asc).to_string(index=False))
    if icu_metrics is not None:
        print("\n[Full-ICU reference metrics]")
        print(icu_metrics.sort_values(pm, ascending=asc).to_string(index=False))
    print("\n[Demo metrics]")
    print(demo_metrics.sort_values(pm, ascending=asc).to_string(index=False))
    for mode, (e1, e2, e3, e4) in results.items():
        print(f"\n---- null = {mode} ({json.dumps(run_info['null'][mode])}) ----")
        sub = e1[e1["metric"] == pm][["model", "demo_value", "subsample_q025", "subsample_q50", "subsample_q975",
                                      "demo_percentile_(better_direction)", "p_two_sided_empirical"]]
        print(sub.to_string(index=False))
        print(e2.to_string(index=False))
        print(e3.to_string(index=False))
        if e4 is not None:
            print(e4.to_string(index=False))
    print("========================================\n")


def parse_args_into_cfg() -> Config:
    import argparse

    p = argparse.ArgumentParser(description="Is the MIMIC-IV Demo a typical Demo-sized draw from Full? "
                                            "(null distributions of metrics, rankings, decisions, importances)")
    p.add_argument("--full_path", type=str, required=True)
    p.add_argument("--demo_path", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--target_col", type=str, default=None,
                   help="Target column (label_mortality / label_ed_admit); auto-detected if unset.")
    p.add_argument("--n_subsamples", type=int, default=CFG.n_subsamples, help="Null draws per null mode.")
    p.add_argument("--subsample_n", type=int, default=None,
                   help="Null subsample size in rows. Default (recommended): the Demo row count.")
    p.add_argument("--null_modes", nargs="+", default=list(CFG.null_modes), choices=list(NULL_MODES))
    p.add_argument("--n_splits", type=int, default=CFG.n_splits)
    p.add_argument("--seed", type=int, default=CFG.seed)
    p.add_argument("--calibration", type=str, default=CFG.calibration, choices=["sigmoid", "isotonic", "none"])
    p.add_argument("--calibration_cv", type=int, default=CFG.calibration_cv)
    p.add_argument("--primary_metric", type=str, default=CFG.primary_metric, choices=METRICS)
    p.add_argument("--run_importance_stability", action=argparse.BooleanOptionalAction,
                   default=CFG.run_importance_stability)
    p.add_argument("--perm_repeats", type=int, default=CFG.perm_repeats)
    p.add_argument("--importance_test_cap", type=int, default=CFG.importance_test_cap)
    p.add_argument("--min_per_class", type=int, default=None,
                   help="Minimum rows of each class in a null draw (default max(n_splits, calibration_cv)).")
    p.add_argument("--subsample_max_retries", type=int, default=CFG.subsample_max_retries)
    p.add_argument("--group_col", type=str, default=CFG.group_col)
    p.add_argument("--cv_mode", type=str, default=CFG.cv_mode, choices=["group", "row"])
    p.add_argument("--svc_max_train_n", type=int, default=None,
                   help="Cap SVC_RBF's training-fold size (Full-scale runtime). No effect on Demo-sized data.")
    p.add_argument("--n_jobs", type=int, default=CFG.n_jobs,
                   help="Worker processes for the null draws (and threads for the Full reference fits).")
    p.add_argument("--progress_every", type=int, default=CFG.progress_every)
    p.add_argument("--null_pool", type=str, default=CFG.null_pool, choices=["all", "icu"],
                   help="Patients the null draws are taken from: all Full patients (default) or only Full "
                        "patients with >= 1 ICU stay (needs --icu_subjects_path; use a separate --outdir).")
    p.add_argument("--icu_subjects_path", type=str, default=None,
                   help="CSV with a subject_id column listing Full patients with >= 1 ICU stay "
                        "(01_prepare_data/make_icu_subject_list.py). Used with --null_pool icu only.")
    args = p.parse_args()
    if args.null_pool == "icu" and not args.icu_subjects_path:
        p.error("--null_pool icu requires --icu_subjects_path")

    return Config(
        full_path=args.full_path,
        demo_path=args.demo_path,
        outdir=Path(args.outdir),
        plots_dir=Path(args.outdir) / "plots",
        n_subsamples=args.n_subsamples,
        subsample_n=args.subsample_n,
        n_splits=args.n_splits,
        seed=args.seed,
        target_col=(None if args.target_col in (None, "", "auto") else args.target_col),
        calibration=(None if args.calibration == "none" else args.calibration),
        calibration_cv=args.calibration_cv,
        primary_metric=args.primary_metric,
        run_importance_stability=args.run_importance_stability,
        perm_repeats=args.perm_repeats,
        importance_test_cap=args.importance_test_cap,
        min_per_class_in_subsample=args.min_per_class,
        subsample_max_retries=args.subsample_max_retries,
        group_col=args.group_col,
        cv_mode=args.cv_mode,
        svc_max_train_n=args.svc_max_train_n,
        null_modes=tuple(dict.fromkeys(args.null_modes)),
        n_jobs=(args.n_jobs if args.n_jobs >= 1 else max(1, (os.cpu_count() or 1) + 1 + args.n_jobs)),
        progress_every=max(1, args.progress_every),
        null_pool=args.null_pool,
        icu_subjects_path=args.icu_subjects_path,
    )


if __name__ == "__main__":
    main(parse_args_into_cfg())
