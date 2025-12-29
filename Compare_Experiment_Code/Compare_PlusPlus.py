# demo_trustworthiness_merged.py
# ------------------------------------------------------------
# Merged version:
# - Keeps your original structure/metrics extraction style
# - Adds best parts from Compare_Plus.py:
#   * Config dataclass
#   * robust binary y coercion
#   * AUPRC metric
#   * Optional Experiment 4: importance stability via permutation importance
#   * Per-model AUROC hist plots w/ DEMO vertical line
#
# EXTRA (your request):
# - Adds progress + ETA (download-like) reporting for long loops
#   * Subsample loop (Exp0)
#   * Exp4 loop
# - Keeps everything else the same (metrics, files, flow, outputs)
# ------------------------------------------------------------

from __future__ import annotations

import warnings
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss,
    average_precision_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# -----------------------------
# CONFIG
# -----------------------------
@dataclass
class Config:
    full_path: str = "Full_analytic_dataset_mortality_all_admissions.csv"
    demo_path: str = "demo_analytic_dataset_mortality_all_admissions.csv"

    outdir: Path = Path("outputs_demo_trust")
    plots_dir: Path = Path("outputs_demo_trust") / "plots"

    n_subsamples: int = 1000
    subsample_n: int = 252
    n_splits: int = 5
    seed: int = 42

    # If auto-detect fails, set explicitly
    target_col: Optional[str] = None

    # Choose calibration: "sigmoid", "isotonic", or None
    calibration: Optional[str] = "sigmoid"

    # Calibration CV folds (will be auto-adjusted down if needed)
    calibration_cv: int = 3

    # Main selection metric for rank/decision stability
    primary_metric: str = "AUROC"  # AUROC, LogLoss, Brier, AUPRC

    # Top-k overlap checks
    topk_list: Tuple[int, ...] = (1, 2, 3)

    # Add-on: importance stability
    run_importance_stability: bool = True
    perm_repeats: int = 10  # per permutation importance call (keep modest)

    # Progress printing controls
    progress_every: int = 10  # print every N runs for long loops

    # Option B robustness knobs
    # Guarantee at least this many samples per class in each subsample:
    min_per_class_in_subsample: Optional[int] = None
    # How many times to retry drawing a valid stratified subsample (in case class is extremely rare)
    subsample_max_retries: int = 50


CFG = Config()


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def ensure_outdirs(cfg: Config) -> None:
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)


def find_target_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "mortality", "Mortality", "death", "Death", "outcome", "Outcome",
        "label", "Label", "target", "Target", "y", "Y",
        "hospital_expire_flag", "HOSPITAL_EXPIRE_FLAG",
        "in_hospital_mortality", "IN_HOSPITAL_MORTALITY",
    ]
    for c in candidates:
        if c in df.columns:
            vals = df[c].dropna().unique()
            if len(vals) <= 2:
                return c

    for c in df.columns:
        vals = df[c].dropna().unique()
        if len(vals) <= 2 and df[c].dtype != "O":
            return c

    return None


def coerce_binary_y(y: pd.Series | np.ndarray) -> np.ndarray:
    """
    Make y strictly {0,1}. Robust to booleans, {1,2}, {"yes","no"} (if 2 unique).
    """
    if isinstance(y, np.ndarray):
        y_ser = pd.Series(y)
    else:
        y_ser = y.copy()

    y_ser = y_ser.dropna()
    uniq = list(pd.unique(y_ser))
    if len(uniq) != 2:
        raise ValueError(f"Target must be binary-like. Found unique={uniq[:10]} (n={len(uniq)})")

    try:
        uniq_sorted = sorted(uniq)
    except Exception:
        uniq_sorted = sorted([str(u) for u in uniq])

    mapping = {uniq_sorted[0]: 0, uniq_sorted[1]: 1}
    if not set(mapping.keys()).issubset(set(uniq)):
        mapping = {str(uniq_sorted[0]): 0, str(uniq_sorted[1]): 1}
        y01 = pd.Series(y).astype(str).map(mapping).to_numpy()
    else:
        y01 = pd.Series(y).map(mapping).to_numpy()

    if set(np.unique(y01[~pd.isna(y01)])) - {0, 1}:
        raise ValueError("Failed to coerce y into {0,1}.")
    return y01.astype(int)


def split_X_y(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
    y_raw = df[target_col].values
    y = coerce_binary_y(y_raw)
    X = df.drop(columns=[target_col])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )
    return pre


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
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def _min_class_count(y: np.ndarray) -> int:
    y = np.asarray(y)
    vals, cnts = np.unique(y, return_counts=True)
    if len(vals) < 2:
        return 0
    return int(cnts.min())


# -----------------------------
# Progress helpers
# -----------------------------
def _fmt_seconds(sec: float) -> str:
    if sec is None or not np.isfinite(sec) or sec < 0:
        return "?:??"
    sec = int(round(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def _progress_line(label: str, done: int, total: int, t0: float) -> str:
    elapsed = time.time() - t0
    rate = elapsed / max(done, 1)
    remaining = (total - done) * rate
    pct = (done / total) * 100.0 if total > 0 else 0.0

    bar_len = 24
    filled = int(round(bar_len * done / total)) if total > 0 else 0
    filled = min(max(filled, 0), bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    return (
        f"[{label}] {pct:6.2f}% |{bar}| "
        f"{done}/{total} "
        f"Elapsed {_fmt_seconds(elapsed)} "
        f"ETA {_fmt_seconds(remaining)}"
    )


# -----------------------------
# Model zoo
# -----------------------------
@dataclass
class ModelSpec:
    name: str
    estimator: object
    needs_scaling: bool = True


def get_model_zoo(seed: int = 42) -> List[ModelSpec]:
    return [
        ModelSpec(
            "Logistic_L2",
            LogisticRegression(max_iter=5000, solver="lbfgs", n_jobs=None, class_weight="balanced"),
        ),
        ModelSpec(
            "RandomForest",
            RandomForestClassifier(
                n_estimators=400, random_state=seed, n_jobs=-1,
                class_weight="balanced_subsample", min_samples_leaf=2
            ),
            needs_scaling=False,
        ),
        ModelSpec(
            "ExtraTrees",
            ExtraTreesClassifier(
                n_estimators=600, random_state=seed, n_jobs=-1,
                class_weight="balanced", min_samples_leaf=2
            ),
            needs_scaling=False,
        ),
        ModelSpec(
            "GradientBoosting",
            GradientBoostingClassifier(random_state=seed),
            needs_scaling=False,
        ),
        ModelSpec("GaussianNB", GaussianNB()),
        ModelSpec(
            "MLP",
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                random_state=seed,
                max_iter=400,
                early_stopping=True,
            ),
        ),
        ModelSpec(
            "SVC_RBF",
            SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        ),
    ]


# -----------------------------
# Option B: robust stratified subsampling
# -----------------------------
def stratified_subsample_indices(
    y: np.ndarray,
    n: int,
    rng: np.random.Generator,
    min_per_class: int,
) -> np.ndarray:
    """
    Draw stratified subsample with guaranteed minimum samples per class.
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError("y must have exactly 2 classes for stratified_subsample_indices.")

    c0, c1 = classes[0], classes[1]
    idx0 = np.where(y == c0)[0]
    idx1 = np.where(y == c1)[0]

    if n < 2 * min_per_class:
        raise ValueError(f"subsample_n={n} too small for min_per_class={min_per_class} (need >= {2*min_per_class})")

    # target proportion (based on full y)
    p1 = len(idx1) / len(y)
    n1 = int(round(n * p1))
    n1 = max(min_per_class, min(n - min_per_class, n1))
    n0 = n - n1

    # If not enough in one class, clamp and re-balance
    n0 = min(n0, len(idx0))
    n1 = min(n1, len(idx1))
    if n0 + n1 < n:
        # fill from the class that has remaining
        remain = n - (n0 + n1)
        room0 = len(idx0) - n0
        room1 = len(idx1) - n1
        take0 = min(remain, room0)
        n0 += take0
        remain -= take0
        take1 = min(remain, room1)
        n1 += take1
        remain -= take1

    # final safety: ensure minimum per class if possible
    if n0 < min_per_class or n1 < min_per_class:
        raise ValueError(
            f"Cannot satisfy min_per_class={min_per_class} with available counts "
            f"(class0={len(idx0)}, class1={len(idx1)})."
        )

    s0 = rng.choice(idx0, size=n0, replace=False)
    s1 = rng.choice(idx1, size=n1, replace=False)
    idx = np.concatenate([s0, s1])
    rng.shuffle(idx)
    return idx


def choose_safe_folds(min_class: int, desired: int) -> int:
    """
    Choose a safe StratifiedKFold n_splits given smallest class count.
    Must be at least 2 to do CV; otherwise return 0 (caller decides).
    """
    if min_class < 2:
        return 0
    return int(min(desired, min_class))


def choose_safe_calibration_cv(min_class: int, desired: int) -> int:
    """
    CalibratedClassifierCV requires each class count >= cv folds.
    If too small, return 0 -> disable calibration.
    """
    if min_class < 2:
        return 0
    return int(min(desired, min_class))


# -----------------------------
# Fit + OOF prediction (robust)
# -----------------------------
def fit_predict_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    model: ModelSpec,
    seed: int,
    n_splits_desired: int,
    calibration: Optional[str],
    calibration_cv_desired: int,
) -> np.ndarray:
    y = np.asarray(y).astype(int)
    min_class = _min_class_count(y)

    # Outer CV folds
    n_splits = choose_safe_folds(min_class, n_splits_desired)
    if n_splits < 2:
        # No valid CV possible; fallback: fit once and predict on same data
        # (keeps pipeline running, avoids crashing; metrics may be optimistic)
        pre = build_preprocessor(X)
        pipe = Pipeline(steps=[("preprocess", pre), ("clf", model.estimator)])

        # Calibration choice (also needs enough class count)
        cal_cv = choose_safe_calibration_cv(min_class, calibration_cv_desired)
        if calibration in ("sigmoid", "isotonic") and cal_cv >= 2:
            clf = CalibratedClassifierCV(pipe, method=calibration, cv=cal_cv)
        else:
            clf = pipe

        clf.fit(X, y)
        proba = clf.predict_proba(X)[:, 1]
        return np.asarray(proba, dtype=float)

    pre = build_preprocessor(X)
    pipe = Pipeline(steps=[("preprocess", pre), ("clf", model.estimator)])

    # Calibration CV folds (inner)
    cal_cv = choose_safe_calibration_cv(min_class, calibration_cv_desired)
    if calibration in ("sigmoid", "isotonic") and cal_cv >= 2:
        clf = CalibratedClassifierCV(pipe, method=calibration, cv=cal_cv)
    else:
        clf = pipe

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)

    for tr_idx, te_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_te = X.iloc[te_idx]

        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]
        oof[te_idx] = proba

    return oof


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    out = {}
    # If a run accidentally has 1 class, metrics like AUROC break — handle gracefully.
    if len(np.unique(y_true)) < 2:
        out["AUROC"] = float("nan")
        out["AUPRC"] = float("nan")
        y_prob_clip = np.clip(y_prob, 1e-15, 1 - 1e-15)
        out["LogLoss"] = float(log_loss(y_true, y_prob_clip, labels=[0, 1]))
        out["Brier"] = float(brier_score_loss(y_true, y_prob))
        out["ECE"] = float(expected_calibration_error(y_true, y_prob, n_bins=10))
        return out

    out["AUROC"] = float(roc_auc_score(y_true, y_prob))
    out["AUPRC"] = float(average_precision_score(y_true, y_prob))
    y_prob_clip = np.clip(y_prob, 1e-15, 1 - 1e-15)
    out["LogLoss"] = float(log_loss(y_true, y_prob_clip))
    out["Brier"] = float(brier_score_loss(y_true, y_prob))
    out["ECE"] = float(expected_calibration_error(y_true, y_prob, n_bins=10))
    return out


def rank_models(df_metrics: pd.DataFrame, metric: str) -> pd.Series:
    higher_better = metric in ("AUROC", "AUPRC")
    vals = df_metrics.set_index("model")[metric]
    if higher_better:
        return (-vals).rank(method="min").astype(int)
    return (vals).rank(method="min").astype(int)


def topk_set(df_metrics: pd.DataFrame, metric: str, k: int) -> set:
    higher_better = metric in ("AUROC", "AUPRC")
    df = df_metrics.sort_values(metric, ascending=not higher_better)
    return set(df["model"].head(k).tolist())


# -----------------------------
# Experiment runner
# -----------------------------
def eval_dataset_once(df: pd.DataFrame, target_col: str, cfg: Config, seed: int) -> pd.DataFrame:
    X, y = split_X_y(df, target_col)
    models = get_model_zoo(seed=seed)

    rows = []
    for ms in models:
        y_prob = fit_predict_oof(
            X, y, ms,
            seed=seed,
            n_splits_desired=cfg.n_splits,
            calibration=cfg.calibration,
            calibration_cv_desired=cfg.calibration_cv,
        )
        m = compute_metrics(y, y_prob)
        rows.append({"model": ms.name, **m})
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def run_subsample_experiments(full_df: pd.DataFrame, target_col: str, cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    models = get_model_zoo(seed=cfg.seed)
    X_full, y_full = split_X_y(full_df, target_col)

    n = len(full_df)
    if cfg.subsample_n > n:
        raise ValueError(f"subsample_n={cfg.subsample_n} > full n={n}")

    # Option B: choose min_per_class automatically if not set:
    # Need enough for BOTH:
    # - outer CV: cfg.n_splits
    # - calibration CV: cfg.calibration_cv (if calibration enabled)
    need_for_cal = cfg.calibration_cv if cfg.calibration in ("sigmoid", "isotonic") else 2
    min_per_class = cfg.min_per_class_in_subsample or max(cfg.n_splits, need_for_cal, 2)

    all_rows = []
    t0 = time.time()
    total = cfg.n_subsamples

    for run_id in range(cfg.n_subsamples):
        # Retry drawing a valid stratified subsample if the outcome is too rare
        idx = None
        for _ in range(cfg.subsample_max_retries):
            try:
                idx_try = stratified_subsample_indices(y_full, cfg.subsample_n, rng, min_per_class=min_per_class)
                y_try = y_full[idx_try]
                if len(np.unique(y_try)) == 2:
                    idx = idx_try
                    break
            except Exception:
                continue

        if idx is None:
            # As last resort: random choice without guarantee; still do not crash.
            idx = rng.choice(n, size=cfg.subsample_n, replace=False)

        X = X_full.iloc[idx].reset_index(drop=True)
        y = y_full[idx]

        for ms in models:
            y_prob = fit_predict_oof(
                X, y, ms,
                seed=cfg.seed + run_id,
                n_splits_desired=cfg.n_splits,
                calibration=cfg.calibration,
                calibration_cv_desired=cfg.calibration_cv,
            )
            m = compute_metrics(y, y_prob)
            all_rows.append({"run_id": run_id, "model": ms.name, **m})

        done = run_id + 1
        if done % cfg.progress_every == 0 or done == total:
            print(_progress_line("Subsample", done, total, t0))

    return pd.DataFrame(all_rows)


def exp1_demo_percentile(demo_metrics: pd.DataFrame, subsample_long: pd.DataFrame) -> pd.DataFrame:
    metrics = ["AUROC", "AUPRC", "LogLoss", "Brier", "ECE"]
    rows = []

    for model in demo_metrics["model"].unique():
        demo_row = demo_metrics.loc[demo_metrics["model"] == model].iloc[0]
        dist = subsample_long[subsample_long["model"] == model]

        for metric in metrics:
            vals = dist[metric].dropna().values
            demo_val = float(demo_row[metric])
            if len(vals) == 0 or not np.isfinite(demo_val):
                continue

            higher_better = metric in ("AUROC", "AUPRC")
            if higher_better:
                pct = float((vals < demo_val).mean() * 100.0)
            else:
                pct = float((vals > demo_val).mean() * 100.0)

            q025, q50, q975 = np.quantile(vals, [0.025, 0.5, 0.975])
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "demo_value": demo_val,
                    "subsample_q025": float(q025),
                    "subsample_q50": float(q50),
                    "subsample_q975": float(q975),
                    "demo_percentile_(better_direction)": pct,
                }
            )

    return pd.DataFrame(rows)


def exp2_rank_stability(full_metrics: pd.DataFrame, demo_metrics: pd.DataFrame, subsample_long: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    full_rank = rank_models(full_metrics, cfg.primary_metric)
    demo_rank = rank_models(demo_metrics, cfg.primary_metric)

    merged_demo = pd.DataFrame({"rank_full": full_rank, "rank_demo": demo_rank}).dropna()
    demo_spearman = float(merged_demo["rank_full"].corr(merged_demo["rank_demo"], method="spearman"))

    spearmans = []
    topk_overlaps = {k: [] for k in cfg.topk_list}

    full_top = {k: topk_set(full_metrics, cfg.primary_metric, k) for k in cfg.topk_list}

    for _, g in subsample_long.groupby("run_id"):
        g_metrics = g[["model", "AUROC", "AUPRC", "LogLoss", "Brier", "ECE"]].copy()
        sub_rank = rank_models(g_metrics, cfg.primary_metric)

        merged = pd.DataFrame({"rank_full": full_rank, "rank_sub": sub_rank}).dropna()
        spearmans.append(float(merged["rank_full"].corr(merged["rank_sub"], method="spearman")))

        for k in cfg.topk_list:
            sub_top = topk_set(g_metrics, cfg.primary_metric, k)
            topk_overlaps[k].append(len(full_top[k] & sub_top) / k)

    rows = []
    rows.append(
        {
            "summary": "demo_vs_full",
            "spearman_rank_corr": demo_spearman,
            **{f"top{k}_overlap": len(full_top[k] & topk_set(demo_metrics, cfg.primary_metric, k)) / k for k in cfg.topk_list},
        }
    )

    spearmans = np.array(spearmans, dtype=float)
    row_sub = {
        "summary": "subsample_vs_full_distribution",
        "spearman_mean": float(np.nanmean(spearmans)),
        "spearman_q025": float(np.nanquantile(spearmans, 0.025)),
        "spearman_q50": float(np.nanquantile(spearmans, 0.50)),
        "spearman_q975": float(np.nanquantile(spearmans, 0.975)),
    }
    for k in cfg.topk_list:
        arr = np.array(topk_overlaps[k], dtype=float)
        row_sub[f"top{k}_overlap_mean"] = float(np.nanmean(arr))
        row_sub[f"top{k}_overlap_q025"] = float(np.nanquantile(arr, 0.025))
        row_sub[f"top{k}_overlap_q50"] = float(np.nanquantile(arr, 0.50))
        row_sub[f"top{k}_overlap_q975"] = float(np.nanquantile(arr, 0.975))

    rows.append(row_sub)
    return pd.DataFrame(rows)


def exp3_decision_stability(full_metrics: pd.DataFrame, demo_metrics: pd.DataFrame, subsample_long: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    def best_model(df: pd.DataFrame) -> str:
        higher_better = cfg.primary_metric in ("AUROC", "AUPRC")
        dd = df.sort_values(cfg.primary_metric, ascending=not higher_better)
        return dd.iloc[0]["model"]

    full_best = best_model(full_metrics)
    demo_best = best_model(demo_metrics)
    demo_hit = 1 if demo_best == full_best else 0

    hits = []
    cover_k = {k: [] for k in cfg.topk_list}

    for _, g in subsample_long.groupby("run_id"):
        g_metrics = g[["model", "AUROC", "AUPRC", "LogLoss", "Brier", "ECE"]].copy()
        sub_best = best_model(g_metrics)
        hits.append(1 if sub_best == full_best else 0)

        for k in cfg.topk_list:
            sub_top = topk_set(g_metrics, cfg.primary_metric, k)
            cover_k[k].append(1 if full_best in sub_top else 0)

    hits = np.array(hits, dtype=float)

    out = {
        "primary_metric": cfg.primary_metric,
        "full_best_model": full_best,
        "demo_best_model": demo_best,
        "demo_best_matches_full": int(demo_hit),
        "subsample_P(best_matches_full)": float(np.nanmean(hits)),
        "subsample_q025": float(np.nanquantile(hits, 0.025)),
        "subsample_q50": float(np.nanquantile(hits, 0.50)),
        "subsample_q975": float(np.nanquantile(hits, 0.975)),
    }

    for k in cfg.topk_list:
        arr = np.array(cover_k[k], dtype=float)
        out[f"subsample_P(full_best_in_top{k})"] = float(np.nanmean(arr))

    return pd.DataFrame([out])


# -----------------------------
# Exp4: importance stability (robust)
# -----------------------------
def fit_importance_once(X: pd.DataFrame, y: np.ndarray, cfg: Config, seed: int) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        # no valid roc_auc scoring possible; return empty
        return {}

    pre = build_preprocessor(X)
    base_lr = LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs")
    pipe = Pipeline([("pre", pre), ("clf", base_lr)])
    pipe.fit(X, y)

    pi = permutation_importance(
        pipe, X, y,
        scoring="roc_auc",
        n_repeats=cfg.perm_repeats,
        random_state=seed,
        n_jobs=-1
    )

    # NOTE: permutation_importance returns features after preprocessing; but we used ColumnTransformer,
    # so feature names are not aligned with X.columns. To keep it consistent and avoid errors,
    # we aggregate by original columns using a simple fallback:
    # If lengths mismatch, just return index by position.
    imps = pi.importances_mean
    if len(imps) == len(X.columns):
        imp = pd.Series(imps, index=X.columns).sort_values(ascending=False)
    else:
        imp = pd.Series(imps, index=[f"feat_{i}" for i in range(len(imps))]).sort_values(ascending=False)
    return imp.to_dict()


def spearman_corr_dict(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = sorted(set(a.keys()) & set(b.keys()))
    if len(keys) < 2:
        return float("nan")
    va = np.array([a[k] for k in keys], dtype=float)
    vb = np.array([b[k] for k in keys], dtype=float)
    ra = pd.Series(va).rank(method="average").to_numpy()
    rb = pd.Series(vb).rank(method="average").to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def exp4_importance_stability(full_df: pd.DataFrame, demo_df: pd.DataFrame, target_col: str, cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    X_demo, y_demo = split_X_y(demo_df, target_col)
    demo_imp = fit_importance_once(X_demo, y_demo, cfg, seed=cfg.seed + 999)
    if len(demo_imp) == 0:
        # Demo has 1 class -> cannot evaluate importance stability
        return pd.DataFrame({"run_id": [], "spearman_importance_corr_vs_demo": []})

    X_full, y_full = split_X_y(full_df, target_col)

    # same min_per_class logic as subsample
    need_for_cal = cfg.calibration_cv if cfg.calibration in ("sigmoid", "isotonic") else 2
    min_per_class = cfg.min_per_class_in_subsample or max(cfg.n_splits, need_for_cal, 2)

    rows = []
    t0 = time.time()
    total = cfg.n_subsamples

    for run_id in range(cfg.n_subsamples):
        idx = None
        for _ in range(cfg.subsample_max_retries):
            try:
                idx_try = stratified_subsample_indices(y_full, cfg.subsample_n, rng, min_per_class=min_per_class)
                if len(np.unique(y_full[idx_try])) == 2:
                    idx = idx_try
                    break
            except Exception:
                continue

        if idx is None:
            idx = rng.choice(len(full_df), size=cfg.subsample_n, replace=False)

        X_sub = X_full.iloc[idx].reset_index(drop=True)
        y_sub = y_full[idx]

        sub_imp = fit_importance_once(X_sub, y_sub, cfg, seed=cfg.seed + run_id)
        corr = spearman_corr_dict(sub_imp, demo_imp) if len(sub_imp) else float("nan")
        rows.append({"run_id": run_id, "spearman_importance_corr_vs_demo": corr})

        done = run_id + 1
        if done % cfg.progress_every == 0 or done == total:
            print(_progress_line("Exp4-Importance", done, total, t0))

    return pd.DataFrame(rows)


# -----------------------------
# Plots
# -----------------------------
def plot_auroc_hist_per_model(subsample_long: pd.DataFrame, demo_metrics: pd.DataFrame, cfg: Config) -> None:
    subs = subsample_long.copy()
    for model in sorted(subs["model"].unique()):
        vals = subs.loc[subs["model"] == model, "AUROC"].dropna().to_numpy(dtype=float)
        demo_row = demo_metrics.loc[demo_metrics["model"] == model, "AUROC"]
        demo_val = float(demo_row.iloc[0]) if len(demo_row) else float("nan")

        if len(vals) == 0 or not np.isfinite(demo_val):
            continue

        plt.figure(figsize=(7, 4))
        plt.hist(vals, bins=30)
        plt.axvline(demo_val, linewidth=2)
        plt.title(f"AUROC subsample distribution (n={cfg.subsample_n}) - {model}")
        plt.xlabel("AUROC")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(cfg.plots_dir / f"auroc_hist_{model}.png")
        plt.close()


def plot_importance_stability(exp4_df: pd.DataFrame, cfg: Config) -> None:
    vals = exp4_df["spearman_importance_corr_vs_demo"].dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return
    plt.figure(figsize=(7, 4))
    plt.hist(vals, bins=30)
    plt.title("Importance stability (Spearman corr vs DEMO)")
    plt.xlabel("Spearman correlation")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(cfg.plots_dir / "exp4_importance_stability_hist.png")
    plt.close()


# -----------------------------
# main
# -----------------------------
def main(cfg: Config) -> None:
    ensure_outdirs(cfg)

    full_path = Path(cfg.full_path)
    demo_path = Path(cfg.demo_path)

    if not full_path.exists():
        fallback = Path("analytic_dataset_mortality_all_admissions.csv")
        if fallback.exists():
            print(f"[INFO] {cfg.full_path} not found. Using fallback: {fallback.name}")
            full_path = fallback
        else:
            raise FileNotFoundError(f"Cannot find {cfg.full_path} (or fallback analytic_dataset_mortality_all_admissions.csv)")

    if not demo_path.exists():
        raise FileNotFoundError(f"Cannot find {cfg.demo_path}")

    full_df = pd.read_csv(full_path)
    demo_df = pd.read_csv(demo_path)

    target = cfg.target_col or find_target_column(full_df)
    if target is None or target not in full_df.columns:
        raise ValueError("Target column auto-detection failed. Please set cfg.target_col explicitly.")
    if target not in demo_df.columns:
        raise ValueError(f"Demo.csv does not contain target column '{target}'.")

    print(f"[OK] Using target column: {target}")
    print(f"[OK] Full n={len(full_df):,} | Demo n={len(demo_df):,}")

    # FULL + DEMO (reference)
    full_metrics = eval_dataset_once(full_df, target, cfg, seed=cfg.seed)
    demo_metrics = eval_dataset_once(demo_df, target, cfg, seed=cfg.seed)

    full_metrics.to_csv(cfg.outdir / "full_reference_metrics.csv", index=False)
    demo_metrics.to_csv(cfg.outdir / "demo_metrics.csv", index=False)
    print("[SAVED] full_reference_metrics.csv, demo_metrics.csv")

    # Subsamples
    rng = set_seed(cfg.seed)
    subsample_long = run_subsample_experiments(full_df, target, cfg, rng=rng)
    subsample_long.to_csv(cfg.outdir / "subsample_metrics_long.csv", index=False)
    print("[SAVED] subsample_metrics_long.csv")

    # Exp1
    demo_pct = exp1_demo_percentile(demo_metrics, subsample_long)
    demo_pct.to_csv(cfg.outdir / "exp1_demo_percentiles.csv", index=False)
    print("[SAVED] exp1_demo_percentiles.csv")

    # Exp2
    exp2 = exp2_rank_stability(full_metrics, demo_metrics, subsample_long, cfg)
    exp2.to_csv(cfg.outdir / "exp2_rank_stability.csv", index=False)
    print("[SAVED] exp2_rank_stability.csv")

    # Exp3
    exp3 = exp3_decision_stability(full_metrics, demo_metrics, subsample_long, cfg)
    exp3.to_csv(cfg.outdir / "exp3_decision_stability.csv", index=False)
    print("[SAVED] exp3_decision_stability.csv")

    # Exp4 (optional)
    if cfg.run_importance_stability:
        exp4 = exp4_importance_stability(full_df, demo_df, target, cfg)
        exp4.to_csv(cfg.outdir / "exp4_importance_stability.csv", index=False)
        print("[SAVED] exp4_importance_stability.csv")
    else:
        exp4 = None

    # Plots
    plot_auroc_hist_per_model(subsample_long, demo_metrics, cfg)
    if exp4 is not None and len(exp4) > 0:
        plot_importance_stability(exp4, cfg)
    print("[SAVED] plots in", cfg.plots_dir)

    # Console summary
    print("\n================ SUMMARY ================")
    print("PRIMARY_METRIC:", cfg.primary_metric)
    print("\n[Full reference metrics]")
    print(full_metrics.sort_values(cfg.primary_metric, ascending=(cfg.primary_metric not in ("AUROC", "AUPRC"))).to_string(index=False))
    print("\n[Demo metrics]")
    print(demo_metrics.sort_values(cfg.primary_metric, ascending=(cfg.primary_metric not in ("AUROC", "AUPRC"))).to_string(index=False))
    print("\n[Decision stability]")
    print(exp3.to_string(index=False))
    if exp4 is not None and len(exp4) > 0:
        print("\n[Importance stability]")
        print(exp4["spearman_importance_corr_vs_demo"].describe().to_string())
    print("========================================\n")


if __name__ == "__main__":
    main(CFG)