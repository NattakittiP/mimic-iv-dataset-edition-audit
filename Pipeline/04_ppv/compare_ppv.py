# compare_ppv.py  (= Compare_PPV.py, refactored to take file paths as CLI args)
#
# Compares prevalence-standardized PPV (from run_ppv.py) between the Full and
# the Demo dataset of the same domain.
#
# STATISTICAL DESIGN (revised):
#  * Demo and Full are DIFFERENT datasets with DIFFERENT CV splits, so Demo
#    fold k and Full fold k have nothing to do with each other. The previous
#    version merged on (fold, model) and ran PAIRED tests (paired t, Wilcoxon
#    signed-rank, paired Cohen's d, bootstrap of paired differences) -- whose
#    p-values changed across an equally valid renumbering of the folds. All
#    comparisons here are UNPAIRED.
#  * PRIMARY estimate per side = PPV_std computed from confusion counts POOLED
#    over the folds of each CV repeat (every patient tested once per repeat),
#    averaged over repeats -- matching run_ppv.py's ppv_std_pooled_mean.
#    Averaging per-fold PPV is biased when folds contain few positives (Demo).
#  * CI for the Full - Demo difference: independent bootstrap of folds on each
#    side (resample folds with replacement, pool their counts, recompute
#    PPV_std). Folds of a repeated CV are correlated, so treat this CI and the
#    unpaired tests (Welch t, Mann-Whitney U on per-fold values) as DESCRIPTIVE.
#  * Hard checks: both files must use the same pi_ref and the same target
#    sensitivity -- otherwise the standardized PPVs are not comparable.
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # never open windows: plt.show() hangs/crashes unattended Windows runs
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu

COUNT_COLS = ["TP", "FP", "TN", "FN"]


def _ppv_std_from_counts(tp, fp, tn, fn, pi_ref):
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    if not (np.isfinite(sens) and np.isfinite(spec)):
        return np.nan
    denom = sens * pi_ref + (1.0 - spec) * (1.0 - pi_ref)
    return float(sens * pi_ref / denom) if denom > 0 else np.nan


def _require_repeat(df: pd.DataFrame) -> pd.DataFrame:
    if "repeat" in df.columns:
        return df
    raise ValueError("per-fold file has no 'repeat' column -- re-run run_ppv.py (current version) to regenerate it.")


def _pooled_estimate(sub: pd.DataFrame, pi_ref: float) -> float:
    """Mean over repeats of PPV_std from counts pooled over each repeat's folds."""
    vals = []
    for _, r in sub.groupby("repeat"):
        tp, fp, tn, fn = (float(r[c].sum()) for c in COUNT_COLS)
        vals.append(_ppv_std_from_counts(tp, fp, tn, fn, pi_ref))
    vals = np.asarray(vals, dtype=float)
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan


def _boot_pooled(sub: pd.DataFrame, pi_ref: float, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """Bootstrap folds (with replacement), pool their counts, recompute PPV_std."""
    counts = sub[COUNT_COLS].to_numpy(dtype=float)
    n = len(counts)
    out = np.full(n_boot, np.nan)
    for b in range(n_boot):
        c = counts[rng.integers(0, n, size=n)].sum(axis=0)
        out[b] = _ppv_std_from_counts(*c, pi_ref)
    return out


def _hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sp = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if sp == 0:
        return np.nan
    j = 1.0 - 3.0 / (4.0 * (nx + ny) - 9.0)
    return float(j * (np.mean(x) - np.mean(y)) / sp)


def _q(a: np.ndarray, q: float) -> float:
    return float(np.nanquantile(a, q)) if np.isfinite(a).any() else np.nan


def _check_same(full: pd.DataFrame, demo: pd.DataFrame, col: str, tol: float) -> float:
    vf = full[col].dropna().unique()
    vd = demo[col].dropna().unique()
    if len(vf) != 1 or len(vd) != 1:
        raise ValueError(f"'{col}' is not constant within a file (full: {vf}, demo: {vd}).")
    if abs(float(vf[0]) - float(vd[0])) > tol:
        raise ValueError(
            f"Full and Demo were run with different {col} ({vf[0]} vs {vd[0]}). "
            f"Standardized PPVs are only comparable with the SAME {col}: re-run the Demo side "
            f"with --pi_ref_from <Full CSV> (or --pi_ref with all digits of the Full pi_ref) and the same "
            f"--target_sens.")
    return float(vf[0])


def _check_same_model_params(full_csv: str, demo_csv: str) -> None:
    """run_ppv.py writes ppv_run_info.json next to its CSVs. Both sides must
    have been fitted with the same model settings (e.g. never a re-run MLP on one side only)."""
    import json
    infos = []
    for p in (full_csv, demo_csv):
        fp = os.path.join(os.path.dirname(os.path.abspath(p)), "ppv_run_info.json")
        infos.append(json.load(open(fp, encoding="utf-8")) if os.path.exists(fp) else None)
    f, d = infos
    if f is None and d is None:
        print("[compare_ppv] NOTE: no ppv_run_info.json on either side (older run_ppv.py); model settings not checked.")
        return
    if f is None or d is None:
        raise ValueError("[compare_ppv] only one side has ppv_run_info.json -> Full and Demo were made by different "
                         "run_ppv.py versions (e.g. MLP re-run on one side only). Re-run the other side.")
    if f.get("model_params") != d.get("model_params"):
        diff = sorted(k for k in set(f["model_params"]) | set(d["model_params"])
                      if f["model_params"].get(k) != d["model_params"].get(k))
        raise ValueError(f"[compare_ppv] Full and Demo used different model settings for {diff} -- not comparable.")


def compare_full_vs_demo(per_fold_full: pd.DataFrame, per_fold_demo: pd.DataFrame, *,
                         outdir: str, n_boot: int = 5000, seed: int = 42, make_plots: bool = True):
    os.makedirs(outdir, exist_ok=True)
    need = {"model", "fold", "repeat", "pi_ref", "target_sens_train", "ppv_standardized_pi_ref",
            "sens_test", "spec_test", *COUNT_COLS}
    for name, df in [("full", per_fold_full), ("demo", per_fold_demo)]:
        missing = sorted(need - set(df.columns))
        if missing:
            raise ValueError(f"{name} per-fold file missing columns: {missing} (re-run run_ppv.py).")

    pi_ref = _check_same(per_fold_full, per_fold_demo, "pi_ref", tol=1e-9)
    target_sens = _check_same(per_fold_full, per_fold_demo, "target_sens_train", tol=1e-9)

    models_f = set(per_fold_full["model"])
    models_d = set(per_fold_demo["model"])
    if models_f != models_d:
        print(f"[compare_ppv] WARNING: model sets differ; comparing the intersection. "
              f"Only full: {sorted(models_f - models_d)}; only demo: {sorted(models_d - models_f)}")
    models = sorted(models_f & models_d)
    if not models:
        raise ValueError("No model in common between the Full and Demo files.")

    rng = np.random.default_rng(seed)
    rows = []
    for m in models:
        F = per_fold_full[per_fold_full["model"] == m]
        D = per_fold_demo[per_fold_demo["model"] == m]
        est_f = _pooled_estimate(F, pi_ref)
        est_d = _pooled_estimate(D, pi_ref)
        bf = _boot_pooled(F, pi_ref, rng, n_boot)
        bd = _boot_pooled(D, pi_ref, rng, n_boot)
        diff = bf - bd
        xf = F["ppv_standardized_pi_ref"].to_numpy(dtype=float)
        xd = D["ppv_standardized_pi_ref"].to_numpy(dtype=float)
        ff, dd = xf[np.isfinite(xf)], xd[np.isfinite(xd)]
        p_welch = float(ttest_ind(ff, dd, equal_var=False).pvalue) if len(ff) >= 2 and len(dd) >= 2 else np.nan
        try:
            p_mwu = float(mannwhitneyu(ff, dd, alternative="two-sided").pvalue) if len(ff) and len(dd) else np.nan
        except ValueError:
            p_mwu = np.nan
        rows.append({
            "model": m,
            "n_folds_full": int(len(ff)), "n_folds_demo": int(len(dd)),
            "ppv_std_pooled_full": est_f, "ppv_std_pooled_demo": est_d,
            "diff_full_minus_demo": est_f - est_d,
            "diff_boot_ci_low": _q(diff, 0.025), "diff_boot_ci_high": _q(diff, 0.975),
            "full_boot_ci_low": _q(bf, 0.025), "full_boot_ci_high": _q(bf, 0.975),
            "demo_boot_ci_low": _q(bd, 0.025), "demo_boot_ci_high": _q(bd, 0.975),
            "sens_mean_full": float(F["sens_test"].mean()), "sens_mean_demo": float(D["sens_test"].mean()),
            "spec_mean_full": float(F["spec_test"].mean()), "spec_mean_demo": float(D["spec_test"].mean()),
            "per_fold_mean_full": float(np.mean(ff)) if len(ff) else np.nan,
            "per_fold_mean_demo": float(np.mean(dd)) if len(dd) else np.nan,
            "p_welch_unpaired_descriptive": p_welch,
            "p_mannwhitney_unpaired_descriptive": p_mwu,
            "hedges_g_per_fold": _hedges_g(ff, dd),
        })

    summary = pd.DataFrame(rows).sort_values("diff_full_minus_demo", ascending=False).reset_index(drop=True)
    summary.insert(1, "pi_ref", pi_ref)
    summary.insert(2, "target_sens", target_sens)
    summary.to_csv(os.path.join(outdir, "per_model_full_vs_demo_unpaired.csv"), index=False)

    if make_plots:
        xs = np.arange(len(summary))
        fu, de = summary["ppv_std_pooled_full"], summary["ppv_std_pooled_demo"]
        plt.figure(figsize=(8, 5))
        plt.errorbar(xs - 0.1, fu, yerr=[fu - summary["full_boot_ci_low"], summary["full_boot_ci_high"] - fu],
                     fmt="o", label="Full")
        plt.errorbar(xs + 0.1, de, yerr=[de - summary["demo_boot_ci_low"], summary["demo_boot_ci_high"] - de],
                     fmt="s", label="Demo")
        plt.xticks(xs, summary["model"], rotation=45, ha="right")
        plt.ylabel(f"PPV standardized to pi_ref={pi_ref:.4f} (pooled per repeat)")
        plt.title(f"Standardized PPV at target sensitivity {target_sens:.2f}: Full vs Demo")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "plot_ppvstd_full_vs_demo.png"), dpi=200)
        plt.close()

        y = summary["diff_full_minus_demo"].to_numpy(float)
        plt.figure(figsize=(8, 5))
        plt.axhline(0.0, color="grey", lw=1)
        plt.errorbar(xs, y, yerr=[y - summary["diff_boot_ci_low"], summary["diff_boot_ci_high"] - y], fmt="o")
        plt.xticks(xs, summary["model"], rotation=45, ha="right")
        plt.ylabel("Full - Demo (standardized PPV)")
        plt.title("Difference with independent-bootstrap 95% CI (descriptive)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "plot_ppvstd_diff_ci.png"), dpi=200)
        plt.close()

    meta = {"pi_ref": pi_ref, "target_sens": target_sens, "models": models, "outdir": outdir}
    return summary, meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare prevalence-standardized PPV between Full and Demo (unpaired).")
    p.add_argument("--full_per_fold", type=str, required=True, help="Full side's ppv_std_per_fold.csv (run_ppv.py).")
    p.add_argument("--demo_per_fold", type=str, required=True, help="Demo side's ppv_std_per_fold.csv (run_ppv.py).")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--n_boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    full = _require_repeat(pd.read_csv(args.full_per_fold))
    demo = _require_repeat(pd.read_csv(args.demo_per_fold))
    _check_same_model_params(args.full_per_fold, args.demo_per_fold)
    summary, meta = compare_full_vs_demo(full, demo, outdir=args.outdir, n_boot=args.n_boot,
                                         seed=args.seed, make_plots=not args.no_plots)
    print("META:", meta)
    print(summary[["model", "ppv_std_pooled_full", "ppv_std_pooled_demo", "diff_full_minus_demo",
                   "diff_boot_ci_low", "diff_boot_ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
