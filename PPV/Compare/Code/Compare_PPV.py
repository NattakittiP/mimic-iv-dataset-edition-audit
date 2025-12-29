import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# High-quality comparison suite
# -----------------------------
def _cohen_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for paired samples (mean(diff)/std(diff))."""
    d = x - y
    if len(d) < 2:
        return np.nan
    sd = np.std(d, ddof=1)
    return float(np.mean(d) / sd) if sd > 0 else np.nan

def _bootstrap_ci_of_mean_diff(d: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 42):
    """Bootstrap CI for mean(d)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return (float(d.mean()), float(lo), float(hi))

def _paired_wilcoxon_if_available(x: np.ndarray, y: np.ndarray):
    """Return Wilcoxon signed-rank p-value if scipy is available, else NaN."""
    try:
        from scipy.stats import wilcoxon
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            return np.nan
        if np.allclose(x[m], y[m]):
            return 1.0
        stat, p = wilcoxon(
            x[m], y[m],
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            mode="auto"
        )
        return float(p)
    except Exception:
        return np.nan

def _paired_ttest_if_available(x: np.ndarray, y: np.ndarray):
    """Return paired t-test p-value if scipy is available, else NaN."""
    try:
        from scipy.stats import ttest_rel
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            return np.nan
        if np.allclose(x[m], y[m]):
            return 1.0
        stat, p = ttest_rel(x[m], y[m], nan_policy="omit")
        return float(p)
    except Exception:
        return np.nan

def compare_full_vs_demo(
    per_fold_full: pd.DataFrame,
    per_fold_demo: pd.DataFrame,
    *,
    outdir: str = "/content/compare_ppvstd",
    metric_cols=(
        "ppv_standardized_pi_ref",
        "ppv_observed_test",
        "sens_test",
        "spec_test",
        "thr_from_train_inner_oof",
        "pi_test",
    ),
    key_cols=("fold", "model"),
    n_boot: int = 5000,
    seed: int = 42,
    make_plots: bool = True,
):
    os.makedirs(outdir, exist_ok=True)

    # --------- 1) Sanity checks ----------
    needed = set(key_cols) | set(metric_cols)
    for name, df in [("per_fold_full", per_fold_full), ("per_fold_demo", per_fold_demo)]:
        missing = sorted(list(needed - set(df.columns)))
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")

    # Align fold/model pairs
    A = per_fold_full[list(key_cols) + list(metric_cols)].copy()
    B = per_fold_demo[list(key_cols) + list(metric_cols)].copy()

    merged = A.merge(B, on=list(key_cols), how="inner", suffixes=("_full", "_demo"))
    if len(merged) == 0:
        raise ValueError(
            "No matching (fold, model) rows between full and demo. "
            "Ensure both used same CV scheme (fold numbering)."
        )

    # Identify dropped rows (if any)
    only_full = A.merge(B[list(key_cols)], on=list(key_cols), how="left", indicator=True)
    only_demo = B.merge(A[list(key_cols)], on=list(key_cols), how="left", indicator=True)
    n_only_full = int((only_full["_merge"] == "left_only").sum())
    n_only_demo = int((only_demo["_merge"] == "left_only").sum())

    # --------- 2) Per-model paired comparisons ----------
    results = []
    for model, g in merged.groupby("model", sort=True):
        row = {"model": model, "n_pairs": int(len(g))}
        for m in metric_cols:
            x = g[f"{m}_full"].to_numpy(dtype=float)
            y = g[f"{m}_demo"].to_numpy(dtype=float)
            d = x - y

            row[f"{m}_mean_full"] = float(np.nanmean(x))
            row[f"{m}_mean_demo"] = float(np.nanmean(y))
            row[f"{m}_mean_diff_full_minus_demo"] = float(np.nanmean(d))
            row[f"{m}_sd_diff"] = float(np.nanstd(d, ddof=1)) if np.isfinite(d).sum() >= 2 else np.nan
            row[f"{m}_cohen_d_paired"] = _cohen_d_paired(x, y)

            mean_d, lo, hi = _bootstrap_ci_of_mean_diff(d, n_boot=n_boot, seed=seed)
            row[f"{m}_diff_boot_mean"] = mean_d
            row[f"{m}_diff_boot_ci_low"] = lo
            row[f"{m}_diff_boot_ci_high"] = hi

            row[f"{m}_p_ttest"] = _paired_ttest_if_available(x, y)
            row[f"{m}_p_wilcoxon"] = _paired_wilcoxon_if_available(x, y)

        results.append(row)

    summary = pd.DataFrame(results)

    # Ranking by standardized PPV difference
    rank_col = "ppv_standardized_pi_ref_mean_diff_full_minus_demo"
    if rank_col in summary.columns:
        summary = summary.sort_values(rank_col, ascending=False).reset_index(drop=True)

    # --------- 3) Overall pooled paired comparison for main metric ----------
    main = "ppv_standardized_pi_ref"
    pooled = None
    if f"{main}_full" in merged.columns:
        x = merged[f"{main}_full"].to_numpy(float)
        y = merged[f"{main}_demo"].to_numpy(float)
        d = x - y
        mean_d, lo, hi = _bootstrap_ci_of_mean_diff(d, n_boot=n_boot, seed=seed)
        pooled = pd.DataFrame([{
            "metric": main,
            "n_pairs": int(np.isfinite(d).sum()),
            "mean_full": float(np.nanmean(x)),
            "mean_demo": float(np.nanmean(y)),
            "mean_diff_full_minus_demo": float(np.nanmean(d)),
            "boot_ci_low": lo,
            "boot_ci_high": hi,
            "p_ttest": _paired_ttest_if_available(x, y),
            "p_wilcoxon": _paired_wilcoxon_if_available(x, y),
            "cohen_d_paired": _cohen_d_paired(x, y),
        }])

    # --------- 4) Save outputs ----------
    merged.to_csv(f"{outdir}/per_fold_merged_full_vs_demo.csv", index=False)
    summary.to_csv(f"{outdir}/per_model_paired_comparison.csv", index=False)
    if pooled is not None:
        pooled.to_csv(f"{outdir}/pooled_main_metric_comparison.csv", index=False)

    # --------- 5) Plots ----------
    if make_plots:
        # (A) per-model mean standardized PPV (full vs demo)
        if ("ppv_standardized_pi_ref_mean_full" in summary.columns and
            "ppv_standardized_pi_ref_mean_demo" in summary.columns):
            dfp = summary[["model", "ppv_standardized_pi_ref_mean_full", "ppv_standardized_pi_ref_mean_demo"]].copy()
            dfp = dfp.sort_values("ppv_standardized_pi_ref_mean_full", ascending=False)
            xidx = np.arange(len(dfp))

            plt.figure()
            plt.plot(xidx, dfp["ppv_standardized_pi_ref_mean_full"], marker="o", label="FULL")
            plt.plot(xidx, dfp["ppv_standardized_pi_ref_mean_demo"], marker="o", label="DEMO")
            plt.xticks(xidx, dfp["model"], rotation=45, ha="right")
            plt.ylabel("Mean PPV (standardized to pi_ref)")
            plt.title("Per-model mean standardized PPV: FULL vs DEMO")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{outdir}/plot_mean_ppvstd_full_vs_demo.png", dpi=160)
            plt.show()

        # (B) distribution of paired diffs for main metric (pooled)
        if f"{main}_full" in merged.columns:
            d = (merged[f"{main}_full"] - merged[f"{main}_demo"]).to_numpy(float)
            d = d[np.isfinite(d)]
            plt.figure()
            plt.hist(d, bins=30)
            plt.xlabel("Diff (FULL - DEMO) of standardized PPV")
            plt.ylabel("Count")
            plt.title("Pooled paired differences: standardized PPV (FULL - DEMO)")
            plt.tight_layout()
            plt.savefig(f"{outdir}/hist_diff_ppvstd_pooled.png", dpi=160)
            plt.show()

        # (C) per-model CI bars for standardized PPV diff (bootstrap)
        ci_cols = [
            "ppv_standardized_pi_ref_diff_boot_ci_low",
            "ppv_standardized_pi_ref_diff_boot_mean",
            "ppv_standardized_pi_ref_diff_boot_ci_high"
        ]
        if all(c in summary.columns for c in ci_cols):
            dfc = summary[["model"] + ci_cols].copy()
            dfc = dfc.sort_values("ppv_standardized_pi_ref_diff_boot_mean", ascending=False)
            xidx = np.arange(len(dfc))
            y = dfc["ppv_standardized_pi_ref_diff_boot_mean"].to_numpy(float)
            lo = dfc["ppv_standardized_pi_ref_diff_boot_ci_low"].to_numpy(float)
            hi = dfc["ppv_standardized_pi_ref_diff_boot_ci_high"].to_numpy(float)

            plt.figure()
            plt.axhline(0.0)
            plt.errorbar(xidx, y, yerr=[y - lo, hi - y], fmt="o")
            plt.xticks(xidx, dfc["model"], rotation=45, ha="right")
            plt.ylabel("Bootstrap mean diff (FULL - DEMO)")
            plt.title("Per-model standardized PPV difference with bootstrap CI")
            plt.tight_layout()
            plt.savefig(f"{outdir}/plot_ci_diff_ppvstd_per_model.png", dpi=160)
            plt.show()

    # --------- 6) Compact “topline” view ----------
    topline_cols = [
        "model", "n_pairs",
        "ppv_standardized_pi_ref_mean_full",
        "ppv_standardized_pi_ref_mean_demo",
        "ppv_standardized_pi_ref_mean_diff_full_minus_demo",
        "ppv_standardized_pi_ref_diff_boot_ci_low",
        "ppv_standardized_pi_ref_diff_boot_ci_high",
        "ppv_standardized_pi_ref_p_wilcoxon",
        "ppv_standardized_pi_ref_p_ttest",
        "ppv_standardized_pi_ref_cohen_d_paired",
    ]
    topline = summary[[c for c in topline_cols if c in summary.columns]].copy()

    meta = {
        "n_merged_rows": int(len(merged)),
        "n_only_full_rows_dropped": n_only_full,
        "n_only_demo_rows_dropped": n_only_demo,
        "outdir": outdir,
    }

    return merged, summary, topline, pooled, meta


# -----------------------------
# RUN comparison using attached files
# -----------------------------
full_per_fold_path = "full_ppv_std_per_fold.csv"
demo_per_fold_path = "demo_ppv_std_per_fold.csv"

per_fold_full = pd.read_csv(full_per_fold_path)
per_fold_demo = pd.read_csv(demo_per_fold_path)

merged, per_model, topline, pooled_main, meta = compare_full_vs_demo(
    per_fold_full=per_fold_full,
    per_fold_demo=per_fold_demo,
    outdir="content/compare_ppvstd",
    n_boot=5000,
    seed=42,
    make_plots=True,
)

print("META:", meta)
print("\n=== TOPLINE (per model) ===")
print(topline.to_string(index=False))

if pooled_main is not None:
    print("\n=== POOLED MAIN METRIC ===")
    print(pooled_main.to_string(index=False))