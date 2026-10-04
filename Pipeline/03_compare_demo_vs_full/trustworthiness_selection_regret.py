"""
Selection regret of Demo-sized null draws (derived from existing trustworthiness outputs).

For each domain, pool and null mode, every draw "selects" the model with the highest
AUROC on that draw, exactly as exp3 of compare_trustworthiness.py does. The selection
regret of a draw is

    regret_u = AUROC_ref(best_ref) - AUROC_ref(selected_u),

where AUROC_ref are the whole-Full reference AUROCs of the same trustworthiness zoo
(full_reference_metrics.csv) and best_ref is the Full-best model. The script also
reproduces the exp3 hit probability as a consistency check against
exp3_decision_stability*.csv.

Inputs (read only):  Results/compare/<dom>/<pool>/{full_reference_metrics,demo_metrics,
                     subsample_metrics_long[_prevmatched],exp3_decision_stability[_prevmatched]}.csv
Output (new file):   Results/compare/trustworthiness_selection_regret.csv
No model is refitted; no existing result file is modified.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TOL = (0.005, 0.01)


def best(df: pd.DataFrame) -> str:
    # identical rule to compare_trustworthiness.topk_set(..., "AUROC", 1)
    d = df.dropna(subset=["AUROC"]).sort_values("AUROC", ascending=False)
    return d["model"].iloc[0]


def main(results: Path) -> None:
    rows = []
    for dom in ("hosp", "ed"):
        for pool, pool_name in (("trustworthiness", "all_patients"), ("trustworthiness_icu", "icu_patients")):
            base = results / "compare" / dom / pool
            ref = pd.read_csv(base / "full_reference_metrics.csv").set_index("model")["AUROC"]
            demo = pd.read_csv(base / "demo_metrics.csv")
            ref_best = best(ref.reset_index())
            demo_sel = best(demo)
            demo_regret = float(ref[ref_best] - ref[demo_sel])
            for suf, mode in (("", "random"), ("_prevmatched", "prevalence_matched")):
                long = pd.read_csv(base / f"subsample_metrics_long{suf}.csv")
                sel = long.groupby("run_id").apply(best)
                reg = (ref[ref_best] - ref.loc[sel.values].values).astype(float)
                hit = float(np.mean(sel.values == ref_best))
                exp3 = pd.read_csv(base / f"exp3_decision_stability{suf}.csv")
                exp3_hit = float(exp3["subsample_P(best_matches_full)"].iloc[0])
                assert abs(hit - exp3_hit) < 1e-12, (dom, pool, mode, hit, exp3_hit)
                pct = 100.0 * (np.mean(reg < demo_regret) + 0.5 * np.mean(reg == demo_regret))
                row = {
                    "domain": dom, "pool": pool_name, "null": mode, "n_draws": len(reg),
                    "full_best_model": ref_best, "full_best_auroc": float(ref[ref_best]),
                    "P_exact_full_best": hit,
                    "regret_mean": float(reg.mean()), "regret_median": float(np.median(reg)),
                    "regret_p90": float(np.quantile(reg, 0.9)), "regret_max": float(reg.max()),
                    "demo_selected_model": demo_sel, "demo_regret": demo_regret,
                    "demo_regret_percentile": float(pct),
                }
                for t in TOL:
                    row[f"P_regret_le_{t}"] = float(np.mean(reg <= t + 1e-12))
                row["ref_auroc_by_model"] = "; ".join(f"{m}={v:.6f}" for m, v in ref.sort_values(ascending=False).items())
                rows.append(row)
    out = pd.DataFrame(rows)
    path = results / "compare" / "trustworthiness_selection_regret.csv"
    out.to_csv(path, index=False)
    print(out.drop(columns=["ref_auroc_by_model"]).to_string())
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[2] / "Results")
    main(ap.parse_args().results)
