"""
Demo-vs-Full ranking under RSCE_RSC (derived from existing RSCE outputs).

RSCE_RSC = (0.4 R + 0.3 S_ratio + 0.2 C_linear) / 0.9 for every model, the like-for-like
score that removes the asymmetry in E availability. The script reports, per domain and
score (RSCE_full and RSCE_RSC), the leader on Full and on Demo, the rank of each leader
on the other release, and Spearman/Kendall agreement between the Demo and Full scores.

Inputs (read only):  Results/<dom>_{full,demo}/rsce/rsce_scores.csv
Output (new file):   Results/compare/rsce_rsc_rank_agreement.csv
No model is refitted; no existing result file is modified.
"""
import argparse
from pathlib import Path
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def main(results: Path) -> None:
    rows = []
    for dom, (full, demo) in {"hosp": ("hosp_full", "hosp_demo"), "ed": ("ed_full", "ed_demo")}.items():
        F = pd.read_csv(results / full / "rsce" / "rsce_scores.csv").set_index("model")
        D = pd.read_csv(results / demo / "rsce" / "rsce_scores.csv").set_index("model").loc[F.index]
        for col in ("RSCE_full", "RSCE_RSC"):
            rf, rd = F[col].rank(ascending=False), D[col].rank(ascending=False)
            assert rf.is_unique and rd.is_unique, "ties in ranks"
            fl, dl = F[col].idxmax(), D[col].idxmax()
            rows.append({
                "domain": dom, "score": col, "full_leader": fl, "demo_leader": dl,
                "full_leader_rank_on_demo": int(rd[fl]), "demo_leader_rank_on_full": int(rf[dl]),
                "spearman_rho": float(spearmanr(F[col], D[col])[0]),
                "kendall_tau": float(kendalltau(F[col], D[col])[0]),
                "full_order": " > ".join(F[col].sort_values(ascending=False).index),
                "demo_order": " > ".join(D[col].sort_values(ascending=False).index),
            })
    out = pd.DataFrame(rows)
    path = results / "compare" / "rsce_rsc_rank_agreement.csv"
    out.to_csv(path, index=False)
    print(out.to_string())
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[2] / "Results")
    main(ap.parse_args().results)
