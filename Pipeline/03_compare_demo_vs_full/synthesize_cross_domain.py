#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synthesize_cross_domain.py
===========================
The Demo-vs-Full comparison is run in two structurally different domains:
Hospital-module mortality and ED-module admission disposition. Every other
script in this folder compares Demo vs Full WITHIN one domain; this one asks
the cross-domain question: does the Demo-vs-Full relationship (rank
agreement, direction/size of ΔRSCE, sign-test result) look the SAME in both
domains, or does it diverge?

This script reads the compare_pro.py outputs for the hosp track and the ED
track and puts them side by side, plus a short auto-generated narrative
flagging whether the two domains agree or disagree on each headline number.

Usage:
  python synthesize_cross_domain.py \
      --hosp_compare_dir ../Results/compare/hosp \
      --ed_compare_dir   ../Results/compare/ed \
      --outdir ../Results/compare/cross_domain
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def log(msg: str) -> None:
    print(f"[synthesize_cross_domain] {msg}")


def _read_csv_safe(path: Path) -> pd.DataFrame:
    # Missing inputs are an ERROR: a typo in --ed_compare_dir used to produce a
    # summary saying "not available" and exit 0.
    if not path.exists():
        raise FileNotFoundError(f"missing {path} -- run compare_pro.py for this domain first.")
    return pd.read_csv(path)


def load_domain(compare_dir: Path, tag: str) -> Dict[str, object]:
    compare_dir = Path(compare_dir)
    rsce_cmp = _read_csv_safe(compare_dir / "rsce_comparison_demo_vs_full.csv")
    rank_agree = _read_csv_safe(compare_dir / "rank_agreement.csv")
    sign_test = _read_csv_safe(compare_dir / "sign_test_summary.csv")
    decomp = _read_csv_safe(compare_dir / "delta_rsce_decomposition.csv")

    out: Dict[str, object] = {"domain": tag}
    if rsce_cmp is not None and "RSCE_full_delta" in rsce_cmp.columns:
        out["n_models"] = int(len(rsce_cmp))
        out["mean_delta_rsce_full_minus_demo"] = float(rsce_cmp["RSCE_full_delta"].mean())
        out["median_delta_rsce_full_minus_demo"] = float(rsce_cmp["RSCE_full_delta"].median())
        out["pct_models_full_gt_demo"] = float((rsce_cmp["RSCE_full_delta"] > 0).mean() * 100.0)
        out["min_delta"] = float(rsce_cmp["RSCE_full_delta"].min())
        out["max_delta"] = float(rsce_cmp["RSCE_full_delta"].max())
        out["best_model_full"] = rsce_cmp.sort_values("RSCE_full_full", ascending=False).iloc[0]["model"] if "RSCE_full_full" in rsce_cmp.columns else None
        out["best_model_demo"] = rsce_cmp.sort_values("RSCE_full_demo", ascending=False).iloc[0]["model"] if "RSCE_full_demo" in rsce_cmp.columns else None
        out["best_model_agrees"] = (out["best_model_full"] == out["best_model_demo"]) if out["best_model_full"] and out["best_model_demo"] else None
    if rank_agree is not None and len(rank_agree):
        out["spearman_rank_demo_vs_full"] = float(rank_agree.iloc[0].get("spearman_rank", np.nan))
        out["kendall_rank_demo_vs_full"] = float(rank_agree.iloc[0].get("kendall_rank", np.nan))
    if sign_test is not None and len(sign_test):
        out["sign_test_p"] = float(sign_test.iloc[0].get("p_value_two_sided_binomtest", np.nan))
        out["sign_test_positive"] = int(sign_test.iloc[0].get("positive", np.nan))
        out["sign_test_negative"] = int(sign_test.iloc[0].get("negative", np.nan))
    if decomp is not None and len(decomp):
        for comp in ("R", "S", "C", "E"):
            col = f"contrib_{comp}"
            if col in decomp.columns:
                out[f"mean_contrib_{comp}"] = float(decomp[col].mean())
    return out


def build_narrative(hosp: Dict[str, object], ed: Dict[str, object]) -> str:
    lines = ["# Cross-domain synthesis: Hospital module vs ED module\n"]
    lines.append(
        "Auto-generated from `compare_pro.py` outputs for each domain. This is descriptive "
        "(no new hypothesis test across domains -- domains use different cohorts/targets and "
        "aren't directly poolable), meant to make it easy to see, at a glance, whether the "
        "Demo-vs-Full pattern generalizes.\n"
    )

    def g(d, k):
        return d.get(k, None)

    lines.append("## Direction of ΔRSCE (Full - Demo)\n")
    for tag, d in [("Hospital", hosp), ("ED", ed)]:
        md = g(d, "mean_delta_rsce_full_minus_demo")
        pct = g(d, "pct_models_full_gt_demo")
        p = g(d, "sign_test_p")
        if md is not None:
            if p is not None and np.isfinite(p) and p < 0.05:
                direction = "Full systematically HIGHER than Demo" if md > 0 else "Full systematically LOWER than Demo"
            else:
                direction = "no consistent direction (sign test p >= 0.05)"
            lines.append(f"- **{tag}**: mean ΔRSCE = {md:.4f}, median = {g(d, 'median_delta_rsce_full_minus_demo'):.4f}, "
                         f"range [{g(d, 'min_delta'):.4f}, {g(d, 'max_delta'):.4f}]; {pct:.0f}% of models have Full > Demo "
                         f"-> {direction}.")

    def verdict(d):
        p, md = d.get("sign_test_p"), d.get("mean_delta_rsce_full_minus_demo")
        if p is None or md is None or not np.isfinite(p) or p >= 0.05:
            return 0
        return 1 if md > 0 else -1
    vh, ve = verdict(hosp), verdict(ed)
    if vh == 0 and ve == 0:
        cons = "NO SYSTEMATIC DIRECTION in either domain (sign tests p >= 0.05) -- Demo neither over- nor under-states RSCE consistently"
    elif vh == ve:
        cons = "CONSISTENT: both domains show the same systematic direction"
    elif 0 in (vh, ve):
        cons = "PARTIAL: a systematic direction in one domain only"
    else:
        cons = "INCONSISTENT: the domains show opposite systematic directions -- discuss explicitly, do not average"
    lines.append(f"\n**Cross-domain consistency (based on the sign tests, not the sign of the mean):** {cons}.\n")

    lines.append("\n## Which component drives ΔRSCE (mean weighted contribution, exact decomposition)\n")
    for tag, d in [("Hospital", hosp), ("ED", ed)]:
        parts = [f"{c}={d[f'mean_contrib_{c}']:+.4f}" for c in ("R", "S", "C", "E") if d.get(f"mean_contrib_{c}") is not None]
        lines.append(f"- **{tag}**: " + (", ".join(parts) if parts else "not available."))

    lines.append("\n## Rank agreement (Spearman, Demo vs Full)\n")
    for tag, d in [("Hospital", hosp), ("ED", ed)]:
        rho = g(d, "spearman_rank_demo_vs_full")
        lines.append(f"- **{tag}**: Spearman rho = {rho:.3f}" if rho is not None else f"- **{tag}**: not available.")

    lines.append("\n## Best model: does Demo pick the same winner as Full?\n")
    for tag, d in [("Hospital", hosp), ("ED", ed)]:
        agrees = g(d, "best_model_agrees")
        bf, bd = g(d, "best_model_full"), g(d, "best_model_demo")
        if agrees is not None:
            lines.append(f"- **{tag}**: Full-best={bf}, Demo-best={bd} -> {'MATCH' if agrees else 'MISMATCH'}.")
        else:
            lines.append(f"- **{tag}**: not available.")

    lines.append("\n## Sign test on ΔRSCE (two-sided binomial)\n")
    for tag, d in [("Hospital", hosp), ("ED", ed)]:
        p = g(d, "sign_test_p")
        pos, neg = g(d, "sign_test_positive"), g(d, "sign_test_negative")
        if p is not None:
            lines.append(f"- **{tag}**: {pos} positive / {neg} negative deltas, p={p:.4f}.")
        else:
            lines.append(f"- **{tag}**: not available.")

    lines.append(
        "\n---\n*Interpretation note:* Hospital-module and ED-module datasets have different "
        "cohorts, targets, prevalence and feature sets, so this is a consistency check on the "
        "DIRECTION and QUALITATIVE pattern of the Demo-vs-Full relationship, not a pooled "
        "statistical test. If the two domains agree, that's evidence the Demo-vs-Full finding "
        "generalizes beyond a single dataset; if they disagree, "
        "that is itself a finding worth reporting rather than a bug to fix."
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthesize hosp-domain and ED-domain compare_pro.py outputs into one cross-domain summary.")
    p.add_argument("--hosp_compare_dir", type=str, required=True, help="--outdir used for compare_pro.py on the hosp track.")
    p.add_argument("--ed_compare_dir", type=str, required=True, help="--outdir used for compare_pro.py on the ED track.")
    p.add_argument("--outdir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hosp = load_domain(Path(args.hosp_compare_dir), "hosp")
    ed = load_domain(Path(args.ed_compare_dir), "ed")

    combined = pd.DataFrame([hosp, ed])
    combined.to_csv(outdir / "cross_domain_summary.csv", index=False)

    narrative = build_narrative(hosp, ed)
    (outdir / "cross_domain_summary.md").write_text(narrative, encoding="utf-8")

    print("\n[Done] Cross-domain synthesis written to:", outdir)
    print(" - cross_domain_summary.csv")
    print(" - cross_domain_summary.md")
    print("\n" + narrative)


if __name__ == "__main__":
    main()
