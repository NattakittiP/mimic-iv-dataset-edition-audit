#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_world_heatmap.py
========================
World-to-world rank agreement heatmap: for each pair of "worlds" (perturbations),
how consistently do models get ranked against each other? Built separately for
Demo and Full from the `metrics_aggregated_deltas_demo_vs_full.csv` produced by
compare_pro.py (compare_metrics_aggregated()).

Input/output locations are CLI args, so it works for any dataset variant
(hosp or ED) without editing the file.

Usage:
  python compare_world_heatmap.py \
      --input results/compare/hosp/metrics_aggregated_deltas_demo_vs_full.csv \
      --outdir results/compare/hosp \
      --score_col AUROC_mean
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: these scripts only ever plt.savefig()/plt.close(),
                        # never plt.show() -- forcing Agg avoids a real crash seen on Windows where the
                        # auto-selected interactive TkAgg backend hit a Tcl/Tk "wrong thread" assertion
                        # (Tcl_AsyncDelete) and killed the whole process (exit code 0x80000003) partway
                        # through a run_rsce.py SHAP-plotting pass.
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def build_world_rank_matrix(df: pd.DataFrame, score_col: str, out_csv: Path, out_png: Path, title: str) -> None:
    pivot = df.pivot_table(index="world", columns="model", values=score_col, aggfunc="mean")
    # order worlds by perturbation severity (clean first) instead of alphabetically
    sev_col = next((c for c in ("severity", "severity_full", "severity_demo") if c in df.columns), None)
    if sev_col is not None:
        order = df.groupby("world")[sev_col].first().sort_values().index.tolist()
        pivot = pivot.reindex([w for w in order if w in pivot.index])
    ranks = pivot.rank(axis=1, ascending=False, method="average")

    worlds = ranks.index.tolist()
    M = np.zeros((len(worlds), len(worlds)), dtype=float)
    for i, wi in enumerate(worlds):
        for j, wj in enumerate(worlds):
            rho, _ = spearmanr(ranks.loc[wi].values, ranks.loc[wj].values)
            M[i, j] = rho

    mat = pd.DataFrame(M, index=worlds, columns=worlds)
    mat.to_csv(out_csv, index=True)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(mat.values, vmin=-1, vmax=1, aspect="auto", cmap="coolwarm")  # diverging, centred on 0
    plt.colorbar(im, label="Spearman rank correlation")
    plt.xticks(range(len(worlds)), worlds, rotation=45, ha="right")
    plt.yticks(range(len(worlds)), worlds)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="World-to-world rank agreement heatmap (Demo vs Full) from a metrics_aggregated_deltas file.")
    p.add_argument("--input", type=str, required=True, help="Path to metrics_aggregated_deltas_demo_vs_full.csv (from compare_pro.py).")
    p.add_argument("--outdir", type=str, required=True, help="Output folder.")
    p.add_argument("--score_col_full", type=str, default=None, help="Column to use for the FULL heatmap (default: <score_col>_full).")
    p.add_argument("--score_col_demo", type=str, default=None, help="Column to use for the DEMO heatmap (default: <score_col>_demo).")
    p.add_argument("--score_col", type=str, default="AUROC_mean", help="Base metric column name (e.g. AUROC_mean, ECE_mean).")
    p.add_argument("--dataset_tag", type=str, default="dataset", help="Label used in plot titles.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    score_full = args.score_col_full or f"{args.score_col}_full"
    score_demo = args.score_col_demo or f"{args.score_col}_demo"

    build_world_rank_matrix(
        df=df,
        score_col=score_full,
        out_csv=outdir / "world_rank_spearman_matrix_full.csv",
        out_png=outdir / "fig_world_rank_spearman_heatmap_full.png",
        title=f"World-to-World Rank Agreement (Full, {args.dataset_tag}) using {score_full}",
    )

    build_world_rank_matrix(
        df=df,
        score_col=score_demo,
        out_csv=outdir / "world_rank_spearman_matrix_demo.csv",
        out_png=outdir / "fig_world_rank_spearman_heatmap_demo.png",
        title=f"World-to-World Rank Agreement (Demo, {args.dataset_tag}) using {score_demo}",
    )

    print(f"[Done] World rank heatmaps written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
