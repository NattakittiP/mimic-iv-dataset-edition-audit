import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# 1) Load your aggregated metrics (from the zip output)
df = pd.read_csv("metrics_aggregated_deltas_demo_vs_full.csv")

def build_world_rank_matrix(df, score_col, out_csv, out_png, title):
    # Pivot: rows=world, cols=model, values=score (e.g., AUROC_mean_full)
    pivot = df.pivot_table(index="world", columns="model", values=score_col, aggfunc="mean")

    # Convert scores to ranks within each world (higher AUROC = better rank)
    ranks = pivot.rank(axis=1, ascending=False, method="average")

    worlds = ranks.index.tolist()
    M = np.zeros((len(worlds), len(worlds)), dtype=float)

    for i, wi in enumerate(worlds):
        for j, wj in enumerate(worlds):
            rho, _ = spearmanr(ranks.loc[wi].values, ranks.loc[wj].values)
            M[i, j] = rho

    mat = pd.DataFrame(M, index=worlds, columns=worlds)
    mat.to_csv(out_csv, index=True)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(mat.values, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, label="Spearman rank correlation")
    plt.xticks(range(len(worlds)), worlds, rotation=45, ha="right")
    plt.yticks(range(len(worlds)), worlds)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.show()

# FULL
build_world_rank_matrix(
    df=df,
    score_col="AUROC_mean_full",
    out_csv="world_rank_spearman_matrix_full.csv",
    out_png="fig_world_rank_spearman_heatmap_full.png",
    title="World-to-World Rank Agreement (Full) using AUROC_mean_full"
)

# DEMO
build_world_rank_matrix(
    df=df,
    score_col="AUROC_mean_demo",
    out_csv="world_rank_spearman_matrix_demo.csv",
    out_png="fig_world_rank_spearman_heatmap_demo.png",
    title="World-to-World Rank Agreement (Demo) using AUROC_mean_demo"
)
