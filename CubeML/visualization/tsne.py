from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from CubeML.visualization.overview import encode_features_as_numeric, style_axes


DEFAULT_TSNE_MAX_SAMPLES = 5000


def create_tsne_figure(
    df: pd.DataFrame,
    misplaced_stickers: pd.Series,
    max_samples: int = DEFAULT_TSNE_MAX_SAMPLES,
    random_state: int = 42,
) -> plt.Figure:
    # t-SNE is expensive, so we keep a reproducible sample for a fast visual overview.
    if len(df) > max_samples:
        sampled_index = df.sample(n=max_samples, random_state=random_state).index
        sampled_df = df.loc[sampled_index]
        sampled_misplaced = misplaced_stickers.loc[sampled_index]
    else:
        sampled_df = df
        sampled_misplaced = misplaced_stickers

    X = encode_features_as_numeric(sampled_df).to_numpy(dtype=np.float32)
    X_scaled = StandardScaler().fit_transform(X)

    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    X_tsne = tsne.fit_transform(X_scaled)

    figure, (moves_ax, complexity_ax) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    move_codes, move_labels = pd.factorize(sampled_df["MOVE"], sort=True)
    cmap = plt.get_cmap("tab20", max(len(move_labels), 1))
    moves_ax.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=move_codes,
        cmap=cmap,
        s=6,
        alpha=0.55,
        linewidths=0,
        rasterized=True,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(i),
            markeredgecolor="none",
            markersize=6,
            label=str(label),
        )
        for i, label in enumerate(move_labels)
    ]
    legend_cols = 2 if len(move_labels) <= 10 else 3
    moves_ax.legend(handles=legend_handles, title="Move", frameon=False, ncol=legend_cols, loc="best")
    moves_ax.set_title("t-SNE colored by Move")
    moves_ax.set_xlabel("t-SNE dimension 1")
    moves_ax.set_ylabel("t-SNE dimension 2")
    style_axes(moves_ax, grid_axis="both")

    complexity_scatter = complexity_ax.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=sampled_misplaced.to_numpy(),
        cmap="viridis",
        s=6,
        alpha=0.55,
        linewidths=0,
        rasterized=True,
    )
    colorbar = figure.colorbar(complexity_scatter, ax=complexity_ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Misplaced stickers")
    complexity_ax.set_title("t-SNE colored by Complexity")
    complexity_ax.set_xlabel("t-SNE dimension 1")
    complexity_ax.set_ylabel("t-SNE dimension 2")
    style_axes(complexity_ax, grid_axis="both")

    figure.suptitle(
        f"CubeML t-SNE Overview (n={len(sampled_df):,})",
        fontsize=14,
        fontweight="bold",
    )
    return figure
