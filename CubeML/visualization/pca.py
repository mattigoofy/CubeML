from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from CubeML.visualization.overview import encode_features_as_numeric, style_axes


def create_pca_figure(df: pd.DataFrame, misplaced_stickers: pd.Series) -> plt.Figure:
    feature_df = encode_features_as_numeric(df)
    feature_names = feature_df.columns.tolist()
    X = feature_df.to_numpy(dtype=np.float32)

    pca = PCA(random_state=42)
    pipeline = Pipeline([("scaler", StandardScaler()), ("pca", pca)])
    X_pca = pipeline.fit_transform(X)
    X_scaled = pipeline.named_steps["scaler"].transform(X)

    explained = pca.explained_variance_ratio_

    figure, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    var_ax = axes[0, 0]
    arrows_ax = axes[0, 1]
    moves_ax = axes[1, 0]
    complexity_ax = axes[1, 1]

    # --- Plot 1: Explained variance per component ---
    n_show = min(20, len(explained))
    tick_step = 2 if n_show > 10 else 1
    var_ax.bar(range(1, n_show + 1), explained[:n_show])
    var_ax.set_xlabel("Component")
    var_ax.set_ylabel("Explained variance ratio")
    var_ax.set_title("Explained Variance per Component")
    var_ax.set_xticks(range(1, n_show + 1, tick_step))
    style_axes(var_ax, grid_axis="y")
    cumulative = np.cumsum(explained[:n_show])
    var_ax_right = var_ax.twinx()
    var_ax_right.plot(range(1, n_show + 1), cumulative, color="red", marker="o", markersize=3, linewidth=1.2)
    var_ax_right.set_ylabel("Cumulative explained variance", color="red")
    var_ax_right.tick_params(axis="y", labelcolor="red")
    var_ax_right.set_ylim(0, 1.05)

    # --- Plot 2: Principal component arrows in original feature space ---
    # Pick the 2 tile features with the highest absolute loadings on PC1
    pc1_loadings = np.abs(pca.components_[0])
    top2 = np.argsort(pc1_loadings)[-2:]
    f1_idx, f2_idx = int(top2[0]), int(top2[1])
    f1_name, f2_name = feature_names[f1_idx], feature_names[f2_idx]

    arrows_ax.scatter(X_scaled[:, f1_idx], X_scaled[:, f2_idx], alpha=0.15, s=3, color="steelblue", rasterized=True)
    colors = ["red", "blue"]
    for i, color in enumerate(colors):
        dx = pca.components_[i, f1_idx] * pca.explained_variance_[i]
        dy = pca.components_[i, f2_idx] * pca.explained_variance_[i]
        arrows_ax.arrow(0, 0, dx, dy, width=0.02, color=color, label=f"PC{i + 1} (var={pca.explained_variance_[i]:.2f})")
    arrows_ax.set_xlabel(f1_name)
    arrows_ax.set_ylabel(f2_name)
    arrows_ax.set_title("Principal Components in Feature Space")
    arrows_ax.legend(frameon=False)
    style_axes(arrows_ax, grid_axis="both")

    # --- Plot 3: 2D scatter colored by move ---
    move_codes, move_labels = pd.factorize(df["MOVE"], sort=True)
    cmap = plt.get_cmap("tab20", max(len(move_labels), 1))
    moves_ax.scatter(X_pca[:, 0], X_pca[:, 1], c=move_codes, cmap=cmap, s=5, alpha=0.25, linewidths=0, rasterized=True)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(i), markeredgecolor="none", markersize=6, label=str(lbl))
        for i, lbl in enumerate(move_labels)
    ]
    legend_cols = 2 if len(move_labels) <= 10 else 3
    moves_ax.legend(handles=legend_handles, title="Move", frameon=False, ncol=legend_cols, loc="best")
    moves_ax.set_title("PCA colored by Move")
    moves_ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
    moves_ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
    style_axes(moves_ax, grid_axis="both")

    # --- Plot 4: 2D scatter colored by complexity ---
    sc = complexity_ax.scatter(X_pca[:, 0], X_pca[:, 1], c=misplaced_stickers.to_numpy(), cmap="viridis", s=5, alpha=0.25, linewidths=0, rasterized=True)
    cb = figure.colorbar(sc, ax=complexity_ax, fraction=0.046, pad=0.04)
    cb.set_label("Misplaced stickers")
    complexity_ax.set_title("PCA colored by Complexity")
    complexity_ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
    complexity_ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
    style_axes(complexity_ax, grid_axis="both")

    figure.suptitle(
        f"CubeML PCA Overview — PC1={explained[0]:.1%}, PC2={explained[1]:.1%}, total={float(explained[:2].sum()):.1%}",
        fontsize=14,
        fontweight="bold",
    )
    return figure
