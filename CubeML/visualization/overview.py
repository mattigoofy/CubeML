from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif


COMPLEXITY_BIN_EDGES = (0, 18, 24, 30, 36, 55)
COMPLEXITY_BIN_LABELS = ("0-17", "18-23", "24-29", "30-35", "36-54")


def encode_features_as_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert feature columns to numeric codes for mutual information and PCA."""
    feature_frame = df.loc[:, df.columns != "MOVE"]
    encoded = pd.DataFrame(index=feature_frame.index)
    for column in feature_frame.columns:
        column_values = feature_frame[column]
        if pd.api.types.is_numeric_dtype(column_values):
            encoded[column] = column_values.astype(np.float32)
        else:
            encoded[column] = pd.Categorical(column_values).codes.astype(np.float32)
    return encoded


def style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, linestyle="--", linewidth=0.6, alpha=0.35)


def compute_move_distribution(df: pd.DataFrame) -> pd.Series:
    return df["MOVE"].value_counts().sort_values(ascending=False)


def compute_misplaced_sticker_counts(df: pd.DataFrame, face_order: tuple[str, ...]) -> pd.Series:
    misplaced = np.zeros(len(df), dtype=np.int16)
    for face in face_order:
        available_tiles = [i for i in range(1, 10) if f"TILE_{face}{i}" in df.columns]
        if not available_tiles:
            continue

        if f"TILE_{face}5" in df.columns:
            reference_values = df[f"TILE_{face}5"]
        else:
            face_tile_cols = [f"TILE_{face}{i}" for i in available_tiles]
            all_values = df[face_tile_cols].values.flatten()
            reference_values = pd.Series(all_values).mode()[0]

        face_columns = [f"TILE_{face}{i}" for i in available_tiles]
        misplaced += df[face_columns].ne(reference_values, axis=0).sum(axis=1).to_numpy(np.int16)

    return pd.Series(misplaced, index=df.index, name="misplaced_stickers")


def compute_tile_mutual_information(df: pd.DataFrame) -> pd.Series:
    feature_frame = encode_features_as_numeric(df)
    target = df["MOVE"]
    scores = mutual_info_classif(
        feature_frame.to_numpy(),
        target,
        discrete_features=True,
        random_state=42,
    )
    return pd.Series(scores, index=feature_frame.columns).sort_values(ascending=False)


def compute_move_share_by_complexity(
    df: pd.DataFrame,
    misplaced_stickers: pd.Series,
) -> pd.DataFrame:
    complexity_bins = pd.cut(
        misplaced_stickers,
        bins=COMPLEXITY_BIN_EDGES,
        labels=COMPLEXITY_BIN_LABELS,
        include_lowest=True,
        right=False,
    )
    table = pd.crosstab(
        np.asarray(complexity_bins.astype(str)),
        df["MOVE"].to_numpy(),
        normalize="index",
    ).fillna(0.0)
    return table.reindex(COMPLEXITY_BIN_LABELS, fill_value=0.0)


def plot_move_distribution(ax: plt.Axes, move_counts: pd.Series) -> None:
    ax.bar(move_counts.index, move_counts.values, color="#4c6a92")
    ax.set_title("Next Move Distribution")
    ax.set_xlabel("Move")
    ax.set_ylabel("Samples")
    style_axes(ax)


def plot_misplaced_histogram(ax: plt.Axes, misplaced_stickers: pd.Series) -> None:
    bins = np.arange(misplaced_stickers.min(), misplaced_stickers.max() + 2) - 0.5
    ax.hist(misplaced_stickers, bins=bins, color="#7f8c8d", edgecolor="white")
    ax.set_title("Scramble Complexity")
    ax.set_xlabel("Misplaced stickers")
    ax.set_ylabel("Samples")
    style_axes(ax)


def plot_top_tile_information(ax: plt.Axes, tile_information: pd.Series) -> None:
    most_informative = tile_information.head(6)
    least_informative = tile_information.sort_values(ascending=True).head(6)

    labels = most_informative.index.tolist() + ["..."] + least_informative.index.tolist()
    values = most_informative.values.tolist() + [0.0] + least_informative.values.tolist()
    colors = ["#5d8f62"] * len(most_informative) + ["#00000000"] + ["#9aa3ad"] * len(least_informative)
    positions = np.arange(len(labels))

    ax.barh(positions, values, color=colors)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title("Most and Least Informative Tile Positions")
    ax.set_xlabel("Mutual information with next move")
    ax.set_ylabel("Tile position")
    style_axes(ax, grid_axis="x")


def plot_move_share_by_complexity(ax: plt.Axes, move_share_by_complexity: pd.DataFrame) -> None:
    ordered_columns = sorted(move_share_by_complexity.columns)
    cumulative = np.zeros(len(move_share_by_complexity), dtype=float)
    colors = ["#4c6a92", "#a85d5d", "#679267", "#a78642", "#74639b", "#7f8c8d"]

    for color_index, move in enumerate(ordered_columns):
        values = move_share_by_complexity[move].to_numpy()
        ax.bar(
            move_share_by_complexity.index.astype(str),
            values,
            bottom=cumulative,
            label=move,
            color=colors[color_index % len(colors)],
        )
        cumulative += values

    ax.set_title("Move Share by Complexity Bin")
    ax.set_xlabel("Misplaced sticker range")
    ax.set_ylabel("Share of samples")
    ax.set_ylim(0.0, 1.0)
    style_axes(ax)
    ax.legend(title="Move", frameon=False, ncol=3, loc="upper center")


def create_overview_figure(
    move_counts: pd.Series,
    misplaced_stickers: pd.Series,
    tile_information: pd.Series,
    move_share_by_complexity: pd.DataFrame,
) -> plt.Figure:
    figure, axes = plt.subplot_mosaic(
        [["moves", "misplaced_hist"], ["tile_info", "complexity_mix"]],
        figsize=(14, 10),
        constrained_layout=True,
    )

    plot_move_distribution(axes["moves"], move_counts)
    plot_misplaced_histogram(axes["misplaced_hist"], misplaced_stickers)
    plot_top_tile_information(axes["tile_info"], tile_information)
    plot_move_share_by_complexity(axes["complexity_mix"], move_share_by_complexity)

    figure.suptitle("CubeML Dataset Overview", fontsize=16, fontweight="bold")
    return figure
