from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = ROOT_DIR / "cfop-dataset-processed" / "dataset.pkl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
FACE_ORDER = ("L", "U", "F", "D", "R", "B")
COMPLEXITY_BIN_EDGES = (0, 18, 24, 30, 36, 55)
COMPLEXITY_BIN_LABELS = ("0-17", "18-23", "24-29", "30-35", "36-54")


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_pickle(dataset_path)
    required_columns = {
        f"TILE_{face}{tile_index}"
        for face in FACE_ORDER
        for tile_index in range(1, 10)
    }
    required_columns.add("MOVE")

    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        missing_preview = ", ".join(missing_columns[:5])
        raise ValueError(f"Dataset is missing expected columns: {missing_preview}")

    return df


def compute_move_distribution(df: pd.DataFrame) -> pd.Series:
    return df["MOVE"].value_counts().sort_values(ascending=False)


def compute_misplaced_sticker_counts(df: pd.DataFrame) -> pd.Series:
    misplaced = np.zeros(len(df), dtype=np.int16)
    for face in FACE_ORDER:
        face_columns = [f"TILE_{face}{tile_index}" for tile_index in range(1, 10)]
        center_values = df[f"TILE_{face}5"]
        misplaced += df[face_columns].ne(center_values, axis=0).sum(axis=1).to_numpy(np.int16)

    return pd.Series(misplaced, index=df.index, name="misplaced_stickers")


def compute_tile_mutual_information(df: pd.DataFrame) -> pd.Series:
    feature_frame = df.loc[:, df.columns != "MOVE"]
    target = df["MOVE"]
    scores = mutual_info_classif(
        feature_frame,
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


def _style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, linestyle="--", linewidth=0.6, alpha=0.35)


def plot_move_distribution(ax: plt.Axes, move_counts: pd.Series) -> None:
    ax.bar(move_counts.index, move_counts.values, color="#4c6a92")
    ax.set_title("Next Move Distribution")
    ax.set_xlabel("Move")
    ax.set_ylabel("Samples")
    _style_axes(ax)


def plot_misplaced_histogram(ax: plt.Axes, misplaced_stickers: pd.Series) -> None:
    bins = np.arange(misplaced_stickers.min(), misplaced_stickers.max() + 2) - 0.5
    ax.hist(misplaced_stickers, bins=bins, color="#7f8c8d", edgecolor="white")
    ax.set_title("Scramble Complexity")
    ax.set_xlabel("Misplaced stickers")
    ax.set_ylabel("Samples")
    _style_axes(ax)


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
    _style_axes(ax, grid_axis="x")


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
    _style_axes(ax)
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


def save_overview_plot(output_dir: Path, figure: plt.Figure) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dataset_overview.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    return output_path


def print_summary(
    df: pd.DataFrame,
    move_counts: pd.Series,
    misplaced_stickers: pd.Series,
    tile_information: pd.Series,
    output_path: Path,
) -> None:
    print(f"Loaded {len(df):,} samples from the processed dataset.")
    print(f"Saved visualization to {output_path}")
    print(
        "Majority-class baseline accuracy: "
        f"{move_counts.iloc[0] / len(df):.2%} by always predicting {move_counts.index[0]}"
    )
    print("Move distribution:")
    for move, count in move_counts.items():
        print(f"  {move}: {count:,}")

    print(
        "Misplaced sticker summary: "
        f"min={misplaced_stickers.min()}, "
        f"median={int(misplaced_stickers.median())}, "
        f"mean={misplaced_stickers.mean():.2f}, "
        f"max={misplaced_stickers.max()}"
    )
    print("Most informative tile positions:")
    for tile_name, score in tile_information.head(5).items():
        print(f"  {tile_name}: {score:.3f}")
    print("Least informative tile positions:")
    for tile_name, score in tile_information.sort_values(ascending=True).head(5).items():
        print(f"  {tile_name}: {score:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate simple visual summaries for the processed Rubik's cube dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the processed dataset pickle file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the PNG output will be written.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving it.",
    )
    return parser.parse_args()


def main() -> None:
    plt.style.use("default")
    args = parse_args()
    df = load_dataset(args.dataset)

    move_counts = compute_move_distribution(df)
    misplaced_stickers = compute_misplaced_sticker_counts(df)
    tile_information = compute_tile_mutual_information(df)
    move_share_by_complexity = compute_move_share_by_complexity(df, misplaced_stickers)

    figure = create_overview_figure(
        move_counts=move_counts,
        misplaced_stickers=misplaced_stickers,
        tile_information=tile_information,
        move_share_by_complexity=move_share_by_complexity,
    )
    output_path = save_overview_plot(args.output_dir, figure)
    print_summary(df, move_counts, misplaced_stickers, tile_information, output_path)

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()