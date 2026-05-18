from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from CubeML.utils.model import load_dataset
from CubeML.visualization.overview import (
    compute_move_distribution,
    compute_misplaced_sticker_counts,
    compute_tile_mutual_information,
    compute_move_share_by_complexity,
    create_overview_figure,
)
from CubeML.visualization.pca import create_pca_figure
from CubeML.visualization.tsne import create_tsne_figure


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = ROOT_DIR / "datasets" / "cfop-dataset-processed" / "dataset_with_prime.pkl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_FACE_ORDER = ("L", "U", "F", "D", "R", "B")


def extract_faces_from_dataset(df: pd.DataFrame, preferred_order: tuple[str, ...] = DEFAULT_FACE_ORDER) -> tuple[str, ...]:
    tile_columns = [col for col in df.columns if col.startswith("TILE_")]
    faces = {col[5:-1] for col in tile_columns}
    ordered = [f for f in preferred_order if f in faces] + sorted(faces - set(preferred_order))
    return tuple(ordered)


def save_figure(output_dir: Path, figure: plt.Figure, filename_prefix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{filename_prefix}_{timestamp}.png"
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
    data_df, moves_df = load_dataset(str(args.dataset))
    df = data_df.assign(MOVE=moves_df)

    # Extract face order dynamically from dataset
    face_order = extract_faces_from_dataset(df, preferred_order=DEFAULT_FACE_ORDER)

    move_counts = compute_move_distribution(df)
    misplaced_stickers = compute_misplaced_sticker_counts(df, face_order=face_order)
    tile_information = compute_tile_mutual_information(df)
    move_share_by_complexity = compute_move_share_by_complexity(df, misplaced_stickers)
    overview_figure = create_overview_figure(
        move_counts=move_counts,
        misplaced_stickers=misplaced_stickers,
        tile_information=tile_information,
        move_share_by_complexity=move_share_by_complexity,
    )
    output_path = save_figure(args.output_dir, overview_figure, "dataset_overview")

    # pca_figure = create_pca_figure(df, misplaced_stickers)
    # pca_output_path = save_figure(args.output_dir, pca_figure, "dataset_pca")

    # tsne_figure = create_tsne_figure(df, misplaced_stickers)
    # tsne_output_path = save_figure(args.output_dir, tsne_figure, "dataset_tsne")

    print_summary(df, move_counts, misplaced_stickers, tile_information, output_path)
    # print(f"Saved PCA visualization to {pca_output_path}")
    # print(f"Saved t-SNE visualization to {tsne_output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(overview_figure)
        # plt.close(pca_figure)
        # plt.close(tsne_figure)


if __name__ == "__main__":
    main()