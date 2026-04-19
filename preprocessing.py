import pandas as pd
import os
from tqdm import tqdm
from utils.cube import execute_move_str
from utils.states import cast_state

from multiprocessing import Pool, Manager
from functools import partial


def preprocess(filepath_in: list, filepath_out: str):
    """
    Preprocess the whole cfop-dataset folder.
    """

    data: set[str] = set()

    for fp in filepath_in:
        lines: list[str] = list()
        with open(fp, "r") as file:
            lines = file.read().strip().splitlines()
            # lines = lines[:100]

        for i in tqdm(range(0, len(lines) - 1, 2)):
            cube_state = lines[i]
            solution = lines[i + 1]
            # print(cube_state)

            if tuple(cube_state) in data:
                continue

            # 6 sides, 9 facelets
            if len(cube_state.split(" ")) != 6 * 9:
                continue

            # No final states
            if solution == "#":
                continue

            if len(solution) > 1:
                # U2 -> U + U
                if solution[1] == "2":
                    single_move = solution[0]
                    next_state = cube_state
                    for i in range(2):
                        if tuple(next_state) in data:
                            continue

                        # data.add({"state": cast_state(next_state), "solution": single_move})
                        data.add((next_state, single_move))
                        next_state = execute_move_str(single_move, next_state)

                # U' -> U + U + U
                elif solution[1] == "'":
                    single_move = solution[0]
                    next_state = cube_state
                    for i in range(3):
                        if tuple(next_state) in data:
                            continue

                        # data.add({"state": cast_state(next_state), "solution": single_move})
                        data.add((next_state, single_move))
                        next_state = execute_move_str(single_move, next_state)

            else:
                # data.add({"state": cast_state(cube_state), "solution": solution})
                data.add((cube_state, solution))

    df = pd.DataFrame()
    for state, solution in tqdm(data):
        subframe = cast_state(state.split(" "))
        subframe["MOVE"] = solution
        df = pd.concat([df, subframe])

    df.to_pickle(filepath_out)


def process_file(args: tuple) -> pd.DataFrame:
    """Process a single file and return records."""
    fp, position = args

    # Strategy B: Use set with strings for faster lookups
    # Avoid repeated tuple() conversions which are expensive
    seen_states = set()
    collected_data = []

    with open(fp, "r") as file:
        lines = file.read().strip().splitlines()
        # lines = lines[:500]

    for i in tqdm(range(0, len(lines) - 1, 2), desc=fp, position=position, leave=True):
        cube_state = lines[i]
        solution = lines[i + 1]

        # Strategy B: Store state as string directly for faster hashing
        state_key = cube_state
        if state_key in seen_states:
            continue

        # 6 sides, 9 facelets
        if len(cube_state.split(" ")) != 6 * 9:
            continue

        # No final states
        if solution == "#":
            continue

        if len(solution) > 1:
            # U2 -> U + U
            if solution[1] == "2":
                single_move = solution[0]
                next_state = cube_state
                for j in range(2):
                    next_state_key = next_state
                    if next_state_key in seen_states:
                        next_state = execute_move_str(single_move, next_state)
                        continue

                    seen_states.add(next_state_key)
                    collected_data.append((next_state, single_move))
                    next_state = execute_move_str(single_move, next_state)

            # U' -> U + U + U
            elif solution[1] == "'":
                # seen_states.add(state_key)
                # collected_data.append((cube_state, solution))

                single_move = solution[0]
                next_state = cube_state
                for j in range(3):
                    next_state_key = next_state
                    if next_state_key in seen_states:
                        next_state = execute_move_str(single_move, next_state)
                        continue

                    seen_states.add(next_state_key)
                    collected_data.append((next_state, single_move))
                    next_state = execute_move_str(single_move, next_state)

        else:
            seen_states.add(state_key)
            collected_data.append((cube_state, solution))

    # Strategy A: Build DataFrame all at once instead of concatenating in a loop
    # This is ~100-1000x faster for large datasets
    all_rows = []
    all_moves = []

    for state_str, solution in tqdm(
        collected_data, desc=f"Building DF from {fp}", position=position, leave=True
    ):
        state_list = state_str.split(" ")

        # Convert state using existing cast_state function
        state_dict = cast_state(state_list)
        all_rows.append(state_dict)
        all_moves.append(solution)

    # Concatenate all rows at once, then add MOVE column
    if all_rows:
        df = pd.concat(all_rows, ignore_index=True)
        df["MOVE"] = all_moves
    else:
        df = pd.DataFrame()

    return df


def preprocess_faster(filepath_in: list, filepath_out: str):
    """
    Preprocess the whole cfop-dataset folder, in parallel per file.
    """

    args = [(fp, i) for i, fp in enumerate(filepath_in)]
    with Pool(processes=2) as pool:
        # Each file is processed in a separate process
        all_records: list[pd.DataFrame] = pool.map(process_file, args)

    # Flatten, deduplicate across files, and build DataFrame
    df = pd.DataFrame()
    for i in all_records:
        df = pd.concat([df, i])

    df.to_pickle(filepath_out)


if __name__ == "__main__":
    path_in: list = [
        "./cfop-dataset/training.seq.0",
        "./cfop-dataset/training.seq.1",
        "./cfop-dataset/training.seq.2",
        "./cfop-dataset/training.seq.3",
        "./cfop-dataset/training.seq.4",
        "./cfop-dataset/training.seq.5",
        "./cfop-dataset/training.seq.6",
        "./cfop-dataset/training.seq.7",
        "./cfop-dataset/training.seq.8",
        "./cfop-dataset/training.seq.9",
        "./cfop-dataset/training.seq.99"
    ]
    path_out_dir: str = "./cfop-dataset-processed"
    path_out_file: str = "dataset_test.pkl"
    path_out: str = os.path.join(path_out_dir, path_out_file)
    for p in path_in:
        if not os.path.isfile(p):
            raise Exception(f"Path {p} is not a directory.")

    if not os.path.isdir(path_out_dir):
        os.mkdir(path_out_dir)

    preprocess_faster(path_in, path_out)
