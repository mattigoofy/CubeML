import pandas as pd
import os
from tqdm import tqdm
from cube import execute_move_str
from states import cast_state

def preprocess(filepath_in: str, filepath_out: str):
    """
    Preprocess the whole cfop-dataset folder.
    """

    data: set[str] = set()

    lines: list[str] = list()
    with open(filepath_in, "r") as file:
        lines = file.read().strip().splitlines()
        # lines = lines[:100]

    for i in tqdm(range(0, len(lines)-1, 2)):
        cube_state = lines[i]
        solution = lines[i+1]
        # print(cube_state)

        if tuple(cube_state) in data:
            continue

        # 6 sides, 9 facelets
        if len(cube_state.split(" ")) != 6*9:
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
    for (state, solution) in tqdm(data):
        subframe = cast_state(state.split(" "))
        subframe["MOVE"] = solution
        df = pd.concat([df, subframe])

    df.to_pickle(filepath_out)

if __name__ == "__main__":
    path_in: str = "./cfop-dataset/training.seq.0"
    path_out_dir: str = "./cfop-dataset-processed"
    path_out_file: str = "dataset.pkl"
    path_out: str = os.path.join(path_out_dir, path_out_file)
    if not os.path.isfile(path_in):
        raise Exception(f"Path {path_in} is not a directory.")
    
    if not os.path.isdir(path_out_dir):
        os.mkdir(path_out_dir)
        
    preprocess(path_in, path_out)
