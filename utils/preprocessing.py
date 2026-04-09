import pandas as pd
import os

def cast_state(state: str) -> pd.DataFrame:
    """
    Cast a state from the dataset (e.g. `3 0 2 2 0 2 4 ...` to usable classes (e.g. TILE_L1, TILE_L2, ..., TILE_L9, TILE_U1, TILE_U2, ...)
    """
    state_split = state.split(" ")
    faces = ["L", "U", "F", "D", "R", "B"]
    d: dict[str, int] = {}
    for i, state in enumerate(state_split):
        face = faces[(i // 9) % 9]
        tile_idx = (i % 9) + 1
        d.update({ f"TILE_{face}{tile_idx}": [int(state_split[i])] })
        
    return pd.DataFrame(data=d)


def preprocess(filepath_in: str, filepath_out: str):
    """
    Preprocess the whole cfop-dataset folder.
    """

    data: set[str] = {}

    with open(filepath_in, "r") as file:
        lines = file.read().strip().splitlines()

        for i in range(0, len(lines)-1, 2):

                        
            cube_state = lines[i].split()
            solution = lines[i+1]

            if cube_state in data:
                continue
    
            # 6 sides, 9 facelets
            if len(cube_state) != 6*9:
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
                        data.append({"state": cast_state(next_state), "solution": single_move})
                        next_state = execute_move_list(single_move, next_state)
    
                # U' -> U + U + U
                elif solution[1] == "'":
                    single_move = solution[0]
                    next_state = cube_state
                    for i in range(3):
                        data.append({"state": cast_state(next_state), "solution": single_move})
                        next_state = execute_move_list(single_move, next_state)

        
        
                    
            data.add({"state": cast_state(cube_state), "solution": solution})

    df = pd.DataFrame(data=data)
    df.to_pickle(filepath_out)

if __name__ == "__main__":
    path_in: str = "../cfop-dataset/"
    path_out: str = "../cfop-dataset-processed/"
    if not os.path.isdir(path_in):
        raise Exception(f"Path {path_in} is not a directory.")
    
    if not os.path.isdir(path_out):
        os.mkdir(path_out)
        
    preprocess(path_in, path_out)
