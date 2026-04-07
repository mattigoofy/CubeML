import pandas as pd

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
