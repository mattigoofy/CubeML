import pandas as pd

def uncast_state(df: pd.DataFrame) -> list:
    """
    Uncast a state from a DataFrame (e.g. TILE_L1, TILE_L2, ...) back to the original string format (e.g. `3 0 2 2 0 2 4 ...`)
    """
    faces = ["L", "U", "F", "D", "R", "B"]
    values = []
    for face in faces:
        for tile_idx in range(1, 10):
            col = f"TILE_{face}{tile_idx}"
            values.append(str(df[col].iloc[0]))

    return values

def cast_state(state: list) -> pd.DataFrame:
    """
    Cast a state from the dataset (e.g. `3 0 2 2 0 2 4 ...` to usable classes (e.g. TILE_L1, TILE_L2, ..., TILE_L9, TILE_U1, TILE_U2, ...)
    """
    state_split = state
    faces = ["L", "U", "F", "D", "R", "B"]
    d: dict[str, int] = {}
    for i, state in enumerate(state_split):
        face = faces[(i // 9) % 9]
        tile_idx = (i % 9) + 1
        d.update({ f"TILE_{face}{tile_idx}": [int(state_split[i])] })
        
    return pd.DataFrame(data=d)


