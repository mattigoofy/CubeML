import os
import json
import pandas as pd
import pycuber as pc
from pycuber.solver import CFOPSolver
from IPython.display import display, HTML, SVG
import uuid
import time
from states import cast_state, uncast_state

def visualize_scramble(state: pd.DataFrame):
    state_uncasted = uncast_state(state)
    cube = pc.Cube(pc.array_to_cubies(state_uncasted))    # create a rubiks cube
    py_solver = CFOPSolver(cube)
    solve_alg = py_solver.solve(suppress_progress_messages=True)                # solve the state

    # Inverse the algorithm to show the original state
    inverse_order_alg = solve_alg[::-1]
    inverse_move_alg = [str(i.inverse()) for i in inverse_order_alg]
    str_alg = " ".join(inverse_move_alg)

    # get to correct starting state
    str_alg = "Y2 " +  str_alg

    cube_id = f"cube-{uuid.uuid4().hex}"  # unique ID per call

    display(HTML(f"""
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js"></script>
        <script src="https://molarmanful.github.io/gCube/gcube.min.js"></script>
        <div id="{cube_id}-wrapper" style="width:200px; height:300px; overflow:hidden;">
          <g-cube id="{cube_id}"></g-cube>
        </div>
        
        <script>
            setTimeout(function() {{
                $("#{cube_id}").gscramble("{str_alg}");
            }}, 500);
        </script>
    """))

    time.sleep(0.5)

def execute_move(move: str, state: pd.DataFrame) -> pd.DataFrame:
    state_uncasted = uncast_state(state)
    cube = pc.Cube(pc.array_to_cubies(state_uncasted))    
    alg = pc.Formula(move)
    cube(alg)

    color_map = {"[r]": "0", "[y]": "1", "[g]": "2", "[w]": "3", "[o]": "4", "[b]": "5"}
    faces = ["L", "U", "F", "D", "R", "B"] 
    result = []
    for face in faces:
        face_array = cube.get_face(face)
        for row in face_array:
            for cubie in row:
                result.append(color_map[str(cubie).lower()])

    result_str = " ".join(result)
    return cast_state(result_str)
    
def execute_move_list(move: str, state: list) -> list:
    cube = pc.Cube(pc.array_to_cubies(state))    
    alg = pc.Formula(move)
    cube(alg)

    color_map = {"[r]": "0", "[y]": "1", "[g]": "2", "[w]": "3", "[o]": "4", "[b]": "5"}
    faces = ["L", "U", "F", "D", "R", "B"] 
    result = []
    for face in faces:
        face_array = cube.get_face(face)
        for row in face_array:
            for cubie in row:
                result.append(color_map[str(cubie).lower()])

    return result

def execute_move_str(move: str, state: str) -> str:
    state_list = state.split(" ")
    cube = pc.Cube(pc.array_to_cubies(state_list))    
    alg = pc.Formula(move)
    cube(alg)

    color_map = {"[r]": "0", "[y]": "1", "[g]": "2", "[w]": "3", "[o]": "4", "[b]": "5"}
    faces = ["L", "U", "F", "D", "R", "B"] 
    result = []
    for face in faces:
        face_array = cube.get_face(face)
        for row in face_array:
            for cubie in row:
                result.append(color_map[str(cubie).lower()])

    return " ".join(result)
