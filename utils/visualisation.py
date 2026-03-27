import os
import json
import pycuber as pc
from pycuber.solver import CFOPSolver
from IPython.display import display, HTML, SVG
import requests
import uuid
import time

def visualize_scramble(state: list):
    cube = pc.Cube(pc.array_to_cubies(state))    # create a rubiks cube
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