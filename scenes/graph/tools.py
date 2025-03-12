from io import StringIO
from langchain_core.tools import tool 
from typing import Tuple,Annotated
from langgraph.prebuilt import InjectedState
import sys
import os
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import pandas as pd
import sklearn

persistent_vars = {}
plotly_saving_code = """import pickle
import uuid
import plotly

for figure in plotly_figures:
    pickle_filename = f"images/plotly_figures/pickle/{uuid.uuid4()}.pkl"
    with open(pickle_filename, 'wb') as f:
        pickle.dump(figure, f)
"""
@tool(parse_docstring=True)
def python_code_executor(graph_state:Annotated[dict,InjectedState],thought:str, python_code:str) -> Tuple[str,dict]:
    """tool to execute the python code

    Args:
        thought (str): Internal thought about the next action to be taken, and the reasoning behind it.This should be formatted in MARKDOWN and be of high quality.
        python_code (str):python code given for execution

    Returns:
        Tuple[str,dict]: the output of the python code and updated graph state.
    """
    current_variables = graph_state["current_variables"] if "current_variables" in graph_state else {}
    for input_dataset in graph_state["input_data"]:
        if input_dataset.variable_name not in current_variables:
            current_variables[input_dataset.variable_name] = pd.read_csv(input_dataset.data_path)
    
    if not os.path.exists("images/plotly_figures/pickle"):
        os.makedirs("images/plotly_figures/pickle")
    
    current_image_pickle_files = os.listdir("images/plotly_figures/pickle")
    try:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        exec_globals = globals().copy()
        exec_globals.update(persistent_vars)
        exec_globals.update(current_variables)
        exec_globals.update({"plotly_figures": []})
        
        exec(python_code,exec_globals)
        persistent_vars.update({k: v for k, v in exec_globals.items() if k not in globals()})
        
        output = sys.stdout.getvalue()
        
        sys.out = old_stdout
        
        updated_state = {
            "intermediate_outputs":[{"thought": thought, "code": python_code, "output": output}],
            "current_variables": persistent_vars
        }
        
        if 'plotly_figures' in exec_globals:
            exec(plotly_saving_code, exec_globals)
            # Check if any images were created
            new_image_folder_contents = os.listdir("images/plotly_figures/pickle")
            new_image_files = [file for file in new_image_folder_contents if file not in current_image_pickle_files]
            if new_image_files:
                updated_state["output_image_paths"] = new_image_files
            
            persistent_vars["plotly_figures"] = []
        
        return output,updated_state
    except Exception as e:
        return str(e),{"intermediate_outputs":[{"thought":thought,"code":python_code,"output":str(e)}]}