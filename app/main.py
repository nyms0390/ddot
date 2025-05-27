import numpy as np

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List

from odeformer.odebench.strogatz_equations import equations
from odeformer.odebench.solve_and_plot import process_equations, solve_equations
from scripts.inference import create_sample

from . import solver, plot
from .model import evaluator, predict  # Import predict
app = FastAPI()

# %%
class ODERequest(BaseModel):
    system_id: int= 54
    method: str = "LSODA"
    y0: List[float] = [1.0, 1.0, 1.0]
    t_start: float = 0.0
    t_end: float = 10.0
    steps: int = 150
    step_size: float = 1e-6

@app.post("/simulate")
def simulate(req: ODERequest):
    # Find the equation by id
    eq = next((e for e in equations if e['id'] == req.system_id), None)
    if eq is None:
        return {"error": "ODE system not found."}

    # Prepare config for solve_equations
    config = {
        "t_span": (req.t_start, req.t_end),
        "method": req.method,
        "rtol": 1e-5,
        "atol": 1e-7,
        "first_step": req.step_size,
        "t_eval": np.linspace(req.t_start, req.t_end, req.steps),
        "min_step": 1e-10,
    }

    # Process and solve the selected equation
    process_equations([eq])
    solve_equations([eq], config)

    # Create a sample from the solved equation
    sample = create_sample(evaluator, eq)

    # Get the sequence (trajectory) from the sample
    sequence = sample["train"]["trajectories"][0]
    image = plot.plot_sequence(sequence)

    # Get the predicted tree from the model
    pred_tree = predict(sample)

    return {
        "sequence": sequence.tolist(),
        "plot_base64": image,
        "predicted_tree": str(pred_tree)  # or pred_tree.infix() if available
    }

