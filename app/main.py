import numpy as np

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List

from odeformer.odebench.strogatz_equations import equations
from odeformer.odebench.solve_and_plot import process_equations, solve_equations
from scripts.inference import create_sample

from . import solver, plot
from .model import evaluator
app = FastAPI()

# %%
class ODERequest(BaseModel):
    system_id: int
    method: str = "LSODA"
    y0: List[float]
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

    return {
        "sequence": sequence.tolist(),
        "plot_base64": image
    }


@app.post("/simulate/html", response_class=HTMLResponse)
def simulate_html(
    system: str = Form(...),
    y0: str = Form(...),
    t_start: float = Form(0.0),
    t_end: float = Form(25.0),
    steps: int = Form(1000)
):
    y0_list = [float(val) for val in y0.split(",")]  # Parse y0 from comma-separated string
    t_eval = [t_start + i * (t_end - t_start) / steps for i in range(steps)]
    t, sequence = solver.solve_ode(system, y0_list, (t_start, t_end), t_eval)
    image = plot.plot_sequence(sequence)

    html_content = f"""
    <html>
        <body>
            <h1>ODE Simulation Result</h1>
            <p>System: {system}</p>
            <img src="data:image/png;base64,{image}" alt="ODE Plot"/>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)