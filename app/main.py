import numpy as np
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List
import json

from odeformer.odebench.strogatz_equations import equations
from odeformer.odebench.solve_and_plot import process_equations, solve_equations
from scripts.inference import create_sample

from . import solver, plot
from .model import evaluator, predict

app = FastAPI()

class ODERequest(BaseModel):
    system_id: int = 54
    method: str = "LSODA"
    y0: List[float] = [1.0, 1.0, 1.0]
    t_start: float = 0.0
    t_end: float = 10.0
    steps: int = 50
    step_size: float = 1e-6

@app.post("/simulate")
def simulate(req: ODERequest):
    eq = next((e for e in equations if e['id'] == req.system_id), None)
    if eq is None:
        return {"error": "ODE system not found."}
    config = {
        "t_span": (req.t_start, req.t_end),
        "method": req.method,
        "rtol": 1e-5,
        "atol": 1e-7,
        "first_step": req.step_size,
        "t_eval": np.linspace(req.t_start, req.t_end, req.steps),
        "min_step": 1e-10,
    }
    process_equations([eq])
    solve_equations([eq], config)
    sample = create_sample(evaluator, eq)
    sequence = sample["train"]["trajectories"][0]
    image = plot.plot_sequence(sequence)

    app.state.last_sample = sample
    return {
        "sequence": sequence.tolist(),
        "plot_base64": image,
    }

@app.post("/predict")
def predict_ode():
    sample = getattr(app.state, "last_sample", None)
    if sample is None:
        return {"error": "No sample available. Please run /simulate first."}
    latex_str = predict(sample)
    return {"predicted_tree": latex_str}

@app.get("/equations")
def get_equations():
    eq_info = [
        {
            "id": eq["id"],
            "eq": eq["eq"],
            "dim": eq["dim"],
            "consts": eq["consts"][0],
            "desc": eq["eq_description"]
        }
        for eq in equations
    ]
    return eq_info

@app.get("/equation/{eq_id}")
def get_equation(eq_id: int):
    eq = next((e for e in equations if e['id'] == eq_id), None)
    if eq is None:
        return {"error": "Equation not found"}
    return {
        "id": eq["id"],
        "eq": eq["eq"],
        "dim": eq["dim"],
        "consts": eq["consts"][0],
        "desc": eq["eq_description"]
    }

@app.get("/", response_class=HTMLResponse)
def form_page():
    return f"""
    <html>
    <head>
        <title>ODE Simulation</title>
        <script id="MathJax-script" async 
                src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    </head>
    <body>
        <h2>ODE Simulation</h2>
        <form id="simulation_form">
            <label>System ID: <input id="system_id" name="system_id" type="number" onchange="updateEquationInfo()"></label><br>
            <label>
                Initial Conditions (comma-separated):
                <input id="y0" name="y0" type="text" placeholder="e.g. 1.0,1.0,1.0">
                <span id="y0_hint" style="font-size:smaller;color:gray;"></span>
            </label><br>
            <span id="y0_hint" style="font-size:smaller;color:gray;"></span><br>
            <label>t_start: <input name="t_start" type="number" value="0"></label><br>
            <label>t_end: <input name="t_end" type="number" value="10"></label><br>
            <label>steps: <input name="steps" type="number" value="50"></label><br>
            <input type="submit" value="Simulate">
        </form>
        <button type="button" onclick="predictODE()">Predict ODE from Last Simulation</button>
        <hr>
        <div id="ode_info" style="display: flex; gap: 40px;">
            <div>
                <b>Original Equation:</b><br>
                <span id="eq_str"></span>
                <br><br>
                <b>Description:</b><br>
                <span id="eq_desc"></span>
            </div>
            <div>
                <b>Predicted ODE:</b><br>
                <span id="pred_eq"></span>
            </div>
        </div>
        <div id="simulation_result"></div>
        <script>
        let eqInfo = [];
        function updateEquationInfo() {{
            const id = parseInt(document.getElementById('system_id').value);
            fetch('/equation/' + id)
                .then(response => response.json())
                .then(found => {{
                    if (found && !found.error) {{
                        let eqStr = '';
                        (found.eq.split('|')).forEach((rawLine, i) => {{
                            let eqLine = rawLine;
                            (found.consts || []).forEach((c, j) => {{
                                const regex = new RegExp(`c_${{j}}`, 'g');
                                eqLine = eqLine.replace(regex, String(c));
                            }});
                            eqStr += `\\\\[ \\\\frac{{dx_${{i}}}}{{dt}} = ${{eqLine}} \\\\]`;
                        }});
                        document.getElementById('eq_str').innerHTML = eqStr;
                        document.getElementById('eq_desc').innerText = found.desc;
                        document.getElementById('y0').placeholder = 'e.g. ' + Array(found.dim).fill(1.0).join(',');
                        document.getElementById('y0_hint').innerText = 'Enter ' + found.dim + ' values';
                        MathJax.typeset();
                    }} else {{
                        document.getElementById('eq_str').innerText = '';
                        document.getElementById('eq_desc').innerText = '';
                        document.getElementById('y0_hint').innerText = '';
                    }}
                }});
        }}

        document.getElementById('simulation_form').onsubmit = function(e) {{
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = {{
                system_id: parseInt(formData.get('system_id')),
                y0: formData.get('y0').split(',').map(Number),
                t_start: parseFloat(formData.get('t_start')),
                t_end: parseFloat(formData.get('t_end')),
                steps: parseInt(formData.get('steps')),
                method: "LSODA",
                step_size: 1e-6
            }};
            fetch('/simulate', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify(data)
            }})
            .then(response => response.json())
            .then(result => {{
                if (result.error) {{
                    document.getElementById('simulation_result').innerHTML = `<span style="color:red">${{result.error}}</span>`;
                }} else {{
                    document.getElementById('simulation_result').innerHTML = `
                        <h3>Simulation Result</h3>
                        <img src="data:image/png;base64,${{result.plot_base64}}"/><br>
                        <!-- Optional: <button onclick="downloadSequence()">Download Data</button> -->
                    `;
                }}
            }});
        }};

        function predictODE() {{
            fetch('/predict', {{method: 'POST'}})
                .then(response => response.json())
                .then(result => {{
                    if (result.predicted_tree) {{
                        let equationsHtml = '';
                        result.predicted_tree.forEach((latexStr, i) => {{
                            equationsHtml += `<div id="pred_eq_${{i}}">\\\\[ \\\\frac{{dx_${{i}}}}{{dt}} = ${{latexStr}} \\\\]</div>`;
                        }});
                        document.getElementById('pred_eq').innerHTML = `
                            ${{equationsHtml}}
                        `;
                        
                        console.log('HTML updated. Attempting MathJax typesetting...');
                        if (window.MathJax) {{
                            MathJax.typesetPromise(['#pred_eq'])
                                .then(() => {{
                                    console.log('MathJax typesetting successful!');
                                }})
                                .catch((err) => {{
                                    console.error('MathJax typesetting error:', err);
                                }});
                        }} else {{
                            console.warn('MathJax is not loaded or available.');
                        }}
                    }} else if (result.error) {{
                        document.getElementById('pred_eq').innerHTML = `<span style="color:red">${{result.error}}</span>`;
                    }}
                }});
        }}
        // Initialize on page load
        window.onload = updateEquationInfo;
        </script>
    </body>
    </html>
    """

@app.post("/simulate/html", response_class=HTMLResponse)
async def simulate_html(request: Request):
    form = await request.form()
    system_id = int(form.get("system_id", 54))
    y0 = form.get("y0", "1.0,1.0,1.0")
    t_start = form.get("t_start", "0")
    t_end = form.get("t_end", "10")
    steps = form.get("steps", "150")
    y0_list = [float(x) for x in y0.split(",")]
    req = ODERequest(system_id=system_id, y0=y0_list, t_start=float(t_start), t_end=float(t_end), steps=int(steps))
    result = simulate(req)
    return f"""
    <html>
    <body>
        <h2>Simulation Result</h2>
        <img src="data:image/png;base64,{result['plot_base64']}"/><br>
        <pre>{result['sequence']}</pre>
        <form action="/" method="get">
            <input type="hidden" name="system_id" value="{system_id}">
            <input type="hidden" name="y0" value="{y0}">
            <input type="hidden" name="t_start" value="{t_start}">
            <input type="hidden" name="t_end" value="{t_end}">
            <input type="hidden" name="steps" value="{steps}">
            <button type="submit">Back</button>
        </form>
    </body>
    </html>
    """

@app.post("/predict/html", response_class=HTMLResponse)
async def predict_html():
    result = predict_ode()
    return f"""
    <html>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async
            src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
    </script>
    <body>
        <h2>Predicted ODE</h2>
        <div id="pred_eq">
            \\[
            {result['predicted_tree']}
            \\]
        </div>
        <a href="/">Back</a>
        <script>
            MathJax.typeset();
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app='app.main:app', host="0.0.0.0", port=8000, reload=True, debug=True)