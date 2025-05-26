from scipy.integrate import solve_ivp
import numpy as np
from . import odes

ODE_MAP = {
    "lorenz": odes.lorenz,
    "rossler": odes.rossler,
}

def solve_ode(system_name, y0, t_span=(0, 25), t_eval=None):
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
    ode_func = ODE_MAP[system_name]
    result = solve_ivp(ode_func, t_span, y0, t_eval=t_eval)
    return t_eval, result.y.T  # shape: [time][features]
