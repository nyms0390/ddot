# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def ode1(state, t):
    x, y = state
    dxdt = -y
    dydt = x
    return [dxdt, dydt]

def ode2(state, t):
    x, y = state
    dxdt = -y + 0.1*x
    dydt = x + 0.1*y
    return [dxdt, dydt]

# %%
t = np.linspace(0, 10, 200)

initial_state = [1, 0]
traj1 = odeint(ode1, initial_state, t)
traj2 = odeint(ode2, initial_state, t)

x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

u1 = -Y
v1 = X

plt.rcParams.update({'font.size': 16})
fig, axes = plt.subplots(1, 3, figsize=(25, 8), sharex=True, sharey=True)

# Reconstruction
axes[0].quiver(X, Y, u1, v1, color='gray', alpha=0.5)
axes[0].plot(traj1[:, 0], traj1[:, 1], 'r-', label='Trajectory of ground truth ODE')
axes[0].plot(traj2[:, 0], traj2[:, 1], 'b--', label='Trajectory of predicted ODE')
axes[0].plot(1, 0, 'go', label='Original Initial Point (1, 0)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Reconstruction')
axes[0].legend(loc='upper left')
axes[0].grid()


# Generalization
initial_state = [0, 1]
traj1 = odeint(ode1, initial_state, t)
traj2 = odeint(ode2, initial_state, t)

axes[1].quiver(X, Y, u1, v1, color='gray', alpha=0.5)
axes[1].plot(traj1[:, 0], traj1[:, 1], 'r-', label='Trajectory of Ground Truth ODE')
axes[1].plot(traj2[:, 0], traj2[:, 1], 'b--', label='Trajectory of Predicted ODE')
axes[1].plot(0, 1, 'go', label='New Initial Point (0, 1)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Generalization')
axes[1].legend(loc='upper left')
axes[1].grid()

# Divergence
axes[2].quiver(X, Y, u1, v1, color='r', alpha=0.5, label='Vector Field of Ground Truth ODE')

u1 = -Y + 0.1 * X
v1 = X + 0.1 * Y

axes[2].quiver(X, Y, u1, v1, color='b', alpha=0.5, label='Vector Field of Predicted ODE')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
axes[2].set_title('Divergence')
axes[2].legend(loc='upper left')
axes[2].grid()

plt.show()
plt.savefig('intro_exp.png')
# %%
