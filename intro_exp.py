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

init_point = [1, 0]
traj1 = odeint(ode1, init_point, t)
traj2 = odeint(ode2, init_point, t)

x = np.linspace(-4, 4, 20)
y = np.linspace(-4, 4, 20)
X, Y = np.meshgrid(x, y)

u1 = -Y
v1 = X

plt.rcParams.update({'font.size': 16})
fig, axes = plt.subplots(1, 3, figsize=(25, 8), sharex=True, sharey=True)

# Reconstruction
axes[0].quiver(X, Y, u1, v1, color='gray', alpha=0.5)
axes[0].plot(traj1[:, 0], traj1[:, 1], 'r-', label='Trajectory of ground truth ODE')
axes[0].plot(traj2[:, 0], traj2[:, 1], 'b--', label='Trajectory of predicted ODE')
axes[0].plot(init_point[0], init_point[1], 'go', label=f'Original Initial Point ({init_point[0]}, {init_point[1]})')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Reconstruction')
axes[0].legend(loc='upper left')
axes[0].grid()


# Generalization
init_point = [2, 0]
traj1 = odeint(ode1, init_point, t)
traj2 = odeint(ode2, init_point, t)

axes[1].quiver(X, Y, u1, v1, color='gray', alpha=0.5)
axes[1].plot(traj1[:, 0], traj1[:, 1], 'r-', label='Trajectory of Ground Truth ODE')
axes[1].plot(traj2[:, 0], traj2[:, 1], 'b--', label='Trajectory of Predicted ODE')
axes[1].plot(init_point[0], init_point[1], 'go', label=f'New Initial Point ({init_point[0]}, {init_point[1]})')
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

plt.tight_layout()
plt.savefig('intro_exp.png', format='png', transparent=True)
plt.show()

# %%
plt.quiver(X, Y, u1, v1, color='gray', alpha=0.5)

# dx = np.diff(traj2
# [0], init_point[1], 'go', label=f'Initial Point ({init_point[0]}, {init_point[1]})')
plt.xlabel('x')
plt.ylabel('y')
# plt.title('Reconstruction')
# plt.legend(loc='upper left')
plt.grid()

plt.tight_layout()
plt.savefig('data_gen_01.png', format='png', transparent=True)
plt.show()
# %%
