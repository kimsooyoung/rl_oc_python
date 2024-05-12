import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Pendulum:

    def __init__(
        self, 
        mass=1.0, length=0.5, damping=0.1, gravity=9.81,
        coulomb_fric=0.0, inertia=None, torque_limit=np.inf
    ):

        self.m = mass
        self.l = length
        self.b = damping
        self.g = gravity
        self.coulomb_fric = coulomb_fric
        if inertia is None:
            self.inertia = mass*length*length
        else:
            self.inertia = inertia

        self.torque_limit = torque_limit

        self.dof = 1
        self.n_actuators = 1

    def rhs(self, t, state, tau):

        if isinstance(tau, (list, tuple, np.ndarray)):
            torque = tau[0]
        else:
            torque = tau

        torque = np.clip(tau, -np.asarray(self.torque_limit), np.asarray(self.torque_limit))

        accn = (torque - self.m * self.g * self.l * np.sin(state[0]) -
                self.b * state[1] -
                np.sign(state[1]) * self.coulomb_fric) / self.inertia

        res = np.zeros(2)
        res[0] = state[1]
        res[1] = accn
        return res

    def forward_kinematics(self, pos):
        ee_pos_x = float(self.l * np.sin(pos))
        ee_pos_y = float(-self.l * np.cos(pos))
        return [ee_pos_x, ee_pos_y]

def runge_integrator(plant, t, y, dt, tau):
    k1 = plant.rhs(t, y, tau)
    k2 = plant.rhs(t + 0.5 * dt, y + 0.5 * dt * k1, tau)
    k3 = plant.rhs(t + 0.5 * dt, y + 0.5 * dt * k2, tau)
    k4 = plant.rhs(t + dt, y + dt * k3, tau)
    return (k1 + 2 * (k2 + k3) + k4) / 6.0

def euler_integrator(plant, t, y, tau):
    return plant.rhs(t, y, tau)

def step(plant, t, state, tau, dt, integrator="runge_kutta"):

    if integrator == "runge_kutta":
        new_state = state + dt * runge_integrator(plant, t, state, dt, tau)
    elif integrator == "euler":
        new_state = state + dt * euler_integrator(plant, t, state, dt, tau)
    else:
        raise NotImplementedError(
            f"Sorry, the integrator {integrator} is not implemented."
        )

    new_t = t + dt
    return new_t, new_state

pendulum = Pendulum()

fig = plt.figure(figsize=(5, 5))
animation_ax = plt.axes()
animation_plots = []

animation_ax.set_xlim(-2, 2)
animation_ax.set_ylim(-2, 2)

line, = animation_ax.plot([], [], 'bo')
(bar_plot,) = animation_ax.plot([], [], "-", lw=5, color="black")
(ee_plot,) = animation_ax.plot([], [], "o", markersize=10.0, color="blue")
text_plot = animation_ax.text(0.15, 0.85, [], fontsize=10, transform=fig.transFigure)

animation_plots.append(bar_plot)
animation_plots.append(ee_plot)
animation_plots.append(text_plot)

tf = 10.0
dt = 0.01
num_steps = int(tf / dt)
par_dict = {}
par_dict["dt"] = dt
par_dict["plant"] = pendulum

global state
global t

state = [3.0, 0.0]
t = 0.0

frames = num_steps * [par_dict]

def update(par_dict):

    global state
    global t

    dt = par_dict["dt"]
    pendulum = par_dict["plant"]

    new_t, new_state = step(pendulum, t, state, 0, dt)
    ee_pos = pendulum.forward_kinematics(new_state[0])

    x = [0.0, ee_pos[0]]
    y = [0.0, ee_pos[1]]

    animation_plots[0].set_data(x, y)
    animation_plots[1].set_data(ee_pos[0], ee_pos[1])
    animation_plots[2].set_text(f"t = {t}")

    state = new_state
    t = new_t

    return animation_plots

ani = FuncAnimation(
    fig, 
    update,
    frames=frames,
    blit=True,
    repeat=False,
    interval=10,
)
plt.show()