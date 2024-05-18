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

class Simulator:

    def __init__(self, plant):
        self.plant = plant

        self.x = np.zeros(2 * self.plant.dof)  # position, velocity
        self.t = 0.0  # time
        self.step_counter = 0

        self.reset_data_recorder()

    def reset_data_recorder(self):
        self.t_values = []
        self.x_values = []
        self.tau_values = []
        self.step_counter = 0
        self._animation_plots = []

    def record_data(self, t, x, tau):
        self.t_values.append(np.copy(t))
        self.x_values.append(np.copy(x))
        self.tau_values.append(np.copy(tau))

    def runge_integrator(self, t, x, dt, tau):
        k1 = self.plant.rhs(t, x, tau)
        k2 = self.plant.rhs(t + 0.5 * dt, x + 0.5 * dt * k1, tau)
        k3 = self.plant.rhs(t + 0.5 * dt, x + 0.5 * dt * k2, tau)
        k4 = self.plant.rhs(t + dt, x + dt * k3, tau)
        return (k1 + 2 * (k2 + k3) + k4) / 6.0

    def euler_integrator(self, t, x, tau):
        return self.plant.rhs(t, x, tau)

    # def step(self, t, state, tau, dt, integrator="runge_kutta"):
    def step(self, tau, dt, integrator="runge_kutta"):

        if integrator == "runge_kutta":
            self.x += dt * self.runge_integrator(self.t, self.x, dt, tau)
        elif integrator == "euler":
            self.x += dt * self.euler_integrator(self.t, self.x, tau)
        else:
            raise NotImplementedError(
                f"Sorry, the integrator {integrator} is not implemented."
            )

        self.t += dt
        self.record_data(self.t, self.x.copy(), tau)

        return new_t, new_state

    def _animation_step(self, par_dict):

        dt = par_dict["dt"]
        pendulum = par_dict["plant"]
        integrator = par_dict["integrator"]

        self.step(0, dt, integrator=integrator)
        ee_pos = self.plant.forward_kinematics(new_state[0])

        x = [0.0, ee_pos[0]]
        y = [0.0, ee_pos[1]]

        self._animation_plots[0].set_data(x, y)
        self._animation_plots[1].set_data(ee_pos[0], ee_pos[1])
        self._animation_plots[2].set_text(f"t = {t}")

        state = new_state
        t = new_t

        return self._animation_plots

    def simulate_and_animate(
        self, t0, x0, tf, dt, 
        controller=None, integrator="runge_kutta",
    ):
        self.reset_data_recorder()

        fig = plt.figure(figsize=(5, 5))
        animation_ax = plt.axes()

        animation_ax.set_xlim(-2, 2)
        animation_ax.set_ylim(-2, 2)

        line, = animation_ax.plot([], [], 'bo')
        (bar_plot,) = animation_ax.plot([], [], "-", lw=5, color="black")
        (ee_plot,) = animation_ax.plot([], [], "o", markersize=10.0, color="blue")
        text_plot = animation_ax.text(0.15, 0.85, [], fontsize=10, transform=fig.transFigure)

        self._animation_plots.append(bar_plot)
        self._animation_plots.append(ee_plot)
        self._animation_plots.append(text_plot)

        num_steps = int(tf / dt)
        par_dict = {}
        par_dict["dt"] = dt
        par_dict["plant"] = pendulum
        par_dict["integrator"] = integrator
        frames = num_steps * [par_dict]

        self.animation = FuncAnimation(
            fig,
            _animation_step,
            frames=frames,
            blit=True,
            repeat=False,
            interval=10,
        )
        plt.show()

        return self.t_values, self.x_values, self.tau_values

pendulum = Pendulum()
sim = Simulator(plant=pendulum)

dt = 0.01
tf = 10.0

T, X, U = sim.simulate_and_animate(
    t0=0.0, x0=[3.1, 0.0], tf=tf, dt=dt,
    controller=None, integrator="runge_kutta"
)