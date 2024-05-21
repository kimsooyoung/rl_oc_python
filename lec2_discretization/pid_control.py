import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt

from utils import load_trajectory
from matplotlib.animation import FuncAnimation

platform = None 
if os.name == 'nt':
    platform = "Windows"

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
        self.base = [0, 0]
        self.workspace_range = [
            [-1.2*self.l, 1.2*self.l],
            [-1.2*self.l, 1.2*self.l]
        ]

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

    def set_state(self, time, x, step_counter=0):
        self.x = np.copy(x)
        self.t = np.copy(float(time))

    def reset_data_recorder(self):
        self.t_values = []
        self.x_values = []
        self.tau_values = []
        self.step_counter = 0

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


    def _animation_init(self):
        self._animation_ax.set_xlim(
            self.plant.workspace_range[0][0], self.plant.workspace_range[0][1]
        )
        self._animation_ax.set_ylim(
            self.plant.workspace_range[1][0], self.plant.workspace_range[1][1]
        )
        self._animation_ax.set_xlabel("x position [m]")
        self._animation_ax.set_ylabel("y position [m]")

        return self._animation_plots

    def _animation_step(self, par_dict):

        dt = par_dict["dt"]
        pendulum = par_dict["plant"]
        controller = par_dict["controller"]
        integrator = par_dict["integrator"]

        if controller is not None:
            t0 = time.time()
            pos, vel, tau = controller.get_control_output(
                meas_pos=self.x[0],
                meas_vel=self.x[1],
                meas_tau=np.zeros(self.plant.dof),
                meas_time=self.t,
            )
            if time.time() - t0 > dt:
                controller_slow = True
        else:
            tau = np.zeros(self.plant.n_actuators)
        self.step(tau, dt, integrator=integrator)

        ee_pos = self.plant.forward_kinematics(self.x[0])
        x = [0.0, ee_pos[0]]
        y = [0.0, ee_pos[1]]

        self._animation_plots[0].set_data(x, y)
        if platform == "Windows":
            self._animation_plots[1].set_data([x[1]], [y[1]])
        else:
            self._animation_plots[1].set_data(x[1], y[1])
        t = float(self._animation_plots[2].get_text()[4:])
        t = round(t + dt, 3)
        self._animation_plots[2].set_text(f"t = {t}")

        return self._animation_plots

    def simulate_and_animate(
        self, t0, x0, tf, dt, 
        controller=None, integrator="runge_kutta",
    ):
        self.set_state(t0, x0)
        self.reset_data_recorder()

        fig = plt.figure(1, figsize=(5, 5))
        self._animation_ax = plt.axes()
        self._animation_plots = []

        line, = self._animation_ax.plot([], [], 'bo')
        (bar_plot,) = self._animation_ax.plot([], [], "-", lw=5, color="black")
        (ee_plot,) = self._animation_ax.plot([], [], "o", markersize=10.0, color="blue")
        text_plot = self._animation_ax.text(0.15, 0.85, [], fontsize=10, transform=fig.transFigure)

        self._animation_plots.append(bar_plot)
        self._animation_plots.append(ee_plot)

        self._animation_plots.append(text_plot)
        self._animation_plots[-1].set_text("t = 0.000")

        num_steps = int(tf / dt)
        par_dict = {}
        par_dict["dt"] = dt
        par_dict["plant"] = pendulum
        par_dict["controller"] = controller
        par_dict["integrator"] = integrator
        frames = num_steps * [par_dict]

        self.animation = FuncAnimation(
            fig,
            self._animation_step,
            frames=frames,
            init_func=self._animation_init,
            blit=True,
            repeat=False,
            interval=dt * 1000,
        )
        plt.show()

        return self.t_values, self.x_values, self.tau_values

class GravityCompController():

    def __init__(
        self,
        mass=1.0, length=0.5, gravity=9.81,
    ):
        # mass=1.0, length=0.5, damping=0.1, gravity=9.81,
        # coulomb_fric=0.0, inertia=None, torque_limit=np.inf

        self.counter = 0
        self.u = 0
        self.g = gravity
        self.m = mass
        self.l = length

    def get_control_output(self, meas_pos, meas_vel, meas_tau, meas_time):
        # compensate gravity with input torque
        des_tau = self.m * self.g * self.l * np.sin(meas_pos)

        # since this is a pure torque controller, set pos_des and vel_des to 0
        des_pos = 0
        des_vel = 0
        print(f"des_tau: {des_tau}")

        return des_pos, des_vel, des_tau

    def set_goal(self, x):
        self.goal = x


class PIDController(AbstractController):
    """
    Controller acts on a predefined trajectory and adds PID control gains.
    """
    def __init__(self, data_dict, Kp, Ki, Kd, use_feed_forward=True):

        self.traj_time = data_dict["des_time"]
        self.traj_pos = data_dict["des_pos"]
        self.traj_vel = data_dict["des_vel"]
        if use_feed_forward:
            self.traj_tau = data_dict["des_tau"]

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.use_feed_forward = use_feed_forward

        self.counter = 0
        self.errors = []
        self.dt = self.traj_time[1] - self.traj_time[0]
        self.last_pos = 0.0
        self.last_vel = 0.0

    def init(self, x0):
        self.counter = 0
        self.errors = []
        self.last_pos = 0.0
        self.last_vel = 0.0

    def set_goal(self, x):
        pass

    def get_control_output(self, meas_pos=None, meas_vel=None, meas_tau=None,
                           meas_time=None):

        des_pos = self.last_pos
        des_vel = self.last_vel
        des_tau = 0.0

        if self.counter < len(self.traj_time):
            des_pos = self.traj_pos[self.counter]
            des_vel = self.traj_vel[self.counter]
            if self.use_feed_forward:
                des_tau = self.traj_tau[self.counter]

            self.last_pos = des_pos
            self.last_vel = des_vel

        e = des_pos - meas_pos
        e = (e + np.pi) % (2*np.pi) - np.pi
        self.errors.append(e)

        P = self.Kp*e
        I = self.Ki*np.sum(self.errors)*self.dt
        if len(self.errors) > 2:
            D = self.Kd*(self.errors[-1]-self.errors[-2])/self.dt
        else:
            D = 0.0

        des_tau = des_tau + P + I + D

        self.counter += 1

        return des_pos, des_vel, des_tau


def plot_state(T, X, U):

    plt.figure(2, figsize=(8, 5))

    plt.subplot(3, 1, 1)
    plt.plot(T, np.asarray(X).T[0], color='r', label=r'$\theta$')
    plt.ylabel("angle [rad]")
    plt.legend(loc="best")

    plt.subplot(3, 1, 2)
    plt.plot(T, np.asarray(X).T[1], color='g', label=r'$\dot{\theta}$')
    plt.ylabel("angular velocity [rad/s]")
    plt.legend(loc="best")

    plt.subplot(3, 1, 3)
    plt.plot(T, U, color='b', label=r'$u$')
    plt.xlabel("time [s]")
    plt.ylabel("input torque [Nm]")
    plt.legend(loc="best")

    plt.show()


pendulum = Pendulum()
sim = Simulator(plant=pendulum)
gravity_comp_controller = GravityCompController()
gravity_comp_controller.set_goal([np.pi/3, 0])

dt = 0.01
tf = 3.0
derise_state = [1.5707+1.0, 0.0]

T, X, U = sim.simulate_and_animate(
    t0=0.0, x0=derise_state, tf=tf, dt=dt,
    controller=gravity_comp_controller, integrator="runge_kutta"
)

plot_state(T, X, U)

