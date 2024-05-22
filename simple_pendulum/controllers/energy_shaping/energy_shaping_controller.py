"""
Energy Shaping Controller
=========================
"""


# Other imports
import numpy as np

# Local imports
from simple_pendulum.model.pendulum_plant import PendulumPlant
from simple_pendulum.controllers.abstract_controller import AbstractController
from simple_pendulum.controllers.lqr.lqr_controller import LQRController


class EnergyShapingController(AbstractController):
    """
    Controller which swings up the pendulum by regulating its energy.
    """
    def __init__(self,
                 mass=1.0,
                 length=0.5,
                 damping=0.1,
                 gravity=9.81,
                 torque_limit=2.0,
                 k=1.0):
        """
        Controller which swings up the pendulum by regulating its energy.

        **Parameters**

        mass : float, default=1.0
            mass of the pendulum [kg]
        length : float, default=0.5
            length of the pendulum [m]
        damping : float, default=0.1
            damping factor of the pendulum [kg m/s]
        gravity : float, default=9.81
            gravity (positive direction points down) [m/s^2]
        torque_limit : float, default=2.0
            torque limit of the motor [Nm]
        k : float, default=1.0
            the weight determining the output torque with respect to the
            current energy level.
        """
        self.m = mass
        self.l = length
        self.b = damping
        self.g = gravity
        self.torque_limit = torque_limit
        self.k = k
        self.plant = PendulumPlant(mass=mass,
                                   length=length,
                                   damping=damping,
                                   gravity=gravity,
                                   coulomb_fric=0.0,
                                   inertia=mass*length**2,
                                   torque_limit=torque_limit)

    def set_goal(self, x):
        """
        Set the goal for the controller.
        This function calculates the energy of the goal state.

        Parameters
        ----------
        x : array-like
            the goal state for the pendulum
        """
        self.goal = [x[0], x[1]]
        self.desired_energy = self.plant.total_energy(x)

    def get_control_output(self, meas_pos, meas_vel,
                           meas_tau=0, meas_time=0):
        """
        The function to compute the control input for the pendulum actuator

        Parameters
        ----------
        meas_pos : float
            the position of the pendulum [rad]
        meas_vel : float
            the velocity of the pendulum [rad/s]
        meas_tau : float
            the meastured torque of the pendulum [Nm]
            (not used)
        meas_time : float
            the collapsed time [s]
            (not used)

        Returns
        -------
        des_pos : float
            the desired position of the pendulum [rad]
            (not used, returns None)
        des_vel : float
            the desired velocity of the pendulum [rad/s]
            (not used, returns None)
        des_tau : float
            the torque supposed to be applied by the actuator [Nm]
        """
        pos = float(np.squeeze(meas_pos))
        vel = float(np.squeeze(meas_vel))

        if np.abs(pos) < 0.05 and np.abs(vel) < 0.05:
            des_tau = 1.0*self.torque_limit
        else:
            total_energy = self.plant.total_energy([pos, vel])

            des_tau = -self.k*vel*(total_energy - self.desired_energy) + \
                self.b*vel

        des_tau = min(des_tau, self.torque_limit)
        des_tau = max(des_tau, -self.torque_limit)

        # since this is a pure torque controller,
        # set des_pos and des_vel to None
        des_pos = None
        des_vel = None

        return des_pos, des_vel, des_tau


class EnergyShapingAndLQRController(AbstractController):
    """
    Controller which swings up the pendulum with the energy shaping
    controller and stabilizes the pendulum with the lqr controller.
    """
    def __init__(self, mass=1.0, length=0.5, damping=0.1, coulomb_fric=0.0,
                 gravity=9.81, torque_limit=np.inf, k=1.0,
                 Q=np.diag((10, 1)), R=np.array([[1]]), compute_RoA=False):
        """
        Controller which swings up the pendulum with the energy shaping
        controller and stabilizes the pendulum with the lqr controller.

        Parameters
        ----------
        mass : float, default=1.0
            mass of the pendulum [kg]
        length : float, default=0.5
            length of the pendulum [m]
        damping : float, default=0.1
            damping factor of the pendulum [kg m/s]
        coulomb_fric : float, default=0.0
            friction term, (independent of magnitude of velocity), unit: Nm
        gravity : float, default=9.81
            gravity (positive direction points down) [m/s^2]
        torque_limit : float, default=np.inf
            the torque_limit of the pendulum actuator
        k : float, default=1.0
            (energy controller) the weight determining the output torque with
            respect to the current energy level.
        Q : array-like, default=np.diag(10, 1)
            (LQR controller) the state cost matrix, np.shape(Q) = (2,2)
        R : array-like, default=np.array([[1]])
            (LQR controller) the control cost matrix, np.shape(R) = (1,1)
        compute_RoA : bool, default=False
            (LQR controller) whether to compute the region of attraction of the
            LQR controller (requires drake)
        """

        self.m = mass
        self.l = length
        self.b = damping
        self.cf = coulomb_fric
        self.g = gravity

        self.energy_shaping_controller = EnergyShapingController(mass=mass,
                                                                 length=length,
                                                                 damping=damping,
                                                                 gravity=gravity,
                                                                 torque_limit=torque_limit,
                                                                 k=k)
        self.lqr_controller = LQRController(mass=mass,
                                            length=length,
                                            damping=damping,
                                            coulomb_fric=coulomb_fric,
                                            gravity=gravity,
                                            torque_limit=torque_limit,
                                            Q=Q,
                                            R=R,
                                            compute_RoA=compute_RoA)

        self.active_controller = "none"
        self.swingup_time = None
        self.eps = [0.2, 0.1]

    def set_goal(self, x):
        """
        Set the goal for the controller.

        Parameters
        ----------
        x : array-like
            the goal state for the pendulum
        """
        self.energy_shaping_controller.set_goal(x)
        self.lqr_controller.set_goal(x)

    def get_control_output(self, meas_pos, meas_vel,
                           meas_tau=0, meas_time=0, verbose=False):
        """
        The function to compute the control input for the pendulum actuator

        Parameters
        ----------
        meas_pos : float
            the position of the pendulum [rad]
        meas_vel : float
            the velocity of the pendulum [rad/s]
        meas_tau : float
            the meastured torque of the pendulum [Nm]
            (not used)
        meas_time : float
            the collapsed time [s]
            (not used)
        verbose : bool, default=False
            whether to print when the controller switches between
            energy shaping and lqr

        Returns
        -------
        des_pos : float
            the desired position of the pendulum [rad]
            (not used, returns None)
        des_vel : float
            the desired velocity of the pendulum [rad/s]
            (not used, returns None)
        des_tau : float
            the torque supposed to be applied by the actuator [Nm]
        """
        des_pos, des_vel, u = self.lqr_controller.get_control_output(meas_pos,
                                                                     meas_vel)
        th = meas_pos + np.pi
        th = (th + np.pi) % (2*np.pi) - np.pi
        if (self.swingup_time is None and
            np.abs(th) < self.eps[0] and
            np.abs(meas_vel) < self.eps[1]):
            self.swingup_time = meas_time

        if u is not None:
            if self.active_controller != "lqr":
                self.active_controller = "lqr"
                if verbose:
                    print("Switching to lqr control")
        else:
            if self.active_controller != "EnergyShaping":
                self.active_controller = "EnergyShaping"
                if verbose:
                    print("Switching to energy shaping control")
            des_pos, des_vel, u = (self.energy_shaping_controller.
                                   get_control_output(meas_pos, meas_vel))
        return des_pos, des_vel, u

    def get_swingup_time(self):
        return self.swingup_time
