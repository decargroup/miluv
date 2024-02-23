""" 
Module containing Extended Kalman Filter.
"""
from utils.inputs import Input
from utils.states import State
from utils.models import ProcessModel
from utils import StateWithCovariance
import numpy as np
from scipy.stats.distributions import chi2
from utils.measurement import Measurement

def check_outlier(error: np.ndarray, covariance: np.ndarray):
    """
    Performs the Normalized-Innovation-Squared (NIS) test to identify
    an outlier.
    """
    error = error.reshape((-1, 1))
    nis = np.ndarray.item(error.T @ np.linalg.solve(covariance, error))
    if nis > chi2.ppf(0.99, df=error.size):
        is_outlier = True
    else:
        is_outlier = False

    return is_outlier

class ExtendedKalmanFilter:
    """
    On-manifold nonlinear Kalman filter.
    """

    __slots__ = ["process_model", "reject_outliers", "bias"]

    def __init__(self, process_model: ProcessModel, reject_outliers=False):
        """
        Parameters
        ----------
        process_model : ProcessModel
            process model to be used in the prediction step
        reject_outliers : bool, optional
            whether to apply the NIS test to measurements, by default False
        """
        self.process_model = process_model
        self.reject_outliers = reject_outliers

    def predict(
        self,
        x: StateWithCovariance,
        u: Input,
        dt: float = None,
        x_jac: State = None,
        output_details: bool = False,
    ) -> StateWithCovariance:
        """
        Propagates the state forward in time using a process model. The user
        must provide the current state, input, and time interval

        .. note::
            If the time interval ``dt`` is not provided in the arguments, it will
            be taken as the difference between the input stamp and the state stamp.

        Parameters
        ----------
        x : StateWithCovariance
            The current state estimate.
        u : Input
            Input measurement to be given to process model
        dt : float, optional
            Duration to next time step. If not provided, dt will be calculated
            with ``dt = u.stamp - x.state.stamp``.
        x_jac : State, optional
            Evaluation point for the process model Jacobian. If not provided, the
            current state estimate will be used.

        Returns
        -------
        StateWithCovariance
            New predicted state
        """

        # Make a copy so we dont modify the input
        x_new = x.copy()

        # If state has no time stamp, load from measurement.
        # usually only happens on estimator start-up
        if x.state.stamp is None:
            t_km1 = u.stamp
        else:
            t_km1 = x.state.stamp

        if dt is None:
            dt = u.stamp - t_km1

        if dt < 0:
            raise RuntimeError("dt is negative!")

        # Load dedicated jacobian evaluation point if user specified.
        if x_jac is None:
            x_jac = x.state

        details_dict = {}
        if u is not None:
            Q = self.process_model.covariance(x_jac, u, dt)
            x_new.state, A = self.process_model.evaluate_with_jacobian(
                x.state, u, dt
            )
            x_new.covariance = A @ x.covariance @ A.T + Q
            x_new.symmetrize()
            x_new.state.stamp = t_km1 + dt

            details_dict = {"A": A, "Q": Q}

        if output_details:
            return x_new, details_dict
        else:
            return x_new

    def correct(
        self,
        x: StateWithCovariance,
        y: Measurement,
        u: Input,
        x_jac: State = None,
        reject_outlier: bool = None,
        output_details: bool = False,
    ) -> StateWithCovariance:
        """
        Fuses an arbitrary measurement to produce a corrected state estimate.
        If a measurement model returns ``None`` from its ``evaluate()`` method,
        the measurement will not be fused.

        Parameters
        ----------
        x : StateWithCovariance
            The current state estimate.
        y : Measurement
            Measurement to be fused into the current state estimate.
        u: Input
            Most recent input, to be used to predict the state forward
            if the measurement stamp is larger than the state stamp. If set to
            None, no prediction will be performed and the correction will
            just be done with the current state estimate.
        x_jac : State, optional
            valuation point for the process model Jacobian. If not provided, the
            current state estimate will be used.
        reject_outlier : bool, optional
            Whether to apply the NIS test to this measurement, by default None,
            in which case the value of ``self.reject_outliers`` will be used.
        output_details : bool, optional
            Whether to output intermediate computation results (innovation,
            innovation covariance) in an additional returned dict.
        Returns
        -------
        StateWithCovariance
            The corrected state estimate
        """
        # Make copy to avoid modifying the input
        x = x.copy()

        if x.state.stamp is None:
            x.state.stamp = y.stamp

        # Load default outlier rejection option
        if reject_outlier is None:
            reject_outlier = self.reject_outliers

        # If measurement stamp is later than state stamp, do prediction step
        # until current time.
        if y.stamp is not None:
            dt = y.stamp - x.state.stamp
            if dt < -1e10:
                raise RuntimeError(
                    "Measurement stamp is earlier than state stamp"
                )
            elif u is not None and dt > 1e-11:
                x = self.predict(x, u, dt)

        if x_jac is None:
            x_jac = x.state

        y_check, G = y.model.evaluate_with_jacobian(x.state)

        details_dict = {}
        if y_check is not None:
            P = x.covariance
            R = np.atleast_2d(y.model.covariance(x_jac))
            G = np.atleast_2d(G)
            z = y.minus(y_check)
            S = G @ P @ G.T + R

            outlier = False

            # Test for outlier if requested.
            if reject_outlier:
                outlier = check_outlier(z, S)

            if not outlier:
                # Do the correction
                K = np.linalg.solve(S.T, (P @ G.T).T).T
                dx = K @ z
                dx = dx.ravel()
                x.state = x.state.plus(dx.ravel())

                # dx_attitude = dx[0:3]
                # dx_velocity = dx[3:6]
                # dx_position = dx[6:9]

                # for value in x.state.value:
                #     if isinstance(value, SE23State):
                #         pose_state = value
                
                # pose_state.attitude = pose_state.attitude @ SO3.Exp(-dx_attitude)
                # pose_state.velocity = pose_state.velocity - dx_velocity
                # pose_state.position = pose_state.position - dx_position
                
                # if self.bias:
                #     dx_bias = dx[9:15]
                #     bias_state = x.state.get_state_by_id('bias')
                #     bias_state.value = bias_state.value - dx_bias
                #     x.state.value = [pose_state, bias_state]
                # else:
                #     x.state.value = [pose_state]

                x.covariance = (np.identity(x.state.dof) - K @ G) @ P
                x.symmetrize()

            details_dict = {"z": z, "S": S, "is_outlier": outlier}

        if output_details:
            return x, details_dict
        else:
            return x