""" 
Extended Kalman Filter.
"""
from utils.meas import Measurement
from utils.states import StateWithCovariance
import numpy as np
from scipy.stats.distributions import chi2

def check_outlier(error: np.ndarray, covariance: np.ndarray):
    """
    Normalized-Innovation-Squared (NIS) test to identify an outlier.
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

    def __init__(self, process_model, reject_outliers=False):

        self.process_model = process_model
        self.reject_outliers = reject_outliers

    def predict(
        self,
        x: StateWithCovariance, u, dt: float = None) -> StateWithCovariance:
        
        x_new = x.copy()
        if u is not None:
            Q = self.process_model.covariance(x.state, u, dt)
            x_new.state, A = self.process_model.evaluate_with_jacobian(
                x.state, u, dt)
            x_new.covariance = A @ x.covariance @ A.T + Q
            x_new.symmetrize()
            x_new.state.stamp = x.state.stamp + dt

        return x_new

    def correct(
        self,
        x: StateWithCovariance, y: Measurement, u, 
        x_jac = None, reject_outlier: bool = None, ) -> StateWithCovariance:
        
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
            if u is not None and dt > 1e-11:
                x = self.predict(x, u, dt)

        y_check, G = y.model.evaluate_with_jacobian(x.state)

        if y_check is not None:
            P = x.covariance
            R = np.atleast_2d(y.model.covariance(x.state))
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
                x.state = x.state.plus((K @ z).ravel())
                x.covariance = (np.identity(x.state.dof) - K @ G) @ P
                x.symmetrize()
        return x