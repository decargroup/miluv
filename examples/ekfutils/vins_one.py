import numpy as np
from pymlg import SE3

import examples.ekfutils.common as common

# EKF parameters
state_dimension = 6
np.random.seed(0)

# Covariance matrices
P0 = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
Q = np.diag([0.001, 0.001, 0.001, 0.1 , 0.1, 0.1])
R_range = 0.3
R_height = 0.3

class EKF:
    def __init__(self, state, anchors, tag_moment_arms):
        # Add noise to the initial state using P0 to reflect uncertainty in the initial state
        self.x = state @ SE3.Exp(np.random.multivariate_normal(np.zeros(state_dimension), P0))
        
        self.P = P0
        self.anchors = anchors
        self.tag_moment_arms = tag_moment_arms["ifo001"]

    def predict(self, u, dt):
        A = self._process_jacobian(self.x, u, dt)
        Qd = self._process_covariance(self.x, u, dt)
        
        self.x = self._process_model(self.x, u, dt)
        self.P = A @ self.P @ A.T + Qd
    
    def correct(self, y):
        if "range" in y:
            anchor_id = y["to_id"]
            tag_id = y["from_id"]
            H = self._range_jacobian(self.x, anchor_id, tag_id)
            R = R_range
            actual_measurement = y["range"]
            predicted_measurement = self._range_measurement(self.x, anchor_id, tag_id)
        elif "height" in y:
            H = self._height_jacobian(self.x)
            R = R_height
            actual_measurement = y["height"]
            predicted_measurement = self._height_measurement(self.x)
        else:
            return
        
        z = np.array([actual_measurement - predicted_measurement])
        S = H @ self.P @ H.T + R
        
        if common.is_outlier(z, S):
            return
        
        K = self.P @ H.T / S
        
        self.x = self.x @ SE3.Exp(K @ z)
        self.P = (np.eye(6) - K @ H) @ self.P
        
        # Ensure symmetric covariance matrix
        self.P = 0.5 * (self.P + self.P.T)
        
    @staticmethod
    def _process_model(x, u, dt):
        return x @ SE3.Exp(u * dt)
    
    @staticmethod
    def _process_jacobian(x, u, dt):
        return SE3.adjoint(np.linalg.inv(SE3.Exp(u * dt)))
    
    @staticmethod
    def _process_covariance(x, u, dt):
        return dt * SE3.left_jacobian(-dt * u) @ Q @ SE3.left_jacobian(-dt * u).T
    
    def _range_measurement(self, x, anchor_id, tag_id):
        Pi = np.hstack([np.eye(3), np.zeros((3, 1))])
        r_tilde = np.vstack((self.tag_moment_arms[tag_id], 1)).reshape(4, 1)
        
        return np.linalg.norm(self.anchors[anchor_id] - Pi @ x @ r_tilde)
    
    def _range_jacobian(self, x, anchor_id, tag_id):
        Pi = np.hstack([np.eye(3), np.zeros((3, 1))])
        r_tilde = np.vstack((self.tag_moment_arms[tag_id], 1)).reshape(4, 1)
        
        vector = (self.anchors[anchor_id] - Pi @ x @ r_tilde).reshape(3, 1)
        
        return -1 * vector.T / np.linalg.norm(vector) @ Pi @ x @ SE3.odot(r_tilde)
    
    @staticmethod
    def _height_measurement(x):
        return x[2, 3]
    
    @staticmethod
    def _height_jacobian(x):
        a = np.array([0, 0, 1, 0]).reshape(4, 1)
        b = np.array([0, 0, 0, 1])
        return (a.T @ x @ SE3.odot(b))