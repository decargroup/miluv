import os
import numpy as np
import matplotlib.pyplot as plt
from pymlg import SE3

import examples.ekfutils.common as common
import miluv.utils as utils

# EKF parameters
state_dimension = 6
np.random.seed(0)

# Covariance matrices
P0 = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1]) # Initial state covariance

imu_noise_params = utils.get_imu_noise_params("ifo001", "px4")
Q = np.diag([
    imu_noise_params["gyro"][0]**2, 
    imu_noise_params["gyro"][1]**2,
    imu_noise_params["gyro"][2]**2,
    0.1 , 0.1, 0.1
]) # Process noise covariance

R_range = 0.3**2 # Range measurement noise covariance
R_height = 0.5**2 # Height measurement noise covariance

class EKF:
    def __init__(self, state: SE3, anchors: dict[int, np.ndarray], tag_moment_arms: dict[str, np.ndarray]):
        # Add noise to the initial state using P0 to reflect uncertainty in the initial state
        self.x = state @ SE3.Exp(np.random.multivariate_normal(np.zeros(state_dimension), P0))
        
        self.P = P0
        self.anchors = anchors
        self.tag_moment_arms = tag_moment_arms["ifo001"]

    def predict(self, u: np.ndarray, dt: float) -> None:
        A = self._process_jacobian(self.x, u, dt)
        Qd = self._process_covariance(self.x, u, dt)
        
        self.x = self._process_model(self.x, u, dt)
        self.P = A @ self.P @ A.T + Qd
    
    def correct(self, y: dict) -> None:
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
    def _process_model(x: SE3, u: np.ndarray, dt: float) -> SE3:
        return x @ SE3.Exp(u * dt)
    
    @staticmethod
    def _process_jacobian(x: SE3, u: np.ndarray, dt: float) -> np.ndarray:
        return SE3.adjoint(np.linalg.inv(SE3.Exp(u * dt)))
    
    @staticmethod
    def _process_covariance(x: SE3, u: np.ndarray, dt: float) -> np.ndarray:
        return dt * SE3.left_jacobian(-dt * u) @ Q @ SE3.left_jacobian(-dt * u).T
    
    def _range_measurement(self, x: SE3, anchor_id: int, tag_id: int) -> float:
        Pi = np.hstack([np.eye(3), np.zeros((3, 1))])
        r_tilde = np.vstack((self.tag_moment_arms[tag_id], 1)).reshape(4, 1)
        
        return np.linalg.norm(self.anchors[anchor_id] - Pi @ x @ r_tilde)
    
    def _range_jacobian(self, x: SE3, anchor_id: int, tag_id: int) -> np.ndarray:
        Pi = np.hstack([np.eye(3), np.zeros((3, 1))])
        r_tilde = np.vstack((self.tag_moment_arms[tag_id], 1)).reshape(4, 1)
        
        vector = (self.anchors[anchor_id] - Pi @ x @ r_tilde).reshape(3, 1)
        
        return -1 * vector.T / np.linalg.norm(vector) @ Pi @ x @ SE3.odot(r_tilde)
    
    @staticmethod
    def _height_measurement(x: SE3) -> float:
        return x[2, 3]
    
    @staticmethod
    def _height_jacobian(x: SE3) -> np.ndarray:
        a = np.array([0, 0, 1, 0]).reshape(4, 1)
        b = np.array([0, 0, 0, 1])
        return (a.T @ x @ SE3.odot(b))

class EvaluateEKF:
    def __init__(self, gt_se3: list[SE3], ekf_history: common.StateHistory, exp_name: str):
        self.timestamps, vector_states, self.covariances = ekf_history.get()
        self.states = np.array([SE3.Exp(state) for state in vector_states])
        self.gt_se3 = gt_se3
        self.exp_name = exp_name
        
        self.error = np.zeros((len(self.gt_se3), 6))
        for i in range(0, len(self.gt_se3)):
            self.error[i, :] = SE3.Log(SE3.inverse(self.gt_se3[i]) @ self.states[i]).ravel()
            
        self.error_titles = [r"$\delta_{\phi_x}$", r"$\delta_{\phi_y}$", r"$\delta_{\phi_z}$", 
                             r"$\delta_{x}$", r"$\delta_{y}$", r"$\delta_{z}$"]

    def plot_poses(self) -> None:
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))
        fig.suptitle("Ground Truth vs. EKF Poses")
        
        gt = np.array([SE3.Log(pose).ravel() for pose in self.gt_se3])
        est = np.array([SE3.Log(pose).ravel() for pose in self.states])
        for i in range(0, 6):
            axs[i % 3, int(i > 2)].plot(self.timestamps, gt[:, i], label="GT")
            axs[i % 3, int(i > 2)].plot(self.timestamps, est[:, i], label="Est")
            axs[i % 3, int(i > 2)].set_ylabel(self.error_titles[i])
        axs[2, 0].set_xlabel("Time [s]")
        axs[2, 1].set_xlabel("Time [s]")
        axs[0, 0].legend()
        
        if not os.path.exists('results/plots/ekf_vins_one_robot'):
            os.makedirs('results/plots/ekf_vins_one_robot')
        plt.savefig(f"results/plots/ekf_vins_one_robot/{self.exp_name}_poses.pdf")
        plt.close()

    def plot_error(self) -> None:
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))
        fig.suptitle("Three-Sigma Error Plots")
        
        for i in range(0, 6):
            axs[i % 3, int(i > 2)].plot(self.timestamps, self.error[:, i])
            axs[i % 3, int(i > 2)].fill_between(
                self.timestamps,
                -1 * 3*np.sqrt(self.covariances[:, i, i]), 
                3*np.sqrt(self.covariances[:, i, i]), 
                alpha=0.5
            )
            axs[i % 3, int(i > 2)].set_ylabel(self.error_titles[i])
        axs[2, 0].set_xlabel("Time [s]")
        axs[2, 1].set_xlabel("Time [s]")
        
        if not os.path.exists('results/plots/ekf_vins_one_robot'):
            os.makedirs('results/plots/ekf_vins_one_robot')
        plt.savefig(f"results/plots/ekf_vins_one_robot/{self.exp_name}_error.pdf")
        plt.close()

    def save_results(self) -> None:
        pos_rmse, att_rmse = self.get_rmse()
        
        myCsvRow = f"{self.exp_name},{pos_rmse},{att_rmse}\n"
        
        if not os.path.exists('results/ekf_vins_one_robot.csv'):
            with open('results/ekf_vins_one_robot.csv','w') as file:
                file.write("exp_name,pos_rmse,att_rmse\n")
        
        with open('results/ekf_vins_one_robot.csv','a') as file:
            file.write(myCsvRow)
                
    def get_rmse(self) -> tuple[float, float]:
        pos_rmse = np.sqrt(np.mean(self.error[:, 3:] ** 2))
        att_rmse = np.sqrt(np.mean(self.error[:, :3] ** 2))
        return pos_rmse, att_rmse