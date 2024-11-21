import os
import numpy as np
import matplotlib.pyplot as plt
from pymlg import SE23
from dataclasses import dataclass

import examples.ekfutils.imu_models as imu_models
import examples.ekfutils.common as common
import miluv.utils as utils

np.random.seed(0)

# EKF parameters
pose_dimension = 9
bias_dimension = 6
full_state_dimension = pose_dimension + bias_dimension

# Covariance matrices
P0 = np.diag([
    0.01, 0.01, 0.01, # Orientation
    0.01, 0.01, 0.01, # Velocity
    0.1, 0.1, 0.1,    # Position
    0.01, 0.01, 0.01, # Gyro bias
    0.1, 0.1, 0.1,    # Accel bias
]) # Initial state covariance

imu_noise_params = utils.get_imu_noise_params("ifo001", "px4")
Q = np.diag([
    imu_noise_params["gyro"][0]**2, 
    imu_noise_params["gyro"][1]**2,
    imu_noise_params["gyro"][2]**2,
    imu_noise_params["accel"][0]**2,
    imu_noise_params["accel"][1]**2,
    imu_noise_params["accel"][2]**2,
    imu_noise_params["gyro_bias"][0]**2,
    imu_noise_params["gyro_bias"][1]**2,
    imu_noise_params["gyro_bias"][2]**2,
    imu_noise_params["accel_bias"][0]**2,
    imu_noise_params["accel_bias"][1]**2,
    imu_noise_params["accel_bias"][2]**2
]) # Process noise covariance

R_range = 0.3**2 # Range measurement noise covariance
R_height = 0.5**2 # Height measurement noise covariance

@dataclass
class State:
    pose: SE23
    bias: np.ndarray

class EKF:
    def __init__(self, state: SE23, anchors: dict[int: np.ndarray], tag_moment_arms: dict[str: dict[int: np.ndarray]]):
        # Add noise to the initial state using P0 to reflect uncertainty in the initial state
        self.x = State(
            pose=state @ SE23.Exp(np.random.multivariate_normal(np.zeros(pose_dimension), P0[:pose_dimension, :pose_dimension])),
            bias=np.zeros(bias_dimension),
        )
        
        self.P = P0
        self.anchors = anchors
        self.tag_moment_arms = tag_moment_arms["ifo001"]

    def predict(self, u: np.ndarray, dt: float) -> None:
        if dt == 0:
            return
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
        
        self.x.pose = self.x.pose @ SE23.Exp(K[:pose_dimension, :] @ z)
        self.x.bias = self.x.bias + K[pose_dimension:, :] @ z
        
        self.P = (np.eye(full_state_dimension) - K @ H) @ self.P
        
        # Ensure symmetric covariance matrix
        self.P = 0.5 * (self.P + self.P.T)
        
    @staticmethod
    def _process_model(x: State, u: np.ndarray, dt: float) -> State:
        x.pose = imu_models.G_mat(dt) @ x.pose @ imu_models.U_mat(x.bias, u, dt)
        
        return x
    
    @staticmethod
    def _process_jacobian(x: State, u: np.ndarray, dt: float) -> np.ndarray:
        jac = np.zeros((full_state_dimension, full_state_dimension))
        jac[:pose_dimension, :pose_dimension] = imu_models.U_adjoint_mat(imu_models.U_inv_mat(x.bias, u, dt))
        jac[:pose_dimension, pose_dimension:] = -imu_models.L_mat(x.bias, u, dt)
        
        jac[pose_dimension:, pose_dimension:] = np.eye(bias_dimension)
        
        return jac
    
    @staticmethod
    def _process_covariance(x: State, u: np.ndarray, dt: float) -> np.ndarray:
        noise_jac = np.zeros((full_state_dimension, Q.shape[0]))
        noise_jac[:pose_dimension, :bias_dimension] = imu_models.L_mat(x.bias, u, dt)
        noise_jac[pose_dimension:, bias_dimension:] = np.eye(bias_dimension)
        
        return (noise_jac @ Q @noise_jac.T) / dt
    
    def _range_measurement(self, x: State, anchor_id: int, tag_id: int) -> float:
        Pi = np.hstack([np.eye(3), np.zeros((3, 2))])
        r_tilde = np.vstack((self.tag_moment_arms[tag_id], 0, 1)).reshape(5, 1)
        
        return np.linalg.norm(self.anchors[anchor_id] - Pi @ x.pose @ r_tilde)
    
    def _range_jacobian(self, x: State, anchor_id: int, tag_id: int) -> np.ndarray:
        Pi = np.hstack([np.eye(3), np.zeros((3, 2))])
        r_tilde = np.vstack((self.tag_moment_arms[tag_id], 0, 1)).reshape(5, 1)
        
        vector = (self.anchors[anchor_id] - Pi @ x.pose @ r_tilde).reshape(3, 1)
        
        jac = np.zeros((1, full_state_dimension))
        jac[:, :pose_dimension] = -1 * vector.T / np.linalg.norm(vector) @ Pi @ x.pose @ SE23.odot(r_tilde)
        return jac
    
    @staticmethod
    def _height_measurement(x: State) -> float:
        return x.pose[2, 4]
    
    @staticmethod
    def _height_jacobian(x: State) -> np.ndarray:
        a = np.array([0, 0, 1, 0, 0]).reshape(5, 1)
        b = np.array([0, 0, 0, 0, 1])
        
        jac = np.zeros((1, full_state_dimension))
        jac[:, :pose_dimension] = a.T @ x.pose @ SE23.odot(b)
        return jac
    
    @property
    def pose(self) -> SE23:
        return self.x.pose
    
    @property
    def pose_covariance(self) -> np.ndarray:
        return self.P[:pose_dimension, :pose_dimension]
    
    @property
    def bias(self) -> np.ndarray:
        return self.x.bias
    
    @property
    def bias_covariance(self) -> np.ndarray:
        return self.P[pose_dimension:, pose_dimension:]

class EvaluateEKF:
    def __init__(self, gt_se23: list[SE23], gt_bias: np.ndarray, ekf_history: dict, exp_name: str):
        self.timestamps, self.states, self.covariances = ekf_history["pose"].get()
        self.timestamps_bias, self.bias, self.covariances_bias = ekf_history["bias"].get()
        
        self.gt_se23 = gt_se23
        self.gt_bias = gt_bias
        self.exp_name = exp_name
        
        self.pose_error = np.zeros((len(self.gt_se23), pose_dimension))
        for i in range(0, len(self.gt_se23)):
            self.pose_error[i, :] = SE23.Log(SE23.inverse(self.gt_se23[i]) @ self.states[i]).ravel()

    def plot_poses(self) -> None:
        pose_titles = [r"$\phi_x$", r"$\phi_y$", r"$\phi_z$",
                       r"$v_x$", r"$v_y$", r"$v_z$",
                       r"$x$", r"$y$", r"$z$"]
        
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle("Ground Truth vs. EKF Poses")
        
        gt = np.array([SE23.Log(pose).ravel() for pose in self.gt_se23])
        est = np.array([SE23.Log(pose).ravel() for pose in self.states])
        for i in range(0, pose_dimension):
            axs[i % 3, i // 3].plot(self.timestamps, gt[:, i], label="GT")
            axs[i % 3, i // 3].plot(self.timestamps, est[:, i], label="Est")
            axs[i % 3, i // 3].set_ylabel(pose_titles[i])
        axs[2, 0].set_xlabel("Time [s]")
        axs[2, 1].set_xlabel("Time [s]")
        axs[2, 2].set_xlabel("Time [s]")
        axs[0, 0].legend()
        
        if not os.path.exists('results/plots/ekf_imu_one_robot'):
            os.makedirs('results/plots/ekf_imu_one_robot')
        plt.savefig(f"results/plots/ekf_imu_one_robot/{self.exp_name}_poses.pdf")
        plt.close()

    def plot_error(self) -> None:
        error_titles = [r"$\delta_{\phi_x}$", r"$\delta_{\phi_y}$", r"$\delta_{\phi_z}$", 
                        r"$\delta_{v_x}$", r"$\delta_{v_y}$", r"$\delta_{v_z}$", 
                        r"$\delta_{x}$", r"$\delta_{y}$", r"$\delta_{z}$"]
        
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle("Three-Sigma Error Plots")
        
        for i in range(0, pose_dimension):
            axs[i % 3, i // 3].plot(self.timestamps, self.pose_error[:, i])
            axs[i % 3, i // 3].fill_between(
                self.timestamps,
                -1 * 3*np.sqrt(self.covariances[:, i, i]), 
                3*np.sqrt(self.covariances[:, i, i]), 
                alpha=0.5
            )
            axs[i % 3, i // 3].set_ylabel(error_titles[i])
        axs[2, 0].set_xlabel("Time [s]")
        axs[2, 1].set_xlabel("Time [s]")
        axs[2, 2].set_xlabel("Time [s]")
        
        if not os.path.exists('results/plots/ekf_imu_one_robot'):
            os.makedirs('results/plots/ekf_imu_one_robot')
        plt.savefig(f"results/plots/ekf_imu_one_robot/{self.exp_name}_error.pdf")
        plt.close()
        
    def plot_bias_error(self) -> None:
        bias_error_titles = [r"$\delta_{\beta_{\omega_x}}$", r"$\delta_{\beta_{\omega_y}}$", r"$\delta_{\beta_{\omega_z}}$", 
                             r"$\delta_{\beta_{a_x}}$", r"$\delta_{\beta_{a_y}}$", r"$\delta_{\beta_{a_z}}$"]
        
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))
        fig.suptitle("Three-Sigma Bias Error Plots")
        
        for i in range(0, bias_dimension):
            axs[i % 3, i // 3].plot(self.timestamps_bias, self.bias[:, i] - self.gt_bias[:, i])
            axs[i % 3, i // 3].fill_between(
                self.timestamps_bias,
                -1 * 3*np.sqrt(self.covariances_bias[:, i, i]), 
                3*np.sqrt(self.covariances_bias[:, i, i]), 
                alpha=0.5
            )
            axs[i % 3, i // 3].set_ylabel(bias_error_titles[i])
        axs[2, 0].set_xlabel("Time [s]")
        axs[2, 1].set_xlabel("Time [s]")
        
        if not os.path.exists('results/plots/ekf_imu_one_robot'):
            os.makedirs('results/plots/ekf_imu_one_robot')
        plt.savefig(f"results/plots/ekf_imu_one_robot/{self.exp_name}_bias_error.pdf")
        plt.close()
        

    def save_results(self) -> None:
        pos_rmse, att_rmse = self.get_rmse()
        
        myCsvRow = f"{self.exp_name},{pos_rmse},{att_rmse}\n"
        
        if not os.path.exists('results/ekf_imu_one_robot.csv'):
            with open('results/ekf_imu_one_robot.csv','w') as file:
                file.write("exp_name,pos_rmse,att_rmse\n")
        
        with open('results/ekf_imu_one_robot.csv','a') as file:
            file.write(myCsvRow)
                
    def get_rmse(self) -> tuple[float, float]:
        pos_rmse = np.sqrt(np.mean(self.pose_error[:, 6:] ** 2))
        att_rmse = np.sqrt(np.mean(self.pose_error[:, :3] ** 2))
        return pos_rmse, att_rmse
