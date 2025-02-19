import os
import numpy as np
import matplotlib.pyplot as plt
from pymlg import SE23
import scipy as sp
from dataclasses import dataclass

import examples.ekfutils.imu_models as imu_models
import examples.ekfutils.common as common
import miluv.utils as utils

np.random.seed(0)

# EKF parameters
robot_names = ["ifo001", "ifo002", "ifo003"]
num_robots = len(robot_names)

pose_dimension = 9
bias_dimension = 6
single_robot_state_dimension = pose_dimension + bias_dimension
full_state_dimension = num_robots * single_robot_state_dimension

# This dictionary maps robot names to the start and end indices of their state in the full state vector
x_idx = {
    robot: {
        "pose_start": single_robot_state_dimension * i, 
        "pose_end": single_robot_state_dimension * i + pose_dimension,
        "bias_start": single_robot_state_dimension * i + pose_dimension,
        "bias_end": single_robot_state_dimension * (i + 1)
    } for i, robot in enumerate(robot_names)
}

# Covariance matrices
P0 = np.diag([
    0.01, 0.01, 0.01, # Orientation
    0.01, 0.01, 0.01, # Velocity
    0.1, 0.1, 0.1,    # Position
    0.01, 0.01, 0.01, # Gyro bias
    0.1, 0.1, 0.1,    # Accel bias
]) # Initial state covariance

imu_noise_params = {robot: utils.get_imu_noise_params(robot, "px4") for robot in robot_names}
Q = {robot: np.diag([
    imu_noise_params[robot]["gyro"][0]**2, 
    imu_noise_params[robot]["gyro"][1]**2,
    imu_noise_params[robot]["gyro"][2]**2,
    imu_noise_params[robot]["accel"][0]**2,
    imu_noise_params[robot]["accel"][1]**2,
    imu_noise_params[robot]["accel"][2]**2,
    imu_noise_params[robot]["gyro_bias"][0]**2,
    imu_noise_params[robot]["gyro_bias"][1]**2,
    imu_noise_params[robot]["gyro_bias"][2]**2,
    imu_noise_params[robot]["accel_bias"][0]**2,
    imu_noise_params[robot]["accel_bias"][1]**2,
    imu_noise_params[robot]["accel_bias"][2]**2
]) for robot in robot_names} # Process noise covariance

R_range = 0.3**2 # Range measurement noise covariance
R_height = 0.5**2 # Height measurement noise covariance

@dataclass
class State:
    pose: SE23
    bias: np.ndarray

class EKF:
    def __init__(self, state: dict[str: SE23], anchors: dict[int: np.ndarray], tag_moment_arms: dict[str: dict[int: np.ndarray]]):
        # Add noise to the initial state using P0 to reflect uncertainty in the initial state
        self.x = {robot: State(
            pose=state[robot] @ SE23.Exp(np.random.multivariate_normal(
                np.zeros(pose_dimension), P0[:pose_dimension, :pose_dimension]
            )),
            bias=np.zeros(bias_dimension),
        ) for robot in robot_names
        }
        
        self.P = sp.linalg.block_diag(*[P0 for _ in range(num_robots)])
        self.anchors = anchors
        self.tag_moment_arms = tag_moment_arms

    def predict(self, u: dict[str: np.ndarray], dt: float) -> None:
        if dt == 0:
            return
        A = self._process_jacobian(self.x, u, dt)
        Qd = self._process_covariance(self.x, u, dt)
        
        self.x = self._process_model(self.x, u, dt)
        self.P = A @ self.P @ A.T + Qd
    
    def correct(self, y: dict) -> None:
        if "range" in y:
            to_id = y["to_id"]
            from_id = y["from_id"]
            
            if to_id in self.anchors: # note that anchors cannot initiate a range measurement in our setup
                H = self._range_jacobian_w_anchor(self.x, to_id, from_id)
                predicted_measurement = self._range_measurement_w_anchor(self.x, to_id, from_id)
            else:
                H = self._range_jacobian_between_tags(self.x, to_id, from_id)
                predicted_measurement = self._range_measurement_between_tags(self.x, to_id, from_id)
            
            R = R_range
            actual_measurement = y["range"]
        elif "height" in y:
            H = self._height_jacobian(self.x, y["robot"])
            R = R_height
            actual_measurement = y["height"]
            predicted_measurement = self._height_measurement(self.x, y["robot"])
        else:
            return
        
        z = np.array([actual_measurement - predicted_measurement])
        S = H @ self.P @ H.T + R
        
        K = self.P @ H.T / S
        
        for robot in robot_names:
            pose_start = x_idx[robot]["pose_start"]
            pose_end = x_idx[robot]["pose_end"]
            bias_start = x_idx[robot]["bias_start"]
            bias_end = x_idx[robot]["bias_end"]
            
            self.x[robot].pose = self.x[robot].pose @ SE23.Exp(K[pose_start:pose_end, :] @ z)
            self.x[robot].bias = self.x[robot].bias + K[bias_start:bias_end, :] @ z
        
        self.P = (np.eye(full_state_dimension) - K @ H) @ self.P
        
        # Ensure symmetric covariance matrix
        self.P = 0.5 * (self.P + self.P.T)
        
    @staticmethod
    def _process_model(x: dict[str: State], u: dict[str: np.ndarray], dt: float) -> dict[str: State]:
        def subproc(robot: str) -> State:
            return State(
                pose=imu_models.G_mat(dt) @ x[robot].pose @ imu_models.U_mat(x[robot].bias, u[robot], dt),
                bias=x[robot].bias
            )
        return {robot: subproc(robot) for robot in robot_names}
    
    @staticmethod
    def _process_jacobian(x: dict[str: State], u: dict[str: np.ndarray], dt: float) -> np.ndarray:
        # TODO: replace all these subfunctions with the one from the single robot model
        def subjac(robot: str) -> np.ndarray:
            jac = np.zeros((single_robot_state_dimension, single_robot_state_dimension))
            jac[:pose_dimension, :pose_dimension] = imu_models.U_adjoint_mat(imu_models.U_inv_mat(x[robot].bias, u[robot], dt))
            jac[:pose_dimension, pose_dimension:] = -imu_models.L_mat(x[robot].bias, u[robot], dt)
            jac[pose_dimension:, pose_dimension:] = np.eye(bias_dimension)

            return jac
        
        return sp.linalg.block_diag(*[subjac(robot) for robot in robot_names])
    
    @staticmethod
    def _process_covariance(x: dict[str: State], u: dict[str: np.ndarray], dt: float) -> np.ndarray:
        def subcov(robot: str) -> np.ndarray:
            noise_jac = np.zeros((single_robot_state_dimension, Q[robot].shape[0]))
            noise_jac[:pose_dimension, :bias_dimension] = imu_models.L_mat(x[robot].bias, u[robot], dt)
            noise_jac[pose_dimension:, bias_dimension:] = np.eye(bias_dimension)
            
            return (noise_jac @ Q[robot] @ noise_jac.T) / dt
        
        return sp.linalg.block_diag(*[subcov(robot) for robot in robot_names])
    
    def _range_measurement_w_anchor(self, x: dict[str: State], anchor_id: int, tag_id: int) -> float:
        robot = common.get_robot_from_tag(tag_id, self.tag_moment_arms)
        
        Pi = np.hstack([np.eye(3), np.zeros((3, 2))])
        r_tilde = np.vstack((self.tag_moment_arms[robot][tag_id], 0, 1)).reshape(5, 1)
        
        return np.linalg.norm(self.anchors[anchor_id] - Pi @ x[robot].pose @ r_tilde)
    
    def _range_jacobian_w_anchor(self, x: dict[str: State], anchor_id: int, tag_id: int) -> np.ndarray:
        robot = common.get_robot_from_tag(tag_id, self.tag_moment_arms)
        
        Pi = np.hstack([np.eye(3), np.zeros((3, 2))])
        r_tilde = np.vstack((self.tag_moment_arms[robot][tag_id], 0, 1)).reshape(5, 1)
        
        vector = (self.anchors[anchor_id] - Pi @ x[robot].pose @ r_tilde).reshape(3, 1)
        
        jac = np.zeros((1, full_state_dimension))
        
        start_idx = x_idx[robot]["pose_start"]
        end_idx = x_idx[robot]["pose_end"]
        jac[0, start_idx : end_idx] = -1 * vector.T / np.linalg.norm(vector) @ Pi @ x[robot].pose @ SE23.odot(r_tilde)
            
        return jac
    
    def _range_measurement_between_tags(self, x: dict[str: State], to_id: int, from_id: int) -> float:
        robot_to = common.get_robot_from_tag(to_id, self.tag_moment_arms)
        robot_from = common.get_robot_from_tag(from_id, self.tag_moment_arms)
        
        Pi = np.hstack([np.eye(3), np.zeros((3, 2))])
        r_tilde_to = np.vstack((self.tag_moment_arms[robot_to][to_id], 0, 1)).reshape(5, 1)
        r_tilde_from = np.vstack((self.tag_moment_arms[robot_from][from_id], 0, 1)).reshape(5, 1)
        
        return np.linalg.norm(Pi @ (x[robot_to].pose @ r_tilde_to - x[robot_from].pose @ r_tilde_from))
    
    def _range_jacobian_between_tags(self, x: dict[str: State], to_id: int, from_id: int) -> np.ndarray:
        robot_to = common.get_robot_from_tag(to_id, self.tag_moment_arms)
        robot_from = common.get_robot_from_tag(from_id, self.tag_moment_arms)
        
        Pi = np.hstack([np.eye(3), np.zeros((3, 2))])
        r_tilde_to = np.vstack((self.tag_moment_arms[robot_to][to_id], 0, 1)).reshape(5, 1)
        r_tilde_from = np.vstack((self.tag_moment_arms[robot_from][from_id], 0, 1)).reshape(5, 1)
        
        vector = (Pi @ (x[robot_to].pose @ r_tilde_to - x[robot_from].pose @ r_tilde_from)).reshape(3, 1)
        
        jac = np.zeros((1, full_state_dimension))
        
        start_idx = x_idx[robot_to]["pose_start"]
        end_idx = x_idx[robot_to]["pose_end"]
        jac[0, start_idx : end_idx] = vector.T / np.linalg.norm(vector) @ Pi @ x[robot_to].pose @ SE23.odot(r_tilde_to)
        
        start_idx = x_idx[robot_from]["pose_start"]
        end_idx = x_idx[robot_from]["pose_end"]
        jac[0, start_idx : end_idx] = -1 * vector.T / np.linalg.norm(vector) @ Pi @ x[robot_from].pose @ SE23.odot(r_tilde_from)
            
        return jac
    
    @staticmethod
    def _height_measurement(x: dict[str: State], robot: str) -> float:
        return x[robot].pose[2, 4]
    
    @staticmethod
    def _height_jacobian(x: dict[str: State], robot: str) -> np.ndarray:
        a = np.array([0, 0, 1, 0, 0]).reshape(5, 1)
        b = np.array([0, 0, 0, 0, 1])
        
        jac = np.zeros((1, full_state_dimension))
        jac[:, x_idx[robot]["pose_start"]:x_idx[robot]["pose_end"]] = a.T @ x[robot].pose @ SE23.odot(b)
        return jac
    
    @property
    def pose(self) -> dict[str: SE23]:
        return {robot: self.x[robot].pose for robot in robot_names}
    
    @property
    def pose_covariance(self) -> dict[str: np.ndarray]:
        return {robot: self.P[
            x_idx[robot]["pose_start"]:x_idx[robot]["pose_end"], x_idx[robot]["pose_start"]:x_idx[robot]["pose_end"]
        ] for robot in robot_names}
    
    @property
    def bias(self) -> dict[str: np.ndarray]:
        return {robot: self.x[robot].bias for robot in robot_names}
    
    @property
    def bias_covariance(self) -> dict[str: np.ndarray]:
        return {robot: self.P[
            x_idx[robot]["bias_start"]:x_idx[robot]["bias_end"], x_idx[robot]["bias_start"]:x_idx[robot]["bias_end"]
        ] for robot in robot_names}

class EvaluateEKF:
    def __init__(
        self, 
        gt_se23: dict[str: SE23], 
        gt_bias: dict[str: np.ndarray],
        ekf_history: dict[str: dict], 
        exp_name: str
    ):
        self.timestamps = {}
        self.states = {}
        self.covariances = {}
        self.timestamps_bias = {}
        self.bias = {}
        self.covariances_bias = {}
        self.gt_se23 = {}
        self.gt_bias = {}
        self.pose_error = {}
        
        for robot in robot_names:
            self.timestamps[robot], self.states[robot], self.covariances[robot] = ekf_history[robot]["pose"].get()
            self.timestamps_bias[robot], self.bias[robot], self.covariances_bias[robot] = ekf_history[robot]["bias"].get()
            self.gt_se23[robot] = gt_se23[robot]
            self.gt_bias[robot] = gt_bias[robot]
            
            self.pose_error[robot] = np.zeros((len(self.gt_se23[robot]), pose_dimension))
            for i in range(0, len(self.gt_se23[robot])):
                self.pose_error[robot][i, :] = SE23.Log(SE23.inverse(self.gt_se23[robot][i]) @ self.states[robot][i]).ravel()

        self.exp_name = exp_name

    def plot_poses(self) -> None:
        pose_titles = [r"$\phi_x$", r"$\phi_y$", r"$\phi_z$",
                       r"$v_x$", r"$v_y$", r"$v_z$",
                       r"$x$", r"$y$", r"$z$"]
        
        for robot in robot_names:
            fig, axs = plt.subplots(3, 3, figsize=(10, 10))
            fig.suptitle("Ground Truth vs. EKF Poses " + robot)
            
            gt = np.array([SE23.Log(pose).ravel() for pose in self.gt_se23[robot]])
            est = np.array([SE23.Log(pose).ravel() for pose in self.states[robot]])
            for i in range(0, pose_dimension):
                axs[i % 3, i // 3].plot(self.timestamps[robot], gt[:, i], label="GT")
                axs[i % 3, i // 3].plot(self.timestamps[robot], est[:, i], label="Est")
                axs[i % 3, i // 3].set_ylabel(pose_titles[i])
            axs[2, 0].set_xlabel("Time [s]")
            axs[2, 1].set_xlabel("Time [s]")
            axs[2, 2].set_xlabel("Time [s]")
            axs[0, 0].legend()
            
            if not os.path.exists('results/plots/ekf_imu_three_robots'):
                os.makedirs('results/plots/ekf_imu_three_robots')
            plt.savefig(f"results/plots/ekf_imu_three_robots/{self.exp_name}_poses_{robot}.pdf")
            plt.close()

    def plot_error(self) -> None:
        error_titles = [r"$\delta_{\phi_x}$", r"$\delta_{\phi_y}$", r"$\delta_{\phi_z}$", 
                        r"$\delta_{v_x}$", r"$\delta_{v_y}$", r"$\delta_{v_z}$", 
                        r"$\delta_{x}$", r"$\delta_{y}$", r"$\delta_{z}$"]
        
        for robot in robot_names:
            fig, axs = plt.subplots(3, 3, figsize=(10, 10))
            fig.suptitle("Three-Sigma Error Plots " + robot)
            
            for i in range(0, pose_dimension):
                axs[i % 3, i // 3].plot(self.timestamps[robot], self.pose_error[robot][:, i])
                axs[i % 3, i // 3].fill_between(
                    self.timestamps[robot],
                    -1 * 3*np.sqrt(self.covariances[robot][:, i, i]), 
                    3*np.sqrt(self.covariances[robot][:, i, i]), 
                    alpha=0.5
                )
                axs[i % 3, i // 3].set_ylabel(error_titles[i])
            axs[2, 0].set_xlabel("Time [s]")
            axs[2, 1].set_xlabel("Time [s]")
            axs[2, 2].set_xlabel("Time [s]")
            
            if not os.path.exists('results/plots/ekf_imu_three_robots'):
                os.makedirs('results/plots/ekf_imu_three_robots')
            plt.savefig(f"results/plots/ekf_imu_three_robots/{self.exp_name}_error_{robot}.pdf")
            plt.close()
            
    def plot_bias_error(self) -> None:
        bias_error_titles = [r"$\delta_{\beta_{\omega_x}}$", r"$\delta_{\beta_{\omega_y}}$", r"$\delta_{\beta_{\omega_z}}",
                             r"$\delta_{\beta_{a_x}}$", r"$\delta_{\beta_{a_y}}$", r"$\delta_{\beta_{a_z}}"]
        
        for robot in robot_names:
            fig, axs = plt.subplots(3, 2, figsize=(10, 10))
            fig.suptitle("Three-Sigma Bias Error Plots " + robot)
            
            for i in range(0, bias_dimension):
                axs[i % 3, i // 3].plot(self.timestamps_bias[robot], self.bias[robot][:, i] - self.gt_bias[robot][:, i])
                axs[i % 3, i // 3].fill_between(
                    self.timestamps_bias[robot],
                    -1 * 3*np.sqrt(self.covariances_bias[robot][:, i, i]), 
                    3*np.sqrt(self.covariances_bias[robot][:, i, i]), 
                    alpha=0.5
                )
                axs[i % 3, i // 3].set_ylabel(bias_error_titles[i])
            axs[2, 0].set_xlabel("Time [s]")
            axs[2, 1].set_xlabel("Time [s]")
            
            if not os.path.exists('results/plots/ekf_imu_three_robots'):
                os.makedirs('results/plots/ekf_imu_three_robots')
            plt.savefig(f"results/plots/ekf_imu_three_robots/{self.exp_name}_bias_error_{robot}.pdf")
            plt.close()

    def save_results(self) -> None:
        pos_rmse = {}
        att_rmse = {}
        for robot in robot_names:
            pos_rmse[robot], att_rmse[robot] = self.get_rmse(robot)
        
        myCsvRow = f"{self.exp_name}," + ",".join([f"{pos_rmse[robot]},{att_rmse[robot]}" for robot in robot_names]) + "\n"
        
        if not os.path.exists('results/ekf_imu_three_robots.csv'):
            with open('results/ekf_imu_three_robots.csv','w') as file:
                file.write("exp_name," + ",".join([f"{robot}_pos_rmse,{robot}_att_rmse" for robot in robot_names]) + "\n")
        
        with open('results/ekf_imu_three_robots.csv','a') as file:
            file.write(myCsvRow)
                
    def get_rmse(self, robot: str) -> tuple[float, float]:
        pos_rmse = np.sqrt(np.mean(self.pose_error[robot][:, 6:] ** 2))
        att_rmse = np.sqrt(np.mean(self.pose_error[robot][:, :3] ** 2))
        return pos_rmse, att_rmse
