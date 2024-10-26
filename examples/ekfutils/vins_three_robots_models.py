import os
import numpy as np
import matplotlib.pyplot as plt
from pymlg import SE3
import scipy as sp

import examples.ekfutils.common as common
import miluv.utils as utils

np.random.seed(0)

# EKF parameters
robot_names = ["ifo001", "ifo002", "ifo003"]
num_robots = len(robot_names)

single_robot_state_dimension = 6
full_state_dimension = num_robots * single_robot_state_dimension

# This dictionary maps robot names to the start and end indices of their state in the full state vector
x_idx = {
    robot: {"start": 6 * i, "end": 6 * (i + 1)} for i, robot in enumerate(robot_names)
}

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
    def __init__(self, state: dict[str: SE3], anchors: dict[int: np.ndarray], tag_moment_arms: dict[str: dict[int: np.ndarray]]):
        # Add noise to the initial state using P0 to reflect uncertainty in the initial state
        self.x = {
            robot: state[robot] @ SE3.Exp(np.random.multivariate_normal(np.zeros(single_robot_state_dimension), P0)) 
            for robot in robot_names
        }
        
        self.P = sp.linalg.block_diag(*[P0 for _ in range(num_robots)])
        self.anchors = anchors
        self.tag_moment_arms = tag_moment_arms

    def predict(self, u: dict[str: np.ndarray], dt: float) -> None:
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
        
        if common.is_outlier(z, S):
            return
        
        K = self.P @ H.T / S
        
        for robot in robot_names:
            K_robot = K[x_idx[robot]["start"] : x_idx[robot]["end"]]
            self.x[robot] = self.x[robot] @ SE3.Exp(K_robot @ z)
            
        self.P = (np.eye(full_state_dimension) - K @ H) @ self.P
        
        # Ensure symmetric covariance matrix
        self.P = 0.5 * (self.P + self.P.T)
      
    def get_covariance(self, robot: str) -> np.ndarray:
        return self.P[
            x_idx[robot]["start"] : x_idx[robot]["end"], 
            x_idx[robot]["start"] : x_idx[robot]["end"]
        ]
        
    @staticmethod
    def _process_model(x: dict[str: SE3], u: dict[str: np.ndarray], dt: float) -> dict[str: SE3]:
        def subproc(robot: str) -> SE3:
            return x[robot] @ SE3.Exp(u[robot] * dt)
        return {robot: subproc(robot) for robot in robot_names}
    
    @staticmethod
    def _process_jacobian(x: dict[str: SE3], u: dict[str: np.ndarray], dt: float) -> np.ndarray:
        def subjac(robot: str) -> np.ndarray:
            return SE3.adjoint(np.linalg.inv(SE3.Exp(u[robot] * dt)))
        return sp.linalg.block_diag(*[subjac(robot) for robot in robot_names])
    
    @staticmethod
    def _process_covariance(x: dict[str: SE3], u: dict[str: np.ndarray], dt: float) -> np.ndarray:
        def subcov(robot: str) -> np.ndarray:
            return SE3.left_jacobian(-dt * u[robot]) @ Q @ SE3.left_jacobian(-dt * u[robot]).T
        return dt * sp.linalg.block_diag(*[subcov(robot) for robot in robot_names])
    
    def _range_measurement_w_anchor(self, x: dict[str: SE3], anchor_id: int, tag_id: int) -> float:
        robot = common.get_robot_from_tag(tag_id, self.tag_moment_arms)
        
        Pi = np.hstack([np.eye(3), np.zeros((3, 1))])
        r_tilde = np.vstack((self.tag_moment_arms[robot][tag_id], 1)).reshape(4, 1)
        
        return np.linalg.norm(self.anchors[anchor_id] - Pi @ x[robot] @ r_tilde)
    
    def _range_jacobian_w_anchor(self, x: dict[str: SE3], anchor_id: int, tag_id: int) -> np.ndarray:
        robot = common.get_robot_from_tag(tag_id, self.tag_moment_arms)
        
        Pi = np.hstack([np.eye(3), np.zeros((3, 1))])
        r_tilde = np.vstack((self.tag_moment_arms[robot][tag_id], 1)).reshape(4, 1)
        
        vector = (self.anchors[anchor_id] - Pi @ x[robot] @ r_tilde).reshape(3, 1)
        
        jac = np.zeros((1, full_state_dimension))
        
        start_idx = x_idx[robot]["start"]
        end_idx = x_idx[robot]["end"]
        jac[0, start_idx : end_idx] = -1 * vector.T / np.linalg.norm(vector) @ Pi @ x[robot] @ SE3.odot(r_tilde)
            
        return jac
    
    def _range_measurement_between_tags(self, x: dict[str: SE3], to_id: int, from_id: int) -> float:
        robot_to = common.get_robot_from_tag(to_id, self.tag_moment_arms)
        robot_from = common.get_robot_from_tag(from_id, self.tag_moment_arms)
        
        Pi = np.hstack([np.eye(3), np.zeros((3, 1))])
        r_tilde_to = np.vstack((self.tag_moment_arms[robot_to][to_id], 1)).reshape(4, 1)
        r_tilde_from = np.vstack((self.tag_moment_arms[robot_from][from_id], 1)).reshape(4, 1)
        
        return np.linalg.norm(Pi @ (x[robot_to] @ r_tilde_to - x[robot_from] @ r_tilde_from))
    
    def _range_jacobian_between_tags(self, x: dict[str: SE3], to_id: int, from_id: int) -> np.ndarray:
        robot_to = common.get_robot_from_tag(to_id, self.tag_moment_arms)
        robot_from = common.get_robot_from_tag(from_id, self.tag_moment_arms)
        
        Pi = np.hstack([np.eye(3), np.zeros((3, 1))])
        r_tilde_to = np.vstack((self.tag_moment_arms[robot_to][to_id], 1)).reshape(4, 1)
        r_tilde_from = np.vstack((self.tag_moment_arms[robot_from][from_id], 1)).reshape(4, 1)
        
        vector = (Pi @ (x[robot_to] @ r_tilde_to - x[robot_from] @ r_tilde_from)).reshape(3, 1)
        
        jac = np.zeros((1, full_state_dimension))
        
        start_idx = x_idx[robot_to]["start"]
        end_idx = x_idx[robot_from]["end"]
        jac[0, start_idx : end_idx] = vector.T / np.linalg.norm(vector) @ Pi @ x[robot_to] @ SE3.odot(r_tilde_to)
        
        start_idx = x_idx[robot_from]["start"]
        end_idx = x_idx[robot_from]["end"]
        jac[0, start_idx : end_idx] = -1 * vector.T / np.linalg.norm(vector) @ Pi @ x[robot_from] @ SE3.odot(r_tilde_from)
            
        return jac        
    
    @staticmethod
    def _height_measurement(x: dict[str: SE3], robot: str) -> float:
        return x[robot][2, 3]
    
    @staticmethod
    def _height_jacobian(x: dict[str: SE3], robot: str) -> np.ndarray:
        a = np.array([0, 0, 1, 0]).reshape(4, 1)
        b = np.array([0, 0, 0, 1])
        
        jac = np.zeros((1, full_state_dimension))
        jac[0, x_idx[robot]["start"] : x_idx[robot]["end"]] = a.T @ x[robot] @ SE3.odot(b)
        return jac

class EvaluateEKF:
    def __init__(self, gt_se3: dict[str: SE3], ekf_history: dict[str: common.MatrixStateHistory], exp_name: str):        
        self.timestamps = {}
        self.states = {}
        self.covariances = {}
        self.gt_se3 = {}
        self.error = {}
        
        for robot in robot_names:
            self.timestamps[robot], self.states[robot], self.covariances[robot] = ekf_history[robot].get()
            self.gt_se3[robot] = gt_se3[robot]
            
            self.error[robot] = np.zeros((len(self.gt_se3[robot]), 6))
            for i in range(0, len(self.gt_se3[robot])):
                self.error[robot][i, :] = SE3.Log(SE3.inverse(self.gt_se3[robot][i]) @ self.states[robot][i]).ravel()

        self.exp_name = exp_name
        self.error_titles = [r"$\delta_{\phi_x}$", r"$\delta_{\phi_y}$", r"$\delta_{\phi_z}$", 
                             r"$\delta_{x}$", r"$\delta_{y}$", r"$\delta_{z}$"]

    def plot_poses(self) -> None:
        for robot in robot_names:
            fig, axs = plt.subplots(3, 2, figsize=(10, 10))
            fig.suptitle("Ground Truth vs. EKF Poses for " + robot)
            
            gt = np.array([SE3.Log(pose).ravel() for pose in self.gt_se3[robot]])
            est = np.array([SE3.Log(pose).ravel() for pose in self.states[robot]])
            for i in range(0, 6):
                axs[i % 3, int(i > 2)].plot(self.timestamps[robot], gt[:, i], label="GT")
                axs[i % 3, int(i > 2)].plot(self.timestamps[robot], est[:, i], label="Est")
                axs[i % 3, int(i > 2)].set_ylabel(self.error_titles[i])
            axs[2, 0].set_xlabel("Time [s]")
            axs[2, 1].set_xlabel("Time [s]")
            axs[0, 0].legend()
            
            if not os.path.exists('results/plots/ekf_vins_three_robots'):
                os.makedirs('results/plots/ekf_vins_three_robots')
            plt.savefig(f"results/plots/ekf_vins_three_robots/{self.exp_name}_poses_{robot}.pdf")
            plt.close()

    def plot_error(self) -> None:
        for robot in robot_names:
            fig, axs = plt.subplots(3, 2, figsize=(10, 10))
            fig.suptitle("Three-Sigma Error Plots for " + robot)
            
            for i in range(0, 6):
                axs[i % 3, int(i > 2)].plot(self.timestamps[robot], self.error[robot][:, i])
                axs[i % 3, int(i > 2)].fill_between(
                    self.timestamps[robot],
                    -1 * 3*np.sqrt(self.covariances[robot][:, i, i]), 
                    3*np.sqrt(self.covariances[robot][:, i, i]), 
                    alpha=0.5
                )
                axs[i % 3, int(i > 2)].set_ylabel(self.error_titles[i])
            axs[2, 0].set_xlabel("Time [s]")
            axs[2, 1].set_xlabel("Time [s]")
            
            if not os.path.exists('results/plots/ekf_vins_three_robots'):
                os.makedirs('results/plots/ekf_vins_three_robots')
            plt.savefig(f"results/plots/ekf_vins_three_robots/{self.exp_name}_error_{robot}.pdf")
            plt.close()

    def save_results(self) -> None:
        pos_rmse = {}
        att_rmse = {}
        for robot in robot_names:        
            pos_rmse[robot], att_rmse[robot] = self.get_rmse(robot)
        
        myCsvRow = f"{self.exp_name}," + ",".join([f"{pos_rmse[robot]},{att_rmse[robot]}" for robot in robot_names]) + "\n"
        
        if not os.path.exists('results/ekf_vins_three_robots.csv'):
            with open('results/ekf_vins_three_robots.csv','w') as file:
                file.write("exp_name," + ",".join([f"{robot}_pos_rmse,{robot}_att_rmse" for robot in robot_names]) + "\n")
        
        with open('results/ekf_vins_three_robots.csv','a') as file:
            file.write(myCsvRow)
                
    def get_rmse(self, robot: str) -> tuple[float, float]:
        pos_rmse = np.sqrt(np.mean(self.error[robot][:, 3:] ** 2))
        att_rmse = np.sqrt(np.mean(self.error[robot][:, :3] ** 2))
        return pos_rmse, att_rmse