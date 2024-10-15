import os
import numpy as np
import matplotlib.pyplot as plt

from pymlg import SE3

class StateHistory:
    def __init__(self):
        self.timestamps = np.empty(0)
        self.states = np.empty((0, 4, 4))
        self.covariances = np.empty((0, 6, 6))

    def add(self, timestamp: float, state: np.ndarray, covariance: np.ndarray) -> None:
        self.timestamps = np.append(self.timestamps, timestamp)
        self.states = np.append(self.states, state.reshape(1, 4, 4), axis=0)
        self.covariances = np.append(self.covariances, covariance.reshape(1, 6, 6), axis=0)

    def get(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.timestamps, self.states, self.covariances
    
class EvaluateEKF:
    def __init__(self, gt_se3: list[SE3], ekf_history: StateHistory, exp_name: str):
        self.timestamps, self.states, self.covariances = ekf_history.get()
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
        
        
        