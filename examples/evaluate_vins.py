# %%
from miluv.data import DataLoader
from miluv.utils import load_vins, align_frames, compute_position_rmse
import pandas as pd
import matplotlib.pyplot as plt
import sys

def evaluate_vins(exp_name, robot_id, visualize):
    # Read sensor and mocap data
    mv = DataLoader(exp_name, barometer=False, cir=False)

    # Read vins data
    vins = load_vins(exp_name, robot_id)

    # Drop the last 10 seconds of vins data
    vins = vins[vins["timestamp"] < vins["timestamp"].iloc[-1] - 10]

    # Get mocap data at vins timestamps
    pos = mv.data[robot_id]["mocap_pos"](vins["timestamp"])
    quat = mv.data[robot_id]["mocap_quat"](vins["timestamp"])

    # Align frame
    df_mocap = pd.DataFrame({
        "timestamp": vins["timestamp"],
        "pose.position.x": pos[0],
        "pose.position.y": pos[1],
        "pose.position.z": pos[2],
        "pose.orientation.x": quat[0],
        "pose.orientation.y": quat[1],
        "pose.orientation.z": quat[2],
        "pose.orientation.w": quat[3],
    })
    vins = align_frames(vins, df_mocap)
    
    rmse = compute_position_rmse(vins, df_mocap)
    print(f"Position RMSE for Experiment {exp_name} and Robot {robot_id}: {rmse} m")

    if visualize:
        # Compare vins and mocap data
        fig = plt.figure()
        ax = plt.axes(projection ='3d')
        ax.plot3D(vins["pose.position.x"], vins["pose.position.y"], vins["pose.position.z"], label="vins")
        ax.plot3D(pos[0], pos[1], pos[2], label="mocap")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.legend()
        ax.grid()   

        fig, axs = plt.subplots(3, 1)
        fig.suptitle("VINS vs. Mocap Position")
        axs[0].plot(vins["timestamp"], vins["pose.position.x"], label="vins")
        axs[0].plot(vins["timestamp"], pos[0], label="mocap")
        axs[0].set_ylabel("x [m]")
        axs[0].legend()
        axs[1].plot(vins["timestamp"], vins["pose.position.y"], label="vins")
        axs[1].plot(vins["timestamp"], pos[1], label="mocap")
        axs[1].set_ylabel("y [m]")
        axs[1].legend()
        axs[2].plot(vins["timestamp"], vins["pose.position.z"], label="vins")
        axs[2].plot(vins["timestamp"], pos[2], label="mocap")
        axs[2].set_ylabel("z [m]")
        axs[2].legend()
        plt.legend()
        axs[0].grid()
        axs[1].grid()
        axs[2].grid()

        fig, axs = plt.subplots(3, 1)
        fig.suptitle("VINS vs. Mocap Position Error")
        axs[0].plot(vins["timestamp"], vins["pose.position.x"] - pos[0], label="x")
        axs[0].set_ylabel("x [m]")
        axs[1].plot(vins["timestamp"], vins["pose.position.y"] - pos[1], label="y")
        axs[1].set_ylabel("y [m]")
        axs[2].plot(vins["timestamp"], vins["pose.position.z"] - pos[2], label="z")
        axs[2].set_ylabel("z [m]")
        plt.legend()
        axs[0].grid()
        axs[1].grid()
        axs[2].grid()

        plt.show(block=True)
        
    return rmse

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Not enough arguments. Usage: python vins.py exp_name robot_id")
        sys.exit(1)
    exp_name = sys.argv[1]
    robot_id = sys.argv[2]
    
    if len(sys.argv) == 4:
        visualize = sys.argv[3]
    else:
        visualize = False
    
    evaluate_vins(exp_name, robot_id, visualize)

# %%
