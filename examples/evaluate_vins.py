# %%
from miluv.data import DataLoader
from miluv.utils import load_vins, align_frames, compute_position_rmse, save_vins, apply_transformation
import pandas as pd
import matplotlib.pyplot as plt
import sys
import yaml
from scipy.spatial.transform import Rotation

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
    results = align_frames(vins, df_mocap)
    vins = results["data"]
    frame_alignment = {
        "phi_vm": Rotation.from_matrix(results["C"]).as_rotvec().tolist(),
        "r_vm_m": results["r"].tolist(),
    }
    
    # Save frame_alignment to a yaml file
    with open(f"data/vins/{exp_name}/{robot_id}_alignment_pose.yaml", "w") as file:
        yaml.dump(frame_alignment, file)
    save_vins(vins, exp_name, robot_id, suffix="_aligned_and_shifted")
    
    rmse_loop = compute_position_rmse(vins, df_mocap)
    print(f"Position RMSE w Loop Closure for \Experiment {exp_name} \
                                        and Robot {robot_id}: {rmse_loop} m")
    
    # Apply transformation to vins without loop closure
    vins_no_loop = load_vins(exp_name, robot_id, loop=False)
    vins_no_loop = vins_no_loop[vins_no_loop["timestamp"] < vins_no_loop["timestamp"].iloc[-1] - 10]
    pos_no_loop = mv.data[robot_id]["mocap_pos"](vins_no_loop["timestamp"])
    quat_no_loop = mv.data[robot_id]["mocap_quat"](vins_no_loop["timestamp"])
    df_mocap_no_loop = pd.DataFrame({
        "timestamp": vins_no_loop["timestamp"],
        "pose.position.x": pos_no_loop[0],
        "pose.position.y": pos_no_loop[1],
        "pose.position.z": pos_no_loop[2],
        "pose.orientation.x": quat_no_loop[0],
        "pose.orientation.y": quat_no_loop[1],
        "pose.orientation.z": quat_no_loop[2],
        "pose.orientation.w": quat_no_loop[3],
    })
    vins_no_loop = apply_transformation(vins_no_loop, results["C"], results["r"])
    save_vins(vins_no_loop, exp_name, robot_id, loop=False, suffix="_aligned_and_shifted")
    rmse_no_loop = compute_position_rmse(vins_no_loop, df_mocap_no_loop)
    print(f"Position RMSE w/o Loop Closure for Experiment {exp_name} \
                                        and Robot {robot_id}: {rmse_no_loop} m")

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
        
    return {"rmse_loop": rmse_loop, "rmse_no_loop": rmse_no_loop}

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Not enough arguments. Usage: python evaluate_vins.py exp_name robot_id")
        sys.exit(1)
    exp_name = sys.argv[1]
    robot_id = sys.argv[2]
    
    if len(sys.argv) == 4:
        visualize = sys.argv[3]
    else:
        visualize = False
    
    evaluate_vins(exp_name, robot_id, visualize)

# %%
