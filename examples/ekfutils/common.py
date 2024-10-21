import numpy as np
from scipy.stats import chi2
import pandas as pd

from pymlg import SE3

def get_robot_from_tag(tag_id: int, tag_moment_arms: dict[str, dict[int, np.ndarray]]) -> str:
    """
    Get the robot name from a tag ID.
    
    Args:
    - tag_id: The tag ID.
    - tag_moment_arms: Tag moment arms separated by robot.
    
    Returns:
    str
    - The robot name.
    """
    for robot in tag_moment_arms:
        if tag_id in tag_moment_arms[robot]:
            return robot
    return None

def is_outlier(error: np.ndarray, covariance: np.ndarray) -> bool:
    """
    Perform a normalized innovation squared (NIS) test on the given error and covariance.
    This is essentially a chi-squared test in the context of filtering. More details can be found
    in the book "Estimation with Applications to Tracking and Navigation" by Bar-Shalom et al.
    in Section 5.4.2.
    
    Args:
    - error: The error vector to test.
    - covariance: The covariance matrix of the error vector.
    
    Returns:
    bool
    - True if the NIS test passes and the error is within the 99% confidence interval.
    """
    error = error.reshape((-1, 1))
    nis = np.ndarray.item(error.T @ np.linalg.solve(covariance, error))
    return (nis > chi2.ppf(0.99, df=error.size))

def convert_vins_velocity_to_body_frame(vins: pd.DataFrame, gt_se3: list[SE3]) -> pd.DataFrame:
    """
    Convert VINS velocity data to the robot's body frame.
    
    Args:
    - vins: VINS data.
    - gt_se3: Ground truth SE3 poses.
    
    Returns:
    pd.DataFrame
    - VINS data with velocity data in the robot's body frame.
    """
    
    for i in range(len(vins)):
        R = gt_se3[i][:3, :3]
        v = np.array([vins.iloc[i]["twist.linear.x"],
                      vins.iloc[i]["twist.linear.y"],
                      vins.iloc[i]["twist.linear.z"]])
        v_body = R.T @ v
        vins.at[i, "twist.linear.x"] = v_body[0]
        vins.at[i, "twist.linear.y"] = v_body[1]
        vins.at[i, "twist.linear.z"] = v_body[2]
    
    return vins

class MatrixStateHistory:
    """
    A class to store the history of the state and covariance at each timestamp, 
    when the state is represented using a matrix.
    """
    def __init__(self, state_dim: int, covariance_dim: int) -> None:
        """
        Constructor for the StateHistory class.
        
        Args:
        - state_dim: The dimension of the state matrix.
        """
        self.state_dim = state_dim
        self.coveriance_dim = covariance_dim
        
        self.timestamps = np.empty(0)
        self.states = np.empty((0, self.state_dim, self.state_dim))
        self.covariances = np.empty((0, self.coveriance_dim, self.coveriance_dim))

    def add(self, timestamp: float, state: np.ndarray, covariance: np.ndarray) -> None:
        """
        Add a timestamped state and covariance to the history.
        
        Args:
        - timestamp: The timestamp of the state and covariance.
        - state: The state matrix.
        - covariance: The covariance matrix of the state vector.
        """
        self.timestamps = np.append(self.timestamps, timestamp)
        self.states = np.append(
            self.states, 
            state.reshape(1, self.state_dim, self.state_dim), 
            axis=0
        )
        self.covariances = np.append(
            self.covariances, 
            covariance.reshape(1, self.coveriance_dim, self.coveriance_dim), 
            axis=0,
        )

    def get(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the timestamps, states, and covariances stored in the history.

        Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]
        - The timestamps, states, and covariances.
        """
        return self.timestamps, self.states, self.covariances
    
class VectorStateHistory:
    """
    A class to store the history of the state and covariance at each timestamp,
    when the state is represented using a vector.
    """
    def __init__(self, state_dim: int) -> None:
        """
        Constructor for the StateHistory class.
        
        Args:
        - state_dim: The dimension of the state vector.
        """
        self.state_dim = state_dim
        
        self.timestamps = np.empty(0)
        self.states = np.empty((0, self.state_dim))
        self.covariances = np.empty((0, self.state_dim, self.state_dim))

    def add(self, timestamp: float, state: np.ndarray, covariance: np.ndarray) -> None:
        """
        Add a timestamped state and covariance to the history.
        
        Args:
        - timestamp: The timestamp of the state and covariance.
        - state: The state vector.
        - covariance: The covariance matrix of the state vector.
        """
        self.timestamps = np.append(self.timestamps, timestamp)
        self.states = np.append(self.states, state.reshape(1, self.state_dim), axis=0)
        self.covariances = np.append(
            self.covariances, 
            covariance.reshape(1, self.state_dim, self.state_dim), 
            axis=0,
        )

    def get(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the timestamps, states, and covariances stored in the history.

        Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]
        - The timestamps, states, and covariances.
        """
        return self.timestamps, self.states, self.covariances