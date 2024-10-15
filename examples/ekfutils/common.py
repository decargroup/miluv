import numpy as np
from scipy.stats import chi2
import pandas as pd

from pymlg import SE3

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
