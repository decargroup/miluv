import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from typing import List, Tuple, Any
from utils.states import StateWithCovariance
from miluv.data import DataLoader
from utils.meas_models import (
    Measurement,
    RangePoseToAnchorById, 
    RangePoseToPose,
    AltitudeById,
)
from utils.states import IMUState
from pymlg import SE23, SO3

""" Functions related to storing results into lists """
class GaussianResult:
    """
    A data container that simultaneously computes the error and three-sigma
    bounds of a state estimate.
    """
    def __init__( self, estimate: StateWithCovariance, state_true,):
        self.stamp = estimate.state.stamp
        self.state = estimate.state
        self.state_true = state_true
        self.covariance = estimate.covariance
        self.error = state_true.minus(estimate.state).ravel()
        self.three_sigma = 3 * np.sqrt(np.diag(estimate.covariance))

class GaussianResultList:
    """
    Makes a list of results out of List[GaussianResult] objects.
    """
    def __init__(self, result_list: List[GaussianResult]):
        for attr in ['stamp', 'state', 'state_true', 
                     'covariance', 'error', 'three_sigma']:
            setattr(self, attr, np.array([getattr(r, attr) for r in result_list]))


""" Functions related to Measurement data """
class RangeData:
    # Class for storing UWB range data.
    def __init__(self, dataloader: DataLoader):

        # remove all data which are not "uwb_range"
        data = dataloader.data
        self.setup = dataloader.setup
        self._sensors = ["uwb_range"]
        self.data = {id: {sensor: data[id][sensor] 
                          for sensor in data[id] if 
                          sensor in self._sensors} for id in data}


    def copy(self):
        return copy.deepcopy(self)


    def filter_by_bias(self, max_bias:float, 
                       min_bias :float = None) -> "RangeData":
        out = self.copy()
        for id in out.data:
            for sensor in out.data[id]:
                condition = (abs(out.data[id][sensor]['bias']) <= max_bias)
                if min_bias is not None:
                    condition &= (abs(out.data[id][sensor]['bias']) >= min_bias)
                out.data[id][sensor] = out.data[id][sensor][condition]

        return out

    def merge_range(self, robot_id:List=None, sensors:List=None) -> pd.DataFrame:

        if robot_id is None:
            robot_id = list(self.data.keys())

        if sensors is not None and not all(
            sensor in self._sensors for sensor in sensors):
                raise ValueError(f"Invalid sensor type. Must be one of {self._sensors}")
        else:
            sensors = self._sensors
        sensors = [sensors] if type(sensors) is str else sensors

        # create a pd dataframe with all the range data
        out = {sensor: [] for sensor in sensors}
        for id in robot_id:
            for sensor in sensors:
                if sensor in self.data[id]:
                    out[sensor].append(self.data[id][sensor])
        out = {sensor: pd.concat(out[sensor]) 
               for sensor in sensors}
        
        # sort the data by timestamp
        for sensor in sensors:
            out[sensor] = out[sensor].sort_values(by="timestamp")
            # remove duplicates
            out[sensor] = out[sensor].drop_duplicates(
                subset=["from_id", "to_id", "timestamp"], 
                keep="last")
            
        return out

    def to_measurements(self, reference_id: Any = 'world',) -> List[Measurement]:

        # Convert to a list of Measurement objects.

        data = self.merge_range()
        measurements = []
        for i, data in data['uwb_range'].iterrows():

            tags = self.setup['uwb_tags']
            from_tag = tags.loc[tags['tag_id'] == data.from_id].iloc[0]
            to_tag = tags.loc[tags['tag_id'] == data.to_id].iloc[0]

            variance = data["std"]**2
            if to_tag.parent_id == reference_id:
                model = RangePoseToAnchorById(
                    to_tag[['position.x','position.y','position.z']].tolist(),
                    from_tag[['position.x','position.y','position.z']].tolist(),
                    from_tag.parent_id, 
                    variance, )
            else:
                model = RangePoseToPose(
                    from_tag[['position.x','position.y','position.z']].tolist(),
                    to_tag[['position.x','position.y','position.z']].tolist(),
                    from_tag.parent_id,
                    to_tag.parent_id,
                    variance, )
                
            measurements.append(Measurement(data.range, data.timestamp, model))

        return measurements
    

def height_measurements(dataloader, start_time, end_time, R, bias):

    height = dataloader.by_timerange(start_time, end_time, 
                            sensors=["height"])
    height = [dataloader.data[id]["height"] for id in dataloader.data]
    min_height = [h['range'].min() for h in height]

    measurements = []
    for n, id in enumerate(dataloader.data):
        for i in range(len(height[n])):
            y = Measurement(
                value = height[n].iloc[i]['range'],
                stamp = height[n].iloc[i]['timestamp'],
                model = AltitudeById(R = R[n], 
                    state_id = id,
                    minimum = min_height[n],
                    bias = bias[n]))
            measurements.append(y)
    return measurements

""" Functions related to IMUKinematics process model"""

def get_unbiased_imu(x: IMUState, u):
    u = u.copy()
    u.gyro = u.gyro.ravel() - x.value[1].value.ravel()
    u.accel = u.accel.ravel() - x.value[2].value.ravel()
    return u

def U_matrix(omega, accel, dt):
    phi = omega * dt
    a = accel.reshape((-1, 1))
    U = np.identity(5)
    U[:3, :3] = SO3.Exp(phi)
    U[:3, 3] = np.ravel(dt * SO3.left_jacobian(phi) @ a)
    U[:3, 4] = np.ravel(dt**2 / 2 * SO3.wedge(a) @ a)
    U[3, 4] = dt
    return U

def G_matrix(gravity, dt):
    G = np.identity(5)
    G[:3, 3] = dt * gravity
    G[:3, 4] = -0.5 * dt**2 * gravity
    G[3, 4] = -dt
    return G

def L_matrix(unbiased_gyro, unbiased_accel, dt):
    """
    Computes the jacobian of the SE23_state with respect to the input.
    """
    a = unbiased_accel
    om = unbiased_gyro
    omdt = om * dt
    J_att_inv_times_N = SO3.left_jacobian_inv(omdt) @ SO3.wedge(omdt)
    xi = np.zeros((9,))
    xi[:3] = dt * om
    xi[3:6] = dt * a
    xi[6:9] = (dt**2 / 2) * J_att_inv_times_N @ a
    J = SE23.left_jacobian(-xi)
    Om = SO3.wedge(omdt)
    OmOm = Om @ Om
    A = SO3.wedge(a)
    Up = dt * np.eye(9, 6)
    Up[6:9, 0:3] = (-0.5 * (dt**2 / 2) * ((1 / 360) * (dt**3)
            * (OmOm @ A + Om @ (SO3.wedge(Om @ a)) + SO3.wedge(OmOm @ a))
            - (1 / 6) * dt * A))
    Up[6:9, 3:6] = (dt**2 / 2) * J_att_inv_times_N
    L = J @ Up
    return L

"""Error plotting functions """

def plot_error(
    results: GaussianResultList,
    label: str = None,
    sharey: bool = False,
    color=None,
    bounds=True,
    separate_figs = False,
) -> List[Tuple[plt.Figure, List[plt.Axes]]]:
    """
    A generic three-sigma bound plotter.
    """
    if separate_figs:
        no_of_plots = len(results.state[0].value)
    else:
        no_of_plots = 1
    dim = int(results.error.shape[1] / no_of_plots)

    if dim < 3:
        n_rows = dim
    else:
        n_rows = 3

    n_cols = int(np.ceil(dim / 3))
    figs_axes = [] 
    for n in range(no_of_plots):
        fig, axs = plt.subplots(n_rows, n_cols, sharex=True, sharey=sharey)

        axs_og = axs
        kwargs = {}
        if color is not None:
            kwargs["color"] = color

        axs: List[plt.Axes] = axs.ravel("F")
        
        for i in range(dim):
            if bounds:
                axs[i].fill_between(
                    results.stamp,
                    results.three_sigma[:, n*dim + i],
                    -results.three_sigma[:, n*dim + i],
                    alpha=0.5,
                    **kwargs,
                )
            axs[i].plot(results.stamp, results.error[:, n*dim + i], label=label, **kwargs)

        fig: plt.Figure = fig  # For type hinting
        figs_axes.append((fig, axs_og))
    return figs_axes