import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import matplotlib.patches as patches
import mpl_toolkits.mplot3d.art3d as art3d
import itertools
import pandas as pd
from utils.states import (
    State, 
    StateWithCovariance
)
from miluv.data import DataLoader

class GaussianResult:
    """
    A data container that simultaneously computes various interesting metrics
    about a Gaussian filter's state estimate, given the ground-truth value of
    the state.
    """

    __slots__ = [
        "stamp",
        "state",
        "state_true",
        "covariance",
        "error",
        "ees",
        "nees",
        "md",
        "three_sigma",
        "rmse",
    ]

    def __init__(
        self,
        estimate: StateWithCovariance,
        state_true: State,
    ):
        """
        Parameters
        ----------
        estimate : StateWithCovariance
            Estimated state and corresponding covariance.
        state_true : State
            The true state, which will be used to compute various error metrics.
        """

        state = estimate.state
        covariance = estimate.covariance

        #:float: timestamp
        self.stamp = state.stamp
        #:State: estimated state
        self.state = state
        #:State: true state
        self.state_true = state_true
        #:numpy.ndarray: covariance associated with estimated state
        self.covariance = covariance

        e = state_true.minus(state).reshape((-1, 1))
        #:numpy.ndarray: error vector between estimated and true state
        self.error = e.ravel()
        #:float: sum of estimation error squared (EES)
        self.ees = np.ndarray.item(e.T @ e)
        #:float: normalized estimation error squared (NEES)
        self.nees = np.ndarray.item(e.T @ np.linalg.solve(covariance, e))
        #:float: root mean squared error (RMSE)
        self.rmse = np.sqrt(self.ees / state.dof)
        #:float: Mahalanobis distance
        self.md = np.sqrt(self.nees)
        #:numpy.ndarray: three-sigma bounds on each error component
        self.three_sigma = 3 * np.sqrt(np.diag(covariance))


class GaussianResultList:
    """
    A data container that accepts a list of ``GaussianResult`` objects and
    stacks the attributes in numpy arrays. Convenient for plotting. This object
    does nothing more than array-ifying the attributes of ``GaussianResult``
    """

    __slots__ = [
        "stamp",
        "state",
        "state_true",
        "covariance",
        "error",
        "ees",
        "nees",
        "md",
        "three_sigma",
        "value",
        "value_true",
        "dof",
        "rmse",
    ]

    def __init__(self, result_list: List[GaussianResult]):
        """
        Parameters
        ----------
        result_list : List[GaussianResult]
            A list of GaussianResult, intended such that each element corresponds
            to a different time point


        Let ``N = len(result_list)``
        """
        #:numpy.ndarray with shape (N,):  timestamp
        self.stamp = np.array([r.stamp for r in result_list])
        #:numpy.ndarray with shape (N,): numpy array of State objects
        self.state: List[State] = np.array([r.state for r in result_list])
        #:numpy.ndarray with shape (N,): numpy array of true State objects
        self.state_true: List[State] = np.array(
            [r.state_true for r in result_list]
        )
        #:numpy.ndarray with shape (N,dof,dof): covariance
        self.covariance: np.ndarray = np.array(
            [r.covariance for r in result_list]
        )
        #:numpy.ndarray with shape (N, dof): error throughout trajectory
        self.error = np.array([r.error for r in result_list])
        #:numpy.ndarray with shape (N,): EES throughout trajectory
        self.ees = np.array([r.ees for r in result_list])
        #:numpy.ndarray with shape (N,): EES throughout trajectory
        self.rmse = np.array([r.rmse for r in result_list])
        #:numpy.ndarray with shape (N,): NEES throughout trajectory
        self.nees = np.array([r.nees for r in result_list])
        #:numpy.ndarray with shape (N,): Mahalanobis distance throughout trajectory
        self.md = np.array([r.md for r in result_list])
        #:numpy.ndarray with shape (N, dof): three-sigma bounds
        self.three_sigma = np.array([r.three_sigma for r in result_list])
        #:numpy.ndarray with shape (N,): state value. type depends on implementation
        self.value = np.array([r.state.value for r in result_list])
        #:numpy.ndarray with shape (N,): dof throughout trajectory
        self.dof = np.array([r.state.dof for r in result_list])
        #:numpy.ndarray with shape (N,): true state value. type depends on implementation
        self.value_true = np.array([r.state_true.value for r in result_list])

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

    Parameters
    ----------
    results : GaussianResultList
        Contains the data to plot
    axs : List[plt.Axes], optional
        Axes to draw on, by default None. If None, new axes will be created.
    label : str, optional
        Text label to add, by default None
    sharey : bool, optional
        Whether to have a common y axis or not, by default False
    color : color, optional
        specify the color of the error/bounds.

    Returns
    -------
    List[Tuple[plt.Figure, List[plt.Axes]]]
        List of tuples, each containing a handle to figure and a list of handles to axes that were drawn on.
    """
    if separate_figs:
        no_of_plots = len(results.value[0])
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


def plot_uav3d(state, 
               ax: plt.Axes = None, 
               length=0.17, 
               tag_body_position = [[0.17, 0.17], 
                                    [-0.17, 0.17]],
               alpha = 1,
               color = None,
               details = False,
               animate = False):
    
    colors = ['#40E0D0', '#C1FFC1', '#FFB6C1']
    if color is None:
        color = colors[-1]
    else:
        color = color

    if ax is None:
        fig, ax = plt.subplots(1,1)
    elif isinstance(ax, plt.Axes):
        fig = ax.get_figure()
    
    if state.size == 2:
        center = state[0:2]
        C = np.eye(2)
    
    elif state.size == 9 or 12:
        center = state[:,-1][0:2]
        C = state[0:2,0:2]

    vertices = np.array([[length, length],
                         [-length, length],
                         [-length, -length],
                         [length, -length]])
    tag_position_no = []
    tag_position = []
    for i, position in enumerate(tag_body_position):
        tag_body_position[i] = position[0:2]


    tag_body_position = np.array(tag_body_position)

    # Calculate the Euclidean distance between vertices and tags
    distances = np.linalg.norm(vertices[:, np.newaxis] - tag_body_position, axis=2)

    # Find the indices where the distances are less than 1e-4
    tag_position_no = np.where(distances < 1e-4)[0]


    rotated_vertices = vertices @ C.T
    translated_vertices = rotated_vertices + center

    if animate:
        patches_list = []
        lines_list = []

    for i in range(4):
        if i in tag_position_no:
            tag_position.append(translated_vertices[i])
            rotor = patches.Circle(translated_vertices[i], 0.2 * length, fc='#FF6103', ec='gray', linewidth=1.5, zorder=10, alpha = alpha)
        else:
            rotor = patches.Circle(translated_vertices[i], 0.2 * length, fc='gray', ec=color, linewidth=1.5, zorder=10, alpha = alpha)
        if animate:
            patches_list.append(rotor)
        else:
            ax.add_patch(rotor)
            art3d.pathpatch_2d_to_3d(rotor, z=state[:,-1][2], zdir="z")
            

    for i in range(2):
        x = [translated_vertices[i][0], translated_vertices[i+2][0]]
        y = [translated_vertices[i][1], translated_vertices[i+2][1]]
        z = [state[:,-1][2], state[:,-1][2]]  # assuming z is constant along the line
        if animate:
            lines_list.append((x, y, z))
        else:
            ax.plot(x, y, z, color='black', linewidth=2, zorder=0, alpha=alpha)

    frame = patches.Circle(center, 0.5 * length, fc=color, ec='black', linewidth=2, zorder=10, alpha = alpha)
    x_axis = ax.quiver(center[0], center[1], state[:,-1][2], length*C[0,0], length*C[1,0], 0, color='r', alpha=alpha, zorder=15)
    y_axis = ax.quiver(center[0], center[1], state[:,-1][2], length*C[0,1], length*C[1,1], 0, color='g', alpha = alpha, zorder=15)
    z_axis = ax.quiver(center[0], center[1], state[:,-1][2], 0, 0, length, color='b', alpha = alpha, zorder=15)
    t = []
    for tag in tag_position:
        t.append(np.array([tag[0], tag[1], state[:,-1][2]]))
    tag_position = np.array(t)
    # plt.axis('scaled')
    
    if details == True:
        ax.add_patch(frame)
        art3d.pathpatch_2d_to_3d(frame, z=state[:,-1][2], zdir="z")
        return fig, ax, tag_position
    
    else:
        ax.add_patch(frame)
        art3d.pathpatch_2d_to_3d(frame, z=state[:,-1][2], zdir="z")
        return fig, ax

def plot_range3d(tag_positions, ax, alpha=1):
    for i, j in itertools.combinations(range(len(tag_positions)), 2):
        for k, l in itertools.product(range(len(tag_positions[0])), repeat=2):
            x = [tag_positions[i][k][0], tag_positions[j][l][0]]
            y = [tag_positions[i][k][1], tag_positions[j][l][1]]
            z = [tag_positions[i][k][2], tag_positions[j][l][2]]  # assuming z is the third coordinate
            ax.plot(x, y, z, color="orange", linewidth=1, alpha=alpha)
    return ax

def plot_trajectory3d(gt_data, ax: plt.Axes, colors =  None):
    x =[]
    if colors is None:
        colors = ['C' + str(i) for i in range(len(gt_data[0].value))]
    no_of_robots = len(gt_data[0].value)
    for i in range(no_of_robots):
        x.append(np.array([data.value[i].position for data in gt_data]))
    for i in range(no_of_robots):
        ax.scatter3D(x[i][:, 0], x[i][:, 1], x[i][:, 2], color = colors[i], s = 0.5)