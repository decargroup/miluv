import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import matplotlib.patches as patches
import mpl_toolkits.mplot3d.art3d as art3d
import itertools
import pandas as pd
from utils.states import (
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
        state_true,
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
        self.state: List = np.array([r.state for r in result_list])
        #:numpy.ndarray with shape (N,): numpy array of true State objects
        self.state_true: List = np.array(
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