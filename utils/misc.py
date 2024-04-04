import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from utils.states import StateWithCovariance
from navlie.utils import GaussianResult

class GaussianResult:
    """
    A data container that simultaneously computes the error and three-sigma
    bounds of a state estimate.
    """
    def __init__(
        self,
        estimate: StateWithCovariance,
        state_true,
    ):
        state = estimate.state
        covariance = estimate.covariance
        self.stamp = state.stamp
        self.state = state
        self.state_true = state_true
        self.covariance = covariance
        self.error = state_true.minus(state).ravel()
        self.three_sigma = 3 * np.sqrt(np.diag(covariance))

class GaussianResultList:
    """
    A data container that accepts a list of ``GaussianResult`` objects and
    stacks the attributes in numpy arrays. Convenient for plotting.
    """
    def __init__(self, result_list: List[GaussianResult]):
        self.stamp = np.array([r.stamp for r in result_list])
        self.state: List = np.array([r.state for r in result_list])
        self.state_true: List = np.array([r.state_true for r in result_list])
        self.covariance = np.array([r.covariance for r in result_list])
        self.error = np.array([r.error for r in result_list])
        self.three_sigma = np.array([r.three_sigma for r in result_list])

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