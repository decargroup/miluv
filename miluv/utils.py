import matplotlib.pyplot as plt
import seaborn as sns

def set_plotting_env() -> None:
    """Set the plotting environment
    """
    sns.set_palette("colorblind")
    sns.set_style('whitegrid')

    plt.rc('figure', figsize=(16, 9))
    plt.rc('lines', linewidth=2)
    plt.rc('axes', grid=True)
    plt.rc('grid', linestyle='--')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=35)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rc('legend', facecolor=[1,1,1])
    plt.rc('legend', fontsize=30)
    plt.rcParams['figure.constrained_layout.use'] = True



