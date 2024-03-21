import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
from os.path import join
import numpy as np
from csaps import csaps
from scipy.spatial.transform import Rotation
import scipy as sp
from typing import Tuple

def get_experiment_info(path):
    exp_name = path.split('/')[-1]
    df = pd.read_csv(join("config", "experiments.csv"))
    row: pd.DataFrame = df[df["experiment"] == exp_name]
    return row.to_dict(orient="records")[0]

def get_anchors(anchor_constellation):
    with open('config/uwb/anchors.yaml', 'r') as file:
        return yaml.safe_load(file)[anchor_constellation]

def get_tags(flatten=False):
    with open('config/uwb/tags.yaml', 'r') as file:
        moment_arms = yaml.safe_load(file)

    if flatten:
        arms = {}
        for robot, arm in moment_arms.items():
            arms.update(arm)
        return arms
    else:
        return moment_arms