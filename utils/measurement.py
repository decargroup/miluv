import numpy as np
from typing import Any, List
import pandas as pd
from utils.models import (
    RangePoseToAnchorById,
    RangePoseToPose,
)
from miluv.data import DataLoader
import copy

class Measurement:
    """
    A data container containing a measurement's value, timestamp,
    and corresponding model.
    """
    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        model = None,
        state_id: Any = None,
    ):
        self.value = np.array(value) if np.isscalar(value) else value
        self.stamp = stamp
        self.model = model
        self.state_id = state_id

    def minus(self, y_check: np.ndarray) -> np.ndarray:
        return self.value.reshape((-1, 1)) - y_check.reshape((-1, 1))