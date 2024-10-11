---
title: Data Loading
parent: Examples
nav_order: 1
---

# Data Loading
Using the MILUV devkit, retrieving sensor data by timestamp from experiment ``3a`` can be implemented as:
```py
from miluv.data import DataLoader
import numpy as np

mv_3a = DataLoader(
        "3a",
        cir=False,
        barometer=False,
        height=False,
    )

timestamps = np.arange(0, 10, 1)  # Time in sec

data_at_timestamps = mv_3a.data_from_timestamps(timestamps)
```

This example can be made elaborate by selecting specific robots and sensors to fetch from at the given timestamps.
```py
from miluv.data import DataLoader
import numpy as np

mv_3a = DataLoader(
        "3a",
        cir=False,
        barometer=False,
        height=False,
    )

timestamps = np.arange(0, 10, 1)  # Time in sec

robots = ["ifo001", "ifo002"]  # We are leaving out ifo003
sensors = ["imu_px4", "imu_cam"]  # Fetching just the imu data

data_at_timestamps = mv_3a.data_from_timestamps(
    timestamps,
    robots,
    sensors,
    )
```
