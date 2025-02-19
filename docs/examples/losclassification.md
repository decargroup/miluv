---
title: LOS/NLOS Classification
parent: Examples
nav_order: 5
---

# LOS/NLOS Classification
## Overview
The MILUV devkit provides an example of LOS/NLOS for users to run and modify. To run this example, run
```
python examples/ex_los_nlos_classification.py
```
in the repository's root directory.

The output from this example is show below.
![](https://decargroup.github.io/miluv/assets/lazy_classifier_results.png)

## More details
This example demonstrates how to use CIR data and how to set up MILUV data for machine learning purposes as shown below,
```py
import numpy as np
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier

from miluv.data import DataLoader

# List of anchor IDs that are NLOS
tag_ids_nlos = [
    1,  # styrofoam
    3,  # plastic
    4,  # wood
]

mv = DataLoader(
    "cirObstacles_3_random_0",
    barometer=False,
)

data = mv.data
X = []
y = []

for robot_id in data.keys():
    for anchor_id, cir_data in zip(
            data[robot_id]["uwb_cir"]["to_id"],
            data[robot_id]["uwb_cir"]["cir"],
    ):
        cir_data = cir_data.replace("[", "").replace("]", "").split(", ")
        cir_data = [int(x) for x in cir_data]
         X.append(cir_data)
        if anchor_id in tag_ids_nlos:
            y.append(1)
        else:
            y.append(0)
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
)
# Continued below...
```

As an example, we use the `lazypredict` library to run a suite of classification techniques for the LOS/NLOS task. This code can easily modified to substitute what is below for another classification algorithm.
```py
#... Continued from above

# YOUR ML CLASSIFIER GOES HERE
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)
```
