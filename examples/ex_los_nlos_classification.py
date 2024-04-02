# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# Modified for MILUV by Nicholas Dahdah
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC

from miluv.data import DataLoader

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
]

robot_ids = [
    "ifo001",
    "ifo002",
    "ifo003",
]

tag_ids_nlos = [
    1,  # styrofoam
    3,  # plastic
    4,  # wood
]

mv = DataLoader(
    "12c",
    barometer=False,
)

data = mv.data

X = []
y = []

for robot_id in robot_ids:
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
    test_size=0.3,
    random_state=0,
)

grid_c_gamma = False
if grid_c_gamma:
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, y)

    print("The best parameters are %s with a score of %0.2f" %
          (grid.best_params_, grid.best_score_))

    gamma = grid.best_params_['gamma'],
    C = grid.best_params_['C'],
else:
    gamma = 1E-9
    C = 0.01

classifiers = [
    KNeighborsClassifier(3),
    SVC(
        kernel="linear",
        C=0.025,
        random_state=42,
    ),
    SVC(
        gamma=gamma,
        C=C,
        random_state=42,
    ),
    GaussianProcessClassifier(
        1.0 * RBF(1.0),
        random_state=42,
    ),
    DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    ),
    RandomForestClassifier(
        max_depth=5,
        n_estimators=10,
        max_features=1,
        random_state=42,
    ),
    MLPClassifier(
        alpha=1,
        max_iter=1000,
        random_state=42,
    ),
    AdaBoostClassifier(
        algorithm="SAMME",
        random_state=42,
    ),
    GaussianNB(),
]

# iterate over classifiers
for name, clf in zip(names, classifiers):
    clf = make_pipeline(StandardScaler(), clf)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f"{name}: {score}")
