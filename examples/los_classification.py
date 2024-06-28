import numpy as np
from miluv.data import DataLoader

# To run this example, you need to install the following extra packages (found in requirements_dev.txt):
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier


def main():
    # List of anchor IDs that are NLOS
    tag_ids_nlos = [
        1,  # styrofoam
        3,  # plastic
        4,  # wood
    ]

    mv = DataLoader(
        "3l",
        barometer=False,
        cir=True
    )

    X = []
    y = []

    for robot_id in mv.data.keys():
        for anchor_id, cir_data in zip(
                mv.data[robot_id]["uwb_cir"]["to_id"],
                mv.data[robot_id]["uwb_cir"]["cir"],
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

    # YOUR ML CLASSIFIER GOES HERE
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    print(models)


if __name__ == "__main__":
    main()
