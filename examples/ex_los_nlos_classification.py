from miluv.data import DataLoader
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def main():
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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        random_state=0,
    )

    nn = MLPClassifier(
        hidden_layer_sizes=(
            100,
            100,
        ),
        max_iter=100,
    )

    nn.fit(X_train, y_train)
    nn.predict(X_test)
    print(nn.score(X_test, y_test))

    # Visualize results
    plt.plot(nn.loss_curve_)
    plt.show()


if __name__ == "__main__":
    main()
