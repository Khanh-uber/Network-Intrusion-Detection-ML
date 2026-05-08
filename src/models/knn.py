from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def train_knn(X_train, X_test, y_train, y_test):

    print("Training KNN...")

    model = KNeighborsClassifier(
        n_neighbors=5
    )

    model.fit(X_train, y_train)

    print("Predicting...")

    y_pred = model.predict(X_test)

    print("\n=== KNN ===")

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    print(report)

    # Save report
    with open("models_saved/knn_results.txt", "w") as f:
        f.write(report)

    return model, y_pred