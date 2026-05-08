from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report


def train_svm(X_train, X_test, y_train, y_test):

    print("Training SVM...")

    model = SGDClassifier(
        loss='hinge',
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Predicting...")

    y_pred = model.predict(X_test)

    print("\n=== SVM ===")

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    print(report)

    with open("models_saved/svm_results.txt", "w") as f:
        f.write(report)

    return model, y_pred