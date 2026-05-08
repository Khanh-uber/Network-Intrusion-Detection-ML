from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def train_logistic(X_train, X_test, y_train, y_test):

    model = LogisticRegression(
        max_iter=5000,
        n_jobs=-1
    )

    print("\nTraining Logistic Regression...")

    model.fit(X_train, y_train)

    print("Predicting...")

    y_pred = model.predict(X_test)

    print("\n=== LOGISTIC REGRESSION ===")

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    print(report)

    return model, y_pred