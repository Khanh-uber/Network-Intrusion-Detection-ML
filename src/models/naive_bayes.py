from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

def train_naive_bayes(X_train, X_test, y_train, y_test):

    model = GaussianNB()

    print("\nTraining Naive Bayes...")

    model.fit(X_train, y_train)

    print("Predicting...")

    y_pred = model.predict(X_test)

    print("\n=== NAIVE BAYES ===")

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    print(report)

    return model, y_pred