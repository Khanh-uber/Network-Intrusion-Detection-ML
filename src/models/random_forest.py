from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib


def train_rf(X_train, X_test, y_train, y_test):

    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )

    print("Training Random Forest...")

    model.fit(X_train, y_train)

    print("Predicting...")

    y_pred = model.predict(X_test)

    print("\n=== RANDOM FOREST ===")

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    print(report)


    # Save classification report
    with open("models_saved/rf_results.txt", "w") as f:
        f.write(report)


    # Save model
    joblib.dump(
        model,
        "models_saved/best_rf_model.pkl"
    )

    print("\nRandom Forest model saved!")

    return model, y_pred