from src.models.random_forest import train_rf
from src.models.knn import train_knn
from src.models.svm import train_svm
from src.models.logistic_regression import train_logistic
from src.models.naive_bayes import train_naive_bayes

from src.preprocessing import (
    load_and_preprocess,
    clean_data,
    prepare_train_test
)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# =========================
# LOAD & PREPROCESS DATA
# =========================

print("Loading dataset...")

df = load_and_preprocess("dataset")

print("Cleaning data...")

df = clean_data(df)

print("Preparing train/test split...")

X_train, X_test, y_train, y_test = prepare_train_test(df)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# =========================
# RANDOM FOREST
# =========================

rf_model, rf_pred = train_rf(
    X_train,
    X_test,
    y_train,
    y_test
)


# =========================
# KNN
# =========================

knn_model, knn_pred = train_knn(
    X_train,
    X_test,
    y_train,
    y_test
)


# =========================
# SVM
# =========================

svm_model, svm_pred = train_svm(
    X_train,
    X_test,
    y_train,
    y_test
)
# =========================
# Logistic Regresison
# =========================

train_logistic(
    X_train,
    X_test,
    y_train,
    y_test
)

# =========================
# Naive Bayes
# =========================

train_naive_bayes(
    X_train,
    X_test,
    y_train,
    y_test
)

# =========================
# CONFUSION MATRIX (RF)
# =========================

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(12, 8))

sns.heatmap(cm, annot=True, fmt='d')

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# =========================
# REAL-TIME ALERT DEMO
# =========================

sample = X_test.iloc[[0]]

prediction = rf_model.predict(sample)[0]

print("\n===== REAL-TIME IDS ALERT =====")

if prediction != 0:
    print(f"[ALERT] Suspicious traffic detected!")
    print(f"Attack Type: {prediction}")
else:
    print("[INFO] Normal BENIGN traffic")