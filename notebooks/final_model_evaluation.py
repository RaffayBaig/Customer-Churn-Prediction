import pandas as pd
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay
)

MODEL_PATH = "model/final_model.pkl"
PREPROCESSOR_PATH = "model/preprocessor.pkl"
TEST_DATA_PATH = "Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"


# ---------------------------
# Load Model & Preprocessor
# ---------------------------
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found")

    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError("Preprocessor not found")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    print("Model & Preprocessor Loaded")
    return model, preprocessor


# ---------------------------
# Evaluation Function
# ---------------------------
def evaluate_model(model, preprocessor, df, threshold=0.5):

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})

    X_processed = preprocessor.transform(X)

    y_prob = model.predict_proba(X_processed)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print("\n===== MODEL PERFORMANCE =====")
    print("Accuracy :", accuracy_score(y, y_pred))
    print("Precision:", precision_score(y, y_pred))
    print("Recall   :", recall_score(y, y_pred))
    print("F1 Score :", f1_score(y, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc = roc_auc_score(y, y_prob)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC = {auc:.3f})")
    plt.show()


# ---------------------------
# Threshold Tuning (IMPORTANT)
# ---------------------------
def find_best_threshold(model, preprocessor, df):

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0})

    X_processed = preprocessor.transform(X)
    y_prob = model.predict_proba(X_processed)[:, 1]

    best_threshold = 0.5
    best_recall = 0

    for t in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_prob >= t).astype(int)
        recall = recall_score(y, y_pred)

        if recall > best_recall:
            best_recall = recall
            best_threshold = t

    print("\nBest Threshold for Recall:", best_threshold)
    print("Best Recall:", best_recall)


# ---------------------------
# Main Runner
# ---------------------------
def main():
    df = pd.read_csv(TEST_DATA_PATH)

    model, preprocessor = load_artifacts()

    find_best_threshold(model, preprocessor, df)
    evaluate_model(model, preprocessor, df, threshold=0.5)


if __name__ == "__main__":
    main()
