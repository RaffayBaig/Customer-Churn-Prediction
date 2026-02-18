import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Fix src import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import make_preprocessor, save_preprocessor



#Load Dataset


DATA_PATH = r"Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully")
print("Shape:", df.shape)



#Target Encoding


df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop ID column
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

# Sanity check
assert y.isnull().sum() == 0, "Target contains NaN values"



#Column Identification


num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("Numerical Features:", len(num_cols))
print("Categorical Features:", len(cat_cols))



#Train Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)



#Preprocessing Pipeline


preprocessor = make_preprocessor(num_cols, cat_cols)

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

save_preprocessor(preprocessor)

print("Preprocessing Done & Saved")



#Model + Hyperparameter Grid


rf = RandomForestClassifier(random_state=42, n_jobs=-1)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=2
)



#Hyperparameter Tuning


print("\nStarting Grid Search Hyperparameter Tuning...\n")

grid_search.fit(X_train_proc, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print("Best CV ROC-AUC:", grid_search.best_score_)


# Final Model Training

best_model = grid_search.best_estimator_
best_model.fit(X_train_proc, y_train)



#  Final Evaluation


y_pred_proba = best_model.predict_proba(X_test_proc)[:, 1]
test_auc = roc_auc_score(y_test, y_pred_proba)

print("\nFinal Test ROC-AUC:", round(test_auc, 4))


#Saving the final model

os.makedirs("model", exist_ok=True)

joblib.dump(best_model, "model/final_model.pkl")

print("\nModel saved successfully to model/final_model.pkl")
