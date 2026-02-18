import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib


def make_numerical_pipeline():
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])


def make_categorical_pipeline():
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])


def make_preprocessor(numeric_columns, categorical_columns):
    return ColumnTransformer(transformers=[
        ("num", make_numerical_pipeline(), numeric_columns),
        ("cat", make_categorical_pipeline(), categorical_columns)
    ])


def get_preprocessed_data(X_train, X_test, preprocessor):
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    return X_train_processed, X_test_processed


def save_preprocessor(preprocessor, path="model/preprocessor.pkl"):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path="model/preprocessor.pkl"):
    return joblib.load(path)


def get_feature_names(preprocessor):
    num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
    cat_features = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out()
    return np.concatenate([num_features, cat_features])
