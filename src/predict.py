
#Here we can upload new csv files which will be preprocessed and predicted using the saved model and preprocessor artifacts.
#The output will be a csv file with the original data along with two new columns: "churn_probability" and "churn_prediction".

import pandas as pd
import sys
import os
import joblib
import pandas as pd
import numpy as np

#src import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MODEL_PATH = "model/final_model.pkl"
PREPROCESSOR_PATH = "model/preprocessor.pkl"


#Load Model & Preprocessor

def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Train the model first.")

    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError("Preprocessor file not found. Train the model first.")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    print("Model & Preprocessor Loaded Successfully")

    return model, preprocessor



#Prediction Function

def predict_churn(df, model, preprocessor, threshold=0.5):
    """
    df : pandas DataFrame of new customer data
    threshold : probability threshold for churn decision
    """

    X_processed = preprocessor.transform(df)

    churn_prob = model.predict_proba(X_processed)[:, 1]
    churn_pred = (churn_prob >= threshold).astype(int)

    results = df.copy()
    results["churn_probability"] = churn_prob
    results["churn_prediction"] = churn_pred
    results["churn_prediction"] = results["churn_prediction"].map({1: "Yes", 0: "No"})

    return results



# Main Runner (CSV Mode)


def main():

    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_new_data.csv>")
        sys.exit(1)

    data_path = sys.argv[1]

    if not os.path.exists(data_path):
        raise FileNotFoundError("Input CSV file not found")

    df = pd.read_csv(data_path)

    print("New Data Loaded")
    print("Shape:", df.shape)

    if "customerID" in df.columns:
        customer_ids = df["customerID"]
        df = df.drop(columns=["customerID"])

    model, preprocessor = load_artifacts()

    predictions = predict_churn(df, model, preprocessor)

    os.makedirs("predictions", exist_ok=True)
    output_path = "predictions/churn_predictions.csv"

    predictions.to_csv(output_path, index=False)

    print("\nPredictions saved to:", output_path)
    print(predictions.head())



# Entry Point


if __name__ == "__main__":
    main()
    
df = pd.read_csv("Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")

sample = df.sample(10).drop(columns=["customerID","Churn"])

model, preprocessor = load_artifacts()

output1=predict_churn(sample, model, preprocessor)
print(output1)

