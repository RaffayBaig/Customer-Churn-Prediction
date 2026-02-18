📊 Customer Churn Prediction System
An end-to-end Machine Learning solution designed to identify high-risk customers likely to discontinue service. This project transforms raw historical data into actionable business intelligence through a deployable web interface.

🚀 Key Features
Automated Pipeline: Clean, encode, and scale data with a unified preprocessing pipeline.

Real-time Inference: Interactive Streamlit dashboard for instant churn probability.

Reproducible Research: Documented EDA and model selection process in Jupyter Notebooks.

Scalable Storage: Large model files managed via Git LFS (Large File Storage).

📈 Model Performance & Evaluation
I evaluated three distinct models to understand the trade-offs between precision and recall. Depending on the business objective (e.g., minimizing false alarms vs. capturing every potential churner), different models offer unique advantages.
Model,Precision,Recall,F1 Score,Accuracy,Best Use Case
Logistic Regression,0.63,0.54,0.58,0.79,High Explainability
Decision Tree,0.53,0.77,0.63,0.76,Balanced Recall
Random Forest,0.46,0.81,0.59,0.70,Maximum Churn Capture
