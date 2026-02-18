# 📊 Customer Churn Prediction System

A complete Machine Learning project that predicts whether a customer is likely to churn based on historical data.  
The project covers the **end-to-end ML workflow**: data preprocessing, model training, evaluation, and deployment readiness.

---
## 🚀 Project Overview

Customer churn is a critical business problem where companies aim to identify customers who are likely to stop using their services.  
This project uses supervised machine learning techniques to predict churn and help businesses take proactive retention actions.

---

## 🧠 Machine Learning Approach
- **Problem Type:** Binary Classification  
- **Target Variable:** Customer Churn (Yes / No)
- **Models Used:** Classical Machine Learning Models  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score

---
## 🗂️ Project Structure
Customer-Churn-Prediction/
│
├── Dataset/ # Raw and processed datasets
├── notebooks/ # EDA and model training notebooks
├── model/
│ ├── final_model.pkl # Trained ML model (Git LFS)
│ └── preprocessor.pkl # Data preprocessing pipeline (Git LFS)
│
├── src/ # Source code (training, utilities)
├── app.py # Streamlit application
├── requirements.txt # Project dependencies
└── README.md # Project documentation


---

## 🔄 Data Preprocessing

- Handling missing values  
- Encoding categorical variables  
- Feature scaling  
- Pipeline-based preprocessing for consistency  

The preprocessing logic is saved separately to ensure **reproducible predictions**.

---

## 📈 Model Training & Evaluation

- Multiple models were trained and evaluated
- Final model selected based on performance metrics
- The trained model and preprocessing pipeline are saved using `pickle/joblib`

---

## 🖥️ Web Application (Streamlit)

A simple and interactive **Streamlit web app** allows users to:
- Input customer details
- Get real-time churn predictions
- Visualize prediction outcomes

---

## 🛠️ Technologies Used

- **Programming Language:** Python  
- **Libraries:** NumPy, Pandas, Scikit-learn  
- **Model Serialization:** Pickle / Joblib  
- **Web Framework:** Streamlit  
- **Version Control:** Git, Git LFS  

---

## 📌 How to Run the Project Locally

```bash
git clone https://github.com/your-username/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
pip install -r requirements.txt
streamlit run app.py

                                                                                                                           👤 Author

                                                                                                                       Mirza Abdul Raffay Baig
                                                                                                   Computer Science (AI) | Data Analysis & Machine Learning

