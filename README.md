# 📊 Customer Churn Prediction System

An end-to-end Machine Learning solution designed to identify high-risk customers likely to discontinue service. This project transforms raw historical data into actionable business intelligence through a deployable web interface.

---

## 🚀 Key Features

- **Automated Pipeline:** Clean, encode, and scale data using a unified preprocessing pipeline  
- **Real-time Inference:** Interactive Streamlit dashboard for instant churn prediction  
- **Reproducible Research:** Well-documented EDA and model training notebooks  
- **Scalable Storage:** Large model files managed using Git LFS (Large File Storage)

---

## 📈 Model Performance & Evaluation

Multiple models were evaluated to analyze trade-offs between precision and recall.  
Depending on business objectives (e.g., minimizing false positives vs. capturing all potential churners), different models provide distinct advantages.

| Model                | Precision | Recall | F1 Score | Accuracy | Best Use Case              |
|---------------------|-----------|--------|----------|----------|----------------------------|
| Logistic Regression | 0.63      | 0.54   | 0.58     | 0.79     | High Explainability        |
| Decision Tree       | 0.53      | 0.77   | 0.63     | 0.76     | Balanced Recall            |
| Random Forest       | 0.46      | 0.81   | 0.59     | 0.70     | Maximum Churn Capture      |

---

## 🧠 Model Selection Insights

- **Logistic Regression** achieved the highest overall accuracy (79%) and offers strong interpretability  
- **Random Forest** delivered the highest recall (0.81), making it ideal for capturing maximum churners  
- **Decision Tree** provided the best balance with the highest F1 Score (0.63)

---

## 🗂️ Project Structure
Customer-Churn-Prediction/
├── Dataset/             # Raw and processed CSV files
├── notebooks/           # Exploratory Data Analysis (EDA) & Model Training
├── model/
│   ├── final_model.pkl  # Trained classifier (Git LFS)
│   └── preprocessor.pkl # Preprocessing pipeline (Git LFS)
├── src/                 # Modular Python scripts for training/utils
├── app.py               # Streamlit UI 
├── requirements.txt     # Dependency list
└── README.md            # Project documentation

---

## 🛠️ Tech Stack

- **Language:** Python 3.x  
- **Data Processing & ML:** Pandas, NumPy, Scikit-learn  
- **Web Framework:** Streamlit  
- **Serialization:** Pickle, Joblib  
- **Version Control:** Git, Git LFS  

---

## 📌 Installation & Local Setup
Clone the repository:
```bash
git clone https://github.com/your-username/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

Install dependencies:
pip install -r requirements.txt

Run the application:
streamlit run app.py
```
# 👤 Author
Mirza Abdul Raffay Baig Computer Science (AI) | Data Analysis & Machine Learning
