# FraudTransactionDetection
🚨 Fraud Transaction Detection using Machine Learning

A machine learning project designed to detect fraudulent financial transactions using data analysis, feature engineering, and predictive modeling. The goal is to identify suspicious transactions with high recall while minimizing false negatives in highly imbalanced datasets.

📌 Objective

The primary objective of this project is to:

Detect fraudulent transactions using supervised machine learning

Handle class imbalance effectively

Analyze fraud patterns through exploratory data analysis (EDA)

Build a deployable model for real-time transaction risk scoring

📂 Project Structure
Fraud-Transaction-Detection/
│
├── Accredian.ipynb                # Main notebook (EDA → Training → Evaluation)
├── Fraud.csv                      # Dataset
├── fraud_detection_model.pkl      # Trained ML model
├── fraud_model_bundle.pkl         # Model + feature metadata
└── README.md                      # Project documentation

🛠️ Tech Stack

Python

Pandas & NumPy – Data processing

Matplotlib – Visualization

Scikit-learn – Machine Learning

Joblib – Model serialization

📊 Workflow
1️⃣ Data Loading

Imported transaction dataset

Created separate copies for EDA and modeling

2️⃣ Exploratory Data Analysis (EDA)

Class distribution analysis

Transaction pattern visualization

Fraud vs non-fraud comparison

3️⃣ Data Cleaning

Handled missing/inconsistent values

Removed unnecessary features

4️⃣ Feature Engineering

Generated model-ready features

One-hot encoding for categorical variables

Feature scaling for logistic regression

5️⃣ Data Preparation

Train-test split

Handling severe class imbalance

6️⃣ Model Training

Logistic Regression (with balanced class weights)

Standard scaling applied before training

7️⃣ Model Evaluation

Metrics used:

Precision

Recall

F1 Score

Confusion Matrix

ROC-AUC

Key Result:
The model achieves very high fraud recall (~98%), making it effective for fraud detection where missing fraud cases is costly.

📈 Model Performance (Example)
Metric	Value
Accuracy	~95%
Fraud Recall	~98%
Fraud Precision	Low (expected due to imbalance)

⚠️ Note: In fraud detection, high recall is prioritized over precision to minimize missed fraud cases.

🔍 Fraud Pattern Insights

Fraud transactions show distinctive behavior in:

Transaction amount

Account balance changes

Transaction types

Feature importance analysis helps explain model decisions.

💾 Model Saving & Loading
import joblib

# Save model
joblib.dump(model, "fraud_detection_model.pkl")

# Load model
model = joblib.load("fraud_detection_model.pkl")

⚙️ Deployment Plan

Expose model as REST API

Real-time transaction scoring

Integration with banking or fintech pipelines

📊 Monitoring KPIs

Fraud detection rate

False positive rate

Model drift monitoring

Prediction latency

🚀 Future Improvements

Use advanced models (XGBoost / LightGBM)

Threshold tuning for business needs

Explainable AI (SHAP/LIME)

Real-time streaming inference

Auto retraining pipeline

🧠 Learning Outcomes

Handling highly imbalanced datasets

Feature engineering for financial data

Model evaluation for risk-focused systems

End-to-end ML workflow

👨‍💻 Author

#Rahul Pariary
Data Analyst | Aspiring Data Scientist
Passionate about solving real-world problems using data and machine learning.
