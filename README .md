# 🚨 Fraud Transaction Detection using Machine Learning

> End-to-end machine learning pipeline to detect fraudulent financial transactions using imbalance-aware modeling, feature engineering, and performance-focused evaluation.

---

## 📌 Project Overview

Financial fraud causes billions of dollars in losses every year.  
This project builds a **machine learning system** that identifies fraudulent transactions with **high recall**, ensuring suspicious activities are flagged early for investigation.

The project covers the complete data science lifecycle:

✔ Data cleaning  
✔ Exploratory Data Analysis (EDA)  
✔ Feature engineering  
✔ Model building & evaluation  
✔ Model saving for deployment  

---

## 🧠 Problem Statement

Fraud detection datasets are highly imbalanced — fraudulent transactions represent only a tiny fraction of total transactions.

**Goal:**  
Build a model that minimizes **false negatives** (missed fraud cases) while maintaining reasonable precision.

---

## 🏗️ Project Architecture

```
Raw Transaction Data
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Exploratory Data Analysis (EDA)
        │
        ▼
Feature Engineering
        │
        ▼
Train/Test Split
        │
        ▼
Logistic Regression Model
        │
        ▼
Evaluation (Recall, F1, ROC-AUC)
        │
        ▼
Model Serialization (.pkl)
```

---

## 📂 Repository Structure

```
Fraud-Transaction-Detection/
│
├── Accredian.ipynb                # Full project notebook
├── Fraud.csv                      # Dataset
├── fraud_detection_model.pkl      # Trained ML model
├── fraud_model_bundle.pkl         # Model + feature metadata
└── README.md                      # Project documentation
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|------|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib |
| ML Framework | Scikit-learn |
| Model Storage | Joblib |

---

## 📊 Exploratory Data Analysis Highlights

- Analyzed fraud vs non-fraud distribution
- Identified extreme class imbalance
- Studied transaction amount patterns
- Investigated balance change behavior before/after transactions
- Visualized feature relationships with fraud occurrence

---

## ⚙️ Model Development

### Model Used
- **Logistic Regression**
- Class weighting to handle imbalance
- Standard feature scaling

### Why Logistic Regression?
- Fast training
- Interpretable
- Strong baseline for anomaly/fraud detection
- Easy deployment in production environments

---

## 📈 Evaluation Metrics

Since this is a risk-focused system, **Recall** is prioritized.

| Metric | Score (Approx.) |
|--------|----------------|
| Accuracy | ~95% |
| Fraud Recall | ~98% |
| Precision | Lower (expected) |
| F1 Score | Balanced |

💡 High recall ensures fraudulent activities are rarely missed.

---

## 🔍 Key Insights

- Fraud cases exhibit unusual balance transitions.
- Specific transaction types are more fraud-prone.
- Model performs well despite severe class imbalance.

---

## 💾 Model Saving & Loading

```python
import joblib

# Save
joblib.dump(model, "fraud_detection_model.pkl")

# Load
model = joblib.load("fraud_detection_model.pkl")
```

---

## 🚀 Deployment Roadmap (Next Steps)

- Build REST API using Flask / FastAPI
- Real-time prediction endpoint
- Threshold tuning based on business risk
- Model monitoring & drift detection
- Dashboard integration for fraud analysts

---

## 📊 Business Impact

A high-recall fraud detection system can:

- Reduce financial losses
- Improve trust in financial platforms
- Assist fraud analysts with early alerts
- Automate risk assessment pipelines

---

## 🎯 Skills Demonstrated

- Data preprocessing & cleaning
- Handling imbalanced datasets
- Feature engineering
- ML model evaluation for real-world constraints
- End-to-end ML workflow design

---

## 👨‍💻 Author

**Rahul Pariary**  
Data Analyst | Aspiring Data Scientist  

Driven by data, problem-solving, and building intelligent systems that create real impact.

---

## ⭐ Support

If you found this project useful:

🌟 Star the repository  
🍴 Fork for improvements  
🤝 Contributions are welcome
