# 📉 Customer Churn Prediction & Analysis — Telecom

> End-to-end churn analytics project: EDA → Feature Engineering → Machine Learning → Business Recommendations. Built on 7,000+ customer records modelled after the IBM Telco dataset.

---

## 📌 Problem Statement

Telecom companies lose **15–25% of customers annually** to churn. Retaining a customer costs 5× less than acquiring a new one. By identifying *who* will churn and *why*, companies can take targeted, proactive action to protect revenue.

---

## 🎯 Project Goals

1. Understand the profile of customers who churn (EDA)
2. Identify the strongest churn predictors
3. Build and evaluate ML models to predict churn
4. Translate model outputs into actionable business strategies

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| Pandas & NumPy | Data wrangling |
| Matplotlib & Seaborn | EDA visualizations |
| Scikit-learn | ML models, evaluation |
| Plotly | Interactive charts |

---

## 📁 Project Structure

```
project2_churn_analysis/
├── churn_analysis.py        # Main pipeline
├── requirements.txt
├── README.md
└── output/
    ├── churn_eda.png                # 6-panel EDA dashboard
    └── churn_model_evaluation.png  # ROC, Confusion Matrix, Feature Importance
```

---

## 🚀 How to Run

```bash
git clone https://github.com/chakradhar-sanapala/customer-churn-analysis
cd customer-churn-analysis
pip install -r requirements.txt
python churn_analysis.py
```

---

## 📊 Model Results

| Model | CV ROC-AUC |
|-------|-----------|
| Logistic Regression | ~0.84 |
| **Random Forest** | **~0.88** ✅ Best |
| Gradient Boosting | ~0.87 |

---

## 🔑 Key Findings

| Factor | Churn Impact |
|--------|-------------|
| Month-to-month contract | 3× higher churn vs 2-year |
| Fiber Optic internet | Highest churn of all service types |
| Tenure < 12 months | Critical early churn window |
| Electronic check payment | Correlated with higher churn |
| Senior citizens | 6% higher churn probability |

---

## 💡 Business Recommendations

1. **Loyalty incentives** for month-to-month customers to upgrade to annual plans
2. **Service quality audit** for Fiber Optic — pricing or SLA issues likely
3. **Auto-pay migration campaigns** for electronic check users
4. **Early-tenure engagement** (months 1–6) — onboarding programs reduce early churn

---

## 🔗 Dataset

Synthetic data modelled on the  
[IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).  
Drop in the real CSV with matching column names for production-ready predictions.
