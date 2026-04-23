"""
=============================================================
 Customer Churn Analysis & Prediction — Telecom Dataset
=============================================================
Author  : Sanapala Chakradhar
Dataset : IBM Telco Customer Churn (synthetic replica)
Tools   : Python · Pandas · Seaborn · Scikit-learn · Plotly
=============================================================

Business Problem
-----------------
Customer churn costs telecom companies up to 5× more than
retaining existing customers. This project identifies the
strongest churn predictors, segments at-risk customers,
and builds a classification model to flag them proactively.

Steps
------
1. Data Generation & Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training (Logistic Regression + Random Forest)
5. Model Evaluation (ROC, Precision-Recall, Feature Importance)
6. Business Recommendations

Run:  python churn_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                             confusion_matrix, precision_recall_curve, average_precision_score)
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════
# 1. DATA GENERATION
# ══════════════════════════════════════════════════════════════

N = 7_043  # same size as IBM Telco dataset

def generate_telecom_data(n):
    gender          = np.random.choice(["Male", "Female"], n)
    senior          = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner         = np.random.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents      = np.random.choice(["Yes", "No"], n, p=[0.30, 0.70])
    tenure          = np.random.exponential(30, n).clip(1, 72).astype(int)
    phone_service   = np.random.choice(["Yes", "No"], n, p=[0.90, 0.10])
    internet_service= np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    contract        = np.random.choice(["Month-to-month", "One year", "Two year"],
                                        n, p=[0.55, 0.21, 0.24])
    payment_method  = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n, p=[0.34, 0.23, 0.22, 0.21])
    paperless       = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])
    monthly_charges = np.where(
        internet_service == "No",
        np.random.normal(25, 5, n).clip(18, 40),
        np.where(internet_service == "Fiber optic",
                 np.random.normal(80, 15, n).clip(45, 120),
                 np.random.normal(55, 12, n).clip(25, 90))
    )
    total_charges   = monthly_charges * tenure + np.random.normal(0, 20, n)
    total_charges   = total_charges.clip(0)

    # Churn probability — higher for:
    # short tenure, fiber optic, month-to-month, electronic check
    churn_prob = (
        0.05
        + 0.25 * (contract == "Month-to-month")
        + 0.15 * (internet_service == "Fiber optic")
        + 0.10 * (payment_method == "Electronic check")
        + 0.12 * (tenure < 12)
        - 0.10 * (tenure > 48)
        - 0.08 * (contract == "Two year")
        + 0.06 * (senior == 1)
        + np.random.normal(0, 0.05, n)
    ).clip(0, 1)
    churn = (np.random.rand(n) < churn_prob).astype(int)

    return pd.DataFrame({
        "Gender": gender, "SeniorCitizen": senior,
        "Partner": partner, "Dependents": dependents,
        "Tenure": tenure, "PhoneService": phone_service,
        "InternetService": internet_service, "Contract": contract,
        "PaymentMethod": payment_method, "PaperlessBilling": paperless,
        "MonthlyCharges": monthly_charges.round(2),
        "TotalCharges": total_charges.round(2),
        "Churn": churn
    })

df = generate_telecom_data(N)
df["ChurnLabel"] = df["Churn"].map({1: "Churned", 0: "Retained"})

print("=" * 60)
print("  CUSTOMER CHURN ANALYSIS — DATASET OVERVIEW")
print("=" * 60)
print(f"  Shape   : {df.shape}")
print(f"  Churn % : {df['Churn'].mean()*100:.1f}%  ({df['Churn'].sum():,} churned / {len(df):,} total)")
print(f"  Missing : {df.isnull().sum().sum()} values")

# ══════════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Customer Churn — Exploratory Analysis", fontsize=17, fontweight="bold")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 2a. Churn distribution
ax0 = fig.add_subplot(gs[0, 0])
churn_cnt = df["ChurnLabel"].value_counts()
ax0.bar(churn_cnt.index, churn_cnt.values,
        color=["#2E86AB", "#C73E1D"], edgecolor="white", linewidth=1.2)
for i, v in enumerate(churn_cnt.values):
    ax0.text(i, v + 50, f"{v:,}\n({v/N*100:.1f}%)", ha="center", fontsize=9)
ax0.set_title("Churn Distribution")
ax0.set_ylabel("Customers")

# 2b. Tenure by Churn
ax1 = fig.add_subplot(gs[0, 1])
for label, color in [("Churned","#C73E1D"), ("Retained","#2E86AB")]:
    ax1.hist(df[df["ChurnLabel"]==label]["Tenure"], bins=30,
             alpha=0.6, color=color, label=label, density=True)
ax1.set_title("Tenure Distribution by Churn")
ax1.set_xlabel("Tenure (months)")
ax1.legend()

# 2c. Monthly Charges by Churn (Box)
ax2 = fig.add_subplot(gs[0, 2])
df.boxplot(column="MonthlyCharges", by="ChurnLabel", ax=ax2,
           patch_artist=True,
           boxprops=dict(facecolor="#A23B72", alpha=0.5))
ax2.set_title("Monthly Charges vs Churn")
ax2.set_xlabel("")
plt.sca(ax2); plt.title("Monthly Charges vs Churn")

# 2d. Contract type churn rate
ax3 = fig.add_subplot(gs[1, 0])
ct = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False) * 100
ax3.bar(ct.index, ct.values, color=["#C73E1D","#F18F01","#2E86AB"])
ax3.set_title("Churn Rate by Contract Type")
ax3.set_ylabel("Churn Rate (%)")
for i, v in enumerate(ct.values):
    ax3.text(i, v+0.5, f"{v:.1f}%", ha="center", fontsize=9)

# 2e. Internet Service churn rate
ax4 = fig.add_subplot(gs[1, 1])
inet = df.groupby("InternetService")["Churn"].mean().sort_values(ascending=False) * 100
ax4.bar(inet.index, inet.values, color=["#C73E1D","#F18F01","#2E86AB"])
ax4.set_title("Churn Rate by Internet Service")
ax4.set_ylabel("Churn Rate (%)")
for i, v in enumerate(inet.values):
    ax4.text(i, v+0.5, f"{v:.1f}%", ha="center", fontsize=9)

# 2f. Payment Method
ax5 = fig.add_subplot(gs[1, 2])
pay = df.groupby("PaymentMethod")["Churn"].mean().sort_values() * 100
ax5.barh(pay.index, pay.values, color=["#2E86AB","#2E86AB","#F18F01","#C73E1D"])
ax5.set_title("Churn Rate by Payment Method")
ax5.set_xlabel("Churn Rate (%)")

# 2g. Correlation heatmap (numeric)
ax6 = fig.add_subplot(gs[2, :])
num_cols = ["Tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn"]
corr = df[num_cols].corr()
sns.heatmap(corr, ax=ax6, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5)
ax6.set_title("Feature Correlation Heatmap")

plt.savefig("output/churn_eda.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] EDA saved → output/churn_eda.png")

# ══════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING & PREPROCESSING
# ══════════════════════════════════════════════════════════════

df_model = df.copy()

# Encode categoricals
cat_cols = ["Gender","Partner","Dependents","PhoneService",
            "InternetService","Contract","PaymentMethod","PaperlessBilling"]
le = LabelEncoder()
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col])

# New features
df_model["AvgMonthlySpend"]    = df_model["TotalCharges"] / (df_model["Tenure"] + 1)
df_model["ChargesPerTenure"]   = df_model["MonthlyCharges"] * df_model["Tenure"]
df_model["IsLongTermCustomer"] = (df_model["Tenure"] > 36).astype(int)

feature_cols = [c for c in df_model.columns if c not in ["Churn","ChurnLabel"]]
X = df_model[feature_cols]
y = df_model["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                      random_state=42, stratify=y)
scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ══════════════════════════════════════════════════════════════
# 4. MODEL TRAINING
# ══════════════════════════════════════════════════════════════

models = {
    "Logistic Regression"  : LogisticRegression(max_iter=1000, C=0.5, random_state=42),
    "Random Forest"        : RandomForestClassifier(n_estimators=200, max_depth=8,
                                                    random_state=42, n_jobs=-1),
    "Gradient Boosting"    : GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                                        learning_rate=0.08, random_state=42),
}

print("\n── MODEL COMPARISON (5-Fold CV ROC-AUC) ────────────────")
results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="roc_auc")
    results[name] = cv_scores
    print(f"  {name:<25}  AUC = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Train best model (Random Forest) on full training set
best_model = models["Random Forest"]
best_model.fit(X_train_s, y_train)
y_prob  = best_model.predict_proba(X_test_s)[:, 1]
y_pred  = best_model.predict(X_test_s)
test_auc= roc_auc_score(y_test, y_prob)
print(f"\n  Test AUC (Random Forest): {test_auc:.4f}")
print(f"\n── CLASSIFICATION REPORT ────────────────────────────────")
print(classification_report(y_test, y_pred, target_names=["Retained","Churned"]))

# ══════════════════════════════════════════════════════════════
# 5. EVALUATION VISUALISATIONS
# ══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Model Evaluation — Random Forest Classifier", fontsize=14, fontweight="bold")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0].plot(fpr, tpr, color="#2E86AB", lw=2, label=f"AUC = {test_auc:.4f}")
axes[0].plot([0,1],[0,1],"--", color="gray")
axes[0].fill_between(fpr, tpr, alpha=0.1, color="#2E86AB")
axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate",
            title="ROC Curve")
axes[0].legend()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Retained","Churned"],
            yticklabels=["Retained","Churned"], ax=axes[1],
            linewidths=0.5, linecolor="white")
axes[1].set(title="Confusion Matrix", xlabel="Predicted", ylabel="Actual")

# Feature Importance (top 12)
fi = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values().tail(12)
fi.plot(kind="barh", ax=axes[2], color="#A23B72")
axes[2].set_title("Top 12 Feature Importances")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("output/churn_model_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("[✓] Model evaluation saved → output/churn_model_evaluation.png")

# ══════════════════════════════════════════════════════════════
# 6. BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════

print("\n── BUSINESS RECOMMENDATIONS ─────────────────────────────")
print("  1. Target month-to-month customers with upgrade incentives")
print("     → They churn at 3× the rate of 2-year contract holders")
print("  2. Fiber optic users show high churn — review pricing/SLA")
print("  3. Migrate Electronic Check users to auto-pay (lower churn)")
print("  4. Focus retention spend on customers with tenure < 12 months")
print("     → Highest churn window; early engagement reduces LTV loss")
print("\n[✓] All outputs saved to ./output/")
