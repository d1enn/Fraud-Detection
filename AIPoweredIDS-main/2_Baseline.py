import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import kagglehub
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, f1_score, precision_score, recall_score)

#keep the styling consistent with the EDA file
sns.set_theme(style="whitegrid", palette="muted")
FRAUD_PALETTE = {0: "#4878CF", 1: "D65F5F"}
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

def save_fig(name):
    plt.savefig(os.path.join(FIG_DIR, name), dpi=300, bbox_inches="tight")
    plt.close()
    print(f" ✓ saved → figures/{name}")

""" 1- LOAD THE DATA - uses the same cache"""

print ("Loading dataset...")
path = kagglehub.dataset_download("arunavakrchakraborty/financial-payment-services-fraud-data")
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
df = pd.read_csv(os.path.join(path, csv_files[0]))
print(f" Loaded: {df.shape[0]:,} rows")

""" 2- PREPROCESSING PIPELINE - METHODOLOY"""

print("\nPreprocessing...")

# STEP 1: filter to fraud revelevant transactions only so based on EDA that is TRANSFER and CASH_OUT
df_filtered = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
print(f" After type filter: {df_filtered.shape[0]:,} rows" f"({df_filtered['isFraud'].mean()*100:.3f}% fraud)")

# STEP 2: ENGINEER BALANCE DELTA FEATURES
# raw balances < inofrmative than changes in balance
#based on EDA transacrtion that drains an account to 0 is a strong signal of fraud -> so encode this behaviuor

df_filtered["orig_balance_delta"] = (df_filtered["newbalanceOrig"] - df_filtered["oldbalanceOrg"])
df_filtered["dest_balance_delta"] = (df_filtered["newbalanceDest"] - df_filtered["oldbalanceDest"])

#did the account reach 0 after the transaction -> then flag it
df_filtered["orig_drained"] = (df_filtered["newbalanceOrig"] == 0).astype(int)

# STEP 3: SELECT FINAL FEATURE SET
#DROP - nameOrig, nameDest, isFlaggedFraud, type
#make type binary TRANSFER = 1, CASHOUT = 0

df_filtered["type_encoded"] = (df_filtered["type"] == "TRANSFER").astype(int)

FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "orig_balance_delta",
    "dest_balance_delta",
    "orig_drained",
    "type_encoded"
]
TARGET = "isFraud"

X = df_filtered[FEATURES]
y = df_filtered[TARGET]

print(f"\ Feature set: {FEATURES}")
print(f"Class distribution after filtering:")
print(f"Fraud: {y.sum():,} ({y.mean()*100:.3f}%)")
print(f"Legit: {(y==0).sum():,}")

# STEP 4: TRAIN/TEST SPLIT
#stratify=y makes sure that the fraud ratio is preserved in both splits -> important with imbalance data, state in the methodology

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")

# STEP 5: SCALE FEATURES
#logisitc regression is sensitive to feature scale
# fit oNLY on the training data -> on full set would be datat leakage

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""LOGISTIC REGRESSION - BASELINE MODEL"""
#class_weight = 'balanced' adjust for imbaalnce without resampling

print("\nTraining Logistic Regression Baseline...")

lr = LogisticRegression(
    class_weight="balanced", # compensates for the 0.13% fraud rate
    max_iter=1000,
    random_state=42,
    n_jobs=1
)
lr.fit(X_train_scaled, y_train)

y_pred = lr.predict(X_test_scaled)
y_pred_prob = lr.predict_proba(X_test_scaled)[:, 1]

""" EVALUATION - DISSO FIGURES"""
print("\n" + "=" * 55)
print("BASELINE RESULTS - LOGISTIC REGRESSION")
print("=" * 55)
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

roc_auc = roc_auc_score(y_test, y_pred_prob)
avg_prc = average_precision_score(y_test, y_pred_prob)
print(f" ROC-AUC Score          : {roc_auc:.4f}")
print(f"Avg Precision (PR-AUC)  : {avg_prc:.4f}")
print("=" * 55)

# FIGURE 7 - CONFUSION MATRIX
# shows false +/- explicityly -> discuss the cost of each error type. false neg = missing fraud vs false pos = blocking legit transaction

cm = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figure 7 - Logistic Regression: Confusion Matrix", 
             fontsize=14, fontweight="bold")

for ax, data, fmt, title in zip(
    axes,
    [cm, cm_pct],
    ["d", ".1f"],
    ["Raw Counts", "Row-Normalised (%)"]
):
    sns.heatmap(
        data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
        xticklabels=["Pred: Legit", "Pred: Fraud"],
        yticklabels=["Actual: Legit", "Actual: Fraud"],
        cbar=False
    )

    ax.set_title(title)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

plt.tight_layout()
save_fig("fig7_confusion_matric_lr.png")

#FIGURE 8 - ROC CURVE
# diagonal = random classifier and the curve should be above it
#ROC-AUC is the number i will compare across all the 3 models

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color="#D65F5F", lw=2, label=f"logistic Regression (AUC = {roc_auc:.4f})")
plt.plot([0,1], [0,1], "k--", lw=1.5, label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positve Rate (Recall)")
plt.title("Figure 8 - ROC Curve: Logistic Regression Baseline")
plt.legend(loc="lower right")
plt.tight_layout()
save_fig("Fig8_roc_curve_lr.png")

#FIGURE 9 - PRECISION RECALL CURVE
# more informative that ROC for imbalanced datasets
# high ROC-AUC can still have bad precision on fraud -> so in method argue why PR-AUC is the actual metric that matters here

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color="#D65F5F", lw=2, label=f"Logistic Regression (AP = {avg_prc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Figure 9 - Precision Recall Curve: Logistic Regression Baseline")
plt.legend(loc="upper right")
plt.tight_layout()
save_fig("fig9_pr_curve_lr.png")

#FIGURE 10 - FEATURE COEFFICIENTS 
#shows which features dirve the logistic regression decison -> use to justify feature enginneering choices

coef_df = pd.DataFrame({
    "feature" : FEATURES,
    "coefficient" : lr.coef_[0]
}).sort_values("coefficient", key=abs, ascending=True)

colors_coef = ["#D65F5F" if c > 0 else "#4878CF" for c in coef_df["coefficient"]]

plt.figure(figsize=(10,6))
plt.barh(coef_df["feature"], coef_df["coefficient"], color=colors_coef)
plt.axvline(x=0, color="black", linewidth=0.8)
plt.xlabel("Coefficient Value")
plt.title("Figure 1- - Logistic Regression Feature Coefficients\n"
         "(red = pushes towards fraud, blue = pushes towards legitimate)" )
plt.tight_layout()
save_fig("fig10_lr_coefficients.png")

print("\nAll Phase 2 figures saved to /figures")
print("Complete baseline results section.")

os.makedirs("metrics", exist_ok=True)

metrics = {
    "model" : "Logistic Regression",
    "precision" : round(precision_score(y_test, y_pred), 4),
    "recall" : round(recall_score(y_test, y_pred), 4),
    "f1" : round(f1_score(y_test, y_pred), 4),
    "roc_auc" : round(roc_auc, 4),
    "pr_auc" : round(avg_prc, 4)
}

pd.DataFrame([metrics]).to_csv("metrics/baseline_lr.csv", index=False)
print("\n ✓ Metrics saved → metrics/baseline_lr.csv ")
print(f"Precision : {metrics['precision']}")
print(f"Recall    : {metrics['recall']}")
print(f"F1 Score  : {metrics['f1']}")
print(f"ROC-AUC   : {metrics['roc_auc']}")
print(f"PR-AUC    : {metrics['pr_auc']}")