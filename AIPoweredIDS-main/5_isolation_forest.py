import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import kagglehub
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import( classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, f1_score, precision_score, recall_score)

#style
sns.set_theme(style="whitegrid", palette="muted")
FRAUD_PALETTE = {0: "#4878CF", 1: "#D65F5F"}
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs("metrics", exist_ok=True)

def save_fig(name):
    plt.savefig(os.path.join(FIG_DIR, name), dpi=300, bbox_inches="tight")
    plt.close()
    print(f" saved -> figures/{name}")

#load data
print("Loading dataset...")
path = kagglehub.dataset_download("arunavakrchakraborty/financial-payment-services-fraud-data")
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
df = pd.read_csv(os.path.join(path, csv_files[0]))
print(f" Loaded: {df.shape[0]:,} rows")

"""PREPROCESSING"""
df_filtered = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

df_filtered["orig_balance_delta"] = (df_filtered["newbalanceOrig"] - df_filtered["oldbalanceOrg"])
df_filtered["dest_balance_delta"] = (df_filtered["newbalanceDest"] - df_filtered["oldbalanceDest"])
df_filtered["orig_drained"] = (df_filtered["newbalanceOrig"] == 0).astype(int)
df_filtered["type_encoded"] = (df_filtered["type"] == "TRANSFER").astype(int)

FEATURES = [
    "step", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "orig_balance_delta", "dest_balance_delta",
    "orig_drained", "type_encoded"
]
TARGET = "isFraud"

X = df_filtered[FEATURES]
y = df_filtered[TARGET]

#standard stratified split with the same params from phase 2 and 3 to make sure the test set is identical
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f" Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
print(f" Fraud in test test: {y_test.sum():,} ({y_test.mean()*100:.3f}%)")

""" ISOLATION FOREST - TRAINING """
# take only legit transactions from training set
X_train_legit = X_train_scaled[y_train == 0]
print(f"\n Training Isolation Forest on {X_train_legit.shape[0]:,}"
      f"only legitimate transactions")
print(f" Fraud cases not included in training: {(y_train == 1).sum():,}")

fraud_rate = y_train.mean()
print(f" contamination set to know fraud rate: {fraud_rate:.6f}")

iso = IsolationForest(
    n_estimators=200,
    contamination=fraud_rate,
    max_samples="auto",
    random_state=42,
    n_jobs=-1
)

iso.fit(X_train_legit)
print("Isolation Forest trained on legitimate transactions only")

"""PREDICTION AND SCORING

-ve scores = more anolalous and +ve scores = more normal but will convert -1 to 1 to fraud and 1 to 0 for legit
to compare against the isFaud labels"""

#negate the anomoly scores so higher = more suspicicious
anomaly_scores = -iso.decision_function(X_test_scaled)

# -1 anomoly -> 1 fraud and 1 normal -> 0 legit
raw_predictions = iso.predict(X_test_scaled)
y_pred = np.where(raw_predictions == -1, 1, 0)

print(f"\n Transactions flagged as anomalous: {y_pred.sum():,}"
      f"({y_pred.mean()*100:.2f}% of test set)")

"""EVALUATION"""

roc_auc = roc_auc_score(y_test, anomaly_scores)
avg_prc = average_precision_score(y_test, anomaly_scores)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "=" * 55)
print(" ISOLATION FOREST RESULTS")
print("=" * 55)
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"], zero_division=0))
print(f" ROC-AUC : {roc_auc:.4f}")
print(f" PR-AUC : {avg_prc:.4f}")
print("=" * 55)

""" FIG 24 - CONFUSION MATRIX """

cm = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(14,5))
fig.suptitle("Figure 24 - Isolation Forest: Confusion Matrix", fontsize=13, fontweight="bold")

for ax, data, fmt, title in zip(
    axes,
    [cm, cm_pct],
    ["d", ".1f"],
    ["Raw Counts", "Raw Normalised (%)"]
):
    sns.heatmap(
        data, annot=True, fmt=fmt, cmap="Oranges", ax=ax,
        xticklabels=["Pred: Legit", "Pred: Fraud"],
        yticklabels=["Actual: Legit", "Actual: Fraud"],
        cbar=False
    )
    ax.set_title(title)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

plt.tight_layout()
save_fig("fig24_confusion_matrix_if.png")

"""FIG 24 - ROC CURVE for all 3 models"""

fpr_if, tpr_if, _ = roc_curve(y_test, anomaly_scores)

plt.figure(figsize=(9,6))
plt.plot(fpr_if, tpr_if, color="#E88B2E", lw=2, label=f"Isolation Forest (AUC = {roc_auc:.4f})")

#get the phase 2 and 3 AUC vals from the metrics
lr_path = "metrics/baseline_lr.csv"
xgb_path = "metrics/xgboost.csv"

if os.path.exists(lr_path):
    lr_auc = pd.read_csv(lr_path)["roc_auc"].values[0]
    plt.plot([], [], color="#4878CF", lw=2, label=f"Logistic Regression (AUC = {lr_auc:.4f})")

if os.path.exists(xgb_path):
    xgb_auc = pd.read_csv(xgb_path)["roc_auc"].values[0]
    plt.plot([], [], color= "#D65F5F", lw=2, label=f"XGBoost (AUC = {xgb_auc:.4f})")

plt.plot([0,1], [0,1], "k--", lw=1.5, label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Figure 25 - ROC Curve: All Three Models")
plt.legend(loc="lower right")
plt.tight_layout()
save_fig("fig25_roc_all_models.png")

"""FIGURE 26 - PRECISION RECALL CURVE """

precision_vals, recall_vals, _ = precision_recall_curve(y_test, anomaly_scores)

plt.figure(figsize=(8,6))
plt.plot(recall_vals, precision_vals, color="#E88B2E", lw=2, label=f"Isolation Forest (AP = {avg_prc:.4f})")
plt.axhline(y=y_test.mean(), color="k", linestyle="--", lw=1.5, label=f"Random baseline ({y_test.mean():.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Figure 26 - Precision-Recall Curve: Isolation Forest")
plt.legend(loc="upper right")
plt.tight_layout()
save_fig("fig26_pr_curve_if.png")

"""FIG 27 - ANOMALY SCORE DISTRIBUTION -> only in IF to show what the model has learned"""

scores_legit = anomaly_scores[y_test.values == 0]
scores_fraud = anomaly_scores[y_test.values == 1]

plt.figure(figsize=(10, 5))
plt.hist(scores_legit, bins=100, alpha=0.55, color="#4878CF",
         density=True, label=f"Legitimate (n={len(scores_legit):,})")

plt.hist(scores_fraud, bins=100, alpha=0.55, color="#D65F5F",
         density=True, label=f"Fraudulent (n={len(scores_fraud):,})")

plt.xlabel("Anomaly Score (higher = more anomalous)")
plt.ylabel("Density")
plt.title("Figure 27 — Isolation Forest Anomaly Score Distribution\n"
          "(separation = unsupervised signal quality)")
plt.legend()
plt.tight_layout()
save_fig("fig27_anomaly_score_distribution.png")

""" FIG 28 FALSE POSITIVE ANALYSIS"""

test_df_if = X_test.copy()
test_df_if["y_true"]        = y_test.values
test_df_if["y_pred"]        = y_pred
test_df_if["anomaly_score"] = anomaly_scores

fp_if = test_df_if[(test_df_if["y_true"] == 0) & (test_df_if["y_pred"] == 1)]
tn_if = test_df_if[(test_df_if["y_true"] == 0) & (test_df_if["y_pred"] == 0)]

n_legit_if = (test_df_if["y_true"] == 0).sum()
print(f"\n  Isolation Forest false positives: {len(fp_if):,} "
      f"({len(fp_if)/n_legit_if*100:.2f}% of legit transactions)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Figure 28 — False Positive Analysis: Isolation Forest\n"
    "(compare against Figure 16 — XGBoost Phase 3)",
    fontsize=12, fontweight="bold"
)

cap_if = test_df_if["amount"].quantile(0.99)

axes[0].hist(
    np.log1p(tn_if[tn_if["amount"] <= cap_if]["amount"]),
    bins=60, alpha=0.6, color="#4878CF", density=True, label="True Negatives"
)
axes[0].hist(
    np.log1p(fp_if[fp_if["amount"] <= cap_if]["amount"]),
    bins=60, alpha=0.6, color="#D65F5F", density=True, label="False Positives"
)
axes[0].set_title("Amount Distribution: FP vs True Negatives")
axes[0].set_xlabel("log(Amount + 1)")
axes[0].set_ylabel("Density")
axes[0].legend()

axes[1].hist(fp_if["anomaly_score"], bins=40,
             color="#E88B2E", edgecolor="white")
axes[1].set_title("Anomaly Score Distribution for False Positives\n"
                  "(how anomalous are the misclassified legitimate transactions?)")
axes[1].set_xlabel("Anomaly Score")
axes[1].set_ylabel("Count")

plt.tight_layout()
save_fig("fig28_false_positive_if.png")

"""FIG 29 - MISSED FRAUD OVERLAP ANALYSIS"""
# does the IF catch fraud that the xgboost missed vice versa
# can't load xgboost predictions from the metrics so have to get xgboost predictions for the overlap only
#locked threshold from phase 3

import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    early_stopping_rounds=20
)

# internal val set for early stopping
X_tr_xgb, X_val_xgb, y_tr_xgb, y_val_xgb = train_test_split(
    X_train_scaled, y_train,
    test_size=0.1, random_state=42, stratify=y_train
)
xgb_model.fit(
    X_tr_xgb, y_tr_xgb,
    eval_set=[(X_val_xgb, y_val_xgb)],
    verbose=False,
)

xgb_threshold = pd.read_csv("metrics/xgboost.csv")["threshold"].values[0]
xgb_probs     = xgb_model.predict_proba(X_test_scaled)[:, 1]
xgb_preds     = (xgb_probs >= xgb_threshold).astype(int)

# overlap analysis based on fraud cases only
fraud_idx = y_test.values == 1
fraud_analysis = pd.DataFrame({
    "xgb_caught": xgb_preds[fraud_idx] == 1,
    "if_caught":  y_pred[fraud_idx] == 1
})

both_caught  = (fraud_analysis["xgb_caught"] & fraud_analysis["if_caught"]).sum()
xgb_only     = (fraud_analysis["xgb_caught"] & ~fraud_analysis["if_caught"]).sum()
if_only      = (~fraud_analysis["xgb_caught"] & fraud_analysis["if_caught"]).sum()
neither      = (~fraud_analysis["xgb_caught"] & ~fraud_analysis["if_caught"]).sum()
total_fraud  = fraud_idx.sum()

print(f"\n  MISSED FRAUD OVERLAP ANALYSIS")
print(f" Total fraud cases in test set : {total_fraud:,}")
print(f" A — Caught by both : {both_caught:,} "
      f"({both_caught/total_fraud*100:.1f}%)")
print(f" B — XGBoost only : {xgb_only:,} "
      f"({xgb_only/total_fraud*100:.1f}%)")
print(f" C — Isolation Forest only : {if_only:,} "
      f"({if_only/total_fraud*100:.1f}%)")
print(f" D — Missed by both : {neither:,} "
      f"({neither/total_fraud*100:.1f}%)")

fig, ax = plt.subplots(figsize=(8, 6))

categories  = ["Both\n(A)", "XGBoost\nOnly (B)",
                "Isolation Forest\nOnly (C)", "Neither\n(D)"]
counts      = [both_caught, xgb_only, if_only, neither]
bar_colors  = ["#6ACC65", "#4878CF", "#E88B2E", "#D65F5F"]

bars = ax.bar(categories, counts, color=bar_colors, edgecolor="white", width=0.5)
ax.set_ylabel("Number of Fraud Cases")
ax.set_title("Figure 29 — Fraud Detection Overlap: XGBoost vs Isolation Forest\n"
             "(how many fraud cases does each model uniquely catch?)")

for bar, count in zip(bars, counts):
    pct = count / total_fraud * 100
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 5,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
save_fig("fig29_fraud_overlap_xgb_vs_if.png")

"""FIG 30 - THREE MODEL PRECISION/RECALL/F1 BAR CHART
- show all three models on the same axes """

lr_metrics  = pd.read_csv("metrics/baseline_lr.csv").iloc[0] \
              if os.path.exists("metrics/baseline_lr.csv") else None
xgb_metrics = pd.read_csv("metrics/xgboost.csv").iloc[0] \
              if os.path.exists("metrics/xgboost.csv") else None

model_names = ["Logistic\nRegression", "XGBoost", "Isolation\nForest"]
metric_names = ["Precision", "Recall", "F1"]

if lr_metrics is not None and xgb_metrics is not None:
    values = {
        "Precision": [lr_metrics["precision"], xgb_metrics["precision"], prec],
        "Recall": [lr_metrics["recall"], xgb_metrics["recall"], rec],
        "F1": [lr_metrics["f1"], xgb_metrics["f1"], f1],
    }

    x = np.arange(len(model_names))
    width = 0.25
    colors_bar = ["#4878CF", "#D65F5F", "#6ACC65"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (metric, color) in enumerate(zip(metric_names, colors_bar)):
        bars = ax.bar(x + i * width, values[metric], width,
                      label=metric, color=color, edgecolor="white", alpha=0.85)
        
        for bar, val in zip(bars, values[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title("Figure 30 — Precision / Recall / F1: All Three Models\n"
                 "(supervised vs unsupervised trade off)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    save_fig("fig30_three_model_comparison.png")

"""SAVE THE METRICS"""

metrics = {
    "model": "Isolation Forest",
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1": round(f1, 4),
    "roc_auc": round(roc_auc, 4),
    "pr_auc": round(avg_prc, 4),
    "threshold": "contamination",
    "note": "trained on legitimate transactions only"
}

pd.DataFrame([metrics]).to_csv("metrics/isolation_forest.csv", index=False)
print(f"\n Metrics saved → metrics/isolation_forest.csv")
print(f"\nAll Phase 5 figures saved to /figures")