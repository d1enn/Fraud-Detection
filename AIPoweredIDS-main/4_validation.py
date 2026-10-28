import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import kagglehub
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ( classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, f1_score, precision_score, recall_score)

#styling
sns.set_theme(style="whitegrid", palette="muted")
FRAUD_PALETTE = {0: "#4878CF", 1: "#D65F5F"}
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs("metrics", exist_ok = True)

def save_fig(name):
    plt.savefig(os.path.join(FIG_DIR, name), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved -> figures/{name}")

"""LOAD DATA"""

print("Loading dataset...")
path = kagglehub.dataset_download("arunavakrchakraborty/financial-payment-services-fraud-data")
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
df = pd.read_csv(os.path.join(path, csv_files[0]))
print(f" Loaded: {df.shape[0]:,} rows")

""" PREPROCESSING - identical to baseline and xgboost """

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

"""TEMPORAL SPLIT - checkss validity"""
#SPLIT_STEP = 600 # 80/20 split
SPLIT_STEP = int(df_filtered["step"].quantile(0.8)) # 80th percentile split instead of hardcoded
print(f" Split Step (80th percentile): {SPLIT_STEP}")

train_df = df_filtered[df_filtered["step"] <= SPLIT_STEP].copy()
test_df = df_filtered[df_filtered["step"] > SPLIT_STEP].copy()

# no shuffling
X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_test = test_df[FEATURES]
y_test = test_df[TARGET]

print(f"\n Temporal split at step {SPLIT_STEP}")
print(f" Train: steps 1-{SPLIT_STEP} -> {len(X_train):,} rows | "
      f"fraud rate: {y_train.mean()*100:.4f}%")
print(f" Test: steps {SPLIT_STEP+1}-743 -> {len(X_test):,} rows | "
      f"fraud rate: {y_test.mean()*100:.4f}%")
print(f"\n Fraud counts:")
print(f"    Train fraud: {y_train.sum():,}")
print(f"    Test fraud: {y_test.sum():,}")

"""SCALE - training on partition only

method - scalar fit on training partition -> putting it on one full datatset before splitting = data leakage
-> the model would have indirect knowledge of the statistic properties for future transactions -> error in fraud detection"""

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) #not fit

""" TRAIN XGBOOST ON TEMPORAL TRAINING PARTITION """

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

print(f"\n scale_pos_weight: {scale_pos_weight:.2f}")
print("\nTraining XGBoost on temporal training partition...")

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

# validation is set for early stopping from only the training partition 
# steps 1-600 (no future info) -> not in the test set
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train # safe bc within the training partiticiton
)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)
print(f" Best iteration: {model.best_iteration}")

""" THRESHOLD - locked from phase 3"""

xgb_metrics_path = "metrics/xgboost.csv"

if not os.path.exists(xgb_metrics_path):
    raise FileNotFoundError(
        "metrics/xgboost.csv not found."
        "Run 3_xgboost.py before this file -"
        "the threshold must come from Phase 3, not be re-derived here."
    )

locked_threshold = pd.read_csv(xgb_metrics_path)["threshold"].values[0]
print(f"\n Locked threshold (from Phase 3): {locked_threshold:.4f}")
print(f" This value is NOT re-tuned on temporal test data.")

y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred_tuned = (y_pred_prob >= locked_threshold).astype(int)

"""EVALUATION"""
roc_auc = roc_auc_score(y_test, y_pred_prob)
avg_prc = average_precision_score(y_test, y_pred_prob)
prec = precision_score(y_test, y_pred_tuned)
rec = recall_score(y_test, y_pred_tuned)
f1 = f1_score(y_test, y_pred_tuned)

print("\n" + "=" * 55)
print("TEMPORAL VALIDATION RESULTS")
print("=" * 55)
print(classification_report(y_test, y_pred_tuned, target_names=["Legitimate", "Fraud"]))
print(f" ROC-AUC : {roc_auc:.4f}")
print(f" PR-AUC : {avg_prc:.4f}")
print(f" Threshold: {locked_threshold:.4f} (locked from Phase 3)")
print("=" * 55)

#load phase 3 random split metrics for comparison
phase3 = pd.read_csv(xgb_metrics_path).iloc[0]
print("\n COMPARISON: Random Split (Phase 3) vs Temporal Split (Phase 4)")
print(f" {'Metric':<12} {'Phase 3':>10} {'Phase 4':>10} {'Delta':10}")
print(f" {'-'*44}")

for metric, val in [("Precision", prec), ("Recall", rec),
                     ("F1", f1), ("ROC-AUC", roc_auc), ("PR-AUC", avg_prc)]:
    key = metric.lower().replace("-", "_").replace("roc_auc", "roc_auc").replace("pr_auc", "pr_auc")
    p3val = phase3[key] if key in phase3 else float("nan")
    delta = val - p3val 
    direction = "▼" if delta < 0 else "▲"
    print(f" {metric:<12} {p3val:>10.4f} {val:>10.4f} {direction}{abs(delta):>8.4f}")


"""fig 17 - confusion matrix:temporal test set
- compare against figure 11 (random split)
- false negs
- concept drift """

cm = confusion_matrix(y_test, y_pred_tuned)

fig, axes = plt.subplots(1,2, figsize=(14,5))
fig.suptitle("Figure 17 - Temporal Validation: Confusion Matrix",
fontsize=13, fontweight="bold")

cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

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
save_fig("fig17_confusion_matrix_temporal.png")

"""FIG 18 - FRAUD RATE ACROSS THE TIME STEPS 
shows if fraud patterns are the same across time."""

step_fraud = df_filtered.groupby("step")["isFraud"].mean() * 100
y_max = step_fraud.max() * 1.2
step_max = int(df_filtered["step"].max())

plt.figure(figsize=(12, 5))
plt.plot(step_fraud.index, step_fraud.values, color="#D65F5F", lw=1.5, alpha=0.8)
plt.axvline(x=SPLIT_STEP, color="black", linestyle="--", lw=2, label=f"Train/test split (step {SPLIT_STEP})")
#plt.fill_betweenx([0, step_fraud.max() * 1.2],
   #               0, SPLIT_STEP, alpha=0.07, color="#4878CF", label="Training region")

#training region shading
plt.fill_betweenx([0, y_max], 0, SPLIT_STEP, alpha=0.07, color="#4878CF", label="Training region")
#test region
plt.fill_betweenx([0, y_max], SPLIT_STEP, step_max, alpha=0.07, color="#D65F5F", label="Test region")

plt.xlabel("Time Step")
plt.ylabel("Fraud Rate (%)")
plt.title("Figure 18 - Fraud Rate Across Time Steps \n"
          "(non-stationarity justifies temporal over random splitting)")
plt.legend()
plt.tight_layout()
save_fig("fig18_fraud_rate_over_time.png")

"""FIGURE 19 - ROC Curve: Temporal vs Random Split -> visual represenatative"""

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color="#D65F5F", lw=2, label=f"Temporal Split (AUC = {roc_auc:.4f})")

phase3_auc = phase3["roc_auc"] if "roc_auc" in phase3.index else None
if phase3_auc:
    plt.plot([], [], color="#4878CF", lw=2,
    label=f"Random Split - Phase 3 (AUC = {phase3_auc:.4f})")

plt.plot([0,1], [0,1], "k--", lw=1.5, label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Figure 19 - ROC Curve: Temporal Split vs Random Split")
plt.legend(loc="lower right")
plt.tight_layout()
save_fig("fig19_roc_temporal_vs_random.png")

"""FIGURE 20 - Precision Recall Curve: Temporal vs Random Split"""
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_prob)

plt.figure(figsize=(8,6))
plt.plot(recall_vals, precision_vals, color="#D65F5F", lw=2,
         label=f"Temporal Split (AP = {avg_prc:.4f})")

phase3_prauc = phase3["pr_auc"] if "pr_auc" in phase3.index else None
if phase3_prauc:
    plt.plot([], [], color="#4878CF", lw=2,
    label=f"Random Split - Phase 3 (AP = {phase3_prauc:.4f})")

plt.axhline(y=y_test.mean(), color="k", linestyle="--", lw=1.5,
            label=f"Random baseline ({y_test.mean():.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Figure 20 - Precision-Recall Curve: Temporal Split vs Random Split")
plt.legend(loc="upper right")
plt.tight_layout()
save_fig("fig20_pr_temporal_vs_random.png")

"""FIGURE 21 - False Positive Analysis: Temporal Test Set"""
# for direct comprison between phase 3 and 4 between the positive behaviour on random vs temporal

test_analysis_df = X_test.copy()
test_analysis_df["y_true"] = y_test.values
test_analysis_df["y_pred"] = y_pred_tuned
test_analysis_df["fraud_prob"] = y_pred_prob

false_positives = test_analysis_df[(test_analysis_df["y_true"] == 0) & (test_analysis_df["y_pred"] == 1)]
true_negatives = test_analysis_df[((test_analysis_df["y_true"] == 0) & (test_analysis_df["y_pred"] == 0))]

n_legit = (test_analysis_df["y_true"] == 0).sum() #calc
print(f"\n False positives (temporal): {len(false_positives):,}"
      f"({len(false_positives)/n_legit*100:.2f}% of legit transactions)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figure 21 - False Positive Analysis: Temporal Set", fontsize=12, fontweight="bold")

cap = test_analysis_df["amount"].quantile(0.99)

tn_amounts = true_negatives[true_negatives["amount"] <= cap]["amount"]
fp_amounts = false_positives[false_positives["amount"] <= cap]["amount"]

#when it runs it says only 15 false positives so the original subset might be too small to show anything meaningful
#log output
print(f" True negatives for the plot: {len(tn_amounts):,}")
print(f" False positives for the plot: {len(fp_amounts):,}")

if len(tn_amounts) > 0:
    axes[0].hist(
        np.log1p(tn_amounts), bins=60, alpha=0.6, color="#4878CF", density=True, label="True Negatives"
    )

if len(fp_amounts) > 0:
    axes[0].hist(
        np.log1p(fp_amounts), bins=60, alpha=0.6, color="#D65F5F", density=True, label="False Positives"
    )

elif len(fp_amounts) == 0:
    axes[0].text(
        0.5, 0.5, f"Only {len(false_positives)} false positives in total \n"
        "not efficient for distribution plot", ha="center", va="center", transform=axes[0].transAxes, fontsize=11, color="#D65F5F"
    )

axes[0].set_title("Amount Distribution: False Positives vs True Negatives \n (temporal test set)")
axes[0].set_xlabel("log(Amount + 1)")
axes[0].set_ylabel("Density")
axes[0].legend()

#fp spread at high probabilities = confident errors = very bad
axes[1].hist(false_positives["fraud_prob"], bins=40, color="#D65F5F", edgecolor="white")
axes[1].axvline(x=locked_threshold, color="black", linestyle="--", label=f"Threshold ({locked_threshold:.2f})")
axes[1].set_xlabel("Predicted Fraud Probability")
axes[1].set_ylabel("Count")
axes[1].legend()

plt.tight_layout()
save_fig("fig21_false_positive_temporal.png")

"""FIGURES 22 & 23 - Concept Drift Analysis"""
#using only the rate of fraud is weak (fig18) bc it only shows if the fraud freq changes
#stronger to test if statistical properties change over time

df_filtered["time_bin"] = pd.cut(df_filtered["step"], bins=5)

#layer 2 - amount drift
amount_drift = (
    df_filtered.groupby(["time_bin", "isFraud"], observed=True)["amount"].mean().unstack().rename(columns={0: "Legitimate", 1:"Fraudulent"})
)

#layer 3 - feature level drift
drain_drift = (
    df_filtered[df_filtered["isFraud"] == 1].groupby("time_bin", observed=True)["orig_drained"].mean() *100
)

# if the ratio of transfer to cash_out frauc changes then the type_endoded becomes less reliable later
type_drift = ( 
    df_filtered[df_filtered["isFraud"] == 1].groupby("time_bin", observed=True)["type_encoded"].mean() * 100
)

print("\n CONCEPT DRIFT ANALYSIS")
print("Mean transaction amount by time bin: ")
print(amount_drift.to_string())
print(f"\n orig_drained rate across time bins (% of fraud): ")
print(drain_drift.to_string())
print(f"\n TRANSFER fraud rate across time bins (% of fraud that is TRANSFER): ")
print(type_drift.to_string())

bin_labels = [f"Bin {i+1}" for i in range(len(amount_drift))]

#fig 22 - amount drift
fig, axes = plt.subplots(1,2, figsize=(14,5))
fig.suptitle("Figure 22 - Concept Drift: Transaction Amount Across Time Bins", fontsize=13, fontweight="bold")

axes[0].plot(bin_labels, amount_drift["Legitimate"], marker="o", color="#4878CF", lw=2, label="Legitimate")
axes[0].plot(bin_labels, amount_drift["Fraudulent"], marker="o", color="#D65F5F", lw=2, label="Fraudulent")
axes[0].axvline(x="Bin 4", color="black", linestyle="--", lw=1.5, label="Train/test boundary (~Bin 4)")
axes[0].set_title("Mean Transaction Amount by Time Bin")
axes[0].set_xlabel("Time Bin (chronological)")
axes[0].set_ylabel("Mean Amount ($)")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
axes[0].legend()

axes[1].plot(bin_labels, drain_drift.values, marker="s", color="#D65F5F", lw=2, label="orig_drained rate (%)")
axes[1].plot(bin_labels, type_drift.values, marker="^", color="#4878CF", lw=2, label="TRANSFER fraud rate (%)")
axes[1].axvline(x="Bin 4", color="black", linestyle="--", lw=1.5, label="Train/test boundary (~Bin 4)")
axes[1].set_title("feature-Level Drift: Key fraud Signals Across Time")
axes[1].set_xlabel("Time Bin (chronological)")
axes[1].set_ylabel("Rate (%)")
axes[1].legend()

plt.tight_layout()
save_fig("fig22_concept_drift_amount.png")

#fig 23 - fraud type drift across the bins
#shows which part of the model decreases in quality first
fraud_type_counts = (
    df_filtered[df_filtered["isFraud"] == 1].groupby(["time_bin", "type"], observed=True).size().unstack(fill_value=0)
)

if "TRANSFER" in fraud_type_counts.columns and "CASH_OUT" in fraud_type_counts.columns:
    fraud_type_pct = fraud_type_counts.div(fraud_type_counts.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(10,5))
    plt.plot(bin_labels, fraud_type_pct["TRANSFER"].values, marker="o", color="#4878CF", lw=2, label="TRANSFER fraud (%)")
    plt.plot(bin_labels, fraud_type_pct["CASH_OUT"].values, marker="s", color="#D65F5F", lw=2, label="CASH_OUT fraud (%)")
    plt.axvline(x="Bin 4", color="black", linestyle="--", lw=1.5, label="Train/test boundary (~Bin 4)")
    plt.xlabel("Time Bin (chronological)")
    plt.ylabel("(%) of Fraud Transactions")
    plt.title("Figure 23 - Fraud Type COmposition Across Time\n"
              "(shifts indicate feature level concept drift)")
    plt.legend()
    plt.tight_layout()
    save_fig("fig23_fraud_type_drift.png")

#clean the temp bin column
df_filtered.drop(columns=["time_bin"], inplace=True)

# save the metrics
metrics = {
    "model": "XGBoost (Temporal Split)",
    "precision": round(prec, 4),
    "recall" : round(rec, 4),
    "f1" : round(f1, 4),
    "roc_auc" : round(roc_auc, 4),
    "pr_auc" : round(avg_prc, 4),
    "threshold" : round(locked_threshold, 4),
    "split_type" : "temporal",
    "split_step" : SPLIT_STEP
}

pd.DataFrame([metrics]).to_csv("metrics/xgboost_temporal.csv", index=False)
print(f"\n Metrics saved -> metrics/xgboost_temporal.csv")
print(f"\nAll Phase 4 figures saved to /figures")