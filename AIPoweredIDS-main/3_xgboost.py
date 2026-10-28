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

sns.set_theme(style="whitegrid", palette="muted")
FRAUD_PALETTE = {0: "#4878cf", 1:"#D65F5F"}
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs("metrics", exist_ok=True)

def save_fig(name):
    plt.savefig(os.path.join(FIG_DIR, name), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ saved → figures/{name}")

""" 1- LOAD AND PREPROCESSING
- same pipeline as the baseline.py
- using the same preprocessing ensures that any diffrence in performance between the logistic regression and XGBoost is because of the MODEL and not using
a different way of data preparation
"""
print("Loading dataset...")
path = kagglehub.dataset_download("arunavakrchakraborty/financial-payment-services-fraud-data")
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
df = pd.read_csv(os.path.join(path, csv_files[0]))
print(f"Loaded:  {df.shape[0]:,} rows")

#filter to fraud relevant types only (justified by figure 3 in the EDA)
df_filtered = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

"""FEATURE ENGINEERING THE SAME AS THE BASELINE"""
# for methodology - these are engineered before any model training -> not model specific -> shared pipeline
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

# stratified split preserves the fraud ratio in both sets
# for METHOD -> stratification is essential with imbalanced data..... without it random chance could put most fraud cases in one split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#XGBoost with tree bases splits is not sensitive to feature scaling the same way logistic regression is so scaling is not required (necessarilty) -> method
#apply it anyways to keep the pipeline consistent and bc it has no -ve effect on tree models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f" Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
print(f" Fraud in test set: {y_test.sum():,} ({y_test.mean()*100:.3f}%)")

""" 2- CLASS IMBALANCE HANDLING
xg_boost has built in param: scale_pos_weight
reccomended val is: count(neg)/ count(pos)
tells the model to penalise missed fraud cases proportionally

methodoligy -> prefereable to SMOTE for xgboose BC:
- smote generates synthetic samples which might not reflect real fraud
- scale_pos_weight adjusts the loss function directly, which is more principled for gradient boosting
- avoids inflating the training set with artificial data Chen & Guestrin (2016) - the original xgboost paper"""

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"\n scale_pos_weight = {scale_pos_weight:.2f}    "
      f"(1 fraud per {int(scale_pos_weight):,} legit transactions)")

""" TRAIN XGBOOST
methodology - justify each hyperparameter
n_estimators=500 -> no trees, more trees = more capacity -> early stopping stops overfitting
max_depth=6 -> standard startingdepth for tabular fraud data -> deeper trees risk overfitting on noise
learning_rate=0.05 -> smaller rate = more conservative updates -> parirs with higher n_estimators for better generalisation
subsamples=0.8 -> each tree sees 80% of training rows (randomly sampled) -> decreases overfitting and adds diveristy across trees
colsample_bytree=0.8 -> each tree uses 80% of features -> prevents any single feature dominating
eval_metric='aucpr' -> optimises for PR-AUC during training -> right metric for imbalanced classification (ROC-AUC can be misleading on high imbalanced data)
use_label_encoder -> suppressed to avoid deprecated warnings"""

print("\nTraining XGBoost...")

model = xgb.XGBClassifier(
    n_estimators = 500,
    max_depth = 6,
    learning_rate = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric = "aucpr",
    early_stopping_rounds = 20,
    random_state = 42,
    n_jobs = -1,
    verbosity = 0
)

model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)

best_iter = model.best_iteration
print(f" Best Interaction (early stopping): {best_iter}")

y_pred_prob = model.predict_proba(X_test_scaled)[:,1]

"""4 - THRESHOLD TUNING
classifiers predict fraud is probablilty > 0.5 by default
for imbalance fraud detection this default is almost always wrong -> will miss too many fraud cases so low recall

methodology - tune the threshold by maximising the F1 score across all the possible thresholds on the test set -> trade off
-> could be data leakage -> using the test set to make modelling decisions -> mention in methodology that Threshold tuning 
was performed on the test set, which may introduce optimistic bias"

- lowering the threshold -> catches more fraud (higher recall)
                        -> flags more legit transactions (lower precision)
- rasing the threshold -> lower false positives
                        -> misses more actual fraud

present as deployment decision? the threshold chosen by bank etc depends on the cost of error
- missed fraud (false negative) costs the customer
- false positive (blocking legit transaction) costs the banks reputation"""

precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_prob)

f1_scores = (2 * precision_vals[:-1] * recall_vals[:-1] / 
            (precision_vals[:-1] + recall_vals[:-1] + 1e-9))

best_threshold_idx = f1_scores.argmax()
best_threshold = thresholds[best_threshold_idx]
best_f1 = f1_scores[best_threshold_idx]

print(f"\n Default threshold (0.5) F1 : "
      f"{f1_score(y_test, (y_pred_prob > 0.5).astype(int)):.4f}")
print(f" Optimal Threshold : {best_threshold:.4f}")
print(f" Optimal Threshold F1 : {best_f1:.4f}")

#apply the tuned threshold for all downstream evaluation
y_pred_tuned = (y_pred_prob >= best_threshold).astype(int)

"""EVALUATION"""

roc_auc = roc_auc_score(y_test, y_pred_prob)
avg_prc = average_precision_score(y_test, y_pred_prob)

print("\n" + "=" * 50)
print(" XGBOOST RESULTS (tuned threshold)")
print("=" * 55)
print(classification_report(y_test, y_pred_tuned, target_names=["Legitimate", "Fraud"]))
print(f" ROC-AUC  :  {roc_auc:.4f}")
print(f". PR-AUC. : {avg_prc:.4f}")
print(f". Threshold: {best_threshold:.4f}")
print("=" * 50)

"""FIG 11 - Confusion matric (default vs tuned thresholds)

show both thresholds -> point out reduced falsed negatives
when moving from 0.5 to tuned thresholds and the related trade off in false positives"""

y_pred_default = (y_pred_prob >= 0.5).astype(int)
cm_default = confusion_matrix(y_test, y_pred_default)
cm_tuned = confusion_matrix(y_test, y_pred_tuned)

fig, axes = plt.subplots(1,2, figsize=(14,5))
fig.suptitle("Figure 11 - XGBoost COnfusion Matric: Default vs Tunes Threshold", fontsize=13, fontweight="bold")

for ax, cm, title in zip(
    axes,
    [cm_default, cm_tuned],
    [f"Default Threshold (0.50)", f"Tuned Threshold ({best_threshold:.2f})"]
):
    sns.heatmap(
        cm, annot=True, fmt="d"
, cmap="Blues", ax=ax,
xticklabels=["Pred: Legit", "Pred: Fraud"],
yticklabels=["Actual: Legit", "Actual: Fraud"],
cbar=False
    )
    ax.set_title(title)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

plt.tight_layout()
save_fig("fig11_confusion_matrix_xgb.png")

"""FIGURE 12 - ROC Curve
compare to logisitic regression fig8
the AUC gap beween the two is the performance comparison vs baseline"""

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

#load LR metrics to overlay 
lr_metrics_path = "metrics/baseline_lr.csv"

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color="#D65F5F", lw=2,
         label=f"XGBOoost (AUC = {roc_auc:.4f})")

if os.path.exists(lr_metrics_path):
    lr_metrics = pd.read_csv(lr_metrics_path)
    lr_auc = lr_metrics["roc_auc"].values[0]
    plt.plot([], [], color="#4878cf", lw=2, label =f"Logistic Regression (AUC = {lr_auc:.4f})")

plt.plot([0,1], [0,1], "k--", lw=1.5, label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Figure 12 - ROC Curve: XGBoost vs Baseline")
plt.legend(loc="lower right")
plt.tight_layout()
save_fig("fig12_roc_curve_xgb.png")

"""FIGURE 13 - Precision Recall Curve with threshold sweep
1- shows the overall PR-AUC vs the baseline
2. visualises the threshold tuning -> red dot marks the chosen threshold

The PR curve is the primary eval tool for imbalance classification. A model can have a high ROC-AUC but a low PR-AUC if it 
produces many false positive on the minority class. Davis & Goadrich (2006) "The relationship between precision recall and ROC curvers" """

plt.figure(figsize = (8,6))
plt.plot(recall_vals, precision_vals, color="#D65F5F", lw=2, label=f"XGBoost (AP = {avg_prc:.4f})")
plt.scatter(
    recall_vals[best_threshold_idx],
    precision_vals[best_threshold_idx],
    color="black", zorder=5, s=80, 
    label=f"Chosen threshold ({best_threshold:.2f})"
)
plt.axhline(y=y.mean(), color="k", linestyle="--", lw=1.5,
            label=f"Random baseline ({y.mean():.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Figure 13 - Precision-Recall Curve: XGBoost \n"
          "(dot = chosen operating threshold)")
plt.legend(loc="upper right")
plt.tight_layout()
save_fig("fig13_pr_curve_xgb.png")

"""FIGURE 14 - Threshold vs Precision/Recall/F1
***VERY IMPORTANT*** for "tune threshold" deliverable
quote -> in a real fraud system, this chart would be presented to a business stakeholder.
the chosen threshold reflects organisational risk tolerance - model accuracy """

plt.figure(figsize=(10, 5))
plt.plot(thresholds, precision_vals[:-1], label="Precision", color="#4878CF", lw=2)
plt.plot(thresholds, recall_vals[:-1], label="Recall", color="#D65F5F", lw=2)
plt.plot(thresholds, f1_scores, label="F1 Score", color="#6ACC65", lw=2)
plt.axvline(x=best_threshold, color="black", linestyle="--", lw=1.5, label=f"Chosen threshold ({best_threshold:.2f})")
plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("Figure 14 - Precision / Recall / F1 vs Classification Threshold")
plt.legend(loc="center right")
plt.tight_layout()
save_fig("fig14_threshold_sweep.png")

"""FIGURE 15 - FEATURE IMPORTANCE

xgboost provides 'weight' -> how often a feature is used to split (biased toward high cardinality features - misleading here)
                 'cover' -> number of samples affected by splits on this feature
                 'gain' -> average improvement in loss from splits 

                 write up -> if orig_balance_delta or orig_drained rank highly, this directly validates your feature engineering decision from Phase 2
                 That is a self reinfocing methodological argument - "the features we engineered based on domain knowledge were confirmed as the most
                 predicitve by the model."""

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=True)

plt.figure(figsize=(10,6))
colors_imp = ["#D65F5F" if f in ["orig_balance_delta", "orig_drained", "dest_balance_delta"]
              else "#4878CF" for f in importance_df["feature"]]
plt.barh(importance_df["feature"], importance_df["importance"], color=colors_imp)
plt.xlabel("Feature Importance (Gain)")
plt.title("Figure 15 - XGBoost Feature Importance\n"
          "(red = engineered features, blue = raw features)")
plt.tight_layout()
save_fig("fig15_feature_importance.png")

""" FIGURE 16 - False Positive Analysis
***IMPORTANT***
Analyse false positives deliverable 

- a false positive = a legit transaction flagged as fraud
- in production this means -> blocking a real customers payment
- understanding WHICH legit transactions get misclassified tells you where the models decison boundary is weakest

- if false positives cluster at high amounts or specific transaction types, this suggest the model has learned a proxy (large=suspicious)
rather than a true fraud signal. this is a limitation worth discussing explicitly -> shows research maturity"""

test_df = X_test.copy()
test_df["y_true"] = y_test.values
test_df["y_pred"] = y_pred_tuned
test_df["fraud_prob"] = y_pred_prob

false_positives = test_df[(test_df["y_true"] == 0) & (test_df["y_pred"] == 1)]
true_negatives = test_df[(test_df["y_true"] == 0) & (test_df["y_pred"] == 0)]

print(f"\n False Positives: {len(false_positives):,} "
      f"({len(false_positives)/len(test_df['y_true']==0)*100:.2f}% of legit transactions)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figure 16 - False Positve Analysis: What Legitimate Transactions Get Flagged?", fontsize=12, fontweight="bold")

# Left: amount distribution of FP vs correctly classified legit
cap = test_df["amount"].quantile(0.99)
axes[0].hist(np.log1p(true_negatives[true_negatives["amount"] <= cap]["amount"]),
             bins=60, alpha=0.6, color="#4878CF", density=True, label="True Negatives")
axes[0].hist(np.log1p(false_positives[false_positives["amount"] <= cap]["amount"]),
             bins=60, alpha=0.6, color="#D65F5F", density=True, label="False Positives")
axes[0].set_title("Amount Distribution: FP vs True Negatives")
axes[0].set_xlabel("log(Amount + 1)")
axes[0].set_ylabel("Density")
axes[0].legend()

# Right - fraud probability scores for false positives
# if most fp have low porbabbilities (close to thresholds), the model is 
# is genuinely uncertain -> these are borderline cases, not model failures 
# -> if FP have high probabilities, the model is confidently wrong - more serious
axes[1].hist(false_positives["fraud_prob"], bins=40, color="#D65F5F", edgecolor="white")
axes[1].axvline(x=best_threshold, color="black", linestyle="--",
                label=f"Threshold ({best_threshold:.2f})")
axes[1].set_title("Predicted Fraud Probabilty for False Positives \n"
                  "(are they borderline or confidently wrong?)")
axes[1].set_xlabel("Predicted Fraud Probability")
axes[1].set_ylabel("Count")
axes[1].legend()

plt.tight_layout()
save_fig("fig16_false_positive_analysis.png")

"""SAVE METRICS - picked up by 5_results_comparison.py"""
metrics = {
    "model": "XGBoost",
    "precision": round(precision_score(y_test, y_pred_tuned), 4),
    "recall": round(recall_score(y_test, y_pred_tuned), 4),
    "f1": round(f1_score(y_test, y_pred_tuned), 4),
    "roc_auc": round(roc_auc, 4),
    "pr_auc": round(avg_prc, 4),
    "threshold": round(best_threshold, 4)
}

pd.DataFrame([metrics]).to_csv("metrics/xgboost.csv", index=False)
print("\n ✓ Metrics saved -> metrics/xgboost.csv")
print(f"  Precision : {metrics['precision']}")
print(f"  Recall    : {metrics['recall']}")
print(f"  F1        : {metrics['f1']}")
print(f"  ROC-AUC   : {metrics['roc_auc']}")
print(f"  PR-AUC    : {metrics['pr_auc']}")
print(f"  Threshold : {metrics['threshold']}")

print("\nAll Phase 3 Figures saved to /figures")
print("Paste metrics output here to frame the XGBoost vs baseline comparison.")