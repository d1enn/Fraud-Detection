import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve)

from ucimlrepo import fetch_ucirepo
import xgboost as xgb

import kagglehub 
warnings.filterwarnings('ignore')

""" Plan is:
    Baseline is Linear Regression
    Strong tabular model is XGBoost
    Advanced Isolation Forest
"""

# downlad the dataset
path = kagglehub.dataset_download("arunavakrchakraborty/financial-payment-services-fraud-data")
print("Path to dataset files:", path)

#list all the files in the downloaded dir
print("\nFiles in the directory:")
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        file_path = os.path.join(dirname, filename)
        print(f" - {filename} (full path: {file_path})")


#load the dataset
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
print(f"CSV files found {csv_files}")

if csv_files:
    if len(csv_files) == 1:
        file_path = os.path.join(path, csv_files[0])

# load the dataset
df = pd.read_csv(file_path)

# show some basic information
print(f"\n Datset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nColumn names:")
print(df.columns.tolist())

print(f"\n Basic Info:")
print(df.info())

# 1 - show descriptive statistics
print("\nDescriptive statitics for numerical columns:")
print(df.describe())

#2 - check for the data types
print("\n Data Types:")
print(df.dtypes)

# consistent visual style across all figs
sns.set_theme(style="whitegrid", palette="muted")
FRAUD_PALETTE = {0: "#4878CF", 1: "#D65F5F"} #blue = legit and red = fraud
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

def save_fig(name):
    """Save the figure to /figures"""
    plt.savefig(os.path.join(FIG_DIR, name), dpi=300, bbox_inches="tight")
    plt.close()
    print(f" ✓ saved -> figures/{name}")

# facts to cite
fraud = df[df["isFraud"] == 1]
legit = df[df["isFraud"] == 0]
n_total = len(df)
n_fraud = len(fraud)
n_legit = len(legit)
fraud_pct = n_fraud / n_total * 100
flagged = df["isFlaggedFraud"].sum()

print("=" * 55)
print("KEY FACTS")
print("=" * 55)
print(f" Total transanctions : {n_total:,}")
print(f" Fraudulent : {n_fraud:,} ({fraud_pct:.4f}%)")
print(f" Legitimate : {n_legit:,}")
print(f" Imbalance ratio : 1 fraud per {n_total//n_fraud:,} transactions")
print(f" isFlaggedFraud hits : {flagged} ({flagged/n_fraud*100:.2f}% of actual fraud)")
print("=" * 55)

"""FIG 1 - CLASS IMBALANCE BAR CHART => supports why accuracy is not enough"""
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Figure 1 - Class Imbalance in the Dataset", fontsize=14, fontweight="bold")

counts = df["isFraud"].value_counts().sort_index()
labels = ["Legitimate", "Fraudulent"]
colors = [FRAUD_PALETTE[0], FRAUD_PALETTE[1]]

# left -> raw counts
axes[0].bar(labels, counts.values,  color=colors, edgecolor="white", width=0.5)
axes[0].set_title("Transaction Counts (Raw)")
axes[0].set_ylabel("Number of Transactions")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 20000, f"{v:,}", ha="center", fontsize=10, fontweight="bold")

# right -> percentages
pcts = counts.values / n_total * 100
axes[1].bar(labels, pcts, color=colors, edgecolor="white", width=0.5)
axes[1].set_title("Transaction Counts (Percentage of Total)")
axes[1].set_ylabel("Percentage (%)")
axes[1].set_ylim(0, 105)
for i, v in enumerate(pcts):
    axes[1].text(i, v + 1, f"{v:.2f}%", ha="center", fontsize = 10, fontweight="bold")

plt.tight_layout()
save_fig("fig1_class_imbalance.png")

"""FIG 2 - TRANSACTION AMOUNT DISTRIBUTION (FRAUD VS LEGIT) -> log scale important because of skew/exponential distribution"""

# cap at the 99th percentile to stop extreme values -> METHODOLOGY
cap = df["amount"].quantile(0.99)
df_capped = df[df["amount"] <= cap].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figure 2 - Transaction Amount Distribution: Fraud vs Legit", fontsize=14, fontweight="bold")

#left -> overlapping Kernel Density Estimate on the log scale - shows the differences in shape
for label, subset in df_capped.groupby("isFraud"):
    amounts = subset["amount"]
    axes[0].hist(
        np.log1p(amounts), #to handle the transaction that are 0
        bins=80,
        alpha=0.55,
        color=FRAUD_PALETTE[label],
        label="Fraudulent" if label == 1 else "Legitimate",
        density=True
    )
axes[0].set_title("Distribution of log(Amount + 1)")
axes[0].set_xlabel("log(Amount + 1)")
axes[0].set_ylabel("Density")
axes[0].legend()

#right -> box plot shows median and spread per class more obviously
plot_data = df_capped[["amount", "isFraud"]].copy()
plot_data["Class"] = plot_data["isFraud"].map({0: "Legitimate", 1: "Fraudulent"})
sns.boxplot(
    data=plot_data,
    x="Class", y="amount",
    palette={"Legitimate" : FRAUD_PALETTE[0], "Fraudulent" : FRAUD_PALETTE[1]},
    ax=axes[1],
    width=0.4,
    showfliers=False # anything that is an outlier is not included for readability purposes
)
axes[1].set_title("Amount Distribution by Class \n (outliers are excluded, capped at 99th percentile)")
axes[1].set_ylabel("Transaction Amount")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

plt.tight_layout()
save_fig("fig2_amount_distribution.png")

""" FIG 3 - FRAUD BY TRANSACTION TYPE (in PaySim, fraud only occurs in TRANSFER and CASH_OUT -> needed for feature engineering)"""

type_fraud = df.groupby("type")["isFraud"].agg(["sum", "count"])
type_fraud["fraud_rate"] = type_fraud["sum"] / type_fraud["count"] * 100
type_fraud = type_fraud.sort_values("fraud_rate", ascending=False)

fig, axes = plt.subplots(1,2, figsize=(14, 5))
fig.suptitle("Figure 3 - Fraud Concentration By Transaction Type", fontsize=14, fontweight="bold")

#left: raw fraud count per type
axes[0].bar(type_fraud.index, type_fraud["sum"], color="#D65F5F", edgecolor="white")
axes[0].set_title("Number of Fraudulent Transactions by Type")
axes[0].set_xlabel("Transaction Type")
axes[0].set_ylabel("Fraud Count")
for i, (idx, row) in enumerate(type_fraud.iterrows()):
    axes[0].text(i, row["sum"] + 50, f"{int(row['sum']):,}", ha="center", fontsize=9)

#right: fraud rate per type 
axes[1].bar(type_fraud.index, type_fraud["fraud_rate"], color="#D65F5F", edgecolor="white")
axes[1].set_xlabel("Transaction Type")
axes[1].set_ylabel("Fraud Rate (%)")
for i, (idx, row) in enumerate(type_fraud.iterrows()):
    if row["fraud_rate"] > 0:
        axes[1].text(i, row["fraud_rate"] + 0.05, f"{row['fraud_rate']:.2f}%", ha="center", fontsize=9)

plt .tight_layout()
save_fig("fig3_fraud_by_type.png")

""" FIG 4 - Fraud rate over time"""

time_fraud = df.groupby("step")["isFraud"].mean() * 100

plt.figure(figsize=(10,5))
plt.plot(time_fraud.index, time_fraud.values, color="#D65F5F")

plt.title("Fraud Rate Over Time")
plt.xlabel("Time Step")
plt.ylabel("Fraud Rate (%)")

save_fig("fig4_fraud_over_time.png")

""" FIG 5 - CORRELATION HEATMAP"""

numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(12,8))

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    center=0,
    square=True,
    cbar_kws={"shrink": .8}
)

plt.title("Feature Correlation Heatmap")

save_fig("fig5_correlation_heatmap.png")


"""FIG 6 - BALANCE BEHAVIOUR FRAUD VS LEGIT"""

sample = df.sample(5000, random_state=42)

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=sample,
    x="oldbalanceOrg",
    y="newbalanceOrig",
    hue="isFraud",
    palette=FRAUD_PALETTE,
    alpha=0.4
)

plt.title("Figure 6 - Account Balance changes for Fraud vs Legitimate Transactions")
plt.xlabel("Old Balance (Origin Account)")
plt.ylabel("New Balance (Origin Account)")

save_fig("fig6_balance_behaviour.png")

print("\nAll Phase 1 figures saved to /figures")