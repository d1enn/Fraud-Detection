AI-Powered Financial Fraud Detection
Comparing supervised and unsupervised machine learning on 6.3 million financial transactions

BSc Computing dissertation project | Python · Scikit-learn · XGBoost · Data Analysis
Overview
Financial fraud is difficult to detect because fraudulent transactions are rare, transaction behaviour changes over time, and false alarms can disrupt legitimate customers. This project develops and evaluates a machine learning-based fraud detection research prototype, comparing supervised classification with unsupervised anomaly detection.

Using the PaySim synthetic financial transactions dataset (6,362,620 transactions; 0.13% labelled as fraud), I investigated whether machine learning could improve upon the dataset's existing rule-based fraud flag while maintaining precision under severe class imbalance.

The project compares three models under a consistent preprocessing and evaluation framework:
- Logistic Regression — interpretable supervised baseline.
- XGBoost — gradient-boosted classifier for complex, non-linear transaction patterns.
- Isolation Forest — unsupervised anomaly detection trained on legitimate transactions.
The work also investigates classification threshold optimisation, temporal generalisation, feature drift, and whether anomaly detection catches fraud missed by XGBoost.
Key results
Model / evaluation                              Precision	Recall	F1-score	PR-AUC
Logistic Regression (random split)	             0.0859	  0.9580	0.1577	  0.5185
XGBoost (random split, tuned threshold)	         0.9260	  0.9063	0.9160	  0.9705
XGBoost (temporal holdout, locked threshold)	   0.9953	  0.7527	0.8572	  0.9603
Isolation Forest (random split)	                 0.1188	  0.1376	0.1275	  0.0593


Main findings:
- XGBoost achieved the best overall performance among the three models. Optimising its classification threshold to 0.9677 improved the random-split F1-score to 0.9160.
- Temporal validation remained strong but exposed a recall trade-off: PR-AUC was 0.9603, while recall decreased to 0.7527 on later transactions, compared with 0.9063 on the random split. The temporal test produced 15 false positives across 548,246 legitimate transactions.
- Isolation Forest did not identify additional fraud cases beyond those already detected by XGBoost in the overlap analysis. This result is specific to the dataset and experimental configuration; it does not establish that anomaly detection has no value in other environments.
- The dataset's existing isFlaggedFraud indicator identified just 16 of 8,213 fraudulent transactions (approximately 0.19% recall).
Note: The random-split XGBoost threshold was selected using the test data, which may make those results optimistic. The subsequent temporal evaluation applied that threshold without retuning, providing a more useful check of performance on later transactions. PR-AUC is reported as average precision (AP), as calculated in the scripts.

Methodology
1. Explore the data: analyse transaction types, class imbalance, amounts, account balances and fraud patterns over time.
2. Preprocess and engineer features: retain TRANSFER and CASH_OUT transactions (the types containing fraud in PaySim); engineer origin and destination balance changes and an orig_drained indicator; exclude account identifiers and isFlaggedFraud from model inputs.
3. Build a baseline: train Logistic Regression with imbalance-aware class weighting and a training-only fitted StandardScaler.
4. Train XGBoost: compare it against the baseline and examine precision–recall behaviour, threshold selection and feature importance.
5. Validate over time: train on earlier transactions and evaluate on later transactions, reusing the previously selected 0.9677 threshold without retuning it on the temporal test set.
6. Evaluate anomalies and model overlap: train Isolation Forest on legitimate transactions and compare its detections with those of XGBoost.
Evaluation prioritises precision, recall, F1-score, PR-AUC, ROC-AUC and confusion matrices, rather than overall accuracy, due to the extreme class imbalance.

Technology stack
- Language: Python 3.12
- Data processing: pandas, NumPy
- Machine learning: scikit-learn, XGBoost
- Visualisation: Matplotlib, Seaborn
- Data acquisition: KaggleHub
- Version control: Git and GitHub

Dataset
PaySim / Financial Payment Services Fraud Data — a publicly available synthetic mobile-money transaction dataset hosted on Kaggle.
- Dataset: Financial Payment Services Fraud Data (Kaggle)
- Total transactions: 6,362,620
- Fraudulent transactions: 8,213 (approximately 0.13%)
- Transactions used for modelling: 2,770,409 TRANSFER and CASH_OUT transactions
The dataset is synthetic and is not evidence of live banking performance. Dataset credit and any applicable reuse conditions remain with the original creators and distributor.
Limitations and future improvements
- Synthetic data: benchmark results may not generalise to real banking traffic or new fraud strategies.
- Threshold selection: the random-split threshold was optimised against the test set, introducing potential optimism; future work should use an independent validation set for threshold selection.
- Temporal drift: later data reduced recall even though ranking performance remained high, motivating ongoing monitoring and retraining.
- Explainability and deployment: a production system would need per-decision explanations, real-time serving, monitoring, security controls and testing on suitably governed real data.
Potential extensions include SHAP-based explanations, a more rigorous out-of-time tuning protocol, and drift-aware retraining.
About this project
Developed as my BSc Computing dissertation: An AI-Powered Anomaly-Based Intrusion Detection System for Financial Transactions.

This repository demonstrates applied Python development, data preprocessing and feature engineering, supervised and unsupervised machine learning, experimentation with imbalanced data, and critical evaluation of model performance under temporal distribution shift.
