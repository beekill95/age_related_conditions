# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from __future__ import annotations


import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler

# %%
kaggle_submission = False

# %% [markdown]
# # TabPFN
#
# In this experiment,
# I'll try the most popular approach in the competition: TabPFN.
#
# ## Data

# %%
if kaggle_submission:
    train_path = '/kaggle/input/icr-identify-age-related-conditions/train.csv'
    test_path = '/kaggle/input/icr-identify-age-related-conditions/test.csv'
else:
    train_path = '../data/train.csv'
    test_path = '../data/test.csv'

train_df = pd.read_csv(train_path)
train_df.info()

# %%
test_df = pd.read_csv(test_path)
test_df.info()

# %%
X_df = train_df.drop(columns=['Id', 'Class', 'EJ'])
y = train_df['Class']
ej = train_df['EJ'].replace({'A': -1, 'B': 1})

# %% [markdown]
# ## Model

# %%
def balanced_log_loss(y_true, pred_prob):
    nb_class_0 = np.sum(1 - y_true)
    nb_class_1 = np.sum(y_true)

    prob_0 = np.clip(1. - pred_prob, 1e-10, 1. - 1e-10)
    prob_1 = np.clip(pred_prob, 1e-10, 1. - 1e-10)
    return (-np.sum((1 - y_true) * np.log(prob_0)) / nb_class_0
            - np.sum(y_true * np.log(prob_1)) / nb_class_1) / 2.


# %%
kfold = StratifiedKFold(n_splits=10)

X = X_df.values
train_log_losses = []
train_f1_scores = []
val_log_losses = []
val_f1_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    Xtr, Xte = X[train_idx], X[val_idx]
    ytr, yte = y[train_idx].values, y[val_idx].values

    # Training model:
    # TODO: weird! Probably I don't understand the paper correctly:
    # the way I understand it is that the prediction will be made along with all training data,
    # and no model training required!
    model = TabPFNClassifier(device='cuda', N_ensemble_configurations=32)
    model.fit(Xtr, ytr)

    # Testing.
    ytr_pred = model.predict(Xtr)
    ytr_prob_pred = model.predict_proba(Xtr)[:, 1]
    yte_pred = model.predict(Xte)
    yte_prob_pred = model.predict_proba(Xte)[:, 1]

    # Record model's performance.
    f1_train = f1_score(ytr, ytr_pred)
    precision_train = precision_score(ytr, ytr_pred)
    recall_train = recall_score(ytr, ytr_pred)
    log_loss_train = balanced_log_loss(ytr, ytr_prob_pred)
    print(f'Train - f1={f1_train:.4f} recall={recall_train:.4f} precision={precision_train:.4f} log-loss={log_loss_train:.4f}')

    f1_val = f1_score(yte, yte_pred)
    precision_val = precision_score(yte, yte_pred)
    recall_val = recall_score(yte, yte_pred)
    log_loss_val = balanced_log_loss(yte, yte_prob_pred)
    print(f'Valid - f1={f1_val:.4f} recall={recall_val:.4f} precision={precision_val:.4f} log-loss={log_loss_val:.4f}')

    # Store the results.
    train_f1_scores.append(f1_train)
    train_log_losses.append(log_loss_train)
    val_f1_scores.append(f1_val)
    val_log_losses.append(log_loss_val)

# %%
print(f'Train - F1={np.mean(train_f1_scores):.4f} +- {np.std(train_f1_scores):.4f}'
      f'; Log Loss = {np.mean(train_log_losses):.4f} +- {np.std(train_log_losses):.4f}')
print(f'Valid - F1={np.mean(val_f1_scores):.4f} +- {np.std(val_f1_scores):.4f}'
      f'; Log Loss = {np.mean(val_log_losses):.4f} +- {np.std(val_log_losses):.4f}')

