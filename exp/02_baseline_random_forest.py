# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OneHotEncoder

# %% [markdown]
# # Random Forest Baseline

# %%
df = pd.read_csv('../data/train.csv')
df.info()

# %%
# Split data into features and target.
X = df.drop(columns=['Id', 'Class'])
y = df['Class']

# %%
# Preprocess categorical column 'EJ' into 0-1 encoding.
ej_onehot = pd.get_dummies(X['EJ'])
X = X.drop(columns='EJ').join(ej_onehot)

# %%
# Replace missing values with median.
medians = X.median()
X = X.fillna(medians)

# %%
rf = RandomForestClassifier(class_weight='balanced')
cross_validate(rf, X=X, y=y, cv=6, scoring=['recall', 'precision', 'f1'], return_train_score=True)
