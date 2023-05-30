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
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (RandomForestClassifier,
                              HistGradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# %% [markdown]
# # Multiple Baseline Comparison
# ## Data

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
# medians = X.median()
# X = X.fillna(medians)


# %% [markdown]
# ## Models

# %%
models = [
    ('logistic', LogisticRegression(max_iter = 10000000,
                                    solver = 'liblinear')),
    ('logistic bal', LogisticRegression(max_iter = 10000000,
                                    class_weight = 'balanced',
                                    solver = 'liblinear')),
    ('svc', SVC(max_iter = 1000000, class_weight = 'balanced')),
    # Quite slow!
    #('linearsvc', LinearSVC(max_iter = 1000000, class_weight = 'balanced')),
    ('gaussian', GaussianProcessClassifier()),
    ('mlp', MLPClassifier(100, max_iter=1000000)),
    ('mlp 2', MLPClassifier([100, 100], max_iter=1000000)),
    ('knn 3', KNeighborsClassifier(3)),
    ('knn 6', KNeighborsClassifier(5)),
    ('knn 7', KNeighborsClassifier(7)),
    ('knn 10', KNeighborsClassifier(10)),
    ('randomforest', RandomForestClassifier(class_weight='balanced')),
    ('adaboost', AdaBoostClassifier()),
    ('histboost', HistGradientBoostingClassifier()),
    ('lgbm', LGBMClassifier()),
    ('lgbm balanced', LGBMClassifier(class_weight='balanced')),
    ('dart', LGBMClassifier(boosting_type='dart')),
    ('dart balanced', LGBMClassifier(boosting_type='dart', class_weight='balanced')),
    ('xgboost', XGBClassifier()),
    ('catboost', CatBoostClassifier(verbose=0)),
    ('cb balanced', CatBoostClassifier(verbose=0, auto_class_weights='Balanced')),
]

# %% [markdown]
# ## Cross Validation Routine

# %%
def cross_validate_model(model, *, X, y, cv: int):
    metrics = ['recall', 'precision', 'f1']
    results = cross_validate(model, X=X, y=y,
                             cv=cv,
                             scoring=metrics,
                             return_train_score=True)

    # Convert `results` from dict to pandas' DataFrame.
    folds = list(range(cv))
    train_df = pd.DataFrame({
        'Fold': folds,
        'Data': 'Train',
        **{m: results[f'train_{m}'] for m in metrics},
    })
    test_df = pd.DataFrame({
        'Fold': folds,
        'Data': 'Test',
        **{m: results[f'test_{m}'] for m in metrics},
    })
    df = pd.concat([train_df, test_df]).reset_index()
    return df


for model_name, model in models:
    r = cross_validate_model(
            Pipeline([('imputer', SimpleImputer(strategy='median')),
                      (model_name, model)]),
            X=X, y=y, cv=10)

    train_f1_mean = r['f1'][r['Data'] == 'Train'].mean()
    train_f1_std = r['f1'][r['Data'] == 'Train'].std()
    test_f1_mean = r['f1'][r['Data'] == 'Test'].mean()
    test_f1_std = r['f1'][r['Data'] == 'Test'].std()

    print(f'{model_name:>15} \
          | f1_test: {test_f1_mean:.4f} ± {test_f1_std:.4f} \
          | f1_train: {train_f1_mean:.4f} ± {train_f1_std:.4f}')

# %%
# This time, with standard scaler.
for model_name, model in models:
    r = cross_validate_model(
            Pipeline([('imputer', SimpleImputer(strategy='median')),
                      ('scaler', StandardScaler()),
                      (model_name, model)]),
            X=X, y=y, cv=10)

    train_f1_mean = r['f1'][r['Data'] == 'Train'].mean()
    train_f1_std = r['f1'][r['Data'] == 'Train'].std()
    test_f1_mean = r['f1'][r['Data'] == 'Test'].mean()
    test_f1_std = r['f1'][r['Data'] == 'Test'].std()

    print(f'{model_name:>15} \
          | f1_test: {test_f1_mean:.4f} ± {test_f1_std:.4f} \
          | f1_train: {train_f1_mean:.4f} ± {train_f1_std:.4f}')
