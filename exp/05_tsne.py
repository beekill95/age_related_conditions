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
from itables import init_notebook_mode
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


init_notebook_mode(all_interactive=True)


# %% [markdown]
# # TSNE for Data Visualization
# ## Data Processing
#
# We'll apply data visualization for interaction and quadratic
# terms for important features as found in the previous notebooks.
# Furthermore,
# unlike the previous notebooks,
# the preprocessing steps are different:
#
# 1. Fill missing values with medians.
# 1. Create interaction and quadratic terms.
# 1. Apply standard scaler.

# %%
train_df = pd.read_csv('../data/train.csv')
X_df = train_df.drop(columns=['Id', 'Class', 'EJ'])
y_df = train_df['Class']

# %%
# Specify preprocessing pipeline.
preprocessing = Pipeline([
    # ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

# %%
def create_interaction_terms_between(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    assert all(f in df.columns for f in features)

    interactions = dict()

    for i, fst in enumerate(features):
        for snd in features[i+1:]:
            interactions[f'{fst}*{snd}'] = df[fst] * df[snd]

    return pd.DataFrame(interactions, index=df.index)


# Fill missing values with medians.
X_df = X_df.fillna(X_df.median())

# Select features based on previous notebooks.
most_important_features = ['BQ', 'CR', 'DI', 'DU']
less_important_features = ['CD ', 'CH', 'DN', 'DL', 'EE', 'EP', 'FI', 'GE', 'GF']
features = most_important_features + less_important_features

# Interaction features.
X_interactions_df = create_interaction_terms_between(X_df, features)

# Quadratic terms.
X2_df = X_df[features].pow(2.).rename(columns={
    f: f'{f}^2' for f in features
})

all_df = pd.concat([X_df, X_interactions_df, X2_df], axis=1)
all_df = pd.DataFrame(
    preprocessing.fit_transform(all_df),
    columns=all_df.columns,
    index=all_df.index)
all_df.describe().transpose()

# %% [markdown]
# ## Feature Visualization

# %%
# Specify dimension reduction pipeline.
dimension_reduction_2d = Pipeline([
    ('pca', PCA(n_components=50)),
    ('tsne', TSNE(perplexity=10.))
])

X_reduced = dimension_reduction_2d.fit_transform(all_df.values)
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y_df)

# %%
dimension_reduction_1d = Pipeline([
    ('pca', PCA(n_components=50)),
    ('tsne', TSNE(n_components=1, perplexity=10.))
])

X_reduced = dimension_reduction_1d.fit_transform(all_df.values)
sns.swarmplot(x=X_reduced[:, 0], y=y_df, hue=y_df, orient='h')

# %% [markdown]
# From these plots,
# we can understand why the logistics regression might do a good job here.
# The linear kernel is able to distinguish between class 0 and class 1:
# both classes tend to cluster at the extreme ends of the first feature.
# However, this is not perfect,
# the majority class looks like a uniform distribution.
#
# => We need to do a better job!
