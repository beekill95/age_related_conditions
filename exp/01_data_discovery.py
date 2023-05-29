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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %% [markdown]
# # Data Discovery

# %%
df = pd.read_csv('../data/train.csv')
df.info()

# %% [markdown]
# From the info,
# there 58 columns.
# Only a few of these columns contains missing values,
# in particular, the two columns EL and BQ have many missing values.
# There are other columns that also contain a few missing values:
# CB, CC, DU, FL, FR, FS, GL.

# %%
df.head()

# %%
df.describe().loc['min']

# %%
greek_df = pd.read_csv('../data/greeks.csv')
greek_df.info()

# %%
greek_df.head()

# %% [markdown]
# ## Missing Values
#
# Here,
# we will check which data sample has missing values.

# %%
missing_mask = ~df.isna().values
plt.pcolormesh(missing_mask)

# %% [markdown]
# Here, the vertical axis indicates data samples,
# and the horizontal axis indicates features.
#
# From the look of the plot,
# the two columns EL and BQ don't have missing values at the same samples.
# Thus, it is not reasonable to just remove these missing samples.

# %% [markdown]
# ## Class Distribution

# %%
sns.countplot(df, x='Class', hue='Class')

# %% [markdown]
# We have unbalanced dataset here,
# roughly 500 samples are 'Class 0'
# while the remaining samples are 'Class 1'.

# %% [markdown]
# ## Features Distribution
# ### Metric Features

# %%
def plot_feature_distribution(df: pd.DataFrame):
    feature_columns = df.columns.drop(['Id', 'Class']).values
    _, axes = plt.subplots(nrows=8, ncols=7, figsize=(20, 30), layout='constrained')

    for col, ax in zip(feature_columns, axes.flatten()):
        # If we approach the categorical column, ignore it.
        if col != 'EJ':
            sns.kdeplot(df, x=col, hue='Class', ax=ax, log_scale=True)
        else:
            ax.remove()


plot_feature_distribution(df)

# %% [markdown]
# We observe a couple of things:
#
# * The range of each metric variables are very different from each other.
# Thus, we need normalization and standardization the data.
# * These metric variables' distributions are not normal distribution.
# Some of these variables have bimodal distribution,
# such as BQ, CW, EL, and GL.
# * Many of these distributions are skewed.

# %% [markdown]
# ### Categorical Variable

# %%
sns.countplot(df, x='EJ', hue='Class')

# %% [markdown]
# ## Data Distribution with age-conditions

# %%
df_with_alpha = pd.merge(df, greek_df[['Id', 'Alpha']], on='Id')
sns.histplot(df_with_alpha, x='Alpha', hue='Alpha')


# %%
def plot_feature_distribution_with_alpha(df: pd.DataFrame):
    feature_columns = df.columns.drop(['Id', 'Class', 'Alpha']).values
    _, axes = plt.subplots(nrows=8, ncols=7, figsize=(20, 30), layout='constrained')

    for col, ax in zip(feature_columns, axes.flatten()):
        # If we approach the categorical column, ignore it.
        if col != 'EJ':
            sns.kdeplot(df, x=col, hue='Alpha', ax=ax, log_scale=True)
        else:
            ax.remove()


plot_feature_distribution_with_alpha(df_with_alpha)

# %%
sns.countplot(df_with_alpha, x="EJ", hue="Alpha")
