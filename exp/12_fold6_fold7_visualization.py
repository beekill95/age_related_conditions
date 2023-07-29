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
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.manifold import MDS, TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# # Visualize Fold 6 and Fold 7
#
# First, just apply preprocessing just like we did
# in all previous experiments.
#
# ## Data

# %%
train_path = '../data/train.csv'
test_path = '../data/test.csv'

train_df = pd.read_csv(train_path)
train_df.info()

# %%
test_df = pd.read_csv(test_path)
test_df.info()

# %% [markdown]
# ### Data Preprocessing

# %%
imputer = SimpleImputer(strategy='median')
preprocessing = Pipeline([
    # ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

# %%
X_df = train_df.drop(columns=['Id', 'Class', 'EJ'])
y = train_df['Class']
ej = train_df['EJ'].astype('category')

X_df = pd.DataFrame(
    imputer.fit_transform(X_df),
    columns=X_df.columns,
    index=X_df.index)
y = y.values


# %% [markdown]
# ### Interaction Terms

# %%
def create_interaction_terms_between(df: pd.DataFrame, features: list[str]):
    assert all(f in df.columns for f in features)

    interactions = dict()

    for i, fst in enumerate(features):
        for snd in features[i+1:]:
            interactions[f'{fst}*{snd}'] = df[fst] * df[snd]

    return pd.DataFrame(interactions)


# %% [markdown]
# ### Quadratic Terms

# %%
def create_quadratic_terms(df: pd.DataFrame, features: list[str]):
    assert all(f in df.columns for f in features)

    return X_df[features].pow(2.).rename(columns={
        f: f'{f}^2' for f in features
    })


# %%
Xinteractions_df = create_interaction_terms_between(X_df, X_df.columns)
X2_df = create_quadratic_terms(X_df, X_df.columns)

Xtrain_df = pd.concat([X_df, Xinteractions_df, X2_df], axis=1)


# %% [markdown]
# ### Correlated Features Removal

# %%
def filter_in_uncorrelated_features(X_df: pd.DataFrame,
                                    correlation_threshold: float = 0.7):
    # Calculate Spearman's correlation, and then convert to
    # distances matrix.
    corr = X_df.corr('spearman')
    distances = 1. - corr.abs()

    # Perform clustering using Agglomerative Clustering.
    dist_linkage = hierarchy.ward(squareform(distances))
    clusters = hierarchy.fcluster(dist_linkage,
                                  1. - correlation_threshold,
                                  criterion='distance')

    # Choose a feature from each cluster.
    features = []
    for cluster in np.unique(clusters):
        features_in_cluster = X_df.columns[cluster == clusters]

        # TODO: Here, we use the first feature,
        # but it can be other choices.
        chosen_feature = features_in_cluster[0]
        features.append(chosen_feature)

    # Return a new dataframe with the chosen features.
    return X_df.loc[:, features]


# %% [markdown]
# ## Visualize Folds

# %%
@dataclass
class KFoldData:
    X: pd.DataFrame
    g: pd.Series
    y: np.ndarray


def split_folds(X: pd.DataFrame, g: pd.Series, y: np.ndarray, n_folds: int):
    kfolds = StratifiedKFold(n_splits=n_folds)
    fold_data = []
    for train_idx, test_idx in kfolds.split(X, y):
        train = KFoldData(
            X=X.iloc[train_idx].copy(), g=g.iloc[train_idx], y=y[train_idx])
        test = KFoldData(
            X=X.iloc[test_idx].copy(), g=g.iloc[test_idx], y=y[test_idx])
        fold_data.append((train, test))

    return fold_data


def visualize_folds(*,
                    X: pd.DataFrame, g: pd.Series, y: np.ndarray,
                    n_folds: int, correlation_threshold: float,
                    manifold_kwargs: dict | None = None):
    folds = split_folds(X=X, g=g, y=y, n_folds=n_folds)

    for fold_idx, (train, test) in enumerate(folds):
        # First, filter out correlated features.
        Xtr = filter_in_uncorrelated_features(
            train.X,
            correlation_threshold=correlation_threshold)
        print('Chosen features: \n', Xtr.columns.tolist())
        Xte = test.X.loc[:, Xtr.columns]

        # Append column ej back.
        Xtr.loc[:, 'ej'] = train.g.cat.codes.values
        Xte.loc[:, 'ej'] = test.g.cat.codes.values

        # Using tSNE to visualize the datasets.
        # tsne = (TSNE(n_components=2)
        #         if manifold_kwargs is None
        #         else TSNE(n_components=2, **manifold_kwargs))
        model = (MDS(n_components=2)
                 if manifold_kwargs is None
                 else MDS(n_components=2, **manifold_kwargs))
        Xtemp = pd.concat([Xtr, Xte], axis=0, ignore_index=True)
        print('Xtemp', len(Xtemp))
        Xtemp_2d = pd.DataFrame(
            model.fit_transform(Xtemp),
            columns=['x1', 'x2'])
        Xtemp_2d['y'] = np.concatenate([train.y, test.y])
        Xtemp_2d['data'] = 'train'
        Xtemp_2d.loc[len(Xtr):, 'data'] = 'test'

        # Plot the results.
        fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
        sns.scatterplot(
            data=Xtemp_2d[Xtemp_2d['data'] == 'train'],
            x='x1', y='x2',
            hue='y',
            ax=axes[0])
        axes[0].set_title(f'Train of Fold #{fold_idx + 1}')
        sns.scatterplot(
            data=Xtemp_2d[Xtemp_2d['data'] == 'test'],
            x='x1', y='x2',
            hue='y',
            ax=axes[1])
        axes[1].set_title(f'Test of Fold #{fold_idx + 1}')

        # Show the plot.
        plt.show()
        plt.close(fig)


visualize_folds(
    X=Xtrain_df, g=ej, y=y,
    n_folds=10, correlation_threshold=0.3,
    manifold_kwargs=dict())
