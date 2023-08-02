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
import pickle
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import MDS, TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from typing import Literal

# %% [markdown]
# # Visualize Fold 6 and Fold 7
#
# Perform visualization on the embeddings produced by pretrained models.
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
# ## Pretrained Models
# ### Neural Network Model

# %%
class NNClassifier(nn.Module):
    def __init__(self, input_shape: int) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.LayerNorm(1024),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.LayerNorm(512),

            nn.Linear(512, 64),
            # nn.Tanh(),
        )

        self.output = nn.Sequential(
            # nn.Dropout(),
            nn.LayerNorm(64),
            # nn.Dropout(),
            nn.Linear(64, 1),
        )

    def forward(self, x, *,
                mode: Literal['encoder', 'output', 'full'] = 'full',
                logit: bool = False):
        if mode == 'encoder':
            return self.encoder(x)
        elif mode == 'output':
            x = self.output(x)
            return x if logit else torch.sigmoid(x)
        elif mode == 'full':
            x = self.encoder(x)
            x = self.output(x)
            return x if logit else torch.sigmoid(x)

        raise ValueError(f'Unknown mode={mode}')


@dataclass
class TrainingResult:
    model: NNClassifier
    features: list[str]
    preprocessing: Pipeline
    performance: float


# Load pretrained models.
with open('./20230729_09_48_11.1e.model', 'rb') as inmodel:
    pretrained_models = pickle.load(inmodel)
    print(pretrained_models[0])


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
                    pretrained_models: list[TrainingResult],
                    manifold_kwargs: dict | None = None,
                    device: str = 'cpu'):
    def data_embedding(X: pd.DataFrame, g: pd.Series, model: TrainingResult):
        X_processed = pd.DataFrame(
            pretrained_model.preprocessing.transform(X),
            columns=X.columns
        )[pretrained_model.features]
        X_processed['ej'] = g.cat.codes.values

        # Obtain the embeddeded features.
        X_embed = (
            pretrained_model
            .model(
                torch.tensor(X_processed.values,
                             dtype=torch.float32).to(device),
                mode='encoder')
            .cpu().detach().numpy())
        return X_embed

    # Dimension reduction the original space.
    manifold_kwargs = (
        manifold_kwargs if manifold_kwargs is not None else dict())
    pipeline = Pipeline([
        ('pca', PCA(n_components=50)),
        ('tsne', TSNE(n_components=2, **manifold_kwargs)),
    ])
    X2d = pipeline.fit_transform(X)
    X2d_df = pd.DataFrame({
        'x1': X2d[:, 0],
        'x2': X2d[:, 1],
        'y': y,
    })

    # Cross validation.
    n_folds = len(pretrained_models)
    kfolds = StratifiedKFold(n_splits=n_folds)
    for model_idx, pretrained_model in enumerate(pretrained_models):
        # Preprocess data.
        Xembed = data_embedding(X, g, pretrained_model)

        # Perform data reduction to visualize the folds.
        Xembed_2d = (TSNE(n_components=2, **manifold_kwargs)
                     .fit_transform(Xembed))

        fig = plt.figure(figsize=(36, 16), layout='constrained', dpi=300)
        gs = fig.add_gridspec(
            nrows=4, ncols=int(3 + np.ceil(n_folds / 2)))
        fig.suptitle(f'Model #{model_idx + 1}')

        for fold, (_, test_idx) in enumerate(kfolds.split(Xembed_2d, y)):
            row, col = divmod(fold, n_folds // 2)
            highlight_axes = []

            data = pd.DataFrame({
                'x1': Xembed_2d[:, 0],
                'x2': Xembed_2d[:, 1],
                'y': y,
            })
            data['ds'] = 'train'
            data.loc[test_idx, 'ds'] = 'test'

            ax = fig.add_subplot(gs[row * 2, col + 3])
            highlight_axes.append(ax)
            sns.scatterplot(data[data['ds'] == 'train'],
                            x='x1', y='x2', hue='y',
                            edgecolors='face',
                            ax=ax)
            ax.set_title(f'Fold #{fold + 1}')
            if col == 0:
                ax.set_ylabel('Train')

            ax = fig.add_subplot(gs[row * 2 + 1, col + 3])
            highlight_axes.append(ax)
            if fold == model_idx:
                for i, d in data[data['ds'] == 'test'].iterrows():
                    ax.scatter(x=d['x1'], y=d['x2'],
                               marker=f'${i}$',
                               s=75,
                               fc='blue' if d['y'] == 0 else 'orange')
            else:
                sns.scatterplot(data[data['ds'] == 'test'],
                                x='x1', y='x2', hue='y',
                                ax=ax)
            ax.set_title('')
            if col == 0:
                ax.set_ylabel('Test')

            # Highlight the fold that the model was trained on.
            if fold == model_idx:
                for ax in highlight_axes:
                    ax.tick_params(color='red', labelcolor='red')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('red')

                # Plot original space.
                ax = fig.add_subplot(gs[:3, :3])
                for i, d in X2d_df.iterrows():
                    fc = (('blue' if d['y'] == 0 else 'orange') if i in test_idx
                          else (('green' if d['y'] == 0 else 'lime')))
                    ax.scatter(x=d['x1'], y=d['x2'],
                               marker=f'${i}$',
                               s=500,
                               fc=fc)
                ax.set_title('Original space')

        plt.show()
        plt.close(fig)


visualize_folds(
    X=Xtrain_df, g=ej, y=y,
    pretrained_models=pretrained_models,
    manifold_kwargs=dict())
