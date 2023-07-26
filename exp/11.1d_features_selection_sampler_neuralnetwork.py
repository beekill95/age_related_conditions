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
from imblearn.over_sampling import SMOTENC
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import bernoulli
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.manifold import MDS
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from typing import Literal


numpyro.set_platform('cpu')
numpyro.set_host_device_count(16)

# %%
kaggle_submission = False
run_cv = True

# %% [markdown]
# # Logistic Regression with Tree-based Features Selection and Sampler
#
# In this experiment,
# I'll try perform uncorrelated features removal
# and Bayesian logistic regression.
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
# Xtrain_df['EJ'] = ej.cat.codes

# %% [markdown]
# ### Correlations Removal

# %%
corr = Xtrain_df.corr('spearman')

fig, ax = plt.subplots()
cs = ax.pcolormesh(corr.abs())
fig.colorbar(cs, ax=ax)
fig.tight_layout()

# %%
# Convert the correlation matrix into dissimilarity matrix,
# to be used with MDS.
distance = 1. - np.abs(corr)
mds = MDS(n_components=2, dissimilarity='precomputed')
embeddings = mds.fit_transform(distance)

# Show the results.
sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1])

# %%
# Now, we can use clustering on the MDS's result
# to identify the clusters.
# clustering = DBSCAN(eps=0.1, min_samples=2, metric='precomputed')
# clusters = clustering.fit_predict(distance)
dist_linkage = hierarchy.ward(squareform(distance))
clusters = hierarchy.fcluster(dist_linkage, 0.3, criterion='distance')
unique_clusters = list(np.unique(clusters))
print(f'Clusters: {unique_clusters}')

# Plot the results.
fig, ax = plt.subplots(figsize=(16, 12))
sns.scatterplot(x=embeddings[:, 0],
                y=embeddings[:, 1],
                hue=clusters,
                style=clusters,
                size=clusters*100,
                palette='hsv',
                legend=False,
                ax=ax)
fig.tight_layout()

# %%
# Show the correlation in these clusters.
for cluster in unique_clusters:
    features_in_cluster = Xtrain_df.columns[clusters == cluster]
    X_in_cluster = Xtrain_df[features_in_cluster]
    corr_in_cluster = X_in_cluster.corr('spearman')
    corrs = 1 - squareform(1 - np.abs(corr_in_cluster))
    if len(features_in_cluster) > 1:
        print(f'{cluster=}, nb_features={len(features_in_cluster)}, '
              f'min={np.min(corrs)}, '
              f'max={np.max(corrs)}, mean={np.mean(corrs)}')
    else:
        print(f'{cluster=} has only 1 member.')


# %% [markdown]
# Now, we can use these steps to extract the uncorrelated features.

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
    return X_df[features]


# %% [markdown]
# ## Model
# ### Neural Network Classifier

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


def create_training_and_evaluation_step(
        model: nn.Module,
        lr=1e-3,
        weight_decay=1e-5,
        regularization_weight=1e-2):

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    # Init weights before training.
    def init_weights(m: nn.Module):
        strategy_fn = nn.init.xavier_normal_
        if type(m) in [nn.Linear, nn.LazyLinear]:
            strategy_fn(m.weight)

    model.apply(lambda m: init_weights(m))

    def encoder_regularization_loss(x: torch.Tensor):
        """
        Calculate regularization loss of the encoder's output.

        x should have shape (nb_batches, nb_features)
        """
        # First, the output should be normally distributed.
        l1 = torch.mean(torch.sum(x**2, axis=1))

        # Second, features should not be correlated.
        cov = torch.t(x) @ x
        cov = cov - torch.diag(torch.diag(cov))
        l2 = torch.mean(torch.abs(cov))

        return l1 + l2

    def train_step(dataloader: DataLoader, *,
                   device: str, epoch: int, progress: bool = True,
                   train_noise: float = 0.0):
        model.train()

        train_loss = 0
        regularization_loss = 0

        num_batches = len(dataloader)
        bar = (tqdm(
            enumerate(dataloader), total=num_batches, desc=f'Epoch {epoch}')
            if progress
            else enumerate(dataloader))
        for i, (X, y) in bar:
            X, y = X.to(device), y.to(device)

            # Add noise to the training.
            X = X + train_noise * torch.rand_like(X)

            # Make prediction and calculate loss.
            encoder_output = model(X, mode='encoder')
            pred = model(encoder_output, mode='output', logit=True)

            # Losses.
            encoder_loss = encoder_regularization_loss(encoder_output)
            classification_loss = loss_fn(pred, y)

            loss = regularization_weight * encoder_loss + classification_loss
            regularization_loss += encoder_loss.item()
            train_loss += loss.item()

            # Back-propagation step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Show the MSE.
            if progress:
                bar.set_postfix_str(
                    f'Loss: {(train_loss / (i + 1)):.4f}'
                    f'- Encoder loss: {regularization_loss / (i + 1):.4f}')

        return train_loss / num_batches, regularization_loss / num_batches

    def evaluate_step(dataloader: DataLoader, device: str):
        num_batches = len(dataloader)
        model.eval()

        test_loss = 0
        regularization_loss = 0

        with torch.no_grad():
            for i, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)

                encoder_output = model(X, mode='encoder')
                pred = model(encoder_output, mode='output', logit=True)

                encoder_loss = encoder_regularization_loss(
                    encoder_output).item()
                regularization_loss += encoder_loss
                test_loss += (loss_fn(pred, y).item()
                              + regularization_weight * encoder_loss)

        test_loss /= num_batches
        return test_loss, regularization_loss / num_batches

    return train_step, evaluate_step


def train(model: nn.Module,
          *,
          train_ds: DataLoader,
          val_ds: DataLoader,
          epochs: int,
          train_noise: float = 0.0,
          early_stopping_patience: int = 100,
          device: str = 'cpu',
          lr: float = 1e-3,
          weight_decay: float = 1e-2,
          regularization_weight: float = 1.0):
    def save_checkpoint(model, path):
        torch.save(model.state_dict(), path)

    def load_checkpoint(model, path):
        model.load_state_dict(torch.load(path))
        return model

    model = model.to(device)

    train_step, val_step = create_training_and_evaluation_step(
        model,
        lr=lr,
        weight_decay=weight_decay,
        regularization_weight=regularization_weight)
    train_losses = []
    val_losses = []

    tmp_path = 'tmp_autoencoder.pth'

    patience = 0
    bar = tqdm(range(epochs), total=epochs, desc='Training')
    for epoch in bar:
        train_loss, train_regu_loss = train_step(
            train_ds, device=device, epoch=epoch, progress=False,
            train_noise=train_noise)
        train_losses.append(train_loss)

        val_loss, val_regu_loss = val_step(val_ds, device)
        val_losses.append(val_loss)

        bar.set_postfix_str(
            f'Train: {train_loss:.4f} - Val: {val_loss:.4f}'
            f'-Train Reg: {train_regu_loss:.4f} - Val Reg:{val_regu_loss:.4f}')

        patience += 1
        if val_loss <= np.min(val_losses):
            save_checkpoint(model, tmp_path)
            patience = 0
        else:
            if patience > early_stopping_patience:
                print(f'The validation does not improve for the last {patience} epochs. '
                      'Early stopping!')
                break

    # Best validation score and corresponding train score.
    best_val_idx = np.argmin(val_losses)
    print(
        f'Train: {train_losses[best_val_idx]:.4f} '
        f'- Val: {val_losses[best_val_idx]:.4f} at epoch {best_val_idx}.')

    # Restore the best model.
    print('Restore the best model.')
    return (load_checkpoint(model, tmp_path),
            dict(train_loss=train_losses,
                 val_loss=val_losses,
                 best_epoch=best_val_idx))


# %% [markdown]
# ### Features Selection

# %%
def select_important_features(
        X, y, *, n_estimators: int = 50, important_thresholds: float = 0.5):
    model = ExtraTreesClassifier(n_estimators=n_estimators)
    model.fit(X, y)
    print(model.feature_importances_)
    selector = SelectFromModel(
        model,
        prefit=True,
        threshold=important_thresholds)
    x = selector.fit_transform(X, y)
    return pd.DataFrame(x, columns=selector.get_feature_names_out())


# %% [markdown]
# ### Oversampling

# %%
def sampling(X, y):
    ros = SMOTENC(['ej'], sampling_strategy='all', random_state=0)
    columns = X.columns
    x, y = ros.fit_resample(X, y)
    return pd.DataFrame(x, columns=columns), y


# %% [markdown]
# ### Cross Validation

# %%
def balanced_log_loss(y_true, pred_prob):
    nb_class_0 = np.sum(1 - y_true)
    nb_class_1 = np.sum(y_true)

    prob_0 = np.clip(1. - pred_prob, 1e-10, 1. - 1e-10)
    prob_1 = np.clip(pred_prob, 1e-10, 1. - 1e-10)
    return (-np.sum((1 - y_true) * np.log(prob_0)) / nb_class_0
            - np.sum(y_true * np.log(prob_1)) / nb_class_1) / 2.


def calculate_optimal_prob_prediction(y_preds):
    """
    Calculate the best probability prediction based on the formula
    (in experiment 04_logistics_bayesian_v4*).

    y_preds: numpy array of shape (nb_draws, nb_data_points).
    """
    assert y_preds.ndim == 2, "Only accept 2d numpy array as input."
    _, nb_data = y_preds.shape
    print(y_preds.shape)

    # Calculate number of classes for each draw.
    nb_class_0 = np.sum(1 - y_preds, axis=1)
    print(nb_class_0.shape)
    nb_class_1 = np.sum(y_preds, axis=1)

    best_probs = []
    eps = 1e-15
    for j in range(nb_data):
        cj = np.sum(y_preds[:, j] / (nb_class_1 + eps))
        cj_1 = np.sum((1 - y_preds[:, j]) / (nb_class_0 + eps))

        prob = cj / (cj + cj_1)
        best_probs.append(prob)

    return np.asarray(best_probs)


def estimate_optimal_prob_pred(y_prob, nb_samples: int = 20000):
    ys = bernoulli.rvs(y_prob[:, None], size=(y_prob.shape[0], nb_samples))
    return calculate_optimal_prob_prediction(ys.T)


def f1_recall_precision(y_true, y_pred):
    return tuple(f(y_true, y_pred)
                 for f in [f1_score, recall_score, precision_score])


def plot_train_history(history: dict, epochs: int):
    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')

    train_loss = history['train_loss']
    val_loss = history['val_loss']

    epochs = list(range(len(train_loss)))
    ax.plot(epochs, train_loss, label='Train Loss')
    ax.plot(epochs, val_loss, label='Val Loss')
    ax.vlines(
        history['best_epoch'],
        ymin=min(min(train_loss), min(val_loss)),
        ymax=max(max(train_loss), max(val_loss)),
        label='Best epoch',
        linestyles='dashed')
    ax.legend()

    return fig


def train_and_evaluate(*,
                       Xtr, gtr, ytr,
                       Xte, gte, yte,
                       epochs: int = 100,
                       device: str = 'cpu',
                       lr: float = 1e-3,
                       train_noise: float = 0.0,
                       early_stopping_patience: int = 100,
                       correlation_threshold: float = 0.3,
                       weight_decay: float = 1e-2,
                       regularization_weight: float = 1.0):
    # First, we will normalize the data.
    Xtr = pd.DataFrame(
        preprocessing.fit_transform(Xtr, ytr),
        columns=Xtr.columns)
    Xte = pd.DataFrame(
        preprocessing.transform(Xte),
        columns=Xte.columns)

    # Next, we'll filter out correlated features.
    Xtr = filter_in_uncorrelated_features(
        Xtr, correlation_threshold=correlation_threshold)
    Xte = Xte[Xtr.columns]

    # Store original training dataset.
    Xtr_orig = Xtr.copy()
    Xtr_orig['ej'] = gtr
    ytr_orig = ytr.copy()
    print(f'Before sampling, has {len(ytr_orig)} sammples,\n'
          f'in which there are {ytr_orig.sum()} positive samples.')

    # Next, we'll perform sampling.
    Xtr['ej'] = gtr
    Xte['ej'] = gte

    Xtr, ytr = sampling(Xtr, ytr)
    print(f'After sampling, has {len(ytr)} sammples,\n'
          f'in which there are {ytr.sum()} positive samples.')

    # Then, we use tree-based model to select important features.
    # Xtr = select_important_features(
    #     Xtr, ytr,
    #     n_estimators=1000,
    #     important_thresholds='5*median')
    # Xte = Xte[Xtr.columns]
    nb_features = len(Xtr.columns)
    print('Number of important features: ', nb_features)

    # Training neural network model.
    # print(Xtr.values, ytr[:, None])
    X_train_ds = TensorDataset(
        torch.tensor(Xtr.values, dtype=torch.float32),
        torch.tensor(ytr[:, None], dtype=torch.float32))
    X_val_ds = TensorDataset(
        torch.tensor(Xte.values, dtype=torch.float32),
        torch.tensor(yte[:, None], dtype=torch.float32))

    Xtr_dataloader = DataLoader(X_train_ds, batch_size=64, shuffle=True)
    Xva_dataloader = DataLoader(X_val_ds, batch_size=64)
    model, history = train(NNClassifier(nb_features),
                           train_ds=Xtr_dataloader,
                           val_ds=Xva_dataloader,
                           epochs=epochs,
                           early_stopping_patience=early_stopping_patience,
                           device=device,
                           lr=lr,
                           weight_decay=weight_decay,
                           regularization_weight=regularization_weight,
                           train_noise=train_noise)

    # Plot training history.
    fig = plot_train_history(history, epochs=epochs)
    plt.show()
    plt.close(fig)

    # Evaluate the model.
    ytr_prob = (model(
        torch.tensor(Xtr_orig.values, dtype=torch.float32).to(device))
        .cpu().detach().numpy().squeeze())
    ytr_opt_prob = estimate_optimal_prob_pred(ytr_prob)
    ytr_pred = np.where(ytr_prob > 0.5, 1., 0.)
    (f1_train,
     recall_train,
     precision_train) = f1_recall_precision(ytr_orig, ytr_pred)
    log_loss_train = balanced_log_loss(ytr_orig, ytr_prob)
    opt_log_loss_train = balanced_log_loss(ytr_orig, ytr_opt_prob)
    print(f'Train - f1={f1_train:.4f} recall={recall_train:.4f} '
          f'precision={precision_train:.4f} log-loss={log_loss_train:.4f} '
          f'opt-log-loss={opt_log_loss_train:.4f}')

    yte_prob = (model(
        torch.tensor(Xte.values, dtype=torch.float32).to(device))
        .cpu().detach().numpy().squeeze())
    yte_opt_prob = estimate_optimal_prob_pred(yte_prob)
    yte_pred = np.where(yte_prob > 0.5, 1., 0.)
    (f1_test,
     recall_test,
     precision_test) = f1_recall_precision(yte, yte_pred)
    log_loss_test = balanced_log_loss(yte, yte_prob)
    opt_log_loss_test = balanced_log_loss(yte, yte_opt_prob)
    print(f'Test  - f1={f1_test:.4f} recall={recall_test:.4f} '
          f'precision={precision_test:.4f} log-loss={log_loss_test:.4f} '
          f'opt-log-loss={opt_log_loss_test:.4f}')

    # Return results.
    return dict(
        f1_train=f1_train,
        f1_test=f1_test,
        log_loss_train=log_loss_train,
        opt_log_loss_train=opt_log_loss_train,
        log_loss_test=log_loss_test,
        opt_log_loss_test=opt_log_loss_test,
    )


def cross_validations(*, X, grp, y,
                      n_folds: int = 10,
                      **kwargs):
    results = []

    kfolds = StratifiedKFold(n_splits=n_folds)
    for i, (train_idx, test_idx) in enumerate(kfolds.split(X, y)):
        print(f'\n-- Fold # {i + 1}/{n_folds}:')

        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        gtr, gte = grp[train_idx], grp[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        result = train_and_evaluate(
            Xtr=Xtr, gtr=gtr, ytr=ytr,
            Xte=Xte, gte=gte, yte=yte,
            **kwargs)
        results.append(result)

    return pd.DataFrame(results)


cv_results = cross_validations(
    X=Xtrain_df,
    grp=ej.cat.codes.values,
    y=y,
    n_folds=10,
    epochs=2000,
    correlation_threshold=0.3,
    lr=1e-4,
    early_stopping_patience=100,
    weight_decay=1e-2,
    regularization_weight=1.0,
    train_noise=0.00)

# %%
cv_results.describe()

# %% [markdown]
# * Without optimal prob estimation: (first hidden layer = 2048 -> 512 -> 64 -> 1), correlation_threshold=0.3, lr=1e-3:
#
# |       |  f1_train |   f1_test | log_loss_train | log_loss_test |
# |------:|----------:|----------:|---------------:|--------------:|
# | count | 10.000000 | 10.000000 | 10.000000      | 10.000000     |
# |  mean | 0.993788  | 0.788743  | 0.103287       | 0.355109      |
# |  std  | 0.006351  | 0.110078  | 0.022831       | 0.120017      |
# |  min  | 0.984456  | 0.625000  | 0.080745       | 0.177904      |
# |  25%  | 0.989583  | 0.684211  | 0.088007       | 0.268115      |
# |  50%  | 0.994819  | 0.809091  | 0.100123       | 0.350198      |
# |  75%  | 1.000000  | 0.866460  | 0.107109       | 0.404449      |
# |  max  | 1.000000  | 0.947368  | 0.159215       | 0.600327      |
#
# * With optimal prob estimation:
#
# |       |  f1_train |   f1_test | log_loss_train | opt_log_loss_train | log_loss_test | opt_log_loss_test |
# |------:|----------:|----------:|---------------:|-------------------:|--------------:|------------------:|
# | count | 10.000000 | 10.000000 | 10.000000      | 10.000000          | 10.000000     | 10.000000         |
# |  mean | 0.986986  | 0.800841  | 0.116041       | 0.169590           | 0.355927      | 0.331922          |
# |  std  | 0.020946  | 0.108112  | 0.049985       | 0.049339           | 0.091808      | 0.080742          |
# |  min  | 0.932039  | 0.545455  | 0.081386       | 0.135817           | 0.234064      | 0.249415          |
# |  25%  | 0.986925  | 0.771429  | 0.091549       | 0.145445           | 0.281071      | 0.264392          |
# |  50%  | 0.994872  | 0.800000  | 0.099473       | 0.154371           | 0.351739      | 0.315641          |
# |  75%  | 0.998718  | 0.880952  | 0.118962       | 0.167156           | 0.400678      | 0.363595          |
# |  max  | 1.000000  | 0.909091  | 0.252682       | 0.303835           | 0.500932      | 0.500320          |
#
# * With optimal prob estimation & dropout in output layer:
#
# |       |  f1_train |   f1_test | log_loss_train | opt_log_loss_train | log_loss_test | opt_log_loss_test |
# |------:|----------:|----------:|---------------:|-------------------:|--------------:|------------------:|
# | count | 10.000000 | 10.000000 | 10.000000      | 10.000000          | 10.000000     | 10.000000         |
# |  mean | 0.994306  | 0.785250  | 0.120901       | 0.177781           | 0.367242      | 0.341151          |
# |  std  | 0.007514  | 0.120909  | 0.031118       | 0.031392           | 0.088936      | 0.073599          |
# |  min  | 0.979167  | 0.500000  | 0.098379       | 0.154592           | 0.252296      | 0.251806          |
# |  25%  | 0.990892  | 0.765873  | 0.104631       | 0.162917           | 0.292936      | 0.296841          |
# |  50%  | 0.997436  | 0.832817  | 0.108945       | 0.164559           | 0.371133      | 0.328557          |
# |  75%  | 1.000000  | 0.857143  | 0.116280       | 0.174167           | 0.411316      | 0.376181          |
# |  max  | 1.000000  | 0.909091  | 0.201384       | 0.255014           | 0.524572      | 0.493841          |
#
# * Dropout in output layer and first hidden layer = 1024 -> 512 -> 64 -> 1:
#
# |       |  f1_train |   f1_test | log_loss_train | opt_log_loss_train | log_loss_test | opt_log_loss_test |
# |------:|----------:|----------:|---------------:|-------------------:|--------------:|------------------:|
# | count | 10.000000 | 10.000000 | 10.000000      | 10.000000          | 10.000000     | 10.000000         |
# |  mean | 0.998456  | 0.844092  | 0.110776       | 0.163615           | 0.334471      | 0.315234          |
# |  std  | 0.002486  | 0.071594  | 0.023754       | 0.027730           | 0.106961      | 0.075233          |
# |  min  | 0.994819  | 0.700000  | 0.094138       | 0.144062           | 0.233724      | 0.238359          |
# |  25%  | 0.996154  | 0.814286  | 0.100475       | 0.152444           | 0.253596      | 0.262361          |
# |  50%  | 1.000000  | 0.863354  | 0.103398       | 0.155378           | 0.311073      | 0.286363          |
# |  75%  | 1.000000  | 0.897222  | 0.110759       | 0.162789           | 0.373330      | 0.345970          |
# |  max  | 1.000000  | 0.909091  | 0.176333       | 0.240405           | 0.576234      | 0.483375          |
#
# |       |  f1_train |   f1_test | log_loss_train | opt_log_loss_train | log_loss_test | opt_log_loss_test |
# |------:|----------:|----------:|---------------:|-------------------:|--------------:|------------------:|
# | count | 10.000000 | 10.000000 | 10.000000      | 10.000000          | 10.000000     | 10.000000         |
# |  mean | 0.996378  | 0.793288  | 0.110664       | 0.162491           | 0.368439      | 0.328008          |
# |  std  | 0.004264  | 0.067822  | 0.015997       | 0.019841           | 0.091669      | 0.057852          |
# |  min  | 0.989583  | 0.666667  | 0.097384       | 0.146242           | 0.246263      | 0.234801          |
# |  25%  | 0.994819  | 0.745455  | 0.102548       | 0.150054           | 0.304998      | 0.300062          |
# |  50%  | 0.997436  | 0.800000  | 0.103951       | 0.156243           | 0.364613      | 0.325831          |
# |  75%  | 1.000000  | 0.837461  | 0.114936       | 0.166609           | 0.398645      | 0.354763          |
# |  max  | 1.000000  | 0.888889  | 0.150657       | 0.213085           | 0.579735      | 0.431269          |
#
# * Dropout in output layer and first hidden layer = 512 -> 256 -> 64 -> 1:
#
# |       |  f1_train |   f1_test | log_loss_train | opt_log_loss_train | log_loss_test | opt_log_loss_test |
# |------:|----------:|----------:|---------------:|-------------------:|--------------:|------------------:|
# | count | 10.000000 | 10.000000 | 10.000000      | 10.000000          | 10.000000     | 10.000000         |
# |  mean | 0.997446  | 0.779095  | 0.110231       | 0.165756           | 0.355884      | 0.329831          |
# |  std  | 0.006477  | 0.098293  | 0.026850       | 0.026549           | 0.088056      | 0.070384          |
# |  min  | 0.979592  | 0.636364  | 0.093054       | 0.144840           | 0.260623      | 0.261455          |
# |  25%  | 1.000000  | 0.684211  | 0.101231       | 0.156956           | 0.284495      | 0.282450          |
# |  50%  | 1.000000  | 0.797980  | 0.102794       | 0.159830           | 0.325439      | 0.300359          |
# |  75%  | 1.000000  | 0.851190  | 0.104328       | 0.163339           | 0.398867      | 0.375037          |
# |  max  | 1.000000  | 0.909091  | 0.185507       | 0.238555           | 0.510923      | 0.475515          |

# %% [markdown]
# ---
# * Architecture: 1024 (ReLu, Dropout, LayerNorm) -> 512 (ReLu, Dropout, LayerNorm) -> 64 (LayerNorm) -> 1
# * Correlation Threshold: 0.3
# * LR: 1-4
# * Weight Decay: 1e-2
# * Regularization weight: 1.0
#
# |       |  f1_train |   f1_test | log_loss_train | opt_log_loss_train | log_loss_test | opt_log_loss_test |
# |------:|----------:|----------:|---------------:|-------------------:|--------------:|------------------:|
# | count | 10.000000 | 10.000000 | 10.000000      | 10.000000          | 10.000000     | 10.000000         |
# |  mean | 0.938530  | 0.742627  | 0.092418       | 0.077374           | 0.405253      | 0.312122          |
# |  std  | 0.051880  | 0.166587  | 0.084953       | 0.058434           | 0.204523      | 0.142484          |
# |  min  | 0.855615  | 0.421053  | 0.007355       | 0.007776           | 0.134701      | 0.123478          |
# |  25%  | 0.904306  | 0.727273  | 0.017311       | 0.030913           | 0.286190      | 0.225457          |
# |  50%  | 0.934707  | 0.768421  | 0.068624       | 0.077364           | 0.401148      | 0.307984          |
# |  75%  | 0.991063  | 0.856719  | 0.143486       | 0.111764           | 0.517706      | 0.411845          |
# |  max  | 0.994872  | 0.956522  | 0.235547       | 0.186239           | 0.792556      | 0.572658          |
#
# * Train noise: 0.0
#
# |       |  f1_train |   f1_test | log_loss_train | opt_log_loss_train | log_loss_test | opt_log_loss_test |
# |------:|----------:|----------:|---------------:|-------------------:|--------------:|------------------:|
# | count | 10.000000 | 10.000000 | 10.000000      | 10.000000          | 10.000000     | 10.000000         |
# |  mean | 0.925390  | 0.761399  | 0.114047       | 0.102715           | 0.392968      | 0.303210          |
# |  std  | 0.051459  | 0.138520  | 0.074306       | 0.051025           | 0.149052      | 0.131872          |
# |  min  | 0.863388  | 0.454545  | 0.025461       | 0.025568           | 0.247956      | 0.159256          |
# |  25%  | 0.882754  | 0.712215  | 0.047980       | 0.054976           | 0.261569      | 0.208438          |
# |  50%  | 0.917609  | 0.788889  | 0.114155       | 0.115892           | 0.350852      | 0.258403          |
# |  75%  | 0.966401  | 0.852174  | 0.178983       | 0.140492           | 0.484499      | 0.358687          |
# |  max  | 1.000000  | 0.909091  | 0.229640       | 0.167953           | 0.637699      | 0.536559          |
#
# ---
# ---
# * Architecture: 1024 (ReLu, Dropout, LayerNorm) -> 512 (ReLu, Dropout, LayerNorm) -> 64 (LayerNorm, Droput) -> 1
# * Correlation Threshold: 0.3
# * LR: 1-4
# * Weight Decay: 1e-2
# * Regularization weight: 1.0
#
# |       |  f1_train |   f1_test | log_loss_train | opt_log_loss_train | log_loss_test | opt_log_loss_test |
# |------:|----------:|----------:|---------------:|-------------------:|--------------:|------------------:|
# | count | 10.000000 | 10.000000 | 10.000000      | 10.000000          | 10.000000     | 10.000000         |
# |  mean | 0.915363  | 0.719805  | 0.126651       | 0.111713           | 0.465246      | 0.367487          |
# |  std  | 0.055941  | 0.105371  | 0.076567       | 0.057464           | 0.178936      | 0.146604          |
# |  min  | 0.831050  | 0.500000  | 0.027534       | 0.032577           | 0.258689      | 0.207962          |
# |  25%  | 0.880640  | 0.643939  | 0.059859       | 0.065312           | 0.338920      | 0.265986          |
# |  50%  | 0.904639  | 0.769841  | 0.129620       | 0.116963           | 0.434503      | 0.319276          |
# |  75%  | 0.968530  | 0.795652  | 0.171979       | 0.148954           | 0.576692      | 0.463765          |
# |  max  | 0.984456  | 0.818182  | 0.241858       | 0.195355           | 0.810943      | 0.603020          |
#
# ---
# ---
# * Architecture: 1024 (ReLu, Dropout, LayerNorm) -> 512 (ReLu, Dropout, LayerNorm) -> 64 (LayerNorm) -> 1
# * Correlation Threshold: 0.3
# * LR: 1-4
# * Train Noise: 0.1
# * Weight Decay: 1e-2
# * Regularization weight: 1.0
#
# |       |  f1_train |   f1_test | log_loss_train | opt_log_loss_train | log_loss_test | opt_log_loss_test |
# |------:|----------:|----------:|---------------:|-------------------:|--------------:|------------------:|
# | count | 10.000000 | 10.000000 | 10.000000      | 10.000000          | 10.000000     | 10.000000         |
# |  mean | 0.910933  | 0.756726  | 0.126636       | 0.113645           | 0.469752      | 0.351037          |
# |  std  | 0.075484  | 0.096224  | 0.099148       | 0.087800           | 0.230167      | 0.169355          |
# |  min  | 0.797927  | 0.600000  | 0.014958       | 0.018010           | 0.130632      | 0.122560          |
# |  25%  | 0.854972  | 0.709211  | 0.053874       | 0.034216           | 0.355805      | 0.245166          |
# |  50%  | 0.911154  | 0.761905  | 0.087230       | 0.102653           | 0.455400      | 0.328301          |
# |  75%  | 0.981675  | 0.822055  | 0.193842       | 0.178877           | 0.545046      | 0.433651          |
# |  max  | 1.000000  | 0.909091  | 0.291494       | 0.239619           | 0.972168      | 0.712043          |
#
# ---
# ---
# * Architecture: 1024 (ReLu, Dropout, LayerNorm) -> 512 (ReLu, Dropout, LayerNorm) -> 64 (LayerNorm) -> 1
# * Correlation Threshold: 0.3
# * LR: 1-4
# * Train Noise: 0.01
# * Weight Decay: 1e-2
# * Regularization weight: 1.0
#
# |       |  f1_train |   f1_test | log_loss_train | opt_log_loss_train | log_loss_test | opt_log_loss_test |
# |------:|----------:|----------:|---------------:|-------------------:|--------------:|------------------:|
# | count | 10.000000 | 10.000000 | 10.000000      | 10.000000          | 10.000000     | 10.000000         |
# |  mean | 0.912129  | 0.738721  | 0.123202       | 0.110247           | 0.456864      | 0.354187          |
# |  std  | 0.052634  | 0.114834  | 0.074895       | 0.051595           | 0.204676      | 0.165713          |
# |  min  | 0.849741  | 0.583333  | 0.024473       | 0.038642           | 0.169423      | 0.130206          |
# |  25%  | 0.863509  | 0.623913  | 0.065739       | 0.079485           | 0.295615      | 0.233858          |
# |  50%  | 0.915841  | 0.761905  | 0.108148       | 0.102719           | 0.413764      | 0.310698          |
# |  75%  | 0.950533  | 0.813636  | 0.187869       | 0.137312           | 0.627799      | 0.460436          |
# |  max  | 0.984456  | 0.909091  | 0.237448       | 0.196157           | 0.783205      | 0.666412          |
