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

        self.guess_prob = nn.Parameter(
            torch.randn(1),
            requires_grad=True)

        self.output = nn.Sequential(
            # nn.Dropout(),
            nn.LayerNorm(64),
            # nn.Dropout(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, *,
                mode: Literal['encoder', 'output', 'full'] = 'full',
                include_guess: bool = True):
        if mode == 'encoder':
            return self.encoder(x)
        elif mode == 'output':
            return self._calculate_prob(x, include_guess=include_guess)
        elif mode == 'full':
            x = self.encoder(x)
            return self._calculate_prob(x, include_guess=include_guess)

        raise ValueError(f'Unknown mode={mode}')

    def _calculate_prob(self, x, *, include_guess: bool):
        prob = self.output(x)
        guess_prob = torch.sigmoid(self.guess_prob)
        return ((1 - guess_prob) * prob + 0.5 * guess_prob
                if include_guess
                else prob)


def create_training_and_evaluation_step(
        model: nn.Module,
        lr=1e-3,
        weight_decay=1e-5,
        regularization_weight=1e-2):

    loss_fn = nn.BCELoss()

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
            pred = model(encoder_output, mode='output')

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

    def evaluate_step(dataloader: DataLoader, *,
                      device: str,
                      include_guess: bool = True):
        num_batches = len(dataloader)
        model.eval()

        test_loss = 0
        regularization_loss = 0

        with torch.no_grad():
            for i, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)

                encoder_output = model(X, mode='encoder')
                pred = model(
                    encoder_output, mode='output', include_guess=include_guess)

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
          regularization_weight: float = 1.0,
          evaluate_include_guess: bool = True):
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

        val_loss, val_regu_loss = val_step(
            val_ds,
            device=device,
            include_guess=evaluate_include_guess)
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
                       regularization_weight: float = 1.0,
                       evaluate_include_guess: bool = True):
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
                           train_noise=train_noise,
                           evaluate_include_guess=evaluate_include_guess)

    # Plot training history.
    # fig = plot_train_history(history, epochs=epochs)
    # plt.show()
    # plt.close(fig)

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
        torch.tensor(Xte.values, dtype=torch.float32).to(device),
        include_guess=evaluate_include_guess)
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
                      repeats_per_fold: int = 1,
                      **kwargs):
    results = []

    kfolds = StratifiedKFold(n_splits=n_folds)
    for fold, (train_idx, test_idx) in enumerate(kfolds.split(X, y)):
        for repeat in range(repeats_per_fold):
            print(f'\n-- Fold # {fold + 1}/{n_folds} - '
                  f'Repeat #{repeat + 1}/{repeats_per_fold}:')

            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            gtr, gte = grp[train_idx], grp[test_idx]
            ytr, yte = y[train_idx], y[test_idx]

            result = train_and_evaluate(
                Xtr=Xtr, gtr=gtr, ytr=ytr,
                Xte=Xte, gte=gte, yte=yte,
                **kwargs)

            result['fold'] = fold + 1
            result['repeat'] = repeat + 1
            results.append(result)

    return pd.DataFrame(results)


cv_results = cross_validations(
    X=Xtrain_df,
    grp=ej.cat.codes.values,
    y=y,
    n_folds=10,
    repeats_per_fold=10,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    epochs=2000,
    correlation_threshold=0.3,
    lr=1e-4,
    early_stopping_patience=100,
    weight_decay=1e-2,
    regularization_weight=1.0,
    train_noise=0.01,
    evaluate_include_guess=False)


# %%
cv_results


# %%
def get_optimal_cv_results_each_fold(results: pd.DataFrame,
                                     by: str,
                                     method: Literal['max', 'min'] = 'max'):
    folds = results.groupby('fold')

    optimal_results = []
    for fold, fold_results in folds:
        best_idx = (fold_results[by].argmax()
                    if method == 'max'
                    else fold_results[by].argmin())
        optimal_result = fold_results.iloc[best_idx].to_dict()
        optimal_results.append(optimal_result)

    return pd.DataFrame(optimal_results)


# %% [markdown]
# ## Results
# ### Optimal F1 Test

# %%
# Optimal results per fold by f1_test.
cv_results_optimal_f1_test = get_optimal_cv_results_each_fold(
    cv_results, 'f1_test', 'max')
cv_results_optimal_f1_test

# %%
cv_results_optimal_f1_test.describe()

# %% [markdown]
# ### Optimal `opt_log_loss_test`

# %%
cv_results_optimal_log_loss_test = get_optimal_cv_results_each_fold(
    cv_results, 'opt_log_loss_test', 'min')
cv_results_optimal_log_loss_test

# %%
cv_results_optimal_log_loss_test.describe()

# %% [markdown]
# ### Discussions
#
# From the results above:
#
# * With the inclusion of guess term, the mean f1_score seems to improve, but at the expense of the log_loss score;
# * The fold #4 and #5's results were pretty good in the previous experiment but not in this experiment;
# * Seems to have more overfitting;
# * The fold #6 And #7's results are not improved;
# * Fold #9's result looks a bit better.

# %% [markdown]
# # Results Overview
#
# ---
# * Architecture: 1024 (ReLu, Dropout, LayerNorm) -> 512 (ReLu, Dropout, LayerNorm) -> 64 (LayerNorm) -> 1
# * Correlation Threshold: 0.3
# * LR: 1-4
# * Weight Decay: 1e-2
# * Regularization weight: 1.0
# * Train noise: 0.0
