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


import jax
from jax import random
from jax.scipy.optimize import minimize as jax_minimize
from jax.scipy.special import expit, logit
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.initialization import init_to_median
import numpyro.distributions as dist
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from typing import Literal


# Still using cpu for numpyro,
# not sure because my installation sucks, but using gpu is slower than cpu.
numpyro.set_platform("cpu")
numpyro.set_host_device_count(16)

# %%
kaggle_submission = False

# %% [markdown]
# # Neural Network Classifier (v2)
#
# In this experiment,
# I'll use neural network to classify the data.
# And built upon the previous autoencoder experiment,
# this version will add a regularization term:
# the embedded feature space should be from iid normal gaussian.
# Unlike the previous version of this notebook,
# instead of using Logistic Regression,
# we'll use Bayesian logisitc regression.
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
preprocessing = Pipeline([
    # ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

# %%
X_df = train_df.drop(columns=['Id', 'Class', 'EJ'])
y = train_df['Class']
ej = train_df['EJ'].astype('category')

# Fill missing values with medians.
X_df = X_df.fillna(X_df.median())

X_df = pd.DataFrame(
    preprocessing.fit_transform(X_df),
    columns=X_df.columns,
    index=X_df.index)
y = y.values

# %%
Xtest_df = test_df.drop(columns=['Id', 'EJ'])
ej_test = test_df['EJ'].astype(ej.dtype)

Xtest_df = pd.DataFrame(
    preprocessing.transform(Xtest_df),
    columns=Xtest_df.columns,
    index=Xtest_df.index)

# %% [markdown]
# ## Neural Network Classifier & Bayesian Logistic Regression

# %%
# ================
# Neural Network Classifier
class NNClassifier(nn.Module):
    def __init__(self, input_shape: int) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 2048), nn.ReLU(), nn.Dropout(), nn.LayerNorm(2048),
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(), nn.LayerNorm(512),
            nn.Linear(512, 64),
        )

        self.output = nn.Sequential(
            nn.Linear(64, 1),
        )

    def forward(self, x, mode: Literal['encoder', 'output', 'full'] = 'full', logit: bool = False):
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
        if type(m) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.LazyConv1d, nn.LazyLinear]:
            strategy_fn(m.weight) # pyright: ignore

    model.apply(lambda m: init_weights(m))

    def encoder_regularization_loss(x: torch.Tensor):
        """
        Calculate regularization loss of the encoder's output.

        x should have shape (nb_batches, nb_features)
        """
        # First, the output should be normally distributed.
        l1 = torch.mean(torch.sum(x**2, axis=1)) # pyright: ignore

        # Second, features should not be correlated.
        cov = torch.t(x) @ x
        cov = cov - torch.diag(torch.diag(cov))
        l2 = torch.mean(torch.abs(cov))

        return l1 + l2

    def train_step(dataloader: DataLoader, device: str, epoch: int, progress: bool = True):
        model.train()

        train_loss = 0
        regularization_loss = 0

        num_batches = len(dataloader)
        bar = (tqdm(enumerate(dataloader), total=num_batches, desc=f'Epoch {epoch}')
               if progress
               else enumerate(dataloader))
        for i, (X, y) in bar:
            X, y = X.to(device), y.to(device)

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
                bar.set_postfix_str( # pyright: ignore
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

                encoder_loss = encoder_regularization_loss(encoder_output).item()
                regularization_loss += encoder_loss
                test_loss += (loss_fn(pred, y).item()
                              + regularization_weight * encoder_loss)

        test_loss /= num_batches
        return test_loss, regularization_loss / num_batches

    return train_step, evaluate_step


# 
def train(model: nn.Module,
          *,
          train_ds: DataLoader,
          val_ds: DataLoader,
          epochs: int,
          early_stopping_patience: int = 10,
          device: str = 'cpu'):
    def save_checkpoint(model, path):
        torch.save(model.state_dict(), path)

    def load_checkpoint(model, path):
        model.load_state_dict(torch.load(path))
        return model


    model = model.to(device)

    train_step, val_step = create_training_and_evaluation_step(
        model, weight_decay=1e-2, regularization_weight=1.0)
    train_losses = []
    val_losses = []

    tmp_path = 'tmp_autoencoder.pth'

    bar = tqdm(range(epochs), total=epochs, desc='Training')
    for epoch in bar:
        train_loss, train_regu_loss = train_step(train_ds, device, epoch, progress=False)
        train_losses.append(train_loss)

        val_loss, val_regu_loss = val_step(val_ds, device)
        val_losses.append(val_loss)

        bar.set_postfix_str(f'Train: {train_loss:.4f} - Val: {val_loss:.4f}'
                            f'- Train Regu: {train_regu_loss:.4f} - Val Regu: {val_regu_loss:.4f}')

        if val_loss <= np.min(val_losses):
            save_checkpoint(model, tmp_path)

    # Best validation score and corresponding train score.
    best_val_idx = np.argmin(val_losses)
    print(f'Train: {train_losses[best_val_idx]:.4f} - Val: {val_losses[best_val_idx]:.4f} at epoch {best_val_idx}.')

    # Restore the best model.
    print('Restore the best model.')
    return load_checkpoint(model, tmp_path)


# ================
def balanced_log_loss(y_true, pred_prob):
    nb_class_0 = np.sum(1 - y_true)
    nb_class_1 = np.sum(y_true)

    prob_0 = np.clip(1. - pred_prob, 1e-10, 1. - 1e-10)
    prob_1 = np.clip(pred_prob, 1e-10, 1. - 1e-10)
    return (-np.sum((1 - y_true) * np.log(prob_0)) / nb_class_0
            - np.sum(y_true * np.log(prob_1)) / nb_class_1) / 2.


# ================
# Bayesian Logistic Regression
# and related stuffs.
def bayesian_logistic_regression(*,
                                 X: jax.Array,
                                 group: jax.Array,
                                 nb_groups: int,
                                 y: jax.Array | None = None):
    nb_obs, nb_features = X.shape
    assert nb_obs == group.shape[0]

    # Prior for baselines.
    a0 = numpyro.sample('_a0', dist.Normal(0., 1.))

    # Prior for the coefficients of metric features.
    a_sigma = numpyro.sample('_aSigma', dist.Gamma(1., 1.))
    a = numpyro.sample(
            '_a', dist.StudentT(1., 0., a_sigma).expand((nb_features, ))) # pyright: ignore

    # Prior for the group feature.
    aG = numpyro.sample('_aG', dist.Normal(0., 1.).expand((nb_groups, )))

    # Prior for guess term.
    guess = numpyro.sample('guess', dist.Beta(1., 1.))

    # Observations.
    with numpyro.plate('obs', nb_obs) as idx:
        prob = numpyro.deterministic(
            'prob', expit(a0 + jnp.dot(X[idx], a) + aG[group[idx]]))
        guess_prob = numpyro.deterministic(
                'prob_w_guess', guess * 0.5 + (1 - guess) * prob) # pyright: ignore
        if y is not None:
            numpyro.sample('y', dist.Bernoulli(guess_prob), obs=y[idx])
        else:
            numpyro.sample('y', dist.Bernoulli(guess_prob))


class BayesianLogisticRegression:
    def __init__(self):
        self._mcmc = None

    def fit(self, *,
            X: np.ndarray,
            y: np.ndarray,
            grp: np.ndarray,
            nb_groups: int,
            num_warmup: int = 1000,
            num_samples: int = 20000,
            num_chains: int = 1):
        kernel = NUTS(bayesian_logistic_regression,
                      init_strategy=init_to_median) # pyright: ignore
        mcmc = MCMC(kernel,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains)
        mcmc.run(
            random.PRNGKey(0),
            X=jnp.array(X),
            y=jnp.array(y),
            nb_groups=nb_groups,
            group=jnp.array(grp),
        )

        self._nb_groups = nb_groups
        self._mcmc = mcmc
        return self

    def predict(self, *, X, grp, return_median: bool = True):
        assert self._mcmc is not None, 'Please call `fit` with training data first!'

        predictive = Predictive(bayesian_logistic_regression,
                                self._mcmc.get_samples(),
                                return_sites=['y'])
        predictions = predictive(
            random.PRNGKey(1),
            X=jnp.array(X),
            nb_groups=self._nb_groups,
            group=jnp.array(grp)
        )

        y = predictions['y']
        return jnp.median(y, axis=0) if return_median else y


def calculate_optimal_prob_prediction(y_preds: np.ndarray):
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


# ==================
# Data
X = X_df.values

# Training with 10-fold cross validation.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
kfold = StratifiedKFold(n_splits=10)

train_log_losses = []
train_f1_scores = []
val_log_losses = []
val_f1_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    Xtr = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
    Xva = torch.tensor(X[val_idx], dtype=torch.float32).to(device)
    ejtr = ej[train_idx]
    ytr = y[train_idx]
    ejva = ej[val_idx]
    yva = y[val_idx]

    X_train_ds = TensorDataset(
            Xtr, torch.tensor(ytr[:, None], dtype=torch.float32)) # pyright: ignore
    X_val_ds = TensorDataset(
            Xva, torch.tensor(yva[:, None], dtype=torch.float32)) # pyright: ignore

    Xtr_dataloader = DataLoader(X_train_ds, batch_size=64, shuffle=True)
    Xva_dataloader = DataLoader(X_val_ds, batch_size=64)

    classifier = NNClassifier(X.shape[1]).to(device)
    classifier = train(classifier,
                       train_ds=Xtr_dataloader,
                       val_ds=Xva_dataloader,
                       epochs=400,
                       early_stopping_patience=10,
                       device=device)

    # Use the autoencoder to encode data and train logistic regression model.
    Xtr_encoded = classifier(Xtr, mode='encoder').detach().cpu().numpy()
    Xva_encoded = classifier(Xva, mode='encoder').detach().cpu().numpy()

    logistic = (BayesianLogisticRegression()
                .fit(X=Xtr_encoded,
                     y=ytr,
                     nb_groups=ej.cat.categories.size,
                     grp=ejtr.cat.codes.values,
                     num_chains=4,
                     num_samples=5000))

    # Show the performance measures.
    ytr_preds = logistic.predict(
            X=Xtr_encoded, grp=ejtr.cat.codes.values, return_median=False)
    yva_preds = logistic.predict(
            X=Xva_encoded, grp=ejva.cat.codes.values, return_median=False)

    ytr_pred = np.median(ytr_preds, axis=0)
    ytr_prob = calculate_optimal_prob_prediction(ytr_preds)

    yva_pred = np.median(yva_preds, axis=0)
    yva_prob = calculate_optimal_prob_prediction(yva_preds)

    f1_train = f1_score(ytr, ytr_pred)
    precision_train = precision_score(ytr, ytr_pred)
    recall_train = recall_score(ytr, ytr_pred)
    log_loss_train = balanced_log_loss(ytr, ytr_prob)
    print(f'Train - f1={f1_train:.4f} recall={recall_train:.4f} precision={precision_train:.4f} log-loss={log_loss_train:.4f}')

    f1_val = f1_score(yva, yva_pred)
    precision_val = precision_score(yva, yva_pred)
    recall_val = recall_score(yva, yva_pred)
    log_loss_val = balanced_log_loss(yva, yva_prob)
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

# %% [markdown]
# __Corrected covariance regularization.__
#
# With `regularization_weight = 1e-1`:
#
# - Train - F1=0.9974 +- 0.0026; Log Loss = 0.0106 +- 0.0070
# - Valid - F1=0.8289 +- 0.1110; Log Loss = 0.3920 +- 0.1720
#
# With `regularization_weight = 1.0`:
#
# - Train - F1=0.9927 +- 0.0058; Log Loss = 0.0307 +- 0.0125
# - Valid - F1=0.7967 +- 0.1029; Log Loss = 0.4586 +- 0.2006