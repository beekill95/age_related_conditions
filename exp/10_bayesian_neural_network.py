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
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive
from pyro.nn import PyroModule, PyroSample
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn


# %%
kaggle_submission = False
run_cv = True

# %%
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(f'{device=}')

# %% [markdown]
# # Bayesian Neural Networks
#
# In this experiment,
# I'll use Bayesian deep neural networks for the task.
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
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

# %%
X_df = train_df.drop(columns=['Id', 'Class', 'EJ'])
y = train_df['Class']
ej = train_df['EJ'].astype('category')

X_df = pd.DataFrame(
    preprocessing.fit_transform(X_df),
    columns=X_df.columns,
    index=X_df.index)
y = y.values


# %% [markdown]
# ## Bayesian Deep Neural Networks with Gaussian Priors

# %%
class BayesianDNN(PyroModule):
    def __init__(self, in_dim: int, *,
                 hidden_dim: int = 100,
                 out_dim: int = 1,
                 prior_scale: float = 10):
        super().__init__()

        self.activation = nn.ReLU()
        self.layer1 = PyroModule[nn.Linear](in_dim, hidden_dim)
        self.layer2 = PyroModule[nn.Linear](hidden_dim, out_dim)

        # Set layer parameters as random variables
        self.layer1.weight = PyroSample(
            dist.Normal(0., prior_scale)
            .expand([hidden_dim, in_dim]).to_event(2))
        self.layer1.bias = PyroSample(
            dist.Normal(0., prior_scale).expand([hidden_dim]).to_event(1))
        self.layer2.weight = PyroSample(
            dist.Normal(0., prior_scale)
            .expand([out_dim, hidden_dim]).to_event(2))
        self.layer2.bias = PyroSample(
            dist.Normal(0., prior_scale).expand([out_dim]).to_event(1))

    def forward(self, x, y=None):
        # x = x.reshape(-1, 1)
        x = self.activation(self.layer1(x))
        mu = self.layer2(x).squeeze()

        # FIXME: using guess_prob reduces the performance of the model
        # (while training, see below!)
        # Prior for guess term.
        # guess = pyro.sample(
        #     'guess', dist.Beta(torch.tensor(1., device=x.device), 1.))

        # Probabilities.
        prob = pyro.deterministic('prob', torch.special.expit(mu))

        # FIXME: using guess_prob reduces the performance of the model
        # (while training, see below!)
        # guess_prob = pyro.deterministic(
        #     'guess_prob', guess * 0.5 + (1 - guess) * prob)

        # Sampling model
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Bernoulli(prob), obs=y)

        return obs


# %% [markdown]
# ### Training

# %%
model = BayesianDNN(
    55,
    hidden_dim=50,
    prior_scale=torch.tensor(10., device=device)).to(device)

# Set Pyro random seed
pyro.set_rng_seed(42)

# Define Hamiltonian Monte Carlo (HMC) kernel
# NUTS = "No-U-Turn Sampler" (https://arxiv.org/abs/1111.4246),
# gives HMC an adaptive step size.
# jit_compile=True is faster but requires PyTorch 1.6+
nuts_kernel = NUTS(model, jit_compile=True)

# Define MCMC sampler, get 50 posterior samples
mcmc = MCMC(nuts_kernel, warmup_steps=10, num_samples=10)

# Convert data to PyTorch tensors
x_train = torch.from_numpy(X_df.values).float().to(device)
y_train = torch.from_numpy(y).float().to(device)

# Run MCMC
mcmc.run(x_train, y_train)
mcmc.diagnostics()['divergences']

# %% [markdown]
# ### Predictions

# %%
predictive = Predictive(model=model, posterior_samples=mcmc.get_samples())
y_preds = predictive(x_train)['obs'].cpu().detach().numpy()


# %% [markdown]
# ### Evaluations

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


y_pred = np.median(y_preds, axis=0).astype(np.int64)
y_prob = calculate_optimal_prob_prediction(y_preds)

# Calculate metrics.
y_true = y_train.cpu().detach().numpy().astype(np.int64)
f1_train = f1_score(y_true, y_pred)
precision_train = precision_score(y_true, y_pred)
recall_train = recall_score(y_true, y_pred)
log_loss_train = balanced_log_loss(y_true, y_prob)
print(f'Train - f1={f1_train:.4f} recall={recall_train:.4f} '
      f'precision={precision_train:.4f} log-loss={log_loss_train:.4f}')

# %% [markdown]
# With guess_prob: (200 hidden units)
#
# * Train - f1=0.5290 recall=0.3796 precision=0.8723 log-loss=0.5541
#
# Without guess_prob: (200 hidden units)
#
# * Train - f1=0.7614 recall=0.6204 precision=0.9853 log-loss=3.8616
#
# Without guess_prob: (100 hidden units)
#
# * Train - f1=0.8808 recall=0.7870 precision=1.0000 log-loss=2.4525
#
# Without guess_prob: (50 hidden units)
#
# * Train - f1=0.9907 recall=0.9815 precision=1.0000 log-loss=0.2140


# %% [markdown]
# ### Cross-Validation

# %%
def f1_recall_precision(y_true, y_pred):
    return tuple(f(y_true, y_pred)
                 for f in [f1_score, recall_score, precision_score])


def train_and_evaluate(*, Xtr, ytr, Xte, yte, model_kwargs=dict()):
    model = BayesianDNN(**model_kwargs).to(device)
    nuts_kernel = NUTS(model, jit_compile=True)

    # Define MCMC sampler, get 50 posterior samples
    mcmc = MCMC(nuts_kernel, warmup_steps=50, num_samples=50)

    # Convert data to PyTorch tensors
    x_train = torch.from_numpy(Xtr).float().to(device)
    x_test = torch.from_numpy(Xte).float().to(device)
    y_train = torch.from_numpy(ytr).float().to(device)

    # Run MCMC
    mcmc.run(x_train, y_train)
    divergences = mcmc.diagnostics()['divergences']
    print(f'{divergences=}')

    # Evaluations.
    predictive = Predictive(model=model, posterior_samples=mcmc.get_samples())
    ytr_preds = predictive(x_train)['obs'].cpu().detach().numpy()
    yte_preds = predictive(x_test)['obs'].cpu().detach().numpy()

    ytr_pred = np.median(ytr_preds, axis=0).astype(np.int64)
    ytr_prob = calculate_optimal_prob_prediction(ytr_preds)
    (f1_train,
     recall_train,
     precision_train) = f1_recall_precision(ytr, ytr_pred)
    log_loss_train = balanced_log_loss(ytr, ytr_prob)
    print(f'Train - f1={f1_train:.4f} recall={recall_train:.4f} '
          f'precision={precision_train:.4f} log-loss={log_loss_train:.4f}')

    yte_pred = np.median(yte_preds, axis=0).astype(np.int64)
    yte_prob = calculate_optimal_prob_prediction(yte_preds)
    (f1_test,
     recall_test,
     precision_test) = f1_recall_precision(yte, yte_pred)
    log_loss_test = balanced_log_loss(yte, yte_prob)
    print(f'Test  - f1={f1_test:.4f} recall={recall_test:.4f} '
          f'precision={precision_test:.4f} log-loss={log_loss_test:.4f}')

    # Return results.
    return dict(
        f1_train=f1_train,
        f1_test=f1_test,
        log_loss_train=log_loss_train,
        log_loss_test=log_loss_test,
    )


def cross_validations(X, y, n_folds: int = 10, model_kwargs=dict()):
    results = []

    kfolds = StratifiedKFold(n_splits=n_folds)
    for i, (train_idx, test_idx) in enumerate(kfolds.split(X, y)):
        print(f'\n-- Fold # {i + 1}/{n_folds}:')

        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        result = train_and_evaluate(
            Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, model_kwargs=model_kwargs)
        results.append(result)

    return pd.DataFrame(results)


if run_cv:
    cv_results = cross_validations(
        X_df.values, y, n_folds=10,
        model_kwargs=dict(
            in_dim=55,
            hidden_dim=25,
            prior_scale=torch.tensor(10., device=device)))

# %%
if run_cv:
    print(cv_results.describe())
