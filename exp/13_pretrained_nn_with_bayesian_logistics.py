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
from itables import init_notebook_mode
import jax
from jax import random
import jax.numpy as jnp
from jax.scipy.special import expit
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.initialization import init_to_median
import numpyro.distributions as dist
import pickle
import pandas as pd
from scipy.stats import bernoulli
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from typing import Literal

init_notebook_mode(all_interactive=True)

# Still using cpu for numpyro,
# not sure because my installation sucks, but using gpu is slower than cpu.
numpyro.set_platform("cpu")
numpyro.set_host_device_count(16)

# %%
kaggle_submission = False
run_cv = True

# %% [markdown]
# # Pretrained Neural Networks with Bayesian Logistic Regression
#
# In this experiment,
# I'll use a pretrained neural network (with features preprocessing, removal),
# and used the encoded features to train bayesian logistic model.
# The result will be compared with the trained neural network's results
# to see if bayesian can improve our score.
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

    return df[features].pow(2.).rename(columns={
        f: f'{f}^2' for f in features
    })


# %%
Xinteractions_df = create_interaction_terms_between(X_df, X_df.columns)
X2_df = create_quadratic_terms(X_df, X_df.columns)

Xtrain_df = pd.concat([X_df, Xinteractions_df, X2_df], axis=1)


# %% [markdown]
# ## Models
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


# %% [markdown]
# ### Bayesian Logistic Model

# %%
class BayesianLogisticRegression:
    def __init__(self):
        self._mcmc = None

    def fit(self, *,
            X: np.ndarray,
            y: np.ndarray,
            num_warmup: int = 1000,
            num_samples: int = 20000,
            num_chains: int = 1):
        cls = BayesianLogisticRegression
        kernel = NUTS(cls.bayesian_logistic_regression,
                      init_strategy=init_to_median)  # pyright: ignore
        mcmc = MCMC(kernel,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains)
        mcmc.run(
            random.PRNGKey(0),
            X=jnp.array(X),
            y=jnp.array(y),
        )

        self._mcmc = mcmc
        return self

    def predict(self, *, X, return_median: bool = True):
        assert self._mcmc is not None,\
            'Please call `fit` with training data first!'

        cls = BayesianLogisticRegression
        predictive = Predictive(cls.bayesian_logistic_regression,
                                self._mcmc.get_samples(),
                                return_sites=['y'])
        predictions = predictive(
            random.PRNGKey(1),
            X=jnp.array(X),
        )

        y = predictions['y']
        return jnp.median(y, axis=0) if return_median else y

    @staticmethod
    def bayesian_logistic_regression(*,
                                     X: jax.Array,
                                     # group: jax.Array,
                                     # nb_groups: int,
                                     y: jax.Array | None = None):
        nb_obs, nb_features = X.shape
        # assert nb_obs == group.shape[0]

        # Prior for baselines.
        a0 = numpyro.sample('_a0', dist.Normal(0., 1.))

        # Prior for the coefficients of metric features.
        a_sigma = numpyro.sample('_aSigma', dist.Gamma(1., 1.))
        a = numpyro.sample(
            '_a', dist.StudentT(1., 0., a_sigma).expand((nb_features, )))

        # Prior for the group feature.
        # aG = numpyro.sample('_aG', dist.Normal(0., 1.).expand((nb_groups, )))

        # Prior for guess term.
        guess = numpyro.sample('guess', dist.Beta(1., 1.))

        # Observations.
        with numpyro.plate('obs', nb_obs) as idx:
            # prob = numpyro.deterministic(
            #     'prob', expit(a0 + jnp.dot(X[idx], a) + aG[group[idx]]))
            prob = numpyro.deterministic(
                'prob', expit(a0 + jnp.dot(X[idx], a)))
            guess_prob = numpyro.deterministic(
                'prob_w_guess', guess * 0.5 + (1 - guess) * prob)
            if y is not None:
                numpyro.sample('y', dist.Bernoulli(guess_prob), obs=y[idx])
            else:
                numpyro.sample('y', dist.Bernoulli(guess_prob))


# %% [markdown]
# ### Training and Evaluation Loop

# %%
@dataclass
class TrainingResult:
    model: NNClassifier
    features: list[str]
    preprocessing: Pipeline
    performance: float


def calculate_optimal_prob_prediction(y_preds: np.ndarray):
    """
    Calculate the best probability prediction based on the formula
    (in experiment 04_logistics_bayesian_v4*).

    y_preds: numpy array of shape (nb_draws, nb_data_points).
    """
    assert y_preds.ndim == 2, "Only accept 2d numpy array as input."
    _, nb_data = y_preds.shape

    # Calculate number of classes for each draw.
    nb_class_0 = np.sum(1 - y_preds, axis=1)
    nb_class_1 = np.sum(y_preds, axis=1)

    best_probs = []
    eps = 1e-15
    for j in range(nb_data):
        cj = np.sum(y_preds[:, j] / (nb_class_1 + eps))
        cj_1 = np.sum((1 - y_preds[:, j]) / (nb_class_0 + eps))

        prob = cj / (cj + cj_1)
        best_probs.append(prob)

    return np.asarray(best_probs)


def cross_validation(*,
                     X: pd.DataFrame, g: pd.Series, y: np.ndarray,
                     pretrained_models: list[TrainingResult],
                     device: str):
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

    def f1_recall_precision(y_true, y_pred):
        return tuple(f(y_true, y_pred)
                     for f in [f1_score, recall_score, precision_score])

    def balanced_log_loss(y_true, pred_prob):
        nb_class_0 = np.sum(1 - y_true)
        nb_class_1 = np.sum(y_true)

        prob_0 = np.clip(1. - pred_prob, 1e-10, 1. - 1e-10)
        prob_1 = np.clip(pred_prob, 1e-10, 1. - 1e-10)
        return (-np.sum((1 - y_true) * np.log(prob_0)) / nb_class_0
                - np.sum(y_true * np.log(prob_1)) / nb_class_1) / 2.

    def obtain_nn_perf_results(model: NNClassifier,
                               Xembed: np.ndarray, y: np.ndarray,
                               dstype: Literal['train', 'test'],
                               samples_for_opt_prob_est: int = 20000):
        # Obtaining predictions.
        pred_prob = (
            model(torch.tensor(Xembed, dtype=torch.float32).to(device),
                  mode='output')
            .cpu().detach().numpy().squeeze())
        ypred = np.where(pred_prob > 0.5, 1., 0.)

        # Obtaining optimal probability prediction.
        ysamples = bernoulli.rvs(
            pred_prob[:, None],
            size=(pred_prob.shape[0], samples_for_opt_prob_est))
        opt_prob_est = calculate_optimal_prob_prediction(ysamples.T)

        # Calculate performance results.
        (f1, recall, precision) = f1_recall_precision(y, ypred)
        log_loss = balanced_log_loss(y, pred_prob)
        opt_log_loss = balanced_log_loss(y, opt_prob_est)
        return {
            f'nn_f1_{dstype}': f1,
            f'nn_recall_{dstype}': recall,
            f'nn_precision_{dstype}': precision,
            f'nn_log_loss_{dstype}': log_loss,
            f'nn_opt_log_loss_{dstype}': opt_log_loss,
        }

    def obtain_bayes_perf_results(model: BayesianLogisticRegression,
                                  Xembed: np.ndarray, y: np.ndarray,
                                  dstype: Literal['train', 'test']):
        pred_samples = model.predict(X=Xembed, return_median=False)
        ypred = np.median(pred_samples, axis=0)
        prob_est = calculate_optimal_prob_prediction(pred_samples)

        (f1, recall, precision) = f1_recall_precision(y, ypred)
        log_loss = balanced_log_loss(y, prob_est)
        return {
            f'bayes_f1_{dstype}': f1,
            f'bayes_recall_{dstype}': recall,
            f'bayes_precision_{dstype}': precision,
            f'bayes_log_loss_{dstype}': log_loss,
        }

    cv_results = []
    kfolds = StratifiedKFold(n_splits=len(pretrained_models))
    for fold, (train_idx, test_idx) in enumerate(kfolds.split(X, y)):
        print(f'\nFold #{fold + 1}:')
        # Obtain the corresponding pretrained model.
        pretrained_model = pretrained_models[fold]

        # Split training and testing data.
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        gtr, gte = g.iloc[train_idx], g.iloc[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        # Process the data.
        Xtr_embed = data_embedding(Xtr, gtr, pretrained_model)
        Xte_embed = data_embedding(Xte, gte, pretrained_model)

        # Train the bayesian model.
        logistic = (BayesianLogisticRegression()
                    .fit(X=Xtr_embed,
                         y=ytr,
                         num_chains=4,
                         num_samples=5000))

        # Compare the results between models.
        cv_results.append({
            'fold': fold + 1,
            **obtain_nn_perf_results(
                pretrained_model.model, Xtr_embed, ytr, 'train'),
            **obtain_bayes_perf_results(logistic, Xtr_embed, ytr, 'train'),
            **obtain_nn_perf_results(
                pretrained_model.model, Xte_embed, yte, 'test'),
            **obtain_bayes_perf_results(logistic, Xte_embed, yte, 'test'),
        })

    return pd.DataFrame(cv_results)


# %% [markdown]
# ### Load pretrained models and train Bayesian model
# %%
pretrained_model_path = './20230729_09_48_11.1e.model'
with open(pretrained_model_path, 'rb') as inmodel:
    pretrained_models = pickle.load(inmodel)
    print(pretrained_models[0])

cv_results = cross_validation(
    X=Xtrain_df, g=ej, y=y,
    pretrained_models=pretrained_models,
    device='cuda' if torch.cuda.is_available() else 'cpu')

# %%
interested_columns = [col for col in cv_results.columns
                      if 'f1' in col or 'log_loss' in col]
interested_columns = ['fold'] + interested_columns
cv_results[interested_columns]

# %%
cv_results[interested_columns].describe()
