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
from jax import random
import jax.numpy as jnp
from jax.scipy.special import expit
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Using CPU is a lot faster than GPU.
numpyro.set_platform('cpu')
numpyro.set_host_device_count(16)

# %%
kaggle_submission = False

# %% [markdown]
# # Bayesian Logistics Regression
#
# In this notebook,
# I'll use WoE/IV as features selection.
# The implementation of WoE/IV is copied & modified from
# https://www.kaggle.com/code/levdvernik/logisticregression-icr
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
# ### Features Selection using WoE/IV

# %%
# https://lucastiagooliveira.github.io/datascience/iv/woe/python/2020/12/15/iv_woe.html
def iv_woe(data: pd.DataFrame, target: pd.Series, bins=10, show_woe=False):
    # Empty Dataframe
    newDF, woeDF = pd.DataFrame(), pd.DataFrame()

    # Extract Column Names
    cols = data.columns

    # Run WOE and IV on all the independent variables
    for ivars in cols:
        if ((data[ivars].dtype.kind in 'bifc')
                and (len(np.unique(data[ivars])) > 10)):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': target})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': target})

        # Calculate the number of events in each group (bin)
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']

        # Calculate % of events in each group.
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()

        # Calculate the non events in each group.
        d['Non-Events'] = d['N'] - d['Events']
        # Calculate % of non events in each group.
        d['% of Non-Events'] = np.maximum(
            d['Non-Events'], 0.5) / d['Non-Events'].sum()

        # Calculate WOE by taking natural log of
        # division of % of non-events and % of events
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        temp = pd.DataFrame(
            {"Variable": [ivars], "IV": [d['IV'].sum()]},
            columns=["Variable", "IV"])
        newDF = pd.concat([newDF, temp], axis=0)
        woeDF = pd.concat([woeDF, d], axis=0)

        # Show WOE Table
        if show_woe:
            print(d)

    return newDF, woeDF


iv, woe = iv_woe(X_df, y, bins=15)
iv = iv.set_index("Variable")


# %% [markdown]
# For the feature selection, we will only use features having
# IV value within the recommended range: 0.3 - 0.5.
#
# ### Bayesian Logistic Regression Model

# %%
class BayesianLogisticRegression:
    def fit(self, *, X: np.ndarray, group: np.ndarray, y: np.ndarray,
            num_warmup: int = 1000,
            num_samples: int = 20000,
            num_chains: int = 4,
            print_summary: bool = False):
        kernel = NUTS(self._robust_logistic_regression_model)
        mcmc = MCMC(kernel, num_warmup=num_warmup,
                    num_samples=num_samples, num_chains=num_chains)

        nb_groups = len(np.unique(group))
        mcmc.run(
            random.PRNGKey(0),
            X=jnp.array(X),
            y=jnp.array(y),
            group=jnp.array(group),
            nb_groups=nb_groups,
        )

        self._nb_groups = nb_groups
        self._mcmc = mcmc

        if print_summary:
            mcmc.print_summary()

        return self

    def predict(self, *, X: np.ndarray, group: np.ndarray):
        assert self._mcmc is not None and self._nb_groups is not None

        predictive = Predictive(
            self._robust_logistic_regression_model,
            self._mcmc.get_samples(),
            return_sites=['y', 'prob', 'prob_w_guess'])
        predictions = predictive(
            random.PRNGKey(1),
            X=jnp.array(X),
            nb_groups=self._nb_groups,
            group=jnp.array(group),
        )

        return predictions['y']

    @staticmethod
    def _robust_logistic_regression_model(
            *, X: jnp.ndarray, group: jnp.ndarray,
            nb_groups: int, y: jnp.ndarray | None = None):
        nb_obs, nb_features = X.shape
        assert nb_obs == group.shape[0]

        # Prior for baselines.
        a0 = numpyro.sample('_a0', dist.Normal(0., 1.))

        # Prior for the coefficients of metric features.
        a_sigma = numpyro.sample('_aSigma', dist.Gamma(1., 1.))
        a = numpyro.sample(
            '_a', dist.StudentT(1., 0., a_sigma).expand((nb_features, )))

        # Prior for the group feature.
        aG = numpyro.sample('_aG', dist.Normal(0., 1.).expand((nb_groups, )))

        # Prior for guess term.
        guess = numpyro.sample('guess', dist.Beta(1., 1.))

        # Observations.
        with numpyro.plate('obs', nb_obs) as idx:
            prob = numpyro.deterministic(
                'prob', expit(a0 + jnp.dot(X[idx], a) + aG[group[idx]]))
            guess_prob = numpyro.deterministic(
                'prob_w_guess', guess * 0.5 + (1 - guess) * prob)
            if y is not None:
                numpyro.sample('y', dist.Bernoulli(guess_prob), obs=y[idx])
            else:
                numpyro.sample('y', dist.Bernoulli(guess_prob))

# %% [markdown]
# ### Cross-Validation
#
# We will perform standard cross-validation with 10 folds
# to see how the model performs.

# %%
# Some useful utilities.


def balanced_log_loss(y_true, pred_prob):
    nb_class_0 = np.sum(1 - y_true)
    nb_class_1 = np.sum(y_true)

    prob_0 = np.clip(1. - pred_prob, 1e-10, 1. - 1e-10)
    prob_1 = np.clip(pred_prob, 1e-10, 1. - 1e-10)
    return (-np.sum((1 - y_true) * np.log(prob_0)) / nb_class_0
            - np.sum(y_true * np.log(prob_1)) / nb_class_1) / 2.


def f1_recall_precision(y_true, y_pred):
    return tuple(f(y_true, y_pred)
                 for f in [f1_score, recall_score, precision_score])


def calculate_best_prob_prediction(y_preds: np.ndarray):
    """
    Calculate the best probability prediction based on the above formula.

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


def print_metrics(of: str, *,
                  f1: float, recall: float, precision: float, log_loss: float):
    print(f'{of} - f1={f1:.4f} recall={recall:.4f} '
          f'precision={precision:.4f} log-loss={log_loss:.4f}')


# %%
kfold = StratifiedKFold(n_splits=10)

train_log_losses = []
train_f1_scores = []
val_log_losses = []
val_f1_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_df, y)):
    Xtr_df, ytr = X_df.iloc[train_idx], y[train_idx]
    Xva_df, yva = X_df.iloc[val_idx], y[val_idx]

    # Perform features selection based on WoE/IV:
    # choose the features within 0.3 - 0.5 range.
    iv, _ = iv_woe(Xtr_df, ytr, bins=15)
    iv = iv.set_index("Variable")
    imp_iv_features = iv[(iv['IV'] >= 0.3) & (iv['IV'] <= 0.5)].index

    Xtr_df = Xtr_df[imp_iv_features]
    Xva_df = Xva_df[imp_iv_features]

    # Use these features to train our model.
    model = (BayesianLogisticRegression()
             .fit(X=Xtr_df.values, group=ej[train_idx].cat.codes.values, y=ytr))

    # Perform predictions on on the validation data.
    tr_predictions = model.predict(
        X=Xtr_df.values, group=ej[train_idx].cat.codes.values)
    va_predictions = model.predict(
        X=Xva_df.values, group=ej[val_idx].cat.codes.values)

    # Use median as our point prediciton.
    ytr_pred = np.median(tr_predictions, axis=0)
    yva_pred = np.median(va_predictions, axis=0)

    # Use our approach to calculate the best probability predictions.
    probtr_pred = calculate_best_prob_prediction(tr_predictions)
    probva_pred = calculate_best_prob_prediction(va_predictions)

    # Calculate metrics.
    f1_tr, recall_tr, precs_tr = f1_recall_precision(ytr, ytr_pred)
    log_loss_tr = balanced_log_loss(ytr, probtr_pred)
    print_metrics('Train', f1=f1_tr, recall=recall_tr,
                  precision=precs_tr, log_loss=log_loss_tr)

    f1_va, recall_va, precs_va = f1_recall_precision(yva, yva_pred)
    log_loss_va = balanced_log_loss(yva, probva_pred)
    print_metrics('Validation', f1=f1_va, recall=recall_va,
                  precision=precs_va, log_loss=log_loss_va)

    # Store the metrics.
    store_metrics = [(train_log_losses, log_loss_tr),
                     (train_f1_scores, f1_tr),
                     (val_log_losses, log_loss_va),
                     (val_f1_scores, f1_va)]
    for store, metric in store_metrics:
        store.append(metric)


# %%
print(f'Train - F1={np.mean(train_f1_scores):.4f} +- {np.std(train_f1_scores):.4f}'
      f'; Log Loss = {np.mean(train_log_losses):.4f} +- {np.std(train_log_losses):.4f}')
print(f'Valid - F1={np.mean(val_f1_scores):.4f} +- {np.std(val_f1_scores):.4f}'
      f'; Log Loss = {np.mean(val_log_losses):.4f} +- {np.std(val_log_losses):.4f}')

# %% [markdown]
# Previous run, using normal logistic regression model:
#
# * Train - F1=0.4279 +- 0.0376; Log Loss = 0.4981 +- 0.0234
# * Valid - F1=0.4825 +- 0.1822; Log Loss = 0.5059 +- 0.0742
#
# With the robust regression, we see improvements in the scores:
#
# * Train - F1=0.4860 +- 0.0433; Log Loss = 0.4941 +- 0.0269
# * Valid - F1=0.4993 +- 0.1230; Log Loss = 0.4866 +- 0.0522
