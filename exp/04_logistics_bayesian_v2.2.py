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
from jax import random
from jax import jit, value_and_grad
from jax.scipy.special import expit
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


numpyro.set_host_device_count(4)

# %%
kaggle_submission = False

# %% [markdown]
# # Bayesian Logistics Regression
#
# With a random probability of flipping a coin.
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
X = train_df.drop(columns=['Id', 'Class', 'EJ'])
y = train_df['Class']
ej = train_df['EJ'].astype('category')

X = preprocessing.fit_transform(X)
y = y.values

# %% [markdown]
# ## Model

# %%
def robust_logistic_regression(*,
                               X: jnp.ndarray,
                               group: jnp.ndarray,
                               nb_groups: int,
                               y: jnp.ndarray | None = None):
    nb_obs, nb_features = X.shape
    assert nb_obs == group.shape[0]

    # Prior for baselines.
    a0 = numpyro.sample('_a0', dist.Normal(0., 1.))

    # Prior for the coefficients of metric features.
    a = numpyro.sample('_a', dist.Normal(0., 1.).expand((nb_features, )))

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


kernel = NUTS(robust_logistic_regression)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=5000, num_chains=4)
mcmc.run(
    random.PRNGKey(0),
    X=jnp.array(X),
    y=jnp.array(y),
    nb_groups=ej.cat.categories.size,
    group=jnp.array(ej.cat.codes.values)
)
mcmc.print_summary()

# %%
# Make the prediction on the traininig data.
predictive = Predictive(robust_logistic_regression,
                        mcmc.get_samples(),
                        num_samples=10000,
                        return_sites=['y', 'prob', 'prob_w_guess'])
predictions = predictive(
    random.PRNGKey(1),
    X=jnp.array(X),
    nb_groups=ej.cat.categories.size,
    group=jnp.array(ej.cat.codes.values)
)

# %%
y_probs = predictions['prob']
y_prob = jnp.median(y_probs, axis=0)
y_pred = np.asarray(jnp.where(y_prob > 0.5, 1, 0))

# Compute the scores.
f1 = f1_score(y, y_pred)
recall = recall_score(y, y_pred)
precision = precision_score(y, y_pred)
print(f'{f1=}, {recall=}, {precision=}')

# %%
predictions['prob'].shape


# %%
def balanced_log_loss(y_true, y_pred_prob):
    """
    y_true of shape (nb_draws, nb_data).
    y_pred_prob of shape (nb_data,)
    """
    nb_class_1 = np.sum(1. * y_true, axis=1)
    nb_class_0 = np.sum(1. - y_true, axis=1)

    prob_0 = np.clip(1. - y_pred_prob, 1e-8, 1. - 1e-8)[None, ...]
    prob_1 = np.clip(y_pred_prob, 1e-8, 1. - 1e-8)[None, ...]

    return np.mean((-np.sum((1 - y_true) * np.log(prob_0), axis=1) / nb_class_0 -
                    np.sum(y_true * np.log(prob_1), axis=1) / nb_class_1) / 2.)


def balanced_log_loss_jax(y_true):
    y_true = jnp.asarray(y_true)

    def log_loss_jax(prob):
        nb_class_1 = jnp.sum(y_true, axis=1)
        nb_class_0 = jnp.sum(1 - y_true, axis=1)
        prob_0 = jnp.clip(1. - prob, 1e-8, 1. - 1e-8)[None, ...]
        prob_1 = jnp.clip(prob, 1e-8, 1. - 1e-8)[None, ...]
        return jnp.mean((-jnp.sum((1 - y_true) * jnp.log(prob_0), axis=1) / nb_class_0 -
                         jnp.sum(y_true * jnp.log(prob_1), axis=1) / nb_class_1) / 2.)

    return jit(value_and_grad(log_loss_jax))


def callbackF(Xi):
    global Nfeval
    loss = balanced_log_loss(predictions['y'], Xi)
    print(f'{Nfeval} - Objective: {loss:.4f}')
    Nfeval += 1


Nfeval = 1
# results = minimize(lambda x: balanced_log_loss(predictions['y'], x),
#                    y_prob,
#                    method='Powell',
#                    bounds=[(0., 1.) for _ in y_prob],
#                    callback=callbackF,
#                    options={'disp': True})


# %%
# optim_x = results['x']
# Instead of using minimization results,
# we're going to count the number of predicted.
optim_x = np.sum(predictions['y'], axis=0) / predictions['y'].shape[0]

# %%
balanced_log_loss(predictions['y'], optim_x)

# %%
balanced_log_loss(predictions['y'], y_prob)

# %%
balanced_log_loss(y[None, ...], y_prob)

# %%
balanced_log_loss(y[None, ...], optim_x)

# %% [markdown]
# ## Submission

# %%
# Preprocess test data.
X_test = test_df.drop(columns=['Id', 'EJ'])
ej_test = test_df['EJ'].astype(ej.dtype)

X_test = preprocessing.transform(X_test)

# %%
# Make predictions.
predictions = predictive(
    random.PRNGKey(1),
    X=jnp.array(X_test),
    nb_groups=ej.cat.categories.size,
    group=jnp.array(ej_test.cat.codes.values)
)

y_probs = predictions['prob']
y_prob = np.asarray(jnp.median(y_probs, axis=0))

# %%
# Create .csv submission file.
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'class_0': 1. - y_prob,
    'class_1': y_prob,
})
submission.to_csv('submission.csv', index=False)
