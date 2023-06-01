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
import jax.numpy as jnp
from jax.scipy.optimize import minimize as jax_minimize
from jax.scipy.special import expit, logit
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.initialization import init_to_median
import numpyro.distributions as dist
import pandas as pd
from scipy.optimize import minimize as scipy_minimize
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
# With guessing, and shrinkage on metric variables' coefficients.
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


kernel = NUTS(robust_logistic_regression, init_strategy=init_to_median)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=2500, num_chains=4)
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
def balanced_log_loss_jax(y_true):
    y_true = jnp.asarray(y_true)

    def log_loss_jax(logit):
        prob = expit(logit)
        nb_class_1 = jnp.sum(y_true, axis=1) + 1e-10
        nb_class_0 = jnp.sum(1 - y_true, axis=1) + 1e-10
        prob_0 = jnp.clip(1. - prob, 1e-8, 1. - 1e-8)[None, ...]
        prob_1 = jnp.clip(prob, 1e-8, 1. - 1e-8)[None, ...]
        return jnp.mean((-jnp.sum((1 - y_true) * jnp.log(prob_0), axis=1) / nb_class_0 -
                         jnp.sum(y_true * jnp.log(prob_1), axis=1) / nb_class_1) / 2.)

    return log_loss_jax


results = jax_minimize(
        balanced_log_loss_jax(predictions['y']),
        logit(y_prob),
        method='BFGS',
        options=dict(maxiter=10000))


# %%
print(f'{results.success=}, {results.fun=}, {results.status=}')

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

# %%
# Instead of using median as our predictions,
# we're going to optimize the prediction probability
# such that it minimizes the balanced log loss.
# y_probs = predictions['prob']
# y_prob = np.asarray(jnp.median(y_probs, axis=0))
# results = jax_minimize(
#         balanced_log_loss_jax(predictions['y']),
#         logit(jnp.median(predictions['prob'], axis=0)),
#         method='BFGS',
#         options=dict(maxiter=100000))
results = scipy_minimize(
        balanced_log_loss_jax(predictions['y']),
        logit(jnp.median(predictions['prob'], axis=0)),
        method='Powell',
        options=dict(disp=True))
print(f'{results.success=}, {results.fun=}, {results.status=}')

# Optimimal y_prob.
y_prob_optim = results.x

# %%
# Create .csv submission file.
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'class_0': 1. - y_prob_optim,
    'class_1': y_prob_optim,
})
submission.to_csv('submission.csv', index=False)
