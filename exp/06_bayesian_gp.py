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
import jax
from jax import random
from jax.scipy.special import expit, logit
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.initialization import init_to_median
import numpyro.distributions as dist
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


numpyro.set_host_device_count(4)

# %%
kaggle_submission = False

# %% [markdown]
# # Bayesian Gaussian Process
#
# In this experiment,
# I'll put gaussian process prior on the metric variables.
# The logit of the class_1 will linearly depend on the gaussian process
# and the categorical variable.
#
# The model is similar to what is discussed in
# [Bayesian kernel machine regression](https://pubmed.ncbi.nlm.nih.gov/25532525/).
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

# %%
Xtest_df = test_df.drop(columns=['Id', 'EJ'])
ej_test = test_df['EJ'].astype(ej.dtype)

Xtest_df = pd.DataFrame(
    preprocessing.transform(Xtest_df),
    columns=Xtest_df.columns,
    index=Xtest_df.index)

# %% [markdown]
# ### Gaussian Process Model

# %%
def rbf_kernel(X: jax.Array, *, var, length, eps: float = 1e-9):
    """
    Calculate RBF kernel.

    X: jax.Array
        A 2d array of shape (N, nb_features).
    var: variance.
    length: lenght scale.

    Returns
        A 2d array of shape (N, N).
    """
    squared_dist = jnp.sum(
            (jnp.expand_dims(X, axis=1) - X[None, ...])**2, axis=-1) # pyright: ignore
    k = (var**2) * jnp.exp(-0.5 * squared_dist / (length**2))

    # Make sure that the covariance matrix is positive definite.
    N = X.shape[0]
    k = k + jnp.identity(N, dtype=k.dtype) * eps

    return k


def logistic_regression_with_gp_prior(
        *,
        Xtr: jax.Array,
        y: jax.Array,
        gtr: jax.Array,
        nb_groups: int,
        Xte: jax.Array | None = None,
        gte: jax.Array | None = None):
    """
    Perform logistic regression with Gaussian process prior:
        logit(y) = gp(X_) + g*b + b0

    Xtr: jax.Array
        Observed data points of shape (nb_obs, nb_features).
    y: jax.Array
        A 1d array storing the observed class.
    gtr: jax.Array.
        A 1d array storing which group does a data point belongs to.
    nb_groups: int
        How many groups there are in `g`.
    Xte: jax.Array
        Observed data points that we want to do inference on,
        an array of shape (nb_obs, nb_features).
    gte: jax.Array
        A 1d array storing which group does a test data point belongs to.
    """
    if Xte is not None:
        assert Xte.shape[1] == Xtr.shape[1], 'Number of metric variables must match.'
        assert gte is not None
        nb_obs = Xtr.shape[0] + Xte.shape[0]

        X = jnp.concatenate([Xtr, Xte], axis=0)
        g = jnp.concatenate([gtr, gte], axis=0)
    else:
        nb_obs = Xtr.shape[0]

        X = Xtr
        g = gtr

    assert X.shape[0] == y.shape[0]
    assert X.shape[0] == g.shape[0]

    # Prior for baselines.
    a0 = numpyro.sample('_a0', dist.Normal(0., 10.))

    # Set priors on kernel's parameters.
    var = numpyro.sample('var', dist.LogNormal(0., 10.))
    length = numpyro.sample('length', dist.LogNormal(0., 10.))

    # Calculate kernel values.
    k = rbf_kernel(X, var=var, length=length)
    Lk = jnp.linalg.cholesky(k)

    # Sample from the covariance matrix.
    # h = numpyro.sample(
    #         '_h',
    #         dist.MultivariateNormal(loc=jnp.zeros(nb_obs), # pyright: ignore
    #                                 covariance_matrix=jnp.identity(nb_obs,
    #                                                                dtype=k.dtype)))
    h = numpyro.sample('_h', dist.Normal(jnp.zeros(nb_obs), 1)) # pyright: ignore
    print('Before multiplication', h.shape)
    h = Lk @ h
    print(h.shape)

    # Prior for the group feature.
    aG = numpyro.sample('aG', dist.Normal(0., 10.).expand((nb_groups, )))

    # Prior for guess term.
    guess = numpyro.sample('guess', dist.Beta(1., 1.))

    prob = numpyro.deterministic('prob', expit(a0 + aG[g] + h)) # pyright: ignore
    guess_prob = numpyro.deterministic(
            'prob_w_guess', guess * 0.5 + (1 - guess) * prob) # pyright: ignore

    # Predictions
    yTe_mask = jnp.isnan(y)
    yTe_idx = jnp.nonzero(yTe_mask)
    yTe = numpyro.sample('yTe', dist.Bernoulli(guess_prob[yTe_mask]).mask(False))
    print(y.shape, yTe.shape) # pyright: ignore
    y = y.at[yTe_idx].set(yTe)

    # Observations.
    numpyro.sample('y', dist.Bernoulli(guess_prob), obs=y)


kernel = NUTS(logistic_regression_with_gp_prior,
              init_strategy=init_to_median) # pyright: ignore
mcmc = MCMC(kernel, num_warmup=100, num_samples=100, num_chains=1)
yTe = np.empty(len(Xtest_df))
yTe[:] = np.nan
mcmc.run(
    random.PRNGKey(0),
    Xtr=jnp.array(X_df.values),
    Xte=jnp.array(Xtest_df.values),
    y=jnp.asarray(np.concatenate([y, yTe])),
    nb_groups=ej.cat.categories.size,
    gtr=jnp.array(ej.cat.codes.values),
    gte=jnp.array(ej_test.cat.codes.values),
)
mcmc.print_summary()

# %%
# Make the prediction on the traininig data.
# samples = mcmc.get_samples()
# predictive = Predictive(logistic_regression_with_gp_prior,
#                         {k: samples[k] for k in samples.keys() if k != '_h'},
#                         return_sites=['yTe', 'prob', 'prob_w_guess'])
# predictions = predictive(
#     random.PRNGKey(1),
#     Xtr=jnp.array(X_df.values),
#     y=jnp.array(y),
#     # Xte=jnp.array(X_df.values),
#     nb_groups=ej.cat.categories.size,
#     gtr=jnp.array(ej.cat.codes.values),
#     # gte=jnp.array(ej.cat.codes.values),
# )

# %%
# y_probs = predictions['prob'][617:]
# y_prob = jnp.median(y_probs, axis=0)
# y_pred = np.asarray(jnp.where(y_prob > 0.5, 1, 0))

# # Compute the scores.
# f1 = f1_score(y, y_pred)
# recall = recall_score(y, y_pred)
# precision = precision_score(y, y_pred)
# print(f'{f1=}, {recall=}, {precision=}')
