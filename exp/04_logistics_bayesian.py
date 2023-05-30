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
from jax.scipy.special import expit
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# %%
kaggle_submission = False

# %% [markdown]
# # Bayesian Logistics Regression
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
def logistic_regression_model(*, X: jnp.ndarray, group: jnp.ndarray, nb_groups: int, y: jnp.ndarray | None = None):
    nb_obs, nb_features = X.shape
    assert nb_obs == group.shape[0]

    # Scale the metric features to zero mean and one standard deviation.
    mean_X = jnp.mean(X, axis=0)
    std_X = jnp.std(X, axis=0)
    z_X = (X - mean_X) / std_X

    # Prior for baselines.
    a0 = numpyro.sample('_a0', dist.Normal(0., 1.))

    # Prior for the coefficients of metric features.
    a = numpyro.sample('_a', dist.Normal(0., 1.).expand((nb_features, )))

    # Prior for the group feature.
    aG = numpyro.sample('_aG', dist.Normal(0., 1.).expand((nb_groups, )))

    # Observations.
    with numpyro.plate('obs', nb_obs) as idx:
        logit = numpyro.deterministic('logit', a0 + jnp.dot(z_X[idx], a) + aG[group[idx]])
        if y is not None:
            numpyro.sample('y', dist.BernoulliLogits(logit), obs=y[idx])
        else:
            numpyro.sample('y', dist.BernoulliLogits(logit))


kernel = NUTS(logistic_regression_model)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=5000)
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
predictive = Predictive(logistic_regression_model,
                        mcmc.get_samples(),
                        return_sites=['y', 'logit'])
predictions = predictive(
    random.PRNGKey(1),
    X=jnp.array(X),
    nb_groups=ej.cat.categories.size,
    group=jnp.array(ej.cat.codes.values)
)

# %%
y_probs = expit(predictions['logit'])
y_probs

