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


from jax import random
from jax import vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median
import numpyro.distributions as dist


# We want to use CPU instead of GPU.
# And CPU tends to be faster than GPU, for this task.
numpyro.set_platform('cpu')
numpyro.set_host_device_count(16)


# %% [markdown]
# # Test Gaussian Process
# ## Numpyro's Example
# Code from [Numpyro's Gaussian Process example](https://num.pyro.ai/en/stable/examples/gp.html).

# %%
def rbf_kernel(X, Z, var, length, noise, jitter=1.0e-6, include_noise=True):
    deltaXsq = jnp.power((X[:, None] - Z) / length, 2.0)
    k = var * jnp.exp(-0.5 * deltaXsq)
    if include_noise:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k


def model(X, Y):
    # set uninformative log-normal priors on our three kernel hyperparameters
    var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 10.0))
    noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0))
    length = numpyro.sample("kernel_length", dist.LogNormal(0.0, 10.0))

    # compute kernel
    k = rbf_kernel(X, X, var, length, noise)

    # sample Y according to the standard gaussian process formula
    numpyro.sample(
        "Y",
        dist.MultivariateNormal(loc=jnp.zeros(
            X.shape[0]), covariance_matrix=k),
        obs=Y,
    )


def predict(rng_key, X, Y, X_test, var, length, noise):
    # compute kernels between train and test data, etc.
    k_pp = rbf_kernel(X_test, X_test, var, length, noise, include_noise=True)
    k_pX = rbf_kernel(X_test, X, var, length, noise, include_noise=False)
    k_XX = rbf_kernel(X, X, var, length, noise, include_noise=True)
    K_xx_inv = jnp.linalg.inv(k_XX)
    K = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
    sigma_noise = (jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0))
                   * random.normal(rng_key, X_test.shape[:1]))
    mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y))

    # we return both the mean function and a sample from the posterior
    # predictive for the given set of hyperparameters
    return mean, mean + sigma_noise


# create artificial regression dataset
def get_data(N=30, sigma_obs=0.15, N_test=400):
    np.random.seed(0)
    X = jnp.linspace(-1, 1, N)
    Y = X + 0.2 * jnp.power(X, 3.0) + 0.5 * \
        jnp.power(0.5 + X, 2.0) * jnp.sin(4.0 * X)
    Y += sigma_obs * np.random.randn(N)
    Y -= jnp.mean(Y)
    Y /= jnp.std(Y)

    assert X.shape == (N,)
    assert Y.shape == (N,)

    X_test = jnp.linspace(-1.3, 1.3, N_test)

    return X, Y, X_test


# %%
# Test the model.
num_data = 50
X, Y, X_test = get_data(N=num_data)

# Perform inference.
rng_key, rng_key_predict = random.split(random.PRNGKey(0))
kernel = NUTS(model, init_strategy=init_to_median)
mcmc = MCMC(
    kernel,
    num_warmup=1000,
    num_samples=2000,
    num_chains=1,
    thinning=1,
)
mcmc.run(rng_key, X, Y)
mcmc.print_summary()

# %%
# Perform prediction.
# do prediction
samples = mcmc.get_samples()
vmap_args = (
    random.split(rng_key_predict, samples["kernel_var"].shape[0]),
    samples["kernel_var"],
    samples["kernel_length"],
    samples["kernel_noise"],
)
means, predictions = vmap(
    lambda rng_key, var, length, noise: predict(
        rng_key, X, Y, X_test, var, length, noise
    )
)(*vmap_args)

mean_prediction = np.mean(means, axis=0)
percentiles = np.percentile(predictions, [5.0, 95.0], axis=0)

# Plot the results.
# make plots
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

# plot training data
ax.plot(X, Y, "kx")
# plot 90% confidence level of predictions
ax.fill_between(X_test, percentiles[0, :],
                percentiles[1, :], color="lightblue")
# plot mean prediction
ax.plot(X_test, mean_prediction, "blue", ls="solid", lw=2.0)
ax.set(xlabel="X", ylabel="Y", title="Mean predictions with 90% CI")


# %% [markdown]
# ## My Test

# %%
def my_model(X, Y, X_test):
    X_all = jnp.concatenate([X, X_test], axis=0)

    # set uninformative log-normal priors on our three kernel hyperparameters
    var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 10.0))
    noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0))
    length = numpyro.sample("kernel_length", dist.LogNormal(0.0, 10.0))

    # compute kernel
    k = rbf_kernel(X_all, X_all, var, length, noise)
    Lk = jnp.linalg.cholesky(k)

    # First failed attempt.
    # nb_X = X.shape[0]
    # Lk_X = Lk[:nb_X, :nb_X]

    # # sample Y according to the standard gaussian process formula
    # numpyro.sample(
    #     "Y",
    #     dist.MultivariateNormal(
    #         loc=jnp.zeros(nb_X), scale_tril=Lk_X),
    #     obs=Y)

    # # Predictions.
    # nb_X_test = X_test.shape[0]
    # Lk_X_test = Lk[nb_X:, nb_X:]
    # numpyro.sample(
    #     "Ypred",
    #     dist.MultivariateNormal(loc=jnp.zeros(nb_X_test),
    #                             scale_tril=Lk_X_test))

    # Second attempt - Bayesian Imputation: Success!
    nb_X_test = X_test.shape[0]
    Lk_X_test = Lk[-nb_X_test:, -nb_X_test:]
    Y_pred = numpyro.sample(
        "Ypred",
        dist.MultivariateNormal(loc=jnp.zeros(nb_X_test),
                                scale_tril=Lk_X_test).mask(False))

    Y_all = jnp.concatenate([Y, Y_pred])
    numpyro.sample(
        'Y_all',
        dist.MultivariateNormal(loc=jnp.zeros(X_all.shape[0]), scale_tril=Lk),
        obs=Y_all)


# do inference
rng_key, rng_key_predict = random.split(random.PRNGKey(0))
kernel = NUTS(my_model, init_strategy=init_to_median)
mcmc = MCMC(
    kernel,
    num_warmup=500,
    num_samples=500,
    num_chains=1,
    thinning=1,
)
mcmc.run(rng_key, X=X, Y=Y, X_test=X_test)
mcmc.print_summary()

# %%
# Now, we'll plot the results to see if it is similar
# to that of Numpyro.
samples = mcmc.get_samples()
y_pred_samples = samples['Ypred']
print(f'{y_pred_samples.shape=}')

mean_prediction = np.mean(y_pred_samples, axis=0)
percentiles = np.percentile(y_pred_samples, [5.0, 95.0], axis=0)

# Plot the results.
# make plots
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

# plot training data
ax.plot(X, Y, "kx")
# plot 90% confidence level of predictions
ax.fill_between(X_test, percentiles[0, :],
                percentiles[1, :], color="lightblue")
# plot mean prediction
ax.plot(X_test, mean_prediction, "blue", ls="solid", lw=2.0)
ax.set(xlabel="X", ylabel="Y", title="Mean predictions with 90% CI")
