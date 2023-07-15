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
from imblearn.over_sampling import RandomOverSampler
from jax import random
import jax.numpy as jnp
from jax.scipy.special import expit
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
# ### Bayesian Logistic Regression

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
    ros = RandomOverSampler(random_state=0)
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


def f1_recall_precision(y_true, y_pred):
    return tuple(f(y_true, y_pred)
                 for f in [f1_score, recall_score, precision_score])


def train_and_evaluate(*,
                       Xtr, gtr, ytr,
                       Xte, gte, yte,
                       num_samples: int = 20000,
                       num_warmup: int = 1000,
                       num_chains: int = 4):
    # First, we will normalize the data.
    Xtr = pd.DataFrame(
        preprocessing.fit_transform(Xtr, ytr),
        columns=Xtr.columns)
    Xte = pd.DataFrame(
        preprocessing.transform(Xte),
        columns=Xte.columns)

    # Next, we'll filter out correlated features.
    Xtr = filter_in_uncorrelated_features(Xtr, 0.7)
    Xte = Xte[Xtr.columns]

    # Next, we'll perform sampling.
    Xtr['ej'] = gtr
    # Xtr, ytr = sampling(Xtr, ytr)

    gtr = Xtr['ej'].values
    Xtr = Xtr.drop(columns=['ej'])

    # Then, we use tree-based model to select important features.
    # Xtr = select_important_features(
    #     Xtr, ytr,
    #     n_estimators=1000,
    #     important_thresholds='5*median')
    # Xte = Xte[Xtr.columns]
    print('Number of important features: ', len(Xtr.columns))

    # model = LogisticRegression(class_weight='balanced', max_iter=10000)
    model = BayesianLogisticRegression()
    model.fit(X=Xtr, group=gtr, y=ytr,
              num_warmup=num_warmup,
              num_samples=num_samples,
              num_chains=num_chains)

    ytr_preds = model.predict(X=Xtr, group=gtr)
    ytr_pred = np.median(ytr_preds, axis=0)
    ytr_prob = calculate_optimal_prob_prediction(ytr_preds)
    (f1_train,
     recall_train,
     precision_train) = f1_recall_precision(ytr, ytr_pred)
    log_loss_train = balanced_log_loss(ytr, ytr_prob)
    print(f'Train - f1={f1_train:.4f} recall={recall_train:.4f} '
          f'precision={precision_train:.4f} log-loss={log_loss_train:.4f}')

    yte_preds = model.predict(X=Xte, group=gte)
    yte_pred = np.median(yte_preds, axis=0)
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


def cross_validations(*, X, grp, y,
                      n_folds: int = 10,
                      **kwargs):
    results = []

    kfolds = StratifiedKFold(n_splits=n_folds)
    for i, (train_idx, test_idx) in enumerate(kfolds.split(X, y)):
        print(f'\n-- Fold # {i + 1}/{n_folds}:')

        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        gtr, gte = grp[train_idx], grp[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        result = train_and_evaluate(
            Xtr=Xtr, gtr=gtr, ytr=ytr,
            Xte=Xte, gte=gte, yte=yte,
            **kwargs)
        results.append(result)

    return pd.DataFrame(results)


cv_results = cross_validations(
    X=Xtrain_df,
    grp=ej.cat.codes.values,
    y=y,
    n_folds=10,
    num_warmup=1000,
    num_samples=1000,
    num_chains=4,)

# %%
cv_results.describe()

# %% [markdown]
# * With `correlation_threshold = 0.7` and oversampling:
#
# |       |  f1_train |   f1_test | log_loss_train | log_loss_test |
# |------:|----------:|----------:|---------------:|--------------:|
# | count | 10.000000 | 10.000000 | 10.000000      | 10.000000     |
# |  mean | 0.993466  | 0.708207  | 0.029350       | 0.337743      |
# |  std  | 0.002988  | 0.118231  | 0.011614       | 0.124673      |
# |  min  | 0.990185  | 0.545455  | 0.015218       | 0.135415      |
# |  25%  | 0.990217  | 0.623188  | 0.020611       | 0.236415      |
# |  50%  | 0.994549  | 0.666667  | 0.027627       | 0.394563      |
# |  75%  | 0.995633  | 0.801587  | 0.040340       | 0.409955      |
# |  max  | 0.997821  | 0.909091  | 0.044500       | 0.522006      |
#
# * With `correlation_threshold = 0.7` and __NO__ oversampling:
#
# |       |  f1_train |   f1_test | log_loss_train | log_loss_test |
# |------:|----------:|----------:|---------------:|--------------:|
# | count | 10.000000 | 10.000000 | 10.000000      | 10.000000     |
# |  mean | 0.928349  | 0.708329  | 0.160594       | 0.321467      |
# |  std  | 0.029628  | 0.079225  | 0.019968       | 0.078987      |
# |  min  | 0.864865  | 0.555556  | 0.126313       | 0.211935      |
# |  25%  | 0.911487  | 0.666667  | 0.143092       | 0.255056      |
# |  50%  | 0.932614  | 0.697826  | 0.165937       | 0.329248      |
# |  75%  | 0.947780  | 0.755639  | 0.176564       | 0.376480      |
# |  max  | 0.968421  | 0.833333  | 0.183823       | 0.439812      |
