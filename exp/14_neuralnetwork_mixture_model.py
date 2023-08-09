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
from datetime import datetime
from imblearn.over_sampling import SMOTENC
import numpy as np
import operator
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import bernoulli
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from typing import Callable, Literal


# torch.autograd.set_detect_anomaly(True)

# %%
kaggle_submission = False
run_cv = True

# %% [markdown]
# # Neural Network Classifier with Gaussian Mixture Model
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
# Xtrain_df['EJ'] = ej.cat.codes


# %% [markdown]
# ### Correlations Removal

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
# ### Neural Network Classifier

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
        )

        self.output = nn.Sequential(
            nn.LayerNorm(64),
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
# ### Gaussian Mixture Network

# %%
class GaussianMixtureNetwork(nn.Module):
    def __init__(self, input_shape: int, nb_classes: int):
        super().__init__()
        self.input_shape = input_shape
        self.mu = nn.Linear(nb_classes, input_shape)
        self.sigma = nn.Linear(nb_classes, input_shape)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the mu and sigma of the given class.

        x: torch.Tensor
            A tensor of one-hot vectors of classes.

        Returns:
            A tuple of mu (nb_classes, nb_features)
            and covariance matrix (nb_classes, nb_features, nb_features).
        """
        mu = self.mu(x)
        sigma = torch.exp(self.sigma(x))
        return mu, sigma


# %% [markdown]
# ### Training Loop

# %%
@dataclass(slots=True)
class Losses:
    classification_loss: torch.Tensor | float
    gmm_loss: torch.Tensor | float
    gmm_separation_loss: torch.Tensor | float
    gmm_unit_variance_loss: torch.Tensor | float

    def item(self):
        items = []
        for var in self.__slots__:
            val = self.__getattribute__(var)
            try:
                items.append(val.item())
            except Exception:
                items.append(val)

        return Losses(*items)

    def calc_total_loss(self, weights: dict):
        return sum(self.__getattribute__(var) * weights[var]
                   for var in self.__slots__)

    def __add__(self, other):
        if not isinstance(other, Losses):
            raise ValueError(
                'Does not support addition with other class than `Losses`.')

        results = []
        for var in self.__slots__:
            results.append(self.__getattribute__(var)
                           + other.__getattribute__(var))

        return Losses(*results)

    def __truediv__(self, other):
        if not isinstance(other, int) and not isinstance(other, float):
            raise ValueError(
                'Does not support division values other than int or float')

        results = []
        for var in self.__slots__:
            results.append(self.__getattribute__(var) / other)

        return Losses(*results)


class TrainingLoop:
    NB_CLASSES = 2
    CLASSIFIER_MODEL_TMP = 'tmp_14_classifier.pth'
    GMM_MODEL_TMP = 'tmp_14_gmm.pth'

    def __init__(
        self,
            classifier: NNClassifier,
            gmm: GaussianMixtureNetwork, *,
            device: str = 'cpu'):
        self.classifier = classifier.to(device)
        self.gmm = gmm.to(device)
        self.device = device

    def train(self,
              train_ds: DataLoader, val_ds: DataLoader, *,
              writer: SummaryWriter,
              tag: str,
              loss_weights: dict,
              epochs: int,
              early_stopping_patience: int,
              train_noise: float = 0.0,
              lr: float = 1e-3,
              weight_decay: float = 1e-2,
              on_epoch_end: Callable[[int, NNClassifier], None] | None = None):
        # Create optimizers.
        self._classifier_opt = torch.optim.Adam(
            self.classifier.parameters(),
            lr=lr, weight_decay=weight_decay)
        self._gmm_opt = torch.optim.Adam(
            self.gmm.parameters(),
            lr=lr, weight_decay=weight_decay)

        # Init models' weights.
        self.classifier = self.init_model_weights(self.classifier)
        self.gmm = self.init_model_weights(self.gmm)

        # Early stopping stuffs.
        patience = 0
        min_loss = np.inf

        # Training.
        bar = tqdm(range(epochs), desc='Training', total=epochs)
        for epoch in bar:
            train_losses = self._train_step(
                train_ds,
                train_noise=train_noise,
                loss_weights=loss_weights)
            val_losses = self._eval_step(val_ds)

            bar.set_postfix_str(
                self.pretty_losses_str('Trn', train_losses) + ' '
                + self.pretty_losses_str('Val', val_losses))

            self.log_losses(
                writer, train_losses, loss_weights=loss_weights,
                tag=f'{tag}/Train', epoch=epoch)
            self.log_losses(
                writer, val_losses, loss_weights=loss_weights,
                tag=f'{tag}/Val', epoch=epoch)

            if on_epoch_end is not None:
                on_epoch_end(epoch, self.classifier)

            val_loss = val_losses.calc_total_loss(loss_weights)
            if val_loss <= min_loss:
                min_loss = val_loss
                patience = 0
                self._save_models()
            elif patience > early_stopping_patience:
                print(
                    f'Wait for {patience} epochs without any improvements. '
                    'STOPPING!')
                break
            else:
                patience += 1

        return self._load_models(), None

    def _train_step(self, train_ds: DataLoader, *,
                    train_noise: float, loss_weights: dict):
        """
        Loop over the trainining dataset and train the models.
        Furthermore, it will also calculate the average performance metrics
        of the models on the given dataset.
        """
        classifier_opt, gmm_opt = self._classifier_opt, self._gmm_opt
        device = self.device
        total_losses = Losses(
            classification_loss=0.,
            gmm_loss=0.,
            gmm_separation_loss=0.,
            gmm_unit_variance_loss=0.)

        # Switch to training mode.
        self.classifier.train()
        self.gmm.train()

        # Loop over all training data and train the model.
        for X, y in train_ds:
            X, y = X.to(device), y.to(device)
            X_noisy = X + train_noise * torch.randn_like(X)

            # Aggregate all losses.
            losses = self._calc_losses(X_noisy, y)
            loss = losses.calc_total_loss(loss_weights)
            total_losses += losses.item()

            # Perform back-propagation.
            classifier_opt.zero_grad()
            gmm_opt.zero_grad()
            loss.backward()
            classifier_opt.step()
            gmm_opt.step()

        return total_losses / len(train_ds)

    @torch.no_grad()
    def _eval_step(self, val_ds: DataLoader):
        self.classifier.eval()
        self.gmm.eval()

        device = self.device
        total_losses = Losses(
            classification_loss=0.,
            gmm_loss=0.,
            gmm_separation_loss=0.,
            gmm_unit_variance_loss=0.)

        # Loop over all training data and train the model.
        for X, y in val_ds:
            X, y = X.to(device), y.to(device)

            # Aggregate all losses.
            total_losses += self._calc_losses(X, y).item()

        return total_losses / len(val_ds)

    def _save_models(self):
        torch.save(self.classifier.state_dict(), self.CLASSIFIER_MODEL_TMP)
        torch.save(self.gmm.state_dict(), self.GMM_MODEL_TMP)

    def _load_models(self):
        self.classifier.load_state_dict(
            torch.load(self.CLASSIFIER_MODEL_TMP))
        self.gmm.load_state_dict(
            torch.load(self.GMM_MODEL_TMP))

        return self.classifier, self.gmm

    def _calc_losses(self, X: torch.Tensor, y: torch.Tensor) -> Losses:
        classifier, gmm = self.classifier, self.gmm

        # First, obtain the embedding of X and the predictions separately.
        X_embed = classifier(X, mode='encoder')
        y_pred_logit = classifier(X_embed, mode='output', logit=True)

        # Second, obtain the mu and covariance of the given classes.
        all_classes = F.one_hot(torch.arange(
            self.NB_CLASSES)).float().to(X.device)
        mu, sigma = gmm(all_classes)

        # Calculate classification loss.
        classification_loss = self.calc_classification_loss(
            y, y_pred_logit)

        # Calculate gaussian mixture losses.
        gmm_loss = self.calc_gaussian_mixture_loss(
            X_embed, y, mu=mu, sigma=sigma)

        # Maximize the separation of distributions in the mixture.
        separation_loss = self.calc_mixture_separation_loss(mu)

        # TODO: should I add another loss for covariance matrix
        # such that to shrink the distributions?
        gmm_unit_variance_loss = self.calc_gmm_unit_variance_loss(sigma)

        return Losses(
            classification_loss=classification_loss,
            gmm_loss=gmm_loss,
            gmm_separation_loss=separation_loss,
            gmm_unit_variance_loss=gmm_unit_variance_loss)

    def log_losses(
            self, writer: SummaryWriter, losses: Losses, loss_weights: dict,
            tag: str, epoch: int):
        writer.add_scalar(
            f'{tag}/classification_loss', losses.classification_loss, epoch)
        writer.add_scalar(
            f'{tag}/gmm_loss', losses.gmm_loss, epoch)
        writer.add_scalar(
            f'{tag}/gmm_sep_loss', losses.gmm_separation_loss, epoch)
        writer.add_scalar(
            f'{tag}/gmm_unit_var_loss', losses.gmm_unit_variance_loss, epoch)
        writer.add_scalar(
            f'{tag}/total_loss', losses.calc_total_loss(loss_weights), epoch)

    @staticmethod
    def pretty_losses_str(prefix: str, losses: Losses):
        return (f'{prefix} classification: {losses.classification_loss:.4f}'
                f'- {prefix} GMM: {losses.gmm_loss:.4f}'
                f'- {prefix} GMM sep: {losses.gmm_separation_loss:.4f}'
                f'- {prefix} GMM var: {losses.gmm_unit_variance_loss:.4f}')

    @staticmethod
    def init_model_weights(model: nn.Module,
                           strategy_fn=nn.init.xavier_normal_):
        def init_weights(m: nn.Module):
            if type(m) in [nn.Linear, nn.LazyLinear]:
                strategy_fn(m.weight)

        return model.apply(init_weights)

    @staticmethod
    def calc_classification_loss(y_true, y_pred_logit):
        return F.binary_cross_entropy_with_logits(y_pred_logit, y_true)

    @staticmethod
    def calc_gaussian_mixture_loss(X: torch.Tensor, y: torch.Tensor, *,
                                   mu: torch.Tensor, sigma: torch.Tensor):
        """
        Calculate negative log-likelihood loss for all classes.

        X: torch.Tensor of shape (nb_samples, nb_features).
        y: torch.Tensor of shape (nb_samples,).
        mu: torch.Tensor of shape (nb_classes, nb_features).
        cov: torch.Tensor of shape (nb_classes, nb_features, nb_features).
        """
        def unnormalized_log_prob(x, mu, sigma):
            # x of shape (nb_samples, nb_features)
            # mu of shape (nb_features,)
            # sigma of shape (nb_features,)
            diff = x - mu[None, ...]
            return -0.5 * torch.sum(diff * diff / sigma[None, ...], dim=1)

        loss = 0.

        for i in range(TrainingLoop.NB_CLASSES):
            rows, _ = torch.nonzero(y == i, as_tuple=True)
            log_likelihood = unnormalized_log_prob(X[rows], mu[i], sigma[i])
            loss += torch.mean(-log_likelihood)

        return loss / TrainingLoop.NB_CLASSES

    @staticmethod
    def calc_mixture_separation_loss(mu: torch.Tensor):
        assert mu.shape[0] == 2, 'Only support 2 mu.'
        distances = (mu[0] - mu[1]) ** 2
        losses = torch.exp(-distances)
        return torch.mean(losses)

    @staticmethod
    def calc_gmm_unit_variance_loss(sigma: torch.Tensor):
        diff = sigma - 1
        return torch.mean(torch.mean(diff * diff, axis=1))


# %% [markdown]
# ### Oversampling

# %%
def sampling(X, y):
    ej_idx = len(X.columns) - 1
    ros = SMOTENC([ej_idx], sampling_strategy='all', random_state=0)
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


def estimate_optimal_prob_pred(y_prob, nb_samples: int = 20000):
    ys = bernoulli.rvs(y_prob[:, None], size=(y_prob.shape[0], nb_samples))
    return calculate_optimal_prob_prediction(ys.T)


def f1_recall_precision(y_true, y_pred):
    return tuple(f(y_true, y_pred)
                 for f in [f1_score, recall_score, precision_score])


@torch.no_grad()
def log_embedding_and_performance_metrics(
    epoch: int, model: NNClassifier, *,
        writer: SummaryWriter,
        tag: str,
        Xtr: torch.Tensor, ytr: np.ndarray,
        Xte: torch.Tensor, yte: np.ndarray):

    model.eval()
    writer_train_tag = f'{tag}/Train'
    writer_test_tag = f'{tag}/Test'

    ytr_prob = (model(Xtr).cpu().detach().numpy().squeeze())
    ytr_pred = np.where(ytr_prob > 0.5, 1., 0.)
    (f1_train,
     recall_train,
     precision_train) = f1_recall_precision(ytr, ytr_pred)
    # ytr_opt_prob = estimate_optimal_prob_pred(ytr_prob)
    log_loss_train = balanced_log_loss(ytr, ytr_prob)
    # opt_log_loss_train = balanced_log_loss(ytr_orig, ytr_opt_prob)

    writer.add_scalar(f'{writer_train_tag}/F1', f1_train, epoch)
    writer.add_scalar(f'{writer_train_tag}/Recall', recall_train, epoch)
    writer.add_scalar(f'{writer_train_tag}/Precision', precision_train, epoch)
    writer.add_scalar(f'{writer_train_tag}/Log Loss', log_loss_train, epoch)

    yte_prob = (model(Xte).cpu().detach().numpy().squeeze())
    # yte_opt_prob = estimate_optimal_prob_pred(yte_prob)
    yte_pred = np.where(yte_prob > 0.5, 1., 0.)
    (f1_test,
     recall_test,
     precision_test) = f1_recall_precision(yte, yte_pred)
    log_loss_test = balanced_log_loss(yte, yte_prob)
    # opt_log_loss_test = balanced_log_loss(yte, yte_opt_prob)

    writer.add_scalar(f'{writer_test_tag}/F1', f1_test, epoch)
    writer.add_scalar(f'{writer_test_tag}/Recall', recall_test, epoch)
    writer.add_scalar(f'{writer_test_tag}/Precision', precision_test, epoch)
    writer.add_scalar(f'{writer_test_tag}/Log Loss', log_loss_test, epoch)

    # Log embeddings.
    if epoch % 10 == 0:
        X = torch.concat([Xtr, Xte], axis=0)
        y = np.concatenate([ytr, yte], axis=0)
        dstype = ['train'] * len(ytr) + ['test'] * len(yte)

        Xembed = model(X, mode='encoder').detach().cpu().numpy()
        writer.add_embedding(
            Xembed,
            tag=f'{tag}/Embeddings',
            global_step=epoch,
            metadata=list(zip(y, dstype)),
            metadata_header=['Label', 'Data Type'])


@dataclass
class TrainAndEvaluateResult:
    model: NNClassifier
    features: list[str]
    preprocessing: Pipeline
    metrics: dict


def train_and_evaluate(
    *, Xtr, gtr, ytr,
        Xte, gte, yte,
        preprocessing: Pipeline,
        tag: str,
        epochs: int = 100,
        device: str = 'cpu',
        correlation_threshold: float = 0.3,
        writer: SummaryWriter,
        **train_kwargs) -> TrainAndEvaluateResult:
    # First, we will normalize the data.
    Xtr = pd.DataFrame(
        preprocessing.fit_transform(Xtr, ytr),
        columns=Xtr.columns)
    Xte = pd.DataFrame(
        preprocessing.transform(Xte),
        columns=Xte.columns)

    # Next, we'll filter out correlated features.
    Xtr = filter_in_uncorrelated_features(
        Xtr, correlation_threshold=correlation_threshold)
    uncorrelated_features = Xtr.columns.tolist()
    Xte = Xte[uncorrelated_features]

    # Store original training dataset.
    Xtr_orig = Xtr.copy()
    Xtr_orig['ej'] = gtr
    ytr_orig = ytr.copy()
    print(f'Before sampling, has {len(ytr_orig)} sammples,\n'
          f'in which there are {ytr_orig.sum()} positive samples.')

    # Next, we'll perform sampling.
    Xtr['ej'] = gtr
    Xte['ej'] = gte

    Xtr, ytr = sampling(Xtr, ytr)
    print(f'After sampling, has {len(ytr)} sammples,\n'
          f'in which there are {ytr.sum()} positive samples.')

    nb_features = len(Xtr.columns)
    print('Number of important features: ', nb_features)

    # Training neural network model.
    X_train_ds = TensorDataset(
        torch.tensor(Xtr.values, dtype=torch.float32),
        torch.tensor(ytr[:, None], dtype=torch.float32))
    X_val_ds = TensorDataset(
        torch.tensor(Xte.values, dtype=torch.float32),
        torch.tensor(yte[:, None], dtype=torch.float32))

    Xtr_dataloader = DataLoader(X_train_ds, batch_size=128, shuffle=True)
    Xva_dataloader = DataLoader(X_val_ds, batch_size=128)

    training_loop = TrainingLoop(
        NNClassifier(nb_features),
        GaussianMixtureNetwork(64, 2),
        device=device)
    (model, _), history = training_loop.train(
        train_ds=Xtr_dataloader,
        val_ds=Xva_dataloader,
        epochs=epochs,
        writer=writer,
        tag=tag,
        on_epoch_end=lambda epoch, model:
            log_embedding_and_performance_metrics(
                epoch, model,
                writer=writer,
                tag=tag,
                Xtr=torch.tensor(
                    Xtr_orig.values, dtype=torch.float32).to(device),
                ytr=ytr_orig,
                Xte=torch.tensor(Xte.values, dtype=torch.float32).to(device),
                yte=yte,
            ),
        **train_kwargs)

    # Evaluate the model.
    ytr_prob = (model(
        torch.tensor(Xtr_orig.values, dtype=torch.float32).to(device))
        .cpu().detach().numpy().squeeze())
    ytr_opt_prob = estimate_optimal_prob_pred(ytr_prob)
    ytr_pred = np.where(ytr_prob > 0.5, 1., 0.)
    (f1_train,
     recall_train,
     precision_train) = f1_recall_precision(ytr_orig, ytr_pred)
    log_loss_train = balanced_log_loss(ytr_orig, ytr_prob)
    opt_log_loss_train = balanced_log_loss(ytr_orig, ytr_opt_prob)
    print(f'Train - f1={f1_train:.4f} recall={recall_train:.4f} '
          f'precision={precision_train:.4f} log-loss={log_loss_train:.4f} '
          f'opt-log-loss={opt_log_loss_train:.4f}')

    yte_prob = (model(
        torch.tensor(Xte.values, dtype=torch.float32).to(device))
        .cpu().detach().numpy().squeeze())
    yte_opt_prob = estimate_optimal_prob_pred(yte_prob)
    yte_pred = np.where(yte_prob > 0.5, 1., 0.)
    (f1_test,
     recall_test,
     precision_test) = f1_recall_precision(yte, yte_pred)
    log_loss_test = balanced_log_loss(yte, yte_prob)
    opt_log_loss_test = balanced_log_loss(yte, yte_opt_prob)
    print(f'Test  - f1={f1_test:.4f} recall={recall_test:.4f} '
          f'precision={precision_test:.4f} log-loss={log_loss_test:.4f} '
          f'opt-log-loss={opt_log_loss_test:.4f}')

    return TrainAndEvaluateResult(
        model=model,
        features=uncorrelated_features,
        preprocessing=preprocessing,
        metrics=dict(f1_train=f1_train,
                     f1_test=f1_test,
                     log_loss_train=log_loss_train,
                     opt_log_loss_train=opt_log_loss_train,
                     log_loss_test=log_loss_test,
                     opt_log_loss_test=opt_log_loss_test,
                     ))


@dataclass
class TrainingResult:
    model: NNClassifier
    features: list[str]
    preprocessing: Pipeline
    performance: float


def cross_validations(
    *, X, grp, y,
        n_folds: int = 10,
        repeats_per_fold: int = 1,
        only_folds: list[int] | None = None,
        keep_best_in_fold_method: str = 'f1_test',
        **kwargs):
    assert keep_best_in_fold_method in ['f1_test', 'opt_log_loss_test']

    metrics = []
    best_models: list[TrainingResult] = []

    kfolds = StratifiedKFold(n_splits=n_folds)
    for fold, (train_idx, test_idx) in enumerate(kfolds.split(X, y)):
        if only_folds is not None and (fold + 1) not in only_folds:
            print(f'SKIPPED fold #{fold + 1}')
            continue

        best_model_metric = (0.0
                             if keep_best_in_fold_method == 'f1_test'
                             else np.inf)
        best_model = None

        for repeat in range(repeats_per_fold):
            print(f'\n-- Fold # {fold + 1}/{n_folds} - '
                  f'Repeat #{repeat + 1}/{repeats_per_fold}:')

            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            gtr, gte = grp[train_idx], grp[test_idx]
            ytr, yte = y[train_idx], y[test_idx]

            result = train_and_evaluate(
                Xtr=Xtr, gtr=gtr, ytr=ytr,
                Xte=Xte, gte=gte, yte=yte,
                tag=f'Fold {fold + 1}/Repeat {repeat + 1}',
                **kwargs)

            metric = result.metrics
            metric['fold'] = fold + 1
            metric['repeat'] = repeat + 1
            metrics.append(metric)

            # Compare the model's metric and retain the best model.
            op = (operator.gt
                  if keep_best_in_fold_method == 'f1_test'
                  else operator.lt)
            if op(metric[keep_best_in_fold_method], best_model_metric):
                best_model_metric = metric[keep_best_in_fold_method]
                best_model = TrainingResult(
                    model=result.model,
                    features=result.features,
                    preprocessing=result.preprocessing,
                    performance=best_model_metric,
                )

        # Store the result.
        best_models.append(best_model)

    return pd.DataFrame(metrics), best_models


date = datetime.now().strftime('%Y%m%d_%H_%M')
log_dir = (f'../tensorboard_logs/{date}_14_nn_gaussian_mixture'
           if not kaggle_submission
           else f'tensorboard_logs/{date}_14_nn_gaussian_mixture')
writer = SummaryWriter(log_dir)
print(f'Tensorboard at: {log_dir}')
train_evaluate_kwargs = dict(
    n_folds=10,
    repeats_per_fold=5,
    epochs=2000,
    correlation_threshold=0.3,
    lr=1e-4,
    early_stopping_patience=100,
    weight_decay=1e-2,
    # train_noise=0.01,
    train_noise=0.0,
    loss_weights=dict(
        classification_loss=1.,
        gmm_loss=0.5,
        gmm_separation_loss=0.5,
        gmm_unit_variance_loss=0.5,
    )
)

# Log train metadata.
for key, value in train_evaluate_kwargs.items():
    if isinstance(value, dict):
        for k, v in value.items():
            writer.add_scalar(f'metadata/{key}/{k}', v)
    else:
        writer.add_scalar(f'metadata/{key}', value)

# %%
keep_best_in_fold_method = 'f1_test'
cv_results, models = cross_validations(
    X=Xtrain_df,
    grp=ej.cat.codes.values,
    y=y,
    preprocessing=clone(preprocessing),
    keep_best_in_fold_method=keep_best_in_fold_method,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    writer=writer,
    only_folds=([1, 6, 7, 9] if not kaggle_submission
                else None),
    **train_evaluate_kwargs)

# %%
cv_results


# %% [markdown]
# ## Results
# ### Optimal F1 Test

# %%
def get_optimal_cv_results_each_fold(results: pd.DataFrame,
                                     by: str,
                                     method: Literal['max', 'min'] = 'max'):
    folds = results.groupby('fold')

    optimal_results = []
    for fold, fold_results in folds:
        best_idx = (fold_results[by].argmax()
                    if method == 'max'
                    else fold_results[by].argmin())
        optimal_result = fold_results.iloc[best_idx].to_dict()
        optimal_results.append(optimal_result)

    return pd.DataFrame(optimal_results)


# %%
# Optimal results per fold by f1_test.
cv_results_optimal_f1_test = get_optimal_cv_results_each_fold(
    cv_results, 'f1_test', 'max')
cv_results_optimal_f1_test

# %%
cv_results_optimal_f1_test.describe()

# %% [markdown]
# ### Optimal `opt_log_loss_test`

# %%
cv_results_optimal_log_loss_test = get_optimal_cv_results_each_fold(
    cv_results, 'opt_log_loss_test', 'min')
cv_results_optimal_log_loss_test

# %%
cv_results_optimal_log_loss_test.describe()

# %% [markdown]
# # Results Records
#
# ## No Train Noise, GMM loss = 0.5, GMM separation = 0.5
#
# F1:
#
# |   | f1_train |  f1_test | log_loss_train | opt_log_loss_train | log_loss_test | opt_log_loss_test | fold | repeat |
# |--:|---------:|---------:|---------------:|-------------------:|--------------:|------------------:|-----:|-------:|
# | 0 | 0.989691 | 0.782609 | 0.037188       | 0.071455           | 0.301484      | 0.256083          | 1.0  | 4.0    |
# | 1 | 0.989691 | 0.727273 | 0.021690       | 0.049760           | 0.459746      | 0.376421          | 6.0  | 2.0    |
# | 2 | 0.979592 | 0.571429 | 0.057072       | 0.081473           | 0.708823      | 0.553241          | 7.0  | 4.0    |
# | 3 | 1.000000 | 0.842105 | 0.002505       | 0.005764           | 0.434787      | 0.357118          | 9.0  | 5.0    |
#
# Optimal Log Loss:
#
# |   | f1_train |  f1_test | log_loss_train | opt_log_loss_train | log_loss_test | opt_log_loss_test | fold | repeat |
# |--:|---------:|---------:|---------------:|-------------------:|--------------:|------------------:|-----:|-------:|
# | 0 | 0.989691 | 0.666667 | 0.033394       | 0.053460           | 0.323337      | 0.243919          | 1.0  | 3.0    |
# | 1 | 0.989691 | 0.727273 | 0.021690       | 0.049760           | 0.459746      | 0.376421          | 6.0  | 2.0    |
# | 2 | 0.974619 | 0.545455 | 0.057461       | 0.096276           | 0.612108      | 0.492546          | 7.0  | 3.0    |
# | 3 | 1.000000 | 0.842105 | 0.002505       | 0.005764           | 0.434787      | 0.357118          | 9.0  | 5.0    |
#
# ## Train Noise = 0.01, GMM loss = 0.5, GMM separation = 0.5
#
# F1:
#
# |   | f1_train |  f1_test | log_loss_train | opt_log_loss_train | log_loss_test | opt_log_loss_test | fold | repeat |
# |--:|---------:|---------:|---------------:|-------------------:|--------------:|------------------:|-----:|-------:|
# | 0 | 0.989691 | 0.761905 | 0.027445       | 0.048986           | 0.335425      | 0.260908          | 1.0  | 1.0    |
# | 1 | 0.989691 | 0.700000 | 0.027972       | 0.050983           | 0.600694      | 0.475514          | 6.0  | 2.0    |
# | 2 | 0.979592 | 0.571429 | 0.064876       | 0.100823           | 0.668596      | 0.529985          | 7.0  | 3.0    |
# | 3 | 1.000000 | 0.761905 | 0.000947       | 0.001941           | 0.505035      | 0.442522          | 9.0  | 4.0    |
#
# Optimal Log Loss:
#
# |   | f1_train |  f1_test | log_loss_train | opt_log_loss_train | log_loss_test | opt_log_loss_test | fold | repeat |
# |--:|---------:|---------:|---------------:|-------------------:|--------------:|------------------:|-----:|-------:|
# | 0 | 0.989691 | 0.761905 | 0.027445       | 0.048986           | 0.335425      | 0.260908          | 1.0  | 1.0    |
# | 1 | 0.994872 | 0.636364 | 0.016733       | 0.042278           | 0.542624      | 0.449024          | 6.0  | 3.0    |
# | 2 | 0.959596 | 0.545455 | 0.060397       | 0.095994           | 0.644149      | 0.521739          | 7.0  | 5.0    |
# | 3 | 1.000000 | 0.727273 | 0.001315       | 0.002814           | 0.518738      | 0.441438          | 9.0  | 5.0    |
#
# ## No Train Noise, GMM loss = 0.5, GMM separation = 0.5, GMM unit var = 0.5
#
# F1:
#
# |   | f1_train |  f1_test | log_loss_train | opt_log_loss_train | log_loss_test | opt_log_loss_test | fold | repeat |
# |--:|---------:|---------:|---------------:|-------------------:|--------------:|------------------:|-----:|-------:|
# | 0 | 1.0      | 0.857143 | 0.001318       | 0.004348           | 0.438569      | 0.305311          | 1.0  | 1.0    |
# | 1 | 1.0      | 0.800000 | 0.001423       | 0.004761           | 0.710776      | 0.519708          | 6.0  | 2.0    |
# | 2 | 1.0      | 0.500000 | 0.005280       | 0.013370           | 0.866967      | 0.646218          | 7.0  | 3.0    |
# | 3 | 1.0      | 0.800000 | 0.001023       | 0.001834           | 0.607121      | 0.491031          | 9.0  | 1.0    |
#
# Optimal Log Loss:
#
# |   | f1_train |  f1_test | log_loss_train | opt_log_loss_train | log_loss_test | opt_log_loss_test | fold | repeat |
# |--:|---------:|---------:|---------------:|-------------------:|--------------:|------------------:|-----:|-------:|
# | 0 | 0.989691 | 0.800000 | 0.025421       | 0.043723           | 0.376983      | 0.274818          | 1.0  | 3.0    |
# | 1 | 0.994872 | 0.700000 | 0.005919       | 0.015777           | 0.657027      | 0.482378          | 6.0  | 3.0    |
# | 2 | 0.989691 | 0.476190 | 0.029500       | 0.053289           | 0.727115      | 0.554656          | 7.0  | 4.0    |
# | 3 | 1.000000 | 0.695652 | 0.002594       | 0.006415           | 0.489272      | 0.426375          | 9.0  | 5.0    |
