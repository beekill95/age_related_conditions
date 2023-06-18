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
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

# %%
kaggle_submission = False

# %% [markdown]
# # Autoencoder
#
# In this experiment,
# I'll use autoencoder network to encode sample from feature space
# to a small subspace.
# Then the resulting vector will be fed to logistic regression model.
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
    # ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

# %%
X_df = train_df.drop(columns=['Id', 'Class', 'EJ'])
y = train_df['Class']
ej = train_df['EJ'].replace({'A': -1, 'B': 1})

# Fill missing values with medians.
X_df = X_df.fillna(X_df.median())

X_df = pd.DataFrame(
    preprocessing.fit_transform(X_df),
    columns=X_df.columns,
    index=X_df.index)
y = y.values

# %%
Xtest_df = test_df.drop(columns=['Id', 'EJ'])
ej_test = test_df['EJ'].replace({'A': -1, 'B': 1})

Xtest_df = pd.DataFrame(
    preprocessing.transform(Xtest_df),
    columns=Xtest_df.columns,
    index=Xtest_df.index)

# %%
def create_interaction_terms_between(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    assert all(f in df.columns for f in features)

    interactions = dict()

    for i, fst in enumerate(features):
        for snd in features[i+1:]:
            interactions[f'{fst}*{snd}'] = df[fst] * df[snd]

    return pd.DataFrame(interactions)


features = X_df.columns.tolist()
X_interactions_df = create_interaction_terms_between(
    X_df, features)

# %%
# Quadratic terms.
X2_df = X_df.pow(2.).rename(columns={
    f: f'{f}^2' for f in features
})

all_df = pd.concat([X_df, X_interactions_df, X2_df], axis=1)
all_df = pd.DataFrame(
    preprocessing.fit_transform(all_df),
    columns=all_df.columns,
    index=all_df.index)

# %% [markdown]
# ## Autoencoder Model

# %%
class AutoEncoder(nn.Module):
    def __init__(self, input_shape: int) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 2048), nn.ReLU(), nn.Dropout(), nn.LayerNorm(2048),
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(), nn.LayerNorm(512),
            nn.Linear(512, 64), nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 512), nn.ReLU(), nn.Dropout(), nn.LayerNorm(512),
            nn.Linear(512, 2048), nn.ReLU(), nn.Dropout(), nn.LayerNorm(2048),
            nn.Linear(2048, input_shape),
        )

    def forward(self, x, encoder_only: bool = False):
        x = self.encoder(x)

        if not encoder_only:
            x = self.decoder(x)

        return x


def create_training_and_evaluation_step(model: nn.Module, lr=1e-3, weight_decay=1e-5):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    # Init weights before training.
    def init_weights(m: nn.Module):
        strategy_fn = nn.init.xavier_normal_
        if type(m) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.LazyConv1d, nn.LazyLinear]:
            strategy_fn(m.weight) # pyright: ignore

    model.apply(lambda m: init_weights(m))

    def train_step(dataloader: DataLoader, device: str, epoch: int, progress: bool = True):
        model.train()

        train_loss = 0
        num_batches = len(dataloader)
        bar = (tqdm(enumerate(dataloader), total=num_batches, desc=f'Epoch {epoch}')
               if progress
               else enumerate(dataloader))
        for i, (X, y) in bar:
            X, y = X.to(device), y.to(device)

            # Make prediction and calculate loss.
            pred = model(X)
            loss = loss_fn(pred, y)
            train_loss += loss.item()

            # Back-propagation step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Show the MSE.
            if progress:
                bar.set_postfix_str(f'Loss: {(train_loss / (i + 1)):.4f}')

        return train_loss / num_batches

    def evaluate_step(dataloader: DataLoader, device: str):
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()

        test_loss /= num_batches
        return test_loss

    return train_step, evaluate_step


def train(model: nn.Module,
          *,
          train_ds: DataLoader,
          val_ds: DataLoader,
          epochs: int,
          early_stopping_patience: int = 10,
          device: str = 'cpu'):
    def save_checkpoint(model, path):
        torch.save(model.state_dict(), path)

    def load_checkpoint(model, path):
        model.load_state_dict(torch.load(path))
        return model


    model = model.to(device)

    train_step, val_step = create_training_and_evaluation_step(model, weight_decay=1e-2)
    train_losses = []
    val_losses = []

    tmp_path = 'tmp_autoencoder.pth'

    bar = tqdm(range(epochs), total=epochs, desc='Training')
    for epoch in bar:
        train_loss = train_step(train_ds, device, epoch, progress=False)
        train_losses.append(train_loss)

        val_loss = val_step(val_ds, device)
        val_losses.append(val_loss)

        bar.set_postfix_str(f'Train: {train_loss:.4f} - Val: {val_loss:.4f}')

        if val_loss <= np.min(val_losses):
            save_checkpoint(model, tmp_path)

    # Best validation score and corresponding train score.
    best_val_idx = np.argmin(val_losses)
    print(f'Train: {train_losses[best_val_idx]:.4f} - Val: {val_losses[best_val_idx]:.4f} at epoch {best_val_idx}.')

    # Restore the best model.
    print('Restore the best model.')
    return load_checkpoint(model, tmp_path)


def balanced_log_loss(y_true, pred_prob):
    nb_class_0 = np.sum(1 - y_true)
    nb_class_1 = np.sum(y_true)

    prob_0 = np.clip(1. - pred_prob, 1e-10, 1. - 1e-10)
    prob_1 = np.clip(pred_prob, 1e-10, 1. - 1e-10)
    return (-np.sum((1 - y_true) * np.log(prob_0)) / nb_class_0
            - np.sum(y_true * np.log(prob_1)) / nb_class_1) / 2.


# Data
X = X_df.values

# Training with 10-fold cross validation.
device = 'cpu'
kfold = StratifiedKFold(n_splits=10)

train_log_losses = []
train_f1_scores = []
val_log_losses = []
val_f1_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    Xtr = torch.tensor(X[train_idx], dtype=torch.float32)
    Xva = torch.tensor(X[val_idx], dtype=torch.float32)
    X_train_ds = TensorDataset(Xtr, Xtr) # pyright: ignore
    X_val_ds = TensorDataset(Xva, Xva) # pyright: ignore

    Xtr_dataloader = DataLoader(X_train_ds, batch_size=64, shuffle=True)
    Xva_dataloader = DataLoader(X_val_ds, batch_size=64)

    autoencoder = AutoEncoder(X.shape[1]).to(device)
    autoencoder = train(autoencoder,
                        train_ds=Xtr_dataloader,
                        val_ds=Xva_dataloader,
                        epochs=100,
                        early_stopping_patience=10,
                        device=device)

    # Use the autoencoder to encode data and train logistic regression model.
    Xtr_encoded = autoencoder(Xtr, encoder_only=True).detach().cpu().numpy()
    Xva_encoded = autoencoder(Xva, encoder_only=True).detach().cpu().numpy()
    ytr = y[train_idx]
    yva = y[val_idx]

    logistic = LogisticRegression(max_iter = 10000000,
                                  class_weight = 'balanced',
                                  solver = 'liblinear')
    logistic.fit(Xtr_encoded, ytr)

    # Show the performance measures.
    ytr_pred = logistic.predict(Xtr_encoded)
    yva_pred = logistic.predict(Xva_encoded)

    f1_train = f1_score(ytr, ytr_pred)
    precision_train = precision_score(ytr, ytr_pred)
    recall_train = recall_score(ytr, ytr_pred)
    log_loss_train = balanced_log_loss(ytr, logistic.predict_proba(Xtr_encoded)[:, 1])
    print(f'Train - f1={f1_train:.4f} recall={recall_train:.4f} precision={precision_train:.4f} log-loss={log_loss_train:.4f}')

    f1_val = f1_score(yva, yva_pred)
    precision_val = precision_score(yva, yva_pred)
    recall_val = recall_score(yva, yva_pred)
    log_loss_val = balanced_log_loss(yva, logistic.predict_proba(Xva_encoded)[:, 1])
    print(f'Valid - f1={f1_val:.4f} recall={recall_val:.4f} precision={precision_val:.4f} log-loss={log_loss_val:.4f}')

    # Store the results.
    train_f1_scores.append(f1_train)
    train_log_losses.append(log_loss_train)
    val_f1_scores.append(f1_val)
    val_log_losses.append(log_loss_val)


# %%
print(f'Train - F1={np.mean(train_f1_scores):.4f} +- {np.std(train_f1_scores):.4f}'
      f'; Log Loss = {np.mean(train_log_losses):.4f} +- {np.std(train_log_losses):.4f}')
print(f'Valid - F1={np.mean(val_f1_scores):.4f} +- {np.std(val_f1_scores):.4f}'
      f'; Log Loss = {np.mean(val_log_losses):.4f} +- {np.std(val_log_losses):.4f}')

