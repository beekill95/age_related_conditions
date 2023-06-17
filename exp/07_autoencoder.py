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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
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
# Then the resulting vector will be fed to Bayesian logistic regression model.
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

    def forward(self, x):
        x = self.encoder(x)
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


# Data
X = X_df.values

# Training with 10-fold cross validation.
device = 'cpu'
kfold = KFold(n_splits=10)
for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    autoencoder = AutoEncoder(X.shape[1]).to(device)

    epochs = 100
    train_step, val_step = create_training_and_evaluation_step(autoencoder, weight_decay=1e-2)
    train_losses = []
    val_losses = []

    Xtr = torch.tensor(X[train_idx], dtype=torch.float32)
    Xva = torch.tensor(X[val_idx], dtype=torch.float32)
    X_train_ds = TensorDataset(Xtr, Xtr) # pyright: ignore
    X_val_ds = TensorDataset(Xva, Xva) # pyright: ignore

    Xtr_dataloader = DataLoader(X_train_ds, batch_size=64, shuffle=True)
    Xva_dataloader = DataLoader(X_val_ds, batch_size=64)

    bar = tqdm(range(epochs), total=epochs, desc=f'Fold #{fold + 1}')
    for epoch in bar:
        train_loss = train_step(Xtr_dataloader, device, epoch, progress=False)
        train_losses.append(train_loss)

        val_loss = val_step(Xva_dataloader, device)
        val_losses.append(val_loss)

        bar.set_postfix_str(f'Train: {train_loss:.4f} - Val: {val_loss:.4f}')

    # Best validation score and corresponding train score.
    best_val_idx = np.argmin(val_losses)
    print(f'Train: {train_losses[best_val_idx]:.4f} - Val: {val_losses[best_val_idx]:.4f} at epoch {best_val_idx}.')
