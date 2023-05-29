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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

# %% [markdown]
# # Random Forest Baseline

# %%
df = pd.read_csv('../data/train.csv')
df.info()

# %%
# Split data into features and target.
X = df.drop(columns=['Id', 'Class'])
y = df['Class']

# %%
# Preprocess categorical column 'EJ' into 0-1 encoding.
ej_onehot = pd.get_dummies(X['EJ'])
X = X.drop(columns='EJ').join(ej_onehot)

# %%
# Replace missing values with median.
medians = X.median()
X = X.fillna(medians)

# %%
rf = RandomForestClassifier(class_weight='balanced')
results = cross_validate(rf, X=X, y=y,
                         cv=10,
                         scoring=['recall', 'precision', 'f1'],
                         return_train_score=True)
results


# %%
# Plot cross validation results.
def plot_cross_validation_results(results,
                                  metrics: list[str], *,
                                  train_prefix: str = 'train_',
                                  test_prefix: str = 'test_'):
    def transform_df_structure(df, type, type_prefix):
        type_metrics = [f'{type_prefix}{m}' for m in metrics]

        type_df = df[['Fold', *type_metrics]]
        type_df = pd.melt(type_df,
                       id_vars='Fold',
                       value_vars=type_metrics,
                       var_name='Metric',
                       value_name='Score')
        type_df['Metric'] = type_df['Metric'].replace({
            k:v for k, v in zip(type_metrics, metrics)
        })
        type_df['Data'] = type

        return type_df


    # Convert `results` from dict to pandas' DataFrame.
    df = pd.DataFrame(results)
    df['Fold'] = df.index

    train_df = transform_df_structure(df, 'Train', train_prefix)
    test_df = transform_df_structure(df, 'Test', test_prefix)
    df = pd.concat([train_df, test_df])

    # Print the average for each metric.
    for data_type in ['Train', 'Test']:
        for m in metrics:
            mdf = df[(df['Data'] == data_type) & (df['Metric'] == m)]
            print(f'{data_type=}, metric={m}\n', mdf['Score'].describe())

    # Plot the result.
    sns.catplot(df, x='Fold', y='Score', hue='Metric', col='Data', kind='bar')


plot_cross_validation_results(results, ['f1', 'precision', 'recall'])
