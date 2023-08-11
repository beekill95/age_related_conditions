# ICR - Identifying Age-Related Conditions Kaggle Competition

Competition Homepage: https://www.kaggle.com/competitions/icr-identify-age-related-conditions

## My Approaches
### Bayesian Logistic Regression

* Notebook: [exp/04_logistics_bayesian_v4.1.py](https://github.com/beekill95/age_related_conditions/blob/main/exp/04_logistics_bayesian_v4.1.py);
CV: na / Public LB: 0.25 / Private LB: 0.45 (with interaction terms).
* Noteboook: [exp/04_logistics_bayesian_v4.2.py](https://github.com/beekill95/age_related_conditions/blob/main/exp/04_logistics_bayesian_v4.2.py);
CV: na / Public LB: 0.24 / Private LB: 0.47 (with interaction and quadratic terms).

This was my go-to model when joining this competition
because it is resilient against overfitting,
especially for this small dataset.

The model consists of a linear combination of:

* Metric terms;
* A group (feature EJ) term;
* Interaction terms (basically x*y);
* Quadratic terms (x^2);

To make the model less susceptible to overfitting,
the weights of metric, interaction, and quadratic features distributed as StudentT.
Moreover, I added a guess term to cover the mislabeled data samples.

Mathematically, the model looks like this:

$$
logit_i = a_0 + \mathbf{a^1} * \mathbf{x}_i + \mathbf{a^2} * \mathbf{x_i^{interaction}} + \mathbf{a^3} * \mathbf{x_i^2} + EJ_i \\
prob_i = 0.5*guess + (1 - guess)*e^{logit_i} \\
y_i \sim Bernoulli(prob_i)
$$

The Bayesian model will produce all possible predictions for $y$.
Thus, to produce a final prediction,
I derived an equation for the prediction that minimizes the risk (log loss).
In particular:

We want to optimize the balanced log loss with the following form:
$$
\text{Log Loss}_i(p) = -\frac{1}{\sum_j y_i^{(j)}} \sum_j y_i^{(j)} log p^{(j)} - \frac{1}{\sum_j (1 - y_i^{(j)})} \sum_j (1 - y_i^{(j)}) log(1 - p^{(j)})
$$
where $i$ is an index to a sample/draw,
and $j$ is an index to a data point.

We will want to convert the loss to a loss for each data point $j$.
With some algebra,
the loss for each data point $j$ is:
$$
\text{Log Loss}(p^{(j)}) = \sum_i \frac{-y_i^{(j)}}{\sum_j y_i^{(j)}} log p^{(j)} - \sum_i \frac{1 - y_i^{(j)}}{\sum_j (1 - y_i^{(j)})} log (1 - p^{(j)})
$$

With this form,
we can find the best $p^{(j)}$ that optimizes the prediction.
Now, take the derivative of the above wrt $p^{(j)}$,
we have:
$$
\frac{\partial \mathcal{L}(p^{(j)})}{\partial p^{(j)}} = -\frac{C_j}{p^{(j)}} + \frac{C^{-1}_j}{1 - p^{(j)}}
$$
where $C_j$ is $\sum_i \frac{y_i^{(j)}}{\sum_j y_i^{(j)}}$,
and $C_j^{-1}$ is $\sum_i \frac{1 - y_i^{(j)}}{\sum_j (1 - y_i^{(j)})}$

The prediction that optimizes the log loss satisfies:
$$
p^{(j)} = \frac{C_j}{C_j + C_j^{-1}}
$$

### Neural Network Models
#### 3-layer Neural Network

* Notebook: [exp/11.1e_features_selection_sampler_neuralnetwork.py](https://github.com/beekill95/age_related_conditions/blob/main/exp/11.1e_features_selection_sampler_neuralnetwork.py);
F1 CV: 0.91 ± 0.05 / Log Loss CV: 0.27 ± 0.06 / Public LB: 0.29 / Private LB: 0.56

This model is just a simple 3-layer neural network with 1024 neurons, followed by 512 neurons and finally 64 neurons in the last layer.
This simple model didn't work very well with the current setup,
so I added a couple of regularization techniques:

* Dropout and Layer normalization;
* Weight decay of 1.0;
* Add some gaussian noise to the training data;
* Early stopping;
* Force the output of the last layer to be normally distributed and uncorrelated.

The input data to the model were preprocessed with the following steps:

* Replace missing values with median value;
* Normalize the data;
* Create interaction and quadratic terms using all columns;
* Remove correlated features with correlation threshold over 0.2 using agglomerative clustering;
* Data oversampling using SMOTE.

The model were trained using stratified k-fold with 10 folds.
The best model in each fold is then used to perform out-of-fold predictions.

#### 3-layer Neural Network with Gaussian Mixture Layer

* Notebook: [exp/14_neuralnetwork_mixture_model.py](https://github.com/beekill95/age_related_conditions/blob/main/exp/14_neuralnetwork_mixture_model.py);
F1 CV: 0.91 ± 0.06 / Log Loss CV: 0.30 ± 0.04 / Public LB: 0.35 / Private LB: 0.54

Upon investigating the performance of the 3-layer neural network model above on 10 folds,
I saw that performance was particularly worst in the fold #6, #7, and #9.
Thus, I hypothesized that the performance of the model might improve
if the embeddings (the output of the last layer) followed mixture gaussian distribution.

Therefore, I implemented a gaussian mixture model that receives class membership of each data point (one-hot encoded)
and predicts the mean and variance of the 64 embedding dimensions.
The mean and the variance vectors are then plugged into the unnormalized PDF along with the embeddings to calculate
the negative log-likelihood.