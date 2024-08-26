# Artificial Intelligence

```{image} ../images/AI_vs_ML_vs_DL.png
:alt: AI_vs_ML_vs_DL
:class: bg-primary mb-1
:width: 800px
:align: center
```


```{image} ../images/AI_vs_ML_vs_DL_Evolution.png
:alt: AI_vs_ML_vs_DL_Evolution
:class: bg-primary mb-1
:width: 800px
:align: center
```


## Datasets

In the context of Machine Learning, the split of our modelling dataset into training and testing samples is probably one of the earliest pre-processing steps that we need to undertake. The creation of different samples for training and testing helps us evaluate model performance.

```{thebe-button}
```
In order to practice some of these functions, we will use the `mpg` dataset from the Seaborn database.

First, let's import all the required libraries:
```{code-block} python
:class: thebe
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

df = sns.load_dataset("mpg")
```

### Overfitting & Underfitting

```{image} ../images/underfitting_Overfitting.png
:alt: underfitting_Overfitting
:class: bg-primary mb-1
:width: 800px
:align: center
```

A very common issue when training a model is **overfitting**. This phenomenon occurs when a model performs really well on the data that we used to train it but it fails to generalise well to new, unseen data points. 

There are numerous reasons why this can happen — it could be due to the noise in data or it could be that the model learned to predict specific inputs rather than the predictive parameters that could help it make correct predictions. Typically, the higher the complexity of a model the higher the chance that it will be overfitted.

Techniques to reduce overfitting :
* Increase the training data.
* Reduce model complexity.
* Early stopping during the training phase (have an eye over the loss over the * training period as soon as loss begins to increase stop training).
* Ridge Regularization and Lasso Regularization
* Use dropout for neural networks to tackle overfitting.

On the other hand, underfitting occurs when the model has poor performance even on the data that was used to train it. 

In most cases, underfitting occurs because the model is not suitable for the problem you are trying to solve. Usually, this means that the model is less complex than required in order to learn those parameters that can be proven to be predictive.

In order to reduce underfitting, we can:
* Increase model complexity
* Increase number of features, performing feature engineering
* Remove noise from the data.
* Increase the number of epochs or increase the duration of training to get better results.

### Train & Test Split

At the moment to train a model we need to have a dataset that we can use to train the model.

```{image} ../images/ML_Model-Data.png
:alt: ML_Model-Data
:class: bg-primary mb-1
:width: 800px
:align: center
```

In our case, we can check the size of the original dataset using the `shape` attribute:
```{code-block} python
:class: thebe
print('Original Dataset size: ', df.shape)
df.info()
```

Our original dataset has `9` columns (features) and `398` rows, corresponding to the columns described in the output of the `df.info()` function.

Then, we need to select which will be our features and which will be our target. In this case, we would like to predict the `mpg` column. For that reason, we will remove the `mpg` column from the dataset, and keep all the other features in the dataset.


```{code-block} python
:class: thebe
X = df.drop(columns=['mpg'])
y = df['mpg'].values

print('X shape: ', X.shape)
X.info()
print('------------------------------')
print('y shape: ', y.shape)
```

As we can see the shape of the `X` and `y` variables are now 398 rows, and 8 columns. While the `y` contains (only) the values corresponding of those 398 rows.

The "Training Dataset" (which in practice is a subset of the original dataset), is the dataset that we initially feed into a machine learning algorithm to "train" the algorithm. It is used exclusively as an input to help the algorithm "learn".

Also, we have a so-called "test" dataset which we will use the test dataset as an input to the model after the model has been built to "test" that the model works as needed. So when we build a machine learning model, we usually need a training dataset and a test dataset.

```{image} ../images/train_test_split.png
:alt: train_test_split
:class: bg-primary mb-1
:width: 800px
:align: center
```

We might split teh dataset into two parts "manually", or we can use the **train test split** function from Sklearn library to split the dataset into two parts, using the following syntax:

```python
from sklearn.model_selection import train_test_split
```

Inside the parenthesis, we'll provide the name of the "X" (which is required) input data as the first argument. This data should contain the feature data,   typically in Numpy array format. However, this function might allow other structures like Python lists. This object should be 2-dimensional, so if it's a 1-dimensional Numpy array, you may need to reshape your data. This is commonly done with the code `.reshape(-1,1).`.

Optionally, we can also provide the name of the "y" dataset that contains the label or target dataset. Also, this is typically a 1-dimensional Numpy array (although the function will allow 2D arrays and lists). In the same way, if it's a 1-dimensional object, it should have the same length as the number of rows in `X`. Or if it's 2D, it should have the same number of rows as `X`.

```{image} ../images/train_test_splitSyntax.png
:alt: train_test_splitSyntax
:class: bg-primary mb-1
:width: 800px
:align: center
```

```{code-block} python
:class: thebe
X_train, X_test, y_train, y_test = train_test_split(X, y) # Optional values train_size and random_state
```

Now, we can check the size of the train and test datasets:
```{code-block} python
:class: thebe
print('Train Dataset size: ', X_train.shape)
print('Test Dataset size: ', X_test.shape)
print('------------------------------')
print('Train Dataset size: ', y_train.shape)
print('Test Dataset size: ', y_test.shape)
```

The `test_size` argument enables us to specify the size of the output test set. The default value is 0.25, which means that 25% of the dataset will be included in the test split. The remaining 75% will be used for the training split.

This parameter can be a float between 0 and 1, representing the proportion of the dataset to include in the test split. If it's an integer, the size of the test set will be equal to that number.

In our case, the train dataset has 298 rows corresponding to the 75% of the dataset, and the test dataset has 100 rows corresponding to the 25% of the dataset.

:::{admonition} Select a `train_size` value
:class: note

We can specify the test size, and check the number of rows in the train and test dataset as follows:

```{code-block} python
:class: thebe
trainSize = 0.8
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = trainSize)

print('Train size parameter: ', trainSize)
print('------------------------------')
print('Train Dataset size: ', X_train.shape)
print('Test Dataset size: ', X_test.shape)
print('------------------------------')
print('Train Dataset size: ', y_train.shape)
print('Test Dataset size: ', y_test.shape)
```
:::

Another important parameter is the `random_state` argument, which controls how the pseudo-random number generator randomly selects observations to go into the training set or test set. If it's not provided, the default value is None.

:::{admonition} Random State Value for Course
:class: note
Notice that if we provide an integer as the argument to this parameter, then train_test_split will shuffle the data in the same order prior to the split, every time you use the function with that same integer.

Since this is a introductory course, we will use the default value in order to have consistent results during our exercises.

:::

First, let's use the `train_test_split` function without specify the `random_state` argument, and plot the first rows of the train and test datasets:

```{code-block} python
:class: thebe
trainSize = 0.8
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = trainSize, random_state = None)

print('Train Dataset: ', X_train.head())
print('Test Dataset: ', X_test.head())
```

Now, let's use a random seed value of `42` and plot the first rows of the train and test datasets again:

```{code-block} python
:class: thebe
trainSize = 0.8
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = trainSize, random_state = 42)

print('Train Dataset: ', X_train.head())
print('Test Dataset: ', X_test.head())
```

:::{admonition} Repeat the exercise many times
:class: note
What do you notice when you run the code above many times? Are the values of the first rows of the train and test datasets different?
:::

## Variance & Bias

**Bias** is the difference between the average prediction of our model and the correct value which we are trying to predict. Model with high bias pays very little attention to the training data and oversimplifies the model. It always leads to high error on training and test data.

**Variance** is the variability of model prediction for a given data point or a value which tells us spread of our data. Model with high variance pays a lot of attention to training data and does not generalize on the data which it hasn’t seen before. As a result, such models perform very well on training data but has high error rates on test data.

```{image} ../images/varianceBias.png
:alt: varianceBias
:class: bg-primary mb-1
:width: 800px
:align: center
```

If our model is too simple and has very few parameters then it may have high bias and low variance. On the other hand if our model has large number of parameters then it’s going to have high variance and low bias. So we need to find the right/good balance without overfitting and underfitting the data.

The **bias-variance tradeoff** in complexity is why there is a tradeoff between bias and variance. An algorithm can’t be more complex and less complex at the same time.

An optimal balance of bias and variance would never overfit or underfit the model. Therefore understanding bias and variance is critical for understanding the behavior of prediction models.

## Cross Validation

Cross-validation is a statistical method used to estimate the skill of machine learning models. It is commonly used in applied machine learning to compare and select a model for a given predictive modeling problem because it is easy to understand, easy to implement, and results in skill estimates that generally have a lower bias than other methods.

Two of the most common uses of cross-validation are:
* k-fold cross-validation
* Stratified k-fold cross-validation

