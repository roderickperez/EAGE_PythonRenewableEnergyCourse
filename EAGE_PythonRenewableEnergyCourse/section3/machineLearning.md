# Machine Learning

```{image} ../images/Supervised_Unsupervised_ML.jpg
:alt: Supervised_Unsupervised_ML
:class: bg-primary mb-1
:width: 800px
:align: center
```
:::{admonition} Sklearn

[Scikit-learn (Sklearn)](https://scikit-learn.org/stable/) is the most useful and robust library for machine learning in Python. It provides a selection of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction via a consistence interface in Python.

```{image} ../images/skLearn.png
:alt: skLearn
:class: bg-primary mb-1
:width: 800px
:align: center
```

* **Supervised Learning algorithms**: Almost all the popular supervised learning algorithms, like Linear Regression, Support Vector Machine (SVM), Decision Tree etc., are the part of scikit-learn.

* **Unsupervised Learning algorithms**: On the other hand, it also has all the popular unsupervised learning algorithms from clustering, factor analysis, PCA (Principal Component Analysis) to unsupervised neural networks.

* **Clustering**: This model is used for grouping unlabeled data.

* **Cross Validation**: It is used to check the accuracy of supervised models on unseen data.

* **Dimensionality Reduction**: It is used for reducing the number of attributes in data which can be further used for summarisation, visualisation and feature selection.

* **Ensemble methods*: As name suggest, it is used for combining the predictions of multiple supervised models.

* **Feature extraction**: It is used to extract the features from data to define the attributes in image and text data.

*  **Feature selection**: It is used to identify useful attributes to create supervised models.


We can install and import `scikit-learn` by using the following command:

```python
pip install scikit-learn
import sklearn
```
:::


```{table} Supervised vs Unsupervised Learning
:name: supervised-vs-unsupervised-learning

|                          | **Supervised** | **Unsupervised** |
|--------------------------|:--------------:|:----------------:|
| Input Data               |  Labeled Data  |  Unlabelled Dat  |
| Computational Complexity |     Simple     |      Complex     |
| Accuracy                 |      High      |        Low       |
```

## Supervised Learning

Supervised Learning consist to learns a function to make prediction of a defined label based on the input data. It can be either classifying data into a category (classification problem) or forecasting an outcome (regression algorithms).

Supervised learning can be furthered categorized into classification and regression algorithms. Classification model identifies which category an object belongs to whereas regression model predicts a continuous output.

### Regression

Regression is a technique to find the relationship between independent variables or features ($X$) and a dependent variable or outcome ($y$). It's used as a method for predictive modelling in Machine Learning, in which an algorithm is used to predict continuous outcomes.

#### Linear Regression
The linear regression algorithm is a supervised learning algorithm used in machine learning. Its fundamental basis is to try to find a straight line that indicates the trend of a set of continuous data, where

$$y = mX + b$$

where $m$ is the slope and $b$ is the y-intercept. 

Statistically, linear regression is an approximation to model the linear relationship between a dependent variable ($y$), and one (or more) independent (descriptive) variables that are grouped into $X$.

:::{admonition} Logistic Regression
:class: tip
In the case that the data is discrete, we must use logistic regression.
:::

In order for the algorithm to learn by itself the values of the slope (m) and the intercept (b) of the line, the algorithm measures the error with respect to the input points and the "y" value of the real output. . The algorithm should minimize the cost of the function of a squared error function, and those coefficients will correspond to the optimal line.

There are various methods that allow us to minimize the cost function, the most common being to use a vector version.

```{thebe-button}
```

First, let's import all the required libraries:
```{code-block} python
:class: thebe
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
```

And, let's select the `mpg` dataset from the Seaborn database.
```{code-block} python
:class: thebe
df = sns.load_dataset("mpg")
df.head()
```

:::{admonition} EDA
:class: warning
Since we are in an introductory course we are going to skip the EDA steps as we assume the data is consistent and error free. However, in a real life case, do not forget that an EDA process must be carried out before applying any Machine Learning algorithm.
:::

We are now going to select two of the variables from this dataset. For example, let's say we want to predict `mpg` ($y$) from the values of the `weight` ($X$) variable.

```{code-block} python
:class: thebe
X = df["weight"].values.reshape(-1, 1)
y = df["mpg"].values.reshape(-1, 1)

plt.figure(figsize=(10,6))
plt.scatter(X,y)
plt.xlabel("weight", fontsize=18)
plt.ylabel("mpg", fontsize=18)
plt.title('Linear Regression', fontsize=24)
plt.grid()
plt.show()
```

:::{admonition} Linear Regression
:class: warning
Notice that when we select the `X` and `y` variables, we needed to reshape them into a 2D array.

If we don't do this, Python will return the following error:
```python
ValueError: Expected 2D array, got 1D array instead:
array=[ ... ]
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
```
:::

```{code-block} python
:class: thebe
linearRegression = linear_model.LinearRegression()
```

The first thing we need to do is create a linear regression object, using the sklearn library

```{code-block} python
:class: thebe
linearRegression = linear_model.LinearRegression()
```

```{code-block} python
:class: thebe
linearRegression.fit(X,y)
y_pred = linearRegression.predict(X)
```

After making the prediction, we can then see the slope $m$ and the value where the line intersects the Y-axis (b).

```{code-block} python
:class: thebe
print('Slope: %.3f' % linearRegression.coef_[0])
print('Intercept: %.3f' % linearRegression.intercept_)
```

In order to know the root mean square error of our prediction (`y_pred`) and the real values of our data set (`y`), we use one of the sklearn functions called `sklearn.metrics.mean_squared_error`.

```{code-block} python
:class: thebe
error = mean_squared_error(y, y_pred)
print('Error: %.3f' % error)
```

In the same way, we can calculate the value of the variance, which should be close to 1 if the prediction is good, using the function `sklearn.metrics.r2_score`.

```{code-block} python
:class: thebe
var = r2_score(y, y_pred)
print('Variance: %.3f' % var)
```

Now, with the values of the slope and the intercept, we can plot the regression line
```{code-block} python
:class: thebe
plt.figure(figsize=(10,6))
plt.scatter(X,y)
plt.plot(X, y_pred, color="red", linewidth=3)
plt.xlabel("weight", fontsize=18)
plt.ylabel("mpg", fontsize=18)
plt.title('Linear Regression', fontsize=24)
plt.grid()
plt.show()
```

Also, we can make predictions for new values of the independent variable.
```{code-block} python
:class: thebe
inputValue = float(input("Enter a value for the weight (independent variable): "))
userPediction = linearRegression.predict([[inputValue]])

plt.figure(figsize=(10,6))
plt.scatter(X,y)
plt.plot(X, y_pred, color="red", linewidth=3)
plt.axvline(x=inputValue, color="green", linewidth=2, linestyle="--")
plt.axhline(y=userPediction, color="green", linewidth=2, linestyle="--")
plt.xlabel("weight", fontsize=18)
plt.ylabel("mpg", fontsize=18)
plt.title('Linear Regression', fontsize=24)
plt.grid()
plt.show()


print("The prediction is: %.3f" % userPediction)
```

Now, we are going to repeat the exercise but at this time we will use `train_test_split` to split the data into a training set and a test set.

```{code-block} python
:class: thebe
from sklearn.model_selection import train_test_split

df = sns.load_dataset("mpg")
X = df["weight"].values.reshape(-1, 1)
y = df["mpg"].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (24, 8))
ax[0].scatter(X, y, color = 'blue')
ax[0].set_title('Original data', fontsize=24)
ax[0].set_xlabel("weight", fontsize=18)
ax[0].set_ylabel("mpg", fontsize=18)

ax[1].scatter(X_train, y_train, color = 'red')
ax[1].set_title('Training data', fontsize=24)
ax[1].set_xlabel("weight", fontsize=18)
ax[1].set_ylabel("mpg", fontsize=18)

ax[2].scatter(X_test, y_test, color = 'green')
ax[2].set_title('Test data', fontsize=24)
ax[2].set_xlabel("weight", fontsize=18)
ax[2].set_ylabel("mpg", fontsize=18)
```

Then, instead of train the model with all the data, we will train the model with the training set.

```{code-block} python
:class: thebe
linearRegression_ = linear_model.LinearRegression()

linearRegression_.fit(X_train,y_train)
y_pred_ = linearRegression_.predict(X_train)
```

And, now we can calculate the error and the variance.

```{code-block} python
:class: thebe
print('Slope: %.3f' % linearRegression_.coef_[0])
print('Intercept: %.3f' % linearRegression_.intercept_)

print('----------------------------------------------------')
error_ = mean_squared_error(y_train, y_pred_)
print('Error: %.3f' % error_)
print('----------------------------------------------------')
var_ = r2_score(y_train, y_pred_)
print('Variance: %.3f' % var_)
```

Now, with the values of the slope and the intercept, we can plot the regression line and compare the results when we use all the data and the results when we use the training set.
```{code-block} python
:class: thebe
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (24, 8))
ax[0].scatter(X, y, color = 'blue')
ax[0].plot(X, y_pred, color="red", linewidth=3)
ax[0].set_title('Original data', fontsize=24)
ax[0].set_xlabel("weight", fontsize=18)
ax[0].set_ylabel("mpg", fontsize=18)

ax[1].scatter(X_train, y_train, color = 'blue')
ax[1].plot(X_train, y_pred_, color="red", linewidth=3)
ax[1].set_title('Train data', fontsize=24)
ax[1].set_xlabel("weight", fontsize=18)
ax[1].set_ylabel("mpg", fontsize=18)

```

Now, we can use the test set to test the model.

```{code-block} python
:class: thebe
X_test_ = X_test.reshape(-1, 1)
y_test_ = y_test.reshape(-1, 1)
y_pred_test = linearRegression_.predict(X_test_)

print('----------------------------------------------------')
error_test = mean_squared_error(y_test_, y_pred_test)
print('Error: %.3f' % error_test)
print('----------------------------------------------------')
var_test = r2_score(y_test_, y_pred_test)
print('Variance: %.3f' % var_test)
```

Finally, we can show the results of using the training and test dataset, compared to use the full dataset in the following table:
```{table} All Data vs. Training Dataset
:name: all-vs-training-dataset

|                    | **All Data** | **Training Dataset** | **Test Dataset** |
|:------------------:|:------------:|:--------------------:|:----------------:|
| **Slope (m)**      |    -0.008    |        -0.008        |                  |
| **Intercept (b)**  |    46.317    |        46.782        |                  |
| **Error (r)**      |    18.781    |        19.782        |      14.895      |
| **Varainace (r2)** |     0.692    |         0.684        |       0.723      |
```

Notice that the error is larger and variance is lower in the training set since the model is trained with only 80% of the original data.

:::{admonition} Other Advanced Regression Algorithms
:class: note

* Lasso Regression: Lasso regression is a variation of linear regression with L1 regularization.

$$\sum_{i=1}^{n} (y_{i}-\hat{y}_i)^2 + \lambda |b_i|$$

* Ridge Regression: Ridge regression is a variation of linear regression with L2 regularization.

$$\sum_{i=1}^{n} (y_{i}-\hat{y}_i)^2 + \lambda |b_i|^2$$
:::

#### Multilinear Regression
Now, we are going to extend the exercise of predicting our dependent variable and with more than one input variable for the model. With this, the idea is to have a greater power to the model since we are incorporating more variables to the model. Now, our equation of the line becomes:

$$y = b + m_{1}X_{1}+m_{2}X_{2}+m_{3}X_{3}+...+m_{n}X_{n}$$

For this simple exercise, we are going to use 2 predictor variables to be able to plot the result in a 3D visualization. It is important to note that in 3D, the result is not a line but a plane.

```{code-block} python
:class: thebe
from sklearn.model_selection import train_test_split

df = sns.load_dataset("mpg")
XY = df[["weight", "displacement"]].values
z = df["mpg"].values.reshape(-1, 1)

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (24, 8))
ax[0].scatter(XY[:, 0], z, color = 'blue')
ax[0].set_title('Weight vs MPG', fontsize=24)
ax[0].set_xlabel("weight", fontsize=18)
ax[0].set_ylabel("mpg", fontsize=18)

ax[1].scatter(XY[:, 1], z, color = 'red')
ax[1].set_title('Displacement vs MPG', fontsize=24)
ax[1].set_xlabel("displacement", fontsize=18)
ax[1].set_ylabel("mpg", fontsize=18)
```

Following the same scheme as in the case of linear regression, we first divide the data into training and test data, then we define a linear regression object, which we are going to train. The result of this process are 2 coefficients (which will help us to graph the plane), corresponding to each of our predictive variables. Then we make the prediction with which we will have points on the found plane.

```{code-block} python
:class: thebe
XY_train, XY_test, z_train, z_test = train_test_split(XY, z, test_size=0.2, random_state=42)

multiLinearRegression = linear_model.LinearRegression()

multiLinearRegression.fit(XY_train,z_train)
z_pred = multiLinearRegression.predict(XY_train)

print('Coeffients: ', multiLinearRegression.coef_)
print('Intercept: ', multiLinearRegression.intercept_)
print('----------------------------------------------------')
error = mean_squared_error(z_train, z_pred)
print('Error: %.3f' % error)
print('----------------------------------------------------')
var = r2_score(z_train, z_pred)
print('Variance: %.3f' % var)

```

We can visualize the results comparing the original data and the training data.

```{code-block} python
:class: thebe

print('Slope X_1: %.3f' % multiLinearRegression.coef_[0][0])
print('Slope X_2: %.3f' % multiLinearRegression.coef_[0][1])
print('Intercept: %.3f' % multiLinearRegression.intercept_)



fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (24, 8), sharey = True)
ax[0].scatter(XY_train[:, 0], z_train, color = 'blue', label = 'Original data')
ax[0].scatter(XY_train[:, 0], z_pred, color="red", label = 'Predicted output')
ax[0].plot(XY_train[:, 0], multiLinearRegression.intercept_+multiLinearRegression.coef_[0][0]*XY_train[:, 0], color="red", linewidth=3)

ax[0].set_title('Weight vs MPG', fontsize=24)
ax[0].set_xlabel("weight", fontsize=18)
ax[0].set_ylabel("mpg", fontsize=18)
ax[0].legend()

ax[1].scatter(XY_train[:, 1], z_train, color = 'green', label = 'Original data')
ax[1].scatter(XY_train[:, 1], z_pred, color="red", label = 'Predicted output')
ax[1].set_title('Displacement vs MPG', fontsize=24)
ax[1].set_xlabel("displacement", fontsize=18)
ax[1].plot(XY_train[:, 1], multiLinearRegression.intercept_+multiLinearRegression.coef_[0][1]*XY_train[:, 1], color="red", linewidth=3)
ax[1].legend()
```

In order to visualize the plane we must create a mesh on which we will graph said plane, and then we will calculate the values of the plane for the points `x` and `y`. Next, we calculate the values for z, adding the intercept_ value found.

```{code-block} python
:class: thebe

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(20,20))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)


# Create a meshgrid of x and y values
xx, yy = np.meshgrid(np.arange(min(XY_train[:, 0]), max(XY_train[:, 0]), 1), np.arange(min(XY_train[:, 1]), max(XY_train[:, 1]), 1))

# Calculate z values using the intercept value
z = multiLinearRegression.coef_[0][0]*xx + multiLinearRegression.coef_[0][1]*yy + multiLinearRegression.intercept_

# Plot the plane
ax.plot_surface(xx, yy, z, color='r', alpha=0.2)

# Plot points 
ax.scatter(XY_train[:, 0], XY_train[:, 1], z_train, color='blue', s = 40) # Training data
ax.scatter(XY_train[:, 0], XY_train[:, 1], z_pred, color='red', s = 40) # Predicted data

# Modify camera view
ax.view_init(elev=20., azim=150)

# Set Axis labels and Title
ax.set_xlabel('weight', fontsize=18)
ax.set_ylabel('displacement', fontsize=18)
ax.set_zlabel('mpg', fontsize=18)
ax.set_title('Multilinear Regression', fontsize=24)
```

#### Neural Network*

The neural network is a powerful model that can be used to predict the output of a system. However, in this course we will cover this topic in the a separate section.

### Classification

In machine learning, classification refers to a predictive modeling problem where a class label is predicted for a given example of input data. 

#### Support Vector Machine

Support Vector Machines are powerful and versatile machine learning algorithms that can be used for both classification and regression. The usage area of these algorithms is quite wide. It is dynamically developed and used in many fields from images classification to medical image cancer diagnosis, from text data to bioinformatics. 

SVMs are all about defining a decision boundar, which it can defined as a hyperplane (or a line if in 2 dimensions) where the line decides what class the data belongs to depending on what side it is on.

```{image} ../images/SVM.png
:alt: SVM
:class: bg-primary mb-1
:width: 800px
:align: center
```

In SVM we want to find the decision boundary that **maximises** the distance from the closest points to the boundary of each class. To achieve this goal first we need to take an initial boundary, and find the closest points to it from either class. The closest points to the decision boundary are called **support vectors**, and they are they only points that affect the decision boundary. Once we have the support vectors, we find the line that is furthest away from these support vectors.

First, let's load the `sklearn.datasets import make_blobs` module, which will allow us to create a synthetic dataset with 2 classes, and plot them in a plot.

```{code-block} python
:class: thebe

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

# Create a dataset with 2 classes
X, y = make_blobs(n_samples=100, centers=2, cluster_std=2, random_state=11)

# Plot the data
plt.figure(figsize = (16, 12))
plt.xlabel("$X_1$", fontsize=18)
plt.ylabel("$X_2$", fontsize=18)
plt.scatter(X[:, 0], X[:, 1], c=y, s=100)
plt.grid()
plt.show()
```

As we saw in the previous section, we can split the data in train and test  datasets, and plot them in three different plots to compare.

```{code-block} python
:class: thebe
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=32)

fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (24, 8))
ax[0].scatter(X[:, 0], X[:, 1], c=y, s=100)
ax[0].set_title('Original data', fontsize=24)
ax[0].set_xlabel("$X_1$ (original)", fontsize=18)
ax[0].set_ylabel("$X_2$ (original)", fontsize=18)
ax[0].grid()

ax[1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=100)
ax[1].set_title('Training data (80%)', fontsize=24)
ax[1].set_xlabel("$X_1$ (train) ", fontsize=18)
ax[1].set_ylabel("$X_2$ (train)", fontsize=18)
ax[1].grid()

ax[2].scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=100)
ax[2].set_title('Test data (20%)', fontsize=24)
ax[2].set_xlabel("$X_1$ (test)", fontsize=18)
ax[2].set_ylabel("$X_2$ (test)", fontsize=18)
ax[2].grid()
```


Then, we can construct a decision boundary with the train dataset, and then we can and plot the vector which separates the two classes.

```{code-block} python
:class: thebe

from sklearn.svm import SVC

svc_model = SVC(kernel='linear', random_state=32)
svc_model.fit(X_train, y_train)

w = svc_model.coef_[0]
b = svc_model.intercept_[0]

x_points = np.linspace(-10, 4)    # generating x-points from -1 to 1
y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points

plt.figure(figsize = (16, 12))
plt.plot(x_points, y_points, c='r')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=100)
plt.xlabel("$X_1$", fontsize=18)
plt.ylabel("$X_2$", fontsize=18)
plt.grid()
plt.show()
```

```{image} ../images/SVM1.png
:alt: SVM1
:class: bg-primary mb-1
:width: 800px
:align: center
```

Notice that the equation of the line $ax_{1}+bx_{2}$ can be rewritten in a vector form:

$$y(x) = w^{T}x+b$$

where $w$ is the vector of weights, and $b$ is the bias.

:::{admonition} Multidimensional SVM
:class: note
In case we have multidimensional $n$ features, the equation expands to

$$y(x) = w^{T}x_{1}+w^{T}x_{2}+w^{T}x_{3}+...+w^{T}x_{n}+b$$

:::

In this plot the red line correspond to the decision boundary (hyperplane) of the SVM. The distance between a point and a line can be calculated by 

$$\frac{w^{T}x+b}{||w||} = \frac{ax_{1}+bx_{2}+c}{\sqrt{a^{2}+b^{2}}} $$

In case a point is below the line, that distance will be negative, and if the point was on top of the line, the distance would be positive.

In order to find the "support vectors" of the SVM, which we can interpret as the decision boundary line translated along direction of vector w by the distance equals margin. In order to find these support vectors, we can use the `sklearn.svm.SVC.support_vectors_` attribute. This function first, find a normal vector to the decision boundary, then, calculate a unit vector of that normal vector ($\hat{w}$). Then, we need to get a distance between the lines (margin), and finally translate all points of the decision boundary to a new location by this formula:

$$abovePoints_{new} = hyperplane_{points} + \hat{w} * margin$$

$$belowPoints_{new} = hyperplane_{points} - \hat{w} * margin$$

We know that $w$ is a normal vector we need — it’s always orthogonal to a hyperplane. In order to calculate the unit vector of $w$, which is equal to:

$$\hat{w} = \frac{w}{||w||}$$

We can use the `sklearn.svm.SVC.coef_` attribute whcih returns the coefficients of the support vectors, such as:

```python
w_hat = svc_model.coef_[0] / (np.sqrt(np.sum(svc_model.coef_[0] ** 2)))
```

Now, we can calculatethe distance between the lines (margin)

$$margin_{magnitude} = \frac{1}{||w||}$$

```python
margin = 1 / np.sqrt(np.sum(svc_model.coef_[0] ** 2))
```

and finally translate all points of the decision boundary to a new location by this formula:

```python
new_points_up   = hyperplane_points + w_hat * margin
new_points_down = hyperplane_points - w_hat * margin
```

```{code-block} python
:class: thebe

plt.figure(figsize = (16, 12))

# 1) Encircle support vectors
plt.scatter(svc_model.support_vectors_[:, 0],
            svc_model.support_vectors_[:, 1], 
            s=100, 
            facecolors='none', 
            edgecolors='red', 
            alpha=.5,
            linewidth = 4)

# 2) Calculate unit-vector:
w_hat = svc_model.coef_[0] / (np.sqrt(np.sum(svc_model.coef_[0] ** 2)))

# 3) Calculate margin:
margin = 1 / np.sqrt(np.sum(svc_model.coef_[0] ** 2))

# 4) Calculate points of the margin lines:
decision_boundary_points = np.array(list(zip(x_points, y_points)))
points_of_line_above = decision_boundary_points + w_hat * margin
points_of_line_below = decision_boundary_points - w_hat * margin

# Plot

plt.plot(x_points, y_points, c='r') # Decision Boundary
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=100) # Training Data

# Plot margin lines
plt.plot(points_of_line_above[:, 0], points_of_line_above[:, 1], 'b--', linewidth=2) # Blue margin line above

plt.plot(points_of_line_below[:, 0], points_of_line_below[:, 1], 'g--', linewidth=2) # Green margin line below

plt.xlabel("$X_1$", fontsize=18)
plt.ylabel("$X_2$", fontsize=18)

plt.grid()
plt.show()
```

:::{admonition} Soft-margin SVM
:class: note
In this brief introduction we have assumed the data is linearly separable in the feature space. However, a lot of the time this is not a good assumption. In order to deal with this one can implement a soft-margin SVM.

Soft-margin SVMs allow some of the data to live inside the margin, while applying a small penalty. The derivation for the soft-margin SVM is similar, and introduces the use of a slack variable as a penalty for the number of points inside the margin. The send result is also very similar, with an identical Lagrangian but different constraints.
:::

Imagine that you want to find to which group belows a new set of data points, for example with coordinates `x_1 = 0.0` and `y_1 = -5.0`, and . `x_2 = -5.0` and `y_2 = 5.0`

:::{admonition} Sklearn and Numpy
:class: note
Remember that Sklearn requires that we input the data in the form of a numpy array.
:::

```{code-block} python
:class: thebe
newPoint1 = np.array([[0.0, -5.0]])
newPoint2 = np.array([[-5.0, 5.0]])

# Plot

plt.figure(figsize = (16, 12))
plt.plot(x_points, y_points, c='r') # Decision Boundary
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=100) # Training Data
plt.plot(points_of_line_above[:, 0], points_of_line_above[:, 1], 'b--', linewidth=2) # Blue margin line above
plt.plot(points_of_line_below[:, 0], points_of_line_below[:, 1], 'g--', linewidth=2) # Green margin line below
plt.xlabel("$X_1$", fontsize=18)
plt.ylabel("$X_2$", fontsize=18)

# New point
plt.scatter(newPoint1[:, 0], newPoint1[:, 1], c='orange', s=150)
plt.scatter(newPoint2[:, 0], newPoint2[:, 1], c='magenta', s=150)

plt.grid()
plt.show()
```

Using the fitted model we can classify these points by calling the `predict` function.

```{code-block} python
:class: thebe
print('Class corresponding to Point 1: ', svc_model.predict(newPoint1))
print('Class corresponding to Point 2: ',svc_model.predict(newPoint2))
```

This is similar to calculate the dot product of the new point and adding the bias term.

```{code-block} python
:class: thebe
result = np.dot(svc_model.coef_[0], newPoint1[0]) + svc_model.intercept_


if result <= 0:
    print('Point 1 correspond to Class 0')
else:
    print('Point 1 correspond to Class 1')

```

#### K-Nearest Neighbours
The k-Nearest Neighbors algorithm (KNN) is a very simple classification (and regression) algorithm. It is based on the idea that the observations closest to a given data point are the most "similar" observations in a data set, and we can therefore classify unforeseen points based on the values of the closest existing points. By choosing K, the user can select the number of nearby observations to use in the algorithm. From these neighbors, a summarized prediction is made. Once the neighbors are discovered, the summary prediction can be made by returning the most common outcome or taking the average. 


```{image} ../images/KNN_Distances.png
:alt: KNN_Distances
:class: bg-primary mb-1
:width: 800px
:align: center
```

Euclidean distance is the most common metric used, and is derived from the Pythagorean theorem. Euclidean distance simply refers to the distance between two points. The formula for calculating Euclidean distance:

$$d(x, y)=\sqrt{\sum_{i=1}^{n}(x_{i}-y_{i})^{2}}$$

In the Eucledian distance, for each dimension, we calculate the length of that side in the triangle by subtracting a point’s value from the other’s. Then, square and add it to the running total. The Euclidean distance is the square root of the running total.

In case in the Manhattan distance differs from Euclidean distance when we calculate the difference between two points by using the absolute value of the difference. It describes the distance between point x and point y equals the sum of the absolute differences of the Y value subtracted from the X value in each dimension. 

$$d(x, y)=\sum_{i=1}^{n}|x_{i}-y_{i}|$$

In order to find to which group belows a new set of data point belongs, we need to find first calculate Euclidean Distance, then get the Nearest Neighbors, and finally make a prediction.

:::{admonition} Lazy Learning Method?
:class: note
Since no work is done until a prediction is required, KNN is often referred to as a lazy learning method.
:::

Let's use the `sklearn.datasets import make_blobs` module to create a simple synthetic dataset with 2 classes, and plot them in a plot.

```{code-block} python
:class: thebe

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

# Create a dataset with 2 classes
X, y = make_blobs(n_samples=100, centers=2, cluster_std=3.5, random_state=11)

# Plot the data
plt.figure(figsize = (16, 12))
plt.xlabel("$X_1$", fontsize=18)
plt.ylabel("$X_2$", fontsize=18)
plt.scatter(X[:, 0], X[:, 1], c=y, s=100)
plt.grid()
plt.show()
```

Now, let's evaluate the KNN algorithm calulating the distance between the new point and the existing points in the dataset, for example `x = -1.0` and `y = -2.0`.

```{code-block} python
:class: thebe

newPoint = np.array([[-2.5, -1.0]])

# Plot the data
plt.figure(figsize = (16, 12))
plt.scatter(X[:, 0], X[:, 1], c=y, s=100)
plt.scatter(newPoint[:, 0], newPoint[:, 1], c='lime', s=250, edgecolors='red', alpha=.5, linewidth = 4)
plt.xlabel("$X_1$", fontsize=18)
plt.ylabel("$X_2$", fontsize=18)
plt.grid()
plt.show()
```

```{image} ../images/knn1.png
:alt: knn1
:class: bg-primary mb-1
:width: 800px
:align: center
```

We can use the `KNeighborsClassifier` module to find the nearest neighbors. In this case, we need to specify the number of neighbors to use, by passing the `n_neighbors` parameter.

```{code-block} python
:class: thebe
from sklearn.neighbors import KNeighborsClassifier

# KNN = 1
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X, y)
prediction_knn1 = knn1.predict(newPoint)

# KNN = 2
knn2 = KNeighborsClassifier(n_neighbors=2)
knn2.fit(X, y)
prediction_knn2 = knn2.predict(newPoint)

# KNN = 3
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X, y)
prediction_knn3 = knn3.predict(newPoint)

fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize=(16, 6))

ax[0].scatter(X[:, 0], X[:, 1], c=y, s=100)
ax[0].scatter(newPoint[:, 0], newPoint[:, 1], c=prediction_knn1[0], s=250, edgecolors='red', alpha=.5, linewidth = 4)
ax[0].text(x=newPoint[:, 0]+1.7, y=newPoint[:, 1]-0.7, s=f"Class: {prediction_knn1[0]}", fontsize=22)
ax[0].set_xlabel("$X_1$", fontsize=18)
ax[0].set_ylabel("$X_2$", fontsize=18)
ax[0].grid()
ax[0].set_title("KNN = 1", fontsize=24)

ax[1].scatter(X[:, 0], X[:, 1], c=y, s=100)
ax[1].scatter(newPoint[:, 0], newPoint[:, 1], c=prediction_knn2[0], s=250, edgecolors='red', alpha=.5, linewidth = 4)
ax[1].text(x=newPoint[:, 0]+1.7, y=newPoint[:, 1]-0.7, s=f"Class: {prediction_knn2[0]}", fontsize=22)
ax[1].set_xlabel("$X_1$", fontsize=18)
ax[1].set_ylabel("$X_2$", fontsize=18)
ax[1].grid()
ax[1].set_title("KNN = 2", fontsize=24)

ax[2].scatter(X[:, 0], X[:, 1], c=y, s=100)
ax[2].scatter(newPoint[:, 0], newPoint[:, 1], c=prediction_knn3[0], s=250, edgecolors='red', alpha=.5, linewidth = 4)
ax[2].text(x=newPoint[:, 0]+1.7, y=newPoint[:, 1]-0.7, s=f"Class: {prediction_knn3[0]}", fontsize=22)
ax[2].set_xlabel("$X_1$", fontsize=18)
ax[2].set_ylabel("$X_2$", fontsize=18)
ax[2].grid()
ax[2].set_title("KNN = 3", fontsize=24)
```

:::{admonition} Optimal value of $K$ 

In order to find the optimal value of $K$ in our dataset, let's use a new (and more complex) dataset. First we need to split the dataset in a training and a test set.

```{code-block} python
:class: thebe

X, y = make_blobs(n_samples = 1000, centers = 5, cluster_std = 5, random_state = 11)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize=(16, 6), sharey = True)

ax[0].scatter(X[:, 0], X[:, 1], c=y, s=100)
ax[0].set_xlabel("$X_1$", fontsize=18)
ax[0].set_ylabel("$X_2$", fontsize=18)
ax[0].grid()
ax[0].set_title("Original Dataset", fontsize=18)

ax[1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=100)
ax[1].set_xlabel("$X_1$", fontsize=18)
ax[1].grid()
ax[1].set_title("Train Dataset (80%)", fontsize=18)

ax[2].scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=100)
ax[2].set_xlabel("$X_1$", fontsize=18)
ax[2].grid()
ax[2].set_title("Test Dataset (20%)", fontsize=18)
```

Then, we can use a `for` loop to iterate through all the values of $K$ and find the best one.

```{code-block} python
:class: thebe
from sklearn.metrics import accuracy_score

knn_acc = []

for i in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    knn_acc.append(accuracy_score(y_test, knn_pred))


plt.figure(figsize= (16, 8))
plt.plot(knn_acc, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.scatter(knn_acc.index(max(knn_acc)), max(knn_acc), c='lime', s=250, edgecolors='red', alpha=.5, linewidth = 4)

plt.title('Accuracy Rate vs. K Value', fontsize=24)
plt.xlabel('K', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
plt.grid()
print("Maximum Accuracy: ", max(knn_acc),"at K =", knn_acc.index(max(knn_acc)))
```
:::


#### Decision Tree

Decision Tree Classifier is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. In decision tree classifier, the dependent variable is a binary variable that contains data coded as 1 (yes, success, etc.) or 0 (no, failure, etc.). In other words, the decision tree classifier model predicts $P(Y=1)$ as a function of $X$.

In Decision Trees we will fine the following important terms:

* **Root Node**: It represents the entire population or sample and this further gets divided into two or more homogeneous sets.
* **Splitting**: It is a process of dividing a node into two or more sub-nodes.
* **Decision Node**: When a sub-node splits into further sub-nodes, then it is called the decision node.
* **Leaf / Terminal Node**: Nodes do not split is called Leaf or Terminal node.
* **Pruning**: When we remove sub-nodes of a decision node, this process is called pruning. You can say the opposite process of splitting.
* **Branch / Sub-Tree**: A subsection of the entire tree is called branch or sub-tree.
* **Parent and Child Node**: A node, which is divided into sub-nodes is called a parent node of sub-nodes whereas sub-nodes are the child of a parent node.


Some of the advantages of Decision Tree Algorithm is that it is very easy to implement and it is very efficient since the provided the parameters are tuned optimally. Disregards features that are of little or no importance in prediction, and it is inexpensive to construct with an easy to interpret logic.

However, the disadvantages of Decision Tree Algorithm are that it often tend to overfit the training data. Also, changes in data may lead to unnecessary changes in the result. Even that might be easy to understand, but it is not always easy to implement since large trees can sometimes be very difficult to interpret. Finally, it is biased toward splits on features having a number of levels.

:::{admonition} Decision Tree Algorithm and PCA
:class: note
Since a Decision tree classifier tends to overfit in most cases, it is advantageous to replace a Decision Tree classifier with Principal Component Analysis for datasets with a large number of features.
:::

We are going to use the `sklearn.datasets import make_blobs` module to create a simple synthetic dataset with 2 classes, and plot them in a plot. In this case, the data is not linearly separable as before, and we will have 4 classes.

```{code-block} python
:class: thebe

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

# Create a dataset with 2 classes
X, y = make_blobs(n_samples=250, centers=4, cluster_std=3.5, random_state=11)

# Plot the data
plt.figure(figsize = (16, 8))
plt.xlabel("$X_0$", fontsize=18)
plt.ylabel("$X_1$", fontsize=18)
plt.scatter(X[:, 0], X[:, 1], c=y, s=100)
plt.grid()
plt.show()
```

First, we have to created the decision tree classifier by passing other parameters such as `max_depth`, and `min_sample_leaf` to `DecisionTreeClassifier()`. Finally, we do the training process by using the `model.fit()` method.

```{code-block} python
:class: thebe
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

clf_tree = DecisionTreeClassifier(criterion="gini", random_state=42,max_depth=3, min_samples_leaf=5)   
clf_tree.fit(X, y)
```

And we can plot the tree structure by using the `plot_tree` method.

```{code-block} python
:class: thebe
plt.figure(figsize=(16, 12))
tree.plot_tree(clf_tree);
```

In addition to adding the code to allow you to save your image, the code below tries to make the decision tree more interpretable by adding in feature and class names (as well as setting `filled = True`).

```{code-block} python
:class: thebe
plt.figure(figsize=(16, 12))
tree.plot_tree(clf_tree, filled = True);
```

In the next imaga we can see the decision tree structure, and the leaf nodes. Here we are highlingting that the first node is the Root Node, which represents the entire population (250 samples) and this further gets divided into two or more homogeneous sets. As we can see in the table the data is balanced.

```{image} ../images/decisionTree1.png
:alt: decisionTree1
:class: bg-primary mb-1
:width: 800px
:align: center
```

Gini Inndex corresponds to probability of an object being classified to a specific class. It can be understood as the measure of purity, or as he proportion of the samples that belongs to a class for a particular node. The smaller the value of the index, more purity there is in the node. It is calculated following this formula:

$$Gini_{Index} = 1 - \sum_{i = 1}^{n} (p)^{2}$$

In this case, the Gini Index is 75% since if we calculate to being classified as Class 0 is $63/250 = 0.25$. Using the formula above, we obtain that the Gini Index is 0.75.

In the first split, we can see that the data is split into two sub-nodes, starting from the Root Node, asking the question, "Is the data in the first sub-node less than or equal to -7.509?" If the answer is yes, then the data is in the first sub-node, otherwise the data is in the second sub-node. In the group which is less than -7.509 we have 51 samples, and in the group which is larger than -7.509 we have 199 samples. From the 51 samples, 50 belongs to the Class 0, and one belongs to the Class 1. 


```{image} ../images/decisionTree2.png
:alt: decisionTree2
:class: bg-primary mb-1
:width: 800px
:align: center
```

At the next split, at -9.72 in $X_{0}$, we have 8 samples which as less that this value which 7 samples belongs to Class 0 and 1 sample belong to Class 3. On the other hand, 43 samples are larger than this value, where all belong to Class 0.

```{image} ../images/decisionTree3.png
:alt: decisionTree3
:class: bg-primary mb-1
:width: 800px
:align: center
```
Next, the algorithm split the data into two sub-nodes, at -6.951 in $X_{0}$, we have 56 samples which as less that this value which 5 samples belongs to Class 0, 2 samples to Class 1, 4 to Class 2 and 46 samples belong to Class 3. In this case, 143 samples are not larger than -6.951 in $X_{0}$, which 8 samples correspondes to Class 0, 61 samples to Class 1, 59 to Class 2 and 15 samples belong to Class 3.

```{image} ../images/decisionTree4.png
:alt: decisionTree4
:class: bg-primary mb-1
:width: 800px
:align: center
```

If we continue with the analysis, we might be able to end with the following classification:

```{image} ../images/decisionTree5.png
:alt: decisionTree5
:class: bg-primary mb-1
:width: 800px
:align: center
```

You might be wondering how to select among all the possible parameters. One simple solution is the train many models, and select the one with the highest accuracy. For that, we will create a list with the range of `max_depth` that we would like to evaluate. Then, using a for loop we will plot the results in order to take an informed solution, using the train and test dataset.

```{code-block} python
:class: thebe

from sklearn.model_selection import train_test_split

max_depth_range = list(range(1, 20))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

accuracy = []

for max_depth_tree in max_depth_range:
    DecisionTree_classifier = DecisionTreeClassifier(max_depth=max_depth_tree, 
                                                     random_state = 0)
    DecisionTree_classifier.fit(X_train, y_train)
    score = DecisionTree_classifier.score(X_test, y_test)
    accuracy.append(score)

plt.figure(figsize=(16, 8))
plt.plot(max_depth_range, accuracy)
plt.scatter(9, max(accuracy), s = 200, c = 'red', alpha = 0.5)
plt.grid()
plt.xlabel("Max Depth", fontsize = 18)
plt.ylabel("Accuracy", fontsize = 18)
plt.show()
```

## Unsupervised Learning

Unsupervised learning is a type of algorithm that learns patterns from unlabelled data, in which the users do not need to supervise the model. Instead, it allows the model to work on its own to discover patterns and information that was previously undetected. 

The goal in unsupervised machine learning is to find unknown patterns (structure) in data that data according to similarities, and represent that dataset in a compressed format. It is taken place in real time, so all the input data to be analyzed and labeled in the presence of learners.

:::{admonition} Unsupervised Learning for Regression or Classification
:class: warning
Unsupervised learning cannot be directly applied to a regression or classification problem because unlike supervised learning, we have the input data but no corresponding output data. 
:::

### Clustering 

Clustering is an important concept when it comes to unsupervised learning. It mainly deals with finding a structure or pattern in a collection of uncategorized data. Unsupervised Learning Clustering algorithms will process your data and find natural clusters(groups) if they exist in the data. You can also modify how many clusters your algorithms should identify. It allows you to adjust the granularity of these groups.

In this section we will cover three of the most important unsupervised cluseting learning algorithms, which we will apply into a set of data.

```{image} ../images/UnsupervisedLearningSummary.png
:alt: UnsupervisedLearningSummary
:class: bg-primary mb-1
:width: 800px
:align: center
```

Notice that the result are similar, but there are subtle differences due to the different nature of each algorithm.

#### Hierarchical Clustering

Hierarchical clustering is an algorithm which builds a hierarchy of clusters. It begins with all the data which is assigned to a cluster of their own. Here, two close cluster are going to be in the same cluster. This algorithm ends when there is only one cluster left.

```{code-block} python
:class: thebe
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

# Create a dataset with 4 classes
X, y = make_blobs(n_samples=40, centers=4, cluster_std=2, random_state=13)

# Plot the data
plt.figure(figsize = (16, 12))
plt.xlabel("$X_1$", fontsize=18)
plt.ylabel("$X_2$", fontsize=18)
plt.scatter(X[:, 0], X[:, 1], s=300)
plt.grid()
plt.show()
```

We can generate a dendogram, which shows the hierarchical clustering of row data points based on euclidean distance. It also tells the suitable number of clusters with different colors in the dendrogram.

```{code-block} python
:class: thebe
import scipy.cluster.hierarchy as sch

clusters = sch.linkage(X, method = 'ward', metric='euclidean')

plt.figure(figsize = (16, 12))
sch.dendrogram(Z = clusters)
plt.title('Dendrogram', fontsize=24)
plt.xlabel("$X_1$", fontsize=18)
plt.ylabel('Euclidean Distance', fontsize=18)
plt.show()

```

In the script above, we've generated the clusters and subclusters with our points, defined how our points would link (by applying the ward method), and how to measure the distance between points (by using the euclidean metric).

There are several linkage methods. By understanding more about how they work, you will be able to choose the appropriate one for your needs. Besides that, each of them will yield different results when applied. There is not a fixed rule in clustering analysis, if possible, study the nature of the problem to see which fits its best, test different methods, and inspect the results.

Some of the linkage methods are:

* Single linkage (Nearest Neighbor (NN)): The distance between clusters is defined by the distance between their closest members.

```{image} ../images/hierarchicalClustering1.png
:alt: hierarchicalClustering1
:class: bg-primary mb-1
:width: 800px
:align: center
```

* Complete linkage (Furthest Neighbor (FN), Farthest Point Algorithm, or Voor Hees Algorithm): The distance between clusters is defined by the distance between their furthest members. This method is computationally expensive.

```{image} ../images/hierarchicalClustering2.png
:alt: hierarchicalClustering2
:class: bg-primary mb-1
:width: 800px
:align: center
```

* Average linkage (UPGMA (Unweighted Pair Group Method with Arithmetic mean)): The percentage of the number of points of each cluster is calculated with respect to the number of points of the two clusters if they were merged..

```{image} ../images/hierarchicalClustering3.png
:alt: hierarchicalClustering3
:class: bg-primary mb-1
:width: 800px
:align: center
```

* Centroid linkage (Unweighted Pair Group Method using Centroids (UPGMC)): A point defined by the mean of all points (centroid) is calculated for each cluster and the distance between clusters is the distance between their respective centroids.

```{image} ../images/hierarchicalClustering4.png
:alt: hierarchicalClustering4
:class: bg-primary mb-1
:width: 800px
:align: center
```

* Ward linkage (Minimal Increase of Sum-of-Squares (MISSQ)):  It specifies the distance between two clusters, computes the sum of squares error (ESS), and successively chooses the next clusters based on the smaller ESS. Ward's Method seeks to minimize the increase of ESS at each step. Therefore, minimizing error.

```{image} ../images/hierarchicalClustering5.png
:alt: hierarchicalClustering5
:class: bg-primary mb-1
:width: 800px
:align: center
```

:::{admonition} Optimal Number of Clusters
:class: tip

Finding an interesting number of clusters in a dendrogram is the same as finding the largest horizontal space that doesn't have any vertical lines (the space with the longest vertical lines). This means that there's more separation between the clusters.

The optimal choice of clusters can be based on the horizontal lines in the dendrogram i.e. number of clusters should be 5.
:::

```{code-block} python
:class: thebe
import scipy.cluster.hierarchy as sch

clusters = sch.linkage(X, method = 'ward', metric='euclidean')

plt.figure(figsize = (16, 12))
sch.dendrogram(Z = clusters)
plt.axhline(y = 10, color = 'r', linestyle = '--', linewidth = 4)
plt.title('Dendrogram', fontsize=24)
plt.xlabel("$X_1$", fontsize=18)
plt.ylabel('Euclidean Distance', fontsize=18)
plt.show()

```

After locating the horizontal line, we count how many times our vertical lines were crossed by it - in this example, 4 times. So 4 seems a good indication of the number of clusters that have the most distance between them.

Then, we can create the model to fit the hierarchical means clustering model
using the `AgglomerativeClustering` module from Sklearn.

```{code-block} python
:class: thebe

from sklearn.cluster import AgglomerativeClustering

hierarchicClustering = AgglomerativeClustering(n_clusters = 4, affinity = "euclidean",
                             linkage = 'ward')
hierarchicClustering_pred = hierarchicClustering.fit_predict(X)
```

After that, we can plot the cluster results, and compare the original data with the clustered data.

```{code-block} python
:class: thebe

# Plot the data
plt.figure(figsize = (16, 12))
plt.xlabel("$X_1$", fontsize=18)
plt.ylabel("$X_2$", fontsize=18)
plt.scatter(X[:, 0], X[:, 1], c = hierarchicClustering.labels_, s=300)
plt.grid()
plt.show()

```

#### K-Means

K-means it is an iterative clustering algorithm which helps you to find the highest value for every iteration. Initially, the desired number of clusters are selected. In this clustering method, you need to cluster the data points into k groups. A larger k means smaller groups with more granularity in the same way. A lower k means larger groups with less granularity.

Kmeans Algorithm is an **iterative** algorithm that divides a group of n datasets into $k$ subgroups /clusters based on the similarity and their mean distance from the centroid of that particular subgroup/ formed. In this case, $K$ is the pre-defined number of clusters to be formed by the Algorithm. For example, if $K=3$ it means that the number of clusters to be formed from the dataset is 3.

K-Means Algorithm workflow:

Let's image that we have a unlabeled dataset with $n$ data points.

```{image} ../images/KMeans1.png
:alt: KMeans1
:class: bg-primary mb-1
:width: 600px
:align: center
```

1) Select the value of $K$, to decide the number of clusters to be formed. In this case, let's say $K=2$ to cluster the dataset and to put them into different respective clusters. 

2) Select random $K$ points which will act as centroids. Since $K=2$, we need to choose 2 random which will act as centroid to form the cluster.

```{image} ../images/KMeans2.png
:alt: KMeans2
:class: bg-primary mb-1
:width: 600px
:align: center
```

3) Assign each data point, based on their distance from the randomly selected points (Centroid), to the nearest/closest centroid which will form the predefined clusters.

```{image} ../images/KMeans3.png
:alt: KMeans3
:class: bg-primary mb-1
:width: 600px
:align: center
```

4) Place a new centroid of each cluster.

```{image} ../images/KMeans4.png
:alt: KMeans4
:class: bg-primary mb-1
:width: 600px
:align: center
```

5) Repeat step no.3, which reassign each datapoint to the new closest centroid of each cluster.

6) If any reassignment occurs, then go to step-4 else go to Step 7.

7) Finish

```{image} ../images/KMeans5.png
:alt: KMeans5
:class: bg-primary mb-1
:width: 600px
:align: center
```

We are going to use the same dataset as we previously used in the Hierarchical Clustering section.

```{code-block} python
:class: thebe
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

# Create a dataset with 4 classes
X, y = make_blobs(n_samples=40, centers=4, cluster_std=2, random_state=13)

# Plot the data
plt.figure(figsize = (16, 12))
plt.xlabel("$X_1$", fontsize=18)
plt.ylabel("$X_2$", fontsize=18)
plt.scatter(X[:, 0], X[:, 1], s=300)
plt.grid()
plt.show()
```

Let's create the K-Means model using the `KMeans` module from Sklearn, and select the number of clusters to be formed as $K=4$.

```{code-block} python
:class: thebe
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
```

Let's visualize the results by plotting the data colored by these labels. We will also plot the cluster centers as determined by the k-means estimator, using the `KMeans.cluster_centers_` attribute.

```{code-block} python
:class: thebe
centers = kmeans.cluster_centers_


plt.figure(figsize = (16, 12))
plt.scatter(X[:, 0], X[:, 1], s=300, c = y_kmeans)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=500, alpha=0.5)
plt.xlabel("$X_1$", fontsize=18)
plt.ylabel("$X_2$", fontsize=18)
plt.grid()
plt.show()
```

In order to find the best value for $K$, we can run K-means algorithm across our data for a range of possible values. Since we have 40 data points, so the maximum number of clusters is 40. So for each value $K$ in `range(1,41)`, we train a K-means model and plot the intertia at that number of clusters:

```{code-block} python
:class: thebe
inertias = []

for i in range(1,10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.figure(figsize = (16, 8))
plt.plot(range(1,10), inertias, marker='o')
plt.scatter(4, inertias[3], c='red', s=500, alpha=0.5)
plt.title('Elbow method', fontsize=24)
plt.xlabel('Number of clusters', fontsize=18)
plt.ylabel('Inertia', fontsize=18)
plt.grid()
plt.show()
```

:::{admonition} Limiting the number of clusters to test
:class: note
Since this is an educational exercise, we will limit the number of clusters to test to 10. Remember that larger the number of clusters, the more granular the clusters are. Also, the more clusters you choose, the more time it will take to run the algorithm.

:::

**Inertia** measures how well a dataset was clustered by K-Means. It is calculated by measuring the distance between each data point and its centroid, squaring this distance, and summing these squares across one cluster.

A good model is one with low inertia and a low number of clusters ($K$). Also, noticed that the "elbow" on the graph above (where the interia becomes more linear) is at $K=4$, which corresponds to the optimal number of clusters, since after the 4th cluster, the inertia becomes (almost) linear.

:::{admonition} K-Means is a Hard Clustering Method
:class: note
K-means is that it is a hard clustering method, which means that it will associate each point to one and only one cluster. A limitation to this approach is that there is no uncertainty measure or probability that tells us how much a data point is associated with a specific cluster. 
:::

#### Gaussian Mixture Models

A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of **Gaussian distributions** with unknown parameters. Different than the previous clustering methods described previously, the Gaussian mixture model is a soft clustering method

A Gaussian Mixture is a function that is comprised of several Gaussians, each identified by $k ∈ {1,…, K}$, where $K$ is the number of clusters of our dataset. Each Gaussian k in the mixture is comprised of the following parameters:

* A mean $μ$ that defines its centre.
* A covariance $Σ$ that defines its width. This would be equivalent to the dimensions of an ellipsoid in a multivariate scenario.
* A mixing probability π that defines how big or small the Gaussian function will be.

```{image} ../images/GMM.png
:alt: GMM
:class: bg-primary mb-1
:width: 600px
:align: center
```

In this example, we can see that there are three Gaussian functions, hence $K = 3$. Each Gaussian explains the data contained in each of the three clusters available.

Let’s suppose we have a dataset that looks like this:

```{image} ../images/GMM1.png
:alt: GMM1
:class: bg-primary mb-1
:width: 600px
:align: center
```

Our goal is to find sets of points that appear close together. In this case, we can clearly identify two clusters of points which we will colour red and green, respectively.

```{image} ../images/GMM2.png
:alt: GMM2
:class: bg-primary mb-1
:width: 600px
:align: center
```

We are going to use the same dataset as we previously used in the previous sections.

```{code-block} python
:class: thebe
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

# Create a dataset with 4 classes
X, y = make_blobs(n_samples=40, centers=4, cluster_std=2, random_state=13)

# Plot the data
plt.figure(figsize = (16, 12))
plt.xlabel("$X_1$", fontsize=18)
plt.ylabel("$X_2$", fontsize=18)
plt.scatter(X[:, 0], X[:, 1], s=300)
plt.grid()
plt.show()
```

Then, we will import that Gaussian mixture model from the `sklearn.mixture` module from Sklearn, and select a (random) number of clusters to use, and plot the results.

```{code-block} python
:class: thebe
from sklearn import mixture

gmm = mixture.GaussianMixture(n_components=4)
gmm.fit(X)
y_gmm = gmm.predict(X)

# Plot the data
plt.figure(figsize = (16, 12))
plt.xlabel("$X_1$", fontsize=18)
plt.ylabel("$X_2$", fontsize=18)
plt.scatter(X[:, 0], X[:, 1], s=300, c = y_gmm)
plt.grid()
plt.show()
```

But because GMM contains a probabilistic model under the hood, it is also possible to find probabilistic cluster assignments—in Scikit-Learn this is done using the `predict_proba` method. This returns a matrix of size `[n_samples, n_clusters]` which measures the probability that any point belongs to the given cluster:

```{code-block} python
:class: thebe
probs = gmm.predict_proba(X)
print(probs[:5].round(3))
```

:::{admonition} Only the first 5 rows of the matrix
:class: note
Notice that we have 40 data points, and 2 clusters. However, we are just showing the first 5 rows of the matrix.
:::

We can visualize this uncertainty by, for example, making the size of each point proportional to the certainty of its prediction; looking at the following figure, we can see that it is precisely the points at the boundaries between clusters that reflect this uncertainty of cluster assignment:

```{code-block} python
:class: thebe

size = 500 * probs.max(1) ** 2  # square emphasizes differences
plt.figure(figsize = (16, 12))
plt.xlabel("$X_1$", fontsize=18)
plt.ylabel("$X_2$", fontsize=18)
plt.scatter(X[:, 0], X[:, 1], c = y_gmm, s=size)
plt.grid()
plt.show()
```

As you can see, GMM is very similar to K-Means, except that it is a soft clustering method since it uses an expectation–maximization approach which qualitatively choose a random centroid for each cluster, and then iteratively repeat until converge. The result of this is that each cluster is associated not with a hard-edged boundary, but with a smooth Gaussian model. 

Let's visualize the locations and shapes of the GMM clusters by drawing ellipses based on the GMM output:

```{code-block} python
:class: thebe

from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

def plot_gmm(gmm, X, label=True, ax=None):
    plt.figure(figsize = (16, 12))
    plt.xlabel("$X_1$", fontsize=18)
    plt.ylabel("$X_2$", fontsize=18)
    plt.grid()
    
    
    labels = gmm.fit(X).predict(X)
    if label:
        
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=300, cmap='viridis', zorder=2)
    else:
        plt.scatter(X[:, 0], X[:, 1], s=300, zorder=2)
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
        

gmm = mixture.GaussianMixture(n_components=2, random_state=42)
plot_gmm(gmm, X)
```

:::{admonition} Change the number of clusters
:class: note
From the previous exercise, feel free to change the number of clusters (`n_components=2`) to see how the results change.
:::

Of the most important parameters to change for GMM are the covariance matrix  It is the matrix that describes the uncertainty of the model, and controls the degrees of freedom in the shape of each cluster. The default is `covariance_type="diag"`, which means that the size of the cluster along each dimension can be set independently, with the resulting ellipse constrained to align with the axes.

We can see change this hyperparameter and visually representation of each of these three choices:

```{code-block} python
:class: thebe
gmm = mixture.GaussianMixture(n_components=4, covariance_type='diag', random_state=42)
plot_gmm(gmm, X)
```

A slightly simpler and faster model is `covariance_type="spherical"`, which constrains the shape of the cluster such that all dimensions are equal. The resulting clustering will have similar characteristics to that of k-means, though it is not entirely equivalent. 

A more complicated and computationally expensive model (especially as the number of dimensions grows) is to use `covariance_type="full"`, which allows each cluster to be modeled as an ellipse with arbitrary orientation.


```{code-block} python
:class: thebe
gmm = mixture.GaussianMixture(n_components=4, covariance_type='full', random_state=42)
plot_gmm(gmm, X)
```

### Dimensionality Reduction

Dimensionality reduction refers to techniques that reduce the number of input variables in a dataset.

In theory, more input features often make a predictive modeling task more challenging to model, more generally referred to as the **curse of dimensionality**.

:::{admonition} High-dimensionality
:class: tip
In a real world case, high-dimensionality might mean hundreds, thousands, or even millions of input variables.
:::

:::{admonition} Curse of Dimensionality
:class: note
The curse of dimensionality is one of the most commonly occurring problems in ML. It is a problem that arises when working with data that has a high number of dimensions in the feature space. 

On other words, it means that the error increases with the increase in the number of features. It refers to the fact that algorithms are harder to design in high dimensions and often have a running time exponential in the dimensions. A higher number of dimensions theoretically allow more information to be stored, but practically it rarely helps due to the higher possibility of noise and redundancy in the real-world data.
:::

Fewer input dimensions often mean correspondingly fewer parameters or a simpler structure in the machine learning model, referred to as degrees of freedom. A model with too many degrees of freedom is likely to overfit the training dataset and therefore may not perform well on new data. In general, we desire to have simple models that generalize well, and in turn, input data with few input variables.

Also, it is very important to highlught that dimensionality reduction techniques are is extremely useful for data visualization. When we reduce the dimensionality of higher dimensional data into two or three components, then the data can easily be plotted on a 2D or 3D plot.

There are mainly two types of dimensionality reduction methods. Both methods reduce the number of dimensions but in different ways. One type of method only keeps the most important features in the dataset and removes the redundant features. In this case, no transformation applied to the set of features. 

The other method finds a combination of new features. To accomplish this goal an appropriate transformation is applied to the set of features. It is important to keep in mind that the new set of features contains different values instead of the original values. This method can be divided in linear and non-linear methods. In the Linear methods involve linearly projecting the original data onto a low-dimensional space. Principal Component Analysis (PCA), Factor Analysis (FA), Linear Discriminant Analysis (LDA) and Truncated Singular Value Decomposition (SVD) are examples of linear dimensionality reduction methods.

In case we have non-linear data, which are frequently used in real-world applications, linear methods do not perform well for dimensionality reduction. Kernel PCA, t-distributed Stochastic Neighbor Embedding (t-SNE), Multidimensional Scaling (MDS) and Isometric mapping (Isomap) are examples of non-linear dimensionality reduction methods.


#### PCA

PCA is a linear dimensionality reduction algorithm that transforms a set of correlated variables ($p$) into a smaller $k$ ($k<p$) number of uncorrelated variables called **principal components** while retaining as much of the variation in the original dataset as possible.

In order to undertand how the PCA algorithm works, we will load the Iris dataset, which has 150 samples and 4 features.

```{code-block} python
:class: thebe
import numpy as np 
import pandas as pd

# Import the dataset
df = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')

# Show the first five rows
df.head()
```

Using some of the visualization tools that we have learned about, we can see plot the data in a 2D space (scatterplot).

```{code-block} python
:class: thebe
import matplotlib.pyplot as plt

plt.figure(figsize = (16, 12))
plt.scatter(df['sepal_length'], df['sepal_width'], s = 300)
plt.xlabel('sepal_length', fontsize=18)
plt.ylabel('sepal_width', fontsize=18)
plt.grid()
plt.show()
```

In case we would like to plot all the data might be tedious. For that reason we can use the `pairplot` function from Seaborn to plot all the variables in order to find any trend in the dataset.

```{code-block} python
:class: thebe
import seaborn as sns

plt.figure(figsize = (16, 12))
sns.pairplot(df, hue='species', height=2.5)
plt.show()
```

In PCA we need to calculate the covariance matrix of the dataset and then perform a eigenvalue decomposition of the covariance matrix.

:::{admonition} Variance vs. Covariance
:class: note

##### Variance and Covariance

The variance is the average of the squared deviations of each data point from the mean. The covariance is the average of the products of deviations of each data point from the mean.

On other words, the variance reports variation of a single random variable, and the covariance reports how much two random variables vary, which is the same as the average of the products of deviations of each data point from the mean.

On the diagonal of the covariance matrix we have variances, and other elements in the matrix are the covariances.
:::

We can calculate the covariance matrix of the dataset using the `cov` function from `numpy`.

```{code-block} python
:class: thebe

X = df.drop('species', axis=1).values
cov_mat = np.cov(X.T)
cov_mat[:5]
```

Notice that the covariance matrix is symmetrical.

##### Eigenvalue Decomposition

Eigenvalue is a process that decomposes a square matrix into eigenvectors and eigenvalues. Eigenvectors are simple unit vectors, and eigenvalues are coefficients which give the magnitude to the eigenvectors.

Since our covariance matrix is symmetrical, it means that eigenvectors of symmetric matrices are orthogonal. In the case of PCA, meaning that the first principal component will *explains* most of the variance. Orthogonal to it we will find the second principal component, which explains most of the remaining variance. This is repeated for $N$ number of principal components, where $N$ equals to number of original features.

We can perform the Eigenvalue through Numpy, and it returns a tuple, where the first element represents eigenvalues and the second one represents eigenvectors:

```{code-block} python
:class: thebe

values, vectors = np.linalg.eig(cov_mat)
print('First 5 Eigenvalues: ', values[:5])
print('First 5 Eigenvectors: ', vectors[:5])
```

Then, we can calculate the percentage of explained variance per principal component:

```{code-block} python
:class: thebe
explained_variances = []
for i in range(len(values)):
    explained_variances.append(values[i] / np.sum(values))
 
print(np.sum(explained_variances), '\n', explained_variances)
```

The first value is just the sum of explained variances — and must be equal (or almost) 1. The second value is an array, representing the explained variance percentage per principal component. In this example, the first two principal components account for around 92% of the variance in the data.

As we described in the introduction of the problem, this dataset has 4 features which would be difficult to represent further than 3 dimensions. However, after applying PCA we were able to "explain" almost the entire dataset with the first two principal components where the first component explains 92% of the variance and the second component explains around 5% of the variance.

Now we’ll create a Pandas `DataFrame` object consisting of those two components, alongside the target class.

```{code-block} python
:class: thebe
PCA_1 = X.dot(vectors.T[0])
PCA_2 = X.dot(vectors.T[1])

df_PCA = pd.DataFrame(PCA_1, columns=['PCA_1'])
df_PCA['PCA_2'] = PCA_2
df_PCA['species'] = df['species']
df_PCA.head()
```

Finally, we can plot the new PCA data corresponding to the 4 (initial) features.

```{code-block} python
:class: thebe
import seaborn as sns

plt.figure(figsize = (16, 12))
sns.scatterplot(x = 'PCA_1', y = 'PCA_2', data = df_PCA, hue='species', s=200)
plt.ylim(-7, -4);
```

We can use the PCA module of `scikit-learn` to perform the PCA on the dataset. The most important hyperparameter in that class is `n_components`. Then, we the `fit` method to fit the model to the data. The variable `pca` stores the transformed values of the principal components returned by the `PCA()` class. 

```{code-block} python
:class: thebe
from sklearn.decomposition import PCA

pca = PCA(n_components=4, random_state=42)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PCA 1', 'PCA 2', 'PCA 3', 'PCA 4'])
principalDf['species'] = df['species']
principalDf.head()
```

We can check the explained variance percentage of each principal components. The `explained_variance_ratio_` attribute of the `PCA()` class returns a one-dimensional numpy array which contains the values of the percentage of variance explained by each of the selected components.

```{code-block} python
:class: thebe
print ('Variance explained by the first principal component: %.2f%%' % (np.cumsum(pca.explained_variance_ratio_[0] * 100)))
print ('Variance explained by the second principal component: %.2f%%' % (np.cumsum(pca.explained_variance_ratio_[1] * 100)))
print ('Variance explained by the third principal component: %.2f%%' % (np.cumsum(pca.explained_variance_ratio_[2] * 100)))
```

If we get the cumulative sum of the above array, and plot it, we can see how much variance is explained by each principal component.

```{code-block} python
:class: thebe
plt.figure(figsize = (16, 8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components', fontsize=18)
plt.ylabel('Cumulative explained variance', fontsize=18)
plt.title('PCA: Explained variance by number of components', fontsize=24)
plt.grid()
plt.show()
```

Finally, we can plot the new PCA data corresponding to the 4 (initial) features.

```{code-block} python
:class: thebe
plt.figure(figsize = (16, 12))
sns.scatterplot(x = 'PCA 1', y = 'PCA 2', data = principalDf, hue='species', s=300)
plt.ylim(1.5, -1.5);
```