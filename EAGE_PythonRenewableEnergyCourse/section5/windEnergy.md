# Wind Energy

Amongst other things, in order to achieve Europe’s plan to cut carbon emissions by at least 55% by 2030, Consumer and Industrial electricity users behavior changes will be required, which is the focus for this project. As shown in Fig.1, if the wind electricity is predicted to reach the current 70%, then:

- Industrial users (like Data Centers) can charge batteries for later use.
- Consumers can program their electric appliances to run over those hours, for example: 1) Charge an electric car; 2) Launch washing machine with tumble drier; 3) Increase heat pump etc.

```{image} ../images/wind29.jpg
:alt: wind1
:class: bg-primary mb-1
:width: 800px
:align: center
```

## Case 1

- [Trading Wind Energy: Wind Energy Forecasting Model based on Deep Learning](https://towardsdatascience.com/trading-wind-energy-wind-energy-forecasting-model-based-on-deep-learning-a44f5906d531)

### Goal

- Develop a profitable wind energy demand forecasting model for energy traders based on deep learning

### Motivation

- Creating a steady supply of energy is always vital as our modern society genuinely depends on this.
- Harnessing renewable energy to confront the risk of energy shortfall will continually exist, driving us to use finance.
- Among those renewable energy sources to date, some are dependent on the environment, such as wind energy.
- In the wind energy sector we have three main players: Grid Operators (responsible for providing society with a steady supply of electrical energy), Energy Producers (manage the risk of energy shortfall, wind energy producers), and **Energy Traders** (serve to maximize profits on behalf of their clients).

### Objectives: Energy Trader

- As Energy Trader, the goal is to get a T+18 hour energy forecast every hour.
- Create a forecast model based on time-series datasets using deep learning (neural network) — specifically the difference network architecture.
- Generate a model maximize profits for our client (wind energy producers).

### Trading ALgorithm

Source: Deep Learning Datathon 2020 (by ai4impact)

1. Generate a T+18h forecast of energy production from your client’s windfarms. This forecast is central to the trading
2. Our client is paid 10 euro cents per kWh sold to the grid. You can only sell to the grid what you forecast or that date
3. If the actual energy production exceeds the forecast, this excess is absorbed by the grid, but your client is not compensated for this excess.
4. If the actual energy production is below the forecast, you must buy energy from the spot market (at 20 euro cents/kWh) to supply the grid. You are given a cash reserve at the start of 10,000,000 euro cents to buy energy from the spot market.

### Methodology

1. Examine the data provided along with the statistics.
2. Normalize the data and set the baseline risk (persistence based).
3. Fit the training and test set well on risk function. The test loss should beat the baseline risk.
4. Improve the model performance (keep minimizing the risk while reducing the lag/maintaining zero lag).
5. Check the best model for reproducibility.

### Datasets

- Wind Energy ProductionWind Energy Production
- Wind Forecasts

#### Wind Energy ProductionWind Energy Production

```{image} ../images/wind1.jpg
:alt: wind1
:class: bg-primary mb-1
:width: 800px
:align: center
```

Source: [Réseau de Transport d’Électricité (RTE)](https://www.rte-france.com/en/eco2mix/eco2mix-telechargement-en), the French energy transmission authority

This dataset, named energy-ile-de-france, contains the consolidated near-realtime wind energy production (in kWh) for Île-de-France region surrounding Paris that have been averaged and standardized to a time base of 1 hour. The data is provided from 01 January 2017 to the present.

```{image} ../images/wind2.jpg
:alt: wind2
:class: bg-primary mb-1
:width: 800px
:align: center
```

The data is not really regular, but we can still see some trends. For example, the spikes of energy are most common in the winter, and the transition between seasons. So far, the largest energy produced occurred in the winter 2019–2020 which is up to 89000 kWh. The basic statistics of the data are presented below.

- **Mean** = 17560.44 kWh
- **Median** = 10500.0 kWh
- **Max** = 89000.0 kWh
- **Min** = 0.0 kWh
- **Range** = 89000.0
- **Standard Deviation** = 19146.63

#### Wind Forecasts

Source: [Terra Weather](http://terra-weather.com/)

The data comes with 2 different wind forecast models (A and B), for 8 location wind farms in the Île-de-France region. Hence, there are 16 forecasts where each has 2 variables: wind speed (m/s) and wind direction as a bearing (degrees North — ie. 45 degrees means the wind blows from the northeast). The forecasts are updated daily every 6 hours and have been interpolated to the time base of 1 hour.

```{image} ../images/wind9.jpg
:alt: wind9
:class: bg-primary mb-1
:width: 800px
:align: center
```

```{image} ../images/wind2.jpg
:alt: wind2
:class: bg-primary mb-1
:width: 800px
:align: center
```

The wind speed graph has a similar trend to the energy one, indicating that this forecast data can be useful to our model as input features. Referring to this, the strongest wind occurs in the winter which is up to 12 m/s.

```{image} ../images/wind6.jpg
:alt: wind6
:class: bg-primary mb-1
:width: 800px
:align: center
```

Compared to the wind speed forecasts, the wind direction pattern is tough to decipher. But we will see that even for such data can still bring benefit to our forecast model.

```{image} ../images/wind4.jpg
:alt: wind4
:class: bg-primary mb-1
:width: 800px
:align: center
```

##### Normalize the Data and Set the Baseline Risk

To speed up the training process, we will normalize our data to have zero mean and variance 1 using the formula below.

$$x' = \frac{x-mean}{standardDeviation}$$

```{image} ../images/wind3.jpg
:alt: wind3
:class: bg-primary mb-1
:width: 800px
:align: center
```

Now we have each feature on the same scale. Note that we only normalize energy and wind speed. The wind direction values will have special treatment later.

Next, we will obtain the baseline based on persistence risk. We extract the baseline risk using mean squared error (MSE) and mean absolute error (MAE).

- Persistence risk (MSE): 0.4448637
- Persistence risk (MAE): 0.6486683

##### Model Architecture

Using the difference network architecture, we fit the training and test set well on risk function (MSE and MAE). The difference network helps us achieve better learning to beat this baseline. As a reminder, the objective is to get energy forecasts with a lead time of 18 hours.

```{image} ../images/wind5.jpg
:alt: wind5
:class: bg-primary mb-1
:width: 800px
:align: center
```

The followings are a few hyperparameters we can turn on:

- Windowing input features (Naive, DIFF, momentum and force inputs)
- Statistic input features (Mean, SD, MAX, MIN, etc)
- Optimizer (Adam, SGD)
- Activation functions (Relu, Tanh)
- Number of Hidden layers (2 to 5)
- Regularization (Dropout, L2)
- NN-Size (8 to 256 neurons with 2/3 reduction for the next layer)
- Subnetworks (Input scaling, Autoencoder)
- Type of perceptrons (normal, squared perceptron)
- Losses (MAE, MSE, Momentum loss, force loss)

###### Experiment 1

Input Scaling Subnet + 4 Hidden Layers (with Dropout)

In the first experiment, we try to create a low MAE (and MSE) that will beat the baseline. Therefore, we want our network to be deep and big enough without overfitting. As a result, we use Adam to achieve better learning and add a regularization method called the dropout layer to prevent overfitting. We use a 4-layer network with the multi-configuration as follows:

- Input scaling sub-network
- NN-size: 32/64/128/256
- dropout-prob: 0.05/0.1/0.25
- Optimizer: Adam
- Number of layers: 4

**Feature selection**

Windowing is a basic operation for time-series data. Thus, for the input features, we use a window consisting of 60 hours of past energy produced (T-60). Then we turn the window into DIFF-momentum-force inputs with a lead time of 18 hours. It will result in 72 features. This adjustment helps the model detect movement and its rate to perform better clustering.

We also add an average of 60 hours of past wind speed forecasted and the wind speed forecast at T+18h from each wind model. This generates 4 more features. Thus, we have 76 input features in total ready to feed to the input scaling subnet. Since we use relatively large inputs, this subnet reduces unwanted features before supplying it to the main network.

In summary, these are the list of our input features:

- DIFF+momentum+force inputs of T-60h past energy produced with a lead time of 18h
- the mean of T-60h of past wind speed forecasted (model A)
- the wind speed forecast at T+18h (model A)
- the same applies to model B

**Best Config Loss**

Test Loss: 0.554845 (using MAE as the loss function)

```{image} ../images/wind15.jpg
:alt: wind15
:class: bg-primary mb-1
:width: 800px
:align: center
```

**Evaluation**

```{image} ../images/wind10.jpg
:alt: wind10
:class: bg-primary mb-1
:width: 800px
:align: center
```

Evaluation summary for exp. 1

- Best test loss / Persistence error
- MSE: 0.280589 / 0.4448637
- MAE: 0.554845 / 0.6486683
- Best NN-size: 128
- Best dropout-prob: 0.05

Notice that we have beaten the persistence and achieved zero lag.

There is still a high gap between training and test loss. We can consider using Regularization and adding more features.

###### Experiment 2

Input Scaling Subnet + 4 Hidden Layers
(with Dropout + L2 Regularization)

With the same model as before, we add the L2 regularization into our model. We also run multiconfiguration while taking the best hyperparameters into account.

- Input scaling sub-network
- NN-size: 64/128/256
- dropout-prob: 0.05/0.1
- Weight decay: 1.0E-4/1.0E-5/1.0E-6
- Optimizer: Adam
- Number of layers: 4

**Feature selection**

We add new input features from the wind direction forecast. Although it’s a bit nonsense to add direction data as our input, a steady wind direction does help. Thus, we do not want to normalize direction naively, yet we will use trigonometric functions to _normalize_ it. In addition to the previous one, now we have 84 input features in total.

In summary, these are the addition to our input features:

- (mean) sin function of T-18h of past wind direction forecasted (model A)
- (mean) cos function of T-18h of past wind direction forecasted (model A)
- wind direction forecast at T+18h in sin function (model A)
- wind direction forecast at T+18h in cos function (model A)
- The same applies to model B

**Losses**

```{image} ../images/wind11.jpg
:alt: wind11
:class: bg-primary mb-1
:width: 800px
:align: center
```

Test Loss: 0.549824 (using MAE as the loss function)

**Evaluation**

```{image} ../images/wind12.jpg
:alt: wind12
:class: bg-primary mb-1
:width: 800px
:align: center
```

Evaluation summary for exp. 2

- Best test loss / Persistence error
- MSE: 0.26769 / 0.4448637
- MAE: 0.549824 / 0.6486683
- Best NN-size: 128
- Best dropout-prob: 0.1
- Best Weight decay: 1.0E-4

Notice that we have produced a better test loss while maintaining zero lag (also increase the peak value of lag graph).

We can still improve the performance by adding more input features or layers to the model.

###### Experiment 3 (Final Model)

- Input Scaling Subnet + 4 Hidden Layers (with Dropout + L2 Regularization)

By setting the best hyperparameters fixed, the followings are our network configuration:

- Input scaling sub-network
- NN-size: 128
- dropout-prob: 0.1
- Weight decay: 1.0E-4
- Optimizer: Adam
- Number of layers: 4

**Feature selection**

We include new statistic features as the new additional inputs, taken from energy and wind speed data. In the end, we have 88 input features in total.

- the mean of T-60h of past energy produced
- the standard deviation of T-60h of past energy produced
- the standard deviation of T-60h of past wind speed forecasted (model A)
- the standard deviation of T-60h of past wind speed forecasted (model B)

**Losses**

```{image} ../images/wind17.jpg
:alt: wind17
:class: bg-primary mb-1
:width: 800px
:align: center
```

Test Loss: 0.52758 (using MAE as the loss function)

**Evaluation**

```{image} ../images/wind16.jpg
:alt: wind16
:class: bg-primary mb-1
:width: 800px
:align: center
```

Evaluation summary for exp. 3

- Best test loss / Persistence error
- MSE: 0.258521 / 0.4448637
- MAE: 0.52758 / 0.6486683
- Net profit in euro cents
- MSE: 1.392861351E9
- MAE: 1.447243201E9

We achieve the best test loss using this last model while having no lag. As a result, we have the highest profit of all models.

We have better scatter plots of actual vs training/test predictions. Although we fit well on the training set, obtaining a better scatter plot of actual vs test prediction is still a challenge.

**Reproducibility**

Previously, the final model above has been done 40 repeats of training where each takes a maximum of 10000 iterations. Note that we use MAE for the loss function as it gives a higher profit to the clients. The statistics of the test losses are shown below.

- Mean = 0.540747
- Median = 0.540757
- Max = 0.550977
- Min = 0.527580
- Range = 0.023397
- (Mean-Min)/Standard Deviation = 2.690480

**Final Prediction Model**

```{image} ../images/wind13.jpg
:alt: wind13
:class: bg-primary mb-1
:width: 800px
:align: center
```

```{image} ../images/wind14.jpg
:alt: wind14
:class: bg-primary mb-1
:width: 800px
:align: center
```

```{image} ../images/wind9.jpg
:alt: wind9
:class: bg-primary mb-1
:width: 800px
:align: center
```

- Adding more layers decrease the training error, but increase the test loss and lower the profit, although we have used regularization techniques. Hence, we stick to the 4 layers in the final model.

- The Autoencoder subnet helps reduce the dimension of our input features. However, when added to the network with features no more than 100, it increases the test loss of our model.

- The squared perceptron supposes to provide faster and better learning than the ordinary one. However, during the experiment, it does not improve the performance in terms of lowering the error.

- The momentum and force losses supposed to help reduce lag. However, when we add the losses to the network, the lag graph does not change (still zero lag) and it makes the error higher since the network needs to minimize three losses altogether (test, momentum, and force losses).

##### Summary

The difference networks effectively build a forecasting model with time-series data, even with fewer inputs.

When it comes to historical data, the DIFF window, combined with momentum, force, and statistical features can help the model perform better prediction.

A bigger and deeper network supports the model to memorize well (be careful of overfitting).

Dropout layer (small dropout probability) and L2 regularization help the network handle overfitting problems, hence improving performance.

Although RMSE (or MSE) is also popular as the loss function in time series data, our model produces a higher profit when MAE is used. MSE is inclined to penalize outliers, while MAE is more linear with errors. Since the model has no outliers, MAE turns out to work best for our model.

## Case 2

- [Predicting Excess Wind Electricity in Ireland: Machine Learning against Climate Change](https://towardsdatascience.com/predicting-excess-wind-electricity-in-ireland-machine-learning-against-climate-change-part-1-d042894026a6)

### Goal

- Can Machine Learning algorithms uncover hidden patterns in a complex electricity network for reliable predictions?

### Motivation

Time series predictions between changing consumptions patterns, grid constraints and abruptly changing weather conditions, can be tricky. We are happy to share our experience with a range of ML Algorithms to help us optimize electricity consumptions and reduce our carbon footprint.

### Objectives:

- EDA
- Highlight missing data
- Evaluate data for collinearity, outliers and feature transformations
- Generate a Machine Learning and Deep Learning model

### Problem

- Explore Ireland’s situation and potential smart usage of the wind-generated electricity.
- In Ireland, wind energy contributes 80% of renewable electricity and 30% of total electricity demand.
- Wind energy is a growing industry concern about the amount of wind energy “lost” every year.
- In 2020 this amounted to more than 1.4 million MWh of electricity, nearly double the figure for 2019
- This represent under 11.5 per cent of total production and enough to power more than 300,000 homes

#### "Wasted Power" Problem

When the electricity generation exceeds the consumption, the TSO levers to adjust are limited:

- Redirect electricity to “storage”: in Ireland, pump water up to Turlough Hill Power Station (but limited)
- Export (market-permitting) to UK: max. 1 GW Connection (Ewic + Moyle)
- Ask Gas / Coal Generation Plants to ramp down, however it may take up to a few hours to ramp down
- The current maximum proportion of Wind / Solar Electricity is constrained by the levels of non- synchronous renewables allowed on the system at any given time is System Non-Synchronous Penetration (SNSP) “its current figure of 65% in Q1 2018” and has been increased to 70% recently.
- “Renewable Dispatch-Down” (Constraint and Curtailment): which is basically disconnecting wind farms from the grid causing wind energy to be “lost” as can be seen in EirGrid Group System and Renewable Reports.

Large investments in the electricity grid to support a higher rate of renewable energy will be facilitated through the European Green deal, however renewable capacity will massively increase resulting in far more “wasted” electricity.

Consumer and industrial users behavior changes will also be required, which is the focus for this project. As shown in Fig.2, if the wind electricity is predicted to reach the current 70%, then:

- Industrial users (like Data Centers) can charge batteries for later use.
- Consumers can program their electric appliances to run over those hours, for example: 1) Charge an electric car; 2) Launch washing machine with tumble drier; 3) Increase heat pump etc.

### Dataset

- Met Éirean data: Copyright Met Éireann, Source www.met.ie , Licence Statement: This data is published under a Creative Commons Attribution 4.0 International (CC BY 4.0).
- EirGrid Group Data: Supported by EirGrid Group Data, Source: www.smartgriddashboard.com , Open Data Licence.

a wind power dataset with 145,936 observations spans across Jan 2017 — Feb 2021 is downloaded from EirGrid Group for the island of Ireland, as Republic and Northern Ireland are together as one Integrated Single Electricity Market (I-SEM). The data depicts Wind-generated electricity and electricity demand from samples with a frequency of 15 minutes. To build the complete picture, the total Wind Capacity installed in the island of Ireland which is being reported monthly in the “System and renewable data summary report,” Eirgrid Group, Tech. Rep., 2020.

The historical weather information downloaded from Met Éirean depicts hourly weather (37,225 rows) from each of four meteorological stations located in Shannon Airport, Dublin Airport, Cork Airport and Belmullet, as many grid-connected wind farms are located close by, as well as Dublin as a major population centre for the electricity consumption impact. Furthermore, in a later phase of the work, the predictions of the proposed models, even the best models underestimated the Wind Electricity generation when wind was low in Dublin. We realized that wind speeds were high in the North of Ireland, where we didn’t have specific weather stations data. Hence, the weather data from Malin Head station is also selected into the weather dataset.

#### Data Processing

Overall, the data quality from both sources is excellent for the last 3 years.

In the Eirgrid data set, 66 rows of 15 min periods were Missing Completely at Random (MCAR) and were thus Backfilled.

In the historical Met Éirean, a chunk of data was missing from the start of 2017, so the whole dataset was reduced to start only on July 1st, 2017 without impact on models.

Looking at outliers on temperature and wind data, we found they were consistent with Irish short-term extreme temperature (so extremely rare above 30 degrees!) and storms (more frequent, good for wind!).

```{image} ../images/wind21.jpg
:alt: wind21
:class: bg-primary mb-1
:width: 800px
:align: center
```

We were surprised with some negative values for wind energy, but found Cyclical aerodynamic loads on the turbine blades produced a negative impact on the wind turbine, mainly due to the enhanced wind shear.

```{image} ../images/wind22.jpg
:alt: wind22
:class: bg-primary mb-1
:width: 800px
:align: center
```

Outlier Control charts also provided insights in trends in electricity generation and demand, in particular for the seasonality and the rising amount of wind electricity generation.

```{image} ../images/wind23.jpg
:alt: wind23
:class: bg-primary mb-1
:width: 800px
:align: center
```

To plot Control Charts to help spotting univariate outliers, this code is very handy:

```python
def plotOutliers(df, cols):
  fig = plt.figure(figsize=(10, 60))

  # loop over all vars (total: 14)
  for i in range(0, (len(cols))):
    plt.subplot(14, 1, (i+1))
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.axes.set_title(cols[i] )

    x=df.date
    y1=np.array(df[cols[i]])

    plt.plot(x, y1)
    plt.axhline(y=df[cols[i]].mean())
    plt.axhline(y=df[cols[i]].mean()+3*df[cols[i]].std(),color='r')
    plt.axhline(y=df[cols[i]].mean()-3*df[cols[i]].std(),color='r')

    plt.axhline(y=df[cols[i]].mean()+ df[cols[i]].std(),color='y')
    plt.axhline(y=df[cols[i]].mean()- df[cols[i]].std(),color='y')

    plt.title((cols[i] + " Mean: " + str(np.round(df[cols[i]].mean(), 1)) + " Std: " + str(np.round(df[cols[i]].std(), 1))), fontsize=12)

  plt.tight_layout()

plotOutliers(newdf, attributes)
```

SEAI Monthly generation data was also cross-checked against the Republic of Ireland 15-min data to confirm overall quality.

##### Colinearity

The intuition here is that the total possible production of electricity depends closely on the weather conditions close to the main wind farms, in particular as was found in [2], [3] and [4]: wind speed, wind direction, relative humidity and mean sea level air pressure (in hectopascal). Conversely, the electricity consumption depends on the hour of the day, business vs. weekend days but also on the air temperature.

However, since the weather data from multiple stations is required to have the full view, a lot of the measures will be correlated.

Data collinearity is likely to reduce model performance, as well as obfuscate features impact and should be prevented whenever possible.

We removed very highly correlated features (above 0.9) and high Variance Inflation Factor (VIF), for example temperature in various weather stations, resulting in a more manageable dataset:

```{image} ../images/wind24.jpg
:alt: wind24
:class: bg-primary mb-1
:width: 800px
:align: center
```

To check multi-collinearity, best is to use the variance_inflation_factor. A rule of thumb is if any VIF is greater than 10, then you really need to consider dropping variables from your model.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.api import add_constant

num_df = add_constant(num_df)

vif = [variance_inflation_factor(num_df.to_numpy(), i) for i in range(num_df.to_numpy().shape[1])]

pd.DataFrame(num_df.iloc[:, 1:].columns, vif[1:])
```

#### Features transformations

##### Transform time into 2D

From the Fast Fourier Transform of the temperature, wind speed and actual wind power shown in Fig.8, we can see there are obvious peaks at the day−1 and year−1 frequency components, which means the data have some potential daily and yearly patterns.Transform time into 2D

From the Fast Fourier Transform of the temperature, wind speed and actual wind power shown in Fig.8, we can see there are obvious peaks at the day−1 and year−1 frequency components, which means the data have some potential daily and yearly patterns.

```{image} ../images/wind51.jpg
:alt: wind25
:class: bg-primary mb-1
:width: 800px
:align: center
```

In order to emphasize these patterns in our models, we need to convert the 1D observation timestamps into a 2d periodic radian time space:

```{image} ../images/wind26.jpg
:alt: wind26
:class: bg-primary mb-1
:width: 800px
:align: center
```

Here we transform the time into two radian time spaces: one for the yearly period [yearSin, yearCos] one for the daily period [daySin, dayCos], which are derived by:

```{image} ../images/wind27.jpg
:alt: wind27
:class: bg-primary mb-1
:width: 800px
:align: center
```

##### 2D wind vector

As shown in the previous image, the wind direction is recorded in degrees, which does not make good model inputs, as 360° and 0° should be close to each other, and wrap around smoothly. Also, the wind direction has no effect on the model if the wind speed is high. Therefore, it is more sensible to combine the wind speed and direction to create a 2D wind vector feature $[windSin, windCos]$.

```{image} ../images/wind28.jpg
:alt: wind28
:class: bg-primary mb-1
:width: 800px
:align: center
```

As mentioned, look out for part 2 which will cover model candidates, specific Training / Validation split for a time series with a strong trend and results!

Remember that the model success will be measured mainly by:

- The main relevant metric for this work is the Mean Absolute Error (MAE), as absolute values are what we are trying to measure in order to recommend when to charge batteries.
- Exact predictions are most important when the proportion of actual wind generation is high, as when wind is low the electricity Carbon Intensity will be bad anyway (other renewables like Solar and Hydro have a low impact currently in Ireland)
- The Root Mean Squared Error (RMSE) and explained variance regression score are also measured from the models for better understanding of the model limitations.

#### Split Dataset

The last 2 weeks of March 2021 in the dataset is reserved as the Test set, and the rest of the dataset is split into Training and Validation sets. The standard random split using scikit-learn provides excellent validation results but very poor in the Test results. This is because, for time series data, the models typically predict a value close to the last/next value. With a randomly shuffled set, this value will typically be very close to the actual one, there is, in effect, Data Leakage.

A standard way to split Training / Validation sets in Time Series is to simply split the data at a date roughly at the 80% mark. However, as shown in Part 1, there is a continuous upwards trend in the target variable, so results on the Test set for the most recent data are poor.

For this reason, the dataset is split on a particular day (the 22nd) of each month, so that the Training set includes all the dates to the 22nd of the month and the Validation set dates above the 22nd of the month, thus preserving data in all years (for trend) and month (for seasonality). For a given high- performing model and feature set (Random Forest model and 2DTime), results on the Test set are significantly better with the custom Training-Validation split.

```python
from sklearn.model_selection import train_test_split
import numpy as np

splitOption = 1 # split per day of the month


testSet = dataSet.loc[(inData.date > cutOffTestDate), :]
mainSet = dataSet.loc[(inData.date <= cutOffTestDate), :]

if (splitOption == 0):  # Standard SkLearn train test split, usually not good for time series
  trainSet, validSet = train_test_split(mainSet, test_size=0.2, random_state=42)
elif (splitOption == 1):  # split before / after day of the month
  dataSet["dayInMonth"] = 0

  def setDayInMonth(row):
      row["dayInMonth"] = row.date.day
      return row

  dataSet = dataSet.apply(setDayInMonth, axis=1)

  trainSet = dataSet.loc[(dataSet.dayInMonth > 0) & (dataSet.dayInMonth < 23), :]
  validSet = dataSet.loc[(dataSet.dayInMonth >= 23) & (inData.date <= cutOffTestDate), :]

elif (splitOption == 2):  # split per period
  trainSet = dataSet.loc[(inData.date <= cutOffValidationDate), :]
  validSet = dataSet.loc[(inData.date > cutOffValidationDate), :]


y_train = trainSet.ActualWindMW
y_valid = validSet.ActualWindMW
y_test = testSet.ActualWindMW
```

#### Input Features

In order to explore the influence of each input feature on the models, the models are trained and tested on different sets of features (Table I). The impact of each input feature is examined by comparing the results from different input sets.

```{image} ../images/wind30.jpg
:alt: wind30
:class: bg-primary mb-1
:width: 800px
:align: center
```

#### Model Candidates

The following models are considered:

##### Random Forest

We selected the Random Forest Regression model as the prototype for our early analysis to get an idea of features which make a significant difference, it also handles linear and non-linear relationships quite well as well as bias vs. variance balance. The research for power predictions in [12] also states that they use such models. The default Random Forest parameters lead to fully grown and unpruned trees which can potentially be very large. In this case, the results were very good and time to train was under a few minutes, so they were fine. Note, standard SkLearn GridSearch implementations can be difficult to use with time series because of the possible “data leakage” of close-by hours in nested cross-validation as pointed out above. Best results were found on the “Rhum_Msl” features set which includes the standard wind speeds as well as the Relative Humidity and Sea-level Pressure data.

Random Forest Regression Evaluation for the last 2 weeks of data reserved for Testing (March 15th to 29th, 2021) ⇒ Mean Absolute Error (MAE): 219 . As shown in Fig.2, the validation errors depict a roughly uniform distribution, apart from a few outliers. The period highlighted in the green box is around the 1st lockdown in April 2020 and understandably patterns (mostly in energy demand) changed dramatically at that point.

```{image} ../images/wind31.jpg
:alt: wind31
:class: bg-primary mb-1
:width: 800px
:align: center
```

As shown in the previous picture, the predictions based on the Met Eireann historical data follow closely the actual Wind generation values on the Test set. Note the Eirgrid own Forecast for wind generation (Eirgrid Forecast Wind) tends to overshoot the actual generation when the demand is relatively low. Conversely, predictions from the proposed RF model are more accurate and effectively match the fact that the grid can cope with a maximum ratio of wind electricity.

```{image} ../images/wind32.jpg
:alt: wind32
:class: bg-primary mb-1
:width: 800px
:align: center
```

Features importance (image below according to the Random Forest model have to be taken into account carefully, mostly as there are a number of residual collinearity between weather stations measures, but they give an idea of the features important to the model.

```{image} ../images/wind33.jpg
:alt: wind33
:class: bg-primary mb-1
:width: 800px
:align: center
```

Wind speed in Shannon (wdsp) in knots, as well as wind speeds in Malin Head (wdsp MAL), Cork (wdsp COR) and Belmullet (wdsp BEL) are of course key in predicting the overall wind generation as most wind farms are in those areas. The total wind power capacity in the island of Ireland (TotalWindCapacityMW) has increased year on year and is a major factor too. Day in year and Hour matter for weather patterns and demand seasonality. The current Temperature in Dublin (temp DUB) is also important, presumably because it impacts demand.

##### Artificial Neural Network

The main advantage of ANN models is their self-learning capacity to determine complex relations among variables while keeping high data tolerance. However, in order to achieve accurate prediction, the self-learning processes of ANNs require large amounts of data and the corresponding high cost of computation. Thanks to the explosive growth of available data and computation power, ANN models have been successfully used for modeling non-linear problems and complex systems in forecasting wind power generation and energy consumption.

Therefore, this project also employs the neural network method to compare to other models. The ANN model in this work is built using the Keras library of Tensorflow. There are different versions of the ANN model corresponding to the feature sets shown in Table I. All versions are experimented with different model settings ranging from 2 to 5 dense layers with neurons ranging from 20 to 260 neurons for each layer. According to the results of the experiments, the ANN model is settled with 3 layers with 120 neutrons and a final layer with 10 neurons. The model uses Adam optimizer and rectified linear (ReLU) activation function as ReLU outperforms other functions (such as Softplus, Sigmoid and Hyperbolic functions) in this project.

```python

ann = tf.keras.models.Sequential([
          tf.keras.layers.Dense(units=120, activation='relu', name="Layer_1"),
          tf.keras.layers.Dropout(0.1), #drop-out layer to avoid overfit
          tf.keras.layers.Dense(units=120, activation='relu', name="Layer_2"),
          tf.keras.layers.Dropout(0.1),
          tf.keras.layers.Dense(units=120, activation='relu', name="Layer_3"),
          tf.keras.layers.Dropout(0.1),
          tf.keras.layers.Dense(units=10, activation='relu', name="Layer_4"),
          tf.keras.layers.Dense(units=1, name="output_layer")
          ])

ann.compile(loss=tf.keras.losses.mae, # mae is short for mean absolute error
            optimizer=tf.keras.optimizers.Adam(lr=0.002),
            metrics=["mae"])
```

In next plot, the training and testing results for the different feature sets suggest: 1) The 2D-time feature yields better performance, however the wind vector is not as expected; a) The ANN model for ‘time & rhum’ dataset is chosen for the later evaluation and comparison.

```{image} ../images/wind34.jpg
:alt: wind34
:class: bg-primary mb-1
:width: 800px
:align: center
```

##### LSTM

The LSTM network is sensible for this project due to the ability of learning both short-term and longer-term seasonal patterns off the weather observations.

- A LSTM where the model makes the entire sequence prediction in a single step.
- An Autoregressive LSTM which decomposes this prediction into individual time steps. Then each output can be fed back into itself at each step and predictions can be made conditioned on the previous one, like in the classic Generating Sequences With RNNs.

Both models use a 24-hour window of previous weather values and actual wind power as input, however they don’t use the current weather forecast for the next 24 hours. As a result their performance is suboptimal.

```{image} ../images/wind35.jpg
:alt: wind35
:class: bg-primary mb-1
:width: 800px
:align: center
```

##### Artificial Neural Network (24 Hour Model)

As a result of the findings above, we tried a Neural Network again but based on a single-shot prediction of the whole 24 H, similar to the LSTM above.

The intuition is that wind electricity generation will not only depend on the current winds blowing across Ireland but also on what happened in the hours before. For example, if a gas-fired power station is up and running at a high point and winds start to pick up, as the power station may take a few hours to wind down, wind electricity generation will be “dispatched down” for a little while.

Beside, the Wind generation level immediately preceding the 24 H window may inform the model too, thus a new features set will also include this data.

The new ANN model is also built using the Keras library of Tensorflow, and takes as in input the aggregated 24 H for the required N features and consists of 5 layers of N \* 24 neurons, followed by 2 layers to flatten to a vector of 24 H predictions.

Similarly to the hourly ANN models, the training and testing results for the different feature sets suggest: 1) The 2D-time feature yields better performance, however the wind vector is not as expected; a) The ANN model for ‘time & rhum & prev actual’ dataset is chosen for the later evaluation and comparison.

```{image} ../images/wind36.jpg
:alt: wind36
:class: bg-primary mb-1
:width: 800px
:align: center
```

#### Results

The AI models proposed in this work are evaluated and compared using MAE over the Test set (last 2 weeks of March 2021) for the best features set per model. The predictions of the models are also compared to the benchmark of the work which is the wind energy generation forecasted by EirGrid. As shown in Fig. 18, both the Random Forest and ANN models provide higher accuracy (lower MAE) than EirGrid’s. However, the performance of the LSTM model is the worst. This is because the current LSTM is solely based on the historical data of wind energy generation, and is expected to have improved performance when the weather features are incorporated in future work.

```{image} ../images/wind37.jpg
:alt: wind37
:class: bg-primary mb-1
:width: 800px
:align: center
```

But wait, are those last 2 weeks of March 2021, totally new data for the models as required for a Test set, representative of future performance? We can compare the Validation set to get an idea.

```{image} ../images/wind38.jpg
:alt: wind38
:class: bg-primary mb-1
:width: 800px
:align: center
```

Uh-oh, the results here are not as dramatic, though great to see that the 24H ANN Model still performs best. Why could that be? Is there any pattern to the error which we should be aware of?

In fact, there is, if we look at Errors (Predicted value — Actual Value) vs. the Actual Value :

```{image} ../images/wind39.jpg
:alt: wind39
:class: bg-primary mb-1
:width: 800px
:align: center
```

As we have seen in some examples, the forecast provided by EirGrid tends to overestimate Wind generation when there is a lot of wind and doesn’t seem to take into account the Grid SNSP constraints: we can see above that the error is positively correlated with the Actual value. This is true in particular for yellow points (2021) and orange points (2020) where more Wind Capacity was available, as well as a higher ratio of SNSP support in the grid in 2021.

```{image} ../images/wind40.jpg
:alt: wind40
:class: bg-primary mb-1
:width: 800px
:align: center
```

On the other end, our 24 Hour ANN Model tends to underestimate slightly at lower actual values. The range of errors in general is smaller too.

As the MAE scores are pretty similar, it’s also worth checking the Explained Variance Score, here we can see on this score the EirGrid forecast performance is poorer.

```{image} ../images/wind41.jpg
:alt: wind41
:class: bg-primary mb-1
:width: 800px
:align: center
```

Let’s have a closer look at those last 2 weeks of March:

```{image} ../images/wind42.jpg
:alt: wind42
:class: bg-primary mb-1
:width: 800px
:align: center
```

The predictions are all very good during the 1st week, when there is little wind and little wind generations.

When the wind reaches the maximum capacity of the grid (about 70% of the actual Demand at the time), the EirGrid forecast significantly overshoots, while our best model only slightly underestimates.

**The chosen Machine Learning models, both Neural Networks and Random Forest, are thus able to discover hidden patterns of electricity generation and demand from a few simple weather stations measures, hour and day of the year and connected wind farm capacity.**
