---
layout: post
title: "Multivariate Regression and Binary Classification Competition"
author: Kaysan Merchant
date: 2025-12-07
---

# Multivariate Regression and Binary Classification Competition

## Four Models Prediction

This file contains the process of using 4 models, trained on `train.csv` to predict miles per gallon of cars in the `test.csv` set. This prediction turned out to be quite inaccurate, but it was a valiant attempt to use 4 models to try and get more accurate, like Mr. Andrade wanted.

## Setting the variable


```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence \
import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP.models import (ModelSpec as MS,
summarize,
poly)
from sklearn.metrics import mean_squared_error

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

temp_test, test_d = train_test_split(train, test_size=0.25)

temp_test_2, test_c = train_test_split(temp_test, test_size=0.33)

test_a, test_b = train_test_split(temp_test_2, test_size=0.5)

train_a = pd.concat([test_d, test_c, test_b], ignore_index=True)
train_b = pd.concat([test_d, test_c, test_a], ignore_index=True)
train_c = pd.concat([test_d, test_a, test_b], ignore_index=True)
train_d = pd.concat([test_a, test_c, test_b], ignore_index=True)
```

Above we are turning the train set in 4 test sets, with each test dataset using the other 3 for training.

## All The Work


```python
feature_columns = ['horsepower', 'weight', 'cylinders', 'displacement', 'acceleration', 'year']

model_matrix = MS(feature_columns)

x_train_a = model_matrix.fit_transform(train_a)
x_train_b = model_matrix.fit_transform(train_b)
x_train_c = model_matrix.fit_transform(train_c)
x_train_d = model_matrix.fit_transform(train_d)

target_column = 'mpg'

y_train_a = train_a[target_column]
y_train_b = train_b[target_column]
y_train_c = train_c[target_column]
y_train_d = train_d[target_column]

model_a = sm.OLS(y_train_a, x_train_a)
model_b = sm.OLS(y_train_b, x_train_b)
model_c = sm.OLS(y_train_c, x_train_c)
model_d = sm.OLS(y_train_d, x_train_d)

fitted_model_a = model_a.fit()
fitted_model_b = model_b.fit()
fitted_model_c = model_c.fit()
fitted_model_d = model_d.fit()

summarize(fitted_model_a), summarize(fitted_model_b), summarize(fitted_model_c), summarize(fitted_model_d)
```




    (                 coef  std err       t  P>|t|
     intercept    -16.8617    5.428  -3.106  0.002
     horsepower    -0.0025    0.014  -0.173  0.863
     weight        -0.0074    0.001  -9.822  0.000
     cylinders     -0.3593    0.368  -0.975  0.330
     displacement   0.0136    0.008   1.729  0.085
     acceleration   0.0985    0.113   0.872  0.384
     year           0.7912    0.059  13.304  0.000,
                      coef  std err       t  P>|t|
     intercept    -13.0529    5.432  -2.403  0.017
     horsepower     0.0005    0.017   0.030  0.976
     weight        -0.0075    0.001  -8.851  0.000
     cylinders     -0.2618    0.379  -0.690  0.491
     displacement   0.0135    0.009   1.579  0.115
     acceleration   0.1292    0.124   1.044  0.297
     year           0.7286    0.060  12.070  0.000,
                        coef  std err       t  P>|t|
     intercept    -13.251200    5.312  -2.495  0.013
     horsepower     0.000005    0.016   0.000  1.000
     weight        -0.006100    0.001  -8.382  0.000
     cylinders     -0.447000    0.385  -1.162  0.246
     displacement   0.002600    0.008   0.312  0.755
     acceleration   0.069200    0.112   0.617  0.537
     year           0.736400    0.059  12.564  0.000,
                      coef  std err       t  P>|t|
     intercept    -16.1649    5.775  -2.799  0.005
     horsepower    -0.0030    0.017  -0.180  0.858
     weight        -0.0066    0.001  -8.365  0.000
     cylinders     -0.1128    0.404  -0.279  0.780
     displacement   0.0016    0.009   0.168  0.867
     acceleration   0.0301    0.122   0.246  0.806
     year           0.7819    0.063  12.379  0.000)



This cell creates all the models and fits all datasets to their individual models.

## Residual Sum of Squares


```python
y_train_pred_a = fitted_model_a.predict(x_train_a)
train_a["mpg_pred"] = y_train_pred_a
rmse_a = mean_squared_error(train_a["mpg"], train_a["mpg_pred"])

y_train_pred_b = fitted_model_b.predict(x_train_b)
train_b["mpg_pred"] = y_train_pred_b
rmse_b = mean_squared_error(train_b["mpg"], train_b["mpg_pred"])

y_train_pred_c = fitted_model_c.predict(x_train_c)
train_c["mpg_pred"] = y_train_pred_c
rmse_c = mean_squared_error(train_c["mpg"], train_c["mpg_pred"])

y_train_pred_d = fitted_model_d.predict(x_train_d)
train_d["mpg_pred"] = y_train_pred_d
rmse_d = mean_squared_error(train_d["mpg"], train_d["mpg_pred"])

rmse_a, rmse_b, rmse_c, rmse_d
```




    (10.78088667411845, 11.96507852712704, 11.205104063712366, 12.206016674637468)



This cell above calculates the residual sum of squares. We can see it is inaccurate due to how much each rmse values varies. The less they vary, the more accurate they are.

## Predictions on The Practice Test Sets


```python
x_test_a = model_matrix.transform(test_a)
y_test_pred_a = fitted_model_a.predict(x_test_a)
test_a["mpg_pred"] = y_test_pred_a

x_test_b = model_matrix.transform(test_b)
y_test_pred_b = fitted_model_b.predict(x_test_b)
test_b["mpg_pred"] = y_test_pred_b

x_test_c = model_matrix.transform(test_c)
y_test_pred_c = fitted_model_c.predict(x_test_c)
test_c["mpg_pred"] = y_test_pred_c

x_test_d = model_matrix.transform(test_d)
y_test_pred_d = fitted_model_d.predict(x_test_d)
test_d["mpg_pred"] = y_test_pred_d
```

The above cell uses the fitted models to predict the mpg of the training data set, predicting what is already known.

## Predicting The Real Test


```python
X_test = model_matrix.transform(test)

pred_a = fitted_model_a.predict(X_test)
pred_b = fitted_model_b.predict(X_test)
pred_c = fitted_model_c.predict(X_test)
pred_d = fitted_model_d.predict(X_test)

avg_pred = np.mean([pred_a, pred_b, pred_c, pred_d], axis=0)

test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
      <th>name</th>
      <th>mpg_pred</th>
      <th>mpg</th>
      <th>mpg01</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>70_chevrolet chevelle malibu_alpha_3505</td>
      <td>8</td>
      <td>310.0</td>
      <td>128</td>
      <td>3505</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu_alpha</td>
      <td>15.130576</td>
      <td>15.130576</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71_buick skylark 320_bravo_3697</td>
      <td>8</td>
      <td>351.0</td>
      <td>165</td>
      <td>3697</td>
      <td>11.5</td>
      <td>71</td>
      <td>1</td>
      <td>buick skylark 320_bravo</td>
      <td>14.803855</td>
      <td>14.803855</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>70_plymouth satellite_charlie_3421</td>
      <td>8</td>
      <td>320.0</td>
      <td>150</td>
      <td>3421</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite_charlie</td>
      <td>15.676526</td>
      <td>15.676526</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>68_amc rebel sst_delta_3418</td>
      <td>8</td>
      <td>303.0</td>
      <td>150</td>
      <td>3418</td>
      <td>12.1</td>
      <td>68</td>
      <td>1</td>
      <td>amc rebel sst_delta</td>
      <td>14.135076</td>
      <td>14.135076</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>70_ford torino_echo_3444</td>
      <td>8</td>
      <td>298.0</td>
      <td>141</td>
      <td>3444</td>
      <td>10.4</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino_echo</td>
      <td>15.308729</td>
      <td>15.308729</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>392</th>
      <td>81_ford mustang gl_charlie_2802</td>
      <td>4</td>
      <td>136.0</td>
      <td>88</td>
      <td>2802</td>
      <td>15.5</td>
      <td>81</td>
      <td>1</td>
      <td>ford mustang gl_charlie</td>
      <td>28.472574</td>
      <td>28.472574</td>
      <td>1</td>
    </tr>
    <tr>
      <th>393</th>
      <td>81_vw pickup_delta_2137</td>
      <td>4</td>
      <td>96.0</td>
      <td>51</td>
      <td>2137</td>
      <td>24.5</td>
      <td>81</td>
      <td>2</td>
      <td>vw pickup_delta</td>
      <td>33.511731</td>
      <td>33.511731</td>
      <td>1</td>
    </tr>
    <tr>
      <th>394</th>
      <td>82_dodge rampage_echo_2290</td>
      <td>4</td>
      <td>136.0</td>
      <td>81</td>
      <td>2290</td>
      <td>11.5</td>
      <td>82</td>
      <td>1</td>
      <td>dodge rampage_echo</td>
      <td>32.432247</td>
      <td>32.432247</td>
      <td>1</td>
    </tr>
    <tr>
      <th>395</th>
      <td>81_ford ranger_foxtrot_2611</td>
      <td>4</td>
      <td>118.0</td>
      <td>79</td>
      <td>2611</td>
      <td>18.5</td>
      <td>81</td>
      <td>1</td>
      <td>ford ranger_foxtrot</td>
      <td>29.900864</td>
      <td>29.900864</td>
      <td>1</td>
    </tr>
    <tr>
      <th>396</th>
      <td>84_chevy s-10_golf_2724</td>
      <td>4</td>
      <td>117.0</td>
      <td>80</td>
      <td>2724</td>
      <td>19.4</td>
      <td>84</td>
      <td>1</td>
      <td>chevy s-10_golf</td>
      <td>31.467401</td>
      <td>31.467401</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>397 rows × 12 columns</p>
</div>



Here we predict the mpg of the test data set, this variable is then exported below, as the mpg value and as a binary variable where the value is a 1 if above the median mpg.


```python
test['mpg'] = avg_pred

test[['ID', 'mpg']].to_csv('predicted_4mdl.csv', index=False)
```


```python
median_mpg = test['mpg'].median()
test['mpg01'] = (test['mpg'] > median_mpg).astype(int)
test[['ID', 'mpg01']].to_csv('predicted_mpg01_4mdl.csv', index=False)
```
