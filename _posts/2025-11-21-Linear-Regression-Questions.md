---
layout: post
title: "Linear Regression ISLP Practice Questions"
author: Kaysan Merchant
date: 2025-11-21
---

# Introduction

This analysis explores the relationship between automobile fuel efficiency (mpg) and horsepower using simple linear regression techniques. By fitting a model, visualizing the data, and examining diagnostic plots, we assess how well horsepower predicts mpg and whether the assumptions of linear regression are satisfied. <a href="#multiple-regression-correlations-diagnostics-interactions-and-transformationsquestion-9">**Question 9 is located below this question.**</a>

# Question 8

```python
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("Auto.csv")

# Ensure horsepower is numeric
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df = df.dropna(subset=['horsepower', 'mpg'])

# Fit the simple linear regression model
X = sm.add_constant(df['horsepower'])
y = df['mpg']
model = sm.OLS(y, X).fit()
```

# (a) Linear Regression Analysis

We fit a simple linear regression model using `mpg` as the response and `horsepower` as the predictor.

### **i. Is there a relationship between horsepower and mpg?**
Yes.  
The p-value for horsepower is extremely small (approximately 0), indicating a statistically significant relationship between horsepower and mpg.

---

### **ii. How strong is the relationship?**
The R² value is about **0.60**, meaning approximately **60% of the variation in mpg** can be explained by horsepower alone.  
This indicates a moderately strong linear relationship.

---

### **iii. Is the relationship positive or negative?**
The coefficient for horsepower is **negative**, meaning that as horsepower increases, mpg decreases.  
Thus, the relationship is **negative**.

---

### **iv. Predicted mpg for horsepower = 98**
Using the fitted regression model:

- **Predicted mpg:** 24.47  
- **95% Confidence Interval:** (23.97, 24.96)  
- **95% Prediction Interval:** (14.81, 34.12)

Interpretation:
- The confidence interval gives the likely range for the *mean* mpg of all cars with horsepower = 98.
- The prediction interval gives the likely range for an *individual* car.


```python
# Scatterplot and regression line
fig, ax = plt.subplots(figsize=(8,6))

ax.scatter(df['horsepower'], df['mpg'], alpha=0.7)
ax.plot(df['horsepower'], model.predict(X), linewidth=2)

ax.set_xlabel("Horsepower")
ax.set_ylabel("MPG")
ax.set_title("MPG vs Horsepower with Regression Line")

plt.show()
```


    
<img src="{{ '/images/AutoDataImages/output_3_0.png' | relative_url }}" alt="MPG vs Horsepower scatter with regression line" style="max-width:100%;height:auto;" />
    


# (b) Scatterplot and Regression Line

A scatterplot of `mpg` vs `horsepower` shows a downward trend.  
The least-squares regression line clearly shows a **negative linear relationship**.


```python
# Standard Statsmodels regression diagnostics
fig = plt.figure(figsize=(10,8))
sm.graphics.plot_regress_exog(model, "horsepower", fig=fig)

plt.show()
```


    
<img src="{{ '/images/AutoDataImages/output_5_0.png' | relative_url }}" alt="Regression diagnostics plots" style="max-width:100%;height:auto;" />
    


# (c) Diagnostic Plots and Interpretation

### **1. Residuals vs Fitted Plot**
- Shows a curved pattern.
- Indicates potential **non-linearity**.
- Residual variance increases at lower fitted values → possible **heteroscedasticity**.


```python
fig = plt.figure(figsize=(6,6))
sm.qqplot(model.resid, line='45', fit=True)
plt.title("QQ Plot of Residuals")

plt.show()
```


    <Figure size 600x600 with 0 Axes>



    
<img src="{{ '/images/AutoDataImages/output_7_1.png' | relative_url }}" alt="QQ plot of residuals" style="max-width:100%;height:auto;" />
    


### **2. QQ Plot**
- Deviations from the reference line at the tails.
- Suggests **non-normality** of residuals.


```python
fitted_vals = model.fittedvalues
residuals = model.resid

plt.figure(figsize=(8,6))
plt.scatter(fitted_vals, residuals, alpha=0.7)

plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")

plt.show()
```


    
<img src="{{ '/images/AutoDataImages/output_9_0.png' | relative_url }}" alt="Residuals vs fitted values" style="max-width:100%;height:auto;" />
    


### **3. Residuals vs Horsepower**
- Clear curvature again, reinforcing that a simple straight-line model does not fully capture the relationship.

# **Conclusion**
A simple linear regression captures the general negative trend between horsepower and mpg, but diagnostic plots indicate violations of linear regression assumptions.  
A more suitable model may require:

- Polynomial regression (e.g., quadratic model)  
- Log or Box–Cox transformation  
- Weighted least squares  

These could improve model fit and correct assumption violations.

# Multiple Regression, Correlations, Diagnostics, Interactions, and Transformations(Question 9)
**Dataset:** `Auto.csv`

This notebook produces:
- (a) a scatterplot matrix of all variables (excluding `name`),
- (b) correlation matrix,
- (c) a multiple linear regression of `mpg` on all other predictors (except `name`) with ANOVA,
- (d) regression diagnostic plots and identification of influential observations,
- (e) a model with interactions,
- (f) several transformations of `horsepower` and comparison of fit.


```python
# Cell 1: imports + data load + cleanup
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.regressionplots import influence_plot

# Load dataset
df = pd.read_csv("Auto.csv")

# Convert horsepower to numeric and drop rows with missing important values
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df = df.dropna(subset=['mpg','cylinders','displacement','horsepower','weight','acceleration','year','origin'])

# Quick check
print("Data shape after cleaning:", df.shape)
df.head()
```

    Data shape after cleaning: (392, 9)





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
      <th>mpg</th>
      <th>cylinders</th>
      <th>displacement</th>
      <th>horsepower</th>
      <th>weight</th>
      <th>acceleration</th>
      <th>year</th>
      <th>origin</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.0</td>
      <td>8</td>
      <td>307.0</td>
      <td>130.0</td>
      <td>3504</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>chevrolet chevelle malibu</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>8</td>
      <td>350.0</td>
      <td>165.0</td>
      <td>3693</td>
      <td>11.5</td>
      <td>70</td>
      <td>1</td>
      <td>buick skylark 320</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.0</td>
      <td>8</td>
      <td>318.0</td>
      <td>150.0</td>
      <td>3436</td>
      <td>11.0</td>
      <td>70</td>
      <td>1</td>
      <td>plymouth satellite</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>8</td>
      <td>304.0</td>
      <td>150.0</td>
      <td>3433</td>
      <td>12.0</td>
      <td>70</td>
      <td>1</td>
      <td>amc rebel sst</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17.0</td>
      <td>8</td>
      <td>302.0</td>
      <td>140.0</td>
      <td>3449</td>
      <td>10.5</td>
      <td>70</td>
      <td>1</td>
      <td>ford torino</td>
    </tr>
  </tbody>
</table>
</div>



## (a) Scatterplot matrix (all variables except `name`)


```python
# Cell 2: Scatterplot matrix
vars_for_plot = ['mpg','cylinders','displacement','horsepower','weight','acceleration','year','origin']
plt.figure(figsize=(12,12))
# pandas scatter_matrix uses matplotlib internally
scatter_matrix(df[vars_for_plot], alpha=0.6, diagonal='kde', figsize=(12,12))
plt.suptitle("Scatterplot Matrix of Variables", y=0.92)
plt.show()
```


    <Figure size 1200x1200 with 0 Axes>



    
<img src="{{ '/images/AutoDataImages2/output_3_1.png' | relative_url }}" alt="Scatterplot matrix" style="max-width:100%;height:auto;" />
    


## (b) Correlation matrix


```python
# Cell 3: Correlation matrix
corr_matrix = df[vars_for_plot].corr()
print("Correlation matrix (rounded):\n")
print(corr_matrix.round(3))
```

    Correlation matrix (rounded):
    
                    mpg  cylinders  displacement  horsepower  weight  \
    mpg           1.000     -0.778        -0.805      -0.778  -0.832   
    cylinders    -0.778      1.000         0.951       0.843   0.898   
    displacement -0.805      0.951         1.000       0.897   0.933   
    horsepower   -0.778      0.843         0.897       1.000   0.865   
    weight       -0.832      0.898         0.933       0.865   1.000   
    acceleration  0.423     -0.505        -0.544      -0.689  -0.417   
    year          0.581     -0.346        -0.370      -0.416  -0.309   
    origin        0.565     -0.569        -0.615      -0.455  -0.585   
    
                  acceleration   year  origin  
    mpg                  0.423  0.581   0.565  
    cylinders           -0.505 -0.346  -0.569  
    displacement        -0.544 -0.370  -0.615  
    horsepower          -0.689 -0.416  -0.455  
    weight              -0.417 -0.309  -0.585  
    acceleration         1.000  0.290   0.213  
    year                 0.290  1.000   0.182  
    origin               0.213  0.182   1.000  


## (c) Multiple linear regression: mpg ~ all other predictors (except name)
We use `statsmodels.formula.api.ols()` so we can use `anova_lm()` easily.


```python
# Cell 4: Multiple regression using formula API and ANOVA
formula = "mpg ~ cylinders + displacement + horsepower + weight + acceleration + year + origin"
model_formula = smf.ols(formula, data=df).fit()

# Print full summary
print(model_formula.summary())

# ANOVA (type II)
anova_results = anova_lm(model_formula, typ=2)
print("\nANOVA (Type II):\n", anova_results)

# predictors with p < 0.05
sig_preds = model_formula.pvalues[model_formula.pvalues < 0.05]
print("\nPredictors with p-value < 0.05:\n", sig_preds)

# Interpret coefficient for 'year'
coef_year = model_formula.params['year']
print(f"\nCoefficient for 'year' = {coef_year:.4f}")
print("Interpretation: holding other predictors constant, a 1-unit increase in 'year' is associated with this change in mpg.")
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                    mpg   R-squared:                       0.821
    Model:                            OLS   Adj. R-squared:                  0.818
    Method:                 Least Squares   F-statistic:                     252.4
    Date:                Sun, 23 Nov 2025   Prob (F-statistic):          2.04e-139
    Time:                        15:19:10   Log-Likelihood:                -1023.5
    No. Observations:                 392   AIC:                             2063.
    Df Residuals:                     384   BIC:                             2095.
    Df Model:                           7                                         
    Covariance Type:            nonrobust                                         
    ================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------
    Intercept      -17.2184      4.644     -3.707      0.000     -26.350      -8.087
    cylinders       -0.4934      0.323     -1.526      0.128      -1.129       0.142
    displacement     0.0199      0.008      2.647      0.008       0.005       0.035
    horsepower      -0.0170      0.014     -1.230      0.220      -0.044       0.010
    weight          -0.0065      0.001     -9.929      0.000      -0.008      -0.005
    acceleration     0.0806      0.099      0.815      0.415      -0.114       0.275
    year             0.7508      0.051     14.729      0.000       0.651       0.851
    origin           1.4261      0.278      5.127      0.000       0.879       1.973
    ==============================================================================
    Omnibus:                       31.906   Durbin-Watson:                   1.309
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               53.100
    Skew:                           0.529   Prob(JB):                     2.95e-12
    Kurtosis:                       4.460   Cond. No.                     8.59e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 8.59e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.
    
    ANOVA (Type II):
                        sum_sq     df           F        PR(>F)
    cylinders       25.791491    1.0    2.329125  1.277965e-01
    displacement    77.612668    1.0    7.008884  8.444649e-03
    horsepower      16.739754    1.0    1.511699  2.196328e-01
    weight        1091.631693    1.0   98.580813  7.874953e-21
    acceleration     7.358417    1.0    0.664509  4.154780e-01
    year          2402.249906    1.0  216.937408  3.055983e-39
    origin         291.134494    1.0   26.291171  4.665681e-07
    Residual      4252.212530  384.0         NaN           NaN
    
    Predictors with p-value < 0.05:
     Intercept       2.401841e-04
    displacement    8.444649e-03
    weight          7.874953e-21
    year            3.055983e-39
    origin          4.665681e-07
    dtype: float64
    
    Coefficient for 'year' = 0.7508
    Interpretation: holding other predictors constant, a 1-unit increase in 'year' is associated with this change in mpg.


## (d) Diagnostic plots and influential point detection
We'll produce:
- Residuals vs fitted
- QQ plot of residuals
- Scale-Location (sqrt(|standardized residual|) vs fitted)
- Influence plot (leverage vs Cook's distance)
And list observations with high Cook's D or high leverage.


```python
# Cell 5: Diagnostic plots
fitted_vals = model_formula.fittedvalues
residuals = model_formula.resid

# 1) Residuals vs Fitted
plt.figure(figsize=(8,6))
plt.scatter(fitted_vals, residuals, alpha=0.7)
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.show()

# 2) QQ plot
plt.figure(figsize=(6,6))
sm.qqplot(residuals, line='45', fit=True)
plt.title("QQ Plot of Residuals")
plt.show()

# 3) Scale-Location plot
standardized_resid = (residuals - np.mean(residuals)) / np.std(residuals, ddof=1)
plt.figure(figsize=(8,6))
plt.scatter(fitted_vals, np.sqrt(np.abs(standardized_resid)), alpha=0.7)
plt.xlabel("Fitted values")
plt.ylabel("Sqrt(|Standardized residuals|)")
plt.title("Scale-Location Plot")
plt.show()

# 4) Influence plot
plt.figure(figsize=(8,6))
influence_plot(model_formula, criterion="cooks")
plt.title("Influence plot (Cook's distance vs Leverage)")
plt.show()

# Identify potentially influential observations
influence = model_formula.get_influence()
summary_frame = influence.summary_frame()
n = len(df)
p = len(model_formula.params)
cooks_threshold = 4 / n
leverage_threshold = 2 * p / n

print(f"Cook's D threshold = {cooks_threshold:.6f}")
print(f"Leverage threshold (2*p/n) = {leverage_threshold:.6f}")

high_cooks = summary_frame[summary_frame['cooks_d'] > cooks_threshold].sort_values('cooks_d', ascending=False)
high_leverage = summary_frame[summary_frame['hat_diag'] > leverage_threshold].sort_values('hat_diag', ascending=False)

print("\nTop rows with Cook's D > 4/n (if any):")
print(high_cooks[['cooks_d','student_resid','hat_diag']].head())

print("\nTop rows with hat_diag > 2*p/n (if any):")
print(high_leverage[['hat_diag','student_resid','cooks_d']].head())
```


    
<img src="{{ '/images/AutoDataImages2/output_9_0.png' | relative_url }}" alt="Residuals vs Fitted (multiple regression)" style="max-width:100%;height:auto;" />
    



    <Figure size 600x600 with 0 Axes>



    
<img src="{{ '/images/AutoDataImages2/output_9_2.png' | relative_url }}" alt="Scale-Location plot" style="max-width:100%;height:auto;" />
    



    
<img src="{{ '/images/AutoDataImages2/output_9_3.png' | relative_url }}" alt="Influence plot (Cook's distance vs leverage)" style="max-width:100%;height:auto;" />
    



    <Figure size 800x600 with 0 Axes>



    
<img src="{{ '/images/AutoDataImages2/output_9_5.png' | relative_url }}" alt="Influence diagnostics image" style="max-width:100%;height:auto;" />
    


    Cook's D threshold = 0.010204
    Leverage threshold (2*p/n) = 0.040816
    
    Top rows with Cook's D > 4/n (if any):
          cooks_d  student_resid  hat_diag
    13   0.077801      -1.632924  0.189913
    393  0.055824       2.968385  0.049172
    326  0.048772       3.690246  0.028743
    386  0.036773       2.953219  0.033265
    325  0.029605       3.494823  0.019567
    
    Top rows with hat_diag > 2*p/n (if any):
        hat_diag  student_resid   cooks_d
    13  0.189913      -1.632924  0.077801
    28  0.089541       0.803682  0.007948
    26  0.067370       0.494049  0.002208
    27  0.066673       0.839418  0.006297
    8   0.062449       1.176401  0.011511


## (e) Models with interactions
We'll add three interactions: horsepower×weight, horsepower×year, displacement×acceleration and test their significance.


```python
# Cell 6: Model with interactions (formula style)
formula_int = ("mpg ~ cylinders + displacement + horsepower + weight + acceleration + year + origin "
               "+ horsepower:weight + horsepower:year + displacement:acceleration")
model_int_formula = smf.ols(formula_int, data=df).fit()

print(model_int_formula.summary())

# Print p-values for the interaction terms
print("\nInteraction term p-values:")
print(model_int_formula.pvalues[['horsepower:weight','horsepower:year','displacement:acceleration']])
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                    mpg   R-squared:                       0.874
    Model:                            OLS   Adj. R-squared:                  0.871
    Method:                 Least Squares   F-statistic:                     265.2
    Date:                Sun, 23 Nov 2025   Prob (F-statistic):          7.86e-165
    Time:                        15:19:10   Log-Likelihood:                -954.58
    No. Observations:                 392   AIC:                             1931.
    Df Residuals:                     381   BIC:                             1975.
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    =============================================================================================
                                    coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------------
    Intercept                   -55.9588     10.828     -5.168      0.000     -77.248     -34.670
    cylinders                     0.3436      0.283      1.215      0.225      -0.212       0.899
    displacement                  0.0185      0.010      1.781      0.076      -0.002       0.039
    horsepower                    0.3188      0.105      3.031      0.003       0.112       0.526
    weight                       -0.0083      0.001     -9.114      0.000      -0.010      -0.006
    acceleration                  0.1254      0.135      0.926      0.355      -0.141       0.392
    year                          1.4221      0.131     10.866      0.000       1.165       1.679
    origin                        0.7407      0.242      3.066      0.002       0.266       1.216
    horsepower:weight          3.828e-05   5.84e-06      6.559      0.000    2.68e-05    4.98e-05
    horsepower:year              -0.0069      0.001     -5.222      0.000      -0.009      -0.004
    displacement:acceleration    -0.0017      0.001     -2.372      0.018      -0.003      -0.000
    ==============================================================================
    Omnibus:                       38.798   Durbin-Watson:                   1.602
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               72.800
    Skew:                           0.583   Prob(JB):                     1.55e-16
    Kurtosis:                       4.760   Cond. No.                     3.09e+07
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 3.09e+07. This might indicate that there are
    strong multicollinearity or other numerical problems.
    
    Interaction term p-values:
    horsepower:weight            1.765587e-10
    horsepower:year              2.915279e-07
    displacement:acceleration    1.819133e-02
    dtype: float64


## (f) Transformations of horsepower and comparison
We try horsepower^2, sqrt(horsepower), and log(horsepower) (replace zero with NaN if needed) and compare adjusted R² and AIC.


```python
# Cell 7: Transformations and comparison
df_tr = df.copy()
df_tr['hp_sq'] = df_tr['horsepower'] ** 2
df_tr['hp_sqrt'] = np.sqrt(df_tr['horsepower'])
df_tr['hp_log'] = np.log(df_tr['horsepower'].replace(0, np.nan))

formula_base = "mpg ~ cylinders + displacement + horsepower + weight + acceleration + year + origin"
formula_hp_sq = "mpg ~ cylinders + displacement + hp_sq + weight + acceleration + year + origin"
formula_hp_sqrt = "mpg ~ cylinders + displacement + hp_sqrt + weight + acceleration + year + origin"
formula_hp_log = "mpg ~ cylinders + displacement + hp_log + weight + acceleration + year + origin"

mod_base = smf.ols(formula_base, data=df_tr).fit()
mod_hp_sq = smf.ols(formula_hp_sq, data=df_tr).fit()
mod_hp_sqrt = smf.ols(formula_hp_sqrt, data=df_tr).fit()
mod_hp_log = smf.ols(formula_hp_log, data=df_tr).fit()

comparison = pd.DataFrame({
    'model': ['base (hp)','hp^2','sqrt(hp)','log(hp)'],
    'adj_r2': [mod_base.rsquared_adj, mod_hp_sq.rsquared_adj, mod_hp_sqrt.rsquared_adj, mod_hp_log.rsquared_adj],
    'AIC': [mod_base.aic, mod_hp_sq.aic, mod_hp_sqrt.aic, mod_hp_log.aic]
})
print(comparison.round(4))

print("\nNote: check coefficients and p-values of the transformed-variable models to see whether the transformation improves interpretability or fit.")
```

           model  adj_r2        AIC
    0  base (hp)  0.8182  2062.9495
    1       hp^2  0.8193  2060.6524
    2   sqrt(hp)  0.8237  2050.9587
    3    log(hp)  0.8340  2027.3834
    
    Note: check coefficients and p-values of the transformed-variable models to see whether the transformation improves interpretability or fit.


## Final brief comments / interpretation (summary)
- Use the ANOVA table printed earlier to decide whether the full set of predictors collectively explains a significant portion of `mpg` variation (look at the PR(>F) for the model or the F-statistic in the model summary).  
- Check individual p-values from the multiple regression summary to see which predictors are statistically significant.  
- The `year` coefficient: the printed coefficient value indicates the change in mpg associated with a one-unit increase in `year` (i.e., newer model year) **holding the other predictors constant**. A positive coefficient suggests newer cars get better mpg controlling for the other variables.  
- Diagnostics: inspect Residuals vs Fitted and Scale-Location for heteroscedasticity or non-linearity, the QQ plot for normality issues, and the influence plot + Cook's D / hat values to find potentially influential observations.  
- Interactions: check p-values of interaction terms printed above — significant interaction terms imply the effect of one predictor depends on the level of another predictor.  
- Transformations: compare adjusted R² and AIC among models with transformed horsepower — choose a transform that improves adjusted R² (or reduces AIC) and makes substantive sense.