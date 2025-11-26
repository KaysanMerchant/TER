---
layout: post
title: "Auto Data Logistic Regression and Predictions"
author: Kaysan Merchant
date: 2025-11-26
---

# Introduction

This post is about using the Auto dataset provided by ISLP resources to practice logistic regression and create a prediction model to use on learning data and test data. This part of the post contains part a and b, where we begin with logistic regression and graphing. Part c will split the data into training and test data and will be added later.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('Auto.csv')

# Fix horsepower column (convert "?" to NaN)
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

# Remove missing data
df = df.dropna()

# Create binary mpg01 column
median_mpg = df['mpg'].median()
df['mpg01'] = (df['mpg'] > median_mpg).astype(int)
```


```python
col = 'cylinders'
X = df[[col]].values
y = df['mpg01'].values

model = LogisticRegression()
model.fit(X, y)

x_vals = np.linspace(df[col].min(), df[col].max(), 300).reshape(-1, 1)
y_pred = model.predict_proba(x_vals)[:, 1]

plt.figure(figsize=(7,5))
plt.scatter(df[col], df['mpg01'], alpha=0.5)
plt.plot(x_vals, y_pred, linewidth=3)
plt.xlabel(col)
plt.ylabel("mpg01")
plt.title("Cylinders vs mpg01")
plt.grid(True)
plt.show()

```


    
<img src="{{ '/images/LogisticRegression/output_1_0.png' | relative_url }}" alt="Cylinders vs mpg01" style="max-width:100%;height:auto;" />
    


## Cylinders vs mpg01

This graph shows a strong inverse relationship between the number of engine cylinders and fuel efficiency. 
Cars with **4 cylinders** almost always achieve high gas mileage (`mpg01 = 1`), while cars with **6 or 8 cylinders** 
are almost exclusively low-MPG vehicles (`mpg01 = 0`).

The logistic sigmoid curve drops sharply as cylinder count increases, indicating that adding cylinders greatly 
reduces the probability of high MPG.

**Conclusion:** Cylinders is a strong and reliable predictor of mpg01.


```python
col = 'displacement'
X = df[[col]].values
y = df['mpg01'].values

model = LogisticRegression()
model.fit(X, y)

x_vals = np.linspace(df[col].min(), df[col].max(), 300).reshape(-1, 1)
y_pred = model.predict_proba(x_vals)[:, 1]

plt.figure(figsize=(7,5))
plt.scatter(df[col], df['mpg01'], alpha=0.5)
plt.plot(x_vals, y_pred, linewidth=3)
plt.xlabel(col)
plt.ylabel("mpg01")
plt.title("Displacement vs mpg01")
plt.grid(True)
plt.show()
```


    
<img src="{{ '/images/LogisticRegression/output_3_0.png' | relative_url }}" alt="Displacement vs mpg01" style="max-width:100%;height:auto;" />
    


## Displacement vs mpg01

Engine displacement shows a clear negative relationship with fuel efficiency. Cars with 
**low displacement (around 100–150 cubic inches)** tend to have high MPG, while those with 
**high displacement (250–400 cubic inches)** almost always fall into the low-MPG category.

The sigmoid curve declines smoothly across the displacement range, reflecting the strong 
negative association between displacement and the probability of high MPG.

**Conclusion:** Displacement is one of the dataset’s strongest predictors of mpg01.


```python
col = 'horsepower'
X = df[[col]].values
y = df['mpg01'].values

model = LogisticRegression()
model.fit(X, y)

x_vals = np.linspace(df[col].min(), df[col].max(), 300).reshape(-1, 1)
y_pred = model.predict_proba(x_vals)[:, 1]

plt.figure(figsize=(7,5))
plt.scatter(df[col], df['mpg01'], alpha=0.5)
plt.plot(x_vals, y_pred, linewidth=3)
plt.xlabel(col)
plt.ylabel("mpg01")
plt.title("Horsepower vs mpg01")
plt.grid(True)
plt.show()
```


    
<img src="{{ '/images/LogisticRegression/output_5_0.png' | relative_url }}" alt="Horsepower vs mpg01" style="max-width:100%;height:auto;" />
    


## Horsepower vs mpg01

Horsepower is strongly and negatively correlated with fuel efficiency. Vehicles with 
**horsepower under 90** generally produce high MPG, whereas cars with horsepower above 120–150 
are almost always low-MPG.

The logistic sigmoid curve shows a steep downward slope, illustrating the fast decline in 
probability of high MPG as horsepower increases.

**Conclusion:** Horsepower is a highly informative predictor of mpg01.


```python
col = 'weight'
X = df[[col]].values
y = df['mpg01'].values

model = LogisticRegression()
model.fit(X, y)

x_vals = np.linspace(df[col].min(), df[col].max(), 300).reshape(-1, 1)
y_pred = model.predict_proba(x_vals)[:, 1]

plt.figure(figsize=(7,5))
plt.scatter(df[col], df['mpg01'], alpha=0.5)
plt.plot(x_vals, y_pred, linewidth=3)
plt.xlabel(col)
plt.ylabel("mpg01")
plt.title("Weight vs mpg01")
plt.grid(True)
plt.show()
```


    
<img src="{{ '/images/LogisticRegression/output_7_0.png' | relative_url }}" alt="Weight vs mpg01" style="max-width:100%;height:auto;" />
    


## Weight vs mpg01

Weight exhibits one of the clearest separations in the dataset. Lighter cars (1500–2500 lbs) 
are strongly associated with high fuel efficiency, while heavier cars (3000–5000 lbs) 
tend to have low MPG.

The sigmoid curve drops steeply as weight increases, showing that heavier vehicles have a 
much lower probability of achieving high MPG.

**Conclusion:** Weight is likely the single strongest predictor of mpg01.


```python
col = 'acceleration'
X = df[[col]].values
y = df['mpg01'].values

model = LogisticRegression()
model.fit(X, y)

x_vals = np.linspace(df[col].min(), df[col].max(), 300).reshape(-1, 1)
y_pred = model.predict_proba(x_vals)[:, 1]

plt.figure(figsize=(7,5))
plt.scatter(df[col], df['mpg01'], alpha=0.5)
plt.plot(x_vals, y_pred, linewidth=3)
plt.xlabel(col)
plt.ylabel("mpg01")
plt.title("Acceleration vs mpg01")
plt.grid(True)
plt.show()
```


    
<img src="{{ '/images/LogisticRegression/output_9_0.png' | relative_url }}" alt="Acceleration vs mpg01" style="max-width:100%;height:auto;" />
    


## Acceleration vs mpg01

Acceleration shows a weaker relationship with mpg01 compared to weight or horsepower. Although 
there is overlap, cars with slower acceleration times (larger values) tend to be more fuel-efficient.

The sigmoid curve here is more gradual, reflecting only moderate predictive power.

**Conclusion:** Acceleration contributes some predictive information but is not a major determinant of mpg01.


```python
col = 'year'
X = df[[col]].values
y = df['mpg01'].values

model = LogisticRegression()
model.fit(X, y)

x_vals = np.linspace(df[col].min(), df[col].max(), 300).reshape(-1, 1)
y_pred = model.predict_proba(x_vals)[:, 1]

plt.figure(figsize=(7,5))
plt.scatter(df[col], df['mpg01'], alpha=0.5)
plt.plot(x_vals, y_pred, linewidth=3)
plt.xlabel(col)
plt.ylabel("mpg01")
plt.title("Year vs mpg01")
plt.grid(True)
plt.show()
```


    
<img src="{{ '/images/LogisticRegression/output_11_0.png' | relative_url }}" alt="Year vs mpg01" style="max-width:100%;height:auto;" />
    


## Year vs mpg01

Model year shows a clear positive association with fuel efficiency. As the year increases, the 
probability of high MPG increases. Newer cars (late 1970s to early 1980s) were generally more efficient, 
likely due to technological improvements and regulatory changes.

The sigmoid curve rises with model year, capturing the positive trend.

**Conclusion:** Year is a moderately strong predictor of mpg01 and reflects improvements in automotive technology.


```python
col = 'origin'
X = df[[col]].values
y = df['mpg01'].values

model = LogisticRegression()
model.fit(X, y)

x_vals = np.linspace(df[col].min(), df[col].max(), 300).reshape(-1, 1)
y_pred = model.predict_proba(x_vals)[:, 1]

plt.figure(figsize=(7,5))
plt.scatter(df[col], df['mpg01'], alpha=0.5)
plt.plot(x_vals, y_pred, linewidth=3)
plt.xlabel(col)
plt.ylabel("mpg01")
plt.title("Origin vs mpg01")
plt.grid(True)
plt.show()
```


    
<img src="{{ '/images/LogisticRegression/output_13_0.png' | relative_url }}" alt="Origin vs mpg01" style="max-width:100%;height:auto;" />
    


## Origin vs mpg01

The origin variable (1 = U.S., 2 = Europe, 3 = Japan) reveals that:

- U.S. cars tend to have low MPG,
- Japanese and European cars tend to have higher MPG.

The sigmoid curve rises from origin 1 to origin 3, indicating that imported cars are more likely 
to achieve high fuel efficiency.

**Conclusion:** Origin is a useful predictor, though less powerful than weight or displacement.
