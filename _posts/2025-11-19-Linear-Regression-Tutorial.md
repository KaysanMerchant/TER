---
layout: post
title: "Linear Regression Tutorial"
author: Kaysan Merchant
date: 2025-11-19
---

# Overview

This post will go over Mr. Andrade's linear regression document and how I fixed the error within it.

Forked Github Repository: [Lin_Reg_Fix](https://github.com/KaysanMerchant/Lin_Reg_Fix)

The following repository is forked from Mr. Andrade, with 3 commits fixing the linear regression tutorial document in the repository.

# What I Fixed

I fixed a few things. The first thing I fixed was an outdated library used in the `.ipynb` file. The library he was using(listed in the `requirements.txt` file) was `sklearn`, which was deprecated. I changed it to `scikit-learn` which is the updated version.

Those changes caused some errors within the graphing and variable processing in the code. That was the second thing I fixed. I changed the processing on some arrays from and outdated library to using `numpy` which fixed majority of the errors. 

Another addition to the files was future-proofing the libraries used. In the `requirements.txt` file, you can type in the name of a library. If you do not specify the version, it will use the most recent. If your document goes unchanged but the libraries update, it might mess with the code you have. This is why I needed to add `==version` to the end of each library in the file. Version is obviiously a palceholder for the actual version number used of that library. For example, scikit-learn might have used version `1.1.1` which would be in the place of `version`.

# Summary

Key Issues:
* Outdated libraries
* Unspecified library version
* Deprecated functions within code

Fixes:
* Using a workaround for array values and processing
* Different graphing methods to produce the same or similar graphs
* Replacing the libraries within `requirements.txt` with the correct libraries(updated)
    * Specifying the version of the libraries within `requirements.txt`

# Homework

## 7. Predicting House Prices Using a Log-Transformed Linear Model

After finishing the basic linear regression example, we move on to a new dataset, **`log_regression_example.csv`**, which explores how house size relates to price.

### Why a Simple Linear Model Didn’t Work
When we initially plotted **Price vs. Size**, the pattern wasn’t linear — prices increased more sharply for larger homes.  
To correct this, we applied a **log transformation** to the Size variable. This straightened the relationship and allowed us to fit a meaningful linear regression model.

### Final Regression Model
After transforming the predictor, the fitted model becomes:

<div style="text-align: center; margin: 20px 0;">
  <p>Price = 0.2089 &middot; log(Size) - 0.0273</p>
</div>

- **Size** is measured in square meters  
- **Price** is expressed in millions of dollars  

### Example: Predicting the Price of a 1000 m² House

<div style="text-align: center; margin: 20px 0;">
  <p>Price(1000) = 0.2089 &middot; log(1000) - 0.0273 &asymp; 1.42 million</p>
</div>

### Using the Model for Your Own Predictions
Simply plug your home’s size into the equation:

<div style="text-align: center; margin: 20px 0;">
  <p>Price(your size) = 0.2089 &middot; log(your size) - 0.0273</p>
</div>

This gives a quick estimate of market value based on the log-adjusted size relationship.

## Property Size vs. Price

The following scatter plot shows a funny distribution that shows a hidden message within the data provided:

<img src="{{ '/images/lin_reg_img/img-2025-11-19-15-22-05.png' | relative_url }}" alt="Property Size vs. Price" style="max-width:100%;height:auto;" />