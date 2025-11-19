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