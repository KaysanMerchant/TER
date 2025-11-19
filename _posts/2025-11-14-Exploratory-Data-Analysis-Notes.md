---
layout: post
title: "Exploratory Data Analysis Notes"
author: Kaysan Merchant
date: 2025-11-14
---

# Introduction

The page introduces Exploratory Data Analysis (EDA) as a crucial early step in any data science workflow, performed after gathering or engineering data but before modelling. The purpose is to understand the structure, quality, patterns, and limitations of the data without making strong assumptions. EDA is iterative: insights from exploration guide modelling, and modelling can reveal areas that need further exploration.

# Types of Data

## Categorial Variables
Categorical variables describe groups or labels.
* Nominal: No inherent order (e.g., colours, country).
* Ordinal: Ordered categories (e.g., “low/medium/high”)

Common tools include frequency tables, bar charts, and waffle charts. The guide discourages pie charts due to poor angle comparison by humans.

## Numerical Data
* Continuous variables (e.g., height, temperature) can take any value within a range.
    * Typical summaries: mean, median, standard deviation, quantiles.
    * Visualizations: histograms, density plots, boxplots, violin plots.
* Discrete variables (e.g., event counts) take distinct integer values.
    * Visualizations: bar charts, dot plots.
* Bivariate/multivariate numerical analysis uses scatter plots, correlation matrices, heatmaps, and pair plots to reveal relationships, clusters, or patterns across multiple features.
* Key considerations include checking for outliers, understanding distribution shape, and applying transformations (e.g., log scaling) when helpful.

# Visualisation
* Multivariate Visualization
    * For multiple features, the guide suggests avoiding complex 3D plots unless necessary, as they are difficult to interpret. For high-dimensional datasets, dimensionality reduction techniques (e.g., PCA, LDA) can simplify the data for visualization.
* Text Data
    * Unstructured text is explored using tools like word frequency counts, word clouds, and topic modelling. The goal is to reduce noise, identify important terms, and uncover thematic structures within the text corpus.
* Image Data
    * Image datasets often require dimensionality reduction (e.g., PCA on pixel intensities). Visualising principal components can reveal meaningful differences—such as facial structure variations—by showing the dominant patterns encoded in the image data.

# Theory

The page concludes with suggested readings and supplementary materials on EDA, descriptive statistics, and practical introductions to exploratory techniques. These resources provide deeper theoretical grounding and practical examples for applying EDA effectively.