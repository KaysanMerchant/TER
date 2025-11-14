---
layout: post
title: Projects
custom_css: /assets/css/custom-styles.css
---

<link rel="stylesheet" href="{{ '/assets/css/custom-styles.css' | relative_url }}">

<details>
<summary>Anscombe's Quartet Analysis, Nov. 12, 2025</summary>

<h3>Overview</h3>
In this analysis, we explore the Anscombe's Quartet dataset, visualize distributions with boxplots, and compute statistical measures like mean, variance, and coefficient of determination.
<br>
<h3>Purpose</h3>
We created multiple different graphs to show the data in different ways. This is due to the data showing similar values mathematical but looking drastically different on graphs.
<br>
<h3>Embedded PDF Report</h3>
<iframe src="AnsQuartFiles/Anscombe Quartet Data Visualisation.pdf" width="100%" height="600px">
  This browser does not support PDFs. Please download the PDF to view it: <a href="AnsQuartFiles/Anscombe Quartet Data Visualisation.pdf">Download PDF</a>
</iframe>
<h3>Interactive scatter</h3>
<iframe src="AnsQuartFiles/anscombe_plotly.html" width="100%" height="300px">
  This browser does not support PDFs. Please download the PDF to view it: <a href="AnsQuartFiles/anscombe_plotly.html">Download the interactive graph</a>
</iframe>

</details>

<details>
<summary>Exploratory Data Analysis Summary, Nov. 14, 2025</summary>
<h2>Introduction</h2>
<br>

The page introduces Exploratory Data Analysis (EDA) as a crucial early step in any data science workflow, performed after gathering or engineering data but before modelling. The purpose is to understand the structure, quality, patterns, and limitations of the data without making strong assumptions. EDA is iterative: insights from exploration guide modelling, and modelling can reveal areas that need further exploration.

<br>
<h2>Types of Data</h2>

<h3>Categorial Variables</h3>
Categorical variables describe groups or labels.
<ul> 
  <li>Nominal: No inherent order (e.g., colours, country).</li>
  <li>Ordinal: Ordered categories (e.g., “low/medium/high”).</li>
</ul>
Common tools include frequency tables, bar charts, and waffle charts. The guide discourages pie charts due to poor angle comparison by humans.

<h3>Numerical Data</h3>
<ul>
  <li>Continuous variables (e.g., height, temperature) can take any value within a range.</li>
    <li>Typical summaries: mean, median, standard deviation, quantiles.</li>
    <li>Visualizations: histograms, density plots, boxplots, violin plots.</li>
  <li>Discrete variables (e.g., event counts) take distinct integer values.</li>
    <li>Visualizations: bar charts, dot plots.</li>
  <li>Bivariate/multivariate numerical analysis uses scatter plots, correlation matrices, heatmaps, and pair plots to reveal relationships, clusters, or patterns across multiple features.</li>
  <li>Key considerations include checking for outliers, understanding distribution shape, and applying transformations (e.g., log scaling) when helpful.</li>
</ul>

<h3>Visualisation</h3>
<ul>
  <li>Multivariate Visualization</li>
    <li>For multiple features, the guide suggests avoiding complex 3D plots unless necessary, as they are difficult to interpret. For high-dimensional datasets, dimensionality reduction techniques (e.g., PCA, LDA) can simplify the data for visualization.</li>
  <li>Text Data</li>
    <li>Unstructured text is explored using tools like word frequency counts, word clouds, and topic modelling. The goal is to reduce noise, identify important terms, and uncover thematic structures within the text corpus.</li>
  <li>Image Data</li>
    <li>Image datasets often require dimensionality reduction (e.g., PCA on pixel intensities). Visualising principal components can reveal meaningful differences—such as facial structure variations—by showing the dominant patterns encoded in the image data.</li>
<ul>

<br>
<h2>Theory</h2>

The page concludes with suggested readings and supplementary materials on EDA, descriptive statistics, and practical introductions to exploratory techniques. These resources provide deeper theoretical grounding and practical examples for applying EDA effectively.
<br>

</details>