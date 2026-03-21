# Additional code

This directory contains code for results in Fig. 8, Supplementary Figs. 2–3, and Supplementary Table 1.  

## Discussion on parameter selection

[`Disc-parameter.py`](Disc-parameter.py) produces results in Fig. 8c.  This script focuses on quantifying topological features as a function of embedding parameters.  More specifically, it is designed to study how topological features, extracted from the embedded time series via persistent homology (PH), vary under change of the time delay $\tau$ and the embedding dimension $d$. It provides the following functionalities.  

- PH computation 

  Uses the Ripser library to compute persistence diagrams on the point clouds obtained from both standard and cyclic time-delay embedding (TDE).  It then extracts the key metrics: 
    - Birth time, i.e., the starting point of a homological feature 
    - Lifetime, i.e., the persistence (duration) of the feature 

- Parallel processing 

  Employs Python’s multiprocessing to compute PH metrics for a range of delay values $\tau$ and embedding dimensions $d$ in parallel, improving performance on large datasets.  

- Comprehensive visualisation 
  - Plots the variation of the birth times and lifetimes as functions of delay for cyclic TDE (CTDE).  
  - Provides additional plots of 3D PCA projections for standard TDE and CTDE with varying $\tau$.  
  - Saves computed results in pickle files for later review or further analysis.  

**Usage**: Update the file path to the pickled time series (the `phone` file) before running.  The script will display several plots, including the original time series, its magnitude spectrum with annotated frequency peaks, and the relationship between PH metrics and embedding parameters.  

## Discussion on additional geometric features

[`Disc-feature.py`](Disc-feature.py) produces results in Fig. 8d.  This script focuses on characterising the dynamical properties of the time series by examining how principal component analysis (PCA) eigenvalues of the embedded data change with embedding parameters.  It provides the following functionalities.  

- PCA eigenvalue analysis 
  - As a function of delay $\tau$: Computes PCA eigenvalues using CTDE with varying $\tau$.  It then plots the first 10 eigenvalues.
  - As a function of embedding dimension $d$: Fixing a delay, it varies the embedding dimension and plots the corresponding PCA eigenvalues.  

- Insight into system dynamics 

  By analysing how the eigenvalues evolve, one can infer the intrinsic dimensionality and the complexity of the underlying dynamics in the time series.  

**Usage**: Make sure the pickled time series (e.g., stored in a file named `phone`) is accessible.  Run the script to generate plots that detail the PCA eigenvalue spectra as functions of $\tau$ and $d$ respectively.  

## Standard vs. cyclic TDE

As in Supplementary Fig. 2, [`Supp-TDE.py`](Supp-TDE.py) generates a grid of 16 interactive 3D plots to comprehensively compare and visualise the methods of TDE in relation to the effects of embedding parameters.  It focuses on visualising the geometry of embedded data and provides the following functionalities.  

- Methods of time-delay embedding 
  - Standard TDE: Uses a fixed delay without wrapping around the time series.  
  - Cyclic TDE: Applies a modulo operation to wrap around the time series, ensuring continuity in cyclic data.  

- 3D Visualisation via PCA 

  Each embedded dataset is reduced to three dimensions using PCA.  The script then plots the projections in a 3D scatter plot, where point colour represents locations of sampled data points in the time series prior to embedding.  

- Parameter variation 
  - First row: Standard embedding with a fixed embedding dimension 10 and varying delays 5, 10, 50, 100.  
  - Second row: Cyclic embedding with the same embedding dimension 10 and a different set of delays 5, 100, 500, 1000.  
  - Third and fourth rows: Standard and cyclic embedding with a fixed delay 1 and varying embedding dimensions 10, 100, 500, 1270.  

**Usage**: Ensure the pickled time series data (commonly referred to as the `phone` file) is available through the specified path.  Run the script to generate a comprehensive set of 3D visualisations which help with selecting optimal embedding parameters.  

## Requirements for running the codes

All scripts above rely on several common Python libraries.  Make sure the following libraries are installed.  

- Numerical and scientific libraries: `NumPy`, `Pickle`, `SciPy` 

- Data processing and visualisation: `Matplotlib`, `Scikit-learn` 

- Topological data analysis: `Ripser` 

- Multiprocessing: Python's built-in `Multiprocessing` modules 

The user can install the required libraries using pip: 

```bash
pip install matplotlib numpy ripser scikit-learn scipy 
```

## Further discussion on parameter selection

[`Supp-Disc.ipynb`](Supp-Disc.ipynb) produces results in Supplementary Fig. 3 and Supplementary Table 1.  

## Codes for results not appearing in the manuscript

- [observation_dimension](additional/observation_dimension.ipynb) illustrates how dimension influences time delay embedding and persistent diagrams.  
- [observation_dimension_plot](additional/observation_dimension_plot.ipynb) includes parameters and graph in the discussion section.
