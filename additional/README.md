# Additional codes and materials

This directory contains code for results in Fig. 8, Supplementary Figs. 2–3, and Supplementary Table 1. It also contains miscellaneous files intended to enable and facilitate the user to reproduce the results and apply the methods in the manuscript.  

## Discussion on parameter selection

[`Disc-parameter.py`](Disc-parameter.py) produces results in Fig. 8c.  This script focuses on quantifying topological features as a function of embedding parameters.  More specifically, it is designed to study how topological features, extracted from the embedded time series via persistent homology (PH), vary with changes in the time delay $\tau$ and the embedding dimension $d$. It provides the following functionalities.  

- PH computation 

  Uses the Ripser library to compute persistence diagrams on the point clouds obtained from both standard and cyclic time-delay embedding (TDE).  It then extracts the key metrics: 
    - Birth time, i.e., the starting point of a homological feature 
    - Lifetime, i.e., the persistence (duration) of the feature 

- Parallel processing 
  - Employs Python’s multiprocessing to compute PH metrics for a range of delay values $\tau$ and embedding dimensions $d$ in parallel, improving performance on large datasets.  

- Comprehensive visualisation 
  - Plots the variation of the birth times and lifetimes as functions of delay for cyclic TDE.  
  - Provides additional plots of 3D PCA projections for standard and cyclic TDE with varying $\tau$.  
  - Saves computed results in pickle files for later review or further analysis.  

For its usage, update the file path to the pickled time series (the `phone` file) before running.  The script will display several plots, including the original time series, its magnitude spectrum with annotated frequency peaks, and the relationship between PH metrics and embedding parameters.  

## Discussion on additional geometric features

[`Disc-feature.py`](Disc-feature.py) produces results in Fig. 8d.  This script focuses on characterising the dynamical properties of the time series by examining how principal component analysis (PCA) eigenvalues of the embedded data change with embedding parameters.  It provides the following functionalities.  

- PCA eigenvalue analysis 
  - **As a Function of Delay (τ):** Computes PCA eigenvalues using circular time delay embeddings while varying τ. It then plots the first 10 eigenvalues versus the delay.
  - **As a Function of Embedding Dimension (d):** For a fixed delay, it varies the embedding dimension and plots the corresponding PCA eigenvalues.

- **Insight into System Dynamics:**  
  By analyzing how the eigenvalues evolve, one can infer the intrinsic dimensionality and the complexity of the underlying dynamics in the time series.

For its usage, make sure the pickled time series (e.g., stored in a file named `phone`) is accessible. Run the script to generate plots that detail the PCA eigenvalue spectra as functions of τ and d.

## Standard vs. cyclic time-delay embedding

As in Supplementary Fig. 2, [`Supp-TDE.py`](Supp-TDE.py) generates a grid of 16 interactive 3D plots to comprehensively compare and visualise the methods of time-delay embedding (TDE) in relation to the effects of embedding parameters.  It focuses on visualising the geometry of embedded data and provides the following functionalities.  

- Methods of time-delay embedding 
  - Standard TDE: Uses a fixed delay without wrapping around the time series.  
  - Cyclic TDE: Applies a modulo operation to wrap around the time series, ensuring continuity in cyclic data.  

- 3D Visualisation via PCA 
  - Each embedded dataset is reduced to three dimensions using PCA.  The script then plots the projections in a 3D scatter plot, where point colour represents locations of sampled data points in the time series prior to embedding.  

- Parameter variation 
  - First row: Standard embedding with a fixed embedding dimension 10 and varying delays 5, 10, 50, 100.  
  - Second row: Cyclic embedding with the same embedding dimension 10 and a different set of delays 5, 100, 500, 1000.  
  - Third and fourth rows: Standard and cyclic embedding with a fixed delay 1 and varying embedding dimensions 10, 100, 500, 1270.  

For its usage, ensure the pickled time series data (commonly referred to as the `phone` file) is available through the specified path.  Run the script to generate a comprehensive set of 3D visualisations which help with selecting optimal embedding parameters.  

## Further discussion on parameter selection

[`Supp-Disc.ipynb`](Supp-Disc.ipynb) produces results in Supplementary Fig. 3 and Supplementary Table 1.  

## Codes for results not appearing in the manuscript

- [observation_dimension](additional/observation_dimension.ipynb) illustrates how dimension influences time delay embedding and persistent diagrams.  
- [observation_dimension_plot](additional/observation_dimension_plot.ipynb) includes parameters and graph in the discussion section.  
- [stft_plot_maker_refine](additional/stft_plot_maker_refine.ipynb)

## Consonant waveforms

[This directory](additional/waveform) contains waveforms of pulmonic consonants.  Audio signals for these consonants are from [here](https://en.wikipedia.org/wiki/List_of_consonants).  

## Figure generation

All figures in the manuscript are generated by Adobe Illustrator from the plots obtained through the above Python scripts.  

## Code formats

All the codes are listed as either `.py` (regular Python file) or `.ipynb` (interactive Python notebook).  While the latter format needs to be run in environments such as Jupyter Notebook or JupyterLab, both are intended to enable and facilitate the user to reproduce the results and understand the methods in the manuscript.  








## Common Requirements

All scripts rely on several common Python libraries. Make sure the following packages are installed:

- **Numerical and Scientific Libraries:**  
  `numpy`, `scipy`, `pickle`

- **Data Processing and Visualization:**  
  `matplotlib`, `scikit-learn`

- **Topological Data Analysis:**  
  `ripser`

- **Multiprocessing:**  
  Python's built-in `multiprocessing` module

You can install the required packages using pip:

```bash
pip install numpy scipy matplotlib scikit-learn ripser
