# Time Delay Embedding Analysis and Visualization Framework

This repository contains a suite of Python scripts for analyzing time series data through time delay embeddings. The tools provided here focus on visualizing the geometry of embedded data, characterizing the dynamical properties via PCA eigenvalues, and quantifying topological features (persistent homology) as a function of embedding parameters. These scripts are particularly useful for studying complex signals (e.g., audio or other sequential data) by revealing hidden structures and dynamical invariants.

---

## File Overview

### 1. `16_figs_S2.py`
This script generates a grid of 16 interactive 3D plots to visualize the effects of different time delay embedding parameters. Key functionalities include:

- **Time Delay Embedding Methods:**  
  Implements two types of embeddings:
  - **Standard Embedding:** Uses a fixed delay without wrapping around the time series.
  - **Circular Embedding:** Applies a modulo operation to wrap around the time series, ensuring continuity in cyclic data.

- **3D Visualization via PCA:**  
  Each embedded dataset is reduced to three dimensions using Principal Component Analysis (PCA). The script then plots the projections in a 3D scatter plot where point colors represent the index in the time series.

- **Parameter Variations:**  
  - *First Row:* Standard embeddings with a fixed embedding dimension (10) and varying delays (e.g., 5, 10, 50, 100).
  - *Second Row:* Circular embeddings with the same embedding dimension (10) and a different set of delays (e.g., 5, 100, 500, 1000).
  - *Third and Fourth Rows:* Embeddings (standard and circular, respectively) with a fixed delay (tau = 1) and varying embedding dimensions (e.g., 10, 100, 500, 1000).

- **Usage:**  
  Ensure the pickled time series data (commonly referred to as the `phone` file) is available at the specified path. Run the script to generate a comprehensive set of 3D visualizations that help in selecting optimal embedding parameters.

---

### 2. `Graph_MP_tau_dim_var.py`
This script is designed to study how topological features, extracted from the embedded time series via persistent homology, vary with changes in the time delay (τ) and the embedding dimension (d). Its features include:

- **Persistent Homology Computation:**  
  Uses the Ripser library to compute persistence diagrams on the point clouds obtained from both standard and circular time delay embeddings. It then extracts the key metrics:
  - **Birth Date:** The starting point of a homological feature.
  - **Lifetime:** The persistence (duration) of the feature.

- **Parallel Processing:**  
  Employs Python’s multiprocessing to compute persistent homology metrics for a range of delay values (τ) and embedding dimensions (d) in parallel, improving performance on large datasets.

- **Comprehensive Visualization:**  
  - Plots the variation of the birth dates and lifetimes as functions of delay for circular embeddings.
  - Provides additional plots of 3D PCA projections for standard and circular embeddings with varying τ.
  - Saves computed results in pickle files for later review or further analysis.

- **Usage:**  
  Update the file path to the pickled time series (the `phone` file) before running. The script will display several plots—including the original time series, its magnitude spectrum with annotated frequency peaks, and the relationship between persistent homology metrics and embedding parameters.

---

### 3. `newChar_pca.py`
This script focuses on characterizing the dynamical properties of the time series by examining how PCA eigenvalues of the embedded data change with embedding parameters. It offers:

- **PCA Eigenvalue Analysis:**  
  - **As a Function of Delay (τ):** Computes PCA eigenvalues using circular time delay embeddings while varying τ. It then plots the first 10 eigenvalues versus the delay.
  - **As a Function of Embedding Dimension (d):** For a fixed delay, it varies the embedding dimension and plots the corresponding PCA eigenvalues.

- **Insight into System Dynamics:**  
  By analyzing how the eigenvalues evolve, one can infer the intrinsic dimensionality and the complexity of the underlying dynamics in the time series.

- **Usage:**  
  Make sure the pickled time series (e.g., stored in a file named `phone`) is accessible. Run the script to generate plots that detail the PCA eigenvalue spectra as functions of τ and d.

---

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
