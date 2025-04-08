import glob
import os
import numpy as np
from scipy.io import wavfile
import shutil

import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import soundfile as sf

from numpy import argmax
import math
import heapq
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
import numpy as np


def time_delay_embedding(time_series, embedding_dim=2, delay=1):
    """
    Perform time delay embedding on a given time series.

    Parameters:
    - time_series: 1D array-like, the input time series data.
    - embedding_dim: int, the dimension of the embedded space.
    - delay: int, the time delay between coordinates in the embedded space.

    Returns:
    - embedded_data: 2D numpy array, the embedded time series data.
    """
    if embedding_dim < 1:
        raise ValueError("Embedding dimension must be at least 1.")
    if delay < 1:
        raise ValueError("Delay must be at least 1.")

    N = len(time_series)
    if N < (embedding_dim - 1) * delay + 1:
        raise ValueError("Not enough data points for the specified embedding dimension and delay.")

    embedded_data = []
    for i in range(N - (embedding_dim - 1) * delay):
        point = [time_series[i + j * delay] for j in range(embedding_dim)]
        embedded_data.append(point)

    return np.array(embedded_data)

def time_delay_embedding_circular(time_series, embedding_dim=3, delay=1):
    """
    Perform circular time delay embedding on a given time series.

    Parameters:
    - time_series: 1D array-like, the input time series data.
    - embedding_dim: int, the dimension of the embedded space.
    - delay: int, the time delay between coordinates in the embedded space.

    Returns:
    - embedded_data: numpy array, the embedded time series data.
    """
    if embedding_dim < 1:
        raise ValueError("Embedding dimension must be at least 1.")
    if delay < 1:
        raise ValueError("Delay must be at least 1.")

    N = len(time_series)
    if N < embedding_dim:
        raise ValueError("Time series length must be at least the embedding dimension.")

    embedded_data = []
    for i in range(N):
        point = [time_series[(i + j * delay) % N] for j in range(embedding_dim)]
        embedded_data.append(point)

    return np.array(embedded_data)


def PCA_eigenvalues_delay(time_series, d=10):
    # Compute PCA eigenvalues for each delay from 1 to N-1
    N = len(time_series)
    eigenvalues_per_delay = []

    # Resampling
    maxnum = 2000
    if N > maxnum:
        # Generate new uniformly distributed points
        x_old = np.linspace(0, 1, N)  # Normalized coordinates of the original array
        x_new = np.linspace(0, 1, maxnum)  # Normalized coordinates for the new array
        # Linear interpolation
        time_series = np.interp(x_new, x_old, time_series)
        N = maxnum

    # Loop over tau values
    for tau in range(1, N):
        embedded_data = time_delay_embedding_circular(time_series, embedding_dim=d, delay=tau)
        pca = PCA(n_components=min(d, N))
        pca.fit(embedded_data)
        eigenvalues = pca.explained_variance_
        eigenvalues_per_delay.append(eigenvalues[:10])  # Keep only the first 10 eigenvalues
        print(f'tau = {tau}')
    
    # Convert to a numpy array for easier manipulation
    eigenvalues_per_delay = np.array(eigenvalues_per_delay)

    # Plot the first 10 eigenvalues as a function of delay
    plt.figure(figsize=(14, 8))

    colors = plt.cm.tab10(np.arange(10))  # Use tab10 colormap for 10 colors

    for i in range(10):
        plt.plot(range(1, N), eigenvalues_per_delay[:, i], color=colors[i], label=f'Eigenvalue {i+1}')

    plt.title('First 10 Eigenvalues of PCA vs Delay')
    plt.xlabel('Delay (tau)')
    plt.ylabel('Eigenvalue')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.show()
    return


def PCA_eigenvalues_dimension(time_series, tau=10):
    # Compute PCA eigenvalues for each embedding dimension from 1 to N
    N = len(time_series)
    eigenvalues_per_dimension = []

    # Resampling
    maxnum = 2000
    if N > maxnum:
        # Generate new uniformly distributed points
        x_old = np.linspace(0, 1, N)  # Normalized coordinates of the original array
        x_new = np.linspace(0, 1, maxnum)  # Normalized coordinates for the new array
        # Linear interpolation
        time_series = np.interp(x_new, x_old, time_series)
        N = maxnum

    # Loop over dimensions from 1 to N
    for d in range(1, N+1):
        embedded_data = time_delay_embedding_circular(time_series, embedding_dim=d, delay=tau)
        pca = PCA(n_components=min(d, N))
        pca.fit(embedded_data)
        eigenvalues = pca.explained_variance_
        # Pad to fixed length N, filling with zeros if necessary
        padded_eigenvalues = np.zeros(N)
        padded_eigenvalues[:len(eigenvalues)] = eigenvalues
        eigenvalues_per_dimension.append(padded_eigenvalues[:10])  # Keep only the first 10 eigenvalues
        print(f'd = {d}')
    
    # Convert to a numpy array for easier manipulation
    eigenvalues_per_dimension = np.array(eigenvalues_per_dimension)

    # Plot the first 10 eigenvalues as a function of embedding dimension
    plt.figure(figsize=(14, 8))

    colors = plt.cm.tab10(np.arange(10))  # Use tab10 colormap for 10 colors

    for i in range(10):
        plt.plot(range(1, N+1), eigenvalues_per_dimension[:, i], color=colors[i], label=f'Eigenvalue {i+1}')

    plt.title(f'First 10 Eigenvalues of PCA vs Embedding Dimension (τ={tau})')
    plt.xlabel('Embedding Dimension (d)')
    plt.ylabel('Eigenvalue')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.show()
    
    return eigenvalues_per_dimension


with open("../phone", "rb") as fp: 
    time_series = pickle.load(fp)


PCA_eigenvalues_delay(time_series, 100)
PCA_eigenvalues_dimension(time_series, 10)

