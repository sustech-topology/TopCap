import glob
import os
import numpy as np
from scipy.io import wavfile
import shutil
import pickle
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import soundfile as sf
from ripser import ripser
from numpy import argmax

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from numpy import argmax
from matplotlib.gridspec import GridSpec

#16幅图

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





# Step 5: Define a function to plot projections on the first three principal components in 3D
def plot_3d_pca_projections_circular(time_series, embedding_dim, delay, ax, cmap='viridis'):
    embedded_data = time_delay_embedding_circular(time_series, embedding_dim=embedding_dim, delay=delay)
    pca = PCA(n_components=3)
    projected_data = pca.fit_transform(embedded_data)
    
    # Use the index as the color value
    indices = np.arange(len(projected_data))
    scatter = ax.scatter(
        projected_data[:, 0],
        projected_data[:, 1],
        projected_data[:, 2],
        c=indices,
        cmap=cmap,
        s=20  # Increase the size of the points
    )
    ax.set_title(f'Circular Embedding (Tau = {delay}, dim = {embedding_dim})')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    fig.colorbar(scatter, ax=ax, label='Index in Time Series')

def plot_3d_pca_projections_standard(time_series, embedding_dim, delay, ax, cmap='viridis'):
    embedded_data = time_delay_embedding(time_series, embedding_dim=embedding_dim, delay=delay)
    pca = PCA(n_components=3)
    projected_data = pca.fit_transform(embedded_data)
    
    indices = np.arange(len(projected_data))
    scatter = ax.scatter(
        projected_data[:, 0],
        projected_data[:, 1],
        projected_data[:, 2],
        c=indices,
        cmap=cmap,
        s=20
    )
    ax.set_title(f'Standard Embedding (Tau = {delay}, dim = {embedding_dim})')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    fig.colorbar(scatter, ax=ax, label='Index in Time Series')



if __name__ == "__main__":
    # 主程序初始化代码
    
    #打开某个特定的音频[NG]
    with open("C:\\Users\\KikyoForever\\Desktop\\Topcap\\phone", "rb") as fp: 
        time_series = pickle.load(fp)


    # Step 7: Plot in 3 rows - original time series, standard embeddings, circular embeddings
    fig = plt.figure(figsize=(20, 15))  # 3行布局

    # 使用 GridSpec 定义 4 行 4 列的布局
    gs = GridSpec(4, 4, figure=fig)

    # 第一行：4 个标准嵌入（固定d=10，变化tau）
    delays_standard = [5, 10, 50, 100]  # 你的延迟参数
    for i, delay in enumerate(delays_standard):
        ax = fig.add_subplot(gs[0, i], projection='3d')  # 第0行，第i列
        plot_3d_pca_projections_standard(time_series, 10, delay, ax)

    # 第二行：4 个循环嵌入（固定d=10，变化tau）
    delays_circular = [5, 100, 500, 1000]  # 你的延迟参数
    for i, delay in enumerate(delays_circular):
        ax = fig.add_subplot(gs[1, i], projection='3d')  # 第1行，第i列
        plot_3d_pca_projections_circular(time_series, 10, delay, ax)

    # 第三行：4 个标准嵌入（固定tau=5，变化d）
    dims_standard = [10, 100, 500, 1000]  # 变化的嵌入维数
    for i, dim in enumerate(dims_standard):
        ax = fig.add_subplot(gs[2, i], projection='3d')  # 第2行，第i列
        plot_3d_pca_projections_standard(time_series, dim, 1, ax)  

    # 第四行：4 个循环嵌入（固定tau=5，变化d）
    dims_circular = [10, 100, 500, 1000]  # 变化的嵌入维数
    for i, dim in enumerate(dims_circular):
        ax = fig.add_subplot(gs[3, i], projection='3d')  # 第3行，第i列
        plot_3d_pca_projections_circular(time_series, dim, 1, ax)  

    plt.tight_layout()
    plt.show()
   
    