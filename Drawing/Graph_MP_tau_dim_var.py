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

#找MP随delay变化有没有关系

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



'''
# 定义音频文件夹路径
audio_folder = 'D:\phonetic\Revised\winter_holiday\phone'  # 替换为实际路径

# 输出TopCap信息的csv文件名
# csv_name = 'G:\\phonetic\\PD_Sample2000_Part_Libri_pcatest.csv'

# 初始化两个空列表,记录音频
valid_voiced_list = []
valid_voiceless_list = []

# 初始化两个空列表,记录文件名
name_voiced_list = []
name_voiceless_list = []

# 定义子文件夹
voiced_folder = os.path.join(audio_folder, 'voiced')
voiceless_folder = os.path.join(audio_folder, 'voiceless')

# 读取 voiced 文件夹中的音频文件
for filename in os.listdir(voiced_folder):
    if filename.endswith('.wav'):
        file_path = os.path.join(voiced_folder, filename)
        sample_rate, data = wavfile.read(file_path)
        valid_voiced_list.append(data)  # 将音频数据添加到列表中
        name_voiced_list.append(filename)  # 将音频文件名添加到列表中

# 读取 voiceless 文件夹中的音频文件
for filename in os.listdir(voiceless_folder):
    if filename.endswith('.wav'):
        file_path = os.path.join(voiceless_folder, filename)
        sample_rate, data = wavfile.read(file_path)
        valid_voiceless_list.append(data)  # 将音频数据添加到列表中
        name_voiceless_list.append(filename)  # 将音频文件名添加到列表中

# 读取 voiced 文件夹中的音频文件
for filename in os.listdir(voiced_folder):
    if filename.endswith('.flac'):
        file_path = os.path.join(voiced_folder, filename)
        data , sample_rate = sf.read(file_path)
        valid_voiced_list.append(data)  # 将音频数据添加到列表中
        name_voiced_list.append(filename)  # 将音频文件名添加到列表中

# 读取 voiceless 文件夹中的音频文件
for filename in os.listdir(voiceless_folder):
    if filename.endswith('.flac'):
        file_path = os.path.join(voiceless_folder, filename)
        data , sample_rate = sf.read(file_path)
        valid_voiceless_list.append(data)  # 将音频数据添加到列表中
        name_voiceless_list.append(filename)  # 将音频文件名添加到列表中

# 输出读取的音频数据列表的长度
print(f'Voiced audio files: {len(valid_voiced_list)}')
print(f'Voiceless audio files: {len(valid_voiceless_list)}')


# 示例时间序列数据
########################################################################
time_series = valid_voiced_list[1]

# Step 1: Generate a simulated time series of length N=209
np.random.seed(42)
N = 200
t = np.linspace(0, 100, N)

# Three sinusoidal waves with different frequencies and amplitudes
wave1 = 0.5 * np.sin(0.1* np.pi * t)
wave2 = 0 * np.cos(0.3 * t + np.pi / 3)
wave3 = 0 * np.sin(0.7 * t + np.pi / 4)

# Add some noise
noise = 0 * np.random.randn(N)

# Combine the waves and noise to form the time series
#time_series = wave1 + wave2 + wave3 + noise
'''



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
    ax.set_title(f'Circular Embedding (Tau = {delay})')
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
    ax.set_title(f'Standard Embedding (Tau = {delay})')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    fig.colorbar(scatter, ax=ax, label='Index in Time Series')



# Step 6: Define Fourier 3 peak frequencies estimated function
def analyze_spectrum(time_series):
    """
    分析时间序列的频谱并返回主要频率和周期
    参数:
        time_series: 输入时间序列
    返回:
        frequencies_estimated: 估计的前三个主要频率
        periods_estimated: 对应的周期
        fig: 创建的图形对象
        axs: 子图数组
    """
    t = np.arange(np.size(time_series))
    
    # 进行傅里叶变换
    n = len(time_series)  # 信号长度
    yf = np.fft.fft(time_series)  # 傅里叶变换
    xf = np.fft.fftfreq(n, d=1.0)  # 频率轴
    
    # 计算单边幅度谱
    yf_single_side = yf[:n//2]
    magnitude_spectrum = 2.0/n * np.abs(yf_single_side)
    
    # 找到前三个最大峰值对应的频率（排除频率为0的情况）
    non_zero_magnitude_spectrum = magnitude_spectrum[1:]
    non_zero_xf = xf[1:]
    
    # 找出所有局部最大值
    peaks = []
    for i in range(1, len(non_zero_magnitude_spectrum) - 1):
        if (non_zero_magnitude_spectrum[i] > non_zero_magnitude_spectrum[i-1]) and \
           (non_zero_magnitude_spectrum[i] > non_zero_magnitude_spectrum[i+1]):
            peaks.append(i)
    
    if len(peaks) < 3:
        raise ValueError("Not enough local maxima to find the top three frequencies.")
    
    # 提取峰值对应的频率和幅度
    peak_frequencies = [non_zero_xf[p] for p in peaks]
    peak_magnitudes = [non_zero_magnitude_spectrum[p] for p in peaks]
    
    # 找到前三个最大峰值
    top_three_indices = sorted(range(len(peak_magnitudes)), key=lambda k: peak_magnitudes[k], reverse=True)[:3]
    frequencies_estimated = [peak_frequencies[i] for i in top_three_indices]
    periods_estimated = [1.0 / freq for freq in frequencies_estimated]
    
    return frequencies_estimated, periods_estimated




#MP_tau
#circular
def compute_mp_for_tau_circular(args):
    """计算单个tau值的MP值的辅助函数"""
    time_series, tau, d = args
    embedded_data = time_delay_embedding_circular(time_series, embedding_dim=d, delay=tau)
    dgms = ripser(embedded_data, maxdim=1)['dgms']
    dgms = dgms[1]
    persistent_time = [ele[1]-ele[0] for ele in dgms]
    index = argmax(persistent_time)
    birth_date = dgms[index][0]
    lifetime = persistent_time[index]
    print(f"完成 tau={tau} 的计算")
    return (tau, birth_date, lifetime)

def MP_delay_parallel_circular(time_series, d=10, n_cores=16):
    # Compute PCA eigenvalues for each delay from 1 to N-1
    N = len(time_series)
    
    # 重采样
    MaxSample = 128
    if N > MaxSample:
        # 生成新的均匀分布的点
        x_old = np.linspace(0, 1, N)  # 原数组的归一化坐标
        x_new = np.linspace(0, 1, MaxSample)  # 新数组的归一化坐标
        # 线性插值
        time_series = np.interp(x_new, x_old, time_series)
        N = MaxSample
    
    # 准备并行计算参数
    tau_range = range(1, N)
    args_list = [(time_series, tau, d) for tau in tau_range]
    
    # 使用多进程池并行计算
    with Pool(processes=n_cores) as pool:
        results = pool.map(compute_mp_for_tau_circular, args_list)
    
    # 整理结果，按tau排序
    results.sort(key=lambda x: x[0])
    tau_values = [r[0] for r in results]
    birth_dates = [r[1] for r in results]
    lifetimes = [r[2] for r in results]
    
    # 转换为numpy数组
    MP_per_delay = np.array([birth_dates, lifetimes]).T
    
    # 绘图
    plt.figure(figsize=(14, 8))
    plt.plot(tau_values, MP_per_delay[:, 0], color='cyan', label='birth date')
    plt.plot(tau_values, MP_per_delay[:, 1], color='red', label='lifetime')
    
    plt.title('lifetime and birth date vs Delay (Circular TDE)')
    plt.xlabel('Delay(tau)')
    plt.ylabel('Time')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.show()
    
    return tau_values, MP_per_delay

#MP_tau, standard TDE

def compute_mp_for_tau_standard(args):
    """计算单个tau值的MP值的辅助函数"""
    time_series, tau, d = args
    embedded_data = time_delay_embedding(time_series, embedding_dim=d, delay=tau)
    dgms = ripser(embedded_data, maxdim=1)['dgms']
    dgms = dgms[1]
    persistent_time = [ele[1]-ele[0] for ele in dgms]
    index = argmax(persistent_time)
    birth_date = dgms[index][0]
    lifetime = persistent_time[index]
    print(f"完成 tau={tau} 的计算")
    return (tau, birth_date, lifetime)

def MP_delay_parallel_standard(time_series, d=10, n_cores=16):
    # Compute PCA eigenvalues for each delay from 1 to N-1
    N = len(time_series)
    
    # 重采样
    MaxSample = 128
    if N > MaxSample:
        # 生成新的均匀分布的点
        x_old = np.linspace(0, 1, N)  # 原数组的归一化坐标
        x_new = np.linspace(0, 1, MaxSample)  # 新数组的归一化坐标
        # 线性插值
        time_series = np.interp(x_new, x_old, time_series)
        N = MaxSample
    
    # 准备并行计算参数
    tau_range = range(1, int(np.floor((N - 1) / (d - 1)) -5))
    args_list = [(time_series, tau, d) for tau in tau_range]
    
    # 使用多进程池并行计算
    with Pool(processes=n_cores) as pool:
        results = pool.map(compute_mp_for_tau_standard, args_list)
    
    # 整理结果，按tau排序
    results.sort(key=lambda x: x[0])
    tau_values = [r[0] for r in results]
    birth_dates = [r[1] for r in results]
    lifetimes = [r[2] for r in results]
    
    # 转换为numpy数组
    MP_per_delay = np.array([birth_dates, lifetimes]).T
    
    # 绘图
    plt.figure(figsize=(14, 8))
    plt.plot(tau_values, MP_per_delay[:, 0], color='cyan', label='birth date')
    plt.plot(tau_values, MP_per_delay[:, 1], color='red', label='lifetime')
    
    plt.title('lifetime and birth date vs Delay (Standard TDE)')
    plt.xlabel('Delay(tau)')
    plt.ylabel('Time')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.show()
    
    return MP_per_delay


#MP_dim
def compute_mp_for_dim_circular(args):
    """计算单个维度d的MP值的辅助函数"""
    time_series, d, fixed_tau = args
    embedded_data = time_delay_embedding_circular(time_series, embedding_dim=d, delay=fixed_tau)
    dgms = ripser(embedded_data, maxdim=1)['dgms']
    dgms = dgms[1]
    persistent_time = [ele[1]-ele[0] for ele in dgms]
    index = argmax(persistent_time)
    birth_date = dgms[index][0]
    lifetime = persistent_time[index]
    print(f"完成 d={d} 的计算")
    return (d, birth_date, lifetime)

def MP_dim_parallel_circular(time_series, fixed_tau=10, n_cores=16):
    """
    并行计算不同嵌入维度d的MP值
    
    参数:
        time_series: 输入时间序列
        fixed_tau: 固定的延迟值
        n_cores: 使用的CPU核心数
    """
    N = len(time_series)
    
    # 重采样
    MaxSample = 128
    if N > MaxSample:
        x_old = np.linspace(0, 1, N)
        x_new = np.linspace(0, 1, MaxSample)
        time_series = np.interp(x_new, x_old, time_series)
        N = MaxSample
    
    # 确定d的范围 
    max_dim = N  
    min_dim = 2  # 最小嵌入维度
    skip = 17
    dim_range = range(min_dim, max_dim + 1, skip)
    
    # 准备并行计算参数
    args_list = [(time_series, d, fixed_tau) for d in dim_range]
    
    # 使用多进程池并行计算
    with Pool(processes=n_cores) as pool:
        results = pool.map(compute_mp_for_dim_circular, args_list)
    
    # 整理结果，按d排序
    results.sort(key=lambda x: x[0])
    dim_values = [r[0] for r in results]
    birth_dates = [r[1] for r in results]
    lifetimes = [r[2] for r in results]
    
    # 转换为numpy数组
    MP_per_dim = np.array([birth_dates, lifetimes]).T
    
    # 绘图
    plt.figure(figsize=(14, 8))
    plt.plot(dim_values, MP_per_dim[:, 0], 'o-', color='cyan', label='birth date')
    plt.plot(dim_values, MP_per_dim[:, 1], 'o-', color='red', label='lifetime')
    
    plt.title(f'Lifetime and Birth Date vs Embedding Dimension (τ={fixed_tau}) Circular TDE')
    plt.xlabel('Embedding Dimension (d)')
    plt.ylabel('Time')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.show()
    
    return dim_values, MP_per_dim

if __name__ == "__main__":
    # 主程序初始化代码
    
    #打开某个特定的音频[NG]
    with open("D:\phonetic\Revised\winter_holiday\phone", "rb") as fp: 
        time_series = pickle.load(fp)

    # 使用函数分析频谱
    frequencies_estimated, periods_estimated = analyze_spectrum(time_series)


    # Step 7: Plot in 3 rows - original time series, standard embeddings, circular embeddings
    fig = plt.figure(figsize=(20, 15))  # 3行布局

    # 使用 GridSpec 定义 3 行 4 列的布局
    gs = GridSpec(3, 4, figure=fig)

    # 第一行：原始时间序列及其频谱（跨所有4列）
    ax_original = fig.add_subplot(gs[0, :2])  # 第一行左侧
    ax_original.plot(time_series)
    ax_original.set_title('Original Time Series')
    ax_original.set_xlabel('Time')
    ax_original.set_ylabel('Value')

    ax1_spectrum = fig.add_subplot(gs[0, 2:])  # 第一行右侧
    n = len(time_series)
    yf = np.fft.fft(time_series)
    xf = np.fft.fftfreq(n, d=1.0/16000)  # 修改这里：添加采样率转换
    yf_single_side = yf[:n//2]
    magnitude_spectrum = 2.0/n * np.abs(yf_single_side)
    ax1_spectrum.plot(xf[:n//2], magnitude_spectrum)
    ax1_spectrum.set_title('Magnitude Spectrum')
    ax1_spectrum.set_xlabel('Frequency [Hz]')  # 修改单位
    ax1_spectrum.set_ylabel('Magnitude')
    ax1_spectrum.set_xlim(0, 16000*0.2)  # 修改x轴范围到Hz单位

    # 在前三个频率处绘制不同颜色的虚线（转换为Hz单位）
    colors = ['r', 'g', 'b']
    for i, freq in enumerate(frequencies_estimated):
        freq_hz = freq * 16000  # 转换为Hz单位
        ax1_spectrum.axvline(
            freq_hz,  # 使用Hz单位
            color=colors[i], 
            linestyle='--', 
            label=f'Peak {i+1}: {freq_hz:.2f} Hz'  # 修改单位显示
        )
    ax1_spectrum.legend()

    # 第二行：4 个标准嵌入（time_delay_embedding）的 3D 图
    delays_standard = [5, 10, 50, 100]  # 你的延迟参数
    for i, delay in enumerate(delays_standard):
        ax = fig.add_subplot(gs[1, i], projection='3d')  # 第1行，第i列
        plot_3d_pca_projections_standard(time_series, 10, delay, ax)

    # 第三行：4 个循环嵌入（time_delay_embedding_circular）的 3D 图
    delays_circular = [5, 100, 500, 1000]  # 你的延迟参数
    for i, delay in enumerate(delays_circular):
        ax = fig.add_subplot(gs[2, i], projection='3d')  # 第2行，第i列
        plot_3d_pca_projections_circular(time_series, 10, delay, ax)

    plt.tight_layout()
    plt.show()

    # 使用示例
    ##MP_delay_parallel_standard(time_series, d=10, n_cores=16)
    
    


    # 修改后的调用方式
    result_file = "D:\\phonetic\\Revised\\winter_holiday\\MP_dim_results.pkl"  # 定义结果文件名
    dim_values, MP_per_dim = MP_dim_parallel_circular(time_series, fixed_tau=10, n_cores=16)

    # 将结果保存到文件
    import pickle
    with open(result_file, 'wb') as f:
        pickle.dump({
            'dim_values': dim_values,
            'MP_per_dim': MP_per_dim,
            'fixed_tau': 10
        }, f)

    print(f"Results saved to {result_file}")

    # 从文件读取并绘图的代码
    with open(result_file, 'rb') as f:
        data = pickle.load(f)

    plt.figure(figsize=(14, 8))
    plt.plot(data['dim_values'], data['MP_per_dim'][:, 0], 'o-', color='cyan', label='birth date')
    plt.plot(data['dim_values'], data['MP_per_dim'][:, 1], 'o-', color='red', label='lifetime')
    plt.title(f'Lifetime and Birth Date vs Embedding Dimension (τ={data["fixed_tau"]}) Circular TDE')
    plt.xlabel('Embedding Dimension (d)')
    plt.ylabel('Time')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.show()
    
    


   

    # 修改后的调用方式
    result_file = "D:\\phonetic\\Revised\\winter_holiday\\MP_delay_results.pkl"  # 定义结果文件名
    tau_values, MP_per_delay = MP_delay_parallel_circular(time_series, d=10, n_cores=16)

    # 将结果保存到文件
    import pickle
    with open(result_file, 'wb') as f:
        pickle.dump({
            'dim_values': dim_values,
            'MP_per_dim': MP_per_dim,
            'fixed_tau': 10
        }, f)

    print(f"Results saved to {result_file}")

    # 从文件读取并绘图的代码
    with open(result_file, 'rb') as f:
        data = pickle.load(f)

    plt.figure(figsize=(14, 8))
    plt.plot(tau_values, MP_per_delay[:, 0], color='cyan', label='birth date')
    plt.plot(tau_values, MP_per_delay[:, 1], color='red', label='lifetime')
    
    plt.title('lifetime and birth date vs Delay (Circular TDE)')
    plt.xlabel('Delay(tau)')
    plt.ylabel('Time')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.show()
