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

# Check if MP changes with delay

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
# Define the audio folder path
audio_folder = 'D:\phonetic\Revised\winter_holiday\phone'  # Replace with the actual path

# CSV file name to output TopCap information
# csv_name = 'G:\\phonetic\\PD_Sample2000_Part_Libri_pcatest.csv'

# Initialize two empty lists to store audio data
valid_voiced_list = []
valid_voiceless_list = []

# Initialize two empty lists to store file names
name_voiced_list = []
name_voiceless_list = []

# Define subfolders
voiced_folder = os.path.join(audio_folder, 'voiced')
voiceless_folder = os.path.join(audio_folder, 'voiceless')

# Read audio files from the voiced folder
for filename in os.listdir(voiced_folder):
    if filename.endswith('.wav'):
        file_path = os.path.join(voiced_folder, filename)
        sample_rate, data = wavfile.read(file_path)
        valid_voiced_list.append(data)  # Add audio data to the list
        name_voiced_list.append(filename)  # Add audio file name to the list

# Read audio files from the voiceless folder
for filename in os.listdir(voiceless_folder):
    if filename.endswith('.wav'):
        file_path = os.path.join(voiceless_folder, filename)
        sample_rate, data = wavfile.read(file_path)
        valid_voiceless_list.append(data)  # Add audio data to the list
        name_voiceless_list.append(filename)  # Add audio file name to the list

# Read audio files (FLAC) from the voiced folder
for filename in os.listdir(voiced_folder):
    if filename.endswith('.flac'):
        file_path = os.path.join(voiced_folder, filename)
        data, sample_rate = sf.read(file_path)
        valid_voiced_list.append(data)  # Add audio data to the list
        name_voiced_list.append(filename)  # Add audio file name to the list

# Read audio files (FLAC) from the voiceless folder
for filename in os.listdir(voiceless_folder):
    if filename.endswith('.flac'):
        file_path = os.path.join(voiceless_folder, filename)
        data, sample_rate = sf.read(file_path)
        valid_voiceless_list.append(data)  # Add audio data to the list
        name_voiceless_list.append(filename)  # Add audio file name to the list

# Output the number of audio files read
print(f'Voiced audio files: {len(valid_voiced_list)}')
print(f'Voiceless audio files: {len(valid_voiceless_list)}')


# Example time series data
time_series = valid_voiced_list[1]

# Step 1: Generate a simulated time series of length N=209
np.random.seed(42)
N = 200
t = np.linspace(0, 100, N)

# Three sinusoidal waves with different frequencies and amplitudes
wave1 = 0.5 * np.sin(0.1 * np.pi * t)
wave2 = 0 * np.cos(0.3 * t + np.pi / 3)
wave3 = 0 * np.sin(0.7 * t + np.pi / 4)

# Add some noise
noise = 0 * np.random.randn(N)

# Combine the waves and noise to form the time series
# time_series = wave1 + wave2 + wave3 + noise
'''

# Step 5: Define a function to plot 3D projections of the first three principal components
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

# Step 6: Define a function to estimate the top three peak frequencies in the Fourier spectrum
def analyze_spectrum(time_series):
    """
    Analyze the spectrum of the time series and return the main frequencies and corresponding periods.
    
    Parameters:
        time_series: the input time series
    Returns:
        frequencies_estimated: the estimated top three main frequencies
        periods_estimated: the corresponding periods
    """
    t = np.arange(np.size(time_series))
    
    # Perform Fourier transform
    n = len(time_series)  # Signal length
    yf = np.fft.fft(time_series)  # Fourier transform
    xf = np.fft.fftfreq(n, d=1.0)  # Frequency axis
    
    # Compute the single-sided magnitude spectrum
    yf_single_side = yf[:n//2]
    magnitude_spectrum = 2.0/n * np.abs(yf_single_side)
    
    # Find the frequencies corresponding to the top three peaks (excluding zero frequency)
    non_zero_magnitude_spectrum = magnitude_spectrum[1:]
    non_zero_xf = xf[1:]
    
    # Find all local maxima
    peaks = []
    for i in range(1, len(non_zero_magnitude_spectrum) - 1):
        if (non_zero_magnitude_spectrum[i] > non_zero_magnitude_spectrum[i-1]) and \
           (non_zero_magnitude_spectrum[i] > non_zero_magnitude_spectrum[i+1]):
            peaks.append(i)
    
    if len(peaks) < 3:
        raise ValueError("Not enough local maxima to find the top three frequencies.")
    
    # Extract frequencies and magnitudes for the peaks
    peak_frequencies = [non_zero_xf[p] for p in peaks]
    peak_magnitudes = [non_zero_magnitude_spectrum[p] for p in peaks]
    
    # Get the top three peaks
    top_three_indices = sorted(range(len(peak_magnitudes)), key=lambda k: peak_magnitudes[k], reverse=True)[:3]
    frequencies_estimated = [peak_frequencies[i] for i in top_three_indices]
    periods_estimated = [1.0 / freq for freq in frequencies_estimated]
    
    return frequencies_estimated, periods_estimated

# MP_tau
# Circular embedding version
def compute_mp_for_tau_circular(args):
    """Helper function to compute the MP value for a single tau value."""
    time_series, tau, d = args
    embedded_data = time_delay_embedding_circular(time_series, embedding_dim=d, delay=tau)
    dgms = ripser(embedded_data, maxdim=1)['dgms']
    dgms = dgms[1]
    persistent_time = [ele[1] - ele[0] for ele in dgms]
    index = argmax(persistent_time)
    birth_date = dgms[index][0]
    lifetime = persistent_time[index]
    print(f"Completed calculation for tau={tau}")
    return (tau, birth_date, lifetime)

def MP_delay_parallel_circular(time_series, d=10, n_cores=16):
    # Compute MP values for each delay from 1 to N-1
    N = len(time_series)
    
    # Resample
    MaxSample = 128
    if N > MaxSample:
        # Generate new uniformly distributed points
        x_old = np.linspace(0, 1, N)  # Normalized coordinates of the original array
        x_new = np.linspace(0, 1, MaxSample)  # Normalized coordinates for the new array
        # Linear interpolation
        time_series = np.interp(x_new, x_old, time_series)
        N = MaxSample
    
    # Prepare parameters for parallel computation
    tau_range = range(1, N)
    args_list = [(time_series, tau, d) for tau in tau_range]
    
    # Use a multiprocessing pool to compute in parallel
    with Pool(processes=n_cores) as pool:
        results = pool.map(compute_mp_for_tau_circular, args_list)
    
    # Sort results by tau
    results.sort(key=lambda x: x[0])
    tau_values = [r[0] for r in results]
    birth_dates = [r[1] for r in results]
    lifetimes = [r[2] for r in results]
    
    # Convert to a numpy array
    MP_per_delay = np.array([birth_dates, lifetimes]).T
    
    # Plot the results
    plt.figure(figsize=(14, 8))
    plt.plot(tau_values, MP_per_delay[:, 0], color='cyan', label='birth date')
    plt.plot(tau_values, MP_per_delay[:, 1], color='red', label='lifetime')
    
    plt.title('Lifetime and Birth Date vs Delay (Circular TDE)')
    plt.xlabel('Delay (tau)')
    plt.ylabel('Time')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.show()
    
    return tau_values, MP_per_delay

# MP_tau, standard TDE version
def compute_mp_for_tau_standard(args):
    """Helper function to compute the MP value for a single tau value."""
    time_series, tau, d = args
    embedded_data = time_delay_embedding(time_series, embedding_dim=d, delay=tau)
    dgms = ripser(embedded_data, maxdim=1)['dgms']
    dgms = dgms[1]
    persistent_time = [ele[1] - ele[0] for ele in dgms]
    index = argmax(persistent_time)
    birth_date = dgms[index][0]
    lifetime = persistent_time[index]
    print(f"Completed calculation for tau={tau}")
    return (tau, birth_date, lifetime)

def MP_delay_parallel_standard(time_series, d=10, n_cores=16):
    # Compute MP values for each delay from 1 to N-1
    N = len(time_series)
    
    # Resample
    MaxSample = 128
    if N > MaxSample:
        # Generate new uniformly distributed points
        x_old = np.linspace(0, 1, N)  # Normalized coordinates of the original array
        x_new = np.linspace(0, 1, MaxSample)  # Normalized coordinates for the new array
        # Linear interpolation
        time_series = np.interp(x_new, x_old, time_series)
        N = MaxSample
    
    # Prepare parameters for parallel computation
    tau_range = range(1, int(np.floor((N - 1) / (d - 1)) - 5))
    args_list = [(time_series, tau, d) for tau in tau_range]
    
    # Use a multiprocessing pool to compute in parallel
    with Pool(processes=n_cores) as pool:
        results = pool.map(compute_mp_for_tau_standard, args_list)
    
    # Sort results by tau
    results.sort(key=lambda x: x[0])
    tau_values = [r[0] for r in results]
    birth_dates = [r[1] for r in results]
    lifetimes = [r[2] for r in results]
    
    # Convert to a numpy array
    MP_per_delay = np.array([birth_dates, lifetimes]).T
    
    # Plot the results
    plt.figure(figsize=(14, 8))
    plt.plot(tau_values, MP_per_delay[:, 0], color='cyan', label='birth date')
    plt.plot(tau_values, MP_per_delay[:, 1], color='red', label='lifetime')
    
    plt.title('Lifetime and Birth Date vs Delay (Standard TDE)')
    plt.xlabel('Delay (tau)')
    plt.ylabel('Time')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.show()
    
    return MP_per_delay

# MP_dim
def compute_mp_for_dim_circular(args):
    """Helper function to compute the MP value for a single embedding dimension d."""
    time_series, d, fixed_tau = args
    embedded_data = time_delay_embedding_circular(time_series, embedding_dim=d, delay=fixed_tau)
    dgms = ripser(embedded_data, maxdim=1)['dgms']
    dgms = dgms[1]
    persistent_time = [ele[1] - ele[0] for ele in dgms]
    index = argmax(persistent_time)
    birth_date = dgms[index][0]
    lifetime = persistent_time[index]
    print(f"Completed calculation for d={d}")
    return (d, birth_date, lifetime)

def MP_dim_parallel_circular(time_series, fixed_tau=10, n_cores=16):
    """
    Parallel computation of MP values for different embedding dimensions d.
    
    Parameters:
        time_series: input time series
        fixed_tau: fixed delay value
        n_cores: number of CPU cores to use
    """
    N = len(time_series)
    
    # Resample
    MaxSample = 128
    if N > MaxSample:
        x_old = np.linspace(0, 1, N)
        x_new = np.linspace(0, 1, MaxSample)
        time_series = np.interp(x_new, x_old, time_series)
        N = MaxSample
    
    # Determine the range for d
    max_dim = N  
    min_dim = 2  # Minimum embedding dimension
    skip = 17
    dim_range = range(min_dim, max_dim + 1, skip)
    
    # Prepare parameters for parallel computation
    args_list = [(time_series, d, fixed_tau) for d in dim_range]
    
    # Use a multiprocessing pool to compute in parallel
    with Pool(processes=n_cores) as pool:
        results = pool.map(compute_mp_for_dim_circular, args_list)
    
    # Sort results by d
    results.sort(key=lambda x: x[0])
    dim_values = [r[0] for r in results]
    birth_dates = [r[1] for r in results]
    lifetimes = [r[2] for r in results]
    
    # Convert to a numpy array
    MP_per_dim = np.array([birth_dates, lifetimes]).T
    
    # Plot the results
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
    # Main program initialization code
    
    # Open a specific audio [NG]
    with open("D:\\phonetic\\Revised\\winter_holiday\\phone", "rb") as fp: 
        time_series = pickle.load(fp)

    # Analyze the spectrum of the time series using the function
    frequencies_estimated, periods_estimated = analyze_spectrum(time_series)

    # Step 7: Plot in 3 rows - original time series, standard embeddings, circular embeddings
    fig = plt.figure(figsize=(20, 15))  # 3-row layout

    # Define a 3 row by 4 column layout using GridSpec
    gs = GridSpec(3, 4, figure=fig)

    # First row: Original time series and its spectrum (spanning 2 columns each)
    ax_original = fig.add_subplot(gs[0, :2])  # First row left side
    ax_original.plot(time_series)
    ax_original.set_title('Original Time Series')
    ax_original.set_xlabel('Time')
    ax_original.set_ylabel('Value')

    ax1_spectrum = fig.add_subplot(gs[0, 2:])  # First row right side
    n = len(time_series)
    yf = np.fft.fft(time_series)
    xf = np.fft.fftfreq(n, d=1.0/16000)  # Modified here: add sampling rate conversion
    yf_single_side = yf[:n//2]
    magnitude_spectrum = 2.0/n * np.abs(yf_single_side)
    ax1_spectrum.plot(xf[:n//2], magnitude_spectrum)
    ax1_spectrum.set_title('Magnitude Spectrum')
    ax1_spectrum.set_xlabel('Frequency [Hz]')  # Modified unit
    ax1_spectrum.set_ylabel('Magnitude')
    ax1_spectrum.set_xlim(0, 16000 * 0.2)  # Set x-axis range in Hz

    # Draw dashed lines in different colors at the top three frequencies (converted to Hz)
    colors = ['r', 'g', 'b']
    for i, freq in enumerate(frequencies_estimated):
        freq_hz = freq * 16000  # Convert to Hz
        ax1_spectrum.axvline(
            freq_hz,  # Using Hz
            color=colors[i], 
            linestyle='--', 
            label=f'Peak {i+1}: {freq_hz:.2f} Hz'  # Modified display of unit
        )
    ax1_spectrum.legend()

    # Second row: 4 standard embedding (time_delay_embedding) 3D plots
    delays_standard = [5, 10, 50, 100]  # Your delay parameters
    for i, delay in enumerate(delays_standard):
        ax = fig.add_subplot(gs[1, i], projection='3d')  # Row 1, Column i
        plot_3d_pca_projections_standard(time_series, 10, delay, ax)

    # Third row: 4 circular embedding (time_delay_embedding_circular) 3D plots
    delays_circular = [5, 100, 500, 1000]  # Your delay parameters
    for i, delay in enumerate(delays_circular):
        ax = fig.add_subplot(gs[2, i], projection='3d')  # Row 2, Column i
        plot_3d_pca_projections_circular(time_series, 10, delay, ax)

    plt.tight_layout()
    plt.show()

    # Example usage
    # MP_delay_parallel_standard(time_series, d=10, n_cores=16)
    
    # Modified function call
    result_file = "D:\\phonetic\\Revised\\winter_holiday\\MP_dim_results.pkl"  # Define result file name
    dim_values, MP_per_dim = MP_dim_parallel_circular(time_series, fixed_tau=10, n_cores=16)

    # Save the results to file
    with open(result_file, 'wb') as f:
        pickle.dump({
            'dim_values': dim_values,
            'MP_per_dim': MP_per_dim,
            'fixed_tau': 10
        }, f)

    print(f"Results saved to {result_file}")

    # Code to read from file and plot
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
    
    # Modified function call
    result_file = "D:\\phonetic\\Revised\\winter_holiday\\MP_delay_results.pkl"  # Define result file name
    tau_values, MP_per_delay = MP_delay_parallel_circular(time_series, d=10, n_cores=16)

    # Save the results to file
    with open(result_file, 'wb') as f:
        pickle.dump({
            'dim_values': dim_values,
            'MP_per_dim': MP_per_dim,
            'fixed_tau': 10
        }, f)

    print(f"Results saved to {result_file}")

    # Code to read from file and plot
    with open(result_file, 'rb') as f:
        data = pickle.load(f)

    plt.figure(figsize=(14, 8))
    plt.plot(tau_values, MP_per_delay[:, 0], color='cyan', label='birth date')
    plt.plot(tau_values, MP_per_delay[:, 1], color='red', label='lifetime')
    
    plt.title('Lifetime and Birth Date vs Delay (Circular TDE)')
    plt.xlabel('Delay (tau)')
    plt.ylabel('Time')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.show()
