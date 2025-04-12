#coding=utf-8
# Convert voiced consonant audio into 2D topological features: birth time and persistence.
# Use Fourier transform to find the fundamental frequency, normalize, add noise, then normalize again.
# Assign different noise to each audio by generating a seed from its index; use None to indicate that no noise is added.
# Trim the audio before embedding and use parallel computation to reduce processing time.

import glob
import os
import numpy as np
from scipy.io import wavfile
import shutil

import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import soundfile as sf
from gudhi.point_cloud import timedelay
from numpy import argmax
import math
from ripser import ripser
from persim import plot_diagrams
from multiprocessing import Pool, cpu_count
from functools import partial

def normalize_signal(signal):
    """
    Normalize the signal to the range [-1, 1].
    """
    max_abs_value = np.max(np.abs(signal))
    if max_abs_value == 0:
        return signal  # If the maximum absolute value is 0, return the original signal.
    return signal / max_abs_value

def add_noise(y, seed=110, noise_type='gaussian', snr_db=5):
    """
    Add specified noise to a speech signal (supports controlling randomness with a seed).
    
    Parameters:
        y : Original audio signal.
        noise_type : Type of noise ('gaussian', 'uniform', 'impulse').
        snr_db : Signal-to-noise ratio (in decibels).
        seed : Random seed (controls reproducibility of noise generation).
    
    Returns:
        noisy_y_normalized : The noisy audio signal after normalization.
    """
    # Normalize the original signal before processing.
    y_normalized = normalize_signal(y)
    
    if snr_db is None:
        return y_normalized
    
    # Create an independent random state (to avoid contaminating the global NumPy random seed).
    rng = np.random.RandomState(seed)
    
    # Compute the power of the original signal.
    signal_power = np.mean(y_normalized**2)
    
    # Generate noise based on the specified type.
    if noise_type == 'gaussian':
        noise = rng.normal(0, 1, len(y_normalized))  # Use rng instead of np.random.
    elif noise_type == 'uniform':
        noise = rng.uniform(-1, 1, len(y_normalized))
    elif noise_type == 'impulse':
        noise = np.zeros_like(y_normalized)
        num_impulses = int(len(y_normalized) * 0.001)  # Adjust the ratio to suit different signal lengths.
        indices = rng.randint(0, len(y_normalized), num_impulses)     # Random positions.
        noise[indices] = rng.uniform(-0.5, 0.5, num_impulses)  # Random amplitudes.
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    # Adjust noise power to achieve the target SNR.
    noise_power = np.maximum(np.mean(noise**2), 1e-10)  # Ensure noise power is not zero.
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scaling_factor = np.sqrt(target_noise_power / noise_power)
    scaled_noise = noise * scaling_factor

    # Mix the signals.
    noisy_y = y_normalized + scaled_noise
    
    # Normalize the final noisy signal.
    noisy_y_normalized = normalize_signal(noisy_y)
    
    return noisy_y_normalized

M = 100
max_edge_length = 1
samplerate = 16000

# wav_fraction_finder finds the corresponding fraction of the wav signal based on an interval.
def wav_fraction_finder(start_time, end_time, sig):
    sig_fraction = sig[int(start_time * samplerate):int(end_time * samplerate)]
    return sig_fraction

# principle_frequency_finder_top3 finds the period of a speech signal by identifying the top three peaks.
def principle_frequency_finder_top3(time_series):
    # Perform Fourier transform.
    n = len(time_series)  # Signal length.
    yf = np.fft.fft(time_series)  # Fourier transform.
    xf = np.fft.fftfreq(n, d=1.0)  # Frequency axis, with d=1.0 meaning each sample is 1 unit time.

    # Compute the single-sided amplitude spectrum.
    yf_single_side = yf[:n // 2]
    magnitude_spectrum = 2.0 / n * np.abs(yf_single_side)

    # Find the frequencies corresponding to the top three peaks (excluding the zero frequency).
    # Exclude the first value (zero frequency).
    non_zero_magnitude_spectrum = magnitude_spectrum[1:]
    non_zero_xf = xf[1:]

    # Find all local maxima.
    peaks = []
    for i in range(1, len(non_zero_magnitude_spectrum) - 1):
        if (non_zero_magnitude_spectrum[i] > non_zero_magnitude_spectrum[i - 1]) and \
           (non_zero_magnitude_spectrum[i] > non_zero_magnitude_spectrum[i + 1]):
            peaks.append(i)

    if len(peaks) < 3:
        raise ValueError("Not enough local maxima to find the top three frequencies.")

    # Extract frequencies and magnitudes corresponding to the peaks.
    peak_frequencies = [non_zero_xf[p] for p in peaks]
    peak_magnitudes = [non_zero_magnitude_spectrum[p] for p in peaks]

    # Get the indices of the top three highest peaks.
    top_three_indices = sorted(range(len(peak_magnitudes)), key=lambda k: peak_magnitudes[k], reverse=True)[:3]
    frequencies_estimated = [peak_frequencies[i] for i in top_three_indices]
    periods_estimated = [1.0 / freq for freq in frequencies_estimated]

    return periods_estimated[0]

# principle_frequency_finder finds the period of a speech signal using its highest local maximum.
def principle_frequency_finder(time_series):
    # Perform Fourier transform.
    n = len(time_series)  # Signal length.
    yf = np.fft.fft(time_series)  # Fourier transform.
    xf = np.fft.fftfreq(n, d=1.0)  # Frequency axis, with d=1.0 meaning each sample is 1 unit time.

    # Compute the single-sided amplitude spectrum.
    yf_single_side = yf[:n // 2]
    magnitude_spectrum = 2.0 / n * np.abs(yf_single_side)

    # Exclude the first value (zero frequency).
    non_zero_magnitude_spectrum = magnitude_spectrum[1:]
    non_zero_xf = xf[1:]

    # Find all local maxima.
    peaks = []
    for i in range(1, len(non_zero_magnitude_spectrum) - 1):
        if (non_zero_magnitude_spectrum[i] > non_zero_magnitude_spectrum[i - 1]) and \
           (non_zero_magnitude_spectrum[i] > non_zero_magnitude_spectrum[i + 1]):
            peaks.append(i)

    # Extract frequencies and magnitudes corresponding to the peaks.
    peak_frequencies = [non_zero_xf[p] for p in peaks]
    peak_magnitudes = [non_zero_magnitude_spectrum[p] for p in peaks]
    
    # Check if no peaks were found.
    if len(peak_magnitudes) == 0:
        return 1  # Or choose another default value as needed.
    
    # Find the index of the maximum peak.
    top_index = np.argmax(peak_magnitudes)
    # Compute the corresponding frequency and period.
    frequency_estimated = peak_frequencies[top_index]
    period_estimated = 1.0 / frequency_estimated
    
    return period_estimated

def principle_frequency_finder_acf(sig):
    t = int(len(sig) / 2)
    corr = np.zeros(t)

    for index in np.arange(t):
        ACF_delay = sig[index:]
        L = (t - index) / 2
        m = np.sum(sig[int(t - L):int(t + L + 1)]**2) + np.sum(ACF_delay[int(t - L):int(t + L + 1)]**2)
        r = np.sum(sig[int(t - L):int(t + L + 1)] * ACF_delay[int(t - L):int(t + L + 1)])
        corr[index] = 2 * r / m

    zc = np.zeros(corr.size - 1)
    zc[(corr[0:-1] < 0) * (corr[1::] > 0)] = 1
    zc[(corr[0:-1] > 0) * (corr[1::] < 0)] = -1

    admiss = np.zeros(corr.size)
    admiss[0:-1] = zc
    for i in range(1, corr.size):
        if admiss[i] == 0:
            admiss[i] = admiss[i - 1]

    maxes = np.zeros(corr.size)
    maxes[1:-1] = (np.sign(corr[1:-1] - corr[0:-2]) == 1) * (np.sign(corr[1:-1] - corr[2::]) == 1)
    maxidx = np.arange(corr.size)
    maxidx = maxidx[maxes == 1]
    max_index = 0
    if len(corr[maxidx]) > 0:
        max_index = maxidx[np.argmax(corr[maxidx])]

    return max_index

# Persistent homology computation.
def process_single_item(args, data_category=1, M=100):
    """
    Generic function to process a single item in parallel.
    data_category: 1 = voiced, 2 = voiceless.
    """
    i, data, delay_values, name_list = args
    
    try:

        # Process the delay parameter.
        if delay_values[i] * M > len(data):
            delay_values[i] = int(np.floor(len(data) / M))
        if delay_values[i] == 0:
            delay_values[i] = 1
            
        # Time delay embedding.
        tau = delay_values[i]
        point_Cloud = timedelay.TimeDelayEmbedding(M, tau, 5)
        
        if data.size == 0:
            return None
            
        # Core computation.
        Points = point_Cloud(data)
        if len(Points) < 40 or np.isnan(Points).any():
            return None
            
        # Persistent homology computation.
        dgms = ripser(Points, maxdim=1)['dgms'][1]
        if dgms.size == 0:
            return None
            
        persistent_time = [ele[1] - ele[0] for ele in dgms]
        index = np.argmax(persistent_time)
        
        # Construct the return result (preserve original index).
        return (
            i,
            name_list[i],
            dgms[index][0],
            persistent_time[index],
            data_category
        )
        
    except Exception as e:
        print(f"Error processing item {i}: {str(e)}")
        return None

def parallel_process_to_csv(csv_name, data_list, delay_list, name_list, category, workers=None):
    """
    Main function for parallel processing.
    category: 1 = voiced, 2 = voiceless.
    """
    # Create a process pool.
    workers = workers or max(1, cpu_count() - 1)
    
    # Generate task parameters (preserving original indices).
    task_args = [(i, data, delay_list, name_list) for i, data in enumerate(data_list)]
    
    with Pool(workers) as pool:
        # Create a partial function with fixed category parameter.
        processor = partial(process_single_item, data_category=category)
        
        # Parallel processing (preserving order).
        results = []
        for idx, result in enumerate(pool.imap(processor, task_args, chunksize=50)):
            if result:
                results.append(result)
            if idx % 1000 == 0:
                print(f"Processed {idx}/{len(data_list)} items")
                
    # Sort by original index and write in batch.
    with open(csv_name, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for res in sorted(results, key=lambda x: x[0]):
            writer.writerow(res)  
            
    print(f"Category {category} completed. Total {len(results)} valid entries.")

if __name__ == "__main__":
    # Main program initialization code.
    M = 100  # Embedding dimension.
    
    # Define the audio folder path (replace with the actual path).
    audio_folder = '../data/train500_audio_segment'

    # Output CSV file name for TopCap information.
    csv_name = '../data/Libri_train500.csv'

    # Initialize two empty lists to record audio.
    valid_voiced_list = []
    valid_voiceless_list = []

    # Initialize two empty lists to record filenames.
    name_voiced_list = []
    name_voiceless_list = []

    # Define subdirectories.
    voiced_folder = os.path.join(audio_folder, 'voiced')
    voiceless_folder = os.path.join(audio_folder, 'voiceless')

    failed_files = []  # Record files that failed to read.

    def read_audio_file(file_path, filename):
        try:
            if filename.endswith('.wav'):
                sample_rate, data = wavfile.read(file_path)
            elif filename.endswith('.flac'):
                data, sample_rate = sf.read(file_path)
            else:
                return None, None
            return data, sample_rate
        except Exception as e:
            failed_files.append((filename, str(e)))
            return None, None

    # Read audio files (WAV and FLAC) from the 'voiced' folder.
    for filename in os.listdir(voiced_folder):
        if filename.endswith(('.wav', '.flac')):
            file_path = os.path.join(voiced_folder, filename)
            data, sample_rate = read_audio_file(file_path, filename)
            if data is not None:
                valid_voiced_list.append(data)
                name_voiced_list.append(filename)
                print(f"Successfully read: {filename}")

    # Read audio files (WAV and FLAC) from the 'voiceless' folder.
    for filename in os.listdir(voiceless_folder):
        if filename.endswith(('.wav', '.flac')):
            file_path = os.path.join(voiceless_folder, filename)
            data, sample_rate = read_audio_file(file_path, filename)
            if data is not None:
                valid_voiceless_list.append(data)
                name_voiceless_list.append(filename)
                print(f"Successfully read: {filename}")

    # Output statistics.
    print(f"Number of valid Voiced files: {len(valid_voiced_list)}")
    print(f"Number of valid Voiceless files: {len(valid_voiceless_list)}")

    # Add noise (each audio uses a unique seed).
    noisy_voiced_list = []
    noisy_voiceless_list = []

    noise_params = {
        'noise_type': 'gaussian',
        'snr_db': None,
        'base_seed': 110  # Base seed; each audio will use (base_seed + its index).
    }

    for idx, y in enumerate(valid_voiced_list):
        # Generate a unique seed for each audio (base_seed + index).
        current_seed = noise_params['base_seed'] + idx
        
        noisy_y = add_noise(
            y, 
            seed=current_seed,  # Use the dynamically generated independent seed.
            noise_type=noise_params['noise_type'],
            snr_db=noise_params['snr_db'],
        )
        noisy_voiced_list.append(noisy_y)

    print(f"Generated {len(noisy_voiced_list)} independent noisy audio signals for voiced files.")

    for idx, y in enumerate(valid_voiceless_list):
        # Generate a unique seed for each audio (base_seed + index).
        current_seed = noise_params['base_seed'] + idx
        
        noisy_y = add_noise(
            y, 
            seed=current_seed,  # Use the dynamically generated independent seed.
            noise_type=noise_params['noise_type'],
            snr_db=noise_params['snr_db'],
        )
        noisy_voiceless_list.append(noisy_y)

    print(f"Generated {len(noisy_voiceless_list)} independent noisy audio signals for voiceless files.")

    valid_voiced_list = noisy_voiced_list
    valid_voiceless_list = noisy_voiceless_list

    # Compute TopCap.
    # Calculate the delay parameter (execute only once in the main process).
    k = 6

    T_voiced = [0] * len(valid_voiced_list)
    for i in range(len(valid_voiced_list)):
        T_voiced[i] = principle_frequency_finder(np.array(valid_voiced_list[i]))
        T_voiced[i] = T_voiced[i]
        print(f"Voiced principle frequency computed for index {i}")

    delay_voiced = [round(ele * k / M) for ele in T_voiced]
    for element in range(len(delay_voiced)):
        if delay_voiced[element] == 0:
            delay_voiced[element] = 1

    T_voiceless = [0] * len(valid_voiceless_list)
    for i in range(len(valid_voiceless_list)):
        T_voiceless[i] = principle_frequency_finder(np.array(valid_voiceless_list[i]))
        T_voiceless[i] = T_voiceless[i]
        print(f"Voiceless principle frequency computed for index {i}")

    delay_voiceless = [round(ele * k / M) for ele in T_voiceless]
    for element in range(len(delay_voiceless)):
        if delay_voiceless[element] == 0:
            delay_voiceless[element] = 1

    # Call parallel processing.
    # Process voiced data.
    parallel_process_to_csv(
        csv_name,
        valid_voiced_list,
        delay_voiced,
        name_voiced_list,
        category=1,
        workers=16
    )

    # Process voiceless data.
    parallel_process_to_csv(
        csv_name,
        valid_voiceless_list,
        delay_voiceless,
        name_voiceless_list,
        category=2,
        workers=16
    )
