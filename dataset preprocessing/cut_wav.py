import librosa
import matplotlib.pyplot as plt
import os
import soundfile as sf
from praatio import textgrid
import csv
import matplotlib as mpl
import numpy as np
from numpy import argmax
import math

# Assume you already have syllable-labeled data which includes each syllable's start and end times.
# Cut the phoneme segments and place them into different folders.

DATASET_PATH = 'D:\\phonetic\\ALLSSTAR_reMFA\\Newcut_NWS'
inputPath = DATASET_PATH

def get_wav_filenames_without_extension(directory):
    return [f[:-4] for f in os.listdir(directory) if f.endswith('.wav')]

# Method usage
wav_filenames = get_wav_filenames_without_extension(DATASET_PATH)

def ensure_directory_exists(file_path):
    """
    Ensure the directory exists; if not, create it.
    Args:
        file_path (str): The file path.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

# Get the folder path where the current script is located.
current_directory = os.path.dirname(os.path.abspath(__file__))

# Create the folder 'audio_segment'
folder_a = os.path.join(current_directory, 'audio_segment')
os.makedirs(folder_a, exist_ok=True)

# In the 'audio_segment' folder, create two subfolders: 'voiced' and 'voiceless'
folder_v1 = os.path.join(folder_a, 'voiced')
os.makedirs(folder_v1, exist_ok=True)

folder_v2 = os.path.join(folder_a, 'voiceless')
os.makedirs(folder_v2, exist_ok=True)

print("Folders created successfully!")

# Define phoneme labels (using uppercase letters)
voiced_phones = ['V', 'L', 'NG', 'M', 'N', 'Y', 'ZH']
voiceless_phones = ['F', 'TH', 'T', 'S', 'K', 'CH']

M = 100
max_edge_length = 1
samplerate = 16000

# wav_fraction_finder finds the corresponding portion of the wav signal based on the interval.
def wav_fraction_finder(start_time, end_time, sig):
    sig_fraction = sig[int(start_time * samplerate): int(end_time * samplerate)]
    return sig_fraction

# head_tail_scissor trims the head and tail of a signal where the amplitude is smaller than 0.03.
# It also checks if the trimmed signal length is greater than 500.
def head_tail_scissor(sig):
    valid_interval = [index for index in range(len(sig)) if (sig[index] > 0.03).any()]
    if len(valid_interval) == 0:
        return False, sig
    head = min(valid_interval)
    tail = max(valid_interval)
    sig = sig[head: tail + 1]
    if tail - head < 500:
        return False, sig
    return True, sig

# principle_frequency_finder finds the period of a speech signal using autocorrelation.
def principle_frequency_finder(sig):
    t = int(len(sig) / 2)
    corr = np.zeros(t)

    for index in np.arange(t):
        ACF_delay = sig[index:]
        L = (t - index) / 2
        m = np.sum(sig[int(t - L): int(t + L + 1)] ** 2) + np.sum(ACF_delay[int(t - L): int(t + L + 1)] ** 2)
        r = np.sum(sig[int(t - L): int(t + L + 1)] * ACF_delay[int(t - L): int(t + L + 1)])
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

    return (max_index, corr)

# Process each TextGrid file in the input path.
for fn in os.listdir(inputPath):
    fileName, ext = os.path.splitext(fn)
    if ext == ".TextGrid":
        tg = textgrid.openTextgrid(os.path.join(inputPath, fn), includeEmptyIntervals=False)
        
        # Use the 'phones' tier from the TextGrid.
        phoneTier = tg.getTier('phones')
        
        wavFile = os.path.join(inputPath, fileName + ".wav")
        sig, samplerate = sf.read(wavFile)
        
        # Filter phoneme entries for voiced and voiceless segments.
        voiced_list = [ele for ele in phoneTier.entries if ele[2] in voiced_phones]
        voiceless_list = [ele for ele in phoneTier.entries if ele[2] in voiceless_phones]
    
        valid_voiced_list = [
            head_tail_scissor(wav_fraction_finder(ele[0], ele[1], sig))[1] 
            for ele in voiced_list if head_tail_scissor(wav_fraction_finder(ele[0], ele[1], sig))[0]
        ]
        valid_voiceless_list = [
            head_tail_scissor(wav_fraction_finder(ele[0], ele[1], sig))[1] 
            for ele in voiceless_list if head_tail_scissor(wav_fraction_finder(ele[0], ele[1], sig))[0]
        ]

        counter = 0
        for audio_segment in valid_voiced_list:
            audio_filename = f'D:\\phonetic\\new_consonant_recognition\\audio_segment\\voiced\\voiced{fileName}_{counter}.wav'
            sf.write(audio_filename, audio_segment, samplerate)
            counter += 1
            print(f"Saved {fileName}_{counter} to the voiced folder.")

        counter = 0
        for audio_segment in valid_voiceless_list:
            audio_filename = f'D:\\phonetic\\new_consonant_recognition\\audio_segment\\voiceless\\voiceless{fileName}_{counter}.wav'
            sf.write(audio_filename, audio_segment, samplerate)
            counter += 1
            print(f"Saved {fileName}_{counter} to the voiceless folder.")

      
      
