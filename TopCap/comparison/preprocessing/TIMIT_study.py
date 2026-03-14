#coding=utf-8
from sphfile import SPHFile
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

# Put all the files in the Train folder of the TIMIT dataset into one folder,
# and include the subfolder path in the file names.

# Source folder path
source_folder = "~/dataset"
# Target folder path
target_folder = "~/dataset1"

# Traverse all folders and files under the source folder
for speaker_folder in os.listdir(source_folder):
    speaker_path = os.path.join(source_folder, speaker_folder)
    
    if os.path.isdir(speaker_path):
        for root, dirs, files in os.walk(speaker_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Get the subfolder name
                nomatter_name, subfolder_name = os.path.split(root)
                # Get the file name and file extension
                file_name, file_extension = os.path.splitext(file)
                # Construct a new file name by inserting the folder name into the file name
                new_file_name = f"{os.path.basename(speaker_path)}_{subfolder_name}_{file}"
                # Target file path
                target_file_path = os.path.join(target_folder, new_file_name)
                # Copy and rename the file
                shutil.copy(file_path, target_file_path)

# The original files cannot be played because they are displayed as WAV files but are actually SPH files.
# After reading them, replace the file extension with .wav.
if __name__ == "__main__":
    path = "../phonetic/dataset1/*.wav"
    sph_files = glob.glob(path)
    print(len(sph_files), "train utterances")
    for i in sph_files:
        sph = SPHFile(i)
        # Get audio data and sample rate
        audio_data = sph.content  # Read the audio content
        sample_rate = 16000
        # Specify the folder path and file name; split the file path
        name, ext = os.path.splitext(i)
        # Split into folder name and file name
        folder_name, file_name = os.path.split(name)
        # Your target folder path
        output_folder = "~/dataset2"  
        # Write the array as a WAV audio file
        wavfile.write(os.path.join(output_folder, file_name + "_nn.wav"), sample_rate, audio_data.astype(np.int16))
        # os.remove(i)    # No need to delete the original SPH file
    print("Completed")

M = 100
max_edge_length = 1
samplerate = 16000

# wav_fraction_finder finds the corresponding portion of a wav signal given an interval.
def wav_fraction_finder(start_time, end_time, sig):
    sig_fraction = sig[int(start_time * samplerate): int(end_time * samplerate)]
    return sig_fraction

# head_tail_scissor erases the parts at the head and tail of a signal with amplitude smaller than 0.05.
# It can also be used to check if the length of the trimmed signal is greater than 500.
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

# principle_frequency_finder finds the period of a speech signal.
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

# Cut out all selected specific phonemes from the Train set, perform time delay embedding (TDE) + persistent homology (PD),
# and record the birth-death times and voiced/voiceless status into three columns in a CSV file.

# Two lists of phonemes.
voiced_phones = ['v', 'l', 'ng', 'm', 'n', 'y', 'zh']
voiceless_phones = ['f', 'th', 't', 's', 'k', 'ch']

# Audio folder path
audio_folder = "~/dataset2"
# PHN folder path
phn_folder = "~/dataset1"

# CSV file name for outputting TopCap information.
csv_name = 'PD_TIMIT.csv'

# Open the CSV file for writing.
with open(csv_name, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)

# Target folder paths for segmented files.
output_folder_1 = "~/TIMIT_voiced"  
output_folder_2 = "~/TIMIT_voiceless" 

# Traverse all files in the audio folder.
for audio_file in os.listdir(audio_folder):
    if audio_file.endswith('.wav'):
        audio_file_path = os.path.join(audio_folder, audio_file)
        
        # Read the audio WAV file.
        sig, sample_rate = sf.read(os.path.join(audio_folder, audio_file))
        
        # Find the corresponding PHN file.
        filename = os.path.splitext(audio_file)[0][:-3]
        phn_file = os.path.splitext(audio_file)[0][:-3] + '.phn'
        phn_file_path = os.path.join(phn_folder, phn_file)
        
        # If the PHN file exists.
        if os.path.exists(phn_file_path):
            # Read the contents of the PHN file.
            with open(phn_file_path, 'r') as phn_file:
                for line in phn_file:
                    start_time, end_time, phone = line.split()
                    
                    # Check if the phoneme is in the voiced phonemes list.
                    if phone in voiced_phones:
                        # Process voiced phonemes; slice the audio based on the start and end time.
                        print(f"Found voiced phone '{phone}' from {start_time} to {end_time} in {audio_file}")
                        segment = sig[int(start_time): int(end_time)]
                        # Write the array as a WAV audio file by multiplying the float data by 32767 (the maximum value for int16)
                        # and converting it to int16.
                        wavfile.write(os.path.join(output_folder_1, f"{filename}_{phone}_{start_time}.wav"), sample_rate, (segment * 32767).astype(np.int16))
                        
                    elif phone in voiceless_phones:
                        # Process voiceless phonemes; slice the audio based on the start and end time.
                        print(f"Found voiceless phone '{phone}' from {start_time} to {end_time} in {audio_file}")
                        segment = sig[int(start_time): int(end_time)]
                        # Write the array as a WAV audio file by multiplying the float data by 32767 and converting to int16.
                        wavfile.write(os.path.join(output_folder_2, f"{filename}_{phone}_{start_time}.wav"), sample_rate, (segment * 32767).astype(np.int16))

# Initialize two empty lists.
valid_voiced_list = []
valid_voiceless_list = []

# Traverse all files in the audio folder.
for audio_file in os.listdir(audio_folder):
    if audio_file.endswith('.wav'):
        audio_file_path = os.path.join(audio_folder, audio_file)
        
        # Read the audio WAV file.
        sig, sample_rate = sf.read(os.path.join(audio_folder, audio_file))
        
        # Find the corresponding PHN file.
        filename = os.path.splitext(audio_file)[0][:-3]
        phn_file = os.path.splitext(audio_file)[0][:-3] + '.phn'
        phn_file_path = os.path.join(phn_folder, phn_file)
        
        # If the PHN file exists.
        if os.path.exists(phn_file_path):
            # Read the contents of the PHN file.
            with open(phn_file_path, 'r') as phn_file:
                for line in phn_file:
                    start_time, end_time, phone = line.split()
                    
                    # Check if the phoneme is in the voiced phonemes list.
                    if phone in voiced_phones:
                        # Process voiced phonemes; slice the audio based on the start and end time
                        # and append the segment to the valid_voiced_list.
                        print(f"Found voiced phone '{phone}' from {start_time} to {end_time} in {audio_file}")
                        segment = sig[int(start_time): int(end_time)]
                        valid_voiced_list.append(segment)
                        
                    elif phone in voiceless_phones:
                        # Process voiceless phonemes; slice the audio based on the start and end time
                        # and append the segment to the valid_voiceless_list.
                        print(f"Found voiceless phone '{phone}' from {start_time} to {end_time} in {audio_file}")
                        segment = sig[int(start_time): int(end_time)]
                        valid_voiceless_list.append(segment)

# Process voiced segments.
T_voiced = [0] * len(valid_voiced_list)
for i in range(len(valid_voiced_list)):
    T_voiced[i], corr = principle_frequency_finder(np.array(valid_voiced_list[i]))
    T_voiced[i] = T_voiced[i]
    print(f"voiced_principle_frequency_find_{i}")

delay_voiced = [round(ele * 6 / M) for ele in T_voiced]
for element in range(len(delay_voiced)):
    if delay_voiced[element] == 0:
        delay_voiced[element] = 1

# Process voiceless segments.
T_voiceless = [0] * len(valid_voiceless_list)
for i in range(len(valid_voiceless_list)):
    T_voiceless[i], corr = principle_frequency_finder(np.array(valid_voiceless_list[i]))
    T_voiceless[i] = T_voiceless[i]
    print(f"voiceless_principle_frequency_find_{i}")

delay_voiceless = [round(ele * 6 / M) for ele in T_voiceless]
for element in range(len(delay_voiceless)):
    if delay_voiceless[element] == 0:
        delay_voiceless[element] = 1

# Write the persistent homology results for voiced segments to the CSV file.
with open(csv_name, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)        
    for i in range(len(valid_voiced_list)):
        data = valid_voiced_list[i]
        if delay_voiced[i] * M > len(data):
            delay_voiced[i] = int(np.floor(len(data) / M))
        if delay_voiced[i] == 0:
            delay_voiced[i] = 1
        point_Cloud = timedelay.TimeDelayEmbedding(M, delay_voiced[i], 5)
        if data.size > 0:
            Points = point_Cloud(data)
        else:
            continue
        if len(Points) < 40:               
            continue
        dgms = ripser(Points, maxdim=1)['dgms']
        dgms = dgms[1]
        if dgms.size == 0:
            continue
        persistent_time = [ele[1] - ele[0] for ele in dgms]            
        index = argmax(persistent_time)
        birth_date = dgms[index][0]
        lifetime = persistent_time[index]
        writer.writerow((birth_date, lifetime, 1))
        print(f"written a row {i}")

# Write the persistent homology results for voiceless segments to the CSV file.
with open(csv_name, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)        
    for i in range(len(valid_voiceless_list)):
        data = valid_voiceless_list[i]
        if delay_voiceless[i] * M > len(data):
            delay_voiceless[i] = int(np.floor(len(data) / M))
        if delay_voiceless[i] == 0:
            delay_voiceless[i] = 1
        point_Cloud = timedelay.TimeDelayEmbedding(M, delay_voiceless[i], 5)
        
        try:
            # This is the block of code that may throw an exception.
            Points = point_Cloud(data)
        except Exception as e:
            # Catch all exceptions.
            print(f"An error occurred: {e}")
    
        if len(Points) < 40:               
            continue
        dgms = ripser(Points, maxdim=1)['dgms']
        dgms = dgms[1]
        if dgms.size == 0:
            continue
        persistent_time = [ele[1] - ele[0] for ele in dgms]            
        index = argmax(persistent_time)
        birth_date = dgms[index][0]
        lifetime = persistent_time[index]
        writer.writerow((birth_date, lifetime, 2))
        print(f"written a row {i}")

                    
                    
                    
                    


