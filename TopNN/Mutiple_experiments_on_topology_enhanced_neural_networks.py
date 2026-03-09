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
#from Gauss_SVM_acc_5 import parallel_gaussian_svm_cv
import pandas as pd
import random
from scipy import signal
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import time
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

import torch.multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool


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

def normalize_signal(signal):
    max_abs_value = np.max(np.abs(signal))
    if max_abs_value == 0:
        return signal  
    return signal / max_abs_value

def add_noise(y, seed=110, noise_type='gaussian', snr_db=5):
    y_normalized = normalize_signal(y)
    
    if snr_db is None:
        return y_normalized
    
    rng = np.random.RandomState(seed)
    signal_power = np.mean(y_normalized**2)
    
    if noise_type == 'gaussian':
        noise = rng.normal(0, 1, len(y_normalized))  
    elif noise_type == 'uniform':
        noise = rng.uniform(-1, 1, len(y_normalized))
    elif noise_type == 'impulse':
        noise = np.zeros_like(y_normalized)
        num_impulses = int(len(y_normalized) * 0.001)  
        indices = rng.randint(0, len(y_normalized), num_impulses)    
        noise[indices] = rng.uniform(-0.5, 0.5, num_impulses)  
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    noise_power = np.maximum(np.mean(noise**2), 1e-10)  # 确保噪声功率不为零
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scaling_factor = np.sqrt(target_noise_power / noise_power)
    scaled_noise = noise * scaling_factor

    noisy_y = y_normalized + scaled_noise
    
    noisy_y_normalized = normalize_signal(noisy_y)
    
    return noisy_y_normalized

M=100
max_edge_length=1
samplerate=16000

def wav_fraction_finder(start_time, end_time,sig):
    sig_fraction=sig[int(start_time*samplerate):int(end_time*samplerate)]
    return sig_fraction

def principle_frequency_finder_top3(time_series):
    n = len(time_series)  
    yf = np.fft.fft(time_series)  
    xf = np.fft.fftfreq(n, d=1.0)  

    yf_single_side = yf[:n//2]
    magnitude_spectrum = 2.0/n * np.abs(yf_single_side)

    non_zero_magnitude_spectrum = magnitude_spectrum[1:]
    non_zero_xf = xf[1:]

    peaks = []
    for i in range(1, len(non_zero_magnitude_spectrum) - 1):
        if (non_zero_magnitude_spectrum[i] > non_zero_magnitude_spectrum[i-1]) and \
        (non_zero_magnitude_spectrum[i] > non_zero_magnitude_spectrum[i+1]):
            peaks.append(i)

    if len(peaks) < 3:
        raise ValueError("Not enough local maxima to find the top three frequencies.")

    peak_frequencies = [non_zero_xf[p] for p in peaks]
    peak_magnitudes = [non_zero_magnitude_spectrum[p] for p in peaks]

    top_three_indices = sorted(range(len(peak_magnitudes)), key=lambda k: peak_magnitudes[k], reverse=True)[:3]
    frequencies_estimated = [peak_frequencies[i] for i in top_three_indices]
    periods_estimated = [1.0 / freq for freq in frequencies_estimated]

    return (periods_estimated[0])

def principle_frequency_finder(time_series):
    n = len(time_series)  
    yf = np.fft.fft(time_series) 
    xf = np.fft.fftfreq(n, d=1.0)  

    yf_single_side = yf[:n//2]
    magnitude_spectrum = 2.0/n * np.abs(yf_single_side)

    non_zero_magnitude_spectrum = magnitude_spectrum[1:]
    non_zero_xf = xf[1:]

    peaks = []
    for i in range(1, len(non_zero_magnitude_spectrum) - 1):
        if (non_zero_magnitude_spectrum[i] > non_zero_magnitude_spectrum[i-1]) and \
        (non_zero_magnitude_spectrum[i] > non_zero_magnitude_spectrum[i+1]):
            peaks.append(i)

    peak_frequencies = [non_zero_xf[p] for p in peaks]
    peak_magnitudes = [non_zero_magnitude_spectrum[p] for p in peaks]
    
    if len(peak_magnitudes) == 0:
        return 1  
    
    top_index = np.argmax(peak_magnitudes)
    frequency_estimated = peak_frequencies[top_index]
    period_estimated = 1.0 / frequency_estimated
    
    return period_estimated
    

def principle_frequency_finder_acf(sig):
    t=int(len(sig)/2)
    corr=np.zeros(t)

    for index in np.arange(t):
        ACF_delay=sig[index:]
        L=(t-index)/2
        m = np.sum(sig[int(t-L):int(t+L+1)]**2) + np.sum(ACF_delay[int(t-L):int(t+L+1)]**2)
        r = np.sum(sig[int(t-L):int(t+L+1)]*ACF_delay[int(t-L):int(t+L+1)])
        corr[index] = 2*r/m

    zc = np.zeros(corr.size-1)
    zc[(corr[0:-1] < 0)*(corr[1::] > 0)] = 1
    zc[(corr[0:-1] > 0)*(corr[1::] < 0)] = -1

    admiss = np.zeros(corr.size)
    admiss[0:-1] = zc
    for i in range(1, corr.size):
        if admiss[i] == 0:
            admiss[i] = admiss[i-1]

    maxes = np.zeros(corr.size)
    maxes[1:-1] = (np.sign(corr[1:-1] - corr[0:-2])==1)*(np.sign(corr[1:-1] - corr[2::])==1)
    maxidx = np.arange(corr.size)
    maxidx = maxidx[maxes == 1]
    max_index = 0
    if len(corr[maxidx]) > 0:
        max_index = maxidx[np.argmax(corr[maxidx])]

    return max_index


def time_delay_embedding_circular_fixnum(time_series, embedding_dim=3, delay=1, fixnum=200):
    if not isinstance(embedding_dim, int) or embedding_dim < 1:
        raise ValueError("Embedding dimension must be a positive integer.")
    if not isinstance(delay, int) or delay < 1:
        raise ValueError("Delay must be a positive integer.")
    if not isinstance(fixnum, int) or fixnum < 1:
        raise ValueError("Fixnum must be a positive integer.")
    if not isinstance(time_series, (list, np.ndarray)):
        raise TypeError("Input time_series must be a list or numpy array.")
    
    N = len(time_series)
    if N < embedding_dim:
        raise ValueError(
            f"Time series length ({N}) must be ≥ embedding dimension ({embedding_dim})."
        )
    
    M = min(N, fixnum)
    
    if N == M:  
        indices = range(N)
    else:  
        indices = [int(i) for i in np.linspace(0, N-1, num=M, dtype=int)]
    
    embedded_data = [
        [time_series[(idx + j * delay) % N] for j in range(embedding_dim)]
        for idx in indices
    ]
    
    return np.array(embedded_data)                       



# Compute Persistent Homology
def process_single_item(args, data_category=1, M=100):
    i, data, delay_values, name_list = args
    
    try:
        if delay_values[i] * M > len(data):
            delay_values[i] = int(np.floor(len(data) / M))
        if delay_values[i] == 0:
            delay_values[i] = 1
            
      
        tau= delay_values[i]
    
        if data.size == 0:
            return None
            
        if np.isnan(data).any():
            return None
        else:
            Points = time_delay_embedding_circular_fixnum(data,M,tau,200)
        
        dgms = ripser(Points, maxdim=1)['dgms'][1]
        if dgms.size == 0:
            return None
            
        persistent_time = [ele[1]-ele[0] for ele in dgms]
        index = np.argmax(persistent_time)
        
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
    workers = workers or max(1, cpu_count()-1)
    
    task_args = [(i, data, delay_list, name_list) 
                for i, data in enumerate(data_list)]
    
    with Pool(workers) as pool:
        processor = partial(process_single_item, data_category=category)
        
        results = []
        for idx, result in enumerate(pool.imap(processor, task_args, chunksize=50)):
            if result:
                results.append(result)
            if idx % 1000 == 0:
                print(f"Processed {idx}/{len(data_list)} items")
                
    with open(csv_name, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for res in sorted(results, key=lambda x: x[0]):
            writer.writerow(res)  
            
    print(f"Category {category} completed. Total {len(results)} valid entries.")


def build_file_index(folder_paths):
    file_index = {}
    for folder in folder_paths:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            file_index[file] = file_path 
    return file_index


def extract_mfcc(file_names, file_index,add_noise_flag=False, 
                noise_params=None):
    consonants_mfcc = []
    missing_files = []

    for idx, filename in enumerate(file_names):
        file_path = file_index.get(filename)  
        if file_path and os.path.exists(file_path):  
            y, sr = librosa.load(file_path, sr=16000)  
            
            if add_noise_flag:
                current_seed = noise_params['base_seed'] + Num[idx]
                y = add_noise(y,seed=current_seed,noise_type=noise_params['noise_type'],snr_db=noise_params['snr_db'],)
    
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=256)
            mfccs = torch.from_numpy(mfccs.T)  
            consonants_mfcc.append(mfccs)            
            
        else:
            missing_files.append(filename)
    
    print(f"Obtain {len(consonants_mfcc)}/{len(file_names)} documents")
    if missing_files:
        print("Invalid documents:", ", ".join(missing_files[:3]) + "...")
    
    return consonants_mfcc

# Create a custom ConsonantDataset and split it into training and test set
class ConsonantDataset(Dataset):
    def __init__(self, sequences, topfeatures, labels):
        self.sequences = sequences
        self.topfeatures=topfeatures
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.topfeatures[idx], self.labels[idx]  
    
def collate_fn(batch):
    sequences,topfeatures, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.int64)
    # sequence padding
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    sequences_padded=sequences_padded.to(dtype=torch.float32)
    topfeatures = torch.stack(topfeatures, dim=0).to(dtype=torch.float32)
    labels = torch.cat(labels, dim=0)
    return sequences_padded, topfeatures,labels, lengths

# Initialize the weight
def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def init_weights_fn(initial_fc_weights=True):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            if initial_fc_weights:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    return init_weights


# Model_1
class TopGRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size,feature_dim, num_layers, output_size):
        super(TopGRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size) 
        self.fc = nn.Linear(hidden_size+feature_dim, output_size)
        self.layer_norm = nn.LayerNorm(hidden_size) 
        
        
    
    def forward(self, x, topfeatures,lengths):
        # Pack padded sequences
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # Encoder
        _, hidden = self.gru(packed)
        gru_features = self.bn(hidden[-1])  
        combined_features = torch.cat((gru_features, topfeatures), dim=1)
        
        # Decoder
        out = self.fc(combined_features)
        return out

# Model 2
class ZeroGRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size,feature_dim, num_layers, output_size):
        super(ZeroGRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size) 
        self.fc = nn.Linear(hidden_size+feature_dim, output_size)
        self.layer_norm = nn.LayerNorm(hidden_size) 
        
    
    def forward(self, x, zerofeatures,lengths):
        # Pack padded sequences
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # Encoder
        _, hidden = self.gru(packed)
        gru_features = self.bn(hidden[-1])  
        combined_features = torch.cat((gru_features, zerofeatures), dim=1)
        
        # Decoder
        out = self.fc(combined_features)
        return out

# Model 3
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,initial_w=None):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size) 
        self.fc = nn.Linear(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(hidden_size) 

        if initial_w is not None:
            with torch.no_grad():
                self.fc.weight[:, :6].copy_(initial_w[:, :6])     
    
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        out = self.bn(hidden[-1])    
        out = self.fc(out)
        return out
        
def run_experiment(n):
    k_folds =5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=n)

    fold_training_accuracies_1 = []
    fold_testing_accuracies_1=[]
    fold_training_accuracies_2 = []
    fold_testing_accuracies_2=[]
    fold_training_accuracies_3 = []
    fold_testing_accuracies_3=[]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)


    for fold, (train_idx, test_idx) in enumerate(kf.split(consonants_topfeatures)):
        seed_number = random.randint(1, 100)

        topfeatures_train, topfeatures_test = consonants_topfeatures[train_idx], consonants_topfeatures[test_idx]
        
        X_train = [consonants_mfcc[i] for i in train_idx]
        X_test = [consonants_mfcc[i] for i in test_idx]
        y_train = [consonant_labels[i] for i in train_idx]
        y_test = [consonant_labels[i] for i in test_idx]
        

    
        train_dataset = ConsonantDataset(X_train, topfeatures_train, y_train)
        test_dataset = ConsonantDataset(X_test, topfeatures_test, y_test)

    
        trainloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)


        set_global_seed(seed_number)
        model_1 =TopGRUClassifier(input_size, hidden_size,feature_dim, num_layers, output_size) 
        model_1.to(device)
        model_1.apply(init_weights_fn(initial_fc_weights=True))
        initial_w= model_1.fc.weight.clone().detach() 
        # print("weight of model1",model_1.fc.weight)
        # initial_w = model_1.fc.weight.clone().detach() 


        # Loss and optimizer
        set_global_seed(seed_number)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model_1.parameters(), lr=learning_rate)

        # Set the training parameters
        running_loss = 0
        running_total = 0
        running_correct = 0
        loss_list_1 = [] 
        loss_list_print_every_1 = [] 
        accuracy_list_print_every_1 =[]
        print_every = 100 # Print out the accuracy every {print_every} batches
        training_accuracy_1=0


        # Train the model
        model_1.train()

        for epoch in range(num_epochs):
            for i, (sequences, topfeatures,labels,lengths) in enumerate(trainloader):
                sequences, topfeatures,labels,lengths = sequences.to(device),topfeatures.to(device), labels.to(device),lengths.to(device)
                optimizer.zero_grad()
                outputs = model_1(sequences,topfeatures,lengths)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Store running loss, total, correct
                running_loss += loss.item()
                outputs=outputs.cpu()
                predicted = (outputs.detach().numpy() >= 0.5).astype(int) 
                predicted = torch.from_numpy(predicted)
                predicted=predicted.to(device)
                running_total += labels.size(0)
                running_correct += (predicted == labels).sum().item()       
                loss_list_1.append(loss.item())

                # Print average training loss and accuracy 
                if (i+1) % print_every == 0:         
                    
                    # Store running loss and accuracy
                    loss_list_print_every_1.append(running_loss/print_every)
                    accuracy_list_print_every_1.append(running_correct/running_total)
                    # print("Epoch: {}/{} -- Batches: {}/{} -- Training loss: {:.3f} -- Training accuracy: {:.3f}"
                    #     .format(epoch+1, num_epochs, i+1, len(trainloader), 
                    #             running_loss/print_every, running_correct/running_total))                        
                    
                    # Reset running loss and accuracy
                    running_loss = 0
                    running_total = 0
                    running_correct = 0


        training_accuracy_1=accuracy_list_print_every_1[-1]
        fold_training_accuracies_1.append(training_accuracy_1)       
        # print("Training complete. Total training time: {:.1f} seconds. Training accuracy:{:.2f}.".format(time.time() - start_time,training_accuracy_1))

        running_loss = 0
        labels_true = np.array([], dtype=int)
        labels_pred = np.array([], dtype=int)

        model_1.eval()

        with torch.no_grad():   
            for i, (sequences,topfeatures, labels,lengths) in enumerate(testloader):
                sequences, topfeatures,labels,lengths = sequences.to(device),topfeatures.to(device), labels.to(device),lengths.to(device)
            
                outputs = model_1(sequences,topfeatures,lengths)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels.float())

                running_loss += loss.item()
                outputs=outputs.cpu()
                predicted = (outputs.numpy() >= 0.5).astype(int) 
            
                # calculate accuracy 
                labels_true = np.append(labels_true, labels.cpu().numpy())
                labels_pred = np.append(labels_pred, predicted)

            
        test_accuracy_1 = np.equal(labels_pred, labels_true).mean()
        fold_testing_accuracies_1.append(test_accuracy_1)            
            
        # print("Evaluating network on {} consonants in test set -- Test loss: {:.3f} -- Test accuracy: {:.3f}"
        # .format(len(testloader.dataset), running_loss/len(testloader), test_accuracy_1))
        
        set_global_seed(seed_number)
        model_2=ZeroGRUClassifier(input_size, hidden_size, feature_dim, num_layers, output_size) 
        model_2.to(device)
        model_2.apply(init_weights_fn(initial_fc_weights=True))
        # print("weight of model2",model_2.fc.weight)
        # initial_w = model_2.fc.weight.clone().detach() 
        
        # Loss and optimizer
        set_global_seed(seed_number)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model_2.parameters(), lr=learning_rate)

        # Record the training time
        start_time = time.time()


        # Set the training parameters
        running_loss = 0
        running_total = 0
        running_correct = 0
        loss_list_2 = [] 
        loss_list_print_every_2 = [] 
        accuracy_list_print_every_2 =[]
        print_every = 20 # Print out the accuracy every {print_every} batches
        training_accuracy_2=0


        # Train the model
        model_2.train()

        for epoch in range(num_epochs):
            for i, (sequences, topfeatures,labels,lengths) in enumerate(trainloader):
                sequences, topfeatures,labels,lengths = sequences.to(device),topfeatures.to(device), labels.to(device),lengths.to(device)
                zerofeatures= torch.zeros_like(topfeatures)
                optimizer.zero_grad()
                outputs = model_2(sequences,zerofeatures,lengths)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Store running loss, total, correct
                running_loss += loss.item()
                outputs=outputs.cpu()
                predicted = (outputs.detach().numpy() >= 0.5).astype(int) 
                predicted = torch.from_numpy(predicted)
                predicted=predicted.to(device)
                running_total += labels.size(0)
                running_correct += (predicted == labels).sum().item()       
                loss_list_2.append(loss.item())

                # Print average training loss and accuracy 
                if (i+1) % print_every == 0:         
                    
                    # Store running loss and accuracy
                    loss_list_print_every_2.append(running_loss/print_every)
                    accuracy_list_print_every_2.append(running_correct/running_total)
                    
                    # Reset running loss and accuracy
                    running_loss = 0
                    running_total = 0
                    running_correct = 0
        
        training_accuracy_2=accuracy_list_print_every_2[-1]
        fold_training_accuracies_2.append(training_accuracy_2)       
        # print("Training complete. Total training time: {:.1f} seconds. Training accuracy:{:.3f}.".format(time.time() - start_time,training_accuracy_2))

        running_loss = 0
        labels_true = np.array([], dtype=int)
        labels_pred = np.array([], dtype=int)

        model_2.eval()

        with torch.no_grad():   
            for i, (sequences,topfeatures, labels,lengths) in enumerate(testloader):
                sequences, topfeatures,labels,lengths = sequences.to(device),topfeatures.to(device), labels.to(device),lengths.to(device)
                zerofeatures= torch.zeros_like(topfeatures)
                outputs = model_2(sequences,zerofeatures,lengths)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels.float())

                running_loss += loss.item()
                outputs=outputs.cpu()
                predicted = (outputs.numpy() >= 0.5).astype(int) 
            
                # calculate accuracy 
                labels_true = np.append(labels_true, labels.cpu().numpy())
                labels_pred = np.append(labels_pred, predicted)

            
        test_accuracy_2 = np.equal(labels_pred, labels_true).mean()
        fold_testing_accuracies_2.append(test_accuracy_2)            
            
        
        set_global_seed(seed_number)
        model_3=GRUClassifier(input_size, hidden_size, num_layers, output_size,initial_w) 
        model_3.to(device)
        model_3.apply(init_weights_fn(initial_fc_weights=False))
        # print("weight of model3",model_3.fc.weight)
        # initial_w = model_3.fc.weight.clone().detach() 

        # Loss and optimizer
        set_global_seed(seed_number)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model_3.parameters(), lr=learning_rate)

        # Record the training time
        start_time = time.time()



        # Set the training parameters
        running_loss = 0
        running_total = 0
        running_correct = 0
        loss_list_3 = [] 
        loss_list_print_every_3 = [] 
        accuracy_list_print_every_3 =[]
        print_every = 20 # Print out the accuracy every {print_every} batches
        training_accuracy_3=0


        # Train the model
        model_3.train()

        for epoch in range(num_epochs):
            for i, (sequences, topfeatures,labels,lengths) in enumerate(trainloader):
                sequences, topfeatures,labels,lengths = sequences.to(device),topfeatures.to(device), labels.to(device),lengths.to(device)
                optimizer.zero_grad()
                outputs = model_3(sequences,lengths)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Store running loss, total, correct
                running_loss += loss.item()
                outputs=outputs.cpu()
                predicted = (outputs.detach().numpy() >= 0.5).astype(int) 
                predicted = torch.from_numpy(predicted)
                predicted=predicted.to(device)
                running_total += labels.size(0)
                running_correct += (predicted == labels).sum().item()       
                loss_list_3.append(loss.item())

                # Print average training loss and accuracy 
                if (i+1) % print_every == 0:         
                    
                    # Store running loss and accuracy
                    loss_list_print_every_3.append(running_loss/print_every)
                    accuracy_list_print_every_3.append(running_correct/running_total)
                    
                    # Reset running loss and accuracy
                    running_loss = 0
                    running_total = 0
                    running_correct = 0

        final_w = model_3.fc.weight.clone().detach()  

        training_accuracy_3=accuracy_list_print_every_3[-1]
        fold_training_accuracies_3.append(training_accuracy_3)       
        # print("Training complete. Total training time: {:.1f} seconds. Training accuracy:{:.2f}.".format(time.time() - start_time,training_accuracy_3))

        running_loss = 0
        labels_true = np.array([], dtype=int)
        labels_pred = np.array([], dtype=int)

        model_3.eval()

        with torch.no_grad():   
            for i, (sequences,topfeatures, labels,lengths) in enumerate(testloader):
                sequences, topfeatures,labels,lengths = sequences.to(device),topfeatures.to(device), labels.to(device),lengths.to(device)
                outputs = model_3(sequences,lengths)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels.float())

                running_loss += loss.item()
                outputs=outputs.cpu()
                predicted = (outputs.numpy() >= 0.5).astype(int) 
            
                # calculate accuracy 
                labels_true = np.append(labels_true, labels.cpu().numpy())
                labels_pred = np.append(labels_pred, predicted)

            
        test_accuracy_3 = np.equal(labels_pred, labels_true).mean()
        fold_testing_accuracies_3.append(test_accuracy_3)            
            
    print(f'Average training accuracy of TopGRU: {np.mean(fold_training_accuracies_1):.4f}')
    print(f'Average testing accuracy of TopGRU: {np.mean(fold_testing_accuracies_1):.4f}')   
    print(f'Average training accuracy of ZeroGRU: {np.mean(fold_training_accuracies_2):.4f}')
    print(f'Average testing accuracy of ZeroGRU: {np.mean(fold_testing_accuracies_2):.4f}') 
    print(f'Average training accuracy of GRU: {np.mean(fold_training_accuracies_3):.4f}')
    print(f'Average testing accuracy of GRU: {np.mean(fold_testing_accuracies_3):.4f}') 
    return (
        np.mean(fold_training_accuracies_1),
        np.mean(fold_testing_accuracies_1),
        np.mean(fold_training_accuracies_2),
        np.mean(fold_testing_accuracies_2),
        np.mean(fold_training_accuracies_3),
        np.mean(fold_testing_accuracies_3),
    )




if __name__ == "__main__":
    M = 100 
    audio_folder = 'D:\\code\\Signal_Recognition\\ALLSSTAR_ALL\\ALLSSTAR_ALL'  
 
    csv_name = 'TIMIT_NoiseNone.csv'

    valid_voiced_list = []
    valid_voiceless_list = []

    name_voiced_list = []
    name_voiceless_list = []

    voiced_folder = os.path.join(audio_folder, 'voiced')
    voiceless_folder = os.path.join(audio_folder, 'voiceless')

    failed_files = [] 

    for filename in os.listdir(voiced_folder):
        if filename.endswith(('.wav', '.flac')):
            file_path = os.path.join(voiced_folder, filename)
            data, sample_rate = read_audio_file(file_path, filename)
            if data is not None:
                valid_voiced_list.append(data)
                name_voiced_list.append(filename)
            

    for filename in os.listdir(voiceless_folder):
        if filename.endswith(('.wav', '.flac')):
            file_path = os.path.join(voiceless_folder, filename)
            data, sample_rate = read_audio_file(file_path, filename)
            if data is not None:
                valid_voiceless_list.append(data)
                name_voiceless_list.append(filename)
              

    print(f'number of valid voiced audio files: {len(valid_voiced_list)}')
    print(f'number of valid voiceless audio files: {len(valid_voiceless_list)}')


    noisy_voiced_list = []
    noisy_voiceless_list = []

    noise_params = {
        'noise_type': 'gaussian',
        'snr_db': None,
        'base_seed': 110  
    }

    for idx, y in enumerate(valid_voiced_list):
        current_seed = noise_params['base_seed'] + idx
        
        noisy_y = add_noise(
            y, 
            seed=current_seed, 
            noise_type=noise_params['noise_type'],
            snr_db=noise_params['snr_db'],
        )
        noisy_voiced_list.append(noisy_y)


    for idx, y in enumerate(valid_voiceless_list):
        current_seed = noise_params['base_seed'] + idx
        
        noisy_y = add_noise(
            y, 
            seed=current_seed, 
            noise_type=noise_params['noise_type'],
            snr_db=noise_params['snr_db'],
        )
        noisy_voiceless_list.append(noisy_y)


    valid_voiced_list = noisy_voiced_list
    valid_voiceless_list = noisy_voiceless_list


    # Extract topological features
    k=6

    T_voiced=[0]*len(valid_voiced_list)
    for i in range(len(valid_voiced_list)):
        T_voiced[i]=principle_frequency_finder(np.array(valid_voiced_list[i]))
        T_voiced[i]=T_voiced[i]
        print(f"voiced_principle_frequency_find_{i}")

    delay_voiced=[round(ele*k/M) for ele in T_voiced]
    for element in range(len(delay_voiced)):
        if delay_voiced[element]==0:
            delay_voiced[element]=1

    T_voiceless=[0]*len(valid_voiceless_list)
    for i in range(len(valid_voiceless_list)):
        T_voiceless[i]=principle_frequency_finder(np.array(valid_voiceless_list[i]))
        T_voiceless[i]=T_voiceless[i]
        print(f"voiceless_principle_frequency_find_{i}")

    delay_voiceless=[round(ele*k/M) for ele in T_voiceless]
    for element in range(len(delay_voiceless)):
        if delay_voiceless[element]==0:
            delay_voiceless[element]=1


    parallel_process_to_csv(
        csv_name,
        valid_voiced_list,
        delay_voiced,
        name_voiced_list,
        category=1,
        workers=12
    )

    parallel_process_to_csv(
        csv_name,
        valid_voiceless_list,
        delay_voiceless,
        name_voiceless_list,
        category=2,
        workers=12
    )

    
    # Load the dataset of consonants and extract features using MFCC
    excel_path = csv_name
    df = pd.read_csv(excel_path).dropna()
    Num = df.iloc[:, 0].values
    Num=np.array(Num)
    topfeatures = df.iloc[:, 3].values
    topfeatures=np.array(topfeatures)
    topfeatures = np.expand_dims(topfeatures, axis=1) 
    consonants_topfeatures = torch.tensor(topfeatures, dtype=torch.float32) 
    print(consonants_topfeatures.shape)


    consonant_labels=df.iloc[:, 4].values
    consonant_labels=torch.tensor(consonant_labels, dtype=torch.float32)-1
    consonant_labels = consonant_labels.unsqueeze(1) 
    print(consonant_labels.shape)


    file_names =df.iloc[:, 1].astype(str).tolist()
    folder_paths = ["D:\\code\\Signal_Recognition\\ALLSSTAR_ALL\\ALLSSTAR_ALL\\voiced","D:\\code\\Signal_Recognition\\ALLSSTAR_ALL\\ALLSSTAR_ALL\\voiceless"] 



    file_index = build_file_index(folder_paths) 
    consonants_mfcc = extract_mfcc(file_names, file_index,
                                add_noise_flag=True,
                                noise_params = {
        'noise_type': 'gaussian',  
        'snr_db': None,              
    'base_seed': 110                 
    })


    # set the parameters
    input_size = 40
    hidden_size = 6
    output_size = 1   
    num_epochs = 30
    batch_size = 128
    learning_rate = 0.001
    num_layers=1
    feature_dim=1

    # Run multiple experiments
    num_experiments = 20 
    train_acc_1, test_acc_1 = [], []
    train_acc_2, test_acc_2 = [], []
    train_acc_3, test_acc_3 = [], []



    for i in range(num_experiments):
        print(f"{i + 1}/{num_experiments} ...")  
        
        t1, tst1, t2, tst2, t3, tst3 = run_experiment(i) 
        train_acc_1.append(t1)
        test_acc_1.append(tst1)
        train_acc_2.append(t2)
        test_acc_2.append(tst2)
        train_acc_3.append(t3)
        test_acc_3.append(tst3)

    print("All the experiments finished!")


    print(train_acc_1)
    print("TopGRU: Training Accuracy - Mean: {:.3f}, Std: {:.4f}".format(np.mean(train_acc_1), np.std(train_acc_1)))
    print("TopGRU: Testing Accuracy  - Mean: {:.3f}, Std: {:.4f}".format(np.mean(test_acc_1), np.std(test_acc_1)))
    print("ZeroGRU: Training Accuracy - Mean: {:.3f}, Std: {:.4f}".format(np.mean(train_acc_2), np.std(train_acc_2)))
    print("ZeroGRU: Testing Accuracy  - Mean: {:.3f}, Std: {:.4f}".format(np.mean(test_acc_2), np.std(test_acc_2)))
    print("GRU: Training Accuracy - Mean: {:.3f}, Std: {:.4f}".format(np.mean(train_acc_3), np.std(train_acc_3)))
    print("GRU: Testing Accuracy  - Mean: {:.3f}, Std: {:.4f}".format(np.mean(test_acc_3), np.std(test_acc_3)))






    


