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
import math

# Data preprocessing
consonant_labels=[]
mfcc_voiced_list = []
mfcc_voiceless_list = []

# Load the dataset of voiced consonants and extract features using MFCC
folder_path ="D:\\code\\Signal_Recognition\\audio_segment\\audio_segment\\voiced"#"D:\\code\\Signal_Recognition\\audio_segment\\voiced"
file_count = 0
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        y, sr = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=40,n_fft=256)
        mfccs = np.transpose(mfccs)
        mfcc_features= torch.from_numpy(mfccs)
    mfcc_voiced_list.append(mfcc_features)
    label = torch.tensor([0])                                                                
    consonant_labels.append(label)
    file_count += 1

    # Select the first 10,0000 data
    if file_count >= 100000:
        break

# Load the dataset of voiceless consonants and extract features using MFCC
folder_path ="D:\\code\\Signal_Recognition\\audio_segment\\audio_segment\\voiceless"
file_count = 0
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        y, sr = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=40,n_fft=256)
        mfccs = np.transpose(mfccs)
        mfcc_features= torch.from_numpy(mfccs)
    mfcc_voiceless_list.append(mfcc_features)
    label = torch.tensor([1])
    consonant_labels.append(label)
    file_count += 1
    
    if file_count >= 100000:
        break

# Merge two datasets and preprocess the data
consonants_mfcc=mfcc_voiced_list+mfcc_voiceless_list

sequence_length = len(max(consonants_mfcc, key=len))
print("Maximum sequence length:", sequence_length)

consonants_mfcc= pad_sequence(consonants_mfcc) # Sequence padding
consonants_mfcc = consonants_mfcc.permute(1,0,2)
consonants_mfcc = consonants_mfcc.to(torch.float32)

# Generate labels for corresponding phonemes
consonant_labels = torch.tensor(consonant_labels,dtype=torch.float32)
consonant_labels = consonant_labels.unsqueeze(1) 

# Create a custom ConsonantDataset and split it into training and test set
class ConsonantDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]  
    
X_train, X_test, y_train, y_test = train_test_split(consonants_mfcc, consonant_labels, test_size=0.2, random_state=10)
    
train_dataset = ConsonantDataset(X_train, y_train)
test_dataset = ConsonantDataset(X_test, y_test)

# Set batch size
batch_size = 64

# Construct the trainloader and testloader
trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


print("Sequences in training data:", len(trainloader.dataset))
print("Sequences in testing data:", len(testloader.dataset))

print("Training batches:", len(trainloader))
print("Batch size:", batch_size)


# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)



# Add absolute position encoding
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe = torch.zeros(max_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, model_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# Construct transformer classifier
class TransformerClassifier(nn.Module):
    def __init__(self,model_dim, output_size, num_heads, num_layers,max_len,dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.positional_encoding = PositionalEncoding(model_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=4*model_dim, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_size)  
        self.dropout = nn.Dropout(dropout)

    def forward(self, x,padding_mask):
        x = self.positional_encoding(x)  
        x = x.permute(1, 0, 2)  
        x = self.transformer(x,src_key_padding_mask=padding_mask)  
        x = x.mean(dim=0) # Global average pooling layer
        x = self.dropout(x)
        x = self.fc(x)  
        return x

#  Generate padding mask   
def generate_padding_mask(x, padding_idx=0):
    return (x == padding_idx)  
    
# Set the training parameters
model_dim = 40  
output_size=1
num_heads = 4 
num_layers = 1  
dropout = 0.1    
max_len = sequence_length 
sequence_length = sequence_length   
learning_rate = 0.0001
num_epochs=20
print_every = 100 # Print out the accuracy every {print_every} epochs


# Instantiate model 
model = TransformerClassifier(model_dim,output_size, num_heads, num_layers,max_len)

# Send model to device
model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Record the training time
start_time = time.time()

print("Start of traing -- Device: {} -- Epochs: {} -- Batches: {} -- Batch size: {}"
      .format(device, num_epochs, len(trainloader), batch_size))

# Initiate variables and lists for training progress
running_loss = 0
running_total = 0
running_correct = 0
loss_list = [] 
loss_list_print_every = [] 
accuracy_list_print_every =[]



# Train the model
model.train()

for epoch in range(num_epochs):
    for i, (lines, labels) in enumerate(trainloader):
        
        
        lines, labels = lines.to(device), labels.to(device)
        
        
        lines = lines.reshape(-1, sequence_length, model_dim)
        padding_mask = generate_padding_mask(lines[:, :, 0], padding_idx=0)  
        
        
        output = model(lines,padding_mask=padding_mask)
        loss = criterion(output, labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        output=output.cpu()
        predicted = (output.detach().numpy() >= 0.5).astype(int) 
        predicted = torch.from_numpy(predicted)
        predicted=predicted.to(device)
        running_total += labels.size(0)
        running_correct += (predicted == labels).sum().item()       
        loss_list.append(loss.item())

        if (i+1) % print_every == 0:         
            print("Epoch: {}/{} -- Batches: {}/{} -- Training loss: {:.3f} -- Training accuracy: {:.3f}"
                  .format(epoch+1, num_epochs, i+1, len(trainloader), 
                          running_loss/print_every, running_correct/running_total))
            
            
            loss_list_print_every.append(running_loss/print_every)
            accuracy_list_print_every.append(running_correct/running_total)
            
            running_loss = 0
            running_total = 0
            running_correct = 0
            
print("Training complete. Total training time: {:.1f} seconds".format(time.time() - start_time))

# Visualization for the training process
plt.figure(figsize=(12,8))

plt.subplot(2,1,1)
xticks = np.arange(1, len(loss_list_print_every)+1, 1)
plt.plot(accuracy_list_print_every)
plt.xticks(xticks)
plt.xlabel(str(print_every)+" Batches")
plt.ylabel("Average accuracy per batch")

plt.subplot(2,1,2)
xticks = np.arange(1, len(loss_list_print_every)+1, 1)
plt.plot(loss_list_print_every)
plt.xticks(xticks)
plt.xlabel(str(print_every)+" Batches")
plt.ylabel("Average loss per batch")

plt.show()


# Evaluate the model
running_loss = 0
labels_true = np.array([], dtype=int)
labels_pred = np.array([], dtype=int)

model.eval()

with torch.no_grad():  
    for i, (lines, labels) in enumerate(testloader):
        lines, labels = lines.to(device), labels.to(device)
        lines = lines.reshape(-1, sequence_length, model_dim).to(device)
        padding_mask = generate_padding_mask(lines[:, :, 0], padding_idx=0)  
        
        # Forward pass
        output = model(lines,padding_mask=padding_mask)
        loss = criterion(output, labels.float())

        running_loss += loss.item()
        output=output.cpu()
        predicted = (output.numpy() >= 0.5).astype(int) 
        
        labels_true = np.append(labels_true, labels.cpu().numpy())
        labels_pred = np.append(labels_pred, predicted)

        
test_accuracy = np.equal(labels_pred, labels_true).mean()         
        
print("Evaluating network on {} consonants in test set -- Test loss: {:.3f} -- Test accuracy: {:.3f}"
      .format(len(testloader.dataset), running_loss/len(testloader), test_accuracy))





