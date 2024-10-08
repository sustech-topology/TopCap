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



# Data preprocessing
consonant_labels=[]
mfcc_voiced_list = []
mfcc_voiceless_list = []

# Load the dataset of voiced consonants and extract features using MFCC
folder_path = "D:\\code\\Signal_Recognition\\audio_segment\\audio_segment\\voiced"#"D:\\code\\Signal_Recognition\\audio_segment\\audio_segment\\voiced\\"
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
folder_path = "D:\\code\\Signal_Recognition\\audio_segment\\audio_segment\\voiceless"
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

# Merge two datasets
consonants_mfcc=mfcc_voiced_list+mfcc_voiceless_list

# Generate labels for corresponding phonemes
consonant_labels = torch.tensor(consonant_labels,dtype=torch.float32)
consonant_labels = consonant_labels.unsqueeze(1) 


# Construct GRU classifier
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size) 
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, lengths):
        # Pack padded sequences
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        out = self.bn(hidden[-1])  
        out = self.fc(out)
        return out




# Create a custom ConsonantDataset and split into training and test set
class ConsonantDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]  
    
def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.int64)
    # sequence padding
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    sequences_padded=sequences_padded.to(dtype=torch.float32)
    labels = torch.cat(labels, dim=0)
    return sequences_padded, labels, lengths

# set the parameters
input_size = 40 
hidden_size = 32  
output_size = 1   
num_epochs = 20  
batch_size = 64
learning_rate = 0.0001
num_layers=1

X_train, X_test, y_train, y_test = train_test_split(consonants_mfcc, consonant_labels, test_size=0.2, random_state=10)

train_dataset = ConsonantDataset(X_train, y_train)
test_dataset = ConsonantDataset(X_test, y_test)

# Construct the trainloader and testloader
trainloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn,shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size,collate_fn=collate_fn,shuffle=False)

print("Sequences in training data:", len(trainloader.dataset))
print("Sequences in testing data:", len(testloader.dataset))


# GRU classification
print("Training batches:", len(trainloader))
print("Batch size:", batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Instantiate model
model = GRUClassifier(input_size, hidden_size, num_layers, output_size) 
model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Record the training time
start_time = time.time()

print("Start of traing -- Device: {} -- Epochs: {} -- Batches: {} -- Batch size: {}"
      .format(device, num_epochs, len(trainloader), batch_size))

# Set the training parameters
running_loss = 0
running_total = 0
running_correct = 0
loss_list = [] 
loss_list_print_every = [] 
accuracy_list_print_every =[]
print_every = 100 # Print out the accuracy every {print_every} epochs


# Train the model
model.train()

for epoch in range(num_epochs):
    for i, (sequences, labels,lengths) in enumerate(trainloader):
        sequences, labels,lengths = sequences.to(device), labels.to(device),lengths.to(device)
        optimizer.zero_grad()
        outputs = model(sequences,lengths)
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
        loss_list.append(loss.item())

        # Print average training loss and accuracy 
        if (i+1) % print_every == 0:         
            print("Epoch: {}/{} -- Batches: {}/{} -- Training loss: {:.3f} -- Training accuracy: {:.3f}"
                  .format(epoch+1, num_epochs, i+1, len(trainloader), 
                          running_loss/print_every, running_correct/running_total))
            
            # Store running loss and accuracy
            loss_list_print_every.append(running_loss/print_every)
            accuracy_list_print_every.append(running_correct/running_total)
            
            # Reset running loss and accuracy
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
    for i, (sequences, labels,lengths) in enumerate(testloader):
        sequences, labels,lengths = sequences.to(device), labels.to(device),lengths.to(device)
        
        outputs = model(sequences,lengths)
        outputs = outputs.squeeze(1)
        loss = criterion(outputs, labels.float())

        running_loss += loss.item()
        outputs=outputs.cpu()
        predicted = (outputs.numpy() >= 0.5).astype(int) 
        
        # calculate accuracy 
        labels_true = np.append(labels_true, labels.cpu().numpy())
        labels_pred = np.append(labels_pred, predicted)

        
test_accuracy = np.equal(labels_pred, labels_true).mean()         
        
print("Evaluating network on {} consonants in test set -- Test loss: {:.3f} -- Test accuracy: {:.3f}"
      .format(len(testloader.dataset), running_loss/len(testloader), test_accuracy))