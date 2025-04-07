# MFCC-based Audio Classification Models

This repository provides two distinct audio classification models that leverage Mel-Frequency Cepstral Coefficients (MFCC) extracted from audio files. Both models are implemented using PyTorch and are designed to distinguish between two classes (e.g., voiced vs. voiceless consonants) based on MFCC features. The models differ in architecture:

- **GRU-based Model** (`MFCC_GRU_classification_model.py`)
- **Transformer-based Model** (`MFCC_Transformer_classification_model.py`)

Both implementations include full pipelines from data loading and MFCC feature extraction to model training and evaluation.

---

## File Overview

### 1. `MFCC_GRU_classification_model.py`  
This script implements a GRU-based classifier for MFCC features.

- **Data Loading & Feature Extraction:**  
  - Loads audio files (WAV format) from separate directories for voiced and voiceless sounds.
  - Extracts 40-dimensional MFCC features (with `n_fft=256`) using Librosa.
  - Transposes and converts features to PyTorch tensors.
  - Generates binary labels (0 for voiced, 1 for voiceless) and constructs a combined dataset.

- **Dataset Preparation:**  
  - Splits the dataset into training and testing sets.
  - Uses a custom PyTorch `Dataset` and a collate function that handles variable-length sequences with padding.

- **GRU Classifier Architecture:**  
  - A single-layer GRU processes the MFCC sequences using packed padded sequences.
  - A Batch Normalization layer is applied to the final hidden state.
  - A fully connected layer outputs a single logit for binary classification.

- **Training & Evaluation:**  
  - Trains the model using Binary Cross Entropy with Logits Loss and the Adam optimizer.
  - Prints training loss and accuracy periodically.
  - Evaluates the model on a test set and reports overall test loss and accuracy.

*(See [57] for full implementation details.)*

---

### 2. `MFCC_Transformer_classification_model.py`  
This script implements a Transformer-based classifier for MFCC features.

- **Data Loading & Feature Extraction:**  
  - Similar to the GRU model, it extracts MFCC features from the audio files in the voiced and voiceless directories.
  - Pads the sequences to a uniform length and converts them into a tensor suitable for Transformer input.
  - Generates corresponding binary labels.

- **Dataset Preparation:**  
  - Splits the data into training and testing sets.
  - Uses PyTorch DataLoader to create batches for training.

- **Transformer Classifier Architecture:**  
  - Incorporates absolute positional encoding to capture sequential information.
  - Uses a Transformer Encoder with multi-head self-attention and feedforward layers.
  - Applies global average pooling across the sequence dimension.
  - Uses a dropout layer and a fully connected output layer for binary classification.

- **Training & Evaluation:**  
  - Trains the Transformer model using the same loss function and optimizer settings.
  - Training progress (loss and accuracy) is printed periodically.
  - Evaluates model performance on the test set and displays training progress graphs.

*(See [58] for full implementation details.)*

---

## Requirements

- **Python Version:** Python 3.x  
- **Key Libraries:**  
  - PyTorch  
  - Librosa  
  - NumPy  
  - Matplotlib  
  - Scikit-learn  

Install the required packages using pip:

```bash
pip install torch librosa numpy matplotlib scikit-learn
