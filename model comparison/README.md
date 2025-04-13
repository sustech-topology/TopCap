# MFCC-based Audio Classification Models

This repository provides two distinct audio classification models that leverage Mel-Frequency Cepstral Coefficients (MFCC) extracted from audio files. Both models are implemented using PyTorch and are designed to distinguish between two classes (e.g., voiced vs. voiceless consonants) based on MFCC features. The models differ in architecture:

- **GRU-based Model** (`MFCC_GRU_classification_model.py`)
- **Transformer-based Model** (`MFCC_Transformer_classification_model.py`)

Both implementations include full pipelines from data loading and MFCC feature extraction to model training and evaluation.

---

## File Overview

### 1. `MFCC_GRU_classification_model.py`  
This script corresponding to scetion2.1.2 of the paper implements a GRU-based classifier for MFCC features.

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


---

### 2. `MFCC_Transformer_classification_model.py`  
This script corresponding to scetion2.1.2 of the paper implements a Transformer-based classifier for MFCC features.

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


---

### 3. `segment_PD.py`
This script performs advanced audio signal processing and topological feature extraction. Its key functionalities include:

- **Signal Normalization and Noise Addition:**  
  Functions are provided to normalize audio signals to the range [-1, 1] and add noise (Gaussian, uniform, or impulse) with controllable signal-to-noise ratios (SNR) using reproducible random seeds.

- **Frequency Estimation:**  
  Multiple methods are implemented to estimate the principal frequency of the audio signal, including Fourier Transform-based peak detection and autocorrelation-based methods.

- **Time Delay Embedding & Persistent Homology:**  
  The script uses time delay embedding (via the Gudhi library) to transform one-dimensional audio signals into multi-dimensional point clouds. Persistent homology is then computed using the Ripser library to extract topological features (e.g., persistence intervals).

- **Parallel Processing:**  
  Leveraging Python's multiprocessing, the script processes multiple audio files (from designated "voiced" and "voiceless" folders) concurrently. It trims and preprocesses each audio sample, computes its topological features, and writes the results (e.g., persistence values, corresponding indices, and data category) into a CSV file.

- **Usage Considerations:**  
  - Adjust file paths for the audio directories and CSV output as needed.  
  - Ensure all required libraries (e.g., NumPy, SciPy, SoundFile, Gudhi, Ripser, Persim) are installed.

---


### 4. `Gauss_SVM_acc.py`
This script evaluates the performance of a Gaussian (RBF) SVM classifier on a dataset using stratified 5-fold cross-validation. Key aspects include:

- **Data Loading and Preprocessing:**  
  - Reads a CSV file (e.g., `Sample_TIMIT_noise0_arr.csv`) where the third and fourth columns represent features and the fifth column represents binary labels.
  - Ensures that the dataset is suitable for binary classification.

- **Pipeline Construction and Cross-Validation:**  
  - Constructs a scikit-learn pipeline that standardizes the features using `StandardScaler` and then applies an SVM classifier with an RBF kernel.
  - Performs stratified 5-fold cross-validation in parallel (using all available CPU cores by default) to assess model performance.

- **Output Metrics:**  
  - Prints individual fold accuracies as well as the mean accuracy and standard deviation across folds.

- **Usage Considerations:**  
  - Verify that the CSV file is in the expected format.  
  - Install necessary libraries such as scikit-learn, Pandas, and NumPy.
  - Adjust the `n_jobs` parameter if needed to optimize parallel processing.

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
