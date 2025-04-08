# Topological enhanced neural neteork

This repository contains a collection of Python scripts designed for advanced audio signal processing, feature extraction using topological data analysis, and machine learning experiments. The provided scripts cover a range of functionalities from preprocessing audio signals and extracting persistent homology features, to training topology-enhanced neural network classifiers and evaluating a Gaussian SVM model.

---

## File Overview

### 1. `segment_PD.py`
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

### 2. `Multiple_experiments_on_Topology_enhanced_neural_networks.py`
This script is focused on performing multiple experiments for classifying audio consonants using topology-enhanced neural networks. Its main functionalities include:

- **Data Preparation and Feature Extraction:**  
  - Reads a CSV file (e.g., `Sample_TIMIT_noise5_arr.csv`) containing pre-extracted topological features, MFCC features, and labels from the TIMIT dataset.  
  - Constructs a file index for audio samples stored in designated "voiced" and "voiceless" folders.  
  - Extracts MFCC features from each audio file using Librosa, with an option to add noise.

- **Custom Dataset and DataLoader:**  
  A custom PyTorch `Dataset` (`ConsonantDataset`) and a collate function are defined to handle variable-length MFCC feature sequences along with corresponding topological features and labels.

- **Neural Network Models:**  
  Three GRU-based classifiers are implemented:
  - **TopGRUClassifier:** Combines GRU-encoded features with topological features.
  - **ZeroGRUClassifier:** Uses GRU features concatenated with zero vectors in place of topological features.
  - **GRUClassifier:** A standard GRU classifier, with optional initialization based on weights from the other models.
  
- **Training and Evaluation with Cross-Validation:**  
  - The script uses k-fold cross-validation to train and evaluate the three models over multiple experiments (default 20 experiments).  
  - Training and testing accuracies are recorded and averaged across folds.
  
- **Usage Considerations:**  
  - Ensure that the CSV file and audio directories are correctly specified.  
  - Install necessary libraries such as PyTorch, scikit-learn, Librosa, NumPy, Pandas, and Matplotlib.  
  - Adjust hyperparameters (e.g., number of epochs, batch size, learning rate) as needed for your dataset.

---

### 3. `Gauss_SVM_acc.py`
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
  - **Common:** NumPy, Pandas, Matplotlib
  - **Audio Processing:** SciPy, SoundFile, Librosa
  - **Topological Data Analysis:** Gudhi, Ripser, Persim
  - **Machine Learning:** scikit-learn
  - **Deep Learning:** PyTorch
  - **Others:** CSV, OS, Multiprocessing modules

You can install most of these packages using pip. For example:

```bash
pip install numpy pandas matplotlib scipy soundfile librosa gudhi ripser persim scikit-learn torch
