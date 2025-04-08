# topological-enhanced neural neteork

This repository contains a collection of Python scripts designed for advanced audio signal processing, feature extraction using topological data analysis, and machine learning experiments. The provided scripts cover a range of functionalities from preprocessing audio signals and extracting persistent homology features, to training topology-enhanced neural network classifiers and evaluating a Gaussian SVM model.

---

## File Overview

### 1. `Multiple_experiments_on_Topology_enhanced_neural_networks.py`
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
