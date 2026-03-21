# TopNN: topology-enhanced neural networks

This directory contains code for results in Fig. 5 and Table 2 of experiments with topology-enhanced neural networks.  It contains a collection of Python scripts designed for advanced audio signal processing, feature extraction using topological data analysis, and machine learning experiments.  The provided scripts cover a range of functionalities from preprocessing audio signals and extracting persistent-homology features, to training topology-enhanced neural network classifiers.  

## TopGRU

[`TopGRU.py`](TopGRU.py) realises the model of topology-enhanced gated recurrent unit (GRU) networks as follows.  

- Data preparation & feature extraction 
  - Loads audio files and constructs a file index for audio samples stored in designated "voiced" and "voiceless" folders, with an option to add noise.  
  - Extracts topological features from each audio file using time-delay embedding and persistent homology.  
  - Extracts MFCC features from each audio file using `Librosa`.  

- Custom dataset and DataLoader 

  A custom `PyTorch` dataset (`ConsonantDataset`) and a collate function are defined to handle variable-length MFCC feature sequences along with corresponding topological features and labels.  

- Neural network models 

  Three GRU-based classifiers are implemented for comparison.  
  - `TopGRUClassifier`: Combines GRU-encoded features with topological features.  
  - `ZeroGRUClassifier`: Uses GRU features concatenated with zero vectors in place of topological features.  
  - `GRUClassifier`: A standard GRU classifier, with optional initialisation based on weights from the other models.  
  
- Training & evaluation with cross-validation 
  - The script uses k-fold cross-validation to train and evaluate the three models over multiple (20 by default) experiments.  
  - Training and testing accuracy rates are recorded and averaged across folds.  
  
- Usage considerations 
  - Ensure that the `.csv` file and audio directories are correctly specified.  
  - Install necessary libraries as listed below.  
  - Adjust hyperparameters (e.g., number of epochs, batch size, learning rate) as needed for the user's dataset.  

## Requirements for running the codes

- Python version: Python 3.x 
- Key libraries 
  - Common: `Matplotlib`, `NumPy`, `Pandas`
  - Audio processing: `Librosa`, `SciPy`, `SoundFile`
  - Topological data analysis: `Gudhi`, `Persim`, `Ripser`
  - Machine learning: `Scikit-learn`
  - Deep learning: `PyTorch`
  - Miscellanies: `CSV`, `OS`, `Multiprocessing` modules 

The user can install most of these libraries using pip: 

```bash
pip install gudhi librosa matplotlib numpy pandas persim ripser scikit-learn scipy soundfile torch 
```
