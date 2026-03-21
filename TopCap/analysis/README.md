# Feature analysis

This directory contains code for results in Fig. 4 of analysing and comparing the features derived from TopCap, STFT, and MFCC.  It contains three Jupyter notebooks that demonstrate various methods for extracting and analysing features from speech signals.  Each notebook focuses on a particular technique and provides step-by-step guidelines along with visualisation to help the user understand the feature extraction process.  

## Code functionality

- [`TopCap.ipynb`](TopCap.ipynb) 
  - Demonstrates the extraction and analysis of persistence diagram features from audio signals.  
  - Includes steps for loading audio data, processing the signal, and visualising the extracted PD features.  
  - Useful for understanding how power or other domain-specific metrics can be derived from audio.  

- [`STFT.ipynb`](STFT.ipynb) 
  - Focuses on extracting features using the short-time Fourier transform (STFT).  
  - Converts time-domain audio signals into a time-frequency representation.  
  - Provides visualisation to analyse how the frequency fingerprint of the audio evolves over time.  

- [`MFCC.ipynb`](MFCC.ipynb) 
  - Shows how to compute mel-frequency cepstral coefficients (MFCCs), a key feature in speech and audio processing.  
  - Walks through preprocessing the audio, extracting MFCCs, and visualising the results.  
  - Ideal for projects related to speech recognition and audio classification.  

## Code features

- Audio preprocessing: Load and preprocess audio data for feature extraction.  

- Diverse feature extraction methods: 
  - Topological PD features for specific audio analysis tasks 
  - STFT for time-frequency representation 
  - MFCC for capturing the spectral properties of audio according to human auditory perception 

- Visualisation: Each notebook includes plots and charts to help interpret the extracted features.  

- Quantification: Use logistic classification to quantify the features.  

## Requirements for running the codes

- Version: Python 3.x

- Environment: Jupyter Notebook or JupyterLab

- Install the necessary libraries using pip: 
```bash
pip install librosa matplotlib numpy pandas 
```
