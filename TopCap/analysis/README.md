# Feature analysis

This directory contains code for results in Fig. 4 of analysing and comparing the features derived from TopCap, STFT, and MFCC.  It contains 3 Jupyter notebooks that demonstrate various methods for extracting and analysing features from speech signals.  Each notebook focuses on a particular technique and provides step-by-step guidelines along with visualisation to help the user understand the feature extraction process.  

## Code functionality overview

- [`TopCap.ipynb`](TopCap.ipynb) 
  - Demonstrates the extraction and analysis of PD features from audio signals.
  - Includes steps for loading audio data, processing the signal, and visualizing the extracted PD features.
  - Useful for understanding how power or other domain-specific metrics can be derived from audio.

- [`STFT.ipynb`](STFT.ipynb) 
  - Focuses on extracting features using the Short-Time Fourier Transform (STFT).
  - Converts time-domain audio signals into a time-frequency representation.
  - Provides visualizations to analyze how the frequency content of the audio evolves over time.

- [`MFCC.ipynb`](MFCC.ipynb) 
  - Shows how to compute Mel-Frequency Cepstral Coefficients (MFCCs), a key feature in speech and audio processing.
  - Walks through preprocessing the audio, extracting MFCCs, and visualizing the results.
  - Ideal for projects related to speech recognition and audio classification.

## Features

- **Audio Preprocessing:** Load and preprocess audio data for feature extraction.
- **Diverse Feature Extraction Methods:** 
  - PD features for specific audio analysis tasks.
  - STFT for time-frequency representation.
  - MFCC for capturing the spectral properties of audio.
- **Visualisation:** Each notebook includes plots and charts to help interpret the extracted features.
- **Quantification:** Use logistic classification to quantify the features.

## Requirements for running the codes

- Version: Python 3.x 
- Environment: Jupyter Notebook or JupyterLab
- Install the necessary packages using pip: 

```bash
pip install librosa matplotlib numpy pandas 
```
