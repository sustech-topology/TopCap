# Comparative models

## MFCC-based speech classification models

We build 2 state-of-the-art comparative models that leverage mel-frequency cepstral coefficients (MFCC) extracted from speech signals.  Both models are implemented using `PyTorch` and are designed to distinguish between two classes (i.e., voiced vs. voiceless consonants) based on the MFCC features.  The models differ in architecture.  Implementation of each includes a full pipeline from data loading and MFCC-feature extraction to model training and evaluation.  

### Gated recurrent unit (GRU)

[`MFCC–GRU.py`](MFCC–GRU.py) realises this model as follows.  

- Data loading & feature extraction 
  - Loads speech files (`.wav` format) from separate directories for voiced and voiceless consonants.  
  - Extracts 40-dimensional MFCC features (with `n_fft=256`) using `Librosa`.  
  - Transposes and converts features to `PyTorch` tensors.  
  - Generates binary labels (0 for voiced, 1 for voiceless) and constructs a combined dataset.  

- Dataset preparation 
  - Splits the dataset into training and test sets.  
  - Uses a custom `PyTorch` `Dataset` and a collate function that handles variable-length sequences with padding.  

- GRU classifier architecture 
  - A single-layer GRU processes the MFCC sequences using packed padded sequences.  
  - A batch normalisation layer is applied to the final hidden state.  
  - A fully connected layer outputs a single logit for binary classification.  

- Training & evaluation 
  - Trains the model using Binary Cross-Entropy with Logits Loss and the Adam optimiser.  
  - Prints training loss and accuracy periodically.  
  - Evaluates the model on a test set and reports overall test loss and accuracy.  

### Transformer

[`MFCC–Transformer.py`](MFCC–Transformer.py) realises this model as follows.  

- Data loading & feature extraction
  - Similar to the GRU model, it extracts MFCC features from the speech files in the voiced and voiceless directories.  
  - Pads the sequences to a uniform length and converts them into a tensor suitable for Transformer input.  
  - Generates corresponding binary labels.

- Dataset preparation
  - Splits the data into training and test sets.  
  - Uses `PyTorch` `DataLoader` to create batches for training.

- Transformer classifier architecture
  - Incorporates absolute positional encoding to capture sequential information.  
  - Uses a Transformer Encoder with multi-head self-attention and feedforward layers.  
  - Applies global average pooling across the sequence dimension.  
  - Uses a dropout layer and a fully connected output layer for binary classification.

- Training & evaluation
  - Trains the Transformer model using the same loss function and optimiser settings as the GRU model.  
  - Training progress (loss and accuracy) is printed periodically.  
  - Evaluates model performance on the test set and displays training progress graphs.  

## STFT-based speech classification models

We also build a comparative model that uses the spectral features derived by short-time Fourier transform (STFT), with a convolutional neural network (CNN) for implementation.  Based on resizing the spectrograms to 2 different dimensions (one $8\times8$, the other $16\times16$), this comparison is divided into two experiments.  The model is implemented using TensorFlow and aims to distinguish between voiced and voiceless consonants based on the STFT features.  The model implementation [`STFT–CNN.py`](STFT–CNN.py) includes a complete pipeline from data loading and STFT feature extraction to model training and evaluation as follows.  

- Data loading & dataset partitioning
  - Loads speech files (`.wav` format) from a specified directory using `tf.keras.utils.audio_dataset_from_directory`.  
  - Partitions the samples into training, validation, and test sets in an 8:1:1 ratio, followed by batching.  

- Feature extraction (STFT spectrogram generation)
  - Applies STFT to audio waveforms and computes their magnitude to generate spectrograms.  
  - Adds a channel dimension to the spectrograms, making them compatible with convolutional layer input format (batch_size, height, width, channels).  
  - Resizes the spectrograms to a fixed dimension.​  According to the experimental setup, spectrograms are uniformly resized to $8\times8$ pixels for the STFT–CNN-8 experiment.  For the other experiment STFT–CNN-16, they are resized to $16\times16$ pixels.  

- CNN classifier architecture
  - The model adopts a sequential structure.  
  - It consists of three convolutional blocks, each comprising a Conv2D layer (using ReLU activation and same padding) followed by a MaxPooling2D layer, with the number of filters increasing progressively across blocks (64, 128, 256).  
  - After the convolutional layers, a flatten layer and two fully connected (dense) layers are added.  A final sigmoid activation function outputs a single probability value for binary classification (voiced vs. voiceless consonants).  
    
- Training & evaluation
  - The model is compiled using the Adam optimiser and binary cross-entropy loss function.  
  - It is trained on the training set for 10 epochs, with performance monitored on the validation set.  
  - The loss and accuracy metrics recorded during training can be accessed via the history object.  

## Requirements for running the codes

- Python version: Python 3.x 
- Key packages: 
  - `Librosa` 
  - `Matplotlib` 
  - `NumPy` 
  - `PyTorch`
  - `TensorFlow`
  - `Scikit-learn` 

Install the required packages using pip: 

```bash
pip install librosa matplotlib numpy scikit-learn tensorflow torch 
```
