# Model comparison

To comprehensively evaluate TopCap's performance, we build multiple state-of-the-art comparison models and benchmark them against a wide range datasets.  

## Data preprocessing

The [`preprocessing`](preprocessing) directory contains code for preprocessing data prior to running [`TopCap`](/TopCap) and the comparison models below.  

## MFCC-based speech classification models

We build 2 state-of-the-art comparison models that leverage mel-frequency cepstral coefficients (MFCC) extracted from speech signals.  Both models are implemented using PyTorch and are designed to distinguish between two classes (i.e., voiced vs. voiceless consonants) based on MFCC features.  The models differ in architecture.  Implementation of each includes a full pipeline from data loading and MFCC-feature extraction to model training and evaluation.  

### Gated recurrent unit (GRU)

[`MFCC–GRU.py`](MFCC–GRU.py) realises this model as follows.  

- Data loading & feature extraction
  - Loads speech files (.wav format) from separate directories for voiced and voiceless consonants.  
  - Extracts 40-dimensional MFCC features (with `n_fft=256`) using Librosa.  
  - Transposes and converts features to PyTorch tensors.  
  - Generates binary labels (0 for voiced, 1 for voiceless) and constructs a combined dataset.

- Dataset preparation
  - Splits the dataset into training and test sets.  
  - Uses a custom PyTorch `Dataset` and a collate function that handles variable-length sequences with padding.

- GRU classifier architecture
  - A single-layer GRU processes the MFCC sequences using packed padded sequences.  
  - A batch normalization layer is applied to the final hidden state.  
  - A fully connected layer outputs a single logit for binary classification.

- Training & evaluation
  - Trains the model using Binary Cross Entropy with Logits Loss and the Adam optimiser.  
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
  - Uses PyTorch DataLoader to create batches for training.

- Transformer classifier architecture
  - Incorporates absolute positional encoding to capture sequential information.  
  - Uses a Transformer Encoder with multi-head self-attention and feedforward layers.  
  - Applies global average pooling across the sequence dimension.  
  - Uses a dropout layer and a fully connected output layer for binary classification.

- Training & evaluation
  - Trains the Transformer model using the same loss function and optimiser settings as the GRU model.  
  - Training progress (loss and accuracy) is printed periodically.  
  - Evaluates model performance on the test set and displays training progress graphs.  

Running the above codes requires: 

- Python version: Python 3.x 
- Key libraries: 
  - Librosa 
  - Matplotlib 
  - NumPy 
  - PyTorch 
  - Scikit-learn 

Install the required packages using pip: 

```bash
pip install torch librosa numpy matplotlib scikit-learn
```

## STFT-based speech classification models

The [`STFT–CNN`](STFT–CNN) directory contains 2 convolutional neural network (CNN) comparison models based on short-time Fourier transform (STFT) features.  
