# Model comparison

To comprehensively evaluate TopCap's performance, we build multiple state-of-the-art comparative models and benchmark them against a wide range of datasets.  This directory contains code for results in Table 1 of comparing TopCap with 4 state-of-the-art methods on 8 small and 4 large datasets.  These datasets are from the LJSpeech, TIMIT, and LibriSpeech repositories, along with 4 additional corpora from ALLSSTAR that do not appear in the [primary](/TopCap/primary) experiments.  

## Data preprocessing

The [`data`](data) directory contains code for preprocessing data prior to running [TopCap](/TopCap/primary) and the comparative models below.  

## Comparative models

The [`model`](model) directory contains code for the comparative models MFCC–GRU, MFCC–Transformer, STFT–CNN-8, and STFT–CNN-16 appearing in Table 1.  
