# TopCap: topological features for machine learning

The code for the snapshot results in Fig. 2 of the varied shapes of vowels, voiced consonants, and voiceless consonants can be found in [`snapshot.ipynb`](TopCap/snapshot.ipynb) (cf. the primary experiments below).  

## Primary experiments

The [`primary`](primary) directory contains code for results in Fig. 3 of machine learning topological features.  

## Model comparison

The [`comparison`](comparison) directory contains code for results in Table 1 of comparing TopCap with 4 state-of-the-art methods on 8 small and 4 large datasets.  These 
datasets are from the LJSpeech, TIMIT, and LibriSpeech repositories, along with 4 additional corpora from ALLSSTAR that do not appear in the primary experiments.  

## Feature analysis

The [`analysis`](analysis) directory contains code for results in Fig. 4 of analysing the features derived from TopCap, STFT, and MFCC.  It contains 3 Jupyter notebooks that demonstrate various methods for extracting and analysing features from speech signals.  Each notebook focuses on a particular technique and provides step-by-step guidelines along with visualisation to help the user understand the feature extraction process.  
