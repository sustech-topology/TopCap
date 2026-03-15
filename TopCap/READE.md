# TopCap: topological features for machine learning

The code for the snapshot results in Fig. 2 of the varied shapes of vowels, voiced consonants, and voiceless consonants can be found in 
[`snapshot.ipynb`](TopCap/snapshot.ipynb).  

### Primary experiments

The following corresponds to results in Fig. 3 of machine learning topological features.  Sections are organised according to the flowchart in Fig. 3e.  

#### Step 1: Deriving phonetic data from natural speech

Use [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/index.html) (MFA) to align each speech signal into phonetic segments through the following steps (cf. Supplementary Sec. 4.1).  Detailed guidelines of MFA can be found on the [Installation](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html) page.  See [this tutorial](https://eleanorchodroff.com/tutorial/montreal-forced-aligner.html) for more explanation.  

- Download the acoustic model and dictionary.  
  ```
  mfa model download acoustic english_us_arpa
  mfa model download dictionary english_us_arpa
  ```
- Convert sampling rate into 16kHz by [here](TopCap/primary/convert.ipynb).  
- Align the speech records.  The output files are in `.TextGrid` format.  
  ```
  mfa align ~/mfa_data/my_corpus english_us_arpa english_us_arpa ~/mfa_data/my_corpus_aligned
  ```

#### Step 2: Time-delay embedding and persistent homology

Topological feature extraction is achieved [here](TopCap/primary/feature.py), which captures the most significant topological features within the segmented phonetic time series.  The output is a `.csv` file containing the birth time and lifetime corresponding to the point in a persistence diagram with the longest lifetime.  

#### Step 3: Machine learning

Use the MATLAB (R2024b) [Classification Learner](https://www.mathworks.com/help/stats/classificationlearner-app.html) application, with 5-fold cross-validation, and set aside 30\% records as test data.  Apply the following built-in algorithms: Optimizable Tree, Optimizable Discriminant, Binary GLM Logistic Regression, Optimizable Naive Bayes, Optimizable SVM, Optimizable KNN, Optimizable Efficient Linear, and Optimizable Ensemble.  The [results](TopCap/primary/results) folder contains ROC and AUC from machine learning (Fig. 3a–b) as well as birth time and lifetime of consonants (Fig. 3c–d).  


### Model comparison

[This repository](TopCap/comparison) corresponds to results in Table 1 of comparing TopCap with 4 state-of-the-art methods on 8 small and 4 large datasets.  

These datasets are from the LJSpeech, TIMIT, and LibriSpeech repositories, along with four additional corpora from ALLSSTAR that do not appear in the primary experiments.  

- [Data preprocessing](TopCap/comparison/preprocessing) 

We build state-of-art comparison models to comprehensively evaluate TopCap's performance: 
- [MFCC–GRU](TopCap/comparison/MFCC–GRU.py) 
- [MFCC–Transformer](TopCap/comparison/MFCC–Transformer.py) 
- [STFT–CNN](TopCap/comparison/STFT–CNN), including both STFT–CNN-8 and STFT–CNN-16 


### Feature analysis

[This repository](TopCap/analysis) corresponds to results in Fig. 4 of analysing the features derived from TopCap, STFT, and MFCC.  It contains three Jupyter notebooks that demonstrate various methods for extracting and analysing features from speech signals.  Each notebook focuses on a particular technique and provides step-by-step guidelines along with visualisation to help the user understand the feature extraction process.  
