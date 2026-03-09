# Topology-enhanced machine learning for consonant recognition

Here is the source code for TopCap and related models from the [manuscript](https://yifeizhu.github.io/tail.pdf).  The [data](data) that support the findings of this study are openly available in [SpeechBox, ALLSSTAR Corpora](https://speechbox.linguistics.northwestern.edu/allsstar), as well as [LJSpeech](https://keithito.com/LJ-Speech-Dataset/), [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1), and [LibriSpeech](https://www.openslr.org/12).  

## TopCap: topological features for machine learning

The code for the snapshot results in Fig. 2 of the varied shapes of vowels, voiced consonants, and voiceless consonants can be found [here](working_flow.ipynb).  

### Primary experiments

The following corresponds to results in Fig. 3 of machine learning topological features.  Sections are organised according to the flowchart in Fig. 3e.  

#### Step 1: Deriving phonetic data from natural speech

Use [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/index.html) (MFA) to align each speech signal into phonetic segments through the following steps (cf. Supplementary Sec. 4.1).  Detailed guidelines of MFA can be found on the [Installation](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html) page.  See [this tutorial](https://eleanorchodroff.com/tutorial/montreal-forced-aligner.html) for more explanation.  

- Download the acoustic model and dictionary.  
  ```
  mfa model download acoustic english_us_arpa
  mfa model download dictionary english_us_arpa
  ```
- Convert sampling rate into 16kHz by [wav_modification](wav_modification.ipynb).  
- Align the speech records.  The output files are in `.TextGrid` format.  
  ```
  mfa align ~/mfa_data/my_corpus english_us_arpa english_us_arpa ~/mfa_data/my_corpus_aligned
  ```

#### Step 2: Time-delay embedding and persistent homology

Topological feature extraction is achieved in [csv_writer_consonant](csv_writer_consonant.ipynb), which captures the most significant topological features within the segmented phonetic time series.  The output is a `.csv` file containing the birth time and lifetime corresponding to the point in a persistence diagram with the longest lifetime.  

#### Step 3: Machine learning

Use the MATLAB (R2024b) [Classification Learner](https://www.mathworks.com/help/stats/classificationlearner-app.html) application, with 5-fold cross-validation, and set aside 30\% records as test data.  Apply the following built-in algorithms: Optimizable Tree, Optimizable Discriminant, Binary GLM Logistic Regression, Optimizable Naive Bayes, Optimizable SVM, Optimizable KNN, Optimizable Efficient Linear, and Optimizable Ensemble.  The [results](/supplements/results) folder contains ROC and AUC from machine learning (Fig. 3a–b) as well as birth time and lifetime of consonants (Fig. 3c–d).  


### Model comparison

[This repository](model%20comparison) corresponds to results in Table 1 of comparing TopCap with 4 state-of-the-art methods on 8 small and 4 large datasets.  

These datasets are from the LJSpeech, TIMIT, and LibriSpeech repositories, along with four additional corpora from ALLSSTAR that do not appear in the primary experiments.  

- [Data preprocessing](dataset%20preprocessing) 

We build state-of-art comparison models to comprehensively evaluate TopCap's performance: 
- [MFCC–GRU_classification_model](model%20comparison/MFCC_GRU_classification_model.py) 
- [MFCC–Transformer_classification_model](model%20comparison/MFCC_Transformer_classification_model.py) 
- [STFT–CNN_classification_models](model%20comparison/STFT_CNN_classification_model), including both STFT–CNN-8 and STFT–CNN-16 


### Feature analysis

[This repository](feature_analysis) corresponds to results in Fig. 4 of analysing the features derived from TopCap, STFT, and MFCC.  It contains three Jupyter notebooks that demonstrate various methods for extracting and analysing features from speech signals.  Each notebook focuses on a particular technique and provides step-by-step guidelines along with visualisation to help the user understand the feature extraction process.  



## TopNN: topology-enhanced neural networks

[This repository](Topology-enhanced%20neural%20network) corresponds to results in Fig. 5 and Table 2 of experiments with topology-enhanced Gated Recurrent Unit networks.  It contains a collection of Python scripts designed for advanced audio signal processing, feature extraction using topological data analysis, and machine learning experiments.  The provided scripts cover a range of functionalities from preprocessing audio signals and extracting persistent homology features, to training topology-enhanced neural network classifiers.  



## Detection of vibration patterns

The following codes correspond to results in Figs. 6 and 7, respectively.  

- [Experiments on synthetic data](fre_amp_av.ipynb) 
- [Experiments on real-world data](observation_result_refined.ipynb)



## Additional codes and materials

- [Code](Drawing/Graph_MP_tau_dim_var.py) for results in Fig. 8c supporting discussion on parameter selection
- [Code](Drawing/newChar_pca.py) for results in Fig. 8d supporting discussion on additional geometric features
- [Code](Drawing/16_figs_S2.py) for visualisation in Supplementary Fig. 2 of embedded point clouds via standard and cyclic TDE
- [Code](observation_skip.ipynb) for results in Supplementary Fig. 3 supporting discussion on how skip affects MP and computation time
- Code for results in Supplementary Table 1 supporting discussion on dependency of MP on multiple parameters for TDE

### Consonant waveforms

[This repository](supplements/consonants_waveforms) contains waveforms of pulmonic consonants.  Audio signals for these consonants are from [here](https://en.wikipedia.org/wiki/List_of_consonants).  


### Figure generation



<!--
Further discussions of TopCap are involved in 
- [observation_dimension](observation_dimension.ipynb) illustrates how dimension influences time delay embedding and persistent diagrams. Notice that: the newest version does not contain this part.
- [observation_dimension_plot](observation_dimension_plot.ipynb) includes parameters and graph in the discussion section. Notice that: the newest version does not contain this part.
-->
