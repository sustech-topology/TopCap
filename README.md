# Topology-enhanced Machine Learning for Consonant Recognition
Code material for Topological Data Analysis in Consonant Recognition. The data that support the findings of this study are openly available in SpeechBox, ALLSSTAR Corpus, L1-ENG division at [Home Page of SpeechBox](https://speechbox.linguistics.northwestern.edu/#!/home)

## Data Preprocessing
Using [Montreal Forced Aligner (MFA)](https://montreal-forced-aligner.readthedocs.io/en/latest/index.html) to align each speech into phonetic segments. The detailed guidance of MFA can be found on the [Installation Page](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html). The following steps help align each speech into phonetic segments. See [Montreal Forced Aligner Tutorial](https://eleanorchodroff.com/tutorial/montreal-forced-aligner.html) for more explanations.

- Download the acoustic model and dictionary.
  ```
  mfa model download acoustic english_us_arpa
  mfa model download dictionary english_us_arpa
  ```
- Convert sampling rate into 16kHz by [wav_modification](https://github.com/AnnFeng233/TDA_Consonant_Recognition/blob/main/wav_modification.ipynb).  
- Align speeches, the output files are in `.TextGrid` format.
  ```
  mfa align ~/mfa_data/my_corpus english_us_arpa english_us_arpa ~/mfa_data/my_corpus_aligned
  ```

## TopCap Construction
Before constructing TopCap, there is a preliminary experiment that measures the performance of topological methods in time series. [fre_amp_av] (https://github.com/AnnFeng233/TDA_Consonant_Recognition/blob/main/fre_amp_av.ipynb) helps understand how topological methods distinguish different vibration patterns in time series. The results are shown in [observation_result_refined].  (https://github.com/AnnFeng233/TDA_Consonant_Recognition/blob/main/observation_result_refined.ipynb). This part corrspondes to section 2.3 of the paper. 

TopCap is achieved in [csv_writer_consonant](https://github.com/AnnFeng233/TDA_Consonant_Recognition/blob/main/csv_writer_consonant.ipynb), which captures the most significant topological features within those segmented phonetic time series. The output is a `.csv` file containing the birthtime and lifetime corresponding to the point in the persistent diagram with the longest lifetime. This part corresponds to section 2.1 of the paper. 

Further discussions of TopCap are involved in 
- [observation_dimension](https://github.com/AnnFeng233/TDA_Consonant_Recognition/blob/main/observation_dimension.ipynb) illustrates how dimension influences time delay embedding and persistent diagrams. Notice that: the newest version does not contain this part.
- [observation_dimension_plot](https://github.com/AnnFeng233/TDA_Consonant_Recognition/blob/main/observation_dimension_plot.ipynb) includes parameters and graph in the discussion section. Notice that: the newest version does not contain this part.
- [observation_skip](https://github.com/AnnFeng233/TDA_Consonant_Recognition/blob/main/observation_skip.ipynb) illustrates how skip influences computation time.

## Machine Learning for Topological Features
Matlab (R2022b) [classification learner application](https://www.mathworks.com/help/stats/classificationlearner-app.html), 5-fold cross-validation, set aside 30\% records as test data. Use the following automatic built-in algorithm: Optimizable Tree, Optimizable Discriminant, Efficient Logistic Regression, Optimizable Naive Bayes, Optimizable SVM, Optimizable KNN, Kernel, and Optimizable Ensemble. This is used in section 2.1 of the paper.

## Model Comparison
We built other state-of-art models for comparison with TopCap to comprehensively evaluate its performance. The MFCC-GRU classification model is obtained in [MFCC_GRU_classification_model](https://github.com/sustech-topology/TopCap/blob/main/model%20comparison/MFCC_GRU_classification_model.py), the MFCC-Transformer classification model is obtained in [MFCC_Transformer_classification_model](https://github.com/sustech-topology/TopCap/blob/main/model%20comparison/MFCC_Transformer_classification_model.py), both the STFT-CNN classification model and the STFT-CNN^+ classification model are obtained in [STFT_CNN_classification_model](https://github.com/sustech-topology/TopCap/blob/main/model%20comparison/STFT_CNN_classification_model). This part corresponds to section 2.1.3 of the paper.

## Data Preprocessing of other data sets
The comparison experiments include the LJSpeech, TIMIT, and LibriSpeech repositories, along with four additional corpora from ALLSSTAR that do not appear in our main experiments. The data preprocessing files can be found in the folder [dataset preprocessing](https://github.com/sustech-topology/TopCap/tree/main/dataset%20preprocessing). 


## Supplements
The folder `supplements` includes supplementary files for this project. This part corresponds to section 2.1.3 of the paper.

- The `results` folder contains ROC, and AUC for machine learning, as well birthtime, lifetime of consonants.
- The `consonants_waveforms` folder contains waveforms of pulmonic consonants. Audio for these consonants comes from [Wiki-List of consonants](https://en.wikipedia.org/wiki/List_of_consonants). This gives consonants concrete shapes for readers.

# Feature analysis

This repository corresponding to Figure 4 in Section 2.1.3, Feature Analysis, of the paper contains three Jupyter Notebooks that demonstrate various methods for extracting and analyzing audio features from audio signals. Each notebook focuses on a different technique and provides step-by-step guidance along with visualizations to help you understand the feature extraction process.

# Topology-enhanced neural network

This repository corresponding to section 2.2 of the paper contains a collection of Python scripts designed for advanced audio signal processing, feature extraction using topological data analysis, and machine learning experiments. The provided scripts cover a range of functionalities from preprocessing audio signals and extracting persistent homology features, to training topology-enhanced neural network classifiers.

# Detection of vibration patterns

Code for Sec. 2.3 is located in the main folder as follows.  
- Experiments on synthetic data: fre_amp_av.ipynb
- Experiments on real-world data: observation_result_refined.ipynb

# Drawing

This repository contains a suite of Python scripts for analyzing time series data through time delay embeddings. The tools provided here focus on visualizing the geometry of embedded data, characterizing the dynamical properties via PCA eigenvalues, and quantifying topological features (persistent homology) as a function of embedding parameters. These scripts are particularly useful for studying complex signals (e.g., audio or other sequential data) by revealing hidden structures and dynamical invariants.



