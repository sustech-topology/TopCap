# Topology-enhanced machine learning for consonant recognition

Here is the source code for TopCap and related models from the [manuscript](https://yifeizhu.github.io/tail.pdf).  The data that support the findings of this study are openly available in [SpeechBox, ALLSSTAR Corpora](https://speechbox.linguistics.northwestern.edu/allsstar), as well as [LJSpeech](https://keithito.com/LJ-Speech-Dataset/), [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1), and [LibriSpeech](https://www.openslr.org/12).  

## TopCap: topological features for machine learning

The following corresponds to overview results in Fig. 2 of the varied shapes of vowels, voiced consonants, and voiceless consonants.  

- Data
- Code

### Primary experiments

The following corresponds to results in Fig. 3 of machine learning topological features.  Sections are organised according to the flowchart in Fig. 3e.  

The folder `supplements` includes supplementary files for this project.  

- The `results` folder contains ROC and AUC from machine learning (Fig. 3a–b) as well as birth time and lifetime of consonants (Fig. 3c–d).  

#### Deriving phonetic data from natural speech

Use [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/index.html) (MFA) to align each speech signal into phonetic segments.  The detailed guidance of MFA can be found on the [Installation Page](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html). The following steps help align each speech into phonetic segments. See [Montreal Forced Aligner Tutorial](https://eleanorchodroff.com/tutorial/montreal-forced-aligner.html) for more explanations.

- Download the acoustic model and dictionary.
  ```
  mfa model download acoustic english_us_arpa
  mfa model download dictionary english_us_arpa
  ```
- Convert sampling rate into 16kHz by [wav_modification](wav_modification.ipynb).  
- Align speeches, the output files are in `.TextGrid` format.
  ```
  mfa align ~/mfa_data/my_corpus english_us_arpa english_us_arpa ~/mfa_data/my_corpus_aligned
  ```

#### Time-delay embedding and persistent homology

TopCap is achieved in [csv_writer_consonant](csv_writer_consonant.ipynb), which captures the most significant topological features within those segmented phonetic time series. The output is a `.csv` file containing the birthtime and lifetime corresponding to the point in the persistent diagram with the longest lifetime.  

#### Machine learning

Matlab (R2022b) [classification learner application](https://www.mathworks.com/help/stats/classificationlearner-app.html), 5-fold cross-validation, set aside 30\% records as test data. Use the following automatic built-in algorithm: Optimizable Tree, Optimizable Discriminant, Efficient Logistic Regression, Optimizable Naive Bayes, Optimizable SVM, Optimizable KNN, Kernel, and Optimizable Ensemble. This is used in section 2.1 of the paper.


### Model comparison

The following corresponds to results in Table 1 of comparing TopCap with 4 state-of-the-art methods on 8 small and 4 large datasets.  

We built other state-of-art models for comparison with TopCap to comprehensively evaluate its performance. The MFCC-GRU classification model is obtained in [MFCC_GRU_classification_model](model%20comparison/MFCC_GRU_classification_model.py), the MFCC-Transformer classification model is obtained in [MFCC_Transformer_classification_model](https://github.com/sustech-topology/TopCap/blob/main/model%20comparison/MFCC_Transformer_classification_model.py), both the STFT-CNN classification model and the STFT-CNN^+ classification model are obtained in [STFT_CNN_classification_model](https://github.com/sustech-topology/TopCap/blob/main/model%20comparison/STFT_CNN_classification_model). This part corresponds to section 2.1.3 of the paper.

The comparison experiments include the LJSpeech, TIMIT, and LibriSpeech repositories, along with four additional corpora from ALLSSTAR that do not appear in our main experiments. The data preprocessing files can be found in the folder [dataset preprocessing](https://github.com/sustech-topology/TopCap/tree/main/dataset%20preprocessing).  


### Feature analysis

The following corresponds to results in Fig. 4 of analysing the features derived from TopCap, STFT, and MFCC.  

This repository corresponding to Figure 4 in Section 2.1.3, Feature Analysis, of the paper contains three Jupyter Notebooks that demonstrate various methods for extracting and analyzing audio features from audio signals. Each notebook focuses on a different technique and provides step-by-step guidance along with visualizations to help you understand the feature extraction process.



## TopNN: topology-enhanced neural networks

The following corresponds to results in Fig. 5 and Table 2 of experiments with topology-enhanced Gated Recurrent Unit networks.  

This repository corresponding to section 2.2 of the paper contains a collection of Python scripts designed for advanced audio signal processing, feature extraction using topological data analysis, and machine learning experiments. The provided scripts cover a range of functionalities from preprocessing audio signals and extracting persistent homology features, to training topology-enhanced neural network classifiers.



## Detection of vibration patterns

The following corresponds to results in Figs. 6 and 7.  

- [Experiments on synthetic data](fre_amp_av.ipynb) 
- [Experiments on real-world data](observation_result_refined.ipynb)



## Supplementary Information

The following corresponds to results in Supplementary Fig. 3 on skip, MP, and persistence execution time.  

- [observation_skip](observation_skip.ipynb) illustrates how skip influences computation time.



## Appendix I: Supplements

- The `consonants_waveforms` folder contains waveforms of pulmonic consonants. Audio for these consonants comes from [Wiki-List of consonants](https://en.wikipedia.org/wiki/List_of_consonants). This gives consonants concrete shapes for readers.

Further discussions of TopCap are involved in 
- [observation_dimension](observation_dimension.ipynb) illustrates how dimension influences time delay embedding and persistent diagrams. Notice that: the newest version does not contain this part.
- [observation_dimension_plot](observation_dimension_plot.ipynb) includes parameters and graph in the discussion section. Notice that: the newest version does not contain this part.



## Appendix II: Drawing the figures

This repository contains a suite of Python scripts for analyzing time series data through time delay embeddings. The tools provided here focus on visualizing the geometry of embedded data, characterizing the dynamical properties via PCA eigenvalues, and quantifying topological features (persistent homology) as a function of embedding parameters. These scripts are particularly useful for studying complex signals (e.g., audio or other sequential data) by revealing hidden structures and dynamical invariants.
