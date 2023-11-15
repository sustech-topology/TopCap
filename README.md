# TDA Consonant Recognition
Code material for Topological Data Analysis in Consonant Recognition. The data that support the findings of this study are openly available in SpeechBox, ALLSSTAR Corpus, L1-ENG division at [Home Page of SpeechBox](https://speechbox.linguistics.northwestern.edu/#!/home)

## Data Preprocessing
Using [Montreal Forced Aligner (MFA)](https://montreal-forced-aligner.readthedocs.io/en/latest/index.html) to align each speech into phonetic segments. To install MFA, simply try the following command:
```
conda create -n aligner -c conda-forge montreal-forced-aligner
```
The detailed installation of MFA can be found on the [Installation Page](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html). After installation, use `conda activate aligner` to activate the aligner environment. The following steps help align each speech into phonetic segments. See [Montreal Forced Aligner Tutorial](https://eleanorchodroff.com/tutorial/montreal-forced-aligner.html) for more explanations.

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
Before constructing TopCap, there is a preliminary experiment that measures the performance of topological methods in time series. [fre_amp_av](https://github.com/AnnFeng233/TDA_Consonant_Recognition/blob/main/fre_amp_av.ipynb) helps understand how topological methods distinguish different vibration patterns in time series. The results are shown in [observation_result_refined](https://github.com/AnnFeng233/TDA_Consonant_Recognition/blob/main/observation_result_refined.ipynb)

TopCap is achieved in [csv_writer_consonant](https://github.com/AnnFeng233/TDA_Consonant_Recognition/blob/main/csv_writer_consonant.ipynb), which captures the most significant topological features within those segmented phonetic time series. The output is a `.csv` file containing the birthtime and lifetime corresponding to the point in the persistent diagram with the longest lifetime.  

Further discussions of TopCap are involved in 
- [observation_dimension](https://github.com/AnnFeng233/TDA_Consonant_Recognition/blob/main/observation_dimension.ipynb) illustrates how dimension influences time delay embedding and persistent diagrams.
- [observation_dimension_plot](https://github.com/AnnFeng233/TDA_Consonant_Recognition/blob/main/observation_dimension_plot.ipynb) includes parameters and graph in the discussion section.
- [observation_skip](https://github.com/AnnFeng233/TDA_Consonant_Recognition/blob/main/observation_skip.ipynb) illustrates how skip influences computation time.

## Machine Learning for Topological Features
Matlab (R2022b) [classification learner application](https://www.mathworks.com/help/stats/classificationlearner-app.html), 5-fold cross-validation, set aside 30\% records as test data. Use the following automatic built-in algorithm: Optimizable Tree, Optimizable Discriminant, Efficient Logistic Regression, Optimizable Naive Bayes, Optimizable SVM, Optimizable KNN, Kernel, Optimizable Ensemble, and Optimizable Neural Network.

## Supplements
The folder `supplements` includes supplementary files for this project. 

- The `results` folder contains ROC, and AUC for machine learning, as well birthtime, lifetime of consonants.
- The `consonants_waveforms` folder contains waveforms of pulmonic consonants. Audio for these consonants comes from [Wiki-List of consonants](https://en.wikipedia.org/wiki/List_of_consonants). This gives consonants concrete shapes for readers.


