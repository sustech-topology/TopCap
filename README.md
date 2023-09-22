# TDA_Consonant_Recognition
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
Before constructing TopCap, there is a preliminary experiment that measures the performance of topological methods in time series. [fre_amp_av](https://github.com/AnnFeng233/TDA_Consonant_Recognition/blob/main/fre_amp_av.ipynb) helps understand how topological methods distinguish different vibration patterns in time series.

TopCap is achieved in [csv_writer_consonant](https://github.com/AnnFeng233/TDA_Consonant_Recognition/blob/main/csv_writer_consonant.ipynb), which captures the most significant topological features within those segmented phonetic time series. The output is a `.csv` file containing the birthtime and lifetime corresponding to the point in the persistent diagram with the longest lifetime.  

