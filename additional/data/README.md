# Sample data

This directory contains sample data associated with the code for the experiments.  The provided datasets are a demo version of the original data used to obtain the results in the [manuscript](https://yifeizhu.github.io/tail.pdf).  We have only uploaded a portion of the complete data due to its large size.  This sample data is intended to help users understand the data structure and format.  For access to the full datasets, please refer to the Data availability section of the manuscript and contact the corresponding author [Yifei Zhu](mailto:zhuyf@sustech.edu.cn).  

- [`ALL_049_F_ENG_ENG_HT1.TextGrid`](ALL_049_F_ENG_ENG_HT1.TextGrid) and [`ALL_049_F_ENG_ENG_HT1.wav`](ALL_049_F_ENG_ENG_HT1.wav) are data needed in the following codes: 
  - [`snapshot.ipynb`](/TopCap/snapshot.ipynb) for results with [TopCap](/TopCap) in Fig. 2 
  - [`Supp-Disc.ipynb.ipynb`](/additional/code/Supp-Disc.ipynb.ipynb) of the [`additional/code`](/additional/code) directory for results in Supplementary Fig. 3 and Supplementary Table 1 
  - [`observation_dimension.ipynb`](/additional/code/observation_dimension.ipynb) of the [`additional/code`](/additional/code) directory 

- The [ALL_ENG_ENG_HT1](`ALL_ENG_ENG_HT1`) directory contains datasets from ALLSSTAR HT1 needed in [`convert.ipynb`](/TopCap/primary/convert.ipynb) in Step 1 of deriving phonetic data from natural speech with [TopCap](/TopCap/primary).  

- The [input](input) and [output](output) directories contain phonetic data before and after applying the Montreal Forced Aligner, respectively.  They appear in the following codes: 
  - [`convert.ipynb`](/TopCap/primary/convert.ipynb) in Step 1 of deriving phonetic data from natural speech with [TopCap](/TopCap/primary) 
  - [`real.ipynb`](/Vibration/real.ipynb) for the experiments on [detection of vibration patterns](/Vibration) with real-world data, containing specifically the vowel [ɑ] for results in Fig. 7 
  - [`observation_dimension_plot.ipynb`](/additional/code/observation_dimension_plot.ipynb) of the [`additional/code`](/additional/code) directory 
  - [`plot.ipynb`](/additional/image/plot.ipynb) of the [`additional/image`](/additional/image) directory 

- [`phone`](phone) records the voiced consonant [ŋ] which supports results in Fig. 8 and Supplementary Fig. 2.  

- [`Sample_TIMIT.csv`](Sample_TIMIT.csv) gives an example of input for [`SVM.py`](/TopCap/primary/SVM.py) in Step 3 of machine learning step with [TopCap](/TopCap/primary).  
