# Primary experiments

We proceed according to the flowchart in Fig. 3e.  The [`results`](results) directory contains ROC and AUC from machine learning (Fig. 3a–b) as well as birth time and lifetime of consonants (Fig. 3c–d). 

## Step 1: Deriving phonetic data from natural speech

Use [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/index.html) (MFA) to align each speech signal into phonetic segments through the following steps (cf. Supplementary Sec. 4.1).  Detailed guidelines of MFA can be found on the [Installation](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html) page.  See [this tutorial](https://eleanorchodroff.com/tutorial/montreal-forced-aligner.html) for more explanation.  

- Download the acoustic model and dictionary.  
  ```
  mfa model download acoustic english_us_arpa
  mfa model download dictionary english_us_arpa
  ```
- Convert sampling rate into 16kHz by [`convert.ipynb`](convert.ipynb).  
- Align the speech records.  The output files are in `.TextGrid` format.  
  ```
  mfa align ~/mfa_data/my_corpus english_us_arpa english_us_arpa ~/mfa_data/my_corpus_aligned
  ```

## Step 2: Time-delay embedding and persistent homology

Topological feature extraction is achieved [`feature.py`](feature.py), which captures the most significant topological features within the segmented phonetic time series.  The output is a `.csv` file containing the birth time and lifetime corresponding to the point in a persistence diagram with the longest lifetime.  

## Step 3: Machine learning

Use the MATLAB (R2024b) [Classification Learner](https://www.mathworks.com/help/stats/classificationlearner-app.html) application, with 5-fold cross-validation, and set aside 30\% records as test data.  Apply the following built-in algorithms: Optimizable Tree, Optimizable Discriminant, Binary GLM Logistic Regression, Optimizable Naive Bayes, Optimizable SVM, Optimizable KNN, Optimizable Efficient Linear, and Optimizable Ensemble.  

To be more specific in this step, we prepare a corresponding code [`SVM.py`](SVM.py) for the SVM algorithm and use it with TopCap in the [model comparison experiments](/TopCap/comparison).  This script evaluates the performance of a Gaussian Radial Basis Function (RBF) SVM classifier on a dataset using stratified 5-fold cross-validation.  Key aspects are as follows.  

- Data loading and preprocessing
  - Reads a `.csv` file (e.g., [`Sample_TIMIT.csv`](Sample_TIMIT.csv)) where the third and fourth columns represent features and the fifth column represents binary labels.  
  - Ensures that the dataset is suitable for binary classification.  

- Pipeline construction and cross-validation
  - Constructs a scikit-learn pipeline that standardises the features using StandardScaler and then applies an SVM classifier with an RBF kernel.  
  - Performs stratified 5-fold cross-validation in parallel (using all available CPU cores by default) to assess model performance.  

- Output metrics
  - Prints individual fold accuracies as well as the mean accuracy and standard deviation across folds.  

- Usage considerations
  - Verify that the `.csv` file is in the expected format.  
  - Install necessary libraries such as scikit-learn, Pandas, and NumPy.  
  - Adjust the `n_jobs` parameter if needed to optimise parallel processing.  
