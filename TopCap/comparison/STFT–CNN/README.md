The difference between STFT_CNN and STFT_CNN^+ is that the spectrogram size of the former is 8×8, while that of the latter is 16×16. The corresponding code is as follows. 
  ```
  spectrogram = tf.image.resize(spectrogram, [16, 16]), 
  ```
which appears in both two programs. 

STFT_CNN_classification_model1 utilizes the entire dataset for CNN and generates plots using t-SNE and UMAP. Regarding the plotting, to ensure the data quantity is consistent between methods TopCap and STFT_CNN, a random selection of 1/10 of the dataset was used with method TopCap, since STFT_CNN uses 1/10 of the whole dataset as validation set.

STFT_CNN_classification_model2 utilizes the sampled dataset solely for CNN, catering to the needs of the post-sampling phase of large datasets.
