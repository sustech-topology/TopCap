# STFT-based speech classification models

The difference between the STFT–CNN-8 and STFT–CNN-16 models is that the spectrogram size of the former is 8×8 while that of the latter is 16×16.  The corresponding code goes as follows: 
  ```
  spectrogram = tf.image.resize(spectrogram, [16, 16]), 
  ```

[`STFT–CNN-8.py`](STFT–CNN-8.py) uses the entire dataset for CNN and generates plots using t-SNE and UMAP.  Regarding the plotting, to ensure the data quantity is consistent between the TopCap and STFT–CNN models, a random selection of 1/10 of the dataset was used with TopCap, since STFT–CNN uses 1/10 of the whole dataset as a validation set.  

[`STFT–CNN-16.py`](STFT–CNN-16.py) uses the sampled dataset solely for CNN, catering to the needs of the post-sampling phase of large datasets.  
