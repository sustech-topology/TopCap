The difference between STFT_CNN and STFT_CNN^+ is that the spectrogram size of the former is 8×8, while that of the latter is 16×16. The corresponding code is as follows. 
 ```
  spectrogram = tf.image.resize(spectrogram, [16, 16]), 
  ```
which appears in both two programs. 

STFT_CNN_classifcation_model1 use the whole datasets to CNN and plot the pictures with t-SNE and UMAP. 

STFT_CNN_classifcation_model1 use sampled datasets to CNN only. 
