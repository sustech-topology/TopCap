import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model
import umap
import pandas as pd
from sklearn.utils import resample

# Corresponding .csv file for audio data
data = pd.read_csv('QQR Program/PD_TIMIT.csv')

# The first two components are features, and the third component is a label.
X = data.iloc[:, :2].values
y = data.iloc[:, 2].values

# Sample each label, taking 2123, 12831 (1/10) points.
X_label1 = X[y == 1]
X_label2 = X[y == 2]
X_label1_sampled = resample(X_label1, n_samples=2123, random_state=0)
X_label2_sampled = resample(X_label2, n_samples=2123, random_state=0)

# Merge the sampled data.
X_sampled = np.vstack((X_label1_sampled, X_label2_sampled))
y_sampled = np.array([1]*2123 + [2]*2123)

# Dimensionality reduction using t-SNE
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X_sampled)

# Dimensionality reduction using UMAP
umap_model = umap.UMAP(n_components=2, random_state=0)
X_umap = umap_model.fit_transform(X_sampled)

# Plot t-SNE results.
plt.figure(figsize=(10, 10))
plt.scatter(X_tsne[y_sampled == 1, 0], X_tsne[y_sampled == 1, 1], c=(0.69,0.58,0.80), alpha=0.6)
plt.scatter(X_tsne[y_sampled == 2, 0], X_tsne[y_sampled == 2, 1], c=(1,0.74,0.49), alpha=0.6)
plt.xticks([X_tsne[:, 0].min(), (X_tsne[:, 0].min() + X_tsne[:, 0].max()) / 2, X_tsne[:, 0].max()], fontsize=18)
plt.yticks([X_tsne[:, 1].min(), (X_tsne[:, 1].min() + X_tsne[:, 1].max()) / 2, X_tsne[:, 1].max()], fontsize=18)
plt.legend()
plt.savefig("G:/Articles and Pictures/Relations between RC and TCNN/Visualisation1/V11.pdf", dpi=300)

# Plot UMAP results.
plt.figure(figsize=(10, 10))
plt.scatter(X_umap[y_sampled == 1, 0], X_umap[y_sampled == 1, 1], c=(0.69,0.58,0.80), alpha=0.6)
plt.scatter(X_umap[y_sampled == 2, 0], X_umap[y_sampled == 2, 1], c=(1,0.74,0.49), alpha=0.6)
plt.xticks([X_umap[:, 0].min(), (X_umap[:, 0].min() + X_umap[:, 0].max()) / 2, X_umap[:, 0].max()], fontsize=18)
plt.yticks([X_umap[:, 1].min(), (X_umap[:, 1].min() + X_umap[:, 1].max()) / 2, X_umap[:, 1].max()], fontsize=18)
plt.legend()
plt.savefig("G:/Articles and Pictures/Relations between RC and TCNN/Visualisation1/V13.pdf", dpi=300)

#Local audio dataset
DATASET_PATH = 'QQR Program/TIMIT_2classes/TIMIT_2classes'

data_dir = pathlib.Path(DATASET_PATH)

#Split into training set, test set, and validation set.
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=16000,
    subset='both')

label_names = np.array(train_ds.class_names)

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

#Obtain spectrogram using STFT.
def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  # Resize the spectrogram to 8*8(STFT_CNN) or 16*16(STFT_CNN^+). 
  spectrogram = tf.image.resize(spectrogram, [8, 8])
  return spectrogram

def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
    break

train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

#Construct and run the neural network.
input_shape = example_spectrograms.shape[1:]
num_labels = len(label_names)

model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(64, 3, activation='relu', padding='same'
                  ),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu', padding='same'
                  ),
    layers.MaxPooling2D(),
    layers.Conv2D(256, 3, activation='relu', padding='same'
                  ),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'), 
    layers.Dense(1, activation='sigmoid'),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

EPOCHS = 10
history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
)

y_pred = model.predict(test_spectrogram_ds)

if y_pred.shape[1] == 2:
    y_pred = tf.argmax(y_pred, axis=1)
# If y_pred is a matrix of shape (num_samples, 1)
else:
    y_pred = tf.cast(y_pred > 0.5, tf.int32)

y_true = tf.concat(list(test_spectrogram_ds.map(lambda s,lab: lab)), axis=0)

x_true = tf.concat(list(train_spectrogram_ds.map(lambda s,lab: lab)), axis=0)
last_conv_layer = model.layers[7]
# Obtain the output of the last convolutional layer.
intermediate_model = Model(inputs=model.input, outputs=last_conv_layer.output)
feature_vectors = intermediate_model.predict(test_spectrogram_ds)

# Reduce the feature vectors to 2D using t-SNE.
tsne = TSNE(n_components=2, random_state=42)
reduced_features = tsne.fit_transform(feature_vectors)

# t-SNE Visualization of CNN Latent Space
plt.figure(figsize=(10, 10))
plt.scatter(reduced_features[y_true == 0, 0], reduced_features[y_true == 0, 1], c=(0.69,0.58,0.80), alpha=0.6)
plt.scatter(reduced_features[y_true == 1, 0], reduced_features[y_true == 1, 1], c=(1,0.74,0.49), alpha=0.6)
plt.xticks([reduced_features[:, 0].min(), (reduced_features[:, 0].min() + reduced_features[:, 0].max()) / 2, 
            reduced_features[:, 0].max()], fontsize=14)
plt.yticks([reduced_features[:, 1].min(), (reduced_features[:, 1].min() + reduced_features[:, 1].max()) / 2, 
            reduced_features[:, 1].max()], fontsize=14)
plt.legend()
plt.savefig("G:/Articles and Pictures/Relations between RC and TCNN/Visualisation1/V12.pdf", dpi=300)

# Reduce the feature vectors to 2D using UMAP.
umap_reducer = umap.UMAP(n_components=2, random_state=42)
umap_reduced_features = umap_reducer.fit_transform(feature_vectors)

# UMAP Visualization of CNN Latent Space
plt.figure(figsize=(10, 10))
plt.scatter(umap_reduced_features[y_true == 0, 0], umap_reduced_features[y_true == 0, 1], c=(0.69,0.58,0.80), alpha=0.6)
plt.scatter(umap_reduced_features[y_true == 1, 0], umap_reduced_features[y_true == 1, 1], c=(1,0.74,0.49), alpha=0.6)
plt.xticks([umap_reduced_features[:, 0].min(), 
            (umap_reduced_features[:, 0].min() + umap_reduced_features[:, 0].max()) / 2, 
            umap_reduced_features[:, 0].max()], fontsize=18)
plt.yticks([umap_reduced_features[:, 1].min(), 
            (umap_reduced_features[:, 1].min() + umap_reduced_features[:, 1].max()) / 2, 
            umap_reduced_features[:, 1].max()], fontsize=18)
plt.legend()
plt.savefig("G:/Articles and Pictures/Relations between RC and TCNN/Visualisation1/V14.pdf", dpi=300)

