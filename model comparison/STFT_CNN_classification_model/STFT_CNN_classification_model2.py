import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

#Sample 2000 audio files, then use CNN.


# Obtain the local file path of the dataset.
DATASET_PATH = 'G:/Python Exercise/QQR Program/train500_audio_segment_wav_all'
data_dir = pathlib.Path(DATASET_PATH)

# Retrieve the dataset and select 2000 samples
original_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=None,  # No batch processing
    validation_split=None,  # No validation set splitting
    seed=0,
    output_sequence_length=16000
)

# Obtain the label names.
label_names = np.array(original_ds.class_names)

# Randomly shuffle the data.
full_ds = original_ds.shuffle(buffer_size=40000, seed=0)

# Select the first 2000 samples.
sampled_data = full_ds.take(2000)

# Split these 2000 samples into training and validation sets.
train_size = int(0.8 * 2000)
val_size = 2000 - train_size

# Use 'skip' and 'take' methods to correctly divide the dataset.
train_ds = sampled_data.take(train_size)
val_ds = sampled_data.skip(train_size)

# Rebatch the dataset.
train_ds = train_ds.batch(64)
val_ds = val_ds.batch(64)

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

  
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
  spectrogram = tf.image.resize(spectrogram, [16, 16])
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


