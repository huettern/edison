#!/usr/bin/env python
# coding: utf-8

# ## MFCC feature extraction and Network training
# 
# In this notebook you will go through an example flow of processing audio data, complete with feature extraction and training.
# 
# Make sure you read the instructions on the exercise sheet and follow the task order.

# #### Task 1 

# In[1]:


import json
import numpy as np
from scipy.io import wavfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tqdm import tqdm

assert(tf.__version__ == "2.1.0")
assert(tf.executing_eagerly() == True)

DataSetPath = "/Users/noah/git/mlmcu-project/audio/data/snips/"

with open(DataSetPath+"train.json") as jsonfile:
    traindata = json.load(jsonfile)

with open(DataSetPath+"test.json") as jsonfile:
    testdata = json.load(jsonfile)


# #### Task 2

# In[2]:


def load_data():
    x_train_list = []
    y_train_list = []

    x_test_list = []
    y_test_list = []

    totalSliceLength = 10 # Length to stuff the signals to, given in seconds,10

    # trainsize = len(traindata) # Number of loaded training samples
    # testsize = len(testdata) # Number of loaded testing samples

    trainsize = 1000 # Number of loaded training samples
    testsize = 100 # Number of loaded testing samples


    fs = 16000 # Sampling rate of the samples
    segmentLength = 1024 # Number of samples to use per segment

    sliceLength = int(totalSliceLength * fs / segmentLength)*segmentLength

    for i in tqdm(range(trainsize)): 
        fs, train_sound_data = wavfile.read(DataSetPath+traindata[i]['audio_file_path']) # Read wavfile to extract amplitudes

        _x_train = train_sound_data.copy() # Get a mutable copy of the wavfile
        _x_train.resize(sliceLength) # Zero stuff the single to a length of sliceLength
        _x_train = _x_train.reshape(-1,int(segmentLength)) # Split slice into Segments with 0 overlap
        x_train_list.append(_x_train.astype(np.float32)) # Add segmented slice to training sample list, cast to float so librosa doesn't complain
        y_train_list.append(traindata[i]['is_hotword']) # Read label 

    for i in tqdm(range(testsize)):
        fs, test_sound_data = wavfile.read(DataSetPath+testdata[i]['audio_file_path'])
        _x_test = test_sound_data.copy()
        _x_test.resize(sliceLength)
        _x_test = _x_test.reshape((-1,int(segmentLength)))
        x_test_list.append(_x_test.astype(np.float32))
        y_test_list.append(testdata[i]['is_hotword'])

    x_train = tf.convert_to_tensor(np.asarray(x_train_list))
    y_train = tf.convert_to_tensor(np.asarray(y_train_list))

    x_test = tf.convert_to_tensor(np.asarray(x_test_list))
    y_test = tf.convert_to_tensor(np.asarray(y_test_list))

    return x_train, y_train, x_test, y_test


# In[3]:


def compute_mfccs(tensor):
    sample_rate = 16000.0
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    frame_length = 1024
    num_mfcc = 13

    stfts = tf.signal.stft(tensor, frame_length=frame_length, frame_step=frame_length, fft_length=frame_length)
    spectrograms = tf.abs(stfts)
    spectrograms = tf.reshape(spectrograms, (spectrograms.shape[0],spectrograms.shape[1],-1))
    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
      upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :num_mfcc]
    return tf.reshape(mfccs, (mfccs.shape[0],mfccs.shape[1],mfccs.shape[2],-1))


# In[4]:


x_train, y_train, x_test, y_test = load_data()


# #### Task 3

# In[5]:


x_train_mfcc = compute_mfccs(x_train)
x_test_mfcc = compute_mfccs(x_test)

print(x_train_mfcc.shape)
print(x_test_mfcc.shape)


# #### Task 4

# In[14]:


batchSize = 10
epochs = 30

train_set = (x_train_mfcc/512 + 0.5)
train_labels = y_train

test_set = (x_test_mfcc/512 + 0.5)
test_labels = y_test

model = tf.keras.models.Sequential()

input_shape=(train_set[0].shape)
print (input_shape)
print(type(x_train_mfcc))

num_classes = 2
model.add(tf.keras.layers.Conv2D(4, (3, 3), padding='same', input_shape=train_set.shape[1:]))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(num_classes))
model.add(tf.keras.layers.Activation('softmax'))

# model.add(tf.keras.layers.Dense(10, activation='relu', input_shape=input_shape))
# model.add(tf.keras.layers.Dense(10, activation='relu'))
# model.add(tf.keras.layers.Dense(1))
# model.add(tf.keras.layers.Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
# model.fit(train_set, y_train, None, epochs, steps_per_epoch=int(1000/batchSize))
model.fit(train_set, y_train, batchSize, epochs)


# In[15]:


model.summary()
model.evaluate(test_set, y_test)


# In[ ]:


model.save(".cache/MFCCmodel.h5")


# #### NNoM Extract

# In[ ]:


# from nnom_utils import * 
# from tensorflow import keras

# nptest = test_set.numpy()

# model = keras.models.load_model("./MFCCmodel.h5")
# generate_model(model, nptest, name='weights.h')


# In[ ]:




