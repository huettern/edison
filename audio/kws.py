# -*- coding: utf-8 -*-
# @Author: Noah Huetter
# @Date:   2020-04-16 16:59:06
# @Last Modified by:   Noah Huetter
# @Last Modified time: 2020-04-20 10:42:05

import audioutils as au
import mfcc_utils as mfu
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
import os
import pathlib
import librosa
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import tensorflow as tf
try:
  tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
except:
  pass

cache_dir = '.cache'
verbose = 1
trainsize = 1000
testsize = 100

##################################################
# Model definition

def get_model(inp_shape, num_classes):
  """
    Build CNN model
  """
  print("Building model with input shape %s" % (inp_shape, ))
  # model = Sequential()
  # model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=inp_shape))
  # model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
  # model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
  # model.add(MaxPooling2D(pool_size=(2, 2)))
  # model.add(Dropout(0.25))
  # model.add(Flatten())
  # model.add(Dense(128, activation='relu'))
  # model.add(Dropout(0.25))
  # model.add(Dense(64, activation='relu'))
  # model.add(Dropout(0.4))
  # model.add(Dense(num_classes, activation='softmax'))
  # model.compile(loss='binary_crossentropy',
  #               optimizer=keras.optimizers.Adam(),
  #               metrics=['accuracy'])

  model = Sequential()
  model.add(Conv2D(4, kernel_size=(3, 3), activation='relu', padding='same', input_shape=inp_shape))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(num_classes, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy'])
  return model

##################################################
# Training
def train(model, batchSize = 10, epochs = 30):
  train_set = x_train_mfcc
  train_labels = y_train
  test_set = x_test_mfcc
  test_labels = y_test

  input_shape=(train_set[0].shape)
  print (input_shape)
  print(type(x_train_mfcc))

  model.fit(train_set, train_labels, batch_size=batchSize, epochs=epochs, 
    verbose=verbose, validation_data=(test_set, test_labels))

  return train_set, train_labels, test_set, test_labels


##################################################
# load data
def load_data():

  # Check if cached data exists
  try:
    x_train_mfcc = np.load(cache_dir+'/x_train_mfcc.npy')
    x_test_mfcc = np.load(cache_dir+'/x_test_mfcc.npy')
    y_train = np.load(cache_dir+'/y_train.npy')
    y_test = np.load(cache_dir+'/y_test.npy')
    assert x_train_mfcc.shape[1:] == x_test_mfcc.shape[1:]
    print('Load data from cache success!')

  except FileNotFoundError:
    print('Loading data from source')
    x_train, y_train, x_test, y_test = au.load_snips_data(trainsize = trainsize, testsize = testsize)

    fs = 16e3
    nSamples = x_train.shape[-1]
    frame_len = 1024
    frame_step = 1024
    frame_count = 0 # calculate automatically
    fft_len = frame_len
    mel_nbins = 80
    mel_lower_hz = 80.0
    mel_upper_hz = 7600.0

    x_train_mfcc = mfu.batch_mfcc(x_train, fs, nSamples, frame_len, frame_step, frame_count,
      fft_len, mel_nbins, mel_lower_hz, mel_upper_hz)
    nSamples = x_test.shape[-1]
    x_test_mfcc = mfu.batch_mfcc(x_test, fs, nSamples, frame_len, frame_step, frame_count,
      fft_len, mel_nbins, mel_lower_hz, mel_upper_hz)

    # We don't use all MFCCs to safe some space, just consider the first N, they contain the most energy
    num_mfcc = 13
    x_train_mfcc = x_train_mfcc[:,:,:num_mfcc]
    x_test_mfcc = x_test_mfcc[:,:,:num_mfcc]
    # expand the mfcc values to 1D arrays to make keras Conv2D input happy
    x_train_mfcc  = x_train_mfcc.reshape(x_train_mfcc.shape+(1,))
    x_test_mfcc  = x_test_mfcc.reshape(x_test_mfcc.shape+(1,))

    # store data
    print('Store mfcc data')
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.save(cache_dir+'/x_train_mfcc.npy', x_train_mfcc)
    np.save(cache_dir+'/x_test_mfcc.npy', x_test_mfcc)
    np.save(cache_dir+'/y_train.npy', y_train)
    np.save(cache_dir+'/y_test.npy', y_test)

  return x_train_mfcc, x_test_mfcc, y_train, y_test

def load_data2(max_len=11):
  """
    Another method for loading and preprocessing data, taken from DeadSimpleSeechRecognizer
  """
  try:
    x_train_mfcc = np.load(cache_dir+'/x_train_mfcc_2.npy')
    x_test_mfcc = np.load(cache_dir+'/x_test_mfcc_2.npy')
    y_train = np.load(cache_dir+'/y_train_2.npy')
    y_test = np.load(cache_dir+'/y_test_2.npy')
    assert x_train_mfcc.shape[1:] == x_test_mfcc.shape[1:]
    print('Load data from cache success!')

  except:
    print('Loading data from source')
    x_train, y_train, x_test, y_test = au.load_snips_data(trainsize = trainsize, testsize = testsize)
  
    print('calculate mfcc with librosa')
    x_train_mfcc = []
    for waveCtr in tqdm(range(x_train.shape[0])):
      mfcc = librosa.feature.mfcc(x_train[waveCtr], sr=16000, n_mfcc=13, hop_length=1024)
      if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
      else:
        mfcc = mfcc[:, :max_len]
      x_train_mfcc.append(mfcc)
    x_test_mfcc = []
    for waveCtr in tqdm(range(x_test.shape[0])):
      mfcc = librosa.feature.mfcc(x_test[waveCtr], sr=16000, n_mfcc=13, hop_length=1024)
      if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
      else:
        mfcc = mfcc[:, :max_len]
      x_test_mfcc.append(mfcc)

    x_train_mfcc = np.array(x_train_mfcc)
    x_test_mfcc = np.array(x_test_mfcc)
    x_train_mfcc  = x_train_mfcc.reshape(x_train_mfcc.shape+(1,))
    x_test_mfcc  = x_test_mfcc.reshape(x_test_mfcc.shape+(1,))

    # store data
    print('Store mfcc data')
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.save(cache_dir+'/x_train_mfcc_2.npy', x_train_mfcc)
    np.save(cache_dir+'/x_test_mfcc_2.npy', x_test_mfcc)
    np.save(cache_dir+'/y_train_2.npy', y_train)
    np.save(cache_dir+'/y_test_2.npy', y_test)

  return x_train_mfcc, x_test_mfcc, y_train, y_test


def load_data3(max_len=11):
  """
    Loading data as in the exercise
  """
  try:
    x_train_mfcc = np.load(cache_dir+'/x_train_mfcc_3.npy')
    x_test_mfcc = np.load(cache_dir+'/x_test_mfcc_3.npy')
    y_train = np.load(cache_dir+'/y_train_3.npy')
    y_test = np.load(cache_dir+'/y_test_3.npy')
    assert x_train_mfcc.shape[1:] == x_test_mfcc.shape[1:]
    print('Load data from cache success!')

  except:
    print('Loading data from source using Tensorflow MFCCs')
    x_train, y_train, x_test, y_test = au.load_snips_data(sample_len=4*16000, trainsize = trainsize, testsize = testsize)

    sample_rate = 16000.0
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    frame_length = 1024
    num_mfcc = 13
    stfts = tf.signal.stft(x_train, frame_length=frame_length, frame_step=frame_length, fft_length=frame_length)
    spectrograms = tf.abs(stfts)
    spectrograms = tf.reshape(spectrograms, (spectrograms.shape[0],spectrograms.shape[1],-1))
    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
      upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :num_mfcc]
    x_train_mfcc = tf.reshape(mfccs, (mfccs.shape[0],mfccs.shape[1],mfccs.shape[2],-1))

    sample_rate = 16000.0
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    frame_length = 1024
    num_mfcc = 13
    stfts = tf.signal.stft(x_test, frame_length=frame_length, frame_step=frame_length, fft_length=frame_length)
    spectrograms = tf.abs(stfts)
    spectrograms = tf.reshape(spectrograms, (spectrograms.shape[0],spectrograms.shape[1],-1))
    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
      upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :num_mfcc]
    x_test_mfcc = tf.reshape(mfccs, (mfccs.shape[0],mfccs.shape[1],mfccs.shape[2],-1))

    x_train_mfcc = x_train_mfcc.numpy()
    x_test_mfcc = x_test_mfcc.numpy()
    
    # store data
    print('Store mfcc data')
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.save(cache_dir+'/x_train_mfcc_3.npy', x_train_mfcc)
    np.save(cache_dir+'/x_test_mfcc_3.npy', x_test_mfcc)
    np.save(cache_dir+'/y_train_3.npy', y_train)
    np.save(cache_dir+'/y_test_3.npy', y_test)

  # return
  return x_train_mfcc, x_test_mfcc, y_train, y_test

##################################################
# plottery
def plotInputDifference(mfccs, names):
  """
    Plot different mfcc inputs
  """
  import matplotlib.pyplot as plt
  from matplotlib import colors
  import matplotlib as mpl
  
  sampleno_left = 5
  sampleno_right = 6

  plt.style.use('seaborn-bright')
  # t = np.linspace(0, nSamples/fs, num=nSamples)
  # f = np.linspace(0.0, fs/2.0, fft_len/2)
  fig, axs = plt.subplots(4, 2)
  fig.set_size_inches(8,8)
  
  # cmap = mpl.cm.cool
  # norm = mpl.colors.Normalize(vmin=0, vmax=100)

  cm = 'cool'#mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
  # fig.colorbar(cm,
  #              cax=ax, orientation='horizontal', label='Some Units')
  # norm = colors.Normalize(vmin=vmin, vmax=vmax)
  # for im in images:
  #   im.set_norm(norm)

  x_train, y_train, x_test, y_test = au.load_snips_data(trainsize = trainsize, testsize = testsize)
  ax=axs[0,0]
  ax.plot(x_test[sampleno_left])
  ax.set_title('raw hotword')
  ax.grid(True)
  ax=axs[0,1]
  ax.plot(x_test[sampleno_right])
  ax.set_title('raw coldword')
  ax.grid(True)

  axctr = 0
  for mfcc in mfccs:
    ax=axs[axctr+1,0]
    cm = ax.pcolor(np.transpose(mfcc[sampleno_left].reshape((mfcc[sampleno_left].shape[0], mfcc[sampleno_left].shape[1]))),cmap='RdBu')
    ax.set_xlabel('frame')
    ax.set_ylabel('MFCC')
    ax.set_xlim(0,62)
    ax.set_ylim(0,13)
    ax.set_title(names[axctr]+' - hotword')
    ax.grid(True)
    fig.colorbar(cm, ax=ax)

    ax=axs[axctr+1,1]
    cm = ax.pcolor(np.transpose(mfcc[sampleno_right].reshape((mfcc[sampleno_right].shape[0], mfcc[sampleno_right].shape[1]))),cmap='RdBu')
    ax.set_xlabel('frame')
    ax.set_ylabel('MFCC')
    ax.set_xlim(0,62)
    ax.set_ylim(0,13)
    ax.set_title(names[axctr]+' - coldword')
    ax.grid(True)
    fig.colorbar(cm, ax=ax)

    axctr = axctr + 1

  fig.tight_layout()
  fig.savefig('test2png.png', dpi=100)
  plt.show()


##################################################
# MAIN
##################################################

# x_train_mfcc, x_test_mfcc, y_train, y_test = load_data()
# x_train_mfcc2, x_test_mfcc2, y_train2, y_test2 = load_data2()
# x_train_mfcc3, x_test_mfcc3, y_train3, y_test3 = load_data3()
# x_test_mfcc2 = x_test_mfcc2.transpose((0,2,1,3))
# print(x_test_mfcc.shape)
# print(x_test_mfcc2.shape)
# print(x_test_mfcc3.shape)
# plotInputDifference([x_test_mfcc, x_test_mfcc2, x_test_mfcc3], ['custom implemetation', 'librosa', 'tensorflow'])
# exit()


x_train_mfcc, x_test_mfcc, y_train, y_test = load_data()

assert x_train_mfcc.shape[1:] == x_test_mfcc.shape[1:]
print(x_train_mfcc.shape)
print(x_test_mfcc.shape)
print(y_train.shape)
print(y_test.shape)
print(y_test)

##################################################
# Build model
model = get_model(inp_shape=x_train_mfcc.shape[1:], num_classes = 1)
model.summary()
train_set, train_labels, test_set, test_labels = train(model, batchSize = 10, epochs = 100)

model.summary()
print(x_test_mfcc.shape)
y_pred = np.rint(model.predict(x_test_mfcc).reshape((-1,))).astype(int)

print('Prediction:')
print(y_pred)
print('True:')
print(y_test)

print(confusion_matrix(y_test, y_pred))
# model.evaluate(test_set, y_test)
model.save(cache_dir+'/mfcc_model.h5')
