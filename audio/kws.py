# -*- coding: utf-8 -*-
# @Author: Noah Huetter
# @Date:   2020-04-16 16:59:06
# @Last Modified by:   Noah Huetter
# @Last Modified time: 2020-05-05 21:13:27

import audioutils as au
import mfcc_utils as mfu
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Softmax
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pathlib
import librosa
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import tensorflow as tf
try:
  tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
except:
  pass

if len(sys.argv) < 2:
  print('specify mode')
  exit()

cache_dir = '.cache/kws_multi'
verbose = 1

# Limit in number of samples to take. make sure the correct wav files are present!
trainsize = 1000
testsize = 100

# training parameters
batchSize = 100
epochs  = 50

# cut/padd each sample to that many seconds
sample_len_seconds = 4.0

# MFCC settings
fs = 16000.0
mel_mtx_scale = 128
lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 32
frame_length = 1024
num_mfcc = 13
nSamples = int(sample_len_seconds*fs)
frame_len = frame_length
frame_step = frame_len
frame_count = 0 # 0 for auto
fft_len = frame_len


##################################################
# Model definition

def get_model(inp_shape, num_classes):
  """
    Build CNN model
  """
  print("Building model with input shape %s and %d classes" % (inp_shape, num_classes))

  model = Sequential()
  model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=inp_shape))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(num_classes, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy'])
  return model

def get_model_multiclass(inp_shape, num_classes):
  """
    CNN model for spotting multiple keywords
  """
  print("Building model with input shape %s and %d classes" % (inp_shape, num_classes))

  # first_filter_width = 8
  # first_filter_height = 10
  # first_filter_count = 8
  
  # first_conv_stride_x = 2
  # first_conv_stride_y = 2

  # model = Sequential()
  # model.add(Conv2D(first_filter_count, 
  #   kernel_size=(first_filter_width, first_filter_height),
  #   strides=(first_conv_stride_x, first_conv_stride_y),
  #   use_bias=True,
  #   activation='relu', 
  #   padding='same', 
  #   input_shape=inp_shape) )
  
  # dropout_rate = 0.25
  # model.add(Dropout(dropout_rate))
  # model.add(Flatten())
  # model.add(Dense(num_classes))
  # model.add(Softmax())
  # model.compile(loss='categorical_crossentropy', 
  #   optimizer='adam', 
  #   metrics=['accuracy'])


  first_filter_width = 8
  first_filter_height = 20
  first_filter_count = 64
  first_conv_stride_x = 1
  first_conv_stride_y = 1

  model = Sequential()
  model.add(Conv2D(first_filter_count, 
    kernel_size=(first_filter_width, first_filter_height),
    strides=(first_conv_stride_x, first_conv_stride_y),
    use_bias=True,
    activation='relu', 
    padding='same', 
    input_shape=inp_shape) )
  
  dropout_rate = 0.25
  model.add(Dropout(dropout_rate))

  second_filter_width = 4
  second_filter_height = 10
  second_filter_count = 64
  second_conv_stride_x = 1
  second_conv_stride_y = 1

  model.add(Conv2D(second_filter_count, 
    kernel_size=(second_filter_width, second_filter_height),
    strides=(second_conv_stride_x, second_conv_stride_y),
    use_bias=True,
    activation='relu', 
    padding='same', 
    input_shape=inp_shape) )
  
  dropout_rate = 0.25
  model.add(Dropout(dropout_rate))

  model.add(Flatten())
  model.add(Dense(num_classes))
  model.add(Softmax())
  
  model.compile(loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy'])
  
  return model


##################################################
# Training
def train(model, x, y, vx, vy, batchSize = 10, epochs = 30):

  model.fit(x, y, batch_size=batchSize, epochs=epochs, 
    verbose=verbose, validation_data=(vx, vy))


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
    x_train, y_train, x_test, y_test = au.load_snips_data(sample_len=int(sample_len_seconds*16000), trainsize = trainsize, testsize = testsize)

    fs = 16000.0
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    frame_length = 1024
    num_mfcc = 13
    stfts = tf.signal.stft(x_train, frame_length=frame_length, frame_step=frame_length, fft_length=frame_length)
    spectrograms = tf.abs(stfts)
    spectrograms = tf.reshape(spectrograms, (spectrograms.shape[0],spectrograms.shape[1],-1))
    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, fs, lower_edge_hertz,
      upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :num_mfcc]
    x_train_mfcc = tf.reshape(mfccs, (mfccs.shape[0],mfccs.shape[1],mfccs.shape[2],-1))

    fs = 16000.0
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    frame_length = 1024
    num_mfcc = 13
    stfts = tf.signal.stft(x_test, frame_length=frame_length, frame_step=frame_length, fft_length=frame_length)
    spectrograms = tf.abs(stfts)
    spectrograms = tf.reshape(spectrograms, (spectrograms.shape[0],spectrograms.shape[1],-1))
    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, fs, lower_edge_hertz,
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

def load_data_mculike():
  """
    Load data and compute MFCC with scaled and custom implementation as it is done on the MCU
  """
  # if in cache, use it
  try:
    x_train_mfcc = np.load(cache_dir+'/x_train_mfcc_mcu.npy')
    x_test_mfcc = np.load(cache_dir+'/x_test_mfcc_mcu.npy')
    y_train = np.load(cache_dir+'/y_train_mcu.npy')
    y_test = np.load(cache_dir+'/y_test_mcu.npy')
    assert x_train_mfcc.shape[1:] == x_test_mfcc.shape[1:]
    print('Load data from cache success!')

  except:
    # failed, load from source wave files
    x_train, y_train, x_test, y_test = au.load_snips_data(sample_len=int(sample_len_seconds*16000), trainsize = trainsize, testsize = testsize)

    # calculate MFCCs of training and test x data
    o_mfcc_train = []
    o_mfcc_test = []
    print('starting mfcc calculation')
    for data in tqdm(x_train):
      o_mfcc = mfu.mfcc_mcu(data, fs, nSamples, frame_len, frame_step, frame_count, fft_len, 
        num_mel_bins, lower_edge_hertz, upper_edge_hertz, mel_mtx_scale)
      o_mfcc_train.append([x['mfcc'][:num_mfcc] for x in o_mfcc])
    for data in tqdm(x_test):
      o_mfcc = mfu.mfcc_mcu(data, fs, nSamples, frame_len, frame_step, frame_count, fft_len, 
        num_mel_bins, lower_edge_hertz, upper_edge_hertz, mel_mtx_scale)
      o_mfcc_test.append([x['mfcc'][:num_mfcc] for x in o_mfcc])

    # add dimension to get (x, y, 1) from to make conv2D input layer happy
    x_train_mfcc = np.expand_dims(np.array(o_mfcc_train), axis = -1)
    x_test_mfcc = np.expand_dims(np.array(o_mfcc_test), axis = -1)

    # store data
    print('Store mfcc data')
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.save(cache_dir+'/x_train_mfcc_mcu.npy', x_train_mfcc)
    np.save(cache_dir+'/x_test_mfcc_mcu.npy', x_test_mfcc)
    np.save(cache_dir+'/y_train_mcu.npy', y_train)
    np.save(cache_dir+'/y_test_mcu.npy', y_test)

  # return
  return x_train_mfcc, x_test_mfcc, y_train, y_test


def load_data_mculike_multi(kwds):
  """
    Load data and compute MFCC with scaled and custom implementation as it is done on the MCU
  """
  # if in cache, use it
  try:
    x_train_mfcc = np.load(cache_dir+'/x_train_mfcc_mcu.npy')
    x_test_mfcc = np.load(cache_dir+'/x_test_mfcc_mcu.npy')
    x_val_mfcc = np.load(cache_dir+'/x_val_mfcc_mcu.npy')
    y_train = np.load(cache_dir+'/y_train_mcu.npy')
    y_test = np.load(cache_dir+'/y_test_mcu.npy')
    y_val = np.load(cache_dir+'/y_val_mcu.npy')
    assert x_train_mfcc.shape[1:] == x_test_mfcc.shape[1:]
    print('Load data from cache success!')

  except:
    # failed, load from source wave files
    x_train, y_train, x_test, y_test, x_validation, y_val = au.load_speech_commands(keywords=kwds, sample_len=2*16000)


    sample_len_seconds = 2.0
    fs = 16000.0
    mel_mtx_scale = 128
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 32
    frame_length = 1024
    num_mfcc = 13
    nSamples = int(sample_len_seconds*fs)
    frame_len = frame_length
    frame_step = frame_len
    frame_count = 0 # 0 for auto
    fft_len = frame_len

    # calculate MFCCs of training and test x data
    o_mfcc_train = []
    o_mfcc_test = []
    o_mfcc_val = []
    print('starting mfcc calculation')
    for data in tqdm(x_train):
      o_mfcc = mfu.mfcc_mcu(data, fs, nSamples, frame_len, frame_step, frame_count, fft_len, 
        num_mel_bins, lower_edge_hertz, upper_edge_hertz, mel_mtx_scale)
      o_mfcc_train.append([x['mfcc'][:num_mfcc] for x in o_mfcc])
    for data in tqdm(x_test):
      o_mfcc = mfu.mfcc_mcu(data, fs, nSamples, frame_len, frame_step, frame_count, fft_len, 
        num_mel_bins, lower_edge_hertz, upper_edge_hertz, mel_mtx_scale)
      o_mfcc_test.append([x['mfcc'][:num_mfcc] for x in o_mfcc])
    for data in tqdm(x_validation):
      o_mfcc = mfu.mfcc_mcu(data, fs, nSamples, frame_len, frame_step, frame_count, fft_len, 
        num_mel_bins, lower_edge_hertz, upper_edge_hertz, mel_mtx_scale)
      o_mfcc_val.append([x['mfcc'][:num_mfcc] for x in o_mfcc])

    # add dimension to get (x, y, 1) from to make conv2D input layer happy
    x_train_mfcc = np.expand_dims(np.array(o_mfcc_train), axis = -1)
    x_test_mfcc = np.expand_dims(np.array(o_mfcc_test), axis = -1)
    x_val_mfcc = np.expand_dims(np.array(o_mfcc_val), axis = -1)

    # convert labels to categorial one-hot coded
    y_train = to_categorical(y_train, num_classes=None)
    y_test = to_categorical(y_test, num_classes=None)
    y_val = to_categorical(y_val, num_classes=None)

    # store data
    print('Store mfcc data')
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.save(cache_dir+'/x_train_mfcc_mcu.npy', x_train_mfcc)
    np.save(cache_dir+'/x_test_mfcc_mcu.npy', x_test_mfcc)
    np.save(cache_dir+'/x_val_mfcc_mcu.npy', x_val_mfcc)
    np.save(cache_dir+'/y_train_mcu.npy', y_train)
    np.save(cache_dir+'/y_test_mcu.npy', y_test)
    np.save(cache_dir+'/y_val_mcu.npy', y_val)

  # return
  return x_train_mfcc, x_test_mfcc, x_val_mfcc, y_train, y_test, y_val

##################################################
# plottery
def plotInputDifference(mfccs, names):
  """
    Plot different mfcc inputs
  """
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

def plotSomeMfcc(x_train, x_test, y_train=None, y_test=None, keywords=None):
  """
    Plot a grid of MFCCs to check train and test data
  """
  frames = np.arange(x_train.shape[1])
  melbin = np.arange(x_train.shape[2])

  fig, axs = plt.subplots(4, 4)
  fig.set_size_inches(8,8)

  vmin = 0
  vmax = 1500
  
  import random 

  for i in range(8):
    ax=axs[i//2, i%2]
    i = random.randint(0,x_train.shape[0])
    data = np.squeeze(x_train[i,:,:].T, axis=0)
    if y_train is not None:
      lbl = ('x_train[%d]:%s' % (i, keywords[np.argmax(y_train[i])]))
    else:
      lbl = ('x_train[%d]' % (i))
    c = ax.pcolor(frames, melbin, data, cmap='PuBu', vmin=vmin, vmax=vmax, label=lbl)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('frame')
    ax.set_ylabel('mfcc bin')
    # fig.colorbar(c, ax=ax)

  for i in range(8):
    ax=axs[i//2, 2+i%2]
    i = random.randint(0,x_test.shape[0])
    data = np.squeeze(x_test[i,:,:].T, axis=0)
    if y_test is not None:
      lbl = ('x_test[%d]:%s' % (i, keywords[np.argmax(y_test[i])]))
    else:
      lbl = ('x_test[%d]' % (i))
    c = ax.pcolor(frames, melbin, data, cmap='PuBu', vmin=vmin, vmax=vmax, label=lbl)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('frame')
    ax.set_ylabel('mfcc bin')
    # fig.colorbar(c, ax=ax)

  return fig, axs

##################################################
# MAIN
# The old model using snips dataset
##################################################
if sys.argv[1] == 'snips':
  # x_train_mfcc, x_test_mfcc, y_train, y_test = load_data()
  # x_train_mfcc2, x_test_mfcc2, y_train2, y_test2 = load_data2()
  # x_train_mfcc3, x_test_mfcc3, y_train3, y_test3 = load_data3()
  # x_test_mfcc2 = x_test_mfcc2.transpose((0,2,1,3))
  # print(x_test_mfcc.shape)
  # print(x_test_mfcc2.shape)
  # print(x_test_mfcc3.shape)
  # plotInputDifference([x_test_mfcc, x_test_mfcc2, x_test_mfcc3], ['custom implemetation', 'librosa', 'tensorflow'])
  # exit()


  # x_train_mfcc, x_test_mfcc, y_train, y_test = load_data()
  x_train_mfcc, x_test_mfcc, y_train, y_test = load_data_mculike()

  assert x_train_mfcc.shape[1:] == x_test_mfcc.shape[1:]
  print('x train shape: ', x_train_mfcc.shape)
  print('x test shape: ', x_test_mfcc.shape)
  print('y train shape: ', y_train.shape)
  print('y test shape: ', y_test.shape)

  # fig, axs = plotSomeMfcc(x_train_mfcc, x_test_mfcc)
  # plt.show()
  # exit()

  ##################################################
  # Build model
  model = get_model(inp_shape=x_train_mfcc.shape[1:], num_classes = 1)
  model.summary()
  train_set, train_labels, test_set, test_labels = train(model, 
    x_train_mfcc, y_train, x_test_mfcc, y_test, batchSize = batchSize, epochs = epochs)
  model.summary()
  y_pred = np.rint(model.predict(x_test_mfcc).reshape((-1,))).astype(int)

  print('Prediction:')
  print(y_pred)
  print('True:')
  print(y_test)

  print('Confusion matrix:')
  print(confusion_matrix(y_test, y_pred))
  # model.evaluate(test_set, y_test)
  from datetime import datetime
  dte = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
  fname = cache_dir+'/kws_model_'+dte+'.h5'
  model.save(fname)
  print('Model saved as %s' % (fname))

##################################################
# MAIN
# for multiple possible keywords
##################################################
if sys.argv[1] == 'multi':

  # load data
  keywords = ['cat','marvin','left','zero']
  x_train_mfcc, x_test_mfcc, x_val_mfcc, y_train, y_test, y_val = load_data_mculike_multi(keywords)
  
  print('x train shape: ', x_train_mfcc.shape)
  print('x test shape: ', x_test_mfcc.shape)
  print('y train shape: ', y_train.shape)
  print('y test shape: ', y_test.shape)

  if sys.argv[2] == 'train':
    # build model
    model = get_model_multiclass(inp_shape=x_train_mfcc.shape[1:], num_classes = len(keywords))

    # train model
    model.summary()
    train(model, x_train_mfcc, y_train, x_val_mfcc, y_val, batchSize = batchSize, epochs = epochs)

    # store model
    from datetime import datetime
    dte = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    fname = cache_dir+'/kws_model_'+dte+'.h5'
    model.save(fname)
    print('Model saved as %s' % (fname))

  else:
    # load model
    model = tf.keras.models.load_model(sys.argv[2])
    model.summary()

  # fig, axs = plotSomeMfcc(x_train_mfcc, x_test_mfcc, y_train, y_test, keywords)
  # plt.show()
  # exit()

  y_pred = model.predict(x_test_mfcc)
  y_pred = 1.0*(y_pred > 0.5) 
  
  # print(y_pred)
  # print(y_pred.shape)
  # print(y_test)
  # print(y_test.shape)

  print('Confusion matrix:')
  cmtx = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
  print(cmtx)

  # true positive
  tp = np.sum(np.diagonal(cmtx))
  # total number of predictions
  tot = np.sum(cmtx)

  print('Correct predicionts: %d/%d (%.2f%%)' % (tp, tot, 100.0/tot*tp))



