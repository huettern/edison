# -*- coding: utf-8 -*-
# @Author: Noah Huetter
# @Date:   2020-04-16 16:59:06
# @Last Modified by:   Noah Huetter
# @Last Modified time: 2020-05-07 16:34:24

import audioutils as au
import mfcc_utils as mfu
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Softmax, Input
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from datetime import datetime
import pathlib
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import tensorflow as tf
try:
  tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
except:
  pass

cache_dir = '.cache/kws_keras'
verbose = 1

# Select model architecture here:
# conv_model: Total params: 202,884
#
#       [Conv2D]<-(weights)
#           v
#       [BiasAdd]<-(bias)
#           v
#         [Relu]
#           v
#       [MaxPool]
#           v
#       [Conv2D]<-(weights)
#           v
#       [BiasAdd]<-(bias)
#           v
#         [Relu]
#           v
#       [MaxPool]
#           v
#       [MatMul]<-(weights)
#           v
#       [BiasAdd]<-(bias)
#           v
#
# low_latency_conv Total params: 3,015,662
#     [Conv2D]<-(weights)
#          v
#      [BiasAdd]<-(bias)
#          v
#        [Relu]
#          v
#      [MatMul]<-(weights)
#          v
#      [BiasAdd]<-(bias)
#          v
#      [MatMul]<-(weights)
#          v
#      [BiasAdd]<-(bias)
#          v
#      [MatMul]<-(weights)
#          v
#      [BiasAdd]<-(bias)
#
# single_fc: Total params: 1,616
#       [MatMul]<-(weights)
#           v
#       [BiasAdd]<-(bias)
#
# tiny_conv:Total params: 4,236
#      [Conv2D]<-(weights)
#          v
#      [BiasAdd]<-(bias)
#          v
#        [Relu]
#          v
#      [MatMul]<-(weights)
#          v
#      [BiasAdd]<-(bias)

# tiny_embedding_conv: Total params: 5,812
#     [Conv2D]<-(weights)
#         v
#     [BiasAdd]<-(bias)
#         v
#       [Relu]
#         v
#     [Conv2D]<-(weights)
#         v
#     [BiasAdd]<-(bias)
#         v
#       [Relu]
#         v
#     [MatMul]<-(weights)
#         v
#     [BiasAdd]<-(bias)

# medium_embedding_conv: Total params: 5,664
#
#       [Conv2D]<-(weights)
#           v
#       [BiasAdd]<-(bias)
#           v
#         [Relu]
#           v
#       [MaxPool]
#           v
#       [Conv2D]<-(weights)
#           v
#       [BiasAdd]<-(bias)
#           v
#         [Relu]
#           v
#       [MaxPool]
#           v
#       [MatMul]<-(weights)
#           v
#       [BiasAdd]<-(bias)
#           v


model_arch = 'medium_embedding_conv'

# training parameters
batchSize = 100
epochs  = 50

# cut/padd each sample to that many seconds
sample_len_seconds = 2.0

# audio settings
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
    CNN model for spotting multiple keywords
  """
  print("Building %s model with input shape %s and %d classes" % (model_arch, inp_shape, num_classes))

  if model_arch == 'tiny_conv':

    first_filter_width = 8
    first_filter_height = 10
    first_filter_count = 8
    
    first_conv_stride_x = 2
    first_conv_stride_y = 2

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
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Softmax())

  if model_arch == 'medium_embedding_conv':
    first_filter_width = 8
    first_filter_height = 8
    first_filter_count = 16
    first_conv_stride_x = 2
    first_conv_stride_y = 2

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
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))

    second_filter_width = 4
    second_filter_height = 4
    second_filter_count = 12
    second_conv_stride_x = 1
    second_conv_stride_y = 1

    model.add(Conv2D(second_filter_count, 
      kernel_size=(second_filter_width, second_filter_height),
      strides=(second_conv_stride_x, second_conv_stride_y),
      use_bias=True,
      activation='relu', 
      padding='same' ) )
    
    dropout_rate = 0.25
    model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Softmax())
    

  if model_arch == 'tiny_embedding_conv':
    first_filter_width = 8
    first_filter_height = 10
    first_filter_count = 8
    first_conv_stride_x = 2
    first_conv_stride_y = 2

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
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))

    second_filter_width = 8
    second_filter_height = 10
    second_filter_count = 8
    second_conv_stride_x = 8
    second_conv_stride_y = 8

    model.add(Conv2D(second_filter_count, 
      kernel_size=(second_filter_width, second_filter_height),
      strides=(second_conv_stride_x, second_conv_stride_y),
      use_bias=True,
      activation='relu', 
      padding='same' ) )
    
    dropout_rate = 0.25
    model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Softmax())

  if model_arch == 'conv_model':
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
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))

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
      padding='same' ) )
    
    dropout_rate = 0.25
    model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Softmax())
    
  
  if model_arch == 'low_latency_conv':
    first_filter_width = 8
    first_filter_height = inp_shape[0] # input time size
    first_filter_count = 186
    first_conv_stride_x = 1
    first_conv_stride_y = 4

    model = Sequential()
    model.add(Conv2D(first_filter_count, 
      kernel_size=(first_filter_width, first_filter_height),
      strides=(first_conv_stride_x, first_conv_stride_y),
      use_bias=True,
      activation='relu', 
      padding='same', 
      input_shape=inp_shape ) )
    
    dropout_rate = 0.25
    model.add(Dropout(dropout_rate))

    first_fc_output_channels = 128
    model.add(Flatten())
    model.add(Dense(first_fc_output_channels, activation=None))

    second_fc_output_channels = 128
    model.add(Dense(second_fc_output_channels, activation=None))
    
    dropout_rate = 0.25
    model.add(Dropout(dropout_rate))

    model.add(Dense(num_classes))
    model.add(Softmax())
    

  if model_arch == 'single_fc':
    model = Sequential()
    model.add(Input(shape=inp_shape))
    model.add(Flatten())
    model.add(Dense(num_classes, activation=None, use_bias=True))
    model.add(Softmax())

  model.compile(loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy'])

  return model


##################################################
# Training
def train(model, x, y, vx, vy, batchSize = 10, epochs = 30):
  
  logdir = "/tmp/edison/train/" + datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

  train_history = model.fit(x, y, batch_size=batchSize, epochs=epochs, 
    verbose=verbose, validation_data=(vx, vy),
    callbacks=[tensorboard_callback])

  return train_history


##################################################
# load data

def load_data(keywords, coldwords, noise):
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
    x_train, y_train, x_test, y_test, x_validation, y_val, keywords = au.load_speech_commands(
      keywords=keywords, sample_len=2*16000, coldwords=coldwords, noise=noise, playsome=False)

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
  return x_train_mfcc, x_test_mfcc, x_val_mfcc, y_train, y_test, y_val, keywords

##################################################
# plottery
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
# for multiple possible keywords
##################################################

# load data
keywords = ['cat','marvin','left','zero']
coldwords=['bed','bird','stop','visual']
noise=['_background_noise_']
x_train_mfcc, x_test_mfcc, x_val_mfcc, y_train, y_test, y_val, keywords = load_data(keywords, coldwords, noise)


print('x train shape: ', x_train_mfcc.shape)
print('x test shape: ', x_test_mfcc.shape)
print('y train shape: ', y_train.shape)
print('y test shape: ', y_test.shape)

if sys.argv[1] == 'train':
  # build model
  model = get_model(inp_shape=x_train_mfcc.shape[1:], num_classes = len(keywords))

  # train model
  model.summary()
  train(model, x_train_mfcc, y_train, x_val_mfcc, y_val, batchSize = batchSize, epochs = epochs)

  # store model
  dte = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
  fname = cache_dir+'/kws_model_'+model_arch+'_'+dte+'.h5'
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



