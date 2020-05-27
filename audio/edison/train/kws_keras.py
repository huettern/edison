# -*- coding: utf-8 -*-
# @Author: Noah Huetter
# @Date:   2020-04-16 16:59:06
# @Last Modified by:   Noah Huetter
# @Last Modified time: 2020-05-27 10:39:11

import edison.audio.audioutils as au
import edison.mfcc.mfcc_utils as mfu

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pathlib
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# import tensorflow.keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Softmax, Input, BatchNormalization, ReLU, MaxPool2D
# from tensorflow.keras.utils import to_categorical

import keras
from keras.models import Sequential, load_model, Model
from keras.layers import *
from keras.utils import to_categorical


import tensorflow as tf
try:
  tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
except:
  pass

from config import *
cache_dir += '/kws_keras/'
speech_data_dir = 'cache/acquire/noah'

verbose = 1

# play some audio samples during first data load
playsome = True

# which model architecture to use
model_arch = 'kws_conv'

# Mfcc settings
mfcc_fun = mfu.mfcc_mcu # use MCU like MFCC calculation
use_mfcc_log = False

# training hyperparameters
epochs = 100
batchSize = 100
initial_learningrate = 0.0005
threshold=0.6 # for a true prediction

################################################################################
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
#
# simple_conv: Total params: 4,846
# Conv2D
# MaxPooling2D
# Dropout
# Flatten
# Dense
# Softmax
# 
# kws_conv: Total params: 43,368
# Stolen from https://github.com/majianjia/nnom/tree/master/examples/keyword_spotting

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
    model.add(tf.keras.layers.Conv2D(first_filter_count, 
      kernel_size=(first_filter_width, first_filter_height),
      strides=(first_conv_stride_x, first_conv_stride_y),
      use_bias=True,
      activation='relu', 
      padding='same', 
      input_shape=inp_shape) )
    
    dropout_rate = 0.25
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))

    second_filter_width = 4
    second_filter_height = 4
    second_filter_count = 12
    second_conv_stride_x = 1
    second_conv_stride_y = 1

    model.add(tf.keras.layers.Conv2D(second_filter_count, 
      kernel_size=(second_filter_width, second_filter_height),
      strides=(second_conv_stride_x, second_conv_stride_y),
      use_bias=True,
      activation='relu', 
      padding='same' ) )
    
    dropout_rate = 0.25
    model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(num_classes))
    model.add(tf.keras.layers.Softmax())
    

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

  if model_arch == 'simple_conv':
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(8, 8), activation='relu', padding='same', input_shape=inp_shape, strides=(1,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Softmax())

  if model_arch == 'kws_conv':
    model = Sequential()

    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='valid', input_shape=inp_shape))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D((2, 1), strides=(2, 1), padding="valid"))

    model.add(Conv2D(32 ,kernel_size=(3, 3), strides=(1, 1), padding="valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D((2, 1),strides=(2, 1), padding="valid"))

    model.add(Conv2D(64 ,kernel_size=(3, 3), strides=(1, 1), padding="valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    #model.add(MaxPooling2D((2, 1), strides=(2, 1), padding="valid"))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(num_classes))

    model.add(Softmax())

  opt = keras.optimizers.Adam(learning_rate=initial_learningrate)
  model.compile(optimizer=opt, loss ='categorical_crossentropy', metrics=['accuracy'])
  
  return model


##################################################
# Training
def train(model, x, y, vx, vy, batchSize = 10, epochs = 30):
  
  early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
  reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-9)

  train_history = model.fit(x, y, batch_size = batchSize, epochs = epochs, 
    validation_data = (vx, vy), 
    callbacks = [early_stopping, reduce_lr], shuffle=True)

  return train_history

##################################################
# load data

def load_data(keywords, coldwords, noise, playsome=False):
  """
    Load data and compute MFCC with scaled and custom implementation as it is done on the MCU
  """
  # if in cache, use it
  try:
    x_train   = np.load(cache_dir+'/x_train.npy')
    x_test    = np.load(cache_dir+'/x_test.npy')
    x_val     = np.load(cache_dir+'/x_val.npy')
    y_train   = np.load(cache_dir+'/y_train.npy')
    y_test    = np.load(cache_dir+'/y_test.npy')
    y_val     = np.load(cache_dir+'/y_val.npy')
    keywords  = np.load(cache_dir+'/keywords.npy')
    print('Load data from cache success!')

  except:
    # failed, load from source wave files
    # failed, load from source wave files
    x_train, y_train, x_test, y_test, x_validation, y_val, keywords = au.load_own_speech_commands(
      speech_data_dir, keywords=keywords, coldwords=coldwords, fs=fs, frame_length=frame_length,
      sample_len=nSamples, playsome=playsome, test_val_size=0.2, noise=noise)
    
    # calculate MFCCs of training and test x data
    o_mfcc_train = []
    o_mfcc_test = []
    o_mfcc_val = []
    print('starting mfcc calculation')
    for data in tqdm(x_train):
      o_mfcc = mfcc_fun(data, fs, nSamples, frame_length, frame_step, frame_count, fft_len, 
        num_mel_bins, lower_edge_hertz, upper_edge_hertz, mel_mtx_scale, use_log=use_mfcc_log)
      o_mfcc_train.append([x['mfcc'][first_mfcc:first_mfcc+num_mfcc] for x in o_mfcc])
    for data in tqdm(x_test):
      o_mfcc = mfcc_fun(data, fs, nSamples, frame_length, frame_step, frame_count, fft_len, 
        num_mel_bins, lower_edge_hertz, upper_edge_hertz, mel_mtx_scale, use_log=use_mfcc_log)
      o_mfcc_test.append([x['mfcc'][first_mfcc:first_mfcc+num_mfcc] for x in o_mfcc])
    for data in tqdm(x_validation):
      o_mfcc = mfcc_fun(data, fs, nSamples, frame_length, frame_step, frame_count, fft_len, 
        num_mel_bins, lower_edge_hertz, upper_edge_hertz, mel_mtx_scale, use_log=use_mfcc_log)
      o_mfcc_val.append([x['mfcc'][first_mfcc:first_mfcc+num_mfcc] for x in o_mfcc])

    # MCU compresses 16bit to 8bit for net input, do this also during training
    o_mfcc_train = np.clip(np.array(o_mfcc_train) * net_input_scale, net_input_clip_min, net_input_clip_max)
    o_mfcc_test = np.clip(np.array(o_mfcc_test) * net_input_scale, net_input_clip_min, net_input_clip_max)
    o_mfcc_val = np.clip(np.array(o_mfcc_val) * net_input_scale, net_input_clip_min, net_input_clip_max)

    # add dimension to get (x, y, 1) from to make conv2D input layer happy
    x_train = np.expand_dims(np.array(o_mfcc_train), axis = -1)
    x_test = np.expand_dims(np.array(o_mfcc_test), axis = -1)
    x_val = np.expand_dims(np.array(o_mfcc_val), axis = -1)

    # convert labels to categorial one-hot coded
    y_train = to_categorical(y_train, num_classes=None)
    y_test = to_categorical(y_test, num_classes=None)
    y_val = to_categorical(y_val, num_classes=None)
  
    # shuffle test data
    per = np.random.permutation(x_test.shape[0])
    x_test = x_test[per, :]
    y_test = y_test[per]
    per = np.random.permutation(x_train.shape[0])
    x_train = x_train[per, :]
    y_train = y_train[per]
    per = np.random.permutation(x_val.shape[0])
    x_val = x_val[per, :]
    y_val = y_val[per]

    # store data
    print('Store mfcc data')
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.save(cache_dir+'/x_train.npy', x_train)
    np.save(cache_dir+'/x_test.npy', x_test)
    np.save(cache_dir+'/x_val.npy', x_val)
    np.save(cache_dir+'/y_train.npy', y_train)
    np.save(cache_dir+'/y_test.npy', y_test)
    np.save(cache_dir+'/y_val.npy', y_val)
    np.save(cache_dir+'/keywords.npy', keywords)


  # return
  return x_train, x_test, x_val, y_train, y_test, y_val, keywords

def predictWithConfMatrix(model, x,y):
  """
    Predict with model and print confusion matrix
  """
  y_pred = model.predict(x)
  y_pred = 1.0*(y_pred > 0.5) 

  print('Confusion matrix:')
  cmtx = confusion_matrix(y.argmax(axis=1), y_pred.argmax(axis=1))
  print(cmtx)
  # true positive
  tp = np.sum(np.diagonal(cmtx))
  # total number of predictions
  tot = np.sum(cmtx)
  print('Correct predicionts: %d/%d (%.2f%%)' % (tp, tot, 100.0/tot*tp))

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

def plotMfcc(keywords):
  import pathlib
  from pathlib import Path
  from scipy.io import wavfile

  fig = plt.figure(constrained_layout=True, figsize=(18,5))
  gs = fig.add_gridspec(len(keywords)//2, 4)
  fig.suptitle('Audio samples used during training', fontsize=16)

  # cant plot noise because it is generated in this script and not available as data
  keywords = np.delete(keywords,np.where(keywords=='_noise'))
  keywords = np.delete(keywords,np.where(keywords=='_cold'))

  mel_xlim = (0,15)
  audio_xlim = (0,1)

  i = 0
  for k in keywords:
    # Load single audio sample
    all_data = [str(x) for x in list(Path(speech_data_dir+'/'+k+'/').rglob("*.wav"))]
    _, data = wavfile.read(all_data[0])
    if data.dtype == 'float32':
      data = ( (2**15-1)*data).astype('int16')
    x = data.copy()
    # Cut/pad sample
    if x.shape[0] < nSamples:
      if len(x) == 0:
        x = np.pad(x, (0, nSamples-x.shape[0]), mode='constant', constant_values=(0, 0))
      else:  
        x = np.pad(x, (0, nSamples-x.shape[0]), mode='edge')
    else:
      x = x[:nSamples]

    # calc mfcc
    o_mfcc = mfu.mfcc_mcu(x, fs, nSamples, frame_length, frame_step, frame_count, fft_len, 
        num_mel_bins, lower_edge_hertz, upper_edge_hertz, mel_mtx_scale)
    o_mfcc_val = np.array([x['mfcc'][first_mfcc:first_mfcc+num_mfcc] for x in o_mfcc])

    t = np.linspace(0, nSamples/fs, num=nSamples)
    frames = np.arange(o_mfcc_val.shape[0])
    melbin = np.arange(o_mfcc_val.shape[1])

    ax = fig.add_subplot(gs[i//2, 0+2*(i%2)])
    ax.plot(t, x, label=k)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('time [s]')
    ax.set_ylabel('amplitude')
    ax.set_xlim(audio_xlim)

    ax = fig.add_subplot(gs[i//2, 1+2*(i%2)])
    c = ax.pcolor(frames, melbin, o_mfcc_val.T, cmap='viridis')
    ax.grid(True)
    ax.set_title('MFCC')
    ax.set_xlabel('frame')
    ax.set_ylabel('Mel bin')
    fig.colorbar(c, ax=ax)
    ax.set_xlim(mel_xlim)

    i += 1


##################################################
# MAIN
# for multiple possible keywords
##################################################
def main(argv):

  if len(argv) < 2:
    print('Usage:')
    print('  kws_keras <mode>')
    print('    Modes:')
    print('    train                     Train model')
    print('    test                      Load model from file and test on it')
    exit()
    
  # load data
  keywords, coldwords, noise = ['edison', 'cinema','bedroom', 'office', 'livingroom','kitchen','on', 'off'], ['_cold_word'], 0.1
 
  x_train, x_test, x_val, y_train, y_test, y_val, keywords = load_data(
    keywords, coldwords, noise, playsome=playsome)

  print('x train shape: ', x_train.shape)
  print('x test shape: ', x_test.shape)
  print('x validation shape: ', x_val.shape)
  print('y train shape: ', y_train.shape)
  print('y test shape: ', y_test.shape)
  print('y validation shape: ', y_val.shape)

  model_name = cache_dir+'/kws_model.h5'


  if argv[1] == 'train':
    # build model
    model = get_model(inp_shape=x_train.shape[1:], num_classes = len(keywords))

    # train model
    model.summary()
    train(model, x_train, y_train, x_val, y_val, batchSize = batchSize, epochs = epochs)

    # store model
    model.save(model_name)
    print('Model saved as %s' % (model_name))

  else:
    # load model
    model = keras.models.load_model(model_name)
    model.summary()
    print('Model loaded %s' % (model_name))

  print('Received keywords:',keywords)
  with open(cache_dir+'/keywords.txt','w') as fd:
    fd.write(np.array2string(keywords).replace('\'','\"').replace('[','').replace(']','').replace(' ',', '))
  print('Wrote keywords to textfile', cache_dir+'/keywords.txt')
  
  if argv[1] == 'test':
    print('Performance on train data')
    predictWithConfMatrix(model, x_train,y_train)
    print('Performance on test data')
    predictWithConfMatrix(model, x_test,y_test)
    print('Performance on val data')
    predictWithConfMatrix(model, x_val,y_val)
  if argv[1] == 'plot':
    plotMfcc(keywords)
    plt.show()


