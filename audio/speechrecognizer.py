# -*- coding: utf-8 -*-
# @Author: Noah Huetter
# @Date:   2020-04-16 16:59:06
# @Last Modified by:   Noah Huetter
# @Last Modified time: 2020-04-19 11:25:55

import audioutils as au
import mfcc_utils as mfu
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
import os
import pathlib

cache_dir = '.cache'
verbose = 1

##################################################
# Model definition

def get_model(inp_shape, num_classes):
  """
    Build CNN model
  """
  print("Building model with input shape %s" % (inp_shape, ))
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=inp_shape))
  model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
  model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.25))
  model.add(Dense(64, activation='relu'))
  model.add(Dropout(0.4))
  model.add(Dense(num_classes, activation='softmax'))
  model.compile(loss='binary_crossentropy',
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])
  return model

##################################################
# Training
def train(model):
  batchSize = 10
  epochs = 30

  train_set = x_train_mfcc
  train_labels = y_train
  test_set = x_test_mfcc
  test_labels = y_test

  model.fit(train_set, train_labels, batch_size=batchSize, epochs=epochs, 
    verbose=verbose, validation_data=(test_set, test_labels))


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
    x_train, y_train, x_test, y_test = au.load_snips_data(trainsize = 100, testsize = 10)

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



##################################################
# MAIN
##################################################

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
train(model)

model.save(cache_dir+'/mfcc_model.h')