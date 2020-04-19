# -*- coding: utf-8 -*-
# @Author: Noah Huetter
# @Date:   2020-04-16 16:59:47
# @Last Modified by:   Noah Huetter
# @Last Modified time: 2020-04-19 10:25:37

snipsDataPath = "/Users/noah/git/mlmcu-project/audio/data/snips/"

def load_snips_data(sample_len=4*16000, trainsize = 1000, testsize = 100):
  """
    Load training and test data from hey snips dataset
    
      sample_len    how long each sample should be. shorter ones get padded, longer are silced
    returns x_train, y_train, x_test, y_test
  """
  import json
  from tqdm import tqdm
  from scipy.io import wavfile
  import numpy as np

  x_train_list = []
  y_train_list = []
  x_test_list = []
  y_test_list = []

  with open(snipsDataPath+'/'+"train.json") as jsonfile:
    traindata = json.load(jsonfile)

  with open(snipsDataPath+'/'+"test.json") as jsonfile:
    testdata = json.load(jsonfile)

  # Length to stuff the signals to, given in seconds
  totalSliceLength = 10 

  # number of training/test sampels to take
  if trainsize < 0:
    trainsize = len(traindata)
  if testsize < 0:
    testsize = len(testdata)

  # get sampling rate from a file, assuming all have same fs
  fs, _ = wavfile.read(snipsDataPath+'/'+traindata[0]['audio_file_path'])

  # segmentLength = 1024 # Number of samples to use per segment

  # sliceLength = int(totalSliceLength * fs / segmentLength)*segmentLength
  # print ('sliceLength=%d' % sliceLength)

  for i in tqdm(range(trainsize)): 
    # Read wavfile
    fs, data = wavfile.read(snipsDataPath+'/'+traindata[i]['audio_file_path'])

    # Get a mutable copy of the wavfile
    _x_train = data.copy()

    # Cut/pad sample
    if _x_train.shape[0] < sample_len:
      _x_train = np.pad(_x_train, (0, sample_len-_x_train.shape[0]))
    else:
      _x_train = _x_train[:sample_len]
    
    # add to sample list
    x_train_list.append(_x_train.astype(np.float32))
    y_train_list.append(traindata[i]['is_hotword'])

  for i in tqdm(range(testsize)): 
    # Read wavfile
    fs, data = wavfile.read(snipsDataPath+'/'+testdata[i]['audio_file_path'])

    # Get a mutable copy of the wavfile
    _x_test = data.copy()

    # Cut/pad sample
    if _x_test.shape[0] < sample_len:
      _x_test = np.pad(_x_test, (0, sample_len-_x_test.shape[0]))
    else:
      _x_test = _x_test[:sample_len]
    
    # add to sample list
    x_test_list.append(_x_test.astype(np.float32))
    y_test_list.append(testdata[i]['is_hotword'])

  x_train = np.asarray(x_train_list)
  y_train = np.asarray(y_train_list)
  x_test = np.asarray(x_test_list)
  y_test = np.asarray(y_test_list)

  return x_train, y_train, x_test, y_test

