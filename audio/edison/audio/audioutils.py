# -*- coding: utf-8 -*-
# @Author: Noah Huetter
# @Date:   2020-04-16 16:59:47
# @Last Modified by:   Noah Huetter
# @Last Modified time: 2020-05-26 17:42:07

scDataPath = 'train/.cache/speech_commands_v0.02'
scDownloadURL = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'

def load_own_speech_commands(data_path, keywords=None, coldwords=None, fs=16000, sample_len=2*16000, frame_length=1024, playsome=False, test_val_size=0.2, noise=0.10):
  """
    Load data from the own recorded set

    X_train, y_train, X_test, y_test, X_val, y_val, keywords = load_speech_commands(keywords=None, sample_len=2*16000, playsome=False, test_val_size=0.2)
  """
  from os import path
  from tqdm import tqdm
  import numpy as np
  from pathlib import Path
  from scipy.io import wavfile

  # if directory does not exist
  if not path.exists(data_path):
    print('Folder not found:', data_path)
    return -1

  all_data = [str(x) for x in list(Path(data_path).rglob("*.wav"))]

  data_to_use = []

  # Extract file names to use for keywords
  if keywords is not None:
    print('use only samples that are in keywords')
  else:
    keywords = list(set([x.split('/')[-2] for x in all_data]))
  data_to_use += [x for x in all_data if x.split('/')[-2] in keywords]
  
  # Extract file names to use for coldwords
  if coldwords is not None:
    print('loading coldwords')
    data_to_use += [x for x in all_data if x.split('/')[-2] in coldwords]
    keywords.append('_cold')
  print('Using keywords: ', keywords)

  print('Loading files count:', len(all_data))
  x_list = []
  y_list = []
  cut_cnt = 0
  already_appended = False
  for i in tqdm(range(len(data_to_use))): 
    fs_in, data = wavfile.read(data_to_use[i])
    if fs_in != fs:
      print('Samplerate mismatch! In',fs_in,'expected',fs)
      exit()
    if data.dtype == 'float32':
      data = ( (2**15-1)*data).astype('int16')
    x = data.copy()
    if len(x) == 0:
      print('PAAANIIIIcc')
      continue
    # Cut/pad sample
    already_appended = False
    if x.shape[0] < sample_len:
      # x = np.pad(x, (0, sample_len-x.shape[0]), mode='edge')
      # print(x.shape)
      # shift samples around
      pad_variants = 1 +(sample_len-x.shape[0])//frame_length
      # fig = plt.figure()
      # ax = fig.add_subplot(111)
      for pad_variant_cnt in range(pad_variants):
        prepad = pad_variant_cnt*frame_length
        postpad = sample_len-x.shape[0]-prepad
        chunk = np.pad(x.copy(), (prepad, postpad), mode='constant', constant_values=(0,0))
        # ax.plot(chunk)
        assert chunk.shape[0] == sample_len
        # add to sample list
        already_appended = True
        x_list.append(chunk)
        if data_to_use[i].split('/')[-2] in keywords:
          y_list.append(keywords.index(data_to_use[i].split('/')[-2]))
        else:
          y_list.append(keywords.index('_cold'))
      # plt.show()
    else:
      cut_cnt += 1
      x = x[:sample_len]

    # add to sample list
    if not already_appended:
      assert x.shape[0] == sample_len
      x_list.append(x)
      if data_to_use[i].split('/')[-2] in keywords:
        y_list.append(keywords.index(data_to_use[i].split('/')[-2]))
      else:
        y_list.append(keywords.index('_cold'))

  print('Had to cut',cut_cnt,'samples')

  # add noise to samples
  noise_ampl = 0.01
  if noise > 0:
    keywords.append('_noise')
    for n in range(int(noise*len(x_list))):
      rnd = np.random.normal(0,1,size=sample_len)
      x_list.append( np.array((2**15-1)*noise_ampl*rnd/rnd.max(), dtype='int16') )
      assert x_list[-1].shape[0] == sample_len
      y_list.append(keywords.index('_noise'))

  x = np.asarray(x_list)
  y = np.asarray(y_list)

  print('Splitting into train/test/validation sets')
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_val_size, random_state=42)
  X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

  print("total files=%d trainsize=%d testsize=%d validationsize=%d fs=%.0f" % 
    (len(data_to_use), len(X_train), len(X_test), len(X_val), fs))

  # play some to check
  if playsome:
    import simpleaudio as sa
    import random

    for i in range(10):
      rset = random.choice(['X_train', 'X_test', 'X_val'])
      if rset == 'X_train':
        idx = random.randint(0, len(X_train)-1)
        print('train keyword',keywords[y_train[idx]])
        data = X_train[idx]
      if rset == 'X_test':
        idx = random.randint(0, len(X_test)-1)
        print('test keyword',keywords[y_test[idx]])
        data = X_test[idx]
      if rset == 'X_val':
        idx = random.randint(0, len(X_val)-1)
        print('validation keyword',keywords[y_val[idx]])
        data = X_val[idx]
      play_obj = sa.play_buffer(data, 1, 2, fs) # data, n channels, bytes per sample, fs
      play_obj.wait_done()
  
  print('sample count for train/test/validation')
  for i in range(len(keywords)):
    print('  %-20s %5d %5d %5d' % (keywords[i],np.count_nonzero(y_train==i),np.count_nonzero(y_test==i),np.count_nonzero(y_val==i)))

  print("Returning data: trainsize=%d  testsize=%d  validationsize=%d with keywords" % 
    (X_train.shape[0], X_test.shape[0], X_val.shape[0]))
  print(keywords)

  return X_train, y_train, X_test, y_test, X_val, y_val, np.array(keywords)

def load_speech_commands(keywords = ['cat','marvin','left','zero'], 
  sample_len=2*16000, coldwords=['bed','bird','stop','visual'], noise=['_background_noise_'],
  playsome=False):
  """
    Loads samples from 
    http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

    keywords a list of which keywords to load

    coldwords   specify directories that are used and classified as coldwords
    noise       specify directory used as noise sources
  """
  import os
  import tarfile
  import urllib
  from os import path
  from scipy.io import wavfile
  from tqdm import tqdm
  import numpy as np

  # if directory does not exist
  if not path.exists(scDataPath):
    print('Please download dataset from', scDownloadURL, 'and extract it to', scDataPath)
    return -1

  
  # all files
  from pathlib import Path
  all_data = [str(x) for x in list(Path(scDataPath).rglob("*.wav"))]
  
  # print(all_data)

  with open(scDataPath+'/'+"testing_list.txt") as fd:
    test_data = [scDataPath+'/'+x.strip() for x in fd.readlines()]
  with open(scDataPath+'/'+"validation_list.txt") as fd:
    validation_data = [scDataPath+'/'+x.strip() for x in fd.readlines()]

  print('use only samples that are in keywords')
  all_data = [x for x in all_data if x.split('/')[-2] in keywords]
  test_data = [x for x in test_data if x.split('/')[-2] in keywords]
  validation_data = [x for x in validation_data if x.split('/')[-2] in keywords]

  print('scrap data files that are not in test/validation data')
  train_data = [x for x in all_data if x not in test_data]
  train_data = [x for x in train_data if x not in validation_data]

  fs, _ = wavfile.read(train_data[0])

  # print(train_data)
  # print(test_data)
  # print(validation_data)

  print("Loading data: trainsize=%d  testsize=%d  validationsize=%d fs=%.0f" % 
    (len(train_data), len(test_data), len(validation_data), fs))

  cut_cnt = 0
  def extract(fnames, sample_len):
    x_list = []
    y_list = []
    for i in tqdm(range(len(fnames))): 
      fs, data = wavfile.read(fnames[i])
      x = data.copy()

      # Cut/pad sample
      if x.shape[0] < sample_len:
        x = np.pad(x, (0, sample_len-x.shape[0]), mode='edge')
      else:
        cut_cnt += 1
        x = x[:sample_len]

      # add to sample list
      x_list.append(x)
      y_list.append(keywords.index(fnames[i].split('/')[-2]))
      
    return np.asarray(x_list), np.asarray(y_list)


  # Will store data here
  x_train, y_train = extract(train_data, sample_len)
  x_test, y_test = extract(test_data, sample_len)
  x_validation, y_validation = extract(validation_data, sample_len)

  # Load noise from wav files
  if noise is not None:
    keywords += ['_noise']
    # print('extracting noise')
    x_list = []
    y_list = []
    for noise_folder in noise:
      # list of files used as noise
      noise_data = [str(x) for x in list(Path(scDataPath+'/'+noise_folder).rglob("*.wav"))]

      for fname in noise_data:
        print('working on file',fname)
        # load data
        fs, data = wavfile.read(fname)
        x = data.copy()
        # print('  file shape',data.shape)
        # split noise samples in junks of sample_len
        n_smp = x.shape[0] // sample_len
        # print('  create nchunks ', n_smp)
        for smp in range(n_smp):
          x_list.append(x[smp*sample_len:(1+smp)*sample_len])
          y_list.append(keywords.index('_noise'))

    x_list = np.asarray(x_list)
    y_list = np.asarray(y_list)

    # split into train/test
    from sklearn.model_selection import train_test_split
    xtrain_noise, xtest_noise, ytrain_noise, ytest_noise = train_test_split(x_list, y_list, test_size=0.33, random_state=42)

    # Append noise to train/test/validation sets
    x_train = np.append(x_train, xtrain_noise, axis=0)
    y_train = np.append(y_train, ytrain_noise, axis=0)
    x_test = np.append(x_test, np.array_split(xtest_noise, 2)[0], axis=0)
    y_test = np.append(y_test, np.array_split(ytest_noise, 2)[0], axis=0)
    x_validation = np.append(x_validation, np.array_split(xtest_noise, 2)[1], axis=0)
    y_validation = np.append(y_validation, np.array_split(ytest_noise, 2)[1], axis=0)

    print('Added to train',xtrain_noise.shape[0],
      'test', np.array_split(xtest_noise, 2)[0].shape[0], 
      'and validation', np.array_split(xtest_noise, 2)[1].shape[0], 'noise samples')

  # Load coldwords from wav files
  print('Start loading coldwords')
  if coldwords is not None:
    keywords += ['_cold']
    x_list = []
    y_list = []
    for cold_folder in coldwords:
      # list of files used as noise
      cold_data = [str(x) for x in list(Path(scDataPath+'/'+cold_folder).rglob("*.wav"))]

      for fname in cold_data:
        # load data
        fs, data = wavfile.read(fname)
        x = data.copy()

        # Cut/pad sample
        if x.shape[0] < sample_len:
          x = np.pad(x, (0, sample_len-x.shape[0]), mode='edge')
        else:
          cut_cnt += 1
          x = x[:sample_len]
        # add to sample list
        x_list.append(x)
        y_list.append(keywords.index('_cold'))

    x_list = np.asarray(x_list)
    y_list = np.asarray(y_list)

    # split into train/test
    from sklearn.model_selection import train_test_split
    xtrain_cold, xtest_cold, ytrain_cold, ytest_cold = train_test_split(x_list, y_list, test_size=0.33, random_state=42)

    # Append noise to train/test/validation sets
    x_train = np.append(x_train, xtrain_cold, axis=0)
    y_train = np.append(y_train, ytrain_cold, axis=0)
    x_test = np.append(x_test, np.array_split(xtest_cold, 2)[0], axis=0)
    y_test = np.append(y_test, np.array_split(ytest_cold, 2)[0], axis=0)
    x_validation = np.append(x_validation, np.array_split(xtest_cold, 2)[1], axis=0)
    y_validation = np.append(y_validation, np.array_split(ytest_cold, 2)[1], axis=0)

    print('Added to train',xtrain_cold.shape[0],
      'test', np.array_split(xtest_cold, 2)[0].shape[0], 
      'and validation', np.array_split(xtest_cold, 2)[1].shape[0], 'cold samples')


  # play some to check
  if playsome:
    import simpleaudio as sa
    import random
    i = len(x_train)-1
    print('train keyword',keywords[y_train[i]])
    play_obj = sa.play_buffer(x_train[i], 1, 2, fs) # data, n channels, bytes per sample, fs
    play_obj.wait_done()
    i = len(x_test)-1
    print('test keyword',keywords[y_test[i]])
    play_obj = sa.play_buffer(x_test[i], 1, 2, fs) # data, n channels, bytes per sample, fs
    play_obj.wait_done()
    i = len(x_validation)-1
    print('validation keyword',keywords[y_validation[i]])
    play_obj = sa.play_buffer(x_validation[i], 1, 2, fs) # data, n channels, bytes per sample, fs
    play_obj.wait_done()
    for i in range(5):
      i = random.randint(0, len(x_train)-1)
      print('train keyword',keywords[y_train[i]])
      play_obj = sa.play_buffer(x_train[i], 1, 2, fs) # data, n channels, bytes per sample, fs
      play_obj.wait_done()
      i = random.randint(0, len(x_test)-1)
      print('test keyword',keywords[y_test[i]])
      play_obj = sa.play_buffer(x_test[i], 1, 2, fs) # data, n channels, bytes per sample, fs
      play_obj.wait_done()
      i = random.randint(0, len(x_validation)-1)
      print('validation keyword',keywords[y_validation[i]])
      play_obj = sa.play_buffer(x_validation[i], 1, 2, fs) # data, n channels, bytes per sample, fs
      play_obj.wait_done()
  
  print('Had to cut',cut_cnt,'samples')

  print('sample count for train/test/validation')
  for i in range(len(keywords)):
    print('  ',keywords[i],'counts',np.count_nonzero(y_train==i),np.count_nonzero(y_test==i),np.count_nonzero(y_validation==i))

  print("Returning data: trainsize=%d  testsize=%d  validationsize=%d with keywords" % 
    (x_train.shape[0], x_test.shape[0], x_validation.shape[0]))
  print(keywords)

  return x_train, y_train, x_test, y_test, x_validation, y_validation, keywords


######################################################################
# main
######################################################################
if __name__ == '__main__':
  # load_speech_commands()
  load_own_speech_commands(playsome=True)