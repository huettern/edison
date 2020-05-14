
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pathlib
from pathlib import Path
from tqdm import tqdm
from scipy.io import wavfile

# import tensorflow as tf

# import tensorflow.keras as keras
# from tensorflow.keras.models import Sequential, load_model, Model
# from tensorflow.keras.layers import *
# from tensorflow.keras.utils import to_categorical

import keras
from keras.models import Sequential, load_model, Model
from keras.layers import *
from keras.utils import to_categorical

from sklearn.metrics import confusion_matrix

use_mfcc_librosa = False
use_mfcc_log = False

# audio and MFCC settings
sample_len_seconds = 2.0
fs = 16000
mel_mtx_scale = 128
lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 32
frame_length = 1024
first_mfcc = 0
num_mfcc = 13
nSamples = int(sample_len_seconds*fs)
frame_step = frame_length
frame_count = 0 # 0 for auto
fft_len = frame_length
n_frames = 1 + (nSamples - frame_length) // frame_step

# mel freq. constants -> https://en.wikipedia.org/wiki/Mel_scale
MEL_HIGH_FREQUENCY_Q = 1127.0
MEL_BREAK_FREQUENCY_HERTZ = 700.0

# training hyperparameters
epochs = 300
batchSize = 100
initial_learningrate = 0.0005
threshold=0.6 # for a true prediction

# storing temporary data and model
cache_dir = '.cache/allinone'
model_name = cache_dir+'/kws_model.h5'

# Where to find data
# data_path = 'train/.cache/speech_commands_v0.02'
data_path = 'acquire/noah'

##################################################
# Model definition
def get_model(inp_shape, num_classes):
  print("Building model with input shape %s and %d classes" % (inp_shape, num_classes))
  
  # first_filter_width = 8
  # first_filter_height = 8
  # first_filter_count = 16
  # first_conv_stride_x = 2
  # first_conv_stride_y = 2

  # inputs = Input(shape=inp_shape)
  # x = Conv2D(first_filter_count, 
  #   kernel_size=(first_filter_width, first_filter_height),
  #   strides=(first_conv_stride_x, first_conv_stride_y),
  #   use_bias=True,
  #   activation='relu', 
  #   padding='same')(inputs)

  # dropout_rate = 0.25
  # x = Dropout(dropout_rate)(x)
  # x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x)

  # second_filter_width = 4
  # second_filter_height = 4
  # second_filter_count = 12
  # second_conv_stride_x = 1
  # second_conv_stride_y = 1

  # x = Conv2D(second_filter_count, 
  #   kernel_size=(second_filter_width, second_filter_height),
  #   strides=(second_conv_stride_x, second_conv_stride_y),
  #   use_bias=True,
  #   activation='relu', 
  #   padding='same' )(x)

  # dropout_rate = 0.25
  # x = Dropout(dropout_rate)(x)

  # x = Flatten()(x)
  # x = Dense(num_classes)(x)
  # predictions = Softmax()(x)
  

  inputs = Input(shape=inp_shape)
  x = Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='valid')(inputs)
  x = BatchNormalization()(x)
  x = ReLU()(x)
  x = MaxPooling2D((2, 1), strides=(2, 1), padding="valid")(x)

  x = Conv2D(32 ,kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
  x = BatchNormalization()(x)
  x = ReLU()(x)
  x = MaxPooling2D((2, 1),strides=(2, 1), padding="valid")(x)

  x = Conv2D(64 ,kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
  x = BatchNormalization()(x)
  x = ReLU()(x)
  #x = MaxPooling2D((2, 1), strides=(2, 1), padding="valid")(x)
  x = Dropout(0.2)(x)

  x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
  x = BatchNormalization()(x)
  x = ReLU()(x)
  x = Dropout(0.3)(x)

  x = Flatten()(x)
  x = Dense(num_classes)(x)

  predictions = Softmax()(x)

  model = Model(inputs=inputs, outputs=predictions)

  opt = keras.optimizers.Adam(learning_rate=initial_learningrate)
  model.compile(optimizer=opt, loss ='categorical_crossentropy', metrics=['accuracy'])
  return model

##################################################
# Model training
def train(model, x, y, vx, vy, batchSize = 10, epochs = 30):
  
  early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
  reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-9)

  train_history = model.fit(x, y, batch_size = batchSize, epochs = epochs, 
    validation_data = (vx, vy), 
    callbacks = [early_stopping, reduce_lr], shuffle=True)

  return train_history

##################################################
# Data loading

def load_speech_commands(keywords=None, coldwords=None, sample_len=2*16000, playsome=False, test_val_size=0.2, noise=0.10):
  """
    Load data from the own recorded set

    X_train, y_train, X_test, y_test, X_val, y_val, keywords = load_speech_commands(keywords=None, sample_len=2*16000, playsome=False, test_val_size=0.2)
  """
  from os import path
  from tqdm import tqdm
  import numpy as np

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
  for i in tqdm(range(len(data_to_use))): 
    fs_in, data = wavfile.read(data_to_use[i])
    if fs_in != fs:
      print('Samplerate mismatch! In',fs_in,'expected',fs)
      exit()
    if data.dtype == 'float32':
      data = ( (2**15-1)*data).astype('int16')
    x = data.copy()
    # Cut/pad sample
    if x.shape[0] < sample_len:
      if len(x) == 0:
        x = np.pad(x, (0, sample_len-x.shape[0]), mode='constant', constant_values=(0, 0))
      else:  
        print('pad len', sample_len-x.shape[0], '//1024', (sample_len-x.shape[0])//1024)
        x = np.pad(x, (0, sample_len-x.shape[0]), mode='edge')
    else:
      cut_cnt += 1
      x = x[:sample_len]
    # add to sample list
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

  return X_train, y_train, X_test, y_test, X_val, y_val, keywords


##################################################
# Feature extraction using MFCC
def frames(data, frame_length=3, frame_step=1):
  """
  Split a data vector into (possibly overlapipng) frames

    frame_length: length of each frame
    frame_step: how many sample to advance the frame each step
  """
  n_frames = 1 + (data.shape[0] - frame_length) // frame_step
  out = np.zeros((n_frames,frame_length))
  for i in range(n_frames):
    out[i] = data[i*frame_step:i*frame_step+frame_length]
  return out

def hertz_to_mel(frequencies_hertz):
  """
  Converts frequencies in `frequencies_hertz` in Hertz to the mel scale.
  """
  return MEL_HIGH_FREQUENCY_Q * np.log(1.0 + (frequencies_hertz / MEL_BREAK_FREQUENCY_HERTZ))

def gen_mel_weight_matrix(num_mel_bins=20, num_spectrogram_bins=129, sample_rate=8000, \
    lower_edge_hertz=125.0, upper_edge_hertz=3800.0):
  """
  Generate mel weight matric from linear frequency spectrum, inspired by 
    https://www.tensorflow.org/api_docs/python/tf/signal/linear_to_mel_weight_matrix

  """
  nyquist_hertz = sample_rate / 2.0
  # excludes DC spectrogram bin
  n_bands_to_zero = 1
  linear_frequencies = np.linspace(0, nyquist_hertz, num_spectrogram_bins)[n_bands_to_zero:]
  # convert linear frequency vector to mel scale
  spectrogram_bins_mel = np.expand_dims( hertz_to_mel(linear_frequencies), 1)
  
  # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
  # center of each band is the lower and upper edge of the adjacent bands.
  # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
  # num_mel_bins + 2 pieces.
  band_edges_mel = frames(
    np.linspace(hertz_to_mel(lower_edge_hertz), hertz_to_mel(upper_edge_hertz), num_mel_bins + 2),
    frame_length=3, frame_step=1)
  
  # Split the triples up and reshape them into [1, num_mel_bins] vectors, one vector for
  # lower edge, one for center and one for uppers
  lower_edge_mel, center_mel, upper_edge_mel = tuple(np.reshape( t, [1, num_mel_bins] ) for t in np.split(band_edges_mel, 3, axis=1))
  
  # Calculate lower and upper slopes for every spectrogram bin. Line segments are 
  # linear in the mel domain, not Hertz.
  lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
    center_mel - lower_edge_mel)
  upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
    upper_edge_mel - center_mel)
  
  # Intersect the line segments with each other and zero.
  mel_weights_matrix = np.maximum(0, np.minimum(lower_slopes, upper_slopes))
  
  # Re-add the zeroed lower bins we sliced out above
  return np.pad(mel_weights_matrix, [[n_bands_to_zero, 0], [0, 0]])

def mfcc_mcu(data, \
  fs, nSamples, frame_len, frame_step, frame_count, \
  fft_len, \
  mel_nbins, mel_lower_hz, mel_upper_hz, mel_mtx_scale):
  """
    Runs windowed mfcc on a strem of data, with similar calculation to MCU and scaled to match
    output of MCU
  """
  from scipy.fftpack import dct

  if use_mfcc_librosa:
    import librosa
    mfcc = librosa.feature.mfcc(y=data.astype('float32')/data.max(), sr=fs, n_mfcc=mel_nbins, hop_length=frame_len, dct_type=2, norm='ortho', lifter=0)
    # squash into expected output fmt
    output = []
    # print(mfcc.shape)
    for frame in mfcc.T:
      el = {}
      el['mfcc'] = frame.copy()
      output.append(el)
    # librosa somehow outputs one frame more than I do
    return output[:-1]

  # Calculate number of frames
  if frame_count == 0:
    frame_count = 1 + (nSamples - frame_len) // frame_step
  output = []
  
  # calculate mel matrix
  mel_weight_matrix = mel_mtx_scale*gen_mel_weight_matrix(num_mel_bins=mel_nbins, 
    num_spectrogram_bins=frame_len//2+1, sample_rate=fs,
    lower_edge_hertz=mel_lower_hz, upper_edge_hertz=mel_upper_hz)

  # Iterate over each frame of data
  for frame_ctr in range(frame_count):
    frame = {}
    frame['t_start'] = frame_ctr*frame_step/fs
    frame['t_end'] = (frame_ctr*frame_step+frame_len)/fs

    # get chunk of data
    chunk = np.array(data[frame_ctr*frame_step : frame_ctr*frame_step+frame_len])
    sample_size = chunk.shape[0]

    # calculate FFT
    frame['fft'] = 1.0/1024*np.fft.fft(chunk)
    
    # calcualte spectorgram
    spectrogram = 1.0/np.sqrt(2)*np.abs(frame['fft'])
    frame['spectrogram'] = spectrogram
    num_spectrogram_bins = len(frame['spectrogram'])

    # calculate mel weights
    frame['mel_weight_matrix'] = mel_weight_matrix

    # dot product of spectrum and mel matrix to get mel spectrogram
    mel_spectrogram = 2.0*np.dot(spectrogram[:(sample_size//2)+1], mel_weight_matrix)
    frame['mel_spectrogram'] = mel_spectrogram
    
    # log(x) is intentionally left out to safe computation resources
    if use_mfcc_log:
      mel_spectrogram = np.log(mel_spectrogram+1e-6)

    # calculate DCT-II
    mfcc = 1.0/64*dct(mel_spectrogram, type=2)
    frame['mfcc'] = mfcc

    # Add frame to output list
    output.append(frame)
  return output


##################################################
# load data
def load_data(keywords, coldwords, noise, playsome=False):

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
    x_train, y_train, x_test, y_test, x_validation, y_val, keywords = load_speech_commands(
      keywords=keywords, coldwords=coldwords, sample_len=nSamples, playsome=playsome, test_val_size=0.2, noise=noise)
    
    # calculate MFCCs of training and test x data
    o_mfcc_train = []
    o_mfcc_test = []
    o_mfcc_val = []
    print('starting mfcc calculation')
    mfcc_fun = mfcc_mcu
    for data in tqdm(x_train):
      o_mfcc = mfcc_fun(data, fs, nSamples, frame_length, frame_step, frame_count, fft_len, 
        num_mel_bins, lower_edge_hertz, upper_edge_hertz, mel_mtx_scale)
      o_mfcc_train.append([x['mfcc'][first_mfcc:first_mfcc+num_mfcc] for x in o_mfcc])
    for data in tqdm(x_test):
      o_mfcc = mfcc_fun(data, fs, nSamples, frame_length, frame_step, frame_count, fft_len, 
        num_mel_bins, lower_edge_hertz, upper_edge_hertz, mel_mtx_scale)
      o_mfcc_test.append([x['mfcc'][first_mfcc:first_mfcc+num_mfcc] for x in o_mfcc])
    for data in tqdm(x_validation):
      o_mfcc = mfcc_fun(data, fs, nSamples, frame_length, frame_step, frame_count, fft_len, 
        num_mel_bins, lower_edge_hertz, upper_edge_hertz, mel_mtx_scale)
      o_mfcc_val.append([x['mfcc'][first_mfcc:first_mfcc+num_mfcc] for x in o_mfcc])

    # add dimension to get (x, y, 1) from to make conv2D input layer happy
    x_train = np.expand_dims(np.array(o_mfcc_train), axis = -1)
    x_test = np.expand_dims(np.array(o_mfcc_test), axis = -1)
    x_val = np.expand_dims(np.array(o_mfcc_val), axis = -1)

    # convert labels to categorial one-hot coded
    y_train = to_categorical(y_train, num_classes=None)
    y_test = to_categorical(y_test, num_classes=None)
    y_val = to_categorical(y_val, num_classes=None)

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



def predictWithConfMatrix(x,y):

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


######################################################################
# Operating mode mic
def micInference(model, keywords, abort_after=1):
  import sounddevice as sd
  
  input_shape = model.input.shape.as_list()[1:]
  mic_data = []
  net_input = np.array([], dtype='int16')
  init = 1
  frame_ctr = 0
  mfcc = np.array([])
  last_pred = 0
  host_preds = []
  
  with sd.Stream(samplerate=fs, channels=1) as stream:
    print('Filling buffer...')
    while True:
      frame, overflowed = stream.read(frame_length)
      frame = ((2**15-1)*frame[:,0]).astype('int16')
      frame_ctr += 1

      # keep track of the last few mic sampels
      mic_data.append(frame)
      if len(mic_data) > nSamples:
        mic_data = mic_data[:nSamples]

      chunk_size = frame_length
      o_mfcc = mfcc_mcu(frame, fs, chunk_size, frame_length, frame_step, frame_count, fft_len, 
        num_mel_bins, lower_edge_hertz, upper_edge_hertz, mel_mtx_scale)
      data_mfcc = np.array([x['mfcc'][first_mfcc:first_mfcc+num_mfcc] for x in o_mfcc])

      if init == 1:
        # append to net input matrix
        if len(net_input) == 0:
          net_input = data_mfcc
        else:  
          #net_input[1:,:]
          net_input = np.r_[net_input, data_mfcc]
          #net_input = np.r_[data_mfcc, net_input]
        if (frame_ctr >= n_frames):
          print('Live!')
          init = 0

      else:
        # predict
        net_input_c = np.array(net_input.reshape([1]+input_shape), dtype='float32')
        host_pred = model.predict(net_input_c)[0]
        host_preds.append(host_pred)

        # remove old data
        #net_input = net_input[:-1,:]
        net_input = net_input[1:,:]
        # append new data
        net_input = np.r_[net_input, data_mfcc]
        #net_input = np.r_[data_mfcc, net_input]
        #print("net_input shape for live:" +str(net_input.shape))
        if (host_pred.max() > threshold):
          spotted_kwd = keywords[np.argmax(host_pred)]
          if spotted_kwd[0] != '_':
            print('Spotted', spotted_kwd, 'with', int(100*host_pred.max()),'% probability')
        np.set_printoptions(suppress=True)
        # print(host_pred)

        abort_after -= 1
        if abort_after == 0:
          # import simpleaudio as sa
          # play_obj = sa.play_buffer(mic_data, 1, 2, fs) # data, n channels, bytes per sample, fs
          # play_obj.wait_done()

          net_input = np.array(net_input.reshape((input_shape[0],-1)), dtype='float32')
          return net_input, np.array(mic_data).ravel(), np.array(host_preds)

def infereFromFile(model, fname):
  fs_in, data = wavfile.read(fname)
  print('Got',len(data),'samples with fs',fs_in)
  input_shape = model.input.shape.as_list()[1:]

  if data.dtype == 'float32':
    data = ( (2**15-1)*data).astype('int16')
  x = data.copy()
  # Cut/pad sample
  if x.shape[0] < nSamples:
      x = np.pad(x, (0, nSamples-x.shape[0]), mode='edge')
  else:
    x = x[:nSamples]

  import simpleaudio as sa
  play_obj = sa.play_buffer(x, 1, 2, fs) # data, n channels, bytes per sample, fs
  play_obj.wait_done()

  # calc mfcc
  o_mfcc = mfcc_mcu(x, fs, nSamples, frame_length, frame_step, frame_count, fft_len, 
      num_mel_bins, lower_edge_hertz, upper_edge_hertz, mel_mtx_scale)
  o_mfcc_val = np.array([x['mfcc'][first_mfcc:first_mfcc+num_mfcc] for x in o_mfcc])

  net_input = np.array(o_mfcc_val.reshape([1]+input_shape), dtype='float32')
  host_pred = model.predict(net_input)[0]

  if (host_pred.max() > threshold):
    spotted_kwd = keywords[np.argmax(host_pred)]
    if spotted_kwd[0] != '_':
      print('Spotted', spotted_kwd, 'with', int(100*host_pred.max()),'% probability')
  np.set_printoptions(suppress=True)
  print('net output:', host_pred)

  net_input = np.array(net_input.reshape((input_shape[0],-1)), dtype='float32')
  return net_input, np.array(x).ravel(), np.array(host_pred)

######################################################################
# Some plot functions
def plotMfcc(keywords):
  fig = plt.figure(constrained_layout=True)
  gs = fig.add_gridspec(len(keywords), 2)
  fig.suptitle('Audio samples used during training', fontsize=16)

  # cant plot noise because it is generated in this script and not available as data
  keywords = np.delete(keywords,np.where(keywords=='_noise'))
  keywords = np.delete(keywords,np.where(keywords=='_cold'))

  i = 0
  for k in keywords:
    # Load single audio sample
    all_data = [str(x) for x in list(Path(data_path+'/'+k+'/').rglob("*.wav"))]
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
    o_mfcc = mfcc_mcu(x, fs, nSamples, frame_length, frame_step, frame_count, fft_len, 
        num_mel_bins, lower_edge_hertz, upper_edge_hertz, mel_mtx_scale)
    o_mfcc_val = np.array([x['mfcc'][first_mfcc:first_mfcc+num_mfcc] for x in o_mfcc])

    t = np.linspace(0, nSamples/fs, num=nSamples)
    frames = np.arange(o_mfcc_val.shape[0])
    melbin = np.arange(o_mfcc_val.shape[1])

    ax = fig.add_subplot(gs[i, 0])
    ax.plot(t, x, label=k)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('time [s]')
    ax.set_ylabel('amplitude')

    ax = fig.add_subplot(gs[i, 1])
    c = ax.pcolor(frames, melbin, o_mfcc_val.T, cmap='viridis')
    ax.grid(True)
    ax.set_title('MFCC')
    ax.set_xlabel('frame')
    ax.set_ylabel('Mel bin')
    fig.colorbar(c, ax=ax)

    i += 1

def plotMfccFromData(mic_data, net_input):
  fig = plt.figure(constrained_layout=True)
  gs = fig.add_gridspec(1, 2)
  fig.suptitle('Last '+str(frame_length)+' MIC samples and their MFCC', fontsize=16)

  t = np.linspace(0, len(mic_data)/fs, num=len(mic_data))
  frames = np.arange(net_input.shape[0])
  melbin = np.arange(net_input.shape[1])

  ax = fig.add_subplot(gs[0, 0])
  ax.plot(t, mic_data, label='mic')
  ax.grid(True)
  ax.legend()
  ax.set_xlabel('time [s]')
  ax.set_ylabel('amplitude')

  ax = fig.add_subplot(gs[0, 1])
  c = ax.pcolor(frames, melbin, net_input.T, cmap='viridis')
  ax.grid(True)
  ax.set_title('MFCC')
  ax.set_xlabel('frame')
  ax.set_ylabel('Mel bin')
  fig.colorbar(c, ax=ax)

def plotPredictions(keywords, mic_data, preds):
  keywords = [x.replace('_','') for x in keywords]
  fig = plt.figure(constrained_layout=True)
  fig.suptitle('Net output wrt. time', fontsize=16)
  gs = fig.add_gridspec(2, 1)

  t = np.linspace(0, len(mic_data)/fs, num=len(mic_data))
  ax = fig.add_subplot(gs[0, 0])
  ax.plot(t, mic_data, label='mic')
  ax.grid(True)
  ax.legend()
  ax.set_title('microphone data')
  ax.set_xlabel('time')
  ax.set_ylabel('amplitude')

  ax = fig.add_subplot(gs[1, 0])
  ax.plot(preds)
  ax.grid(True)
  ax.legend(keywords)
  ax.set_title('Predictions')
  ax.set_xlabel('frame')
  ax.set_ylabel('net output')

def createNnomWeights(model, x_test):
  from nnom_utils import generate_model

  generate_model(model, x_test, name=cache_dir+'/weights.h')


######################################################################
# main
######################################################################
if __name__ == '__main__':
  # load data
  # keywords, noise = ['edison', 'cinema', 'on', 'off', '_cold_word'], 0.1 # keywords, coldwords and noise
  
  # own set, keywords only
  keywords, coldwords, noise = ['edison', 'cinema', 'on', 'off'], ['_cold_word'], 0.1
  
  # for speech commands data set
  # keywords, coldwords, noise = ['marvin', 'zero', 'cat', 'left'], ['sheila', 'seven', 'up', 'right'], 0.1
  
  x_train, x_test, x_val, y_train, y_test, y_val, keywords = load_data(keywords, coldwords, noise, playsome=False)
  print('Received keywords:',keywords)

  print('x train shape: ', x_train.shape)
  print('x test shape: ', x_test.shape)
  print('x validation shape: ', x_val.shape)
  print('y train shape: ', y_train.shape)
  print('y test shape: ', y_test.shape)
  print('y validation shape: ', y_val.shape)

  if sys.argv[1] == 'train':
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

  # fig, axs = plotSomeMfcc(x_train, x_test, y_train, y_test, keywords)
  # plt.show()
  # exit()
  print('Received keywords:',keywords)

  if sys.argv[1] == 'test':
    print('Performance on train data')
    predictWithConfMatrix(x_train,y_train)
    print('Performance on test data')
    predictWithConfMatrix(x_test,y_test)
    print('Performance on val data')
    predictWithConfMatrix(x_val,y_val)
  if sys.argv[1] == 'plot':
    plotMfcc(keywords)
    plt.show()
  if sys.argv[1] == 'mic':
    net_input, mic_data, host_preds = micInference(model, keywords, abort_after=0)
    plotMfccFromData(mic_data, net_input)
    plotMfcc(keywords)
    plotPredictions(keywords, mic_data, host_preds)
    plt.show()
  if sys.argv[1] == 'file':
    net_input, mic_data, pred = infereFromFile(model, sys.argv[2])
    plotMfccFromData(mic_data, net_input)
    plotMfcc(keywords)
    plt.show()
  if sys.argv[1] == 'nnom':
    from nnom_utils import generate_model
    createNnomWeights(model, x_test)
    with open(cache_dir+'/keywords.txt','w') as fd:
      fd.write(np.array2string(keywords).replace('\'','\"').replace('[','').replace(']','').replace(' ',', '))




