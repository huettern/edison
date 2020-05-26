# -*- coding: utf-8 -*-
# @Author: Noah Huetter
# @Date:   2020-04-16 16:23:59
# @Last Modified by:   Noah Huetter
# @Last Modified time: 2020-05-26 16:48:26

import numpy as np
from scipy.fftpack import dct
from tqdm import tqdm


# mel freq. constants -> https://en.wikipedia.org/wiki/Mel_scale
from config import *


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

def batch_mfcc(data, \
  fs, nSamples, frame_len, frame_step, frame_count, \
  fft_len, \
  mel_nbins, mel_lower_hz, mel_upper_hz):
  """
    Runs windowed mfcc on a strem of data
  
      data          input data fo shape [..., samples]
      fs            input sample rate
      nSamples      number of samples in input
      frame_len     length of each frame
      frame_step    how many samples to advance the frame
      frame_count   how many frames to compute
      fft_len       length of FFT, ideally frame_len 
      mel_nbins     number of mel filter banks to create
      mel_lower_hz  lowest frequency of mel bank
      mel_upper_hz  highest frequency of mel bank

  """

  if frame_count == 0:
    frame_count = 1 + (nSamples - frame_len) // frame_step
  print("Running mfcc for %d frames with %d step on %d samples" % (frame_count, frame_step, data.shape[0]))
  # will return a list with a dict for each frame
  output = np.zeros((data.shape[0], frame_count, mel_nbins))

  for sampleCtr in tqdm(range(data.shape[0])):
    for frame_ctr in range(frame_count):      
      # get chunk of data
      chunk = data[sampleCtr][frame_ctr*frame_step : frame_ctr*frame_step+frame_len]

      # calculate FFT
      stfft = np.fft.fft(chunk)[:frame_len//2]
      
      # calcualte spectorgram
      spectrogram = np.abs(stfft)
      num_spectrogram_bins = len(spectrogram)

      # calculate mel weights
      mel_weight_matrix = gen_mel_weight_matrix(num_mel_bins=mel_nbins, 
        num_spectrogram_bins=num_spectrogram_bins, sample_rate=fs,
        lower_edge_hertz=mel_lower_hz, upper_edge_hertz=mel_upper_hz)

      # dot product of spectrum and mel matrix to get mel spectrogram
      mel_spectrogram = np.dot(spectrogram, mel_weight_matrix)
      
      # take log of mel spectrogram
      log_mel_spectrogram = np.log(mel_spectrogram + 1e-6)

      # calculate DCT-II
      mfcc = dct(log_mel_spectrogram, type=2) / np.sqrt(2*mel_nbins)
      frame = np.array(mfcc)

      # Add frame to output list
      output[sampleCtr, frame_ctr, ...] = frame

  return output


def mfcc(data, \
  fs, nSamples, frame_len, frame_step, frame_count, \
  fft_len, \
  mel_nbins, mel_lower_hz, mel_upper_hz, dummy=None):
  """
    Runs windowed mfcc on a strem of data
  
      data          input data
      fs            input sample rate
      nSamples      number of samples in input
      frame_len     length of each frame
      frame_step    how many samples to advance the frame
      frame_count   how many frames to compute
      fft_len       length of FFT, ideally frame_len 
      mel_nbins     number of mel filter banks to create
      mel_lower_hz  lowest frequency of mel bank
      mel_upper_hz  highest frequency of mel bank
  
  """

  if frame_count == 0:
    frame_count = 1 + (nSamples - frame_len) // frame_step
  # print("Running mfcc for %d frames with %d step" % (frame_count, frame_step))
  # will return a list with a dict for each frame
  output = []

  for frame_ctr in range(frame_count):
    frame = {}
    frame['t_start'] = frame_ctr*frame_step/fs
    frame['t_end'] = (frame_ctr*frame_step+frame_len)/fs

    # print("frame %d start %f end %f"%(frame_ctr, frame['t_start'],frame['t_end']))

    # get chunk of data
    chunk = data[frame_ctr*frame_step : frame_ctr*frame_step+frame_len]

    # calculate FFT
    frame['fft'] = np.fft.fft(chunk)[:frame_len//2]
    
    # calcualte spectorgram
    spectrogram = np.abs(frame['fft'])
    frame['spectrogram'] = spectrogram
    num_spectrogram_bins = len(frame['spectrogram'])

    # calculate mel weights
    mel_weight_matrix = gen_mel_weight_matrix(num_mel_bins=mel_nbins, 
      num_spectrogram_bins=num_spectrogram_bins, sample_rate=fs,
      lower_edge_hertz=mel_lower_hz, upper_edge_hertz=mel_upper_hz)
    frame['mel_weight_matrix'] = mel_weight_matrix

    # dot product of spectrum and mel matrix to get mel spectrogram
    mel_spectrogram = np.dot(spectrogram, mel_weight_matrix)
    frame['mel_spectrogram'] = mel_spectrogram
    
    # take log of mel spectrogram
    log_mel_spectrogram = np.log(mel_spectrogram + 1e-6)
    frame['log_mel_spectrogram'] = log_mel_spectrogram

    # calculate DCT-II
    mfcc = dct(log_mel_spectrogram, type=2) / np.sqrt(2*mel_nbins)
    frame['mfcc'] = mfcc

    # Add frame to output list
    output.append(frame)

  return output

def mfcc_tf(data, \
  fs, nSamples, frame_len, frame_step, frame_count, \
  fft_len, \
  mel_nbins, mel_lower_hz, mel_upper_hz, unused=None):
  """
  Calculate same mfcc using tensor flow functions
  """
  import tensorflow as tf
  # sess = tf.InteractiveSession()

  framed = frames(data, frame_length=frame_len, frame_step=frame_step)

  # pack data into a tensor of [1, nFrames, frame_len] so we compute only 1 sample
  tensor = tf.convert_to_tensor(framed.reshape((1,framed.shape[0], framed.shape[1])), dtype=tf.float32)

  # stfts has shape [..., frames, fft_unique_bins], here [1, nFrames, 1, fft_len/2+1)
  stfts = tf.signal.stft(tensor, frame_length=frame_len, frame_step=frame_step, fft_length=fft_len)
  
  spectrograms = tf.abs(stfts)
  # reshape spectrograms to [1, nFrames, fft_len/2+1)
  spectrograms = tf.reshape(spectrograms, (spectrograms.shape[0],spectrograms.shape[1],-1))
  
  num_spectrogram_bins = stfts.shape[-1]
  linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    mel_nbins, num_spectrogram_bins, fs, mel_lower_hz,
    mel_upper_hz)
  # mel_spectrograms has shape [1, nFrames, mel_nbins]
  mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
  # log_mel_spectrograms has shape [1, nFrames, mel_nbins]
  log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
  # mfccs has shape [1, nFrames, mel_nbins]
  mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :mel_nbins]

  # fill in same structure as own mfcc implementation
  # for all spectrograms, cut first element corresponding to the DC component
  if frame_count == 0:
    frame_count = 1 + (nSamples - frame_len) // frame_step
  output = []
  for frame_ctr in range(frame_count):
    frame = {}
    frame['t_start'] = frame_ctr*frame_step/fs
    frame['t_end'] = (frame_ctr*frame_step+frame_len)/fs
    frame['fft'] = tf.reshape(stfts, (stfts.shape[0],stfts.shape[1],-1))[0, frame_ctr, 1:]
    frame['spectrogram'] = spectrograms[0, frame_ctr, 1:].numpy()
    # strip DC component from weights matrix
    frame['mel_weight_matrix'] = linear_to_mel_weight_matrix[1:,...].numpy()
    frame['mel_spectrogram'] = mel_spectrograms[0, frame_ctr, ...]
    frame['log_mel_spectrogram'] = log_mel_spectrograms[0, frame_ctr, ...].numpy()
    frame['mfcc'] = mfccs[0, frame_ctr, ...].numpy()
    output.append(frame)
  return output

def mfcc_mcu(data, \
  fs, nSamples, frame_len, frame_step, frame_count, \
  fft_len, \
  mel_nbins, mel_lower_hz, mel_upper_hz, mel_mtx_scale, use_log=False):
  """
    Runs windowed mfcc on a strem of data, with similar calculation to MCU and scaled to match
    output of MCU
  
      data          input data
      fs            input sample rate
      nSamples      number of samples in input
      frame_len     length of each frame
      frame_step    how many samples to advance the frame
      frame_count   how many frames to compute
      fft_len       length of FFT, ideally frame_len 
      mel_nbins     number of mel filter banks to create
      mel_lower_hz  lowest frequency of mel bank
      mel_upper_hz  highest frequency of mel bank
  
  """

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
    mel_spectrogram = np.dot(spectrogram[:(sample_size//2)+1], mel_weight_matrix)
    mel_spectrogram /= mel_mtx_scale
    frame['mel_spectrogram'] = mel_spectrogram
    
    # log(x) is intentionally left out to safe computation resources
    if use_log:
      mel_spectrogram = np.log(mel_spectrogram+1e-6)

    # calculate DCT-II
    mfcc = 1.0/64*dct(mel_spectrogram, type=2)
    frame['mfcc'] = mfcc

    # Add frame to output list
    output.append(frame)
  return output
def dct2Makhoul(x):
  """
    Calculate DCT-II using N-point FFT as in "A Fast Cosine Transform in O'ne and Two Dimensions" - Makhoul1980
    Source: https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft
  """
  N = x.shape[0]
  k = np.arange(N)

  v = np.empty_like(x)
  v[:(N-1)//2+1] = x[::2]
  if N % 2: # odd length
      v[(N-1)//2+1:] = x[-2::-2]
  else: # even length
      v[(N-1)//2+1:] = x[::-2]
  V = np.fft.fft(v)
  Vr = V * 2 * np.exp(-1j*np.pi*k/(2*N))
  return Vr.real, v, V
