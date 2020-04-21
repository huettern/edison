# -*- coding: utf-8 -*-
# @Author: Noah Huetter
# @Date:   2020-04-20 17:22:06
# @Last Modified by:   Noah Huetter
# @Last Modified time: 2020-04-21 19:27:18

import numpy as np
import matplotlib.pyplot as plt

import mfcc_utils as mfu
import mcu_util as mcu

fname = '.cache/mel_constants.h'
cache_dir = '.cache/mfcc_mcu'
from_files = False

# costant frame size
sample_size = 1024#1024
# From experience, we want 80 mel bins
num_mel_bins = 80#80
# From the FFT, we take only half the spectrum plus the DC component
num_spectrogram_bins = sample_size//2+1
# set the microphone sample rate
sample_rate = 16000
# bound the mel filter banks
lower_edge_hertz = 80.0
upper_edge_hertz = 7600.0
# convert to int16 after scaling up
mel_mtx_scale = 128

######################################################################
# functions
######################################################################

def calcCConstants():
  """
    Calcualte constants used for the C implementation
  """


  # calculate mel matrix
  mel_mtx = mfu.gen_mel_weight_matrix(num_mel_bins=num_mel_bins, num_spectrogram_bins=num_spectrogram_bins, sample_rate=sample_rate, \
      lower_edge_hertz=lower_edge_hertz, upper_edge_hertz=upper_edge_hertz)
  print(type(mel_mtx))
  print('mel matrix shape: %s' % (str(mel_mtx.shape)))
  print('mel matrix bounds: min %f max %f' % (np.amin(mel_mtx), np.amax(mel_mtx)))
  print('mel matrix nonzero elements : %d/%d' % (np.count_nonzero(mel_mtx), np.prod(mel_mtx.shape)) )
  
  # Mel matrix is very sparse, could optimize this..

  mel_mtx_s16 = np.array(mel_mtx_scale*mel_mtx, dtype='int16')
  print('discrete mel matrix shape: %s' % (str(mel_mtx_s16.shape)))
  print('discrete mel matrix bounds: min %f max %f' % (np.amin(mel_mtx_s16), np.amax(mel_mtx_s16)))
  print('discrete mel matrix nonzero elements : %d/%d' % (np.count_nonzero(mel_mtx_s16), np.prod(mel_mtx_s16.shape)) )

  # write mel matrix
  mel_str = 'const int16_t melMtx[%d][%d] = \n' % (mel_mtx_s16.shape[0],mel_mtx_s16.shape[1])
  mel_str += mcu.mtxToC(mel_mtx_s16, prepad=4)
  mel_str += ';'

  f = open(fname, 'w')
  f.write('#define MEL_SAMPLE_SIZE         %5d\n' % sample_size)
  f.write('#define MEL_N_MEL_BINS          %5d\n' % num_mel_bins)
  f.write('#define MEL_N_SPECTROGRAM_BINS  %5d\n' % num_spectrogram_bins)
  f.write('#define MEL_SAMPLE_RATE         %5d\n' % sample_rate)
  f.write('#define MEL_LOWER_EDGE_HZ       %05.3f\n' % lower_edge_hertz)
  f.write('#define MEL_UPPER_EDGE_HZ       %05.3f\n' % upper_edge_hertz)
  f.write('#define MEL_MTX_SCALE           %5d\n\n' % mel_mtx_scale)
  f.write('#define MEL_MTX_ROWS            %5d\n' % mel_mtx_s16.shape[0])
  f.write('#define MEL_MTX_COLS            %5d\n' % mel_mtx_s16.shape[1])
  f.write(mel_str)
  f.close()

######################################################################
# plottery
######################################################################
def plotCompare():
  plt.style.use('seaborn-bright')
  t = np.linspace(0, sample_size/fs, num=sample_size)
  f = np.linspace(0.0, fs/2.0, sample_size//2)
  f2 = np.linspace(-fs/2, fs/2, sample_size)

  fig = plt.figure(constrained_layout=True)
  gs = fig.add_gridspec(3, 2)

  ax = fig.add_subplot(gs[0, :])
  ax.plot(t, y, label='y')
  ax.grid(True)
  ax.legend()
  ax.set_title('input')

  ax = fig.add_subplot(gs[1, 0])
  n = host_fft.shape[0]
  ax.plot(f2, np.real(np.concatenate((host_fft[-n//2:], host_fft[0:n//2]))), label='real')
  ax.plot(f2, np.imag(np.concatenate((host_fft[-n//2:], host_fft[0:n//2]))), label='imag')
  ax.grid(True)
  ax.legend()
  ax.set_title('host FFT')

  ax = fig.add_subplot(gs[1, 1])
  real_part = mcu_fft[0::2]
  ax.plot(f2, np.concatenate((real_part[-n//2:], real_part[0:n//2])), label='real')
  imag_part = mcu_fft[1::2]
  ax.plot(f2, np.concatenate((imag_part[-n//2:], imag_part[0:n//2])), label='imag')
  ax.grid(True)
  ax.legend()
  ax.set_title('MCU FFT')

  ax = fig.add_subplot(gs[2, 0])
  ax.plot(f2, np.concatenate((host_spec[-n//2:], host_spec[0:n//2])), label='y')
  ax.grid(True)
  ax.legend()
  ax.set_title('host spectrum')

  ax = fig.add_subplot(gs[2, 1])
  ax.plot(f2, np.concatenate((mcu_spec[-n//2:], mcu_spec[0:n//2])), label='y')
  ax.grid(True)
  ax.legend()
  ax.set_title('MCU spectrum')

  return fig


######################################################################
# main
######################################################################

# calcCConstants()

# Create synthetic sample
fs = sample_rate
t = np.linspace(0,sample_size/fs, sample_size)
y = np.array(1000*np.cos(2*np.pi*(fs/16)*t)+500*np.cos(2*np.pi*(fs/128)*t), dtype='int16')
# y = np.array((2**15-1)*np.cos(2*np.pi*(fs/80)*t), dtype='int16')
# y = np.array((2**15-1)*np.cos(2*np.pi*(0)*t), dtype='int16')
# y = np.array((2**15-1)*np.cos(2*np.pi*(2*fs/1024)*t), dtype='int16')


if not from_files:
  # Exchange some data
  print('Upload sample')
  mcu.sendData(y, 0)

  print('Download samples')
  mcu_fft, tag = mcu.receiveData()
  print('Received %s type with tag 0x%x len %d' % (mcu_fft.dtype, tag, mcu_fft.shape[0]))
  mcu_spec, tag = mcu.receiveData()
  print('Received %s type with tag 0x%x len %d' % (mcu_spec.dtype, tag, mcu_fft.shape[0]))

  # store this valuable data!
  import pathlib
  pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
  np.save(cache_dir+'/mcu_fft.npy', mcu_fft)
  np.save(cache_dir+'/mcu_spec.npy', mcu_spec)

else:
  mcu_fft = np.load(cache_dir+'/mcu_fft.npy')
  mcu_spec = np.load(cache_dir+'/mcu_spec.npy')


######################################################################
# Same calculations on host
# compensate same bit shift as on MCU
host_fft = np.fft.fft(y) / 2**(9-4)
host_spec = np.abs(host_fft)



######################################################################
# plot

fig = plotCompare()
plt.show()

