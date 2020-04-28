# -*- coding: utf-8 -*-
# @Author: Noah Huetter
# @Date:   2020-04-20 17:22:06
# @Last Modified by:   Noah Huetter
# @Last Modified time: 2020-04-28 18:03:24

import sys

if len(sys.argv) < 2:
  print('Usage:')
  print('  mfcc_on_mcu.py <mode> [from file ]')
  print('    Modes:')
  print('    calc                     Calculate C constants header file')
  print('    single                   Run MFCC on single frame')
  print('    file data/test.wav       Run MFCC on wav file of any length')
  exit()

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from scipy.io import wavfile

import mfcc_utils as mfu
import mcu_util as mcu

fname = '.cache/mel_constants.h'
cache_dir = '.cache/mfcc_mcu'
from_files = 0

mode = sys.argv[1]

from_files = 1 if len(sys.argv) > 2 else from_files

# costant frame size
sample_size = 1024#1024
# From experience, we want 80 mel bins
num_mel_bins = 32#80
# From the FFT, we take only half the spectrum plus the DC component
num_spectrogram_bins = sample_size//2+1
# set the microphone sample rate
sample_rate = 16000
# bound the mel filter banks
lower_edge_hertz = 80.0
upper_edge_hertz = 7600.0
# convert to int16 after scaling up
mel_mtx_scale = 128
# convert to int16 after scaling up of twiddle factors for fast DCT2
mel_twiddle_scale = 128

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
  mel_str += ';\n'

  # calculate LUT for ln(x) for x in [0,32766]
  log_lut = np.array(np.log(np.linspace(1e-6,32766,32767)), dtype='int16')
  log_lug_str = 'const q15_t logLutq15[%d] = \n' % (32767)
  log_lug_str += mcu.vecToC(log_lut, prepad = 4)
  log_lug_str += ';\n'

  # calculate scale vector for fast DCT by Makhoul1980
  N = num_mel_bins
  k = np.arange(N)
  factors = 2 * np.exp(-1j*np.pi*k/(2*N))
  factors_s16_real = np.array(mel_twiddle_scale*factors.real, dtype='int16')
  factors_s16_imag = np.array(mel_twiddle_scale*factors.imag, dtype='int16')
  factors_s16 = np.empty((factors_s16_real.size + factors_s16_imag.size,), dtype='int16')
  factors_s16[0::2] = factors_s16_real
  factors_s16[1::2] = factors_s16_imag
  twiddle_factors_str = 'const q15_t dctTwiddleFactorsq15[%d] = \n' % (2*N)
  twiddle_factors_str += mcu.vecToC(factors_s16, prepad = 4)
  twiddle_factors_str += ';\n'

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
  f.write('#define MEL_LOG_LUT_SIZE        %5d\n' % log_lut.shape[0])
  f.write(log_lug_str)
  f.write('#define MEL_DCT_TWIDDLE_SIZE        %5d\n' % factors_s16.shape[0])
  f.write(twiddle_factors_str)
  f.close()

######################################################################
# plottery
######################################################################
def plotCompare():
  plt.style.use('seaborn-bright')
  t = np.linspace(0, sample_size/fs, num=sample_size)
  f = np.linspace(0.0, fs/2.0, sample_size//2)
  f2 = np.linspace(-fs/2, fs/2, sample_size)
  fmel = np.linspace(0,num_mel_bins,num_mel_bins)

  fig = plt.figure(constrained_layout=True)
  gs = fig.add_gridspec(5, 2)

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
  ax.plot(f2[-mel_mtx.shape[0]:], mel_mtx*(host_spec.max()/mel_mtx.max()), 'k', alpha=0.2, label='')
  ax.plot(f2, np.concatenate((host_spec[-n//2:], host_spec[0:n//2])), label='y')
  ax.grid(True)
  ax.legend()
  ax.set_title('host spectrum')

  ax = fig.add_subplot(gs[2, 1])
  ax.plot(f2, np.concatenate((mcu_spec[-n//2:], mcu_spec[0:n//2])), label='y')
  ax.grid(True)
  ax.legend()
  ax.set_title('MCU spectrum')


  ax = fig.add_subplot(gs[3, 0])
  ax.plot(fmel, host_melspec, label='mel spectrum')
  ax.grid(True)
  ax.legend()
  ax.set_title('host mel spectrum')

  ax = fig.add_subplot(gs[3, 1])
  ax.plot(fmel, mcu_melspec, label='mel spectrum')
  ax.plot(fmel, host_melspec, label='host scaled')
  # ax.plot(fmel, mcu_melspec_manual, label='mel spectrum')

  ax.grid(True)
  ax.legend()
  ax.set_title('MCU mel spectrum')

  ax = fig.add_subplot(gs[4, 0])
  ax.plot(fmel, host_dct, label='mel dct')
  ax.plot(fmel, host_dct_reorder, label='post reorder')
  ax.plot(fmel, host_dct_fft.real, label='post fft r')
  ax.plot(fmel, host_dct_makhoul, 'r--', label='fast dct')
  ax.grid(True)
  ax.legend()
  ax.set_title('host mel DCT-II')

  ax = fig.add_subplot(gs[4, 1])
  ax.plot(fmel, mcu_dct, label='mel dct')
  ax.plot(fmel, host_dct, label='host scaled')
  ax.grid(True)
  ax.legend()
  ax.set_title('MCU mel DCT-II')

  return fig

def plotFileMode():
  plt.style.use('seaborn-bright')
  t = np.linspace(0, nSamples/fs, num=nSamples)
  f = np.linspace(0.0, fs/2.0, sample_size//2)
  f2 = np.linspace(-fs/2, fs/2, sample_size)
  fmel = np.linspace(0,num_mel_bins,num_mel_bins)
  frames = np.arange(mcu_dct.shape[0])
  melbin = np.arange(mcu_dct.shape[1])

  fig = plt.figure(constrained_layout=True)
  gs = fig.add_gridspec(3, 2)

  ax = fig.add_subplot(gs[0, :])
  ax.plot(t, y, label='y')
  ax.grid(True)
  ax.legend()
  ax.set_title('input: '+fname)
  ax.set_xlabel('time [s]')
  ax.set_ylabel('amplitude')

  vmin = 0
  vmax = 0.1e6
  ax = fig.add_subplot(gs[1, 0])
  c = ax.pcolor(frames, f, host_spec.T, cmap='PuBu', vmin=vmin, vmax=vmax)
  ax.grid(True)
  ax.set_title('host spectrogram')
  ax.set_xlabel('frame')
  ax.set_ylabel('frequency [Hz]')
  fig.colorbar(c, ax=ax)

  vmin = 0
  vmax = 200
  ax = fig.add_subplot(gs[1, 1])
  c = ax.pcolor(frames, f, mcu_spec[..., 0:mcu_spec.shape[1]//2].T, cmap='PuBu', vmin=vmin, vmax=vmax)
  ax.grid(True)
  ax.set_title('MCU spectrogram')
  ax.set_xlabel('frame')
  ax.set_ylabel('frequency [Hz]')
  fig.colorbar(c, ax=ax)

  vmin = host_dct.min()
  vmax = 20
  ax = fig.add_subplot(gs[2, 0])
  c = ax.pcolor(frames, melbin, host_dct.T, cmap='PuBu', vmin=vmin, vmax=vmax)
  ax.grid(True)
  ax.set_title('host MFCC')
  ax.set_xlabel('frame')
  ax.set_ylabel('Mel bin')
  fig.colorbar(c, ax=ax)

  vmin = mcu_dct.min()
  vmax = mcu_dct.max()
  ax = fig.add_subplot(gs[2, 1])
  c = ax.pcolor(frames, melbin, mcu_dct.T, cmap='PuBu', vmin=vmin, vmax=vmax)
  ax.grid(True)
  ax.set_title('MCU MFCC')
  ax.set_xlabel('frame')
  ax.set_ylabel('Mel bin')
  fig.colorbar(c, ax=ax)

  return fig

######################################################################
# single
######################################################################
if mode == 'single':
  # Create synthetic sample
  fs = sample_rate
  t = np.linspace(0,sample_size/fs, sample_size)
  y = np.array(1000*np.cos(2*np.pi*(fs/16)*t)+500*np.cos(2*np.pi*(fs/128)*t), dtype='int16')
  # y = np.array((2**15-1)*np.cos(2*np.pi*(fs/80)*t), dtype='int16') # saturating
  # y = np.array((2**15-1)*np.cos(2*np.pi*(0)*t), dtype='int16')
  # y = np.array((2**15-1)*np.cos(2*np.pi*(2*fs/1024)*t), dtype='int16')

  # natural sample
  in_fs, in_data = wavfile.read('data/hey_short_16k.wav')
  in_data = np.pad(in_data, (0,sample_size-in_data.shape[0]), 'constant', constant_values=(4, 6))
  y = in_data

  if not from_files:
    # Exchange some data
    if mcu.sendCommand('mel_one_batch') < 0:
      exit()

    print('Upload sample')
    mcu.sendData(y, 0)

    print('Download samples')
    mcu_fft, tag = mcu.receiveData()
    print('Received %s type with tag 0x%x len %d' % (mcu_fft.dtype, tag, mcu_fft.shape[0]))
    mcu_spec, tag = mcu.receiveData()
    print('Received %s type with tag 0x%x len %d' % (mcu_spec.dtype, tag, mcu_fft.shape[0]))
    mcu_melspec, tag = mcu.receiveData()
    print('Received %s type with tag 0x%x len %d' % (mcu_melspec.dtype, tag, mcu_melspec.shape[0]))
    # mcu_melspec_manual, tag = mcu.receiveData()
    # print('Received %s type with tag 0x%x len %d' % (mcu_melspec_manual.dtype, tag, mcu_melspec_manual.shape[0]))
    mcu_dct, tag = mcu.receiveData()
    print('Received %s type with tag 0x%x len %d' % (mcu_dct.dtype, tag, mcu_dct.shape[0]))

    # store this valuable data!
    import pathlib
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.save(cache_dir+'/mcu_fft.npy', mcu_fft)
    np.save(cache_dir+'/mcu_spec.npy', mcu_spec)
    np.save(cache_dir+'/mcu_melspec.npy', mcu_melspec)
    np.save(cache_dir+'/mcu_dct.npy', mcu_dct)

  else:
    mcu_fft = np.load(cache_dir+'/mcu_fft.npy')
    mcu_spec = np.load(cache_dir+'/mcu_spec.npy')
    mcu_melspec = np.load(cache_dir+'/mcu_melspec.npy')
    mcu_dct = np.load(cache_dir+'/mcu_dct.npy')


  ######################################################################
  # Same calculations on host
  # compensate same bit shift as on MCU
  host_fft = np.fft.fft(y)
  host_spec = np.abs(host_fft)
  mel_mtx = mfu.gen_mel_weight_matrix(num_mel_bins=num_mel_bins, num_spectrogram_bins=num_spectrogram_bins, sample_rate=sample_rate, \
      lower_edge_hertz=lower_edge_hertz, upper_edge_hertz=upper_edge_hertz)
  mel_mtx_s16 = np.array(mel_mtx_scale*mel_mtx, dtype='int16')
  host_melspec = host_spec[:(sample_size//2)+1].dot(mel_mtx_s16)
  host_dct = dct(host_melspec, type=2)
  host_dct_makhoul, host_dct_reorder, host_dct_fft = mfu.dct2Makhoul(host_melspec)

  ######################################################################
  # Print some facts
  scale = np.real(host_fft).max()/mcu_fft[0::2].max()
  print('host/mcu fft scale %f' % (scale) )
  host_fft = host_fft * 1/scale
  scale = host_spec.max()/mcu_spec.max()
  print('host/mcu spectrum scale %f' % (scale) )
  host_spec = host_spec * 1/scale
  scale = host_melspec.max()/mcu_melspec.max()
  print('host/mcu mel spectrum scale %f' % (scale) )
  host_melspec = host_melspec * 1/scale
  scale = host_dct.max()/mcu_dct.max()
  print('host/mcu dct scale %f' % (scale) )
  host_dct = host_dct * 1/scale
  host_dct_makhoul = host_dct_makhoul * 1/scale
  host_dct_reorder = host_dct_reorder * 1/scale
  host_dct_fft = host_dct_fft * 1/scale



  ######################################################################
  # plot

  fig = plotCompare()
  plt.show()

######################################################################
# Calculate C constants
######################################################################
if mode == 'calc':
  calcCConstants()

######################################################################
# File
######################################################################
if mode == 'file':
  if len(sys.argv) < 3:
    print('Specify input file')
    exit()
  fname = sys.argv[2]
  from_files = 1 if len(sys.argv) > 3 else 0
  print('Working with %s' % (fname))

  # Read data
  in_fs, in_data = wavfile.read(fname)
  in_data = np.array(in_data)
  y = in_data

  # Set MFCC settings
  fs = in_fs
  nSamples = len(in_data)
  frame_len = sample_size
  frame_step = 1024
  frame_count = 0 # 0 for auto
  fft_len = sample_size

  # Some info
  print("Frame length in seconds = %.3fs" % (frame_len/fs))
  print("Number of input samples = %d" % (nSamples))

  # calculate mfcc
  o_mfcc = mfu.mfcc(in_data, fs, nSamples, frame_len, frame_step, frame_count, fft_len, num_mel_bins, lower_edge_hertz, upper_edge_hertz)

  host_fft = np.array([x['fft'] for x in o_mfcc])
  host_spec = np.array([x['spectrogram'] for x in o_mfcc])
  host_melspec = np.array([x['mel_spectrogram'] for x in o_mfcc])
  host_dct = np.array([x['mfcc'] for x in o_mfcc])

  # calculate on MCU
  frames = mfu.frames(in_data, frame_length=sample_size, frame_step=frame_step)
  mcu_fft = []
  mcu_spec = []
  mcu_melspec = []
  mcu_dct = []
  frame_ctr = 0
  print('Running on MCU')
  from tqdm import tqdm
  if not from_files:
    for frame in tqdm(frames):
      yf = np.array(frame, dtype='int16')
      # Exchange some data
      if mcu.sendCommand('mel_one_batch') < 0:
        exit()
      # print('Upload sample')
      mcu.sendData(yf, 0, progress=False)
      # print('Download samples')
      dat, tag = mcu.receiveData()
      mcu_fft.append(dat)
      # print('Received %s type with tag 0x%x len %d' % (dat.dtype, tag, dat.shape[0]))
      dat, tag = mcu.receiveData()
      mcu_spec.append(dat)
      # print('Received %s type with tag 0x%x len %d' % (dat.dtype, tag, dat.shape[0]))
      dat, tag = mcu.receiveData()
      mcu_melspec.append(dat)
      # print('Received %s type with tag 0x%x len %d' % (dat.dtype, tag, dat.shape[0]))
      dat, tag = mcu.receiveData()
      mcu_dct.append(dat)
      # print('Received %s type with tag 0x%x len %d' % (dat.dtype, tag, dat.shape[0]))
      frame_ctr += 1

    mcu_fft = np.array(mcu_fft)
    mcu_spec = np.array(mcu_spec)
    mcu_melspec = np.array(mcu_melspec)
    mcu_dct = np.array(mcu_dct)
    import pathlib
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.save(cache_dir+'/mcu_fft_file.npy', mcu_fft)
    np.save(cache_dir+'/mcu_spec_file.npy', mcu_spec)
    np.save(cache_dir+'/mcu_melspec_file.npy', mcu_melspec)
    np.save(cache_dir+'/mcu_dct_file.npy', mcu_dct)

  else:
    mcu_fft = np.load(cache_dir+'/mcu_fft_file.npy')
    mcu_spec = np.load(cache_dir+'/mcu_spec_file.npy')
    mcu_melspec = np.load(cache_dir+'/mcu_melspec_file.npy')
    mcu_dct = np.load(cache_dir+'/mcu_dct_file.npy')

  ######################################################################
  # plot
  print(type(mcu_spec[0]))
  print(mcu_spec.shape)
  fig = plotFileMode()
  plt.show()