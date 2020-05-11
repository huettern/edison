import numpy as np
from time import sleep
import struct
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.io import wavfile
from scipy.fftpack import dct
from tqdm import tqdm

import mfcc_utils as mfu


# Input wav file to use
in_wav = 'data/heysnips_true_16k_16b.wav'


######################################################################
# Plottery
######################################################################
def plotFrame(frame, title=None):
  plt.style.use('seaborn-bright')
  t = np.linspace(0, nSamples/fs, num=nSamples)
  f = np.linspace(0.0, fs/2.0, fft_len//2)
  fig, axs = plt.subplots(2, 2)
  if title:
    fig.suptitle(title, fontsize=16)

  # Input
  axs[0,0].plot(t, in_data, label='input')
  origin = (frame['t_start'],-30000)
  width = frame['t_end']-frame['t_start']
  height = 60000
  rect = patches.Rectangle(origin,width,height,linewidth=0,edgecolor='r',facecolor='r')
  axs[0,0].add_patch(rect)
  axs[0,0].set_xlabel('time [s]')
  axs[0,0].set_ylabel('amplitude')
  axs[0,0].set_xlim(0,nSamples/fs)
  axs[0,0].set_ylim(-25000,25000)
  axs[0,0].grid(True)
  axs[0,0].legend()

  # Spectrum
  axs[1,0].plot(f, frame['spectrogram'], label='input')
  axs[1,0].set_xlabel('frequency [Hz]')
  axs[1,0].set_ylabel('spectrogram')
  axs[1,0].set_xlim(10,fs/2)
  axs[1,0].set_ylim(0,1e6)
  axs[1,0].grid(True)
  axs[1,0].legend()

  # Mel coefficients
  spectrogram_bins_mel = np.expand_dims( mfu.hertz_to_mel(f), 1)
  axs[0,1].plot(f, frame['spectrogram']/np.max(frame['spectrogram']), color='black', label='input')
  axs[0,1].plot(f, frame['mel_weight_matrix'], label='mel filters')
  axs[0,1].text(mel_lower_hz, 0.4, 'mel_lower_hz',rotation=90)
  axs[0,1].text(mel_upper_hz, 0.4, 'mel_upper_hz',rotation=90)
  rect = patches.Rectangle((mel_lower_hz,0),mel_upper_hz-mel_lower_hz,1,linewidth=0,edgecolor='r',facecolor='k',alpha=0.2)
  axs[0,1].add_patch(rect)
  axs[0,1].set_xlim(0,fs/2)
  axs[0,1].set_ylim(0,1)
  axs[0,1].set_xlabel('frequency [Hz]')
  axs[0,1].grid(True)
  axs[0,1].legend()

  # Mel log spectrum
  axs[1,1].plot(frame['log_mel_spectrogram'], label='log_mel')
  axs[1,1].plot(frame['mfcc'], 'r', label='dct mfcc')
  # axs[1,1].set_xlim(0,fs/2)
  axs[1,1].set_ylim(0,100)
  axs[1,1].set_xlabel('mel bin')
  axs[1,1].set_ylabel('log_mel_spectrogram')
  axs[1,1].grid(True)
  axs[1,1].legend()
  return fig

# show
def plotAllFrames(o_mfcc):
  """
  plots all frames and stores each plot as png
  """
  for frame in tqdm(range(len(o_mfcc))):
    fig = plotFrame(o_mfcc[frame])
    fig.tight_layout()
    # plt.show()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig('out/figure%03d.png'%frame)
    plt.close()
    fig.clf()
    del fig

def plotShowSingle(frame):
  """
  Plot and show a single frame
  """
  fig = plotFrame(frame)
  fig.tight_layout()
  plt.show()

def plotSpectrogram(o_mfcc):
  """
    Plots the spectrogram of the sample
  """
  plt.style.use('seaborn-bright')
  t = np.linspace(0, nSamples/fs, num=nSamples)
  f = np.linspace(0.0, fs/2.0, fft_len/2)
  fig, axs = plt.subplots(3, 1)

  # Input
  ax=axs[0]
  ax.plot(t, in_data, label='input')
  ax.set_xlabel('time [s]')
  ax.set_ylabel('amplitude')
  ax.set_xlim(0,nSamples/fs)
  ax.set_ylim(-25000,25000)
  ax.grid(True)
  ax.legend()

  # assemble spectrogram data
  spectrogram = np.ndarray((len(o_mfcc), len(o_mfcc[0]['spectrogram'])))
  for i in range(spectrogram.shape[0]):
    spectrogram[i] = np.log(o_mfcc[i]['spectrogram'])
  spectrogram_mel = np.ndarray((len(o_mfcc), len(o_mfcc[0]['mfcc'])))
  for i in range(spectrogram_mel.shape[0]):
    spectrogram_mel[i] = o_mfcc[i]['mfcc']

  ax = axs[1]
  ax.pcolor(np.transpose(spectrogram),cmap='plasma', label='log spectrum')
  ax.set_xlabel('sample')
  ax.set_ylabel('fft bin')
  ax.legend()

  ax = axs[2]
  ax.pcolor(np.transpose(spectrogram_mel),cmap='plasma', label='mel coefficients')
  ax.set_xlabel('time [s]')
  ax.set_ylabel('sample')
  ax.legend()

  
  # axs[1,0] = pcolor(spectrogram)

def plotNetInput(mfcc, titles):
  """
    Plot net input
  """
  frames = np.arange(mfcc.shape[1])
  melbin = np.arange(mfcc.shape[2])

  rows = int(np.ceil(np.sqrt(mfcc.shape[0])))
  cols = int(np.ceil(mfcc.shape[0] / rows))

  print('rows',rows,'cols',cols)

  fig = plt.figure(constrained_layout=True)
  gs = fig.add_gridspec(rows, cols)

  for i in range(mfcc.shape[0]):
    vmin = mfcc[i].T.min()
    vmax = mfcc[i].T.max()
    ax = fig.add_subplot(gs[i//cols, i%cols])
    c = ax.pcolor(frames, melbin, mfcc[i].T, cmap='PuBu', vmin=vmin, vmax=vmax)
    ax.grid(True)
    ax.set_title(titles[i])
    ax.set_xlabel('frame')
    ax.set_ylabel('Mel bin')
    fig.colorbar(c, ax=ax)

  return fig


######################################################################
# Main
######################################################################

# Read data
in_fs, in_data = wavfile.read(in_wav)
in_data = np.array(in_data)

# Set MFCC settings
fs = in_fs
nSamples = len(in_data)
frame_len = 1024
frame_step = 1024
frame_count = 0 # 0 for auto
fft_len = frame_len
mel_nbins = 32
mel_lower_hz = 80
mel_upper_hz = 7600
mel_mtx_scale = 128

# Some info
print("Frame length in seconds = %.3fs" % (frame_len/fs))
print("Number of input samples = %d" % (nSamples))

# calculate mfcc
o_mfcc = mfu.mfcc(in_data, fs, nSamples, frame_len, frame_step, frame_count, fft_len, mel_nbins, mel_lower_hz, mel_upper_hz)
o_mfcc_tf = mfu.mfcc_tf(in_data, fs, nSamples, frame_len, frame_step, frame_count, fft_len, mel_nbins, mel_lower_hz, mel_upper_hz)
o_mfcc_mcu = mfu.mfcc_mcu(in_data, fs, nSamples, frame_len, frame_step, frame_count, fft_len, mel_nbins, mel_lower_hz, mel_upper_hz, mel_mtx_scale)

# plot
# plotAllFrames(o_mfcc)
# plotShowSingle(o_mfcc[10])
# plotSpectrogram(o_mfcc)
# plt.show()

##
# This makes two figures to compare own implementation with tensorflow
##
fig = plotFrame(o_mfcc[5], 'Own implementation')
fig.tight_layout()
fig = plotFrame(o_mfcc_tf[5], 'Tensorflow')
fig.tight_layout()

##
# Make framed MFCC
##
first_mfcc = 0
num_mfcc = 13
mfccs = []
mfccs.append(np.array([x['mfcc'][first_mfcc:first_mfcc+num_mfcc] for x in o_mfcc]))
mfccs.append(np.array([x['mfcc'][first_mfcc:first_mfcc+num_mfcc] for x in o_mfcc_tf]))
mfccs.append(np.array([x['mfcc'][first_mfcc:first_mfcc+num_mfcc] for x in o_mfcc_mcu]))
mfccs.append(np.array([np.log(x['mfcc'][first_mfcc:first_mfcc+num_mfcc]) for x in o_mfcc_mcu]))
mfccs = np.array(mfccs)
print(mfccs.shape)
fig = plotNetInput(mfccs, ['own', 'tf', 'mcu', 'mcu log'])


plt.show()
