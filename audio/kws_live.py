
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import simpleaudio as sa
import threading
from time import sleep

import tensorflow as tf
import mfcc_utils as mfu

import sys
if len(sys.argv) < 2:
  print('Usage:')
  print('  kws_live.py <mode>')
  print('    Modes:')
  print('    host                   Run live inference on host')
  print('    mcu                    Run live inference on mcu')
  exit()

cache_dir = '.cache/kws_mcu'
# model_file = 'train/.cache/kws_keras/kws_model_medium_embedding_conv.h5'
model_file = 'train/.cache/kws_keras/kws_model_nnom.h5'

keywords = np.load('train/.cache/kws_keras/keywords.npy')
threshold = 0.6

# Load trained model
model = tf.keras.models.load_model(model_file)
model.summary()

input_shape = model.input.shape.as_list()[1:]
input_size = np.prod(input_shape)
output_shape = model.output.shape.as_list()[1:]
output_size = np.prod(output_shape)

# Settings
fs = 16000
sample_len_seconds = 2
sample_len = sample_len_seconds*fs
mel_mtx_scale = 128
lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 32
frame_length = 1024
num_mfcc = 13
nSamples = int(sample_len_seconds*fs)
frame_len = frame_length
frame_step = frame_len
frame_count = 0 # 0 for auto
fft_len = frame_len
n_frames = 1 + (sample_len - frame_length) // frame_step

# data buffer
xdata = [0]
ydata = [np.zeros((1,output_size))]
mic_data = []

# abort when
abort_after = 10

# mfcc_fun = mfu.mfcc_mcu
mfcc_fun = mfu.mfcc
# mfcc_fun = mfu.mfcc_tf

######################################################################
# Host 
######################################################################
def kwsHostThd(xdata, ydata, mic_data):
  global abort_after

  net_input = np.array([], dtype='int16')
  init = 1
  frame_ctr = 0
  mfcc = np.array([])
  last_pred = 0

  import sounddevice as sd
  with sd.Stream(samplerate=fs, channels=1) as stream:
    print('Filling buffer...')
    while True:
      frame, overflowed = stream.read(frame_length)
      # print('read frame of size', len(frame), 'and type', type(frame), 'overflow', overflowed)
      frame = ((2**16/2-1)*frame[:,0]).astype('int16')
      frame_ctr += 1
      mic_data.append(frame)

      nSamples = 1024
      o_mfcc = mfcc_fun(frame, fs, nSamples, frame_len, frame_step, frame_count, fft_len, 
        num_mel_bins, lower_edge_hertz, upper_edge_hertz, mel_mtx_scale)
      data_mfcc = np.array([x['mfcc'][:num_mfcc] for x in o_mfcc])

      if init == 1:
        net_input = np.append(net_input.ravel(), data_mfcc)
        if (frame_ctr >= n_frames):
          print('Live!')
          init = 0

      else:
        net_input = np.array(net_input.reshape([1]+input_shape), dtype='float32')
        host_pred = model.predict(net_input)[0]
        net_input = np.append(data_mfcc, net_input.ravel()[:-num_mfcc])

        if (host_pred.max() > threshold):
          spotted_kwd = keywords[np.argmax(host_pred)]
          if spotted_kwd[0] != '_':
            print('Spotted', spotted_kwd)
        np.set_printoptions(suppress=True)
        # print(host_pred)
        last_pred = host_pred
        
        host_mfcc = net_input.reshape(n_frames,num_mfcc)

        # plotting
        xdata.append(xdata[-1] + frame_length/fs)
        ydata.append( np.array(host_pred).reshape(1,output_size) )

        abort_after -= 1
        if abort_after == 0:
          return


######################################################################
# MCU
######################################################################

def kwsMCUThd(xdata, ydata):
  global abort_after

  init = 1
  frame_ctr = 0
  last_pred = 0

  import mcu_util as mcu

  if mcu.sendCommand('kws_mic_continuous') < 0:
    print('MCU error')
    exit()

  while True:

    net_out, ampl, likely, spotted = mcu.getSingleLiveInference()
    print(net_out)

    # plotting
    xdata.append(xdata[-1] + frame_length/fs)
    ydata.append( np.array(net_out).reshape(1,output_size) )

    abort_after -= 1
    if abort_after == 0:
      mcu.write(b'0')
      return

######################################################################
# Plot animation
######################################################################

# fig = plt.figure()
# ax = plt.axes(xlim=(0, 100), ylim=(0, 1))
# nlines = output_size
# lines = []

# for n in range(nlines):
#   line, = ax.plot([], [], label=('%d'%n))
#   lines.append(line)
#   ax.grid(True)
#   ax.legend()

# def init():
#     for line in lines:
#       line.set_data([], [])
#     return line,
# def animate(i):
#   global xdata, ydata
#   n = 0
#   for line in lines:
#     line.set_data(xdata, np.array(ydata).reshape(-1,output_size)[:,n])
#     n += 1
#   return line,

######################################################################
# Plot after
######################################################################
def plotNetOutputHistory():
  global xdata, ydata, mic_data
  
  fig, ax = plt.subplots(2, 1)
  ax[0].plot(mic_data, label='mic')
  ax[0].legend()
  ax[0].grid(True)

  for n in range(output_size):
    print(keywords[n])
    line, = ax[1].plot(xdata, np.array(ydata).reshape(-1,output_size)[:,n], label=keywords[n].strip('_'))
  
  ax[1].grid(True)
  ax[1].set_xlim((0, xdata[-1]))
  ax[1].set_xlim((0, 1))
  ax[1].legend()


######################################################################
# main
######################################################################
if __name__ == '__main__':
  # Live plotting, slow
  # from matplotlib.animation import FuncAnimation
  # print('output_size', output_size)
  # thd = threading.Thread(target=kwsHostThd, args=(xdata, ydata))
  # thd.start()
  # sleep(2)
  # anim = FuncAnimation(fig, animate, init_func=init, interval=200, blit=True)
  # plt.show()

  import sys

  if sys.argv[1] == 'host':
    # post mortem plot
    print('output_size', output_size)
    kwsHostThd(xdata, ydata, mic_data)

    mic_data = np.array(mic_data).reshape((-1,))

    plotNetOutputHistory()
    plt.show()

  if sys.argv[1] == 'mcu':
    # post mortem plot
    print('output_size', output_size)
    kwsMCUThd(xdata, ydata)

    plotNetOutputHistory()
    plt.show()



