
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
model_file ='.cache/allinone/kws_model.h5'
keywords = np.load('.cache/allinone/keywords.npy')
threshold = 0.6

# Load trained model
model = tf.keras.models.load_model(model_file)
model.summary()

input_shape = model.input.shape.as_list()[1:]
input_size = np.prod(input_shape)
output_shape = model.output.shape.as_list()[1:]
output_size = np.prod(output_shape)

# Settings
from config import *

# data buffer
xdata = [0]
ydata = [np.zeros((1,output_size))]
mic_data = []

# abort when
abort_after = 200

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

  net_outs, ampls, likelys, spotteds = [],[],[],[]
  while True:

    net_out, ampl, likely, spotted = mcu.getSingleLiveInference()
    
    net_outs.append(net_out)
    ampls.append(ampl)
    likelys.append(likely)
    spotteds.append(spotted)
    
    print(net_out)
    if spotted is not None:
      print(spotted)

    # plotting
    xdata.append(xdata[-1] + frame_length/fs)
    ydata.append( np.array(net_out).reshape(1,output_size) )

    abort_after -= 1
    if abort_after == 0:
      mcu.write(b'0')
      return np.array(net_outs), np.array(ampls), np.array(likelys), np.array(spotteds)

def netOutFilt(net_outs, alpha):
  
  netOutFlt = [len(net_outs[0])*[0]]

  for out in net_outs:
    # moving average
    new_el = []

    for i in range(len(out)):
      new_el.append(alpha*netOutFlt[-1][i] + (1.0-alpha)*out[i])

    netOutFlt.append(new_el)

  return np.array(netOutFlt)



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

def plotNetOutputHistory(net_out, title):
  
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)

  i = 0
  for k in keywords:
    a = 0.4 if k in ['_cold', '_noise'] else 1.0
    ax.plot(net_out[:,i], label=k.replace('_',''), alpha=a)
    i += 1

  ax.legend()
  ax.grid(True)
  ax.set_title(title)


######################################################################
# main
######################################################################
if __name__ == '__main__':
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

    # try:
    #   net_outs  = np.load(cache_dir+'/net_outs.npy')
    # except:
    net_outs, ampls, likelys, spotteds = kwsMCUThd(xdata, ydata)
    np.save(cache_dir+'/net_outs.npy', net_outs)

    # print(net_outs)
    plotNetOutputHistory(net_outs, 'raw outs MCU')
    netOutF = netOutFilt(net_outs,0.5)
    plotNetOutputHistory(netOutF, 'MCU out filt 0.5')
    plt.show()



