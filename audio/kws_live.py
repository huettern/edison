
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import simpleaudio as sa
import threading

# import keras
import tensorflow as tf

import mfcc_utils as mfu

cache_dir = '.cache/kws_mcu'
model_file = '../firmware/src/ai/cube/kws/kws_model_medium_embedding_conv.h5'


keywords = ['cat','marvin','left','zero']
threshold = 0.99

# Load trained model
model = tf.keras.models.load_model(model_file)
model.summary()

input_shape = model.input.shape.as_list()[1:]
input_size = np.prod(input_shape)

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

def kwsThd(tstvar):
  mic_data = np.array([], dtype='int16')
  net_input = np.array([], dtype='int16')
  init = 1
  frame_ctr = 0
  mfcc = np.array([])
  last_pred = 0

  import sounddevice as sd
  with sd.Stream() as stream:
    print('Filling buffer...')
    while True:
      frame, overflowed = stream.read(frame_length)
      # print('read frame of size', len(frame), 'and type', type(frame), 'overflow', overflowed)
      frame = ((2**16/2-1)*frame[:,0]).astype('int16')
      frame_ctr += 1

      nSamples = 1024
      o_mfcc = mfu.mfcc_mcu(frame, fs, nSamples, frame_len, frame_step, frame_count, fft_len, 
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
          print('Spotted', spotted_kwd)
        np.set_printoptions(suppress=True)
        # print(host_pred)
        last_pred = host_pred
        
        host_mfcc = net_input.reshape(n_frames,num_mfcc)

def animate(i,tstvar, frame):
  ax1.clear()
  print(frame)
  print(tstvar)
  ax1.plot(frame)

# import matplotlib.pyplot as plt 
# import matplotlib.animation as animation
# import numpy as np

# data = np.loadtxt(".cache/example.txt", delimiter=",")
# x = data[:,0]
# y = data[:,1]

# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# frame_line, = ax1.plot([],[], '-')
# ax2 = fig.add_subplot(212)
# line2, = ax2.plot([],[],'--')
# ax1.set_xlim(np.min(x), np.max(x))
# ax1.set_ylim(np.min(y), np.max(y))
# ax2.set_xlim(np.min(x), np.max(x))
# ax2.set_ylim(np.min(y), np.max(y))

# def animate(i,factor):
#     line.set_xdata(x[:i])
#     line.set_ydata(y[:i])
#     line2.set_xdata(x[:i])
#     line2.set_ydata(factor*y[:i])
#     return line,line2

# K = 0.75 # any factor 
# ani = animation.FuncAnimation(fig, animate, frames=len(x), fargs=(K,),
#                               interval=100, blit=True)
# plt.show()

tstvar = 0

thd = threading.Thread(target=kwsThd, args=(tstvar,))
thd.start()