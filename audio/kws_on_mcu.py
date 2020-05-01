# -*- coding: utf-8 -*-
# @Author: Noah Huetter
# @Date:   2020-04-30 14:43:56
# @Last Modified by:   Noah Huetter
# @Last Modified time: 2020-05-01 10:25:22

import sys

if len(sys.argv) < 2:
  print('Usage:')
  print('  kws_on_mcu.py <mode>')
  print('    Modes:')
  print('    single                   Single inference on random data')
  print('    fileinf <file>           Get file, run MFCC on host and inference on MCU')
  exit()
mode = sys.argv[1]
args = sys.argv[2:]

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# import keras
import tensorflow as tf

import mfcc_utils as mfu
import mcu_util as mcu

cache_dir = '.cache/kws_mcu'
model_file = '../firmware/src/ai/cube/kws/kws_model_2020-05-01_09:22:51.h5'

# Load trained model
model = tf.keras.models.load_model(model_file)
model.summary()

input_shape = model.input.shape.as_list()[1:]
input_size = np.prod(input_shape)

# Settings
fs = 16000
sample_len_seconds = 4
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

######################################################################
# functions
######################################################################
def infereOnMCU(net_input, progress=False):
  """
    Upload, process and download inference
  """
  if mcu.sendCommand('kws_single_inference') < 0:
    exit()
  mcu.sendData(net_input.reshape(-1), 0, progress=progress)
  mcu_pred, tag = mcu.receiveData()
  print('Received %s type with tag 0x%x len %d' % (mcu_pred.dtype, tag, mcu_pred.shape[0]))
  return mcu_pred

def rmse(a, b):
  return np.sqrt(np.mean((a-b)**2))

def compare(host_preds, mcu_preds):
  deviaitons = 100.0 * (1.0 - (mcu_preds+1e-9) / (host_preds+1e-9) )

  print('_________________________________________________________________')
  print('Number of inferences run: %d' % (len(host_preds)))
  print("Deviation: max %.3f%% min %.3f%% avg %.3f%% \nrmse %.3f" % (
    deviaitons.max(), deviaitons.min(), np.mean(deviaitons), rmse(mcu_preds, host_preds)))
  print('_________________________________________________________________')


def singleInference(repeat = 1):
  """
    Run a single inference on MCU
  """

  # generate some random data
  np.random.seed(20)

  host_preds = []
  mcu_preds = []
  for i in range(repeat):
    net_input = np.array(np.random.rand(input_size).reshape([1]+input_shape), dtype='float32')

    # predict on CPU
    host_preds.append(model.predict(net_input)[0][0])

    # predict on MCU
    mcu_preds.append(infereOnMCU(net_input))
    rmserror = rmse(host_preds[-1], mcu_preds[-1])
    print('host prediction: %f mcu prediction: %f rmse: %.3f' % (host_preds[-1], mcu_preds[-1], rmserror))
    
  mcu_preds = np.array(mcu_preds)
  host_preds = np.array(host_preds)
  compare(host_preds, mcu_preds)

def fileInference():
  """
    Read file and comput MFCC, launch inference on host and MCU
  """
  host_preds = []
  mcu_preds = []

  in_fs, data = wavfile.read(args[0])

  if (in_fs != fs):
    print('Sample rate of file %d doesn\'t match %d' % (in_fs, fs))
    exit()

  # Cut/pad sample
  if data.shape[0] < sample_len:
    data = np.pad(data, (0, sample_len-data.shape[0]))
  else:
    data = data[:sample_len]

  # Calculate MFCC
  o_mfcc = mfu.mfcc_mcu(data, fs, nSamples, frame_len, frame_step, frame_count, fft_len, 
    num_mel_bins, lower_edge_hertz, upper_edge_hertz, mel_mtx_scale)
  data_mfcc = np.array([x['mfcc'][:num_mfcc] for x in o_mfcc])
  # make fit shape and dtype
  net_input = np.array(data_mfcc.reshape([1]+input_shape), dtype='float32')
  
  # predict on CPU and MCU
  host_preds.append(model.predict(net_input)[0][0])
  mcu_preds.append(infereOnMCU(net_input))

  print('host prediction: %f mcu prediction: %f' % (host_preds[-1], mcu_preds[-1]))

  mcu_preds = np.array(mcu_preds)
  host_preds = np.array(host_preds)
  compare(host_preds, mcu_preds)
  # print('host prediction: %f ' % (host_preds[-1]))

######################################################################
# main
######################################################################
if __name__ == '__main__':
  if mode == 'single':
    if len(args):
      singleInference(int(args[0]))
    else:
      singleInference(1)
  if mode == 'fileinf':
    fileInference()




