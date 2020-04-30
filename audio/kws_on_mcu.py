# -*- coding: utf-8 -*-
# @Author: Noah Huetter
# @Date:   2020-04-30 14:43:56
# @Last Modified by:   Noah Huetter
# @Last Modified time: 2020-04-30 15:28:53

import sys

if len(sys.argv) < 2:
  print('Usage:')
  print('  kws_on_mcu.py <mode>')
  print('    Modes:')
  print('    single                   Single inference on random data')
  exit()
mode = sys.argv[1]

import numpy as np
import matplotlib.pyplot as plt

import keras
import tensorflow as tf

import mfcc_utils as mfu
import mcu_util as mcu

cache_dir = '.cache/kws_mcu'
model_file = '../firmware/src/ai/cube/kws/mfcc_model.h5'

# Load trained model
model = tf.keras.models.load_model(model_file)
model.summary()

input_shape = model.input.shape.as_list()[1:]
input_size = np.prod(input_shape)

######################################################################
# functions
######################################################################
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
    if mcu.sendCommand('kws_single_inference') < 0:
      exit()
    print('Upload sample')
    mcu.sendData(net_input.reshape(-1), 0)
    print('Receive sample')
    mcu_pred, tag = mcu.receiveData()
    mcu_preds.append(mcu_pred[0])
    print('Received %s type with tag 0x%x len %d' % (mcu_pred.dtype, tag, mcu_pred.shape[0]))
    print('host prediction: %f mcu prediction: %f' % (host_preds[-1], mcu_preds[-1]))

  mcu_preds = np.array(mcu_preds)
  host_preds = np.array(host_preds)

  deviaitons = 100.0 * (1.0 - mcu_preds / host_preds)

  print("Deviation: max %.3f min %.3f avg %.3f" % (deviaitons.max(), deviaitons.min(), np.mean(deviaitons)))



######################################################################
# main
######################################################################
if __name__ == '__main__':
  if mode == 'single':
    singleInference(50)