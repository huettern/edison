# -*- coding: utf-8 -*-
# @Author: Noah Huetter
# @Date:   2020-05-01 14:43:56
# @Last Modified by:   Noah Huetter
# @Last Modified time: 2020-05-22 11:00:37

import edison.mcu.mcu_util as mcu
from time import sleep
import struct
import numpy as np
from tqdm import tqdm

from config import *

def cmd_version():
  cmd = 'version'
  print('--------------------------------------------------')
  print(' Testing command %s' % (cmd))
  print('--------------------------------------------------')

  if mcu.sendCommand(cmd) < 0:
    print('FAIL')
    return -1
  print(mcu.ser.readline())
  print('SUCCESS')
  return 0
  
def cmd_mic_sample_processed_manual():
  cmd = 'mic_sample_processed_manual'
  print('--------------------------------------------------')
  print(' Testing command %s' % (cmd))
  print('--------------------------------------------------')

  if mcu.sendCommand(cmd) < 0:
    print('FAIL')
    return -1
  print('SUCCESS')
  return 0
  
def cmd_ai_info():
  cmd = 'ai_info'
  print('--------------------------------------------------')
  print(' Testing command %s' % (cmd))
  print('--------------------------------------------------')

  if mcu.sendCommand(cmd) < 0:
    print('FAIL')
    return -1
  print('SUCCESS')
  return 0
  
def cmd_audio_info():
  cmd = 'audio_info'
  print('--------------------------------------------------')
  print(' Testing command %s' % (cmd))
  print('--------------------------------------------------')

  if mcu.sendCommand(cmd) < 0:
    print('FAIL')
    return -1
  print('SUCCESS')
  return 0
  
def cmd_mic_sample():
  cmd = 'mic_sample'
  print('--------------------------------------------------')
  print(' Testing command %s' % (cmd))
  print('--------------------------------------------------')

  if mcu.sendCommand(cmd, b'\0\1') < 0:
    print('FAIL')
    return -1
  print('SUCCESS')
  return 0
  
  
def cmd_mic_sample_preprocessed():
  cmd = 'mic_sample_preprocessed'
  print('--------------------------------------------------')
  print(' Testing command %s' % (cmd))
  print('--------------------------------------------------')

  if mcu.sendCommand(cmd, struct.pack('>BH', 8, 16)) < 0:
    print('FAIL')
    return -1
  print('SUCCESS')
  return 0
  
  
def cmd_mel_one_batch():
  cmd = 'mel_one_batch'
  print('--------------------------------------------------')
  print(' Testing command %s' % (cmd))
  print('--------------------------------------------------')

  if mcu.sendCommand(cmd) < 0:
    print('FAIL')
    return -1

  print('Upload sample')
  sample_size = 1024
  y = np.array(65536*np.random.rand(sample_size)-65536//2, dtype='int16') # random
  mcu.sendData(y, 0)

  print('Download samples')
  mcu_fft, tag = mcu.receiveData()
  print('Received %s type with tag 0x%x len %d' % (mcu_fft.dtype, tag, mcu_fft.shape[0]))
  mcu_spec, tag = mcu.receiveData()
  print('Received %s type with tag 0x%x len %d' % (mcu_spec.dtype, tag, mcu_fft.shape[0]))
  mcu_melspec, tag = mcu.receiveData()
  print('Received %s type with tag 0x%x len %d' % (mcu_melspec.dtype, tag, mcu_melspec.shape[0]))
  mcu_dct, tag = mcu.receiveData()
  print('Received %s type with tag 0x%x len %d' % (mcu_dct.dtype, tag, mcu_dct.shape[0]))

  print('SUCCESS')
  return 0
  
  
  
def cmd_kws_single_inference():
  cmd = 'kws_single_inference'
  print('--------------------------------------------------')
  print(' Testing command %s' % (cmd))
  print('--------------------------------------------------')

  if mcu.sendCommand(cmd) < 0:
    print('FAIL')
    return -1
  
  input_shape = [62, 13]
  input_size = np.prod(input_shape)

  net_input = np.array(np.random.rand(input_size).reshape([1]+input_shape), dtype='float32')
  mcu.sendData(net_input.reshape(-1), 0)
  mcu_pred, tag = mcu.receiveData()
  print('Received %s type with tag 0x%x len %d' % (mcu_pred.dtype, tag, mcu_pred.shape[0]))
  
  print('SUCCESS')
  return 0
  
  
  
def cmd_mfcc_kws_frame():
  cmd = 'mfcc_kws_frame'
  print('--------------------------------------------------')
  print(' Testing command %s' % (cmd))
  print('--------------------------------------------------')

  if mcu.sendCommand(cmd) < 0:
    print('FAIL')
    return -1
  
  input_shape = [62, 13]
  input_size = np.prod(input_shape)

  frame_step = 1024
  n_frames = 62
  print('Sending %d frames' % (n_frames))
  for frame in tqdm(range(n_frames)):
    mcu.sendData(np.zeros(frame_step,  dtype='int16'), 0, progress=False)
    mcu.waitForMcuReady()

  # MCU now runs inference, wait for complete
  mcu.waitForMcuReady()

  # MCU returns net input and output
  mcu_mfccs, tag = mcu.receiveData()
  print('Received %s type with tag 0x%x len %d' % (mcu_mfccs.dtype, tag, mcu_mfccs.shape[0]))
  mcu_pred, tag = mcu.receiveData()
  print('Received %s type with tag 0x%x len %d' % (mcu_pred.dtype, tag, mcu_pred.shape[0]))
  
  print('SUCCESS')
  return 0
  

######################################################################
# main
######################################################################
def main():
  ret = 0
  done = 0
  ret -= cmd_version()
  done += 1
  sleep(0.5)
  ret -= cmd_mic_sample_processed_manual()
  done += 1
  sleep(0.5)
  # ret -= cmd_mic_sample()
  # done += 1
  # sleep(0.5)
  # ret -= cmd_mic_sample_preprocessed()
  # done += 1
  # sleep(0.5)
  ret -= cmd_mel_one_batch()
  done += 1
  sleep(0.5)
  # ret -= cmd_kws_single_inference()
  # done += 1
  # sleep(0.5)
  # ret -= cmd_mfcc_kws_frame()
  # done += 1
  # sleep(0.5)

  print('--------------------------------------------------')
  print('Passed %d/%d tests' % (done-ret, done))
  print('--------------------------------------------------')

if __name__ == '__main__':
  main()