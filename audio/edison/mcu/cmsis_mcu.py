# -*- coding: utf-8 -*-
# @Author: Noah Huetter
# @Date:   2020-05-20 18:07:22
# @Last Modified by:   Noah Huetter
# @Last Modified time: 2020-05-21 14:07:21


import mcu_util as mcu
import pickle
with open('.cache/ai_nemo/net_hist.pkl', 'rb') as handle:
    net_hist = pickle.load(handle)

data, tag = mcu.receiveData(timeout=-1)
print('Received %s type with tag 0x%x len %d: %s' % (data.dtype, tag, data.shape[0], data))
inp = data.reshape([1,31,13,1]).transpose([0,3,1,2]) # CMSIS NHWC to torch NCHW

data, tag = mcu.receiveData(timeout=-1)
print('Received %s type with tag 0x%x len %d: %s' % (data.dtype, tag, data.shape[0], data))
conv1 = data.reshape([1,27,9,16]).transpose([0,3,1,2]) # CMSIS NHWC to torch NCHW

data, tag = mcu.receiveData(timeout=-1)
print('Received %s type with tag 0x%x len %d: %s' % (data.dtype, tag, data.shape[0], data))
pool1 = data.reshape([1,13,9,16]).transpose([0,3,1,2]) # CMSIS NHWC to torch NCHW

data, tag = mcu.receiveData(timeout=-1)
print('Received %s type with tag 0x%x len %d: %s' % (data.dtype, tag, data.shape[0], data))
conv2 = data.reshape([1,11,7,32]).transpose([0,3,1,2]) # CMSIS NHWC to torch NCHW

data, tag = mcu.receiveData(timeout=-1)
print('Received %s type with tag 0x%x len %d: %s' % (data.dtype, tag, data.shape[0], data))
pool2 = data.reshape([1,5,7,32]).transpose([0,3,1,2]) # CMSIS NHWC to torch NCHW

data, tag = mcu.receiveData(timeout=-1)
print('Received %s type with tag 0x%x len %d: %s' % (data.dtype, tag, data.shape[0], data))
conv3 = data.reshape([1,3,5,64]).transpose([0,3,1,2]) # CMSIS NHWC to torch NCHW

data, tag = mcu.receiveData(timeout=-1)
print('Received %s type with tag 0x%x len %d: %s' % (data.dtype, tag, data.shape[0], data))
conv4 = data.reshape([1,1,3,32]).transpose([0,3,1,2]) # CMSIS NHWC to torch NCHW

data, tag = mcu.receiveData(timeout=-1)
print('Received %s type with tag 0x%x len %d: %s' % (data.dtype, tag, data.shape[0], data))
oup = data.reshape([1,10])
