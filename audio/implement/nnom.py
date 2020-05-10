# -*- coding: utf-8 -*-
# @Author: Noah Huetter
# @Date:   2020-05-10 11:34:32
# @Last Modified by:   Noah Huetter
# @Last Modified time: 2020-05-10 11:55:59

import tensorflow as tf
import sys

model_file = '../models/kws_model_medium_embedding_conv.h5'

model = tf.keras.models.load_model(model_file)

x_test = np.load('../train/.cache/kws_keras/x_val_mfcc_mcu.npy')

generate_model(model, x_test, name='weights.h')

