# -*- coding: utf-8 -*-
# @Author: Noah Huetter
# @Date:   2020-04-16 16:59:06
# @Last Modified by:   Noah Huetter
# @Last Modified time: 2020-04-16 21:07:55

import audioutils as au
import mfcc_utils as mfu

# load data
x_train, y_train, x_test, y_test = au.load_snips_data()