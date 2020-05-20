import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import nemo
from tqdm import tqdm
import numpy as np

cache_dir = '.cache/ai_nemo'
model_path = cache_dir+'/nemo_model.pt'

import pathlib
pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
##################################################
# Steps

# Enable to train model (takes ~1min)
do_train = False
# Enable to test only on small number of samples to speed things up
dummy_tests = True
# Enable to run implementation
do_implementation = True

input_channel_max_value = 2**16-1

##################################################
# Model definition
#
#   Use same topology as in the allinone.py script
#     TF keras: 'valid' adds no zero padding, 'same' adds padding such that if the stride is 1, the output shape is the same as input shape. 

#     inputs = Input(shape=inp_shape)
#     x = Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='valid')(inputs)
#     x = BatchNormalization()(x)
#     x = ReLU()(x)
#     x = MaxPooling2D((2, 1), strides=(2, 1), padding="valid")(x)

#     x = Conv2D(32 ,kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
#     x = BatchNormalization()(x)
#     x = ReLU()(x)
#     x = MaxPooling2D((2, 1),strides=(2, 1), padding="valid")(x)

#     x = Conv2D(64 ,kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
#     x = BatchNormalization()(x)
#     x = ReLU()(x)
#     #x = MaxPooling2D((2, 1), strides=(2, 1), padding="valid")(x)
#     x = Dropout(0.2)(x)

#     x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid")(x)
#     x = BatchNormalization()(x)
#     x = ReLU()(x)
#     x = Dropout(0.3)(x)

#     x = Flatten()(x)
#     x = Dense(num_classes)(x)

#     predictions = Softmax()(x)

class ConvNet(nn.Module):
  def __init__(self, num_classes):
    super(ConvNet, self).__init__()

    # Torch layer sizes are of form (N,Cin,H,W)

    # First layer
    conv1_in_channels = 1
    conv1_out_channels = 16
    conv1_kernel = (5,5)
    conv1_strides = (1,1)
    conv1_padding_mode = 'zeros' # 'zeros', 'reflect', 'replicate' or 'circular'
    conv1_padding = (0,0)
    self.conv1 = nn.Conv2d(conv1_in_channels, conv1_out_channels, conv1_kernel, 
      stride=conv1_strides, padding=conv1_padding, padding_mode=conv1_padding_mode)
    self.bn1 = nn.BatchNorm2d(conv1_out_channels)
    self.relu1 = nn.ReLU()
    pool1_kernel_size = (2, 1)
    pool1_strides = (2,1)
    pool1_padding = (0,0)
    self.pool1 = nn.MaxPool2d(pool1_kernel_size, stride=pool1_strides, padding=pool1_padding)

    # First hidden layer
    conv2_in_channels = conv1_out_channels
    conv2_out_channels = 32
    conv2_kernel = (3,3)
    conv2_strides = (1,1)
    conv2_padding_mode = 'zeros' # 'zeros', 'reflect', 'replicate' or 'circular'
    conv2_padding = (0,0)
    self.conv2 = nn.Conv2d(conv2_in_channels, conv2_out_channels, conv2_kernel, 
      stride=conv2_strides, padding=conv2_padding, padding_mode=conv2_padding_mode)
    self.bn2 = nn.BatchNorm2d(conv2_out_channels)
    self.relu2 = nn.ReLU()
    pool2_kernel_size = (2, 1)
    pool2_strides = (2,1)
    pool2_padding = (0,0)
    self.pool2 = nn.MaxPool2d(pool2_kernel_size, stride=pool2_strides, padding=pool2_padding)

    # Second hidden layer
    conv3_in_channels = conv2_out_channels
    conv3_out_channels = 64
    conv3_kernel = (3,3)
    conv3_strides = (1,1)
    conv3_padding_mode = 'zeros' # 'zeros', 'reflect', 'replicate' or 'circular'
    conv3_padding = (0,0)
    self.conv3 = nn.Conv2d(conv3_in_channels, conv3_out_channels, conv3_kernel, 
      stride=conv3_strides, padding=conv3_padding, padding_mode=conv3_padding_mode)
    self.bn3 = nn.BatchNorm2d(conv3_out_channels)
    self.relu3 = nn.ReLU()

    # Third hidden layer
    conv4_in_channels = conv3_out_channels
    conv4_out_channels = 32
    conv4_kernel = (3,3)
    conv4_strides = (1,1)
    conv4_padding_mode = 'zeros' # 'zeros', 'reflect', 'replicate' or 'circular'
    conv4_padding = (0,0)
    self.conv4 = nn.Conv2d(conv4_in_channels, conv4_out_channels, conv4_kernel, 
      stride=conv4_strides, padding=conv4_padding, padding_mode=conv4_padding_mode)
    self.bn4 = nn.BatchNorm2d(conv4_out_channels)
    self.relu4 = nn.ReLU()
    
    # output fully connected layer
    self.fc1 = nn.Linear(96, num_classes)

  def forward(self, x):
    x = self.conv1(x)
    # print('x = self.conv1(x) size', x.shape)
    x = self.bn1(x)
    # print('x = self.bn1(x) size', x.shape)
    x = self.relu1(x)
    # print('x = self.relu1(x) size', x.shape)
    x = self.pool1(x)
    # print('x = self.pool1(x) size', x.shape)

    x = self.conv2(x)
    # print('x = self.conv2(x) size', x.shape)
    x = self.bn2(x)
    # print('x = self.bn2(x) size', x.shape)
    x = self.relu2(x)
    # print('x = self.relu2(x) size', x.shape)
    x = self.pool2(x)
    # print('x = self.pool2(x) size', x.shape)

    x = self.conv3(x)
    # print('x = self.conv3(x) size', x.shape)
    x = self.bn3(x)
    # print('x = self.bn3(x) size', x.shape)
    x = self.relu3(x)
    # print('x = self.relu3(x) size', x.shape)
    # x = nn.functional.dropout(x, p=0.2)
    # print('x = self.do1(x) size', x.shape)

    x = self.conv4(x)
    # print('x = self.conv4(x) size', x.shape)
    x = self.bn4(x)
    # print('x = self.bn4(x) size', x.shape)
    x = self.relu4(x)
    # print('x = self.relu4(x) size', x.shape)
    # x = nn.functional.dropout(x, p=0.3)
    # print('x = self.do2(x) size', x.shape)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    output = F.log_softmax(x, dim=1) # <== the softmax operation does not need to be quantized, we can keep it as it is
    return output



# convenience class to keep track of averages
class Metric(object):
  def __init__(self, name):
    self.name = name
    self.sum = torch.tensor(0.)
    self.n = torch.tensor(0.)
  def update(self, val):
    self.sum += val.cpu()
    self.n += 1
  @property
  def avg(self):
    return self.sum / self.n



def train(model, device, train_loader, optimizer, criterion, max_epochs, verbose=True):
  # model.train() # <- dunno what this should do
  train_loss = Metric('train_loss')
  for epoch in range(max_epochs):
    with tqdm(total=len(train_loader),
        desc='Train Epoch  #%d/%d' % (epoch+1, max_epochs),
        disable=not verbose) as t:
      for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print('output', output)
        # print('output.shape', output.shape)
        # print('target', target)
        # print('target.shape', target.shape)
        # loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss.update(loss)
        t.set_postfix({'loss': train_loss.avg.item()})
        t.update(1)
      # Validation
      # with torch.set_grad_enabled(False):
      #     for local_batch, local_labels in validation_generator:
      #         # Transfer to GPU
      #         local_batch, local_labels = local_batch.to(device), local_labels.to(device)

      #         # Model computations
      #         [...]
  return train_loss.avg.item()
  return


def test(model, device, test_loader, integer=False, verbose=True):
  model.eval()
  test_loss = 0
  correct = 0
  test_acc = Metric('test_acc')
  with tqdm(total=len(test_loader),
        desc='Test',
        disable=not verbose) as t:
    with torch.no_grad():
      for data, target in test_loader:
        if integer:      # <== this will be useful when we get to the 
          data *= input_channel_max_value  #     IntegerDeployable stage
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        test_acc.update((pred == target.view_as(pred)).float().mean())
        t.set_postfix({'acc' : test_acc.avg.item() * 100. })
        t.update(1)
  test_loss /= len(test_loader.dataset)
  return test_acc.avg.item() * 100.

def implement(model, dummy_input):

  x = dummy_input

  bufsizes = {}
  actsizes = {}
  funCalls = {}
  c_lines, h_lines = '',''

  for name, m in model.named_modules():
    # print('\n\n\n\n')
    print('------------------------------------------------')
    print(name, '->', m, type(m))
    print('------------------------------------------------')

    if isinstance(m, ConvNet):
      continue

    l = {}
    # If this layer is fc, flatten first
    if isinstance(m, nemo.quant.pact.PACT_Linear):
      x = torch.flatten(x, 1)
    inp_shape = x.shape # NCHW
    x = m.forward(x)
    oup_shape = x.shape

    # if convolution layer
    if isinstance(m, nemo.quant.pact.PACT_Conv2d):
      # arm_convolve_s8:
      # * @param[in]       input           pointer to input tensor. Range: int8, format: [N,H,W,in_ch]
      # * @param[in]       input_x         input tensor width
      # * @param[in]       input_y         input tensor height
      # * @param[in]       input_ch        number of input tensor channels
      # * @param[in]       input_batches   number of input batches

      # * @param[in]       kernel          pointer to kernel weights. Range: int8, format: [out_ch, H, W, in_ch]
      # * @param[in]       output_ch       number of filters, i.e., output tensor channels
      # * @param[in]       kernel_x        filter/kernel width
      # * @param[in]       kernel_y        filter/kernel height

      # * @param[in]       pad_x           padding along width
      # * @param[in]       pad_y           padding along height

      # * @param[in]       stride_x        convolution stride x
      # * @param[in]       stride_y        convolution stride y

      # * @param[in]       bias            pointer to per output channel bias. Range: int32

      # * @param[in,out]   output          pointer to output tensor. format: [H, W, out_ch]

      # * @param[in]       output_shift    pointer to per output channel requantization shift parameter.
      # * @param[in]       output_mult     pointer to per output channel requantization multiplier parameter.

      # * @param[in]       out_offset      output tensor offset. Range: int8
      # * @param[in]       input_offset    input tensor offset. Range: int8
      # * @param[in]       output_activation_min   Minimum value to clamp the output to. Range: int8
      # * @param[in]       output_activation_max   Minimum value to clamp the output to. Range: int8
      # * @param[in]       output_x    output tensor width
      # * @param[in]       output_y    output tensor height
      # * @param[in]       buffer_a    pointer to buffer space used for input optimization(partial im2col) and is necessary
      # *                              when ARM_MATH_DSP is defined.
      # *                              Required space: (2 * input_ch * kernel_x * kernel_y) * sizeof(q15_t) bytes
      # *                              Use arm_convolve_s8_get_buffer_size() to get the size.

      # arm_status arm_convolve_s8(const q7_t *input,
      #                            const uint16_t input_x,
      #                            const uint16_t input_y,
      #                            const uint16_t input_ch,
      #                            const uint16_t input_batches,
      #                            const q7_t *kernel,
      #                            const uint16_t output_ch,
      #                            const uint16_t kernel_x,
      #                            const uint16_t kernel_y,
      #                            const uint16_t pad_x,
      #                            const uint16_t pad_y,
      #                            const uint16_t stride_x,
      #                            const uint16_t stride_y,
      #                            const int32_t *bias,
      #                            q7_t *output,
      #                            const int32_t *output_shift,
      #                            const int32_t *output_mult,
      #                            const int32_t out_offset,
      #                            const int32_t input_offset,
      #                            const int32_t output_activation_min,
      #                            const int32_t output_activation_max,
      #                            const uint16_t output_x,
      #                            const uint16_t output_y,
      #                            q15_t *buffer_a);
      print('Found PACT_Conv2d layer')
      
      l['input_x'] = inp_shape[3]
      l['input_y'] = inp_shape[2]
      l['input_ch'] = inp_shape[1]
      l['input_batches'] = inp_shape[0]

      # convert NCHW to NHWC
      ker = np.transpose(m.weight.data.numpy(), [0, 2, 3, 1]).ravel()
      # l['kernel'] = ker
      l['output_ch'] = oup_shape[1]
      l['kernel_x'] = m.weight.shape[2]
      l['kernel_y'] = m.weight.shape[3]

      l['pad_x'] = m.padding[0]
      l['pad_y'] = m.padding[1]

      l['stride_x'] = m.stride[0]
      l['stride_y'] = m.stride[1]

      # convert NCHW to NHWC
      bias = m.bias.data.numpy().ravel()
      # l['bias'] = bias

      # These are probably done in the model's PACT_IntegerAct
      for i, c in enumerate(name):
        if c.isdigit():
          lay_num = int(name[i:])    
      print('layer number got', lay_num)
      output_shift = np.array(l['output_ch']*[0])
      output_mult = np.array(l['output_ch']*[1])
      found_integer_act = False
      for sname, mod in model.named_modules():
        if sname == 'relu'+str(lay_num) and isinstance(mod, nemo.quant.pact.PACT_IntegerAct):
          print('Found PACT_IntegerAct following Conv layer!!!')
          found_integer_act = True
          output_mult = np.array(l['output_ch']*[(mod.D*mod.eps_in/mod.eps_out).round()], dtype=int)
          output_shift = np.array(l['output_ch']*[np.log2(mod.D)], dtype=int )
          l['output_activation_min'] = 0
          l['output_activation_max'] = int(mod.alpha_out)
      if not found_integer_act:
        print("WARNING: Didn't find a PACT_IntegerAct layer following a conv. This is not handled yet")
      # print(output_mult)
      # print(output_shift)
      # lay_num = [int(s) for s in name.split() if s.isdigit()][0]
      # for sname in model.named_modules():
      #   if sname[-1]

      l['out_offset'] = 0
      l['input_offset'] = 0

      l['output_x'] = oup_shape[3]
      l['output_y'] = oup_shape[2]
      actsizes[name] = 1*l['output_x']*l['output_y']*l['output_ch']

      l['buffer_a_size'] = (2 * l['input_ch'] * l['kernel_x'] * l['kernel_y']) * 2 # 2 == sizeof(q15_t)
      bufsizes[name] = l['buffer_a_size']
      # bias = m.bias.data # tensor
      # weight = m.weight.data # tensor
      # pad = m.padding # tuple, (0,0)
      # stride = m.stride # tuple, (0,0)
      # in_ch = m.in_channels # int
      # out_ch = m.out_channels # int

      ##############################
      # generate C code
      h_lines += '// Layer ' + name + '\n'
      un = name.upper()
      # layer properties
      for k, v in l.items():
        h_lines += '#define ' + un + '_' + k.upper() + '\t' + str(v) + '\n'
      # layer values
      h_lines += '#define ' + un + '_WT {' + ','.join(map(str,ker.astype(int))) + '}\n'
      h_lines += '#define ' + un + '_BIAS {' + ','.join(map(str,bias.astype(int))) + '}\n'
      h_lines += '#define ' + un + '_OUT_MULT {' + ','.join(map(str,output_mult.astype(int))) + '}\n'
      h_lines += '#define ' + un + '_OUT_SHIFT {' + ','.join(map(str,output_shift.astype(int))) + '}\n'
      
      # data arrays for weight and bias
      c_lines += '// Layer ' + name + '\n'
      c_lines += 'static const q7_t ' + name + '_wt [' + un + '_INPUT_CH*' + un + '_KERNEL_X*' + un + '_KERNEL_Y*' + un + '_OUTPUT_CH] = '+un+'_WT;\n'
      c_lines += 'static const q7_t ' + name + '_bias [' + un + '_OUTPUT_CH] = '+un+'_BIAS;\n'
      c_lines += 'static const q7_t ' + name + '_output_mult [' + un + '_OUTPUT_CH] = '+un+'_OUT_MULT;\n'
      c_lines += 'static const q7_t ' + name + '_output_shift [' + un + '_OUTPUT_CH] = '+un+'_OUT_SHIFT;\n'

      funCall = ''.join((
        "status = arm_convolve_s8(__LAYER_IN,",
        "{}, ".format(un+'_input_x'.upper()),
        "{}, ".format(un+'_input_y'.upper()),
        "{}, ".format(un+'_input_ch'.upper()),
        "{}, ".format(un+'_input_batches'.upper()),
        "{}, ".format(name + '_wt'),
        "{}, ".format(un+'_output_ch'.upper()),
        "{}, ".format(un+'_kernel_x'.upper()),
        "{}, ".format(un+'_kernel_y'.upper()),
        "{}, ".format(un+'_pad_x'.upper()),
        "{}, ".format(un+'_pad_y'.upper()),
        "{}, ".format(un+'_stride_x'.upper()),
        "{}, ".format(un+'_stride_y'.upper()),
        "{}, ".format(name + '_bias'),
        "__LAYER_OUT,",
        "{}, ".format(name + '_output_shift'),
        "{}, ".format(name + '_output_mult'),
        "{}, ".format(un+'_out_offset'.upper()),
        "{}, ".format(un+'_input_offset'.upper()),
        "{}, ".format(un+'_output_activation_min'.upper()),
        "{}, ".format(un+'_output_activation_max'.upper()),
        "{}, ".format(un+'_output_x'.upper()),
        "{}, ".format(un+'_output_y'.upper()),
        "__LAYER_BUF);"))
      funCalls[name] = (funCall)
      
    elif isinstance(m, nemo.quant.pact.PACT_IntegerBatchNormNd):
      print('Found PACT_IntegerBatchNormNd layer')

    elif isinstance(m, nemo.quant.pact.PACT_IntegerAct):
      print('Found PACT_IntegerAct layer')
      # Absorbed in conv layer
      l = {}
      l['eps_in'] = m.eps_in.numpy()
      l['eps_out'] = m.eps_out
      l['D'] = m.D.numpy()
      l['alpha_out'] = m.alpha_out
      l['shift'] = np.log2(l['D'])

      l['eps_ratio'] = (m.D*m.eps_in/m.eps_out).round().numpy()

    elif isinstance(m, torch.nn.modules.pooling.MaxPool2d):
      print('Found MaxPool2d layer')

      l['input_x'] = inp_shape[3]
      l['input_y'] = inp_shape[2]
      l['depth'] = inp_shape[1] # input channels

      l['output_x'] = oup_shape[3]
      l['output_y'] = oup_shape[2]

      l['stride_x'] = m.stride[0]
      l['stride_y'] = m.stride[1]
      l['kernel_x'] = m.kernel_size[0]
      l['kernel_y'] = m.kernel_size[1]
      l['pad_x'] = m.padding[0]
      l['pad_y'] = m.padding[1]
      l['act_min'] = -2**(8-1)
      l['act_max'] =  2**(8-1)-1
      actsizes[name] = 1*l['output_x']*l['output_y']*l['depth']

      ##############################
      # generate C code
      h_lines += '// Layer ' + name + '\n'
      un = name.upper()
      # layer properties
      for k, v in l.items():
        h_lines += '#define ' + un + '_' + k.upper() + '\t' + str(v) + '\n'

      funCall = ''.join((
        "status = arm_max_pool_s8_opt(",
        "{}, ".format(un+'_input_y'.upper()),
        "{}, ".format(un+'_input_x'.upper()),
        "{}, ".format(un+'_output_y'.upper()),
        "{}, ".format(un+'_output_x'.upper()),
        "{}, ".format(un+'_stride_y'.upper()),
        "{}, ".format(un+'_stride_x'.upper()),
        "{}, ".format(un+'_kernel_y'.upper()),
        "{}, ".format(un+'_kernel_x'.upper()),
        "{}, ".format(un+'_pad_y'.upper()),
        "{}, ".format(un+'_pad_x'.upper()),
        "{}, ".format(un+'_act_min'.upper()),
        "{}, ".format(un+'_act_max'.upper()),
        "{}, ".format(un+'_depth'.upper()),
        "__LAYER_IN, ",
        "__LAYER_BUF, ",
        "__LAYER_OUT);"))
      funCalls[name] = funCall
       #    /**
       # * @brief s8 DSP optimized max pooling function
       # * @param[in]       input_y     input tensor dimension along y
       # * @param[in]       input_x     input tensor dimension along x
       # * @param[in]       output_y    output tensor dimension along y
       # * @param[in]       output_x    output tensor dimension along x
       # * @param[in]       stride_y    stride along y
       # * @param[in]       stride_x    stride along x
       # * @param[in]       kernel_y    filter kernel size along y
       # * @param[in]       kernel_x    filter kernel size along x
       # * @param[in]       pad_y       padding size along y
       # * @param[in]       pad_x       padding size along x
       # * @param[in]       act_min     Activation min. Lower limit to clamp output to. Range: int8
       # * @param[in]       act_max     Activation max. Upper limit to clamp output to. Range: int8
       # * @param[in]       depth       number of input channels
       # * @param[in]       input       pointer to input tensor
       # * @param[in]       tmp_buffer  Not used.
       # * @param[in,out]   output      pointer to output tensor
       # * @return                      The function returns one of the following
       # *                              <code>ARM_MATH_SIZE_MISMATCH</code> - Unsupported dimension of tensors
       # *                              <code>ARM_MATH_SUCCESS</code> - Successful operation
       # *                              <code>ARM_MATH_ARGUMENT_ERROR</code> - Implementation not available
       # * @note The input data is corrupted by this function.
       # * @details This optimized implementation is recommended when depth is >=  4 and dimensions are large.
       # *
       # */
        # arm_status arm_max_pool_s8_opt(const uint16_t input_y,
        #                                const uint16_t input_x,
        #                                const uint16_t output_y,
        #                                const uint16_t output_x,
        #                                const uint16_t stride_y,
        #                                const uint16_t stride_x,
        #                                const uint16_t kernel_y,
        #                                const uint16_t kernel_x,
        #                                const uint16_t pad_y,
        #                                const uint16_t pad_x,
        #                                const int8_t act_min,
        #                                const int8_t act_max,
        #                                const uint16_t depth,
        #                                int8_t *src,
        #                                int16_t *tmp_buffer,
        #                                int8_t *dst)


    elif isinstance(m, nemo.quant.pact.PACT_Linear):
      print('Found PACT_Linear layer')
      weights = m.weight.detach().numpy().ravel()
      bias = m.bias.detach().numpy().ravel()
      # x_quant = input
      # y = torch.nn.functional.linear(x_quant, W_quant, self.bias)
      print(inp_shape)
      print(oup_shape)

      # l['pInput'] = 
      l['pWeight'] = name + '_wt'
      l['col_dim'] =  inp_shape[1]
      l['row_dim'] =  oup_shape[1]
      l['nb_batches'] =  1
      l['input_offset'] = 0
      l['filter_offset'] = 0
      l['out_mult'] = 1
      l['out_shift'] = 0
      l['output_offset'] = 0
      l['pBias'] = name + '_bias'
      # l['pOut'] = 
      l['output_activation_min'] = -2**(32-1)
      l['output_activation_max'] =  2**(32-1)-1
      l['vec_buffer'] = 'computationBuffer'
      l['vec_buffer_size'] = int(2*l['col_dim'])
      bufsizes[name] = l['vec_buffer_size']
      actsizes[name] = 1*l['row_dim']

      ##############################
      # generate C code
      h_lines += '// Layer ' + name + '\n'
      un = name.upper()
      # layer properties
      for k, v in l.items():
        h_lines += '#define ' + un + '_' + k.upper() + '\t' + str(v) + '\n'
      # layer values
      h_lines += '#define ' + un + '_WT {' + ','.join(map(str,weights.astype(int))) + '}\n'
      h_lines += '#define ' + un + '_BIAS {' + ','.join(map(str,bias.astype(int))) + '}\n'

      # data arrays for weight and bias
      c_lines += '// Layer ' + name + '\n'
      c_lines += 'static const q7_t ' + name + '_wt [ sizeof(q15_t) * ' + un + '_COL_DIM] = '+un+'_WT;\n'
      c_lines += 'static const q7_t ' + name + '_bias [ sizeof(q15_t) * ' + un + '_ROW_DIM] = '+un+'_BIAS;\n'

      funCall = ''.join((
        "status = arm_fully_connected_s8(",
        "__LAYER_IN, ",
        "{}, ".format(name + '_wt'),
        "{}, ".format(un+'_col_dim'.upper()),
        "{}, ".format(un+'_row_dim'.upper()),
        "{}, ".format(un+'_nb_batches'.upper()),
        "{}, ".format(un+'_input_offset'.upper()),
        "{}, ".format(un+'_filter_offset'.upper()),
        "{}, ".format(un+'_out_mult'.upper()),
        "{}, ".format(un+'_out_shift'.upper()),
        "{}, ".format(un+'_output_offset'.upper()),
        "{}, ".format(name + '_bias'),
        "__LAYER_OUT, ",
        "{}, ".format(un+'_output_activation_min'.upper()),
        "{}, ".format(un+'_output_activation_max'.upper()),
        "__LAYER_BUF);"))
      funCalls[name] = funCall

      # /**
      #  * @brief S8 basic fully-connected and matrix multiplication layer function for TF Lite
      #  * @param[in]       pInput                       pointer to pInput vector
      #  * @param[in]       pWeight                      pointer to matrix weights
      #  * @param[in]       col_dim                      dimension of the input vector
      #  * @param[in]       row_dim                      dimension of the output vector
      #  * @param[in]       nb_batches                   number of batches
      #  * @param[in]       input_offset                 tensor offset for input. Range: -127 to 128
      #  * @param[in]       filter_offset                tensor offset for filter. Range: -127 to 128
      #  * @param[in]       out_mult                     requantization parameter
      #  * @param[in]       out_shift                    requantization parameter
      #  * @param[in]       output_offset                tensor offset for output. Range: int8
      #  * @param[in]       pBias                        pointer to bias
      #  * @param[out]      pOut                         pointer to output vector
      #  * @param[in]       output_activation_min        for clamping
      #  * @param[in]       output_activation_max        for clamping
      #  * @param[in]       vec_buffer                   pointer to buffer space used for optimization and is necessary
      #  *                                               when ARM_MATH_DSP is defined but not
      #  *                                               ARM_MATH_MVEI.
      #  *                                               Required space: col_dim * sizeof(q15_t) bytes
      #  *                                               Use arm_fully_connected_s8_get_buffer_size() to get the size.
      #  * @return          The function returns         ARM_MATH_SUCCESS
      #  *
      #  * @details
      #  *
      #  * <b>Buffer size:</b>
      #  *
      #  * vec_buffer size: col_dim of word16.
      #  *
      #  *  This basic function is designed to work with regular pWeight
      #  *  matrix without interleaving.
      #  *
      #  *    1. Supported framework: TensorFlow Lite
      #  *    2. q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
      #  *
      #  */

      # arm_fully_connected_s8(const int8_t *input,
      #                        const int8_t *kernel,
      #                        const uint16_t col_dim,
      #                        const uint16_t row_dim,
      #                        const uint16_t nb_batches,
      #                        const int32_t input_offset,
      #                        const int32_t filter_offset,
      #                        const int32_t out_mult,
      #                        const int32_t out_shift,
      #                        const int32_t output_offset,
      #                        const int32_t *bias,
      #                        int8_t *output,
      #                        const int32_t output_activation_min,
      #                        const int32_t output_activation_max,
      #                        q15_t *vec_buffer)

    else:
      print('ERROR: Unsupported layer', type(m))
      return -1

  print('------------------------------------------------')
  print('Write code')
  print('------------------------------------------------')

  print('Activation buffer per layer:', actsizes)
  print('Scratch buffer per layer:', bufsizes)

  with open(cache_dir+'/cmsis_net.h', 'w') as fd_header:
    fd_header.write(h_lines)
    fd_header.write('\n\narm_status cmsisRunInference (void* input, void* output);\n')

  with open(cache_dir+'/cmsis_net.c', 'w') as fd_code:
    fd_code.write('#include <stdint.h>\n')
    fd_code.write('#include <stdio.h>\n')
    fd_code.write('#include "arm_nnfunctions.h"\n')
    fd_code.write('#include "cmsis_net.h"\n')

    fd_code.write('\n\n\n'+c_lines)

    # Write code
    line = '\n'.join((
      "\n// calculation and activations buffer",
      "static uint8_t tmpBuf[{}];".format(max(bufsizes.items())[1]),
      "static uint8_t activations1[{}];".format(max(actsizes.items())[1]),
      "static uint8_t activations2[{}];".format(max(actsizes.items())[1]),
      "\n// inference",
      "arm_status cmsisRunInference (void* input, void* output)",
      "{",
      "  arm_status status;",
      ))

    c=[key for key,value in funCalls.items()]

    # first and last
    first, last = c[0], c[-1]
    for name,funcall in funCalls.items():
      if name == first:
        inp = 'input'
        oup = 'activations1'
      elif name == last:
        inp = oup
        oup = 'output'
      else:
        inp, oup = oup, inp

      funcall = funcall.replace('__LAYER_BUF', 'tmpBuf')
      funcall = funcall.replace('__LAYER_IN', inp)
      funcall = funcall.replace('__LAYER_OUT', oup)

      line += '\n'.join((
        "\n  // layer: {}, {} -> {}".format(name, inp, oup),
        "  {}".format(funcall),
        "\n"))
      if name == first:
        inp = 'activations2'

    line += '\n'.join((
      "  return status;\n}",
      ))

    fd_code.write(line)


  # for name, param in model.named_parameters():
  #   print(name, param.size())
  #   model.

######################################################################
# main
######################################################################
if __name__ == '__main__':

  # Data loaders
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

  # Convert data from NHWC to NCHW
  x_train   = np.transpose(np.load('.cache/allinone/x_train.npy'), [0, 3, 1, 2])
  x_test    = np.transpose(np.load('.cache/allinone/x_test.npy'), [0, 3, 1, 2])
  x_val     = np.transpose(np.load('.cache/allinone/x_val.npy'), [0, 3, 1, 2])
  # y is one-hot in files, we need it flat
  y_train   = np.load('.cache/allinone/y_train.npy').argmax(axis=1).astype(int)
  y_test    = np.load('.cache/allinone/y_test.npy').argmax(axis=1).astype(int)
  y_val     = np.load('.cache/allinone/y_val.npy').argmax(axis=1).astype(int)
  keywords  = np.load('.cache/allinone/keywords.npy')

  print('x_train.shape    ', x_train.shape)
  # print('x_test.shape     ', x_test.shape)
  # print('x_val.shape      ', x_val.shape)
  print('y_train.shape    ', y_train.shape)
  # print('y_test.shape     ', y_test.shape)
  # print('y_val.shape      ', y_val.shape)

  # convert to torch tensors
  x_train_t   = torch.Tensor(x_train)
  x_test_t    = torch.Tensor(x_test)
  x_val_t     = torch.Tensor(x_val)
  y_train_t   = torch.Tensor(y_train).long()
  y_test_t    = torch.Tensor(y_test).long()
  y_val_t     = torch.Tensor(y_val).long()

  if dummy_tests:
    x_test_t = x_test_t[0:1]
    y_test_t = y_test_t[0:1]
  # print(y_train_t)

  # create data loaders
  trainset = torch.utils.data.TensorDataset(x_train_t,y_train_t)
  testset = torch.utils.data.TensorDataset(x_test_t,y_test_t)
  valset = torch.utils.data.TensorDataset(x_val_t,y_val_t)

  train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)
  val_loader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=True)

  # train at FullPrecision stage
  num_classes = y_train.max()+1
  inp_shape = x_train.shape[1:]
  print('num_classes:',num_classes)
  model = ConvNet(num_classes).to(device)

  # single_item = x_train_t[0:1]
  # print('single_item.shape', single_item.shape)
  # model.forward(single_item)

  # Show model info
  # import torchsummary as tsum
  # tsum.summary(model, inp_shape)

  if do_train:
    learning_rate = 0.001
    max_epochs = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = nn.CrossEntropyLoss()
    train(model, device, train_loader, optimizer, loss, max_epochs, verbose=True)
    torch.save(model.state_dict(), model_path)
  else:
    model.load_state_dict(torch.load(model_path))

  acc = test(model, device, test_loader)
  print("\nFullPrecision accuracy: %.02f%%" % acc)

  ##########################################
  # original net
  #
  batch_size = 1
  inp_shape = x_train.shape[1:]
  perm = lambda x : x
  model.eval()
  dummy_input = perm(torch.randn(batch_size, *inp_shape, device='cuda' if torch.cuda.is_available() else 'cpu'))
  torch.onnx.export(model, dummy_input, cache_dir+'/kws_float.onnx',
   do_constant_folding=True, export_params=True, opset_version=11)
  # exit()
  ##########################################
  # FakeQuantized network

  model = nemo.transform.quantize_pact(model, dummy_input=torch.randn((1,)+x_train.shape[1:]).to(device))
  precision = {
    'conv1': {
        'W_bits' : 15
    },
    'relu1': {
        'x_bits' : 16
    },
    'conv2': {
        'W_bits' : 15
    },
    'relu2': {
        'x_bits' : 16
    },
    'conv3': {
        'W_bits' : 15
    },
    'relu3': {
        'x_bits' : 16
    },
    'conv4': {
        'W_bits' : 15
    },
    'relu4': {
        'x_bits' : 16
    },
    'fc1': {
        'W_bits' : 15
    },
  }
  model.change_precision(bits=1, min_prec_dict=precision)
  acc = test(model, device, test_loader)
  print("\nFakeQuantized @ 16b accuracy (first try): %.02f%%" % acc)

  # calibarte activation quantization
  model.set_statistics_act()
  _ = test(model, device, test_loader)
  model.unset_statistics_act()
  model.reset_alpha_act()
  acc = test(model, device, test_loader)
  print("\nFakeQuantized @ 16b accuracy (calibrated): %.02f%%" % acc)

  # heavier quantization
  precision = {
    'conv1': {
        'W_bits' : 7
    },
    'relu1': {
        'x_bits' : 8
    },
    'conv2': {
        'W_bits' : 7
    },
    'relu2': {
        'x_bits' : 8
    },
    'conv3': {
        'W_bits' : 7
    },
    'relu3': {
        'x_bits' : 8
    },
    'conv4': {
        'W_bits' : 7
    },
    'relu4': {
        'x_bits' : 8
    },
    'fc1': {
        'W_bits' : 7
    },
  }
  model.change_precision(bits=1, min_prec_dict=precision)
  acc = test(model, device, test_loader)
  print("\nFakeQuantized @ mixed-precision accuracy: %.02f%%" % acc)
  nemo.utils.save_checkpoint(model, None, 0, checkpoint_name='kws_fq_mixed')

  # towards a possible deployment: folding of batch-normalization layers
  # folding absorbs them (batch-normalization layers) inside the quantized parameters
  model.fold_bn()
  model.reset_alpha_weights()
  acc = test(model, device, test_loader)
  print("\nFakeQuantized @ mixed-precision (folded) accuracy: %.02f%%" % acc)

  # Re-equalize weights
  # model.equalize_weights_dfq({'conv1':'conv2','conv2':'conv3','conv3':'conv4'})
  model.equalize_weights_dfq({'conv1':'conv2'})
  model.set_statistics_act()
  _ = test(model, device, test_loader)
  model.unset_statistics_act()
  model.reset_alpha_act()
  acc = test(model, device, test_loader)
  print("\nFakeQuantized @ mixed-precision (folded+equalized) accuracy: %.02f%%" % acc)

  # transition to QuantizedDeployable state
  # the inputs are MFCC coefficients in int16 format
  state_dict = torch.load('checkpoint/kws_fq_mixed.pth')['state_dict']
  model.load_state_dict(state_dict, strict=True)
  model = nemo.transform.bn_quantizer(model)
  model.harden_weights()
  model.set_deployment(eps_in=1./input_channel_max_value)
  print(model)
  acc = test(model, device, test_loader)
  print("\nQuantizedDeployable @ mixed-precision accuracy: %.02f%%" % acc)

  # export net in ONNX
  inp_shape = x_train.shape[1:]
  nemo.utils.export_onnx(cache_dir+'/kws_qd_mixed.onnx', 
    model, model, inp_shape, round_params=False) # round_params is active by default because NEMO mainly exports integerized networks!

  # transform the network to the last stage: IntegerDeployable
  model = nemo.transform.integerize_pact(model, eps_in=1.0/input_channel_max_value)
  print(type(model))
  print(model)
  acc = test(model, device, test_loader, integer=True)
  print("\nIntegerDeployable @ mixed-precision accuracy: %.02f%%" % acc)
  # export
  nemo.utils.export_onnx(cache_dir+'/kws_id_mixed.onnx', model, model, inp_shape)

  if do_implementation:
    # run implementation
    perm = lambda x : x
    input_shape = inp_shape = x_train.shape[1:]
    dummy_input = perm(torch.randn(1, *input_shape, device='cuda' if torch.cuda.is_available() else 'cpu'))
    implement(model, dummy_input)


  


