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
    x = nn.functional.dropout(x, p=0.2)
    # print('x = self.do1(x) size', x.shape)

    x = self.conv4(x)
    # print('x = self.conv4(x) size', x.shape)
    x = self.bn4(x)
    # print('x = self.bn4(x) size', x.shape)
    x = self.relu4(x)
    # print('x = self.relu4(x) size', x.shape)
    x = nn.functional.dropout(x, p=0.3)
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

  do_train = False
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
  # FakeQuantized network
  #
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
  print(model)
  acc = test(model, device, test_loader, integer=True)
  print("\nIntegerDeployable @ mixed-precision accuracy: %.02f%%" % acc)
  # export
  nemo.utils.export_onnx(cache_dir+'/kws_id_mixed.onnx', model, model, inp_shape)





