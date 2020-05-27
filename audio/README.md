# Audio Utilities

A collection of scripts to test and train. They are all bundled in the `main.py` scrtipt.

```bash
./main.py
# usage: edison <module> <command> [<args>]

# The modules:
#     mic       Test microphone, analyze bit depths
#     mcu       Tools to communicate with microcontroller
#     mfcc      Mel frequency cepstral coefficient tools
#     acquire   Collect training data
#     train     Train network
#     kws       Experiment with the keyword spotting algorithm
#     deploy    Deploy the algorithm to microcontroller code

# Run edison <module> for commands for the selected module
# main.py: error: the following arguments are required: module
```

## Environment
Setup once:
```bash
cd ../
source bootstrap.sh
```

## Example commands

### mcu

#### hif_test
```bash
./main.py mcu hif_test
```
### mic
#### bit_depth_analyze
```bash
./main.py mic bit_depth_analyze
```

### fetch_mic_samples
```bash
./main.py mic fetch_mic_samples
```

### mfcc
#### host
```bash
# runs MFCC on a audio sample with a custom implementation of MFCC and with Tensorflow.
./main.py mfcc host
```
#### mcu
```bash
# Calculate header file for MCU processing
./main.py mfcc mcu calc
# single frame MFCC on mcu and host with detailed vectors
./main.py mfcc mcu single
# Multi-frame on host and mcu with input/output compare
./main.py mfcc mcu file data/heysnips_true_16k_16b.wav
```
### acqiure
  sample_acq
```bash
./main.py acquire acq
```
### train
#### keras
```bash
# train model
./main.py train keras train
# test model only
./main.py train keras test
```
#### nnom
```bash
# Loads data from keras train, so this has to be run beforehand
./main.py train nnom
```
#### torch
This is used to test an implementation using Nemo. The model is trained using pyTorch and the quantization
is performed in Nemo. The deployment with `impl` is made with a custom script that maps the Nemo layers to
CMSIS-NN API functions.

```bash
# Loads data from keras train, so this has to be run beforehand
./main.py train torch train
./main.py train torch impl
```

### kws
#### live
```bash
# Live on MCU
./main.py kws live mcu
# Live on host (not working currently)
./main.py kws live host
```
#### mcu
```bash
# Single inference on random data
./main.py kws mcu single               
# Get file, run MFCC on host and inference on MCU
./main.py kws mcu fileinf .cache/acquire/noah/office/08848e0a.wav
# Get file, run MFCC and inference on host and on MCU
./main.py kws mcu file .cache/acquire/noah/office/08848e0a.wav          
# Record sample from onboard mic and do stuffs
./main.py kws mcu mic
# Record sample from host mic and do stuffs
./main.py kws mcu host
# Test net on host only using host mic
./main.py kws mcu hostcont
# Test net on host only using host mic and single frame
./main.py kws mcu hostsingle
# Continuous sample on MCU with net input history
# This is not implemented in the current firmware
./main.py kws mcu miccont
```

