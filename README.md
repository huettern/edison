# Edison - Low Power Voice Commands [![Build Status](https://travis-ci.com/noah95/edison.svg?token=W9DfQq55LKsHhNiMPYw5&branch=master)](https://travis-ci.com/noah95/edison)

## Experimenting with MFCC

### Understanding MFCC
This script runs MFCC on a audio sample with a custom implementation of MFCC and with Tensorflow.

```bash
python mfcc.py
```

![](doc/img/mel_own.png)

![](doc/img/mel_tf.png)

## Interfacing with MCU
Build the firmware and follow these instructions to get some data.

### MFCC Single
Working and tested at f86859f.

To test a single 1024 element frame of audio data, run in the `audio` directory
```bash
python mfcc_on_mcu.py single
```

![](doc/img/mfcc_on_mcu_single.png)

### MFCC on audio file
Working and tested at 6ddbdc4.

```bash
python mfcc_on_mcu.py file data/heysnips_true_16k_16b.wav
```

![](doc/img/mfcc_snips.png)
