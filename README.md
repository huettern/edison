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


## From `h5` to MCU

### Using CubeMX

1. Open CubeMX
2. Make sure X-CUBE-AI inst installed under Help -> Manage packages -> STMicroelectronics
3. Access to board selector
4. Search for L475 and select board
5. Say No to initialize to default mode
6. Enable USART1
7. Additional Software -> X-CUBE-AI core selection
8. Artificial Intelligence Application to Validation
9. Platform settings: select USART1 for communication
10. Add network, Keras, Saved model, load .h5
11. Select compression, then hit analyze
12. Generate code with Makefile toolchain
13. To copy the generated code from cube to the firmware directory, run in `firmware`
```bash
make clean && make -j8 OPMODE=CUBE_VERIFICATION
```
14. Flash target
```bash
make flash
```
15. In Cube, hit validate on target to get funky stuff