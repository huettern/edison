# Audio Utilities

A collection of scripts to test and train.

| File | Does |
|--|--|
| `bit_depth_analyze.py` | Takes samples from text file and creates different bit with outputs. Used for determining the required bit depth for voice recognition |
| `fetch_mic_samples.py` | Interfaces with the microcontroller to fetch raw microphone data |
| `mfcc.py` | Experiment with Mel Frequency cepstral coefficients |
| `mfcc_utils.py` | Library to calculate MFCCs |
| `mfcc_on_mcu.py` | Testing MFCC implementation on MCU |

## Environment
Setup once:
```bash
virtualenv -p python3.7 venv
source venv/bin/activate
pip install -r requirements.txt
```

On marvin
```bash
sudo apt install python3-libnvinfer python3-libnvinfer-dev
```

## Speechrecognizer

### Inspirations
https://medium.com/manash-en-blog/building-a-dead-simple-word-recognition-engine-using-convnet-in-keras-25e72c19c12b


mcu
  hif_test
mic
  bit_depth_analyze
```bash
./main.py mic bit_depth_analyze
```
  fetch_mic_samples
```bash
./main.py mic fetch_mic_samples
```

mfcc
  host
  mcu
```bash
# Calculate header file for MCU processing
./main.py mfcc mcu calc
# single frame MFCC on mcu and host with detailed vectors
./main.py mfcc mcu single
# Multi-frame on host and mcu with input/output compare
./main.py mfcc mcu file data/heysnips_true_16k_16b.wav
```
acqiure
  sample_acq
train
  keras
  legacy
  nnom
  (torch)
kws
  live
  mcu
deploy
  nnom


Folder structure


edison.py
config,py
edison
  audio
    - bit_depth_analyze.py
    - audioutils.py
  mcu
    - mcu_util.py
    - fetch_mic_samples.py
    - hif_test.py
  mfcc
    - mfcc_utils.py
    - mfcc.py
    - mfcc_on_mcu.py
  acquire
    - sample_acq.py
  train
    - allinone.py
    - kws_keras.py
    - kws_legacy.py
    - kws_nnom.py
    - visualize.py
  kws
    - kws_live.py
    - kws_on_mcu.py
  deploy
    - nnom repo
    - nnom.py










































