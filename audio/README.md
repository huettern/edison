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



