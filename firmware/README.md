# Edison Firmware

## Disclaimer
```
This directory contains source code from other projects. Please refer to the corresponding folders for license notices.
```

## Opmodes
To run NNoM KWS example, compile with
```bash
make OPMODE=NNOM_KWS_EXAMPLE
```

To enable CubeAI verification of the Cube net, compile with
```bash
make OPMODE=CUBE_VERIFICATION
```

## CMSIS DSP lessons learned
`arm_cfft_q15` 

- bit reversal flag must be set for the output to be interpreted.
- FFT output is not in fractional format, hence applying `arm_cmplx_mag_q15` doesn't work


## Host interface

When `hifRun` is run, a host can communicate with the board via UART. Below is a list of commands and their description follows.

Each command is acknowledged by a 1 byte status byte.

| Command byte | N Args | Command                          | Short Description                               |
|:-------------|:-------|:---------------------------------|:------------------------------------------------|
|'0' | 0 | verPrintWrap | Returns version string |
|'1' | 0 | micHostSampleRequestPreprocessedManualWrap | Sample and preprocess mic for manual inspection |
|'2' | 0 | aiPrintInfoWrap | Print network info |
|'3' | 0 | audioHifInfo | Print audio processing info |
|'4' | 0 | appMicMfccInfereContinuous | Start continuous inference |
|'5' | 0 | appMicMfccInfereBlocks | Start continuous inference in block mode, no sliding window |
|0x0 | 2 | micHostSampleRequestWrap | Raw sample mic |
|0x1 | 3 | micHostSampleRequestPreprocessedWrap | Sample and preprocess mic |
|0x2 | 0 | audioMELSingleBatchWrap | MFCC computation |
|0x3 | 0 | aiRunInferenceHifWrap | Run inference |
|0x4 | 0 | appHifMfccAndInference | Upload samples, MCU computes mfcc and inference |
|0x5 | 0 | appHifMicMfccInfere | Run MFCC and inference with data from microphone |

