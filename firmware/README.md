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
| `'0' = 0x30` | 0      | `VERSION`                        | Returns version string                          |
| `'1' = 0x31` | 0      | `MIC_SAMPLE_PREPROCESSED_MANUAL` | Sample and preprocess mic for manual inspection |
| `0x00`       | 2      | `MIC_SAMPLE`                     | Raw sample mic                                  |
| `0x01`       | 3      | `MIC_SAMPLE_PREPROCESSED`        | Sample and preprocess mic                       |


### `VERSION`
Args:
 - None

Return:
 - ASCII Version string

### `MIC_SAMPLE_PREPROCESSED_MANUAL`
Args:
 - None

Return:
 - Binary preprocessed mic samples

Used for debuggung, samlpe count and bit depth coded in `hostinterface.c`.

### `MIC_SAMPLE`
Args:
 - 2 byte sample count (big-endian)

Return:
 - Raw mic samples, nSample\*4 bytes, signed integer big-endian

### `MIC_SAMPLE_PREPROCESSED`
Args:
  - 1 byte bit depth, 8 or 16
  - 2 byte sample count (big-endian)

Return:
 - Preprocessed mic samples, nSample\*(bitDepth/8) bytes, signed integer big-endian

