/*
* @Author: Noah Huetter
* @Date:   2020-04-15 11:33:22
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-04-21 19:32:48
*/
#include "audioprocessing.h"

#include "arm_math.h"
#include "arm_const_structs.h"
#include "hostinterface.h"

#include "audio/mel_constants.h"

/*------------------------------------------------------------------------------
 * Types
 * ---------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------
 * Settings
 * ---------------------------------------------------------------------------*/

/**
 * Uncomment to use real FFT functions. They dont produce the same clean results
 * as the complex FFT
 */
#define USE_REAL_FFT

/*------------------------------------------------------------------------------
 * Private data
 * ---------------------------------------------------------------------------*/
#ifdef USE_REAL_FFT
static arm_rfft_instance_q15 rfft_q15_i;
#endif

/**
 * Room for FFT and spectrogram
 */
static q15_t bufFft[2*MEL_SAMPLE_SIZE];
static q15_t bufSpect[2*MEL_SAMPLE_SIZE];

/*------------------------------------------------------------------------------
 * Prototypes
 * ---------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------
 * Publics
 * ---------------------------------------------------------------------------*/

/**
 * @brief Init data structures
 * @details 
 */
void audioInit(void)
{
  // initialize fft
  #ifdef USE_REAL_FFT
  if(arm_rfft_init_q15(&rfft_q15_i, (uint32_t)MEL_SAMPLE_SIZE, 0, 1) != ARM_MATH_SUCCESS)
  {
    Error_Handler();
  }   
  #endif
}

/**
 * @brief Calculate MFCCs from a fixed input sample length
 * @details [long description]
 * 
 * @param inp [description]
 */
void audioCalcMFCCs(int16_t * inp, int16_t * oup)
{
  // ---------------------------------------------------------------
  // [1.] Calculate FFT
#ifdef USE_REAL_FFT
  // perform real fft
  // for 1024 size, output is downscaled 9 bits to avoid saturation
  arm_rfft_q15(&rfft_q15_i, inp, bufFft);
  // shift back a bit..
  arm_shift_q15(bufFft, 4, bufFft, 2*MEL_SAMPLE_SIZE);
#else
  // empty buffer and copy input to real samples
  for(int i = 0; i < 2*MEL_SAMPLE_SIZE; i++)
    bufFft[i] = 0;
  for(int i = 0; i < 2*MEL_SAMPLE_SIZE; i+=2)
    bufFft[i] = inp[i/2];
  arm_cfft_q15(&arm_cfft_sR_q15_len1024, bufFft, 0, 1);
#endif

  // ---------------------------------------------------------------
  // [2.] Perform magnitude value for spectrum
  arm_cmplx_mag_q15(bufFft, bufSpect, 2*MEL_SAMPLE_SIZE);
  // arm_shift_q15(bufSpect, -1, bufSpect, 2*MEL_SAMPLE_SIZE);
}

/**
 * @brief Dump values to host
 * @details 
 */
void audioDumpToHost(void)
{
  hiSendS16(bufFft, 2*MEL_SAMPLE_SIZE, 0);
  hiSendS16(bufSpect, MEL_SAMPLE_SIZE, 1);
}

/**
 * @brief Function used during development
 * @details 
 */
void audioDevelop(void)
{
  static q15_t in_frame[MEL_SAMPLE_SIZE];
  static q15_t out_mfcc[MEL_N_MEL_BINS];
  uint32_t len;
  uint8_t tag;

  while(1)
  {
    len = hiReceive((void *)in_frame, 2*MEL_SAMPLE_SIZE, DATA_FORMAT_S16, &tag);

    audioCalcMFCCs(in_frame, out_mfcc);

    audioDumpToHost();
  }
}

/*------------------------------------------------------------------------------
 * Privates
 * ---------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------
 * Callbacks
 * ---------------------------------------------------------------------------*/