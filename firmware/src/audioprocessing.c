/*
* @Author: Noah Huetter
* @Date:   2020-04-15 11:33:22
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-04-20 21:31:37
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
/*------------------------------------------------------------------------------
 * Private data
 * ---------------------------------------------------------------------------*/

static arm_rfft_instance_q15 rfft_q15_i;

/**
 * Room for FFT and spectrogram
 */
static q15_t bufFft[2*MEL_SAMPLE_SIZE];
static q15_t bufSpect[MEL_SAMPLE_SIZE];

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
  if(arm_rfft_init_q15(&rfft_q15_i, (uint32_t)MEL_SAMPLE_SIZE, 0, 0 ) != ARM_MATH_SUCCESS)
  {
    Error_Handler();
  }   
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

  // perform real fft
  arm_rfft_q15(&rfft_q15_i, inp, bufFft);

  // ---------------------------------------------------------------
  // [2.] Perform magnitude value for spectrum

  arm_cmplx_mag_q15(bufFft, bufSpect, MEL_SAMPLE_SIZE);
}

/**
 * @brief Dump values to host
 * @details 
 */
void audioDumpToHost(void)
{
  hiSendS16(bufFft, 2*MEL_SAMPLE_SIZE, 0);
  hiSendS16(bufSpect, MEL_SAMPLE_SIZE/2+1, 1);
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