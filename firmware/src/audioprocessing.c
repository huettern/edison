/*
* @Author: Noah Huetter
* @Date:   2020-04-15 11:33:22
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-04-22 20:25:05
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
// #define USE_REAL_FFT

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
static q31_t bufMelSpect[MEL_N_MEL_BINS];

/*------------------------------------------------------------------------------
 * Prototypes
 * ---------------------------------------------------------------------------*/

static void cmpl_mag_sqd_q15 (q15_t * pSrc, q15_t * pDst, uint32_t blockSize);
static void cmpl_mag_q15 (q15_t * pSrc, q15_t * pDst, uint32_t blockSize);

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
  arm_matrix_instance_q15 mtxq15A, mtxq15B, mtxq15C;
  q31_t tmpq31;

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
  cmpl_mag_q15(bufFft, bufSpect, 2*MEL_SAMPLE_SIZE);

  // ---------------------------------------------------------------
  // [3.] Dot product xxx of the mel matrix with the positive values of
  // the spectrum
  // mtxq15A.numRows = 1; mtxq15A.numCols = MEL_SAMPLE_SIZE/2+1; mtxq15A.pData = bufSpect;
  // mtxq15B.numRows = MEL_MTX_ROWS; mtxq15B.numCols = MEL_MTX_COLS; mtxq15B.pData = melMtx;
  // mtxq15C.numRows = 1; mtxq15C.numCols = MEL_N_MEL_BINS; mtxq15C.pData = bufMelSpect;
  // arm_mat_mult_q15(&mtxq15A, &mtxq15B, &mtxq15C, NULL);

  // manual
  for(int mel = 0; mel < MEL_N_MEL_BINS; mel++)
  {
    tmpq31 = 0;
    for(int frq = 0; frq < MEL_SAMPLE_SIZE/2+1; frq++)
    {
      tmpq31 += bufSpect[frq] * melMtx[frq][mel];
    }
    bufMelSpect[mel] = tmpq31;
  }
  
}

/**
 * @brief Dump values to host
 * @details 
 */
void audioDumpToHost(void)
{
  hiSendS16(bufFft, 2*MEL_SAMPLE_SIZE, 0);
  hiSendS16(bufSpect, MEL_SAMPLE_SIZE, 1);
  hiSendS32(bufMelSpect, sizeof(bufMelSpect)/sizeof(q31_t), 2);

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

  // test

  // q31_t in1, in2, in3, in4;
  // q31_t acc0, acc1;
  // q31_t acc2, acc3;
  // q15_t * pSrc = bufFft;
  // q15_t * pDst= bufSpect;

  bufFft[0]   = -2;
  bufFft[0+1] = 0;
  bufFft[0+2] = -2;
  bufFft[0+3] = -3;
  cmpl_mag_sqd_q15(&bufFft[0], bufSpect, 2);
  printf("|%5d + j(%5d) | = %5d\n", bufFft[0], bufFft[1], bufSpect[0]);
  printf("|%5d + j(%5d) | = %5d\n", bufFft[2], bufFft[3], bufSpect[1]);
  cmpl_mag_q15(&bufFft[0], bufSpect, 2);
  printf("|%5d + j(%5d) | = %5d\n", bufFft[0], bufFft[1], bufSpect[0]);
  printf("|%5d + j(%5d) | = %5d\n------\n", bufFft[2], bufFft[3], bufSpect[1]);

  // bufFft[0]   = 1024;
  // bufFft[0+1] = 1024;
  // bufFft[0+2] = 2048;
  // bufFft[0+3] = 1024;
  // cmpl_mag_sqd_q15(&bufFft[0], bufSpect, 2);
  // printf("|%5d + j(%5d) | = %5d\n", bufFft[0], bufFft[1], bufSpect[0]);
  // printf("|%5d + j(%5d) | = %5d\n", bufFft[2], bufFft[3], bufSpect[1]);
  // cmpl_mag_q15(&bufFft[0], bufSpect, 2);
  // printf("|%5d + j(%5d) | = %5d\n", bufFft[0], bufFft[1], bufSpect[0]);
  // printf("|%5d + j(%5d) | = %5d\n------\n", bufFft[2], bufFft[3], bufSpect[1]);


  // bufFft[0]   = 0x7fff;
  // bufFft[0+1] = 0x7fff;
  // bufFft[0+2] = 0x8000;
  // bufFft[0+3] = 0x8000;
  // cmpl_mag_sqd_q15(&bufFft[0], bufSpect, 2);
  // printf("|%5d + j(%5d) | = %5d\n", bufFft[0], bufFft[1], bufSpect[0]);
  // printf("|%5d + j(%5d) | = %5d\n", bufFft[2], bufFft[3], bufSpect[1]);
  // cmpl_mag_q15(&bufFft[0], bufSpect, 2);
  // printf("|%5d + j(%5d) | = %5d\n", bufFft[0], bufFft[1], bufSpect[0]);
  // printf("|%5d + j(%5d) | = %5d\n------\n", bufFft[2], bufFft[3], bufSpect[1]);


  // for(uint16_t i = 0; i < 10*1024; i+=2)
  // {
  //   bufFft[i]   = i;
  //   bufFft[i+1] = 0;
  //   bufFft[i+2] = i+1;
  //   bufFft[i+3] = 0;
  //   arm_cmplx_mag_q15(&bufFft[i], bufSpect, 2);
  //   printf("|%5d + j(%5d) | = %5d\n", bufFft[i], bufFft[i+1], bufSpect[0]);
  //   printf("|%5d + j(%5d) | = %5d\n", bufFft[i+2], bufFft[i+3], bufSpect[1]);
  //   HAL_Delay(200);
  // }

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
/**
 * @brief Take blockSize complex numbers stored [real, imag, real, image, ..] and calculate
 * the squared magnitude
 * @details 
 * 
 * @param pSrc source
 * @param pDst destination
 * @param blockSize number of complex input numbers
 */
static void cmpl_mag_sqd_q15 (q15_t * pSrc, q15_t * pDst, uint32_t blockSize)
{
  q15_t in;
  q31_t sum;
  while(blockSize--)
  {
    in = *pSrc++;
    sum = ((q31_t) in * in);
    in = *pSrc++;
    sum += ((q31_t) in * in);
    *pDst++ = (q15_t)(sum>>16);
  }
}
/**
 * @brief Take blockSize complex numbers stored [real, imag, real, image, ..] and calculate
 * the magnitude
 * @details 
 * 
 * @param pSrc source
 * @param pDst destination
 * @param blockSize number of complex input numbers
 */
static void cmpl_mag_q15 (q15_t * pSrc, q15_t * pDst, uint32_t blockSize)
{
  q15_t in;
  q31_t sum;
  while(blockSize--)
  {
    in = *pSrc++;
    sum = ((q31_t) in * in);
    in = *pSrc++;
    sum += ((q31_t) in * in);
    arm_sqrt_q31(sum, &sum);
    *pDst++ = (q15_t)(sum>>16);
  }
}

/*------------------------------------------------------------------------------
 * Callbacks
 * ---------------------------------------------------------------------------*/