/*
* @Author: Noah Huetter
* @Date:   2020-04-15 11:33:22
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-05-01 16:13:36
*/
#include "audioprocessing.h"

#include "arm_math.h"
#include "arm_const_structs.h"
#include "hostinterface.h"

#include "audio/mel_constants.h"

/*------------------------------------------------------------------------------
 * Types
 * ---------------------------------------------------------------------------*/

typedef struct
{
  uint16_t N;                         /**< length of the DCT2. */
  uint16_t Nby2;                      /**< half of the length of the DCT2. */
  q15_t *pTwiddle;                    /**< points to the twiddle factor table. */
  arm_rfft_instance_q15 *pRfft;        /**< points to the real FFT instance. */
} dct2_instance_q15;


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

static dct2_instance_q15 dct2_q15_i;
static arm_rfft_instance_q15 dct2_rfft_q15_i;

/**
 * Room for FFT and spectrogram
 */
static q15_t bufFft[2*MEL_SAMPLE_SIZE];
static q15_t bufSpect[2*MEL_SAMPLE_SIZE];
// static q31_t bufMelSpectManual[MEL_N_MEL_BINS];
static q15_t bufMelSpect[MEL_N_MEL_BINS];
static q15_t bufDct[2*MEL_N_MEL_BINS];
static q15_t bufDctInline[MEL_N_MEL_BINS];

/*------------------------------------------------------------------------------
 * Prototypes
 * ---------------------------------------------------------------------------*/

static void cmpl_mag_q15 (q15_t * pSrc, q15_t * pDst, uint32_t blockSize);
static void dct2_q15 (const dct2_instance_q15 * S, q15_t * pState, q15_t * pInlineBuffer);

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
void audioCalcMFCCs(int16_t * inp, int16_t ** oup)
{
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
  // [3.] Dot product mel matrix with the positive values of the spectrum
  // couldn't get this to run, but since mel mtx so sparse, should not spend too much time getting it to work
  // arm_matrix_instance_q15 mtxq15A, mtxq15B, mtxq15C;
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
    // bufMelSpectManual[mel] = tmpq31;
    // this shift is determined by experiment and depends on MEL_MTX_SCALE
    bufMelSpect[mel] = (q15_t)(tmpq31>>6);
  }
  
  // ---------------------------------------------------------------
  // [4.] Here would the log(x) calculation come, let's leave it out for now..


  // ---------------------------------------------------------------
  // [5.] DCT-2
  // calc is inplace, so copy to buffer
  arm_copy_q15(bufMelSpect, bufDctInline,MEL_N_MEL_BINS);
  if(arm_rfft_init_q15(&dct2_rfft_q15_i, (uint32_t)MEL_N_MEL_BINS, 0, 1) != ARM_MATH_SUCCESS)
  {
    Error_Handler();
  } 
  dct2_q15_i.N = MEL_N_MEL_BINS;
  dct2_q15_i.Nby2 = MEL_N_MEL_BINS/2;
  dct2_q15_i.pRfft = &dct2_rfft_q15_i;
  dct2_q15(&dct2_q15_i, bufDct, bufDctInline);

  // Store output
  *oup = bufDctInline;
}

/**
 * @brief Dump values to host
 * @details 
 */
void audioDumpToHost(void)
{
  hiSendS16(bufFft, 2*MEL_SAMPLE_SIZE, 0);
  hiSendS16(bufSpect, MEL_SAMPLE_SIZE, 1);
  hiSendS16(bufMelSpect, sizeof(bufMelSpect)/sizeof(q15_t), 2);
  // hiSendS32(bufMelSpectManual, sizeof(bufMelSpectManual)/sizeof(q31_t), 3);
  // hiSendS16(bufDct, sizeof(bufDct)/sizeof(q15_t), 4);
  hiSendS16(bufDctInline, sizeof(bufDctInline)/sizeof(q15_t), 4); // post FFT


}

/**
 * @brief Runs single batch MEL coefficient calcualtion with host interface
 * @details 
 */
void audioMELSingleBatch(void)
{
  uint8_t tag;

  static q15_t in_frame[MEL_SAMPLE_SIZE];
  static q15_t **out_mfcc;

  (void)hiReceive((void *)in_frame, 2*MEL_SAMPLE_SIZE, DATA_FORMAT_S16, &tag);

  audioCalcMFCCs(in_frame, out_mfcc);

  audioDumpToHost();
}

/*------------------------------------------------------------------------------
 * Privates
 * ---------------------------------------------------------------------------*/

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


/**    
 * @brief Processing function for the Q15 DCT2
 * @param[in]       *S             points to an instance of the Q15 DCT2 structure.   
 * @param[in]       *pState        points to state buffer.   
 * @param[in,out]   *pInlineBuffer points to the in-place input and output buffer.   
 * @return none.   
 *     
 * \par Input an output formats:    
 * Internally inputs are downscaled in the RFFT process function to avoid overflows.    
 * Number of bits downscaled, depends on the size of the transform.    
 * The input and output formats for different DCT sizes and number of bits to upscale are mentioned in the table below:     
 *    
 * \image html dct4FormatsQ15Table.gif    
 */
static void dct2_q15(
  const dct2_instance_q15 * S,
  q15_t * pState,
  q15_t * pInlineBuffer)
{
  uint32_t i;                                    /* Loop counter */
  q15_t *pS1, *pS2, *pbuff;                      /* Temporary pointers for input buffer and pState buffer */
  q15_t in;                                      /* Temporary variable */


  /* DCT4 computation involves DCT2 (which is calculated using RFFT)    
   * along with some pre-processing and post-processing.    
   * Computational procedure is explained as follows:    
   * (b) Calculation of DCT2 using FFT is divided into three steps:    
   *                  Step1: Re-ordering of even and odd elements of input.    
   *                  Step2: Calculating FFT of the re-ordered input.    
   *                  Step3: Taking the real part of the product of FFT output and weights.    
   */

  /* ----------------------------------------------------------------    
   * Step1: Re-ordering of even and odd elements as    
   *             pState[i] =  pInlineBuffer[2*i] and    
   *             pState[N-i-1] = pInlineBuffer[2*i+1] where i = 0 to N/2    
   ---------------------------------------------------------------------*/

  /* pS1 initialized to pState */
  pS1 = pState;

  /* pS2 initialized to pState+N-1, so that it points to the end of the state buffer */
  pS2 = pState + (S->N - 1u);

  /* pbuff initialized to input buffer */
  pbuff = pInlineBuffer;

  /* Initializing the loop counter to N/2 >> 2 for loop unrolling by 4 */
  i = (uint32_t) S->Nby2 >> 2u;

  /* First part of the processing with loop unrolling.  Compute 4 outputs at a time.    
   ** a second loop below computes the remaining 1 to 3 samples. */
  do
  {
    /* Re-ordering of even and odd elements */
    /* pState[i] =  pInlineBuffer[2*i] */
    *pS1++ = *pbuff++;
    /* pState[N-i-1] = pInlineBuffer[2*i+1] */
    *pS2-- = *pbuff++;

    *pS1++ = *pbuff++;
    *pS2-- = *pbuff++;

    *pS1++ = *pbuff++;
    *pS2-- = *pbuff++;

    *pS1++ = *pbuff++;
    *pS2-- = *pbuff++;

    /* Decrement the loop counter */
    i--;
  } while(i > 0u);

  /* pbuff initialized to input buffer */
  pbuff = pInlineBuffer;

  /* pS1 initialized to pState */
  pS1 = pState;

  /* Initializing the loop counter to N/4 instead of N for loop unrolling */
  i = (uint32_t) S->N >> 2u;

  /* Processing with loop unrolling 4 times as N is always multiple of 4.    
   * Compute 4 outputs at a time */
  do
  {
    /* Writing the re-ordered output back to inplace input buffer */
    *pbuff++ = *pS1++;
    *pbuff++ = *pS1++;
    *pbuff++ = *pS1++;
    *pbuff++ = *pS1++;

    /* Decrement the loop counter */
    i--;
  } while(i > 0u);

  /* ---------------------------------------------------------    
   *     Step2: Calculate RFFT for N-point input    
   * ---------------------------------------------------------- */
  /* pInlineBuffer is real input of length N , pState is the complex output of length 2N */
  arm_rfft_q15(S->pRfft, pInlineBuffer, pState);

 /*----------------------------------------------------------------------    
  *  Step3: Multiply the FFT output with the weights.    
  *  this is note done, I got expected result without this step
  *----------------------------------------------------------------------*/
  // arm_cmplx_mult_cmplx_q15(pState, weights, pState, S->N);

  /* The output of complex multiplication is in 3.13 format.    
   * Hence changing the format of N (i.e. 2*N elements) complex numbers to 1.15 format by shifting left by 2 bits. */
  // arm_shift_q15(pState, 2, pState, S->N * 2);

 /*----------------------------------------------------------------------    
  *  Step4: Take only the real part
  *----------------------------------------------------------------------*/
  i = (uint32_t) S->N;
  pbuff = pState;
  pS1 = pInlineBuffer;
  do
  {
    *pS1++ = *pbuff++;
    in     = *pbuff++; // discard imaginary part
    i--;
  } while(i > 0u);
  (void)in;
}


/*------------------------------------------------------------------------------
 * Callbacks
 * ---------------------------------------------------------------------------*/