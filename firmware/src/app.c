/*
* @Author: Noah Huetter
* @Date:   2020-04-15 11:16:05
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-05-15 20:43:33
*/
#include "app.h"
#include <stdlib.h>
#include "arm_math.h"

#include "printf.h"
#include "microphone.h"
#include "ai.h"
#include "audioprocessing.h"
#include "hostinterface.h"
#include "cyclecounter.h"
#include "mfcc.h"
#include "led.h"

/*------------------------------------------------------------------------------
 * Types
 * ---------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------
 * Settings
 * ---------------------------------------------------------------------------*/

/**
 * @brief Consider network output above this thershold as hit
 */
#define TRUE_THRESHOLD 0.9
#define NET_OUT_MOVING_AVG_ALPHA 0.5

#define AMPLITUDE_MOVING_AVG_ALPHA 0.9

/**
 * @brief Enable this to show profiling on arduino Tx pin
 */
#define CYCLE_PROFILING

#ifdef CYCLE_PROFILING
  #define prfStart(x) cycProfStart(x)
  #define prfEvent(x) cycProfEvent(x)
  #define prfStop() cycProfStop()
#else
  #define prfStart(x)
  #define prfEvent(x)
  #define prfStop()
#endif

/*------------------------------------------------------------------------------
 * Private data
 * ---------------------------------------------------------------------------*/

static uint8_t netInputChunk[AI_NET_INSIZE_BYTES];
static uint8_t netOutputChunk[AI_NET_OUTSIZE_BYTES];

#if NET_TYPE == NET_TYPE_CUBE
  static float * netInput = (float*)netInputChunk;
  static float * netOutput = (float*)netOutputChunk;
#elif NET_TYPE == NET_TYPE_NNOM
  static int8_t * netInput = (int8_t*)netInputChunk;
  static int8_t * netOutput = (int8_t*)netOutputChunk;
#endif


static FASTRAM_BSS int16_t tmpBuf[1024*16]; // fills entire region

#define IN_FRAME_BUF_N_FRAMES 16
static int16_t * inFrameBuf = tmpBuf;

static volatile uint8_t audioEvent = 0;
static volatile uint32_t processedFrames;
static volatile float lastAmplitude;
static uint16_t in_x, in_y;

static float netOutFilt[AI_NET_OUTSIZE];

/*------------------------------------------------------------------------------
 * Prototypes
 * ---------------------------------------------------------------------------*/
void mfccToNetInput(int16_t* mfcc, uint16_t in_x, uint16_t in_y, uint32_t xoffset);
void mfccToNetInputPush(int16_t* mfcc, uint16_t in_x, uint16_t in_y);

/*------------------------------------------------------------------------------
 * Publics
 * ---------------------------------------------------------------------------*/
/**
 * @brief Receive samples from host, calculate mfcc and run inference
 * @details 
 * 
 * @param args 
 * @return 
 */
int8_t appHifMfccAndInference(uint8_t *args)
{
  (void)args;
  int ret;
  uint32_t tmp32 = 0;
  
  int16_t *out_mfccs;
  
  uint8_t tag;

  prfStart("appHifMfccAndInference");

  // get net info
  aiGetInputShape(&in_x, &in_y);

  prfEvent("aiGetInputShape");

  for(int frameCtr = 0; frameCtr < in_y; frameCtr++)
  {
    // 2. Calculate MFCC
    ret = hiReceive(inFrameBuf, AUD_MFCC_FRAME_SIZE_BYES, DATA_FORMAT_S16, &tag);    
    printf("received ret %d tag %d \n", ret, tag);

    audioCalcMFCCs(inFrameBuf, &out_mfccs);

    // copy to net in buffer and cast to float
    mfccToNetInput(out_mfccs, in_x, in_y, frameCtr);

    // signal host that we are ready
    hiSendMCUReady();
  }

  prfEvent("receive and MFCC");

  // 3. Run inference
  printf("inference..");
  ret = aiRunInference((void*)netInput, (void*)netOutput);
  printf("Prediction: ");
  for(tmp32 = 0; tmp32 < AI_NET_OUTSIZE; tmp32++) {printf("%.2f ", (float)netOutput[tmp32]);}
    printf("\n");
  prfEvent("inference");
  hiSendMCUReady();

  // 4. report back mfccs and net out
  #if NET_TYPE == NET_TYPE_CUBE
    hiSendF32(netInput, AI_NET_INSIZE, 0x20);
    hiSendF32(netOutput, AI_NET_OUTSIZE, 0x21);
  #elif NET_TYPE == NET_TYPE_NNOM
    hiSendS8(netInput, AI_NET_INSIZE, 0x20);
    hiSendS8(netOutput, AI_NET_OUTSIZE, 0x21);
  #endif
  
  prfStop();
  return ret;
}

/**
 * @brief Collects samples from mic, runs mfcc and inference while reporting 
 * data to host
 * @details 
 * 
 * @param args 
 * @return 
 */

int8_t appHifMicMfccInfere(uint8_t *args)
{
  (void) args;

  int ret;
  uint32_t tmp32 = 0;

  void* netInSnapshot = malloc(AI_NET_INSIZE_BYTES);

  // get net info
  aiGetInputShape(&in_x, &in_y);

  // start continuous mic sampling
  processedFrames = 0;
  micContinuousStart();

  // fill buffer once
  while(processedFrames < in_y);

  // stop sampling
  micContinuousStop();

  // 3. Run inference
  memcpy(netInSnapshot, netInput, AI_NET_INSIZE_BYTES);
  printf("inference..");
  ret = aiRunInference((void*)netInSnapshot, (void*)netOutput);

  printf("Prediction: ");
  for(tmp32 = 0; tmp32 < AI_NET_OUTSIZE; tmp32++) {printf("%2.2f ", (float)netOutput[tmp32]);}
    printf("\n");

  // signal host that we are ready
  hiSendMCUReady();

  // 4. report back mfccs and net out
  hiSendS16(inFrameBuf, 1024*16, 0x30);
  #if NET_TYPE == NET_TYPE_CUBE
    hiSendF32(netInput, AI_NET_INSIZE, 0x31);
    hiSendF32(netOutput, AI_NET_OUTSIZE, 0x32);
  #elif NET_TYPE == NET_TYPE_NNOM
    hiSendS8(netInSnapshot, AI_NET_INSIZE, 0x31);
    hiSendS8(netOutput, AI_NET_OUTSIZE, 0x32);
  #endif

  free(netInSnapshot);

  return ret;
}

/**
 * @brief Runs mic sample - mfcc - inference until button press or uart receive
 * @details 
 * 
 * @param args 
 * @return 
 */
int8_t appMicMfccInfereContinuous (uint8_t *args)
{
  static float netOutFloat[AI_NET_OUTSIZE];
  uint32_t tmp32;
  // uint32_t netInBufOff = 0;
  bool doAbort = false;
  int ret;
  float tmpf;

  void* netInSnapshot = malloc(AI_NET_INSIZE_BYTES);

  // clear net out buff filt
  for(int i = 0; i < AI_NET_OUTSIZE; i++) netOutFilt[i] = 0.0;

  mainSetPrintfUart(&huart1);
  aiGetInputShape(&in_x, &in_y); // x = 13, y = 62 (nframes)
  printf("Input shape x,y: (%d,%d)\n", in_x, in_y);

  // start continuous mic sampling
  processedFrames = 0;
  micContinuousStart();

  // fill buffer once
  while(processedFrames < in_y);

  // now enter a loop where sampling and inference is done simultaneously
  while(!doAbort)
  {
    // wait for ISR to complete before starting
    audioEvent = 0;
    while(!audioEvent);

    // 3. Run inference
    memcpy(netInSnapshot, netInput, AI_NET_INSIZE_BYTES);
    ret = aiRunInference((void*)netInSnapshot, (void*)netOutput);

    // store net input
    // for(int i = 0; i < in_x*in_y; i++)
    //   netInBuf[i+netInBufOff] = netInput[i];
    // netInBufOff += in_x*in_y;

    // report
    printf("pred: [ ");
    for(tmp32 = 0; tmp32 < AI_NET_OUTSIZE; tmp32++)
    {
      netOutFloat[tmp32] = (float)(netOutput[tmp32]);
      printf("%2.2f ", netOutFloat[tmp32]);
    }
    printf("] ret: %d ampl: %.0f", ret, lastAmplitude);

    // moving average filter on net output
    for(int i = 0; i < AI_NET_OUTSIZE; i++) netOutFilt[i] = (NET_OUT_MOVING_AVG_ALPHA*netOutFilt[i] + (1.0-NET_OUT_MOVING_AVG_ALPHA)*netOutFloat[i]);

    arm_max_f32(netOutFilt, AI_NET_OUTSIZE, &tmpf, &tmp32);
    printf(" likely: %s", aiGetKeywordFromIndex(tmp32));
    if( (tmpf > TRUE_THRESHOLD) )
    {
      printf(" spotted %s", aiGetKeywordFromIndex(tmp32));
      ledSet(1<<tmp32);
    }
    else
    {
      ledSet(0);
    }
    printf("\n");

    if(netOutFilt[0] > TRUE_THRESHOLD) LED2_ORA();
    else LED2_BLU();

    // check abort condition
    if(IS_BTN_PRESSED() || (huart1.Instance->ISR & UART_FLAG_RXNE) ) doAbort = true;

    // display amplitude
    ledSetColor(0, ((uint16_t)(lastAmplitude)>>8)/2, (255-((uint16_t)(lastAmplitude)>>8))/2, 0);
    ledUpdate(0);
    // if(netInBufOff > 12*in_x*in_y) doAbort = true;
  }

  // stop sampling
  micContinuousStop();

  // flush input
  (void)huart1.Instance->RDR;

  // send net input history
  // hiSendF32(netInBuf, 12*AI_NET_INSIZE, 0x20);

  free(netInSnapshot);
  ledSet(0);
  mainSetPrintfUart(&huart4);

  return ret;
}

/**
 * @brief Runs mic sample - mfcc - inference until button press or uart receive
 * @details not shifting but starting entire net input with new samples
 * 
 * @param args 
 * @return 
 */
int8_t appMicMfccInfereBlocks (uint8_t *args)
{
  static float netOutFloat[AI_NET_OUTSIZE];
  int16_t *inFrame, *out_mfccs;
  uint16_t in_x, in_y;
  uint32_t tmp32;
  // int32_t netInBufOff = 0;
  bool doAbort = false;
  int ret;
  float tmpf;

  aiGetInputShape(&in_x, &in_y); // x = 13, y = 62 (nframes)
  printf("Input shape x,y: (%d,%d)\n", in_x, in_y);

  // start continuous mic sampling
  processedFrames = 0;
  micContinuousStart();

  // now enter a loop where sampling and inference is done simultaneously
  while(!doAbort)
  {

    // fill buffer once
    for (int frameCtr = 0; (frameCtr < in_y) && !doAbort; frameCtr++)
    {
      // get samples, this call is blocking
      inFrame = micContinuousGet();

      // calc mfccs
      audioCalcMFCCs(inFrame, &out_mfccs); //*inp, **oup

      // copy to net in buffer and cast to float
      mfccToNetInput(out_mfccs, in_x, in_y, frameCtr);

      if(IS_BTN_PRESSED() || (huart1.Instance->ISR & UART_FLAG_RXNE) ) doAbort = true;
    }

    // 3. Run inference
    ret = aiRunInference((void*)netInput, (void*)netOutput);

    // store net input
    // for(int i = 0; i < in_x*in_y; i++)
    //   netInBuf[i+netInBufOff] = netInput[i];
    // netInBufOff += in_x*in_y;

    // report
    printf("pred: [ ");
    for(tmp32 = 0; tmp32 < AI_NET_OUTSIZE; tmp32++)
    {
      netOutFloat[tmp32] = (float)(netOutput[tmp32]);
      printf("%2.2f ", netOutFloat[tmp32]);
    }
    printf("] ret: %d ampl: %f", ret, lastAmplitude);

    arm_max_f32(netOutFloat, AI_NET_OUTSIZE, &tmpf, &tmp32);
    printf(" likely: %s", aiGetKeywordFromIndex(tmp32));
    if( (tmpf > TRUE_THRESHOLD) )
    {
      printf(" SPOTTED %s", aiGetKeywordFromIndex(tmp32));
    }
    printf("\n");

    if(netOutFloat[0] > TRUE_THRESHOLD) LED2_ORA();
    else LED2_BLU();

    // check abort condition
    if(IS_BTN_PRESSED() || (huart1.Instance->ISR & UART_FLAG_RXNE) ) doAbort = true;

    // if(netInBufOff > 12*in_x*in_y) doAbort = true;
  }

  // stop sampling
  micContinuousStop();

  // flush input
  (void)huart1.Instance->RDR;

  // send net input history
  // hiSendF32(netInBuf, 12*AI_NET_INSIZE, 0x20);

  return ret;
}

/*------------------------------------------------------------------------------
 * NNom example
 * ---------------------------------------------------------------------------*/
#ifdef NNOM_KWS_EXAMPLE

static const char label_name[][10] =  {"backward", "bed", "bird", "cat", "dog", "down", "eight","five", "follow", "forward",
                      "four", "go", "happy", "house", "learn", "left", "marvin", "nine", "no", "off", "on", "one", "right",
                      "seven", "sheila", "six", "stop", "three", "tree", "two", "up", "visual", "yes", "zero", "unknow"};

// configuration
#define SAMP_FREQ 16000
#define AUDIO_FRAME_LEN (512) //31.25ms * 16000hz = 512, // FFT (windows size must be 2 power n)
//the mfcc feature for kws
#define MFCC_LEN      (63)
#define MFCC_COEFFS_FIRST (1)   // ignore the mfcc feature before this number
#define MFCC_COEFFS_LEN   (13)    // the total coefficient to calculate
#define MFCC_COEFFS         (MFCC_COEFFS_LEN-MFCC_COEFFS_FIRST)
#define MFCC_FEAT_SIZE  (MFCC_LEN * MFCC_COEFFS)

#define SaturaLH(N, L, H) (((N)<(L))?(L):(((N)>(H))?(H):(N)))

static mfcc_t * mfcc;
static int32_t dma_audio_buffer[AUDIO_FRAME_LEN*2];
static int16_t audio_buffer_16bit[(int)(AUDIO_FRAME_LEN*1.5)]; // an easy method for 50% overlapping
static int8_t mfcc_features[MFCC_LEN][MFCC_COEFFS];   // ring buffer
static int8_t mfcc_features_seq[MFCC_LEN][MFCC_COEFFS]; // sequencial buffer for neural network input. 
static uint32_t mfcc_feat_index = 0;
static bool mfccReady = false;

// msh debugging controls
static bool is_print_abs_mean = false; // to print the mean of absolute value of the mfcc_features_seq[][]
static bool is_print_mfcc  = false;    // to print the raw mfcc features at each update 


static int32_t abs_mean(int8_t *p, size_t size)
{
  int64_t sum = 0;
  for(size_t i = 0; i<size; i++)
  {
    if(p[i] < 0)
      sum+=-p[i];
    else
      sum += p[i];
  }
  return sum/size;
}

/**
 * @brief Init stuff for the Nnom KWS example from https://github.com/majianjia/nnom/tree/master/examples/keyword_spotting
 * @details 
 */
void appNnomKwsInit(void)
{
  // calculate 13 coefficient, use number #2~13 coefficient. discard #1
  mfcc = mfcc_create(MFCC_COEFFS_LEN, MFCC_COEFFS_FIRST, AUDIO_FRAME_LEN, 8, 0.97f); 
  HAL_DFSDM_FilterRegularStart_DMA(&hdfsdm1_filter0, dma_audio_buffer, 1024);
  aiNnomInit();
}

void appNnomKwsRun(void)
{
  int32_t *p_raw_audio;
  uint32_t last_mfcc_index = 0; 
  uint32_t label;
  float prob;

  while(1)
  {
    // wait for event and check which buffer is filled
    while(!audioEvent);

    if(audioEvent & 1)
      p_raw_audio = dma_audio_buffer;
    else
      p_raw_audio = &dma_audio_buffer[AUDIO_FRAME_LEN];

    audioEvent = 0;

    // memory move
    // audio buffer = | 256 byte old data |   256 byte new data 1 | 256 byte new data 2 | 
    //                         ^------------------------------------------|
    memcpy(audio_buffer_16bit, &audio_buffer_16bit[AUDIO_FRAME_LEN], (AUDIO_FRAME_LEN/2)*sizeof(int16_t));

    // convert it to 16 bit. 
    // volume*4
    for(int i = 0; i < AUDIO_FRAME_LEN; i++)
    {
      audio_buffer_16bit[AUDIO_FRAME_LEN/2+i] = SaturaLH((p_raw_audio[i] >> 8)*1, -32768, 32767);
    }

    // MFCC
    // do the first mfcc with half old data(256) and half new data(256)
    // then do the second mfcc with all new data(512). 
    // take mfcc buffer
    
    for(int i=0; i<2; i++)
    {
      mfcc_compute(mfcc, &audio_buffer_16bit[i*AUDIO_FRAME_LEN/2], mfcc_features[mfcc_feat_index]);
      
      // debug only, to print mfcc data on console
      if(is_print_mfcc)
      {
        for(int i=0; i<MFCC_COEFFS; i++)
          printf("%d ",  mfcc_features[mfcc_feat_index][i]);
        printf("\n");
      }
      
      mfcc_feat_index++;
      if(mfcc_feat_index >= MFCC_LEN)
        mfcc_feat_index = 0;
    }
    mfccReady = true;

    // copy mfcc ring buffer to sequance buffer. 
    last_mfcc_index = mfcc_feat_index;
    uint32_t len_first = MFCC_FEAT_SIZE - mfcc_feat_index * MFCC_COEFFS;
    uint32_t len_second = mfcc_feat_index * MFCC_COEFFS;
    memcpy(&mfcc_features_seq[0][0], &mfcc_features[0][0] + len_second,  len_first);
    memcpy(&mfcc_features_seq[0][0] + len_first, &mfcc_features[0][0], len_second);
    
    // debug only, to print the abs mean of mfcc output. use to adjust the dec bit (shifting)
    // of the mfcc computing. 
    if(is_print_abs_mean)
      printf("abs mean:%d\n", abs_mean((int8_t*)mfcc_features_seq, MFCC_FEAT_SIZE));
    
    // ML
    memcpy(aiNnomGetInputBuffer(), mfcc_features_seq, MFCC_FEAT_SIZE);
    int ret = aiNnomPredict(&label, &prob);
    printf("ret: %d\n", ret);
    
    // output
    // if(prob > 0.5f)
    // {
      // last_time = rt_tick_get();
      printf("%s : %d%%\n", (char*)&label_name[label], (int)(prob * 100));
    // }
  }
}
#endif // NNOM example

/*------------------------------------------------------------------------------
 * Callback from microphone ISR
 * ---------------------------------------------------------------------------*/
/**
 * @brief Called after DFSDM sample is done
 * @details 
 * 
 * @param c 1 or 2
 */
void appAudioEvent(uint8_t evt, int16_t *buf)
{
  int16_t *inFrame, *out_mfccs, max, min;
  uint32_t index;
  static uint32_t inFrameLoc = 0;

  audioEvent = evt;

  inFrame = buf;

  arm_max_q15(inFrame, MIC_FRAME_SIZE, &max, &index);
  arm_min_q15(inFrame, MIC_FRAME_SIZE, &min, &index);
  lastAmplitude = AMPLITUDE_MOVING_AVG_ALPHA*lastAmplitude + (1.0-AMPLITUDE_MOVING_AVG_ALPHA)*((float)max - (float)min);

  // calc mfccs
  audioCalcMFCCs(inFrame, &out_mfccs); //*inp, **oup

  // copy to net in buffer and cast to float
  mfccToNetInputPush(out_mfccs, in_x, in_y);

  // push frame to frame buffer
  int16_t *dst, *src;
  dst = &inFrameBuf[0]; src = &inFrameBuf[MIC_FRAME_SIZE];
  for(int i = 0; i < (IN_FRAME_BUF_N_FRAMES-1)*MIC_FRAME_SIZE; i++) *dst++ = *src++;
  for(int i = 0; i < MIC_FRAME_SIZE; i++) *dst++ = inFrame[i];

  processedFrames++;
}

/*------------------------------------------------------------------------------
 * Privates
 * ---------------------------------------------------------------------------*/
/**
 * @brief Copies mfcc to net input buffer depending on used network
 * @details 
 * 
 * @param mfcc pointer to MFCC buffer
 * @param in_x net x size
 * @param in_y net y size
 * @param xoffset x offset in network input
 */
void mfccToNetInput(int16_t* mfcc, uint16_t in_x, uint16_t in_y, uint32_t xoffset)
{ 

  // copy to net in buffer and cast to float
#if NET_TYPE == NET_TYPE_CUBE
  for(int mfccCtr = 0; mfccCtr < in_x; mfccCtr++)
  {
    netInput[xoffset*in_x + mfccCtr] = (float)mfcc[mfccCtr];
  }
#elif NET_TYPE == NET_TYPE_NNOM
  int16_t tmps16;
  for(int mfccCtr = 0; mfccCtr < in_x; mfccCtr++)
  { 
    // scale and clip MFCCs
    tmps16 = mfcc[mfccCtr] / 16;
    tmps16 = (tmps16 >  127) ?  127 : tmps16;
    tmps16 = (tmps16 < -128) ? -128 : tmps16;
    netInput[xoffset*in_x + mfccCtr] = (int8_t)(tmps16);
  }
#endif
}

/**
 * @brief Net dependent rotate and push of new MFCCs to net in buffer
 * @details 
 * 
 * @param mfcc pointer to MFCC buffer
 * @param in_x net x size
 * @param in_y net y size
 * @param xoffset x offset in network input
 */
void mfccToNetInputPush(int16_t* mfcc, uint16_t in_x, uint16_t in_y)
{ 
  uint32_t dstIdx = 0;
  uint32_t srcIdx = 1*in_x;

  // move back
  for(int netInCtr = 0; netInCtr < ( (in_y-1)*in_x ); netInCtr++)
  {
    netInput[dstIdx++] = netInput[srcIdx++];
  }

  // copy new sample in at back
  mfccToNetInput(mfcc, in_x, in_y, in_y-1);
}


/*------------------------------------------------------------------------------
 * Callbacks
 * ---------------------------------------------------------------------------*/