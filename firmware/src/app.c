/*
* @Author: Noah Huetter
* @Date:   2020-04-15 11:16:05
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-05-01 17:00:14
*/
#include "app.h"
#include <stdlib.h>

#include "printf.h"
#include "microphone.h"
#include "ai.h"
#include "audioprocessing.h"
#include "hostinterface.h"

/*------------------------------------------------------------------------------
 * Types
 * ---------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------
 * Settings
 * ---------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------
 * Private data
 * ---------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------
 * Prototypes
 * ---------------------------------------------------------------------------*/
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
  
  int16_t *in_data=NULL, *out_mfccs;
  float *out_data=NULL, *mfccs=NULL, tmpf;
  
  uint8_t tag;
  uint16_t in_x, in_y;
  uint32_t len;

  // get net info
  aiGetInputShape(&in_x, &in_y);

  // printf("Got input shape (%d, %d)\n", in_x, in_y);

  // 1. Receive in_y frames and process their MFCC
  in_data = malloc(AUD_MFCC_FRAME_SIZE_BYES);
  mfccs = malloc(AI_NET_INSIZE_BYTES);
  out_data = malloc(AI_NET_OUTSIZE_BYTES);

  for(int frameCtr = 0; frameCtr < in_y; frameCtr++)
  {
    // 2. Calculate MFCC
    len = hiReceive(in_data, AUD_MFCC_FRAME_SIZE_BYES, DATA_FORMAT_S16, &tag);    
    printf("received %d \n", len);

    audioCalcMFCCs(in_data, &out_mfccs);

    // copy to net in buffer and cast to float
    for(int mfccCtr = 0; mfccCtr < in_x; mfccCtr++)
    {
      tmpf = (float)out_mfccs[mfccCtr];
      mfccs[frameCtr*in_x + mfccCtr] = tmpf;
    }

    // signal host that we are ready
    hiSendMCUReady();
  }

  // 3. Run inference
  ret = aiRunInference((void*)mfccs, (void*)out_data);
  hiSendMCUReady();

  // 4. report back mfccs and net out
  hiSendF32(mfccs, AI_NET_INSIZE, 0x20);
  hiSendF32(out_data, AI_NET_OUTSIZE, 0x21);

  // cleanup
  free(in_data);
  free(mfccs);
  free(out_data);

  return ret;
}

/*------------------------------------------------------------------------------
 * Privates
 * ---------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------
 * Callbacks
 * ---------------------------------------------------------------------------*/