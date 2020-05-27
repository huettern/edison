/*
* @Author: Noah Huetter
* @Date:   2020-04-15 11:16:05
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-05-21 10:09:31
*/

#include "ai_cmsis.h"
#include "cyclecounter.h"

#include <stdlib.h>

#include "arm_nnfunctions.h"

#include "cmsis_net/cmsis_net.h"

/*------------------------------------------------------------------------------
 * settings
 * ---------------------------------------------------------------------------*/
// #define CYCLE_PROFILING

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
 * Data
 * ---------------------------------------------------------------------------*/


/*------------------------------------------------------------------------------
 * Publics
 * ---------------------------------------------------------------------------*/

/**
 * @brief Used to test
 * @details [long description]
 */
void aiCMSISTest(void)
{
  // int8_t* in = (int8_t*)malloc(CONV1_INPUT_X*CONV1_INPUT_Y*CONV1_INPUT_CH);
  static int8_t in[CONV1_INPUT_X*CONV1_INPUT_Y*CONV1_INPUT_CH];
  // int8_t* out = (int8_t*)malloc(FC1_ROW_DIM);
  static int8_t out[FC1_ROW_DIM];

  printf("before\n");

  mainSetPrintfUart(&huart4);
  if(!in || !out)
  {
    printf("malloc() error\n");
  }
  
  memset(in, 0, CONV1_INPUT_X*CONV1_INPUT_Y*CONV1_INPUT_CH);
  for(int i = 0; i < CONV1_INPUT_X*CONV1_INPUT_Y*CONV1_INPUT_CH; i++)
  {
    in[i] = (int8_t)i;
  }

  arm_status status;
  status = cmsisRunInference ((void*) in, (void*) out);
  printf("status = %d\n", status);

  // free(in);
  // free(out);
  mainSetPrintfUart(&huart4);

  printf("after\n");
}

/**
 * @brief Init mode
 * @details [long description]
 *  
 * @return 
 */
void aiCMSISInit(void)
{
  
}

int aiCMSISRunInference(void* in_data, void* out_data)
{
  // nnom_status_t ret;
  // prfStart("aiCMSISRunInference");
  // // copy data to net input
  // memcpy(nnom_input_data, in_data, sizeof(nnom_input_data)*sizeof(int8_t));
  // prfEvent("memcpy");
  // // run prediciton
  // ret = model_run(model);
  // prfEvent("model_run");
  // // copy prediction back
  // memcpy(out_data, nnom_output_data, sizeof(nnom_output_data)*sizeof(int8_t));
  // prfEvent("memcpy");

  // prfStop();
  // return (int)ret;
}

int aiCMSISPredict(uint32_t *label, float *prob)
{
  
  return 0; 
}

void aiCMSISPrintInfo(void)
{

}
int8_t* aiCMSISGetInputBuffer(void)
{
  return 0;
}
int8_t* aiCMSISGetOutputBuffer(void)
{
  return 0;
}