/*
* @Author: Noah Huetter
* @Date:   2020-04-15 11:16:05
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-05-28 15:40:21
*/

#include "ai_nnom.h"

#include "nnom.h"
#include "kws_nnom/weights.h"
#include "cyclecounter.h"

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
nnom_model_t *model;

/*------------------------------------------------------------------------------
 * Publics
 * ---------------------------------------------------------------------------*/

/**
 * @brief Used to test
 * @details [long description]
 */
void aiNnomTest(void)
{
#ifdef NNOM_VERIFICATION
  model = nnom_model_create();
  model_run(model);
#endif
}

/**
 * @brief Init mode
 * @details [long description]
 *  
 * @return 
 */
void aiNnomInit(void)
{
  model = nnom_model_create();
}

int aiNnomRunInference(void* in_data, void* out_data)
{
  nnom_status_t ret;
  prfStart("aiNnomRunInference");
  // copy data to net input
  memcpy(nnom_input_data, in_data, sizeof(nnom_input_data)*sizeof(int8_t));
  prfEvent("memcpy");
  // run prediciton
  ret = model_run(model);
  prfEvent("model_run");
  // copy prediction back
  memcpy(out_data, nnom_output_data, sizeof(nnom_output_data)*sizeof(int8_t));
  prfEvent("memcpy");

  prfStop();
  return (int)ret;
}

/**
 * @brief Run prediction with Nnom
 * @details 
 * 
 * @param m 
 * @param label 
 * @param prob 
 * @return 
 */
int aiNnomPredict(uint32_t *label, float *prob)
{
  nnom_status_t ret;
  ret = nnom_predict(model, label, prob);
  return (int)ret;
}

void aiNnomPrintInfo(void)
{

}
int8_t* aiNnomGetInputBuffer(void)
{
  return nnom_input_data;
}
int8_t* aiNnomGetOutputBuffer(void)
{
  return nnom_output_data;
}