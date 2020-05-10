/*
* @Author: Noah Huetter
* @Date:   2020-04-15 11:16:05
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-05-10 16:28:13
*/

#include "ai_nnom.h"

#include "nnom.h"
#include "kws_nnom/weights.h"

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
  // copy data to net input
  memcpy(nnom_input_data, in_data, sizeof(nnom_input_data)*sizeof(int8_t));
  // run prediciton
  ret = model_run(model);
  // copy prediction back
  memcpy(out_data, nnom_output_data, sizeof(nnom_output_data)*sizeof(int8_t));

  return (int)ret;
}

void aiNnomPrintInfo(void)
{
  
}