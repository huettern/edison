/*
* @Author: Noah Huetter
* @Date:   2020-04-15 11:16:05
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-05-28 16:22:19
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
 * Prototypes
 * ---------------------------------------------------------------------------*/
static void print_layer_info(nnom_layer_t *layer, uint32_t layer_count);
static size_t tensorSize(nnom_tensor_t* t);
static size_t io_mem_size(nnom_layer_io_t *io);

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
  printf("\nNNoM version %d.%d.%d\n", NNOM_MAJORVERSION, NNOM_SUBVERSION, NNOM_REVISION);
  printf("Start compiling model...\n");
  printf("Layer(#)         Activation    output shape    ops(MAC)   mem(in, out, buf)      mem blk lifetime\n");
  printf("-------------------------------------------------------------------------------------------------\n");

  // compile layers, started from list head, nested run till the end of models
  uint32_t layer_count=0;
  nnom_layer_t *layer = model->head;
  while (layer)
  {
    print_layer_info(layer, layer_count++);
    if(layer->out->hook.io->type != 1) break;
    layer = layer->out->hook.io->owner;
    printf("\n");
  }

  printf("\n-------------------------------------------------------------------------------------------------\n");


}
int8_t* aiNnomGetInputBuffer(void)
{
  return nnom_input_data;
}
int8_t* aiNnomGetOutputBuffer(void)
{
  return nnom_output_data;
}

static size_t tensorSize(nnom_tensor_t* t)
{
  size_t size = 0;
  if (t)
  {
    size = t->dim[0];
    for (int i = 1; i < t->num_dim; i++)
      size *= t->dim[i];
  }
  return size;
}
static size_t io_mem_size(nnom_layer_io_t *io)
{
  size_t size = 0;
  if (io != NULL)
  {
    while (io)
    {
      size += tensorSize(io->tensor);
      io = io->aux;
    }
  }
  return size;
}
static void print_layer_info(nnom_layer_t *layer, uint32_t layer_count)
{
  size_t in_size = io_mem_size(layer->in);
  size_t out_size = io_mem_size(layer->out);
  size_t compsize;
  size_t mac = layer->stat.macc;
  if (layer->comp != NULL)
    compsize = shape_size(&layer->comp->shape);
  else
    compsize = 0;
  // names
  printf("#%-3d %-10s - ", layer_count, default_layer_names[layer->type]);
  // activations
  if (layer->actail != NULL)
    printf("%-8s - ", default_activation_names[layer->actail->type]);
  else
    printf("         - ");

  printf("(");
  for (int i = 0; i < 3; i++)
  {
    if (layer->out->tensor->num_dim > i)
      printf("%4d,", layer->out->tensor->dim[i]);
    else 
      printf("     ");
  }
  printf(")  ");
  
  // MAC operation
  if(mac == 0)
    printf("        ");
  else if (mac < 10000)
    printf("%7d ", mac);
  else if (mac < 1000*1000)
    printf("%6dk ", mac/1000);
  else if (mac < 1000*1000*1000)
    printf("%3d.%02dM ", mac/(1000*1000), mac%(1000*1000)/(10*1000)); // xxx.xx M
  else
    printf("%3d.%02dG ", mac/(1000*1000*1000), mac%(1000*1000*1000)/(10*1000*1000)); // xxx.xx G
  
  // memory 
  printf("(%6d,%6d,%6d)", in_size, out_size, compsize);
}