/*
* @Author: Noah Huetter
* @Date:   2020-04-15 11:16:05
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-05-08 13:24:26
*/
#include "ai.h"

#include <stdlib.h>

// cube kws model
#include "cube/kws/kws.h"
#include "cube/kws/kws_data.h"
#include "cube/kws/app_x-cube-ai.h"
#include "cube/kws/constants_ai.h"
#include "cube/kws/aiTestUtility.h"

#include "printf.h"
#include "hostinterface.h"
#include "util.h"
#include "cyclecounter.h"

/*------------------------------------------------------------------------------
 * Types
 * ---------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------
 * Settings
 * ---------------------------------------------------------------------------*/
/**
 * @brief Enable this to show profiling on arduino Tx pin
 */
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

#define AI_BUFFER_NULL(ptr_)  \
    AI_BUFFER_OBJ_INIT( \
    AI_BUFFER_FORMAT_NONE|AI_BUFFER_FMT_FLAG_CONST, \
    0, 0, 0, 0, \
    AI_HANDLE_PTR(ptr_))

// memory required for (intermediate) activations
#define NET_CUBE_KWS_ACTIVATIONS_SIZE AI_KWS_DATA_ACTIVATIONS_SIZE


/*------------------------------------------------------------------------------
 * Private data
 * ---------------------------------------------------------------------------*/

const char* const aiKeywords[] = {"cat", "marvin", "left", "zero", "_cold"};

/**
 * CUBE
 */
#ifdef CUBE_VERIFICATION
static struct ai_network_exec_ctx {
    ai_handle network;
    ai_network_report report;
} net_exec_ctx[AI_MNETWORK_NUMBER] = {0};
#endif
// Handle to the net
static ai_handle kws;
// input and output buffers
static ai_buffer ai_input[NET_CUBE_KWS_IN_NUM] = NET_CUBE_KWS_INPUT ;
static ai_buffer ai_output[NET_CUBE_KWS_OUT_NUM] = NET_CUBE_KWS_OUTPUT ;

static ai_u8 activations[NET_CUBE_KWS_ACTIVATIONS_SIZE];

static uint32_t lastInferenceTimeUs;

/*------------------------------------------------------------------------------
 * Prototypes
 * ---------------------------------------------------------------------------*/

static int cubeNetInit(void);
static int cubeNetRun(const void *in_data, void *out_data);
static void printCubeNetInfo(void);

#ifdef CUBE_VERIFICATION
static int aiBootstrap(const char *nn_name, const int idx);
#endif

/*------------------------------------------------------------------------------
 * Publics
 * ---------------------------------------------------------------------------*/

/**
 * @brief Init AI functions
 * @details 
 */
int aiInitialize(void)
{
  cubeNetInit();
  return 0;
}

/**
 * @brief Run single inference with data from/to host interface
 * @details 
 * 
 * @param netId select net ID
 */
void aiRunInferenceHif(void)
{
  int ret;
  float *in_data=NULL, *out_data=NULL;
  uint32_t len;
  uint8_t tag;

  prfStart("aiRunInferenceHif");

  in_data = malloc(NET_CUBE_KWS_INSIZE_BYTES);
  out_data = malloc(NET_CUBE_KWS_OUTSIZE_BYTES);

  prfEvent("malloc");

  len = hiReceive(in_data, NET_CUBE_KWS_INSIZE_BYTES, DATA_FORMAT_F32, &tag);
  (void)len;
//   printf("Received %d elements with tag %d\n[ ", length, tag);

  prfEvent("receive");

  ret = aiRunInference((void*)in_data, (void*)out_data);
  (void)ret;

  prfEvent("aiRunInference");

  hiSendF32(out_data, NET_CUBE_KWS_OUTSIZE, 0x17);

  prfEvent("send");

  free(in_data);
  free(out_data);

  prfEvent("free");
  prfStop();
}

// /**
//  * @brief Print human readable info about AI module
//  * @details 
//  */
void aiPrintInfo(void)
{
  printCubeNetInfo();
}

/**
 * @brief Fetches the networks input shape and stoers in argument pointer
 * @details 
 * 
 * @param x input with
 * @param y input height
 */
void aiGetInputShape(uint16_t *x, uint16_t *y)
{
  ai_network_report rep;
  (void)ai_kws_get_info(kws, &rep);
  *x = rep.inputs[0].width;
  *y = rep.inputs[0].height;
}

/**
 * @brief Run an inference on the defined net
 * @details 
 * 
 * @param in_data input data pointer
 * @param out_data otuput data pointer
 * 
 * @return 
 */
int aiRunInference(void* in_data, void* out_data)
{
  int ret;

  uint8_t id = utilTic();

#ifdef NET_TYPE_CUBE
  ret = cubeNetRun((void*)in_data, (void*)out_data);
#endif
  
  lastInferenceTimeUs = utilToc(id);
  return ret;
}

const char* aiGetKeywordFromIndex(uint32_t idx)
{
  return aiKeywords[idx];
}

/*------------------------------------------------------------------------------
 * Privates
 * ---------------------------------------------------------------------------*/

/**
 * @brief Init network from Cube AI
 * @details 
 */
static int cubeNetInit()
{
 
  dwtIpInit();
  crcIpInit(); // <- THIS IS STRICTLY NECESSARY, or else, the network can't be created
  logDeviceConf();


#ifdef CUBE_VERIFICATION
  int idx;
  const char *nn_name;
  int res = -1;
  /**
   * @brief Init for verification use
   */
  
  for (idx=0; idx < AI_MNETWORK_NUMBER; idx++) {
      net_exec_ctx[idx].network = AI_HANDLE_NULL;
  }

  /* Discover and init the embedded network */
  idx = 0;
  do {
      nn_name = ai_mnetwork_find(NULL, idx);
      if (nn_name) {
          printf("\r\nFound network \"%s\"\r\n", nn_name);
          res = aiBootstrap(nn_name, idx);
          if (res)
              nn_name = NULL;
      }
      idx++;
  } while (nn_name);

  // our network is the first in the list
  kws = net_exec_ctx[0].network;

  return res;

#else
  ai_error err;
  /**
   * @brief Standard init
   */

  /* 1 - Specific AI data structure to provide the references of the
   * activation/working memory chunk and the weights/bias parameters */
  const ai_network_params params = {
          AI_KWS_DATA_WEIGHTS(ai_kws_data_weights_get()),
          AI_KWS_DATA_ACTIVATIONS(activations)
  };

  /* 2 - Create an instance of the NN */
  err = ai_kws_create(&kws, AI_KWS_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return -1;
  }

  /* 3 - Initialize the NN - Ready to be used */
  if (!ai_kws_init(kws, &params)) {
      err = ai_kws_get_error(kws);
      ai_kws_destroy(kws);
      kws = AI_HANDLE_NULL;
    return -2;
  }

  return 0;

#endif
}

/**
 * @brief Runs an inference with the cube net
 * @details 
 */
static int cubeNetRun(const void *in_data, void *out_data)
{
  ai_i32 nbatch;
  ai_error err;

  /* Parameters checking */
  if (!in_data || !out_data || !kws)
      return -1;

  /* Initialize input/output buffer handlers */
  ai_input[0].n_batches = 1;
  ai_input[0].data = AI_HANDLE_PTR(in_data);
  ai_output[0].n_batches = 1;
  ai_output[0].data = AI_HANDLE_PTR(out_data);

  /* 2 - Perform the inference */
  nbatch = ai_kws_run(kws, &ai_input[0], &ai_output[0]);
  if (nbatch != 1) {
      err = ai_kws_get_error(kws);
      // ...
      return err.code;
  }

  return 0;
}

/**
 * @brief Print network info from CubeAI
 * @details 
 */
static void printCubeNetInfo(void)
{
  ai_network_report rep;
  ai_bool ret;
  ret = ai_kws_get_info(kws, &rep);
  if(!ret)
  {
    printf("ai_kws_get_info Error\n");
  }
  else
  {
    printf("-------------------------------------------------------------\n");
    printf("AI net information\n");
    printf(" name: %s\n", rep.model_name);
    printf(" signature: %s\n", rep.model_signature);
    printf(" datetime: %s\n", rep.model_datetime);
    printf(" compiled: %s\n", rep.compile_datetime);
    printf(" n macc: %d\n", rep.n_macc);
    printf(" n inputs: %d\n", rep.n_inputs);
    printf(" n outputs: %d\n\n", rep.n_outputs);
    printf(" I[0] format: %d batches %d shape (%d, %d, %d)\n", rep.inputs[0].format,
      rep.inputs[0].n_batches, rep.inputs[0].height, rep.inputs[0].width, rep.inputs[0].channels);
    printf(" O[0] format: %d batches %d shape (%d, %d, %d)\n\n", rep.outputs[0].format,
      rep.outputs[0].n_batches, rep.outputs[0].height, rep.outputs[0].width, rep.outputs[0].channels);
    printf(" last inference time: %.2fms\n", (float)lastInferenceTimeUs/1000.0);
  }
}

#ifdef CUBE_VERIFICATION
static int aiBootstrap(const char *nn_name, const int idx)
{
  ai_error err;
  ai_u32 ext_addr, sz;

  /* Creating the network */
  printf("Creating the network \"%s\"..\r\n", nn_name);
  err = ai_mnetwork_create(nn_name, &net_exec_ctx[idx].network, NULL);
  if (err.type) {
      aiLogErr(err, "ai_mnetwork_create");
      return -1;
  }

  /* Initialize the instance */
  printf("Initializing the network\r\n");
  /* build params structure to provide the reference of the
   * activation and weight buffers */
#if !defined(AI_MNETWORK_DATA_ACTIVATIONS_INT_SIZE)
  const ai_network_params params = {
          AI_BUFFER_NULL(NULL),
          AI_BUFFER_NULL(activations) };
#else
  ai_network_params params = {
              AI_BUFFER_NULL(NULL),
              AI_BUFFER_NULL(NULL) };

  if (ai_mnetwork_get_ext_data_activations(net_exec_ctx[idx].network, &ext_addr, &sz) == 0) {
    if (ext_addr == 0xFFFFFFFF) {
      params.activations.data = (ai_handle)activations;
      ext_addr = (ai_u32)activations;
      sz = (ai_u32)AI_BUFFER_SIZE(&net_exec_ctx[idx].report.activations);
    }
    else {
      params.activations.data = (ai_handle)ext_addr;
    }
  }
#endif

  if (!ai_mnetwork_init(net_exec_ctx[idx].network, &params)) {
      err = ai_mnetwork_get_error(net_exec_ctx[idx].network);
      aiLogErr(err, "ai_mnetwork_init");
      ai_mnetwork_destroy(net_exec_ctx[idx].network);
      net_exec_ctx[idx].network = AI_HANDLE_NULL;
      return -4;
  }

  /* Query the created network to get relevant info from it */
  if (ai_mnetwork_get_info(net_exec_ctx[idx].network,
          &net_exec_ctx[idx].report)) {
      aiPrintNetworkInfo(&net_exec_ctx[idx].report);
  } else {
      err = ai_mnetwork_get_error(net_exec_ctx[idx].network);
      aiLogErr(err, "ai_mnetwork_get_info");
      ai_mnetwork_destroy(net_exec_ctx[idx].network);
      net_exec_ctx[idx].network = AI_HANDLE_NULL;
      return -2;
  }

  return 0;
}
#endif

/*------------------------------------------------------------------------------
 * Callbacks
 * ---------------------------------------------------------------------------*/