/*
* @Author: Noah Huetter
* @Date:   2020-04-15 11:16:05
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-04-30 13:28:43
*/
#include "ai.h"

// cube kws model
#include "cube/kws/kws.h"
#include "cube/kws/kws_data.h"
#include "cube/kws/app_x-cube-ai.h"
#include "cube/kws/constants_ai.h"

#include "printf.h"

/*------------------------------------------------------------------------------
 * Types
 * ---------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------
 * Settings
 * ---------------------------------------------------------------------------*/

// #define AI_BUFFER_NULL(ptr_)  \
//     AI_BUFFER_OBJ_INIT( \
//     AI_BUFFER_FORMAT_NONE|AI_BUFFER_FMT_FLAG_CONST, \
//     0, 0, 0, 0, \
//     AI_HANDLE_PTR(ptr_))

// #define NET_CUBE_KWS_ID 0
// // number of inputs, here 1
// #define NET_CUBE_KWS_IN_NUM AI_KWS_IN_NUM
// // input size in number of elements
// #define NET_CUBE_KWS_INSIZE AI_KWS_IN_1_SIZE
// // input size in bytes
// #define NET_CUBE_KWS_INSIZE_BYTES AI_KWS_IN_1_SIZE_BYTES
// // a predefined input structure
// #define NET_CUBE_KWS_INPUT AI_KWS_IN
// // same for output
// #define NET_CUBE_KWS_OUT_NUM AI_KWS_OUT_NUM
// #define NET_CUBE_KWS_OUTSIZE AI_KWS_OUT_1_SIZE
// #define NET_CUBE_KWS_OUTSIZE_BYTES AI_KWS_OUT_1_SIZE_BYTES
// #define NET_CUBE_KWS_OUTPUT AI_KWS_OUT
// // memory required for (intermediate) activations
// #define NET_CUBE_KWS_ACTIVATIONS_SIZE AI_KWS_DATA_ACTIVATIONS_SIZE

// // input format
// // format, height, width, channels, n batches, data
// // AI_BUFFER_FORMAT_FLOAT, 62, 32, 1, 1, NULL
// // output format
// // format, height, width, channels, n batches, data
// // AI_BUFFER_FORMAT_FLOAT, 1, 1, 1, 1, NULL


// /*------------------------------------------------------------------------------
//  * Private data
//  * ---------------------------------------------------------------------------*/

// /**
//  * CUBE
//  */
// #define AI_MNETWORK_NUMBER 1
// static struct ai_network_exec_ctx {
//     ai_handle network;
//     ai_network_report report;
// } net_exec_ctx[AI_MNETWORK_NUMBER] = {0};
// // Handle to the net
// static ai_handle kws;
// // input and output buffers
// static ai_buffer ai_input[NET_CUBE_KWS_IN_NUM] = NET_CUBE_KWS_INPUT ;
// static ai_buffer ai_output[NET_CUBE_KWS_OUT_NUM] = NET_CUBE_KWS_OUTPUT ;

// static ai_u8 activations[NET_CUBE_KWS_ACTIVATIONS_SIZE];

// /*------------------------------------------------------------------------------
//  * Prototypes
//  * ---------------------------------------------------------------------------*/

// static int cubeNetInit(void);
// static void printCubeNetInfo(void);
// static void aiLogErr(const ai_error err, const char *fct);
// ai_u32 aiBufferSize(const ai_buffer* buffer);
// void aiPrintNetworkInfo(const ai_network_report* report);
// __STATIC_INLINE void aiPrintLayoutBuffer(const char *msg, int idx,
//         const ai_buffer* buffer);
// static int aiBootstrap(const char *nn_name, const int idx);

// /*------------------------------------------------------------------------------
//  * Publics
//  * ---------------------------------------------------------------------------*/

// /**
//  * @brief Init AI functions
//  * @details 
//  */
// int aiInitialize(void)
// {
//   int res = -1;

//   cubeNetInit();
//   // ai_error err;
//   // const char *nn_name;
//   // const ai_network_params params = {
//   //   AI_BUFFER_NULL(NULL),
//   //   AI_BUFFER_NULL(activations)
//   // };

//   // // Find a network
//   // nn_name = ai_mnetwork_find(NULL, 0);
//   // if (nn_name) {
//   //   printf("\nFound network: \"%s\"\n", nn_name);
//   // } else {
//   //   printf("E: ai_mnetwork_find\n");
//   //   return -1;
//   // }

//   // // Create the network
//   // err = ai_mnetwork_create(nn_name, &kws, NULL);
//   // if (err.type) {
//   //   printf("E: ai_mnetwork_create\n");
//   //   return -1;
//   // }

//   // // Initialize the network
//   // if (!ai_mnetwork_init(kws, &params)) {
//   //   printf("E: ai_mnetwork_init\n");
//   //   return -1;
//   // }

//   // aiPrintInfo();
//   // return 0;
// }

// /**
//  * @brief Run single inference with data from/to host interface
//  * @details 
//  * 
//  * @param netId select net ID
//  */
// void aiRunInferenceHif(uint8_t netId)
// {

// }

// /**
//  * @brief Print human readable info about AI module
//  * @details 
//  */
// void aiPrintInfo(void)
// {
//   printCubeNetInfo();
// }

// /*------------------------------------------------------------------------------
//  * Privates
//  * ---------------------------------------------------------------------------*/

// /**
//  * @brief Init network from Cube AI
//  * @details 
//  */
// static int cubeNetInit(void)
// {
//   int res = -1;
//   const char *nn_name;
//   int idx;

//   /* Clean all network exec context */
//   for (idx=0; idx < AI_MNETWORK_NUMBER; idx++) {
//       net_exec_ctx[idx].network = AI_HANDLE_NULL;
//   }

//   /* Discover and init the embedded network */
//   idx = 0;
//   do {
//       nn_name = ai_mnetwork_find(NULL, idx);
//       if (nn_name) {
//           printf("\r\nFound network \"%s\"\r\n", nn_name);
//           res = aiBootstrap(nn_name, idx);
//           if (res)
//               nn_name = NULL;
//       }
//       idx++;
//   } while (nn_name);

//   return res;
// }

// /**
//  * @brief Print network info from CubeAI
//  * @details 
//  */
// static void printCubeNetInfo(void)
// {
//   ai_network_report rep;
//   ai_bool ret;
//   ret = ai_kws_get_info(kws, &rep);
//   if(!ret)
//   {
//     printf("ai_kws_get_info Error\n");
//   }
//   else
//   {
//     printf("----------------------------------\n");
//     printf("AI net information\n");
//     printf(" name: %s\n", rep.model_name);
//     printf(" signature: %s\n", rep.model_signature);
//     printf(" datetime: %s\n", rep.model_datetime);
//     printf(" compiled: %s\n", rep.compile_datetime);
//     printf(" n macc: %d\n", rep.n_macc);
//     printf(" n inputs: %d\n", rep.n_inputs);
//     printf(" n outputs: %d\n\n", rep.n_outputs);
//     printf(" in[0] format: %d\n", rep.inputs[0].format);
//     printf(" in[0] n_batches: %d\n", rep.inputs[0].n_batches);
//     printf(" in[0] height: %d\n", rep.inputs[0].height);
//     printf(" in[0] width: %d\n", rep.inputs[0].width);
//     printf(" in[0] channels: %d\n\n", rep.inputs[0].channels);
//     printf(" out[0] format: %d\n", rep.outputs[0].format);
//     printf(" out[0] n_batches: %d\n", rep.outputs[0].n_batches);
//     printf(" out[0] height: %d\n", rep.outputs[0].height);
//     printf(" out[0] width: %d\n", rep.outputs[0].width);
//     printf(" out[0] channels: %d\n", rep.outputs[0].channels);
//   }
// }

// void aiLogErr(const ai_error err, const char *fct)
// {
//     if (fct)
//         printf("E: AI error (%s) - type=%d code=%d\r\n", fct,
//                 err.type, err.code);
//     else
//         printf("E: AI error - type=%d code=%d\r\n", err.type, err.code);
// }

// ai_u32 aiBufferSize(const ai_buffer* buffer)
// {
//     return buffer->height * buffer->width * buffer->channels;
// }

// __STATIC_INLINE void aiPrintLayoutBuffer(const char *msg, int idx,
//         const ai_buffer* buffer)
// {
//     uint32_t type_id = AI_BUFFER_FMT_GET_TYPE(buffer->format);
//     printf("%s[%d] ",msg, idx);
//     if (type_id == AI_BUFFER_FMT_TYPE_Q) {
//         printf(" %s%d,",
//             AI_BUFFER_FMT_GET_SIGN(buffer->format)?"s":"u",
//                 (int)AI_BUFFER_FMT_GET_BITS(buffer->format));
//         if (AI_BUFFER_META_INFO_INTQ(buffer->meta_info)) {
//         ai_float scale = AI_BUFFER_META_INFO_INTQ_GET_SCALE(buffer->meta_info, 0);
//         int zero_point = AI_BUFFER_META_INFO_INTQ_GET_ZEROPOINT(buffer->meta_info, 0);
//         printf(" scale=%f, zero=%d,", (float)scale, (int)zero_point);
//       } else {
//         printf("Q%d.%d,",
//             (int)AI_BUFFER_FMT_GET_BITS(buffer->format)
//           - ((int)AI_BUFFER_FMT_GET_FBITS(buffer->format) +
//           (int)AI_BUFFER_FMT_GET_SIGN(buffer->format)),
//           AI_BUFFER_FMT_GET_FBITS(buffer->format));
//       }
//     }
//     else if (type_id == AI_BUFFER_FMT_TYPE_FLOAT)
//         printf(" float%d,",
//                 (int)AI_BUFFER_FMT_GET_BITS(buffer->format));
//     else
//         printf("NONE");
//     printf(" %ld bytes, shape=(%d,%d,%ld)",
//         AI_BUFFER_BYTE_SIZE(AI_BUFFER_SIZE(buffer), buffer->format),
//       buffer->height, buffer->width, buffer->channels);
//     if (buffer->data)
//       printf(" (@0x%08x)\r\n", (int)buffer->data);
//     else
//       printf("\r\n");
// }

// void aiPrintNetworkInfo(const ai_network_report* report)
// {
//     int i;
//     printf("Network configuration...\r\n");
//     printf(" Model name         : %s\r\n", report->model_name);
//     printf(" Model signature    : %s\r\n", report->model_signature);
//     printf(" Model datetime     : %s\r\n", report->model_datetime);
//     printf(" Compile datetime   : %s\r\n", report->compile_datetime);
//     printf(" Runtime revision   : %s (%d.%d.%d)\r\n", report->runtime_revision,
//             report->runtime_version.major,
//             report->runtime_version.minor,
//             report->runtime_version.micro);
//     printf(" Tool revision      : %s (%d.%d.%d)\r\n", report->tool_revision,
//             report->tool_version.major,
//             report->tool_version.minor,
//             report->tool_version.micro);
//     printf("Network info...\r\n");
//     printf("  nodes             : %ld\r\n", report->n_nodes);
//     printf("  complexity        : %ld MACC\r\n", report->n_macc);
//     printf("  activation        : %ld bytes\r\n",
//             aiBufferSize(&report->activations));
//     printf("  params            : %ld bytes\r\n",
//             aiBufferSize(&report->params));
//     printf("  inputs/outputs    : %u/%u\r\n",
//             report->n_inputs, report->n_outputs);
//     for (i=0; i<report->n_inputs; i++)
//         aiPrintLayoutBuffer("   I", i, &report->inputs[i]);
//     for (i=0; i<report->n_outputs; i++)
//         aiPrintLayoutBuffer("   O", i, &report->outputs[i]);
// }

// static int aiBootstrap(const char *nn_name, const int idx)
// {
//   ai_error err;
//   ai_u32 ext_addr, sz;

//   /* Creating the network */
//   printf("Creating the network \"%s\"..\r\n", nn_name);
//   err = ai_mnetwork_create(nn_name, &net_exec_ctx[idx].network, NULL);
//   if (err.type) {
//       aiLogErr(err, "ai_mnetwork_create");
//       return -1;
//   }

//   /* Initialize the instance */
//   printf("Initializing the network\r\n");
//   /* build params structure to provide the reference of the
//    * activation and weight buffers */
// #if !defined(AI_MNETWORK_DATA_ACTIVATIONS_INT_SIZE)
//   const ai_network_params params = {
//           AI_BUFFER_NULL(NULL),
//           AI_BUFFER_NULL(activations) };
// #else
//   ai_network_params params = {
//               AI_BUFFER_NULL(NULL),
//               AI_BUFFER_NULL(NULL) };

//   if (ai_mnetwork_get_ext_data_activations(net_exec_ctx[idx].network, &ext_addr, &sz) == 0) {
//     if (ext_addr == 0xFFFFFFFF) {
//       params.activations.data = (ai_handle)activations;
//       ext_addr = (ai_u32)activations;
//       sz = (ai_u32)AI_BUFFER_SIZE(&net_exec_ctx[idx].report.activations);
//     }
//     else {
//       params.activations.data = (ai_handle)ext_addr;
//     }
//   }
// #endif

//   if (!ai_mnetwork_init(net_exec_ctx[idx].network, &params)) {
//       err = ai_mnetwork_get_error(net_exec_ctx[idx].network);
//       aiLogErr(err, "ai_mnetwork_init");
//       ai_mnetwork_destroy(net_exec_ctx[idx].network);
//       net_exec_ctx[idx].network = AI_HANDLE_NULL;
//       return -4;
//   }

//   /* Query the created network to get relevant info from it */
//   if (ai_mnetwork_get_info(net_exec_ctx[idx].network,
//           &net_exec_ctx[idx].report)) {
//       aiPrintNetworkInfo(&net_exec_ctx[idx].report);
//   } else {
//       err = ai_mnetwork_get_error(net_exec_ctx[idx].network);
//       aiLogErr(err, "ai_mnetwork_get_info");
//       ai_mnetwork_destroy(net_exec_ctx[idx].network);
//       net_exec_ctx[idx].network = AI_HANDLE_NULL;
//       return -2;
//   }

//   return 0;
// }

/*------------------------------------------------------------------------------
 * Callbacks
 * ---------------------------------------------------------------------------*/