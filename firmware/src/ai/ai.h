#pragma once

#include "main.h"

#include "kws.h"

/*------------------------------------------------------------------------------
 * Data
 * ---------------------------------------------------------------------------*/

/*------------------------------------------------------------------------------
 * SETTINGS
 * ---------------------------------------------------------------------------*/

/**
 * Select which net type us used
 */
#define NET_TYPE_CUBE 1
#define NET_TYPE_NNOM 2
#define NET_TYPE_CMSIS 3

#define NET_TYPE NET_TYPE_CUBE

/**
 * This defines settings for the cube net
 */
#define NET_CUBE_KWS_ID 0
// number of inputs, here 1
#define NET_CUBE_KWS_IN_NUM AI_KWS_IN_NUM
// input size in number of elements
#define NET_CUBE_KWS_INSIZE AI_KWS_IN_1_SIZE
// input size in bytes
#define NET_CUBE_KWS_INSIZE_BYTES AI_KWS_IN_1_SIZE_BYTES
// a predefined input structure
#define NET_CUBE_KWS_INPUT AI_KWS_IN
// same for output
#define NET_CUBE_KWS_OUT_NUM AI_KWS_OUT_NUM
#define NET_CUBE_KWS_OUTSIZE AI_KWS_OUT_1_SIZE
#define NET_CUBE_KWS_OUTSIZE_BYTES AI_KWS_OUT_1_SIZE_BYTES
#define NET_CUBE_KWS_OUTPUT AI_KWS_OUT
    
/**
 * Settings of the used net
 */
#if NET_TYPE == NET_TYPE_CUBE
  #define AI_NET_INNUM           NET_CUBE_KWS_IN_NUM
  #define AI_NET_INSIZE           NET_CUBE_KWS_INSIZE
  #define AI_NET_INSIZE_BYTES     NET_CUBE_KWS_INSIZE_BYTES
  #define AI_NET_OUTNUM           NET_CUBE_KWS_OUT_NUM
  #define AI_NET_OUTSIZE           NET_CUBE_KWS_OUTSIZE
  #define AI_NET_OUTSIZE_BYTES     NET_CUBE_KWS_OUTSIZE_BYTES
#elif NET_TYPE == NET_TYPE_NNOM
  #define AI_NET_INNUM           1
  #define AI_NET_INSIZE           403
  #define AI_NET_INSIZE_BYTES     403
  #define AI_NET_OUTNUM           1
  #define AI_NET_OUTSIZE           6
  #define AI_NET_OUTSIZE_BYTES     6
#elif NET_TYPE == NET_TYPE_CMSIS
  #define AI_NET_INNUM           1
  #define AI_NET_INSIZE           403
  #define AI_NET_INSIZE_BYTES     (1*AI_NET_INNUM)
  #define AI_NET_OUTNUM           1
  #define AI_NET_OUTSIZE           10
  #define AI_NET_OUTSIZE_BYTES     (1*AI_NET_OUTSIZE)
 #endif

/*------------------------------------------------------------------------------
 * Publics
 * ---------------------------------------------------------------------------*/

int aiInitialize(void);
void aiPrintInfo(void);
void aiRunInferenceHif(void);
void aiGetInputShape(uint16_t *x, uint16_t *y);
int aiRunInference(void* in_data, void* out_data);
const char* aiGetKeywordFromIndex(uint32_t idx);
uint32_t aiGetKeywordCount(void);