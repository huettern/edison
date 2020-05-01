#pragma once

#include "main.h"

#include "kws.h"

/*------------------------------------------------------------------------------
 * SETTINGS
 * ---------------------------------------------------------------------------*/

/**
 * Select which net type us used
 */
#define NET_TYPE_CUBE


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
#ifdef NET_TYPE_CUBE
  #define AI_NET_INSIZE           NET_CUBE_KWS_INSIZE
  #define AI_NET_INSIZE_BYTES     NET_CUBE_KWS_INSIZE_BYTES
  #define AI_NET_OUTSIZE           NET_CUBE_KWS_OUTSIZE
  #define AI_NET_OUTSIZE_BYTES     NET_CUBE_KWS_OUTSIZE_BYTES
#endif

/*------------------------------------------------------------------------------
 * Publics
 * ---------------------------------------------------------------------------*/

int aiInitialize(void);
void aiPrintInfo(void);
void aiRunInferenceHif(void);
void aiGetInputShape(uint16_t *x, uint16_t *y);
int aiRunInference(void* in_data, void* out_data);
