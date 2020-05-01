/*
* @Author: Noah Huetter
* @Date:   2020-04-15 11:33:22
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-04-15 11:33:48
*/
#pragma once

#include "main.h"

/*------------------------------------------------------------------------------
 * MFCC
 * ---------------------------------------------------------------------------*/
#define AUD_MFCC_FRAME_LEN 1024
#define AUD_MFCC_SAMPLE_SIZE 2
#define AUD_MFCC_FRAME_SIZE_BYES (AUD_MFCC_FRAME_LEN*AUD_MFCC_SAMPLE_SIZE)

/*------------------------------------------------------------------------------
 * publics
 * ---------------------------------------------------------------------------*/
void audioInit(void);
void audioCalcMFCCs(int16_t * inp, int16_t ** oup);
void audioDumpToHost(void);
void audioMELSingleBatch(void);