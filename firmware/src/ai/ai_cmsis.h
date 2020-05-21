#pragma once

#include "main.h"
#include <stdint.h>

void aiCMSISTest(void);
void aiCMSISInit(void);
int aiCMSISRunInference(void* in_data, void* out_data);
int aiCMSISPredict(uint32_t *label, float *prob);
void aiCMSISPrintInfo(void);
int8_t* aiCMSISGetInputBuffer(void);
int8_t* aiCMSISGetOutputBuffer(void);