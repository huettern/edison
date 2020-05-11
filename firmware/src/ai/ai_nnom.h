#pragma once

#include <stdint.h>

void aiNnomTest(void);
void aiNnomInit(void);
int aiNnomRunInference(void* in_data, void* out_data);
int aiNnomPredict(uint32_t *label, float *prob);
void aiNnomPrintInfo(void);
int8_t* aiNnomGetInputBuffer(void);
int8_t* aiNnomGetOutputBuffer(void);