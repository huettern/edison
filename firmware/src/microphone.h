#pragma once

#include "main.h"

void micInit(void);
uint32_t micSampleSingle(int32_t ** data, uint32_t n);
void micEndlessStream(void);
void micHostSampleRequest(uint16_t nSamples);
void micHostSampleRequestPreprocessed(uint16_t nSamples);
uint32_t micSampleSinglePreprocessed(int8_t ** data, uint32_t n);
