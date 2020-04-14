#pragma once

#include "main.h"

void micInit(void);
uint32_t micSampleSingle(int32_t ** data, uint32_t n);
void micEndlessStream(void);
void micHostSampleRequest(uint16_t nSamples, uint8_t optArg);
