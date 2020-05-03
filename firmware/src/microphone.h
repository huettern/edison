#pragma once

#include "main.h"

void micInit(void);
uint32_t micSampleSingle(int32_t ** data, uint32_t n);
void micEndlessStream(void);
void micHostSampleRequest(uint16_t nSamples);
void micHostSampleRequestPreprocessed(uint16_t nSamples, uint8_t bits);
uint32_t micSampleSinglePreprocessed(void ** data, uint32_t n, uint8_t bits);

void micContinuousStart (void);
void micContinuousStop (void);
int16_t* micContinuousGet (void);