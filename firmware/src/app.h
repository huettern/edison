#pragma once

#include "main.h"

/*------------------------------------------------------------------------------
 * Publics
 * ---------------------------------------------------------------------------*/

void appHostAudioProcess(uint32_t nSamples, uint8_t bits);
int8_t appHifMfccAndInference(uint8_t *args);
int8_t appHifMicMfccInfere(uint8_t *args);
int8_t appMicMfccInfereContinuous (uint8_t *args);
