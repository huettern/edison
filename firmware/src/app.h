#pragma once

#include "main.h"

/*------------------------------------------------------------------------------
 * Publics
 * ---------------------------------------------------------------------------*/

void appHostAudioProcess(uint32_t nSamples, uint8_t bits);
int appHifMfccAndInference(uint8_t *args);

