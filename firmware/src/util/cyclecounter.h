#pragma once

#include "main.h"

/**
 * @brief Resets the internal cycle counter to zero
 * @details 
 */
void ResetTimer(void);

/**
 * @brief Starts the internal cycle counter
 * @details 
 */
void StartTimer(void);

/**
 * @brief Stops the internal cycle counter
 * @details 
 */
void StopTimer(void);

/**
 * @brief Returns the current number of cycles according to the internal cycle counter
 * @details 
 */
unsigned int getCycles(void);

void cycProfStart(const char *profile_name);
void cycProfEvent(const char *event);
void cycProfStop(void);