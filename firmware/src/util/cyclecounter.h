#pragma once

#include "main.h"


static volatile unsigned int *DWT_CYCCNT  ;
static volatile unsigned int *DWT_CONTROL ;
static volatile unsigned int *SCB_DEMCR   ;

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
