/*
* @Author: Noah Huetter
* @Date:   2020-05-14 21:05:15
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-05-14 21:05:15
*/
#pragma once

#include "main.h"

void ledInit(void);
void ledSet(uint8_t s);
uint8_t ledGet(void);

#define LED_CFG_LEDS_CNT                5       /*!< Number of leds in a strip row */

uint8_t ledSetColor(size_t index, uint8_t r, uint8_t g, uint8_t b);
uint8_t ledSetColorAll(uint8_t r, uint8_t g, uint8_t b);
uint8_t ledSetColorRgb(size_t index, uint32_t rgb);
uint8_t ledSetColorAllRgb(uint32_t rgb);
uint8_t ledUpdate(uint8_t block) ;
uint8_t ledIsUpdateFinished(void);