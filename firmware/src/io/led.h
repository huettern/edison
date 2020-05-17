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

#define LED_CFG_LEDS_CNT                10       /*!< Number of leds in a strip row */

// Fade animaiton
typedef struct 
{
  // start color
  float start[3]; //rgb
  // stop color
  float stop[3]; //rgb
  // state
  float state[4]; //rgb - count
  // fade speed
  float speed; // 1/steps from start to stop (1 fast, 255 slow)
  // one-hot encoded leds that are involved in the animation
  uint32_t ledsOneHot;
} animationFade_t;

// Fade animaiton
typedef struct 
{
  // start color
  float start[3]; //rgb
  // stop color
  float stop[3]; //rgb
  // state
  float state[4]; //rgb - count
  // fade speed
  float speed; // 1/steps from start to stop (1 fast, 255 slow)
  // one-hot encoded leds that are involved in the animation
  uint32_t ledsOneHot;
} animationBreath_t;

uint8_t ledSetColor(size_t index, uint8_t r, uint8_t g, uint8_t b);
uint8_t ledSetColorAll(uint8_t r, uint8_t g, uint8_t b);
uint8_t ledSetColorRgb(size_t index, uint32_t rgb);
uint32_t ledGetColorRgb(size_t index);
uint8_t ledSetColorAllRgb(uint32_t rgb);
uint8_t ledUpdate(uint8_t block) ;
uint8_t ledIsUpdateFinished(void);
void ledAnimationCallback(void);

uint8_t ledStartFadeAnimation(animationFade_t *anim);
uint8_t ledStartBreathAnimation(animationBreath_t *anim);

void ledStopAnimation(uint8_t idx);

void ledWaitAnimationComplete(uint8_t idx);