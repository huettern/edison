/*
* @Author: Noah Huetter
* @Date:   2020-05-14 21:05:15
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-05-14 21:24:09
*/
#include "led.h"

static uint8_t leds;

void ledInit(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  leds = 0;
  // D0 -> Rx4
  // D1 -> Tx4

  // D2 -> D14
  // D3 -> B0
  // D4 -> A3
  // D5 -> B4

  // D6 -> B1
  // D7 -> A4
  // D8 -> B2
  // D9 -> A15

  // D10 -> A2
  // D11 -> A7
  // D12 -> A6
  // D13 -> A5
  GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_1|GPIO_PIN_4|GPIO_PIN_2;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);
  GPIO_InitStruct.Pin = GPIO_PIN_3|GPIO_PIN_4|GPIO_PIN_2|GPIO_PIN_5|GPIO_PIN_6|GPIO_PIN_7|GPIO_PIN_15;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
  GPIO_InitStruct.Pin = GPIO_PIN_14;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);
}

void ledSet(uint8_t s)
{
  leds = s;

  // set: GPIOx->BSRR = (uint32_t)GPIO_Pin;
  // reset: GPIOx->BRR = (uint32_t)GPIO_Pin;
  if(leds & 0x01) GPIOD->BSRR = GPIO_PIN_14; else GPIOD->BRR = GPIO_PIN_14;
  if(leds & 0x02) GPIOB->BSRR = GPIO_PIN_0; else GPIOB->BRR = GPIO_PIN_0;
  if(leds & 0x04) GPIOA->BSRR = GPIO_PIN_3; else GPIOA->BRR = GPIO_PIN_3;
  if(leds & 0x08) GPIOB->BSRR = GPIO_PIN_4; else GPIOB->BRR = GPIO_PIN_4;
  
  if(leds & 0x10) GPIOB->BSRR = GPIO_PIN_1; else GPIOB->BRR = GPIO_PIN_1;
  if(leds & 0x20) GPIOA->BSRR = GPIO_PIN_4; else GPIOA->BRR = GPIO_PIN_4;
  if(leds & 0x40) GPIOB->BSRR = GPIO_PIN_2; else GPIOB->BRR = GPIO_PIN_2;
  if(leds & 0x80) GPIOA->BSRR = GPIO_PIN_15; else GPIOA->BRR = GPIO_PIN_15;
}

uint8_t ledGet(void)
{
  return leds;
}