/*
* @Author: Noah Huetter
* @Date:   2020-04-14 13:49:21
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-04-14 13:49:21
*/
#pragma once

#include "main.h"


typedef enum
{
  DATA_FORMAT_U8 = '0',
  DATA_FORMAT_S8 = '1',
  DATA_FORMAT_U16 = '2',
  DATA_FORMAT_S16 = '3',
  DATA_FORMAT_U32 = '4',
  DATA_FORMAT_S32 = '5',
} hiDataFormat_t;

void hifRun(void);

void hiSendU8(uint8_t * data, uint32_t len, uint8_t tag);
void hiSendS8(int8_t * data, uint32_t len, uint8_t tag);
void hiSendU16(uint16_t * data, uint32_t len, uint8_t tag);
void hiSendS16(int16_t * data, uint32_t len, uint8_t tag);
void hiSendU32(uint32_t * data, uint32_t len, uint8_t tag);
void hiSendS32(int32_t * data, uint32_t len, uint8_t tag);

uint32_t hiReceive(void * data, uint32_t maxlen, hiDataFormat_t fmt, uint8_t * tag);