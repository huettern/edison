/*
* @Author: Noah Huetter
* @Date:   2020-04-14 13:49:21
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-04-14 13:49:21
*/
#pragma once

#include "main.h"

void hifRun(void);

void hiSendU8(uint8_t * data, uint32_t len, uint8_t tag);
void hiSendS8(int8_t * data, uint32_t len, uint8_t tag);
void hiSendU16(uint16_t * data, uint32_t len, uint8_t tag);
void hiSendS16(int16_t * data, uint32_t len, uint8_t tag);
void hiSendU32(uint32_t * data, uint32_t len, uint8_t tag);
void hiSendS32(int32_t * data, uint32_t len, uint8_t tag);