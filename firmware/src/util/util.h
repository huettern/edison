/*
* @Author: Noah Huetter
* @Date:   2020-03-13 15:33:21
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-03-22 10:31:20
*/
#pragma once

#include "main.h"

void utilDumpHex(const void* data, size_t size);
void utilMemcpy(uint8_t *dst, const uint8_t *src, uint16_t size);

uint8_t utilTic();
uint32_t utilToc(uint8_t id);