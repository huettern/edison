/*
* @Author: Noah Huetter
* @Date:   2020-04-13 13:51:28
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-04-13 13:51:39
*/

#pragma once

/*------------------------------------------------------------------------------
 * Publics
 * ---------------------------------------------------------------------------*/

void NMI_Handler(void);
void HardFault_Handler(void);
void MemManage_Handler(void);
void BusFault_Handler(void);
void UsageFault_Handler(void);
void SVC_Handler(void);
void DebugMon_Handler(void);
void PendSV_Handler(void);
void SysTick_Handler(void);
void EXTI9_5_IRQHandler(void);
void EXTI15_10_IRQHandler(void);
