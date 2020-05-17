/*
* @Author: Noah Huetter
* @Date:   2020-04-13 13:49:34
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-05-17 18:58:40
*/

#include "main.h"
#include "printf.h"
#include "version.h"
#include "microphone.h"
#include "hostinterface.h"
#include "audioprocessing.h"
#include "ai.h"
#include "ai_nnom.h"
#include "cyclecounter.h"
#include "led.h"

#include <stdarg.h>

/*------------------------------------------------------------------------------
 * Private data
 * ---------------------------------------------------------------------------*/

/*------------------------------------------------------------------------------
 * Prototypes
 * ---------------------------------------------------------------------------*/
static void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_CRC_Init(void);
static void MX_TIM1_Init(void);
static void MX_UART4_Init(void);

/*------------------------------------------------------------------------------
 * Publics
 * ---------------------------------------------------------------------------*/
static UART_HandleTypeDef *printfUart = &huart4;
/**
 * @brief ll function for printf
 * @details 
 * 
 * @param character 
 */
void _putchar(char character)
{
  // send char to console etc.
  HAL_UART_Transmit(printfUart, (uint8_t *)&character, 1, 1000);
}

void mainSetPrintfUart(UART_HandleTypeDef *p)
{
  printfUart = p;
}

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* Configure the system clock */
  SystemClock_Config();

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART1_UART_Init();
  MX_CRC_Init();
  MX_TIM1_Init();
  MX_UART4_Init();
  ledInit();
  ledSetColorAll(0, 0, 0); ledUpdate(0);
  
  printf("%s / %s / %s / %s\n",
             verProgName, verVersion,
             verBuildDate, verGitSha);
  printf("Hello Arduino!\n");

  printf("%s / %s / %s / %s\n",
             verProgName, verVersion,
             verBuildDate, verGitSha);
  micInit();
  audioInit();

  for(int i = 0; i < 7; i++)
  {
    if(i<5) ledSetColor(i, 0, 0, 200); ledUpdate(0);
    ledSet(ledGet() | (1<<(i%8)));
    HAL_Delay(50);
  }
  ledSet(0xff); HAL_Delay(500);
  for(int i = 0; i < 20; i++)
  {
    ledSetColorAll(0, 0, (19-i)*12);
    ledUpdate(0);
    ledSet(0xff);
    HAL_Delay((20-i)*1);
    ledSet(0x00);
    HAL_Delay(i*1);
  }
  
#ifdef CUBE_VERIFICATION
  MX_X_CUBE_AI_Init();
  while(1) MX_X_CUBE_AI_Process();
#endif
    
#ifdef NNOM_VERIFICATION
  aiNnomTest();
#endif

#ifdef NNOM_KWS_EXAMPLE
  printf("Compiled in NNOM_KWS_EXAMPLE mode!\b");
  appNnomKwsInit();
  appNnomKwsRun();
#endif

  aiInitialize();
  aiPrintInfo();
  // aiRunInferenceHif();

  while(1)
  {
    hifRun();
  }

  /* TESTS --------------------------------------------------------*/
  // audioDevelop();


  /* animaion test --------------------------------------------------------*/
  // animationBreath_t anim;
  // anim.start[0] = 0;
  // anim.start[1] = 0;
  // anim.start[2] = 0;
  // anim.stop[0] = 255;
  // anim.stop[1] = 255;
  // anim.stop[2] = 0;
  // anim.speed = 0.01;
  // anim.ledsOneHot = 0x8;
  // uint8_t idx1 = ledStartBreathAnimation(&anim);

  // anim.ledsOneHot = 0x4;
  // anim.stop[1] = 0;
  // anim.speed = 0.1;
  // uint8_t idx2 = ledStartBreathAnimation(&anim);

  // animationFade_t anim2;
  // anim2.start[0] = 255;
  // anim2.start[1] = 255;
  // anim2.start[2] = 255;
  // anim2.stop[0] = 0;
  // anim2.stop[1] = 0;
  // anim2.stop[2] = 0;
  // anim2.speed = 0.001;
  // anim2.ledsOneHot = 0x10;
  // uint8_t idx3 = ledStartFadeAnimation(&anim2);

  // ledWaitAnimationComplete(idx3);
  // ledStopAnimation(idx1);
  // ledStopAnimation(idx2);
  // printf("complete!\n");
  // while(1);

  /* led test --------------------------------------------------------*/
  // while(1)
  // {
  //   ledSetColorAll(0xff, 0x00, 0x00);
  //   ledUpdate(1);

  //   while (1)
  //   {
  //     for (int i = 0; i < LED_CFG_LEDS_CNT; i++) 
  //     {
  //       ledSetColor((i + 0) % LED_CFG_LEDS_CNT, 0x1F, 0, 0);
  //       ledSetColor((i + 1) % LED_CFG_LEDS_CNT, 0, 0x1F, 0);
  //       ledSetColor((i + 2) % LED_CFG_LEDS_CNT, 0, 0, 0x1F);
  //       ledSetColor((i + 3) % LED_CFG_LEDS_CNT, 0, 0, 0);
  //       ledSetColor((i + 4) % LED_CFG_LEDS_CNT, 0, 0, 0);
  //       ledUpdate(1);
  //       ledSetColorAll(0, 0, 0);
        
  //       HAL_Delay(100);
  //     }
  //   }
  // }

  /* net input format --------------------------------------------------------*/
  // static float netInput[AI_NET_INSIZE_BYTES/4];
  // static float netOutput[AI_NET_OUTSIZE_BYTES/4];
  // uint32_t ctr = 0;

  // for(int i = 0; i < AI_NET_INSIZE_BYTES/4; i++) netInput[i] = 0;
  // (void)aiRunInference((void*)netInput, (void*)netOutput);
  // printf("all zero: %f inf\n", netOutput[0]);

  // netInput[ctr] = 1.0;
  // (void)aiRunInference((void*)netInput, (void*)netOutput);
  // printf("[%03d] = 1.0: %f inf\n", ctr, netOutput[0]);

  // for(int i = 0; i < AI_NET_INSIZE_BYTES/4; i++)
  // {
  //   netInput[ctr] = 1.0;
  //   (void)aiRunInference((void*)netInput, (void*)netOutput);
  //   printf("[%03d] = 1.0: %f inf\n", ctr, netOutput[0]);
  //   netInput[ctr] = 0.0;
  //   ctr++;
  // }

  /* Profiler --------------------------------------------------------*/
  // cycProfStart("test");
  // HAL_Delay(1000);
  // cycProfEvent("HAL_Delay(1000)");
  // HAL_Delay(10);
  // cycProfEvent("HAL_Delay(10)");
  // HAL_Delay(1);
  // cycProfEvent("HAL_Delay(1)");
  // HAL_Delay(5463);
  // cycProfEvent("HAL_Delay(5463)");
  // cycProfStop();

  /* Timer 1 --------------------------------------------------------*/
  // uint8_t id1, id2;
  // uint16_t last, delta;
  // uint32_t elapsed;
  // while(1)
  // {
  //   // delta = __HAL_TIM_GET_COUNTER(&htim1)-last;
  //   // last=__HAL_TIM_GET_COUNTER(&htim1);
  //   // printf("tim cnt delta %d\n", delta);
  //   id1 = utilTic();
  //   id2 = utilTic();
  //   HAL_Delay(123);
  //   elapsed = utilToc(id1);
  //   printf("elapsed: %.3fms id %d\n", (float)elapsed/1000.0, id1);
  //   elapsed = utilToc(id2);
  //   printf("elapsed: %.3fms id %d\n", (float)elapsed/1000.0, id2);
  // }
  

  /* Host interface --------------------------------------------------------*/

  // uint8_t tmpu8[8];
  // int8_t tmps8[8];
  // uint16_t tmpu16[8];
  // int16_t tmps16[8];
  // uint32_t tmpu32[8];
  // int32_t tmps32[8];
  // float tmpf32[8];
  // uint32_t length;
  // uint8_t tag;
  

  // for(int i = 0; i < 6; i++)
  // {
  //   tmpu8[i] = i-2;
  //   tmps8[i] = i-2;
  //   tmpu16[i] = i-2;
  //   tmps16[i] = i-2;
  //   tmpu32[i] = i-2;
  //   tmps32[i] = i-2;
  // }

  // hiSendU8(tmpu8, 6, 0xee);
  // hiSendS8(tmps8, 6, 0xee);
  // hiSendU16(tmpu16, 6, 0xee);
  // hiSendS16(tmps16, 6, 0xee);
  // hiSendU32(tmpu32, 6, 0xee);
  // hiSendS32(tmps32, 6, 0xee);
  // while(1)
  // {
  //   HAL_Delay(1000);
  //   // hiSendU8(tmpu8, 6, 0xee);
  // }

  /* ping test --------------------------------------------------------*/

  // while(1)
  // {
  //   length = hiReceive(tmpu8, 8, DATA_FORMAT_U8, &tag);
  //   printf("Received %d elements with tag %d\n[ ", length, tag);
  //   for(int i = 0; i < length; i++) printf("%d ", tmpu8[i]);
  //   printf("]\n");

  //   length = hiReceive(tmps8, 8, DATA_FORMAT_S8, &tag);
  //   printf("Received %d elements with tag %d\n[ ", length, tag);
  //   for(int i = 0; i < length; i++) printf("%d ", tmps8[i]);
  //   printf("]\n");

  //   length = hiReceive(tmpu16, 16, DATA_FORMAT_U16, &tag);
  //   printf("Received %d elements with tag %d\n[ ", length, tag);
  //   for(int i = 0; i < length; i++) printf("%d ", tmpu16[i]);
  //   printf("]\n");

  //   length = hiReceive(tmps16, 16, DATA_FORMAT_S16, &tag);
  //   printf("Received %d elements with tag %d\n[ ", length, tag);
  //   for(int i = 0; i < length; i++) printf("%d ", tmps16[i]);
  //   printf("]\n");

  //   length = hiReceive(tmpu32, 32, DATA_FORMAT_U32, &tag);
  //   printf("Received %d elements with tag %d\n[ ", length, tag);
  //   for(int i = 0; i < length; i++) printf("%u ", tmpu32[i]);
  //   printf("]\n");

  //   length = hiReceive(tmps32, 32, DATA_FORMAT_S32, &tag);
  //   printf("Received %d elements with tag %d\n[ ", length, tag);
  //   for(int i = 0; i < length; i++) printf("%d ", tmps32[i]);
  //   printf("]\n");

  //   length = hiReceive(tmpf32, 32, DATA_FORMAT_F32, &tag);
  //   printf("Received %d elements with tag %d\n[ ", length, tag);
  //   for(int i = 0; i < length; i++) printf("%f ", tmpf32[i]);
  //   printf("]\n");
  // }

  /* pont test --------------------------------------------------------*/

  // while(1)
  // {
  //   HAL_Delay(2000);

  //   HAL_Delay(1000);
  //   hiSendU8(tmpu8, 6, 0xee);

  //   HAL_Delay(1000);
  //   hiSendS8(tmps8, 6, 0xee);

  //   HAL_Delay(1000);
  //   hiSendU16(tmpu16, 6, 0xee);

  //   HAL_Delay(1000);
  //   hiSendS16(tmps16, 6, 0xee);

  //   HAL_Delay(1000);
  //   hiSendU32(tmpu32, 6, 0xee);

  //   HAL_Delay(1000);
  //   hiSendS32(tmps32, 6, 0xee);

  //   tmpf32[0]=0.0;tmpf32[1]=-1.2345;tmpf32[2]=9999.987;
  //   HAL_Delay(1000);
  //   hiSendF32(tmpf32, 6, 0xee);
  // }


  /* pingpong test --------------------------------------------------------*/

  // while(1)
  // {
  //   length = hiReceive(tmpu8, 8, DATA_FORMAT_U8, &tag);
  //   printf("Received %d elements with tag %d\n[ ", length, tag);
  //   for(int i = 0; i < length; i++) printf("%d ", tmpu8[i]);
  //   printf("]\n");
  //   hiSendU8(tmpu8, 6, 0xee);

  //   length = hiReceive(tmps8, 8, DATA_FORMAT_S8, &tag);
  //   printf("Received %d elements with tag %d\n[ ", length, tag);
  //   for(int i = 0; i < length; i++) printf("%d ", tmps8[i]);
  //   printf("]\n");
  //   hiSendS8(tmps8, 6, 0xee);

  //   length = hiReceive(tmpu16, 16, DATA_FORMAT_U16, &tag);
  //   printf("Received %d elements with tag %d\n[ ", length, tag);
  //   for(int i = 0; i < length; i++) printf("%d ", tmpu16[i]);
  //   printf("]\n");
  //   hiSendU16(tmpu16, 6, 0xee);

  //   length = hiReceive(tmps16, 16, DATA_FORMAT_S16, &tag);
  //   printf("Received %d elements with tag %d\n[ ", length, tag);
  //   for(int i = 0; i < length; i++) printf("%d ", tmps16[i]);
  //   printf("]\n");
  //   hiSendS16(tmps16, 6, 0xee);

  //   length = hiReceive(tmpu32, 32, DATA_FORMAT_U32, &tag);
  //   printf("Received %d elements with tag %d\n[ ", length, tag);
  //   for(int i = 0; i < length; i++) printf("%u ", tmpu32[i]);
  //   printf("]\n");
  //   hiSendU32(tmpu32, 6, 0xee);

  //   length = hiReceive(tmps32, 32, DATA_FORMAT_S32, &tag);
  //   printf("Received %d elements with tag %d\n[ ", length, tag);
  //   for(int i = 0; i < length; i++) printf("%d ", tmps32[i]);
  //   printf("]\n");
  //   hiSendS32(tmps32, 6, 0xee);

  //   length = hiReceive(tmpf32, 32, DATA_FORMAT_S32, &tag);
  //   printf("Received %d elements with tag %d\n[ ", length, tag);
  //   for(int i = 0; i < length; i++) printf("%f ", tmpf32[i]);
  //   printf("]\n");
  //   hiSendS32(tmpf32, 6, 0xee);
  // }



  /* mic test --------------------------------------------------------*/

  // int32_t* data;
  // int8_t* data8;
  // int32_t databuf[256];
  
  // micEndlessStream();
  // micReqSampling();
  // micSampleSinglePreprocessed(&data8, 5000);

  // for(int i = 0; i < 5000; i++)
  // {
  //   printf("%d\r\n", data8[i]);
  // }

  /* Infinite loop */
  // while (1)
  // {
  //   // for(int i = 0; i < 256; i++)
  //   // {
  //   //   HAL_Delay(10);
  //   //   micSampleSingle(&databuf[i], 1);
  //   // }
  //   micSampleSingle(&data, 10);

  //   printf(">>>\n");
  //   for(int i = 0; i < 10; i++)
  //     printf("%032b %10d\n",data[i],data[i]);
  //   printf("<<<\n");

  //   // printf("x=[");
  //   // for(int i = 0; i < 10; i++)
  //   //   printf("%d,",data[i]);
  //   // printf("];\n");

  //   HAL_Delay(1000);
  //   // HAL_GPIO_TogglePin(LED2_GPIO_Port, LED2_Pin);
  //   HAL_GPIO_TogglePin(LED3_WIFI__LED4_BLE_GPIO_Port, LED3_WIFI__LED4_BLE_Pin);
  //   HAL_Delay(1000);
  
  //   // printf("Start sampling...\n");
  //   // micSampleSingle(data, 1);
  //   // printf("%5d\n", data[0]);


  // }
  // while (1)
  // {
  //   HAL_Delay(500);
  //   HAL_GPIO_TogglePin(LED2_GPIO_Port, LED2_Pin);
  //   HAL_GPIO_TogglePin(LED3_WIFI__LED4_BLE_GPIO_Port, LED3_WIFI__LED4_BLE_Pin);
  //   HAL_Delay(500);
  //   HAL_GPIO_TogglePin(LED2_GPIO_Port, LED2_Pin);
  //   HAL_GPIO_TogglePin(LED3_WIFI__LED4_BLE_GPIO_Port, LED3_WIFI__LED4_BLE_Pin);
  // }
}

/*------------------------------------------------------------------------------
 * Privates
 * ---------------------------------------------------------------------------*/
/**
  * @brief System Clock Configuration
  * @retval None
  */
static void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

  /** Configure LSE Drive Capability 
  */
  HAL_PWR_EnableBkUpAccess();
  __HAL_RCC_LSEDRIVE_CONFIG(RCC_LSEDRIVE_LOW);
  /** Initializes the CPU, AHB and APB busses clocks 
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_LSE|RCC_OSCILLATORTYPE_MSI;
  RCC_OscInitStruct.LSEState = RCC_LSE_ON;
  RCC_OscInitStruct.MSIState = RCC_MSI_ON;
  RCC_OscInitStruct.MSICalibrationValue = 0;
  RCC_OscInitStruct.MSIClockRange = RCC_MSIRANGE_6;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_MSI;
  RCC_OscInitStruct.PLL.PLLM = 1;
  RCC_OscInitStruct.PLL.PLLN = 40;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV7;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /** Initializes the CPU, AHB and APB busses clocks 
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
  PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_USART1|RCC_PERIPHCLK_USART3
                              |RCC_PERIPHCLK_I2C2|RCC_PERIPHCLK_DFSDM1
                              |RCC_PERIPHCLK_USB;
  PeriphClkInit.Usart1ClockSelection = RCC_USART1CLKSOURCE_PCLK2;
  PeriphClkInit.Usart3ClockSelection = RCC_USART3CLKSOURCE_PCLK1;
  PeriphClkInit.I2c2ClockSelection = RCC_I2C2CLKSOURCE_PCLK1;
  PeriphClkInit.Dfsdm1ClockSelection = RCC_DFSDM1CLKSOURCE_PCLK;
  PeriphClkInit.UsbClockSelection = RCC_USBCLKSOURCE_PLLSAI1;
  PeriphClkInit.PLLSAI1.PLLSAI1Source = RCC_PLLSOURCE_MSI;
  PeriphClkInit.PLLSAI1.PLLSAI1M = 1;
  PeriphClkInit.PLLSAI1.PLLSAI1N = 24;
  PeriphClkInit.PLLSAI1.PLLSAI1P = RCC_PLLP_DIV7;
  PeriphClkInit.PLLSAI1.PLLSAI1Q = RCC_PLLQ_DIV2;
  PeriphClkInit.PLLSAI1.PLLSAI1R = RCC_PLLR_DIV2;
  PeriphClkInit.PLLSAI1.PLLSAI1ClockOut = RCC_PLLSAI1_48M2CLK;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
  {
    Error_Handler();
  }
  /** Configure the main internal regulator output voltage 
  */
  if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1) != HAL_OK)
  {
    Error_Handler();
  }
  /** Enable MSI Auto calibration 
  */
  HAL_RCCEx_EnableMSIPLLMode();
}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
}


/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOE, M24SR64_Y_RF_DISABLE_Pin|M24SR64_Y_GPO_Pin|ISM43362_RST_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, ARD_D10_Pin|SPBTLE_RF_RST_Pin|ARD_D9_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, ARD_D8_Pin|ISM43362_BOOT0_Pin|ISM43362_WAKEUP_Pin|LED2_Pin 
                          |SPSGRF_915_SDN_Pin|ARD_D5_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOD, USB_OTG_FS_PWR_EN_Pin|PMOD_RESET_Pin|STSAFE_A100_RESET_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(SPBTLE_RF_SPI3_CSN_GPIO_Port, SPBTLE_RF_SPI3_CSN_Pin, GPIO_PIN_SET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOC, VL53L0X_XSHUT_Pin|LED3_WIFI__LED4_BLE_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(SPSGRF_915_SPI3_CSN_GPIO_Port, SPSGRF_915_SPI3_CSN_Pin, GPIO_PIN_SET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(ISM43362_SPI3_CSN_GPIO_Port, ISM43362_SPI3_CSN_Pin, GPIO_PIN_SET);

  /*Configure GPIO pins : M24SR64_Y_RF_DISABLE_Pin M24SR64_Y_GPO_Pin ISM43362_RST_Pin ISM43362_SPI3_CSN_Pin */
  GPIO_InitStruct.Pin = M24SR64_Y_RF_DISABLE_Pin|M24SR64_Y_GPO_Pin|ISM43362_RST_Pin|ISM43362_SPI3_CSN_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  /*Configure GPIO pins : USB_OTG_FS_OVRCR_EXTI3_Pin SPSGRF_915_GPIO3_EXTI5_Pin SPBTLE_RF_IRQ_EXTI6_Pin ISM43362_DRDY_EXTI1_Pin */
  GPIO_InitStruct.Pin = USB_OTG_FS_OVRCR_EXTI3_Pin|SPSGRF_915_GPIO3_EXTI5_Pin|SPBTLE_RF_IRQ_EXTI6_Pin|ISM43362_DRDY_EXTI1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  /*Configure GPIO pin : BUTTON_EXTI13_Pin */
  GPIO_InitStruct.Pin = BUTTON_EXTI13_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(BUTTON_EXTI13_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : ARD_A5_Pin ARD_A4_Pin ARD_A3_Pin ARD_A2_Pin 
                           ARD_A1_Pin ARD_A0_Pin */
  GPIO_InitStruct.Pin = ARD_A5_Pin|ARD_A4_Pin|ARD_A3_Pin|ARD_A2_Pin 
                          |ARD_A1_Pin|ARD_A0_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG_ADC_CONTROL;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pins : ARD_D1_Pin ARD_D0_Pin */
  GPIO_InitStruct.Pin = ARD_D1_Pin|ARD_D0_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF8_UART4;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : ARD_D10_Pin SPBTLE_RF_RST_Pin ARD_D9_Pin */
  GPIO_InitStruct.Pin = ARD_D10_Pin|SPBTLE_RF_RST_Pin|ARD_D9_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pin : ARD_D4_Pin */
  GPIO_InitStruct.Pin = ARD_D4_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF1_TIM2;
  HAL_GPIO_Init(ARD_D4_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : ARD_D7_Pin */
  GPIO_InitStruct.Pin = ARD_D7_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG_ADC_CONTROL;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(ARD_D7_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : ARD_D13_Pin ARD_D12_Pin ARD_D11_Pin */
  GPIO_InitStruct.Pin = ARD_D13_Pin|ARD_D12_Pin|ARD_D11_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF5_SPI1;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pin : ARD_D3_Pin */
  GPIO_InitStruct.Pin = ARD_D3_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(ARD_D3_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : ARD_D6_Pin */
  GPIO_InitStruct.Pin = ARD_D6_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG_ADC_CONTROL;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(ARD_D6_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : ARD_D8_Pin ISM43362_BOOT0_Pin ISM43362_WAKEUP_Pin LED2_Pin 
                           SPSGRF_915_SDN_Pin ARD_D5_Pin SPSGRF_915_SPI3_CSN_Pin */
  GPIO_InitStruct.Pin = ARD_D8_Pin|ISM43362_BOOT0_Pin|ISM43362_WAKEUP_Pin|LED2_Pin 
                          |SPSGRF_915_SDN_Pin|ARD_D5_Pin|SPSGRF_915_SPI3_CSN_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pins : LPS22HB_INT_DRDY_EXTI0_Pin LSM6DSL_INT1_EXTI11_Pin ARD_D2_Pin HTS221_DRDY_EXTI15_Pin 
                           PMOD_IRQ_EXTI12_Pin */
  GPIO_InitStruct.Pin = LPS22HB_INT_DRDY_EXTI0_Pin|LSM6DSL_INT1_EXTI11_Pin|ARD_D2_Pin|HTS221_DRDY_EXTI15_Pin 
                          |PMOD_IRQ_EXTI12_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  /*Configure GPIO pins : USB_OTG_FS_PWR_EN_Pin SPBTLE_RF_SPI3_CSN_Pin PMOD_RESET_Pin STSAFE_A100_RESET_Pin */
  GPIO_InitStruct.Pin = USB_OTG_FS_PWR_EN_Pin|SPBTLE_RF_SPI3_CSN_Pin|PMOD_RESET_Pin|STSAFE_A100_RESET_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  /*Configure GPIO pins : VL53L0X_XSHUT_Pin LED3_WIFI__LED4_BLE_Pin */
  GPIO_InitStruct.Pin = VL53L0X_XSHUT_Pin|LED3_WIFI__LED4_BLE_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pins : VL53L0X_GPIO1_EXTI7_Pin LSM3MDL_DRDY_EXTI8_Pin */
  GPIO_InitStruct.Pin = VL53L0X_GPIO1_EXTI7_Pin|LSM3MDL_DRDY_EXTI8_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pin : PMOD_SPI2_SCK_Pin */
  GPIO_InitStruct.Pin = PMOD_SPI2_SCK_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF5_SPI2;
  HAL_GPIO_Init(PMOD_SPI2_SCK_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : PMOD_UART2_CTS_Pin PMOD_UART2_RTS_Pin PMOD_UART2_TX_Pin PMOD_UART2_RX_Pin */
  GPIO_InitStruct.Pin = PMOD_UART2_CTS_Pin|PMOD_UART2_RTS_Pin|PMOD_UART2_TX_Pin|PMOD_UART2_RX_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF7_USART2;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  /*Configure GPIO pins : ARD_D15_Pin ARD_D14_Pin */
  GPIO_InitStruct.Pin = ARD_D15_Pin|ARD_D14_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_OD;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF4_I2C1;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /* EXTI interrupt init*/
  HAL_NVIC_SetPriority(EXTI9_5_IRQn, MAIN_IRQ_EXTI9_5_PRE, MAIN_IRQ_EXTI9_5_SUB);
  HAL_NVIC_EnableIRQ(EXTI9_5_IRQn);

  HAL_NVIC_SetPriority(EXTI15_10_IRQn, MAIN_IRQ_EXTI15_10_PRE, MAIN_IRQ_EXTI15_10_SUB);
  HAL_NVIC_EnableIRQ(EXTI15_10_IRQn);
}

/**
  * @brief CRC Initialization Function
  * @param None
  * @retval None
  */
static void MX_CRC_Init(void)
{
  __HAL_RCC_CRC_CLK_ENABLE();
  hcrc.Instance = CRC;
  hcrc.Init.DefaultPolynomialUse = DEFAULT_POLYNOMIAL_ENABLE;
  hcrc.Init.DefaultInitValueUse = DEFAULT_INIT_VALUE_ENABLE;
  hcrc.Init.InputDataInversionMode = CRC_INPUTDATA_INVERSION_NONE;
  hcrc.Init.OutputDataInversionMode = CRC_OUTPUTDATA_INVERSION_DISABLE;
  hcrc.InputDataFormat = CRC_INPUTDATA_FORMAT_BYTES;
  if (HAL_CRC_Init(&hcrc) != HAL_OK)
  {
    Error_Handler();
  }
}

static void MX_TIM1_Init(void)
{
  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  htim1.Instance = TIM1;
  // ticks in 100us ticks -> can count up to 6.5s with .1ms accuracy
  htim1.Init.Prescaler = 80*MAIN_TIM1_TICK_US;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 0xffff;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim1, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterOutputTrigger2 = TIM_TRGO2_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  __HAL_TIM_ENABLE(&htim1);
  HAL_TIM_Base_Start(&htim1);

  sConfigOC.OCMode = TIM_OCMODE_ACTIVE;
  sConfigOC.Pulse = MAIN_TIM1_CH1_INTERVAL_US/MAIN_TIM1_TICK_US;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCNPolarity = TIM_OCNPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  sConfigOC.OCIdleState = TIM_OCIDLESTATE_RESET;
  sConfigOC.OCNIdleState = TIM_OCNIDLESTATE_RESET;
  if (HAL_TIM_OC_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  // enable TIM1 capture compare interrupt
  HAL_NVIC_SetPriority(TIM1_CC_IRQn, MAIN_IRQ_TIM1_CC_PRE, MAIN_IRQ_TIM1_CC_SUB);
  HAL_NVIC_EnableIRQ(TIM1_CC_IRQn);
  HAL_TIM_OC_Start_IT(&htim1, MAIN_TIM1_ANIMATION_CHANNEL);
}


/**
  * @brief UART4 Initialization Function
  * @param None
  * @retval None
  */
static void MX_UART4_Init(void)
{
  huart4.Instance = UART4;
  huart4.Init.BaudRate = 115200;
  huart4.Init.WordLength = UART_WORDLENGTH_8B;
  huart4.Init.StopBits = UART_STOPBITS_1;
  huart4.Init.Parity = UART_PARITY_NONE;
  huart4.Init.Mode = UART_MODE_TX_RX;
  huart4.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart4.Init.OverSampling = UART_OVERSAMPLING_16;
  huart4.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart4.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart4) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  printf("Error Handler\n");
  while(1);
}

#ifdef  USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{ 
}
#endif /* USE_FULL_ASSERT */

