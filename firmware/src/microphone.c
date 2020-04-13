/*
* @Author: Noah Huetter
* @Date:   2020-04-13 13:56:56
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-04-13 15:52:05
*/
#include "main.h"

/*------------------------------------------------------------------------------
 * Settings
 * ---------------------------------------------------------------------------*/
#define MIC_BUFFER_SIZE 256

/*------------------------------------------------------------------------------
 * Private data
 * ---------------------------------------------------------------------------*/
int32_t dataBuffer [MIC_BUFFER_SIZE];

/*------------------------------------------------------------------------------
 * Prototypes
 * ---------------------------------------------------------------------------*/

/*------------------------------------------------------------------------------
 * Publics
 * ---------------------------------------------------------------------------*/
/**
 * @brief Inits all necessary peripherals for microphone sampling
 * @details 
 */
void micInit(void)
{
  // Init DFSM module
  hdfsdm1_ch1.Instance = DFSDM1_Channel2;
  hdfsdm1_ch1.Init.OutputClock.Activation = ENABLE;
  hdfsdm1_ch1.Init.OutputClock.Selection = DFSDM_CHANNEL_OUTPUT_CLOCK_SYSTEM;
  hdfsdm1_ch1.Init.OutputClock.Divider = 33; // from Cube: 2
  hdfsdm1_ch1.Init.Input.Multiplexer = DFSDM_CHANNEL_EXTERNAL_INPUTS;
  hdfsdm1_ch1.Init.Input.DataPacking = DFSDM_CHANNEL_STANDARD_MODE;
  hdfsdm1_ch1.Init.Input.Pins = DFSDM_CHANNEL_SAME_CHANNEL_PINS;
  hdfsdm1_ch1.Init.SerialInterface.Type = DFSDM_CHANNEL_SPI_RISING;
  hdfsdm1_ch1.Init.SerialInterface.SpiClock = DFSDM_CHANNEL_SPI_CLOCK_INTERNAL;
  hdfsdm1_ch1.Init.Awd.FilterOrder = DFSDM_CHANNEL_FASTSINC_ORDER;
  hdfsdm1_ch1.Init.Awd.Oversampling = 1;
  hdfsdm1_ch1.Init.Offset = 0;
  hdfsdm1_ch1.Init.RightBitShift = 0x00;
  if (HAL_DFSDM_ChannelInit(&hdfsdm1_ch1) != HAL_OK)
  {
    Error_Handler();
  }

  // Init DMA 
  hdma_dfsdm1_flt0.Instance = DMA1_Channel4;
  hdma_dfsdm1_flt0.Init.Request = DMA_REQUEST_0; // ref man page 340 CxS value
  hdma_dfsdm1_flt0.Init.Direction = DMA_PERIPH_TO_MEMORY;
  hdma_dfsdm1_flt0.Init.PeriphInc = DMA_PINC_DISABLE;
  hdma_dfsdm1_flt0.Init.MemInc = DMA_MINC_ENABLE;
  hdma_dfsdm1_flt0.Init.PeriphDataAlignment = DMA_PDATAALIGN_WORD; // BYTE HALFWORD WORD
  hdma_dfsdm1_flt0.Init.MemDataAlignment = DMA_MDATAALIGN_WORD;
  hdma_dfsdm1_flt0.Init.Mode = DMA_CIRCULAR; // cicular
  hdma_dfsdm1_flt0.Init.Priority = DMA_PRIORITY_HIGH; // LOW MEDIUM HIGH VERYHIGH
  __HAL_RCC_DMA1_CLK_ENABLE();
  if (HAL_DMA_Init(&hdma_dfsdm1_flt0) != HAL_OK)
  {
    Error_Handler();
  }
  // Link DMA1_CH4 to DFSDM1_FLT0 regular channel
  __HAL_LINKDMA(&hdfsdm1_filter0,hdmaInj,hdma_dfsdm1_flt0);
  __HAL_LINKDMA(&hdfsdm1_filter0,hdmaReg,hdma_dfsdm1_flt0);

  // Init filter channel
  hdfsdm1_filter0.Instance = DFSDM1_Filter0;
  hdfsdm1_filter0.Init.RegularParam.Trigger = DFSDM_FILTER_SW_TRIGGER;
  hdfsdm1_filter0.Init.RegularParam.FastMode = ENABLE;
  hdfsdm1_filter0.Init.RegularParam.DmaMode = ENABLE;
  hdfsdm1_filter0.Init.FilterParam.SincOrder = DFSDM_FILTER_SINC3_ORDER;
  hdfsdm1_filter0.Init.FilterParam.Oversampling = 100;
  hdfsdm1_filter0.Init.FilterParam.IntOversampling = 1;
  if (HAL_DFSDM_FilterInit(&hdfsdm1_filter0) != HAL_OK)
  {
    Error_Handler();
  }

  // (#) Select regular channel and enable/disable continuous mode using
  //     HAL_DFSDM_FilterConfigRegChannel().
  if (HAL_DFSDM_FilterConfigRegChannel(&hdfsdm1_filter0, DFSDM_CHANNEL_2, DFSDM_CONTINUOUS_CONV_ON) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
 * @brief Samples n samples once and returns pointer to buffer
 * @details 
 * 
 * @param n number of samples to get
 * @return pointer to data buffer
 */
int32_t * micSampleSingle(uint32_t n)
{

  // (#) Start regular conversion using HAL_DFSDM_FilterRegularStart(),
  //     HAL_DFSDM_FilterRegularStart_IT(), HAL_DFSDM_FilterRegularStart_DMA() or
  //     HAL_DFSDM_FilterRegularMsbStart_DMA().
  if (HAL_DFSDM_FilterRegularStart_DMA(&hdfsdm1_filter0, dataBuffer, (n>MIC_BUFFER_SIZE)?MIC_BUFFER_SIZE:n) != HAL_OK)
  {
    Error_Handler();
  }

  // (#) In polling mode, use HAL_DFSDM_FilterPollForRegConversion() to detect
  //     the end of regular conversion.
  // while(1);
  if (HAL_DFSDM_FilterPollForRegConversion(&hdfsdm1_filter0, HAL_MAX_DELAY) != HAL_OK) // block
  {
    Error_Handler();
  }

  // (#) In interrupt mode, HAL_DFSDM_FilterRegConvCpltCallback() will be called
  //     at the end of regular conversion.
  // (#) Get value of regular conversion and corresponding channel using
  //     HAL_DFSDM_FilterGetRegularValue().
  // (#) In DMA mode, HAL_DFSDM_FilterRegConvHalfCpltCallback() and
  //     HAL_DFSDM_FilterRegConvCpltCallback() will be called respectively at the
  //     half transfer and at the transfer complete. Please note that
  //     HAL_DFSDM_FilterRegConvHalfCpltCallback() will be called only in DMA
  //     circular mode.
  // (#) Stop regular conversion using HAL_DFSDM_FilterRegularStop(),
  //     HAL_DFSDM_FilterRegularStop_IT() or HAL_DFSDM_FilterRegularStop_DMA().
  HAL_DFSDM_FilterRegularStop_DMA(&hdfsdm1_filter0);

  return dataBuffer;
}

/*------------------------------------------------------------------------------
 * Privates
 * ---------------------------------------------------------------------------*/


/*------------------------------------------------------------------------------
 * Callbacks
 * ---------------------------------------------------------------------------*/
void HAL_DFSDM_FilterRegConvCpltCallback(DFSDM_Filter_HandleTypeDef *hdfsdm_filter)
{
  printf("HAL_DFSDM_FilterRegConvCpltCallback\n");
}



