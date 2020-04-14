/*
* @Author: Noah Huetter
* @Date:   2020-04-13 13:56:56
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-04-14 08:29:17
*/
#include "main.h"

/**
 * f_ckin     Clock frequency of data line: f_ckin = 80e6 / clockDivider
 * f_osr      Filter oversampling ratio = oversampling
 * f_ord      Filter order, [1,5] for sinc
 * I_osr      Integrator oversampling ratio, [1,256] = intOversampling
 * 
 * Output datarate for FAST=0, SincX filter
 *    DR [Samples/s] = f_ckin / (f_osr * (I_osr - 1 + f_ord ) + (f_ord + 1) )
 * Output datarate for FAST=0, FastSinc filter
 *    DR [Samples/s] = f_ckin / (f_osr * (I_osr - 1 + 4 ) + (2 + 1) )
 * Output datarate for FAST=1
 *    DR [Samples/s] = f_ckin / (f_osr * I_osr)
 *   
 * Cut-off frequency
 */

/*------------------------------------------------------------------------------
 * Types
 * ---------------------------------------------------------------------------*/

typedef struct 
{
  // 2..256
  uint32_t clockDivider;
  // 1..32
  uint16_t oversampling;
  // between Min_Data = 0x00 and Max_Data = 0x1F
  uint32_t rightBitShift;
  // between Min_Data = -8388608 and Max_Data = 8388607
  int32_t offset;
  // DFSDM_FILTER_FASTSINC_ORDER
  // DFSDM_FILTER_SINC1_ORDER
  // DFSDM_FILTER_SINC2_ORDER
  // DFSDM_FILTER_SINC3_ORDER
  // DFSDM_FILTER_SINC4_ORDER
  // DFSDM_FILTER_SINC5_ORDER
  uint32_t sincOrder;
  // 1..256
  uint32_t intOversampling;
} filterSettings_t;
/**
 * Results in 5kHz sample rate
 */
const filterSettings_t fs5k = {33, 484, 0x00, 0, DFSDM_FILTER_SINC3_ORDER,1};


/*------------------------------------------------------------------------------
 * Settings
 * ---------------------------------------------------------------------------*/
#define MIC_BUFFER_SIZE 20000

const filterSettings_t* filterSettings = &fs5k;

/*------------------------------------------------------------------------------
 * Private data
 * ---------------------------------------------------------------------------*/
static int32_t dataBuffer [MIC_BUFFER_SIZE];

static bool regConvCplt = false;
static bool regConvHalfCplt = false;

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
  hdfsdm1_ch1.Init.OutputClock.Divider = filterSettings->clockDivider; // from Cube: 2
  hdfsdm1_ch1.Init.Input.Multiplexer = DFSDM_CHANNEL_EXTERNAL_INPUTS;
  hdfsdm1_ch1.Init.Input.DataPacking = DFSDM_CHANNEL_STANDARD_MODE;
  hdfsdm1_ch1.Init.Input.Pins = DFSDM_CHANNEL_SAME_CHANNEL_PINS;
  hdfsdm1_ch1.Init.SerialInterface.Type = DFSDM_CHANNEL_SPI_RISING;
  hdfsdm1_ch1.Init.SerialInterface.SpiClock = DFSDM_CHANNEL_SPI_CLOCK_INTERNAL;
  hdfsdm1_ch1.Init.Awd.FilterOrder = DFSDM_CHANNEL_FASTSINC_ORDER;
  hdfsdm1_ch1.Init.Awd.Oversampling = 1;
  hdfsdm1_ch1.Init.Offset = filterSettings->offset;
  hdfsdm1_ch1.Init.RightBitShift = filterSettings->rightBitShift;
  if (HAL_DFSDM_ChannelInit(&hdfsdm1_ch1) != HAL_OK)
  {
    Error_Handler();
  }

  // Init DMA 
  __HAL_RCC_DMA1_CLK_ENABLE();
  HAL_NVIC_SetPriority(DMA1_Channel4_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Channel4_IRQn);
  hdma_dfsdm1_flt0.Instance = DMA1_Channel4;
  hdma_dfsdm1_flt0.Init.Request = DMA_REQUEST_0; // ref man page 340 CxS value
  hdma_dfsdm1_flt0.Init.Direction = DMA_PERIPH_TO_MEMORY;
  hdma_dfsdm1_flt0.Init.PeriphInc = DMA_PINC_DISABLE;
  hdma_dfsdm1_flt0.Init.MemInc = DMA_MINC_ENABLE;
  hdma_dfsdm1_flt0.Init.PeriphDataAlignment = DMA_PDATAALIGN_WORD; // BYTE HALFWORD WORD
  hdma_dfsdm1_flt0.Init.MemDataAlignment = DMA_MDATAALIGN_WORD;
  hdma_dfsdm1_flt0.Init.Mode = DMA_CIRCULAR; // cicular
  hdma_dfsdm1_flt0.Init.Priority = DMA_PRIORITY_HIGH; // LOW MEDIUM HIGH VERYHIGH
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
  hdfsdm1_filter0.Init.FilterParam.SincOrder = filterSettings->sincOrder;
  hdfsdm1_filter0.Init.FilterParam.Oversampling = filterSettings->oversampling; // 100->24kHz, 242->10Khz, 484->5kHz
  hdfsdm1_filter0.Init.FilterParam.IntOversampling = filterSettings->intOversampling;
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

void micEndlessStream(void)
{
  int32_t* data = dataBuffer;
  const uint32_t chunksize=128;

  while(1)
  {
    regConvCplt = false; regConvHalfCplt = false;
    
    // start
    if (HAL_DFSDM_FilterRegularStart_DMA(&hdfsdm1_filter0, dataBuffer, chunksize) != HAL_OK)
    {
      Error_Handler();
    }

    while(!regConvHalfCplt);

    for(int i = 0; i < chunksize/2; i++)
    {
      printf("%d\r\n", data[i]>>16);
    }

    while(!regConvCplt);

    for(int i = 0; i < chunksize/2; i++)
    {
      printf("%d\r\n", data[chunksize/2+i]>>16);
    }

    HAL_DFSDM_FilterRegularStop_DMA(&hdfsdm1_filter0);
  }
}

/*------------------------------------------------------------------------------
 * Privates
 * ---------------------------------------------------------------------------*/


/*------------------------------------------------------------------------------
 * Callbacks
 * ---------------------------------------------------------------------------*/
void HAL_DFSDM_FilterRegConvCpltCallback(DFSDM_Filter_HandleTypeDef *hdfsdm_filter)
{
  regConvCplt = true;
  // printf("HAL_DFSDM_FilterRegConvCpltCallback\n");
  // HAL_GPIO_TogglePin(LED2_GPIO_Port, LED2_Pin);
}

void HAL_DFSDM_FilterRegConvHalfCpltCallback(DFSDM_Filter_HandleTypeDef *hdfsdm_filter)
{
  regConvHalfCplt = true;
}

