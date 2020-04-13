
#include "main.h"

/**
  * Initializes the Global MSP.
  */
void HAL_MspInit(void)
{
  __HAL_RCC_SYSCFG_CLK_ENABLE();
  __HAL_RCC_PWR_CLK_ENABLE();

  /* System interrupt init*/
}

static uint32_t DFSDM1_Init = 0;
/**
* @brief DFSDM_Channel MSP Initialization
* This function configures the hardware resources used in this example
* @param hdfsdm_channel: DFSDM_Channel handle pointer
* @retval None
*/
void HAL_DFSDM_ChannelMspInit(DFSDM_Channel_HandleTypeDef* hdfsdm_channel)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if(DFSDM1_Init == 0)
  {

    /* Peripheral clock enable */
    __HAL_RCC_DFSDM1_CLK_ENABLE();
  
    __HAL_RCC_GPIOE_CLK_ENABLE();
    /**DFSDM1 GPIO Configuration    
    PE7     ------> DFSDM1_DATIN2
    PE9     ------> DFSDM1_CKOUT 
    */
    GPIO_InitStruct.Pin = DFSDM1_DATIN2_Pin|DFSDM1_CKOUT_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF6_DFSDM1;
    HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);
  }
}

/**
* @brief DFSDM_Channel MSP De-Initialization
* This function freeze the hardware resources used in this example
* @param hdfsdm_channel: DFSDM_Channel handle pointer
* @retval None
*/
void HAL_DFSDM_ChannelMspDeInit(DFSDM_Channel_HandleTypeDef* hdfsdm_channel)
{
  DFSDM1_Init-- ;
  if(DFSDM1_Init == 0)
    {
    /* Peripheral clock disable */
    __HAL_RCC_DFSDM1_CLK_DISABLE();
  
    /**DFSDM1 GPIO Configuration    
    PE7     ------> DFSDM1_DATIN2
    PE9     ------> DFSDM1_CKOUT 
    */
    HAL_GPIO_DeInit(GPIOE, DFSDM1_DATIN2_Pin|DFSDM1_CKOUT_Pin);
  }
}

/**
* @brief UART MSP Initialization
* This function configures the hardware resources used in this example
* @param huart: UART handle pointer
* @retval None
*/
void HAL_UART_MspInit(UART_HandleTypeDef* huart)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if(huart->Instance==USART1)
  {

    /* Peripheral clock enable */
    __HAL_RCC_USART1_CLK_ENABLE();
  
    __HAL_RCC_GPIOB_CLK_ENABLE();
    /**USART1 GPIO Configuration    
    PB6     ------> USART1_TX
    PB7     ------> USART1_RX 
    */
    GPIO_InitStruct.Pin = ST_LINK_UART1_TX_Pin|ST_LINK_UART1_RX_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF7_USART1;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);


  }
}

/**
* @brief UART MSP De-Initialization
* This function freeze the hardware resources used in this example
* @param huart: UART handle pointer
* @retval None
*/
void HAL_UART_MspDeInit(UART_HandleTypeDef* huart)
{
  if(huart->Instance==USART1)
  {
    /* Peripheral clock disable */
    __HAL_RCC_USART1_CLK_DISABLE();
  
    /**USART1 GPIO Configuration    
    PB6     ------> USART1_TX
    PB7     ------> USART1_RX 
    */
    HAL_GPIO_DeInit(GPIOB, ST_LINK_UART1_TX_Pin|ST_LINK_UART1_RX_Pin);
  }
}
