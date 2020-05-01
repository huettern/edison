HAL_INC = \
	-I$(HAL)/Inc \
	-I$(HAL)/Inc/Legacy 

HAL_CSRC = \
	$(HAL)/Src/stm32l4xx_hal.c \
	$(HAL)/Src/stm32l4xx_hal_dfsdm.c \
	$(HAL)/Src/stm32l4xx_hal_uart_ex.c \
	$(HAL)/Src/stm32l4xx_hal_uart.c \
	$(HAL)/Src/stm32l4xx_hal_gpio.c \
	$(HAL)/Src/stm32l4xx_ll_gpio.c \
	$(HAL)/Src/stm32l4xx_ll_exti.c \
	$(HAL)/Src/stm32l4xx_hal_exti.c \
	$(HAL)/Src/stm32l4xx_ll_dma2d.c \
	$(HAL)/Src/stm32l4xx_hal_dma_ex.c \
	$(HAL)/Src/stm32l4xx_ll_dma.c \
	$(HAL)/Src/stm32l4xx_hal_dma2d.c \
	$(HAL)/Src/stm32l4xx_hal_dma.c \
	$(HAL)/Src/stm32l4xx_hal_rcc.c \
	$(HAL)/Src/stm32l4xx_hal_rcc_ex.c \
	$(HAL)/Src/stm32l4xx_ll_rcc.c \
	$(HAL)/Src/stm32l4xx_hal_flash.c \
	$(HAL)/Src/stm32l4xx_hal_flash_ramfunc.c \
	$(HAL)/Src/stm32l4xx_hal_flash_ex.c \
	$(HAL)/Src/stm32l4xx_hal_pwr_ex.c \
	$(HAL)/Src/stm32l4xx_ll_pwr.c \
	$(HAL)/Src/stm32l4xx_hal_pwr.c \
	$(HAL)/Src/stm32l4xx_hal_cortex.c \
	$(HAL)/Src/stm32l4xx_hal_crc_ex.c \
	$(HAL)/Src/stm32l4xx_hal_crc.c \
	$(HAL)/Src/stm32l4xx_ll_crc.c \
	$(HAL)/Src/stm32l4xx_ll_tim.c \
	$(HAL)/Src/stm32l4xx_hal_tim_ex.c \
	$(HAL)/Src/stm32l4xx_hal_tim.c \
# 	$(HAL)/Src/stm32l4xx_ll_rtc.c \
# 	$(HAL)/Src/stm32l4xx_hal_pka.c \
# 	$(HAL)/Src/stm32l4xx_hal_smartcard.c \
# 	$(HAL)/Src/stm32l4xx_ll_tim.c \
# 	$(HAL)/Src/stm32l4xx_hal_pssi.c \
# 	$(HAL)/Src/stm32l4xx_ll_spi.c \
# 	$(HAL)/Src/stm32l4xx_ll_usb.c \
# 	$(HAL)/Src/stm32l4xx_hal_lptim.c \
# 	$(HAL)/Src/stm32l4xx_ll_opamp.c \
# 	$(HAL)/Src/stm32l4xx_hal_nand.c \
# 	$(HAL)/Src/stm32l4xx_ll_fmc.c \
# 	$(HAL)/Src/stm32l4xx_hal_opamp_ex.c \
# 	$(HAL)/Src/stm32l4xx_hal_i2c_ex.c \
# 	$(HAL)/Src/stm32l4xx_hal_dfsdm_ex.c \
# 	$(HAL)/Src/stm32l4xx_hal_pcd_ex.c \
# 	$(HAL)/Src/stm32l4xx_hal_usart_ex.c \
# 	$(HAL)/Src/stm32l4xx_hal_tim_ex.c \
# 	$(HAL)/Src/stm32l4xx_hal_smbus.c \
# 	$(HAL)/Src/stm32l4xx_hal_i2c.c \
# 	$(HAL)/Src/stm32l4xx_hal_wwdg.c \
# 	$(HAL)/Src/stm32l4xx_hal_nor.c \
# 	$(HAL)/Src/stm32l4xx_ll_crs.c \
# 	$(HAL)/Src/stm32l4xx_hal_swpmi.c \
# 	$(HAL)/Src/stm32l4xx_hal_mmc_ex.c \
# 	$(HAL)/Src/stm32l4xx_hal_pcd.c \
# 	$(HAL)/Src/stm32l4xx_hal_iwdg.c \
# 	$(HAL)/Src/stm32l4xx_hal_dac_ex.c \
# 	$(HAL)/Src/stm32l4xx_hal_crc_ex.c \
# 	$(HAL)/Src/stm32l4xx_hal_sd_ex.c \
# 	$(HAL)/Src/stm32l4xx_ll_i2c.c \
# 	$(HAL)/Src/stm32l4xx_hal_ltdc_ex.c \
# 	$(HAL)/Src/stm32l4xx_ll_utils.c \
# 	$(HAL)/Src/stm32l4xx_hal_rng_ex.c \
# 	$(HAL)/Src/stm32l4xx_ll_usart.c \
# 	$(HAL)/Src/stm32l4xx_hal_smartcard_ex.c \
# 	$(HAL)/Src/stm32l4xx_hal_rtc.c \
# 	$(HAL)/Src/stm32l4xx_ll_pka.c \
# 	$(HAL)/Src/stm32l4xx_hal_sd.c \
# 	$(HAL)/Src/stm32l4xx_hal_spi.c \
# 	$(HAL)/Src/stm32l4xx_hal_spi_ex.c \
# 	$(HAL)/Src/stm32l4xx_ll_sdmmc.c \
# 	$(HAL)/Src/stm32l4xx_hal_tim.c \
# 	$(HAL)/Src/stm32l4xx_hal_firewall.c \
# 	$(HAL)/Src/stm32l4xx_hal_crc.c \
# 	$(HAL)/Src/stm32l4xx_hal_hash_ex.c \
# 	$(HAL)/Src/stm32l4xx_hal_dsi.c \
# 	$(HAL)/Src/stm32l4xx_hal_opamp.c \
# 	$(HAL)/Src/stm32l4xx_hal_rtc_ex.c \
# 	$(HAL)/Src/stm32l4xx_ll_lptim.c \
# 	$(HAL)/Src/stm32l4xx_hal_lcd.c \
# 	$(HAL)/Src/stm32l4xx_hal_mmc.c \
# 	$(HAL)/Src/stm32l4xx_hal_can.c \
# 	$(HAL)/Src/stm32l4xx_ll_adc.c \
# 	$(HAL)/Src/stm32l4xx_hal_rng.c \
# 	$(HAL)/Src/stm32l4xx_hal_hcd.c \
# 	$(HAL)/Src/stm32l4xx_hal_sai_ex.c \
# 	$(HAL)/Src/stm32l4xx_hal_sram.c \
# 	$(HAL)/Src/stm32l4xx_ll_dac.c \
# 	$(HAL)/Src/stm32l4xx_ll_comp.c \
# 	$(HAL)/Src/stm32l4xx_hal_ospi.c \
# 	$(HAL)/Src/stm32l4xx_hal_tsc.c \
# 	$(HAL)/Src/stm32l4xx_hal_comp.c \
# 	$(HAL)/Src/stm32l4xx_ll_rng.c \
# 	$(HAL)/Src/stm32l4xx_hal_qspi.c \
# 	$(HAL)/Src/stm32l4xx_hal_dcmi.c \
# 	$(HAL)/Src/stm32l4xx_hal_cryp_ex.c \
# 	$(HAL)/Src/stm32l4xx_hal_hash.c \
# 	$(HAL)/Src/stm32l4xx_hal_adc.c \
# 	$(HAL)/Src/stm32l4xx_hal_adc_ex.c \
# 	$(HAL)/Src/stm32l4xx_hal_usart.c \
# 	$(HAL)/Src/stm32l4xx_hal_sai.c \
# 	$(HAL)/Src/stm32l4xx_ll_lpuart.c \
# 	$(HAL)/Src/stm32l4xx_hal_dac.c \
# 	$(HAL)/Src/stm32l4xx_hal_cryp.c \
# 	$(HAL)/Src/stm32l4xx_ll_crc.c \
# 	$(HAL)/Src/stm32l4xx_hal_ltdc.c \
# 	$(HAL)/Src/stm32l4xx_hal_irda.c \
# 	$(HAL)/Src/stm32l4xx_hal_gfxmmu.c \
# 	$(HAL)/Src/stm32l4xx_ll_swpmi.c \
