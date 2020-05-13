

##############################################################################
# target
TARGET    = edison
DEVICE    = STM32L475xx
CPU       = cortex-m4

# default operation mode
OPMODE 		?= OPMODE_APP

# mfpu
# None for this target
FPU       = -D__FPU_PRESENT -mfpu=fpv4-sp-d16

# mfloat-abi
# Specifies whether to use hardware instructions or software library functions 
# for floating-point operations, and which registers are used to pass 
# floating-point parameters and return values.
# soft
#     Software library functions for floating-point operations and software 
#     floating-point linkage.
# softfp
#     Hardware floating-point instructions and software floating-point linkage.
# hard
#     Hardware floating-point instructions and hardware floating-point linkage. 
FLOAT-ABI = hard

ARM_MATH  = ARM_MATH_CM4
LDSCRIPT  = src/link/STM32L475VGTx_FLASH.ld
#
##############################################################################


##############################################################################
# buil options

# debug build?
DEBUG = 0
# optimization
OPT = -O3
# flags for gcc
GCC_FLAGS = -Wall -fdata-sections -ffunction-sections
# generate listings for each source
GEN_LST = 1
#
##############################################################################


##############################################################################
# User code

# Build path
USR_INC = -Isrc -Isrc/io -Isrc/sys -Isrc/hal -Isrc/util -Isrc/ai -Isrc/audio

USR_CSRC = 	$(wildcard src/*.c) \
						$(wildcard src/io/*.c) \
						$(wildcard src/sys/*.c) \
						$(wildcard src/hal/*.c) \
						$(wildcard src/util/*.c) \
						$(wildcard src/ai/*.c) \
						$(wildcard src/audio/*.c) \

USR_ASMSRC = src/startup/startup_stm32l475xx.s

USR_DEFINES = 

#
##############################################################################


##############################################################################
# CubeAI network
CUBEAI_LOCATION=build/cube/tmp
CUBEAI_NET_NAME=kws
CUBEAI_INC = -Isrc/lib/ST/AI/Inc -Isrc/ai/cube/$(CUBEAI_NET_NAME)
CUBEAI_LIBDIR = -Lsrc/lib/ST/AI/Lib
CUBEAI_LIB = -l:NetworkRuntime500_CM4_GCC.a
CUBEAI_SRC = 	$(wildcard src/ai/cube/$(CUBEAI_NET_NAME)/*.c)
# src/ai/cube/$(CUBEAI_NET_NAME)/$(CUBEAI_NET_NAME).c \
# 							src/ai/cube/$(CUBEAI_NET_NAME)/$(CUBEAI_NET_NAME)_data.c \
# 							src/ai/cube/$(CUBEAI_NET_NAME)/app_x-cube-ai.c
#
##############################################################################

##############################################################################
# NNoM
NNOM_REPO=../audio/implement/nnom/
NNOM_WEIGHTS=../audio/.cache/allinone/weights.h
NNOM_NAME=kws_nnom
NNOM=src/ai/nnom/
NNOM_CSRC=$(wildcard $(NNOM)/src/backends/*.c) $(wildcard $(NNOM)/src/layers/*.c) $(wildcard $(NNOM)/src/core/*.c)
NNOM_INC=-I$(NNOM)/inc -I$(NNOM)/port -I$(NNOM)
#
##############################################################################


##############################################################################
# paths

# Build path
BUILD_DIR = build

#
##############################################################################

##############################################################################
# CMSIS
CMSIS = src/lib/CMSIS
CMSIS_OPT = -O3
include $(CMSIS)/cmsis.mk
#
##############################################################################

##############################################################################
# HAL
HAL = src/lib/stm32f4/drivers/STM32L4xx_HAL_Driver
HAL_OPT = -O3
include $(HAL)/hal.mk
#
##############################################################################

##############################################################################
# Misc
GIT_VERSION := "$(shell git describe --abbrev=8 --dirty --always --tags)"
BUILD_DATE  := "$(shell date +%Y-%m-%d\ %H:%M:%S\ %Z)"
#
##############################################################################

##############################################################################
# sources and objects
CSRC = $(CMSIS_CSRC) $(HAL_CSRC) $(USR_CSRC) $(TEST_CSRC) $(CUBEAI_SRC) $(NNOM_CSRC)
ASMSRC = $(CMSIS_ASMSRC) $(USR_ASMSRC)
LIB_DIRS = $(CMSIS_LIBDIR) $(CUBEAI_LIBDIR)
LIBS = $(CMSIS_LIB) $(CUBEAI_LIB)

COBJ   = $(patsubst src/%,$(BUILD_DIR)/obj/%,$(CSRC:.c=.o))
ASMOBJ = $(patsubst src/%,$(BUILD_DIR)/obj/%,$(filter %.o,$(ASMSRC:.s=.o) $(ASMSRC:.S=.o)))

OBJ    = $(COBJ) $(ASMOBJ)
#
##############################################################################

##############################################################################
# binaries
PREFIX = arm-none-eabi-
ifdef GCC_PATH
	CC = $(GCC_PATH)/$(PREFIX)gcc
	AS = $(GCC_PATH)/$(PREFIX)gcc -x assembler-with-cpp
	ASM = $(GCC_PATH)/$(PREFIX)as
	CP = $(GCC_PATH)/$(PREFIX)objcopy
	SZ = $(GCC_PATH)/$(PREFIX)size
else
	CC = $(PREFIX)gcc
	AS = $(PREFIX)gcc -x assembler-with-cpp
	ASM = $(PREFIX)as
	CP = $(PREFIX)objcopy
	SZ = $(PREFIX)size
endif
HEX = $(CP) -O ihex
BIN = $(CP) -O binary -S
 
#
##############################################################################

##############################################################################
# CFLAGS

# mcu
MCU = -mcpu=$(CPU) -mthumb $(FPU) -mfloat-abi=$(FLOAT-ABI)

# AS defines
AS_DEFS = 

# C defines
C_DEFS =  -DGIT_SHA=\"$(GIT_VERSION)\" -DBUILD_DATE=\"$(BUILD_DATE)\"\
	-DUSE_HAL_DRIVER \
	-D$(DEVICE) \
	-D$(ARM_MATH) \
	$(USR_DEFINES) \
	$(TEST_DEFINES) \
	-D$(OPMODE)

# AS includes
AS_INCLUDES = 

# C includes
C_INCLUDES =  $(CMSIS_INC) $(HAL_INC) $(USR_INC) $(TEST_INC) $(CUBEAI_INC) $(NNOM_INC)

# compile gcc flags
INC = $(C_INCLUDES) $(AS_INCLUDES)
ASFLAGS = $(MCU) $(AS_DEFS) $(AS_INCLUDES) $(OPT) $(GCC_FLAGS)
CFLAGS  = $(MCU) $(C_DEFS) $(C_INCLUDES) $(OPT) $(GCC_FLAGS)

CMSISCFLAGS  = $(MCU) $(C_DEFS) $(CMSIS_INC) $(CMSIS_OPT) $(GCC_FLAGS)
HALCFLAGS  = $(MCU) $(C_DEFS) $(C_INCLUDES) $(HAL_OPT) $(GCC_FLAGS)

# debug?
ifeq ($(DEBUG), 1)
	CFLAGS += -g -gdwarf-2
endif

# Generate dependency information?
CFLAGS += 

#
##############################################################################

##############################################################################
# LDFLAGS

# libraries
LIBS += -lc -lm -lnosys 
LDFLAGS = $(MCU) -specs=nano.specs -T$(LDSCRIPT) $(LIB_DIRS) $(LIBS) \
	-Wl,-Map=$(BUILD_DIR)/$(TARGET).map,--cref \
	-Wl,--gc-sections \
	-Wl,--print-memory-usage
# perhaps we want to wrap malloc/free with __wrap_malloc/__wrap_free?
LDFLAGS += -Wl,--wrap=malloc -Wl,--wrap=free  
#
##############################################################################
# Targets

# default
all: $(BUILD_DIR)/$(TARGET).elf $(BUILD_DIR)/$(TARGET).hex $(BUILD_DIR)/$(TARGET).bin

# clean
clean:
	-rm -fR $(BUILD_DIR)/obj $(COBJ) $(ASMOBJ)
	-rm -fR $(BUILD_DIR)/$(TARGET).bin $(BUILD_DIR)/$(TARGET).elf $(BUILD_DIR)/$(TARGET).hex $(BUILD_DIR)/$(TARGET).map

cleaner:
	-find . -type f -name '*.o' -delete

# Flash
flash: $(BUILD_DIR)/$(TARGET).bin
	st-flash write $(BUILD_DIR)/$(TARGET).bin 0x8000000

mdebug:
# 	echo $(ASMOBJ)
	echo $(OBJ)
# 	echo $(HAL_INC)
# 	echo $(HAL_CSRC)

import-cubeai-net:
	./scripts/import-cubeai-net.sh $(CUBEAI_LOCATION) $(CUBEAI_NET_NAME) src/lib/ST src/ai/cube/$(CUBEAI_NET_NAME)

import-nnom-net:
	./scripts/import-nnom-net.sh $(NNOM_REPO) $(NNOM_WEIGHTS) $(NNOM_NAME) $(NNOM)

#Pull in dependency info for *existing* .o files
-include $(OBJ:.o=.d)

# Linker
$(BUILD_DIR)/$(TARGET).elf: $(OBJ)
	@echo "LD\t$(notdir $@)"
	$(CC) $^ $(LDFLAGS) -o $@
	$(SZ) $@

# hex file
$(BUILD_DIR)/$(TARGET).hex: $(BUILD_DIR)/$(TARGET).elf
	@echo "HEX\t$(notdir $@)"
	@$(HEX) $< $@

# binary file
$(BUILD_DIR)/$(TARGET).bin: $(BUILD_DIR)/$(TARGET).elf
	@echo "BIN\t$(notdir $@)"
	@$(BIN) $< $@  

# c compiler for CMSIS directory
$(BUILD_DIR)/obj/lib/CMSIS/%.o: src/lib/CMSIS/%.c
	@echo "CC (CMSIS)\t$(notdir $<)"
	@mkdir -p $(dir $@)
	@$(CC) $(CMSISCFLAGS) -c $< -o $@

# c compiler for HAL directory
$(BUILD_DIR)/obj/lib/stm32f4/%.o: src/lib/stm32f4/%.c
	@echo "CC (HAL)\t$(notdir $<)"
	@mkdir -p $(dir $@)
	@$(CC) $(HALCFLAGS) -c $< -o $@

# c compiler
$(BUILD_DIR)/obj/%.o: src/%.c
	@echo "CC\t$(notdir $<)"
	@mkdir -p $(dir $@)
	@$(CC) $(CFLAGS) -c $< -o $@
	@# deps
	@$(CC) $(CFLAGS) $(INC) -MM src/$*.c > $(BUILD_DIR)/obj/$*.d
	@cp -f $(BUILD_DIR)/obj/$*.d $(BUILD_DIR)/obj/$*.d.tmp
	@sed -e 's|.*:|$(BUILD_DIR)/obj/$*.o:|' < $(BUILD_DIR)/obj/$*.d.tmp > $(BUILD_DIR)/obj/$*.d
	@sed -e 's/.*://' -e 's/\\$$//' < $(BUILD_DIR)/obj/$*.d.tmp | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >> $(BUILD_DIR)/obj/$*.d
	@rm -f $(BUILD_DIR)/obj/$*.d.tmp
ifdef GEN_LST
		@$(CC) -S -fverbose-asm $(CFLAGS) -w -o $(@:.o=.s) $<
		@$(ASM) -alhnd $(@:.o=.s) -o $(BUILD_DIR)/obj/a.out > $(@:.o=.lst)
endif

# assembler
$(BUILD_DIR)/obj/%.o: src/%.s
	@echo "AS\t$(notdir $<)"
	@mkdir -p $(dir $@)
	@$(AS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/obj/%.o: src/%.S
	@echo "AS\t$(notdir $<)"
	@mkdir -p $(dir $@)
	@$(AS) $(CFLAGS) -c $< -o $@

.PHONY: all clean cleaner flash mdebug import-cubeai-net import-nnom-net