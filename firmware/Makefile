

##############################################################################
# target
TARGET    = edison
DEVICE    = STM32L475xx
CPU       = cortex-m4

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
DEBUG = 1
# optimization
OPT = -O0
# flags for gcc
GCC_FLAGS = -Wall -fdata-sections -ffunction-sections
# generate listings for each source
GEN_LST = 1
#
##############################################################################


##############################################################################
# User code

# Build path
USR_INC = -Isrc -Isrc/io -Isrc/sys -Isrc/hal -Isrc/util

USR_CSRC = 	$(wildcard src/*.c) \
						$(wildcard src/io/*.c) \
						$(wildcard src/sys/*.c) \
						$(wildcard src/hal/*.c) \
						$(wildcard src/util/*.c) \

USR_ASMSRC = src/startup/startup_stm32l475xx.s

USR_DEFINES = 

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
include $(CMSIS)/cmsis.mk
#
##############################################################################

##############################################################################
# HAL
HAL = src/lib/stm32f4/drivers/STM32L4xx_HAL_Driver
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
CSRC = $(CMSIS_CSRC) $(HAL_CSRC) $(USR_CSRC) $(TEST_CSRC)
ASMSRC = $(CMSIS_ASMSRC) $(USR_ASMSRC)
LIBS = $(CMSIS_LIB)

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
	$(TEST_DEFINES)

# AS includes
AS_INCLUDES = 

# C includes
C_INCLUDES =  $(CMSIS_INC) $(HAL_INC) $(USR_INC) $(TEST_INC)

# compile gcc flags
INC = $(C_INCLUDES) $(AS_INCLUDES)
ASFLAGS = $(MCU) $(AS_DEFS) $(AS_INCLUDES) $(OPT) $(GCC_FLAGS)
CFLAGS  = $(MCU) $(C_DEFS) $(C_INCLUDES) $(OPT) $(GCC_FLAGS)

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
LDFLAGS = $(MCU) -T$(LDSCRIPT) $(LIBS) \
	-Wl,-Map=$(BUILD_DIR)/$(TARGET).map,--cref \
	-Wl,--gc-sections \
	-Wl,--print-memory-usage

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

#Pull in dependency info for *existing* .o files
-include $(OBJ:.o=.d)

# Linker
$(BUILD_DIR)/$(TARGET).elf: $(OBJ)
	@echo "LD\t$(notdir $@)"
	$(CC) $(LDFLAGS) $^ -o $@
	$(SZ) $@

# hex file
$(BUILD_DIR)/$(TARGET).hex: $(BUILD_DIR)/$(TARGET).elf
	@echo "HEX\t$(notdir $@)"
	@$(HEX) $< $@

# binary file
$(BUILD_DIR)/$(TARGET).bin: $(BUILD_DIR)/$(TARGET).elf
	@echo "BIN\t$(notdir $@)"
	@$(BIN) $< $@  

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
		@$(ASM) -alhnd $(@:.o=.s) > $(@:.o=.lst)
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

.PHONY: all clean cleaner flash mdebug