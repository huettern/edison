

#include "util.h"

#include "printf.h"

/**
 * @brief Dumps avlues as hex and asxii
 * @details 
 * 
 * @param data 
 * @param size 
 */
void utilDumpHex(const void* data, size_t size)
{
  char ascii[17];
  size_t i, j;
  ascii[16] = '\0';
  for (i = 0; i < size; ++i) {
    printf("%02X ", ((unsigned char*)data)[i]);
    if (((unsigned char*)data)[i] >= ' ' && ((unsigned char*)data)[i] <= '~') {
      ascii[i % 16] = ((unsigned char*)data)[i];
    } else {
      ascii[i % 16] = '.';
    }
    if ((i+1) % 8 == 0 || i+1 == size) {
      printf(" ");
      if ((i+1) % 16 == 0) {
        printf("|  %s \n", ascii);
      } else if (i+1 == size) {
        ascii[(i+1) % 16] = '\0';
        if ((i+1) % 16 <= 8) {
          printf(" ");
        }
        for (j = (i+1) % 16; j < 16; ++j) {
          printf("   ");
        }
        printf("|  %s \n", ascii);
      }
    }
  }
}

/**
 * @brief Own memcopy implementation
 * @details 
 * 
 * @param dst destination pointer
 * @param src source pointer
 * @param size number of elements to copy
 */
void utilMemcpy(uint8_t *dst, const uint8_t *src, uint16_t size)
{
  while( size-- )
  {
    *dst++ = *src++;
  }
}

typedef struct 
{
  uint8_t used;
  uint16_t start;
} ticToc_t;
static ticToc_t tics[8];

uint8_t utilTic()
{
  static uint8_t init = 0;
  if(!init)
  {
    init = 1;
    for(int i =0; i < sizeof(tics)/sizeof(ticToc_t); i++)
    {
      tics[i].used = 0;
    }
  }

  // get next free tic and start it
  for(int i =0; i < sizeof(tics)/sizeof(ticToc_t); i++)
  {
    if(tics[i].used == 0)
    {
      tics[i].used = 1;
      tics[i].start = __HAL_TIM_GET_COUNTER(&htim1);
      return i;
    }
  }
  return -1;
}

uint32_t utilToc(uint8_t id)
{
  uint32_t stop = __HAL_TIM_GET_COUNTER(&htim1);
  if(id < sizeof(tics)/sizeof(ticToc_t))
  {
    tics[id].used = 0;
    if(stop > tics[id].start) return MAIN_TIM1_TICK_US*(stop-tics[id].start);
    else
    {
      return MAIN_TIM1_TICK_US*(0xffff-tics[id].start+stop);
    }
  }
  return 0;
}
