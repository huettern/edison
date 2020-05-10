#include "cyclecounter.h"

/*------------------------------------------------------------------------------
 * Private data
 * ---------------------------------------------------------------------------*/
static volatile unsigned int *DWT_CYCCNT  ;
static volatile unsigned int *DWT_CONTROL ;
static volatile unsigned int *SCB_DEMCR   ;

/**
 * Profiler by https://github.com/Serj-Bashlayev/STM32_Profiler
 */
#define __PROF_STOPED 0xFF
#define CYC_MAX_EVENT_COUNT 20
static uint32_t   time_start; // profiler start time
static const char *prof_name; // profiler name
static uint32_t   time_event[CYC_MAX_EVENT_COUNT]; // events time
static const char *event_name[CYC_MAX_EVENT_COUNT]; // events name
static uint8_t    event_count = __PROF_STOPED; // events counter



/*------------------------------------------------------------------------------
 * Publics
 * ---------------------------------------------------------------------------*/
void ResetTimer(){
  DWT_CYCCNT   = (unsigned int *)0xE0001004; //address of the register
  DWT_CONTROL  = (unsigned int *)0xE0001000; //address of the register
  SCB_DEMCR    = (unsigned int *)0xE000EDFC; //address of the register
  *SCB_DEMCR   = *SCB_DEMCR | 0x01000000;
  *DWT_CYCCNT  = 0; // reset the counter
  *DWT_CONTROL = 0; 
}

void StartTimer(){
  *DWT_CONTROL = *DWT_CONTROL | 1 ; // enable the counter
}

void StopTimer(){
  *DWT_CONTROL = *DWT_CONTROL & 0 ; // disable the counter    
}

unsigned int getCycles(){
  return *DWT_CYCCNT;
}


void cycProfStart(const char *profile_name)
{
  prof_name = profile_name;
  event_count = 0;

  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk; // enable counter
  //DWT->CYCCNT  = time_start = 0;
  time_start = DWT->CYCCNT;
}

void cycProfEvent(const char *event)
{
  if (event_count == __PROF_STOPED)
    return;

  if (event_count < CYC_MAX_EVENT_COUNT)
  {
    time_event[event_count] = DWT->CYCCNT;
    event_name[event_count] = event;
    event_count++;
  }
}

void cycProfStop(void)
{
  int32_t tick_per_1us;
  int32_t time_prev;
  int32_t timestamp;
  int32_t delta_t, sum_t;

  tick_per_1us = SystemCoreClock / 1000000;

  if (event_count == __PROF_STOPED)
  {
    printf("\r\nWarning: PROFILING_STOP WITHOUT START.\r\n");
    return;
  }

  printf("Profiling \"%s\" sequence: \r\n"
               "--Event-----------------------|--timestamp--|----delta_t---\r\n", prof_name);
  time_prev = 0;

  sum_t = 0;
  for (int i = 0; i < event_count; i++)
  {
    timestamp = (time_event[i] - time_start) / tick_per_1us;
    delta_t = timestamp - time_prev;
    time_prev = timestamp;
    printf("%-30s:%9d us | +%9d us\r\n", event_name[i], timestamp, delta_t);
    sum_t += delta_t;
  }
  printf("%-30s:%9d us | +%9d us\r\n", "Total", 0, sum_t);
  printf("\r\n");
  event_count = __PROF_STOPED;
}

