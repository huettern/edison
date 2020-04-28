/*
* @Author: Noah Huetter
* @Date:   2020-04-14 13:49:21
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-04-28 16:42:36
*/

#include "hostinterface.h"

#include "version.h"
#include "microphone.h"

/**
 * Simple host-microcontroller interface. Commands are issued from the host with 
 * a single-byte command followed by a variable number of arguments. If the command
 * byte is recognized and the correct number of arguments are passed, a return
 * command RET_CMD_OK is sent, else the corresponding error code.
 * 
 * To transfer data, the hiSend* routines are used. A transfer to the host contains of
 *  '>'+format+tag+len <'a' from other side> data+crc
 *    '>' indicates mcu to host transfer
 *    format: 0 u8, 1 s8, 2 u16, 3 ..., see hiDataFormat_t
 *    tag 1 byte user tag
 *    len 4 byte little-endian number of data elements (not bytes)
 *    receiver acknowledges transfer by sending byte 'a'
 *    data little-endian data
 *    crc 2 byte crc which is the byte-wise sum of all data elements to a uint16, little-endian
 *  A transfer is acknowledged by the character '^'
 */

/*------------------------------------------------------------------------------
 * Types
 * ---------------------------------------------------------------------------*/
typedef enum 
{
  CMD_VERSION = '0',
  CMD_MIC_SAMPLE_PREPROCESSED_MANUAL = '1',
  CMD_MIC_SAMPLE = 0x0,
  CMD_MIC_SAMPLE_PREPROCESSED = 0x1,
  CMD_MEL_ONE_BATCH = 0x2,
} hostCommandIDs_t;

typedef struct 
{
  hostCommandIDs_t cmd;
  uint8_t argCount; // argument in number of bytes
} hostCommands_t;

typedef enum
{
  RET_CMD_OK = 0,
  RET_CMD_NOT_FOUND = 1,
  RET_CMD_TOO_FEW_ARGUMENTS = 2,
} returnCommands_t;


/*------------------------------------------------------------------------------
 * Settings
 * ---------------------------------------------------------------------------*/
/**
 * time to wait for command arguments to arrive before exiting
 */
#define ARGUMENT_LISTEN_DELAY 100

#define CRC_SEED 0x1234

/*------------------------------------------------------------------------------
 * Private data
 * ---------------------------------------------------------------------------*/

static const hostCommands_t commands [] = {
  {CMD_VERSION, 0},
  {CMD_MIC_SAMPLE, 2},
  {CMD_MIC_SAMPLE_PREPROCESSED, 3},
  {CMD_MIC_SAMPLE_PREPROCESSED_MANUAL, 0},
  {CMD_MEL_ONE_BATCH, 0}
};

static const uint8_t fmtToNbytes[] = {1,1,2,2,4,4};

/*------------------------------------------------------------------------------
 * Prototypes
 * ---------------------------------------------------------------------------*/
static void runCommand(const hostCommands_t* cmd, uint8_t* args);
static void sendDataTransferHeader(hiDataFormat_t fmt, uint8_t tag, uint32_t len);
static uint16_t calcCcrSum(void * data, uint32_t len);
static int8_t checkSendAck(void);
static int8_t waitForByte (uint8_t b, uint32_t timeout);

/*------------------------------------------------------------------------------
 * Publics
 * ---------------------------------------------------------------------------*/
/**
 * @brief Runs a single interation of the host interface
 * @details 
 */
void hifRun(void)
{
  uint8_t rxData[16];
  hostCommandIDs_t cmdId;
  uint8_t ret;
  const hostCommands_t *cmd;

  // wait for 1 data bytes to arrive
  HAL_UART_Receive(&huart1, &cmdId, 1, HAL_MAX_DELAY);

  // get number of arguments to expect
  cmd = NULL;
  for (int i = 0; i < sizeof(commands)/sizeof(hostCommands_t); i++)
    if(commands[i].cmd == cmdId) cmd = &commands[i];
  // command not found, exit
  if(!cmd)
  {
    ret = RET_CMD_NOT_FOUND;
    HAL_UART_Transmit(&huart1, &ret, 1, HAL_MAX_DELAY);
    return;
  }

  // fetch argument bytes
  if(cmd->argCount>0)
  {
    if (HAL_UART_Receive(&huart1, rxData, cmd->argCount, ARGUMENT_LISTEN_DELAY) != HAL_OK)
    {
      // receive error
      ret = RET_CMD_TOO_FEW_ARGUMENTS;
      HAL_UART_Transmit(&huart1, &ret, 1, HAL_MAX_DELAY);
      return;
    }
  }

  // command accepted
  ret = RET_CMD_OK;
  HAL_UART_Transmit(&huart1, &ret, 1, HAL_MAX_DELAY);

  // execute command
  runCommand(cmd, rxData);
}

/**
 * @brief Transfer 8 bit unsigned data to host
 * @details 
 * 
 * @param data data pointer
 * @param len number of elements
 * @param tag tag to send
 */
void hiSendU8(uint8_t * data, uint32_t len, uint8_t tag)
{
  uint16_t crc = calcCcrSum((void*)data, len);
  sendDataTransferHeader(DATA_FORMAT_U8, tag, len);
  if(waitForByte('a', 1000) < 0) return;
  HAL_UART_Transmit(&huart1, data, len, HAL_MAX_DELAY);
  HAL_UART_Transmit(&huart1, ((uint8_t*)&crc), 2, HAL_MAX_DELAY);
  checkSendAck();
}

void hiSendS8(int8_t * data, uint32_t len, uint8_t tag)
{
  uint16_t crc = calcCcrSum((void*)data, len);
  sendDataTransferHeader(DATA_FORMAT_S8, tag, len);
  if(waitForByte('a', 1000) < 0) return;
  HAL_UART_Transmit(&huart1, (uint8_t*)data, len, HAL_MAX_DELAY);
  HAL_UART_Transmit(&huart1, ((uint8_t*)&crc), 2, HAL_MAX_DELAY);
  checkSendAck();
}

void hiSendU16(uint16_t * data, uint32_t len, uint8_t tag)
{
  uint16_t crc = calcCcrSum((void*)data, 2*len);
  sendDataTransferHeader(DATA_FORMAT_U16, tag, len);
  if(waitForByte('a', 1000) < 0) return;
  HAL_UART_Transmit(&huart1, (uint8_t*)data, 2*len, HAL_MAX_DELAY);
  HAL_UART_Transmit(&huart1, ((uint8_t*)&crc), 2, HAL_MAX_DELAY);
  checkSendAck();
}

void hiSendS16(int16_t * data, uint32_t len, uint8_t tag)
{
  uint16_t crc = calcCcrSum((void*)data, 2*len);
  sendDataTransferHeader(DATA_FORMAT_S16, tag, len);
  if(waitForByte('a', 1000) < 0) return;
  HAL_UART_Transmit(&huart1, (uint8_t*)data, 2*len, HAL_MAX_DELAY);
  HAL_UART_Transmit(&huart1, ((uint8_t*)&crc), 2, HAL_MAX_DELAY);
  checkSendAck();
}

void hiSendU32(uint32_t * data, uint32_t len, uint8_t tag)
{
  uint16_t crc = calcCcrSum((void*)data, 4*len);
  sendDataTransferHeader(DATA_FORMAT_U32, tag, len);
  if(waitForByte('a', 1000) < 0) return;
  HAL_UART_Transmit(&huart1, (uint8_t*)data, 4*len, HAL_MAX_DELAY);
  HAL_UART_Transmit(&huart1, ((uint8_t*)&crc), 2, HAL_MAX_DELAY);
  checkSendAck();
}

void hiSendS32(int32_t * data, uint32_t len, uint8_t tag)
{
  uint16_t crc = calcCcrSum((void*)data, 4*len);
  sendDataTransferHeader(DATA_FORMAT_S32, tag, len);
  if(waitForByte('a', 1000) < 0) return;
  HAL_UART_Transmit(&huart1, (uint8_t*)data, 4*len, HAL_MAX_DELAY);
  HAL_UART_Transmit(&huart1, ((uint8_t*)&crc), 2, HAL_MAX_DELAY);
  checkSendAck();
}

/**
 * @brief Receive data from host
 * @details 
 * 
 * @param data pointer to data buffer to store data
 * @param fmt data format to filter. if = 0, is ignored
 * @param maxlen maximum size of received data, in bytes
 * @param tag received tag
 * @return number of elements received and stored. 0 on error
 */
uint32_t hiReceive(void * data, uint32_t maxlen, hiDataFormat_t fmt, uint8_t * tag)
{
  uint32_t nBytes, length;
  uint16_t crc_in, crc_out;
  uint8_t tmp[7]; 
  hiDataFormat_t inFmt;

  // wait for start byte
  waitForByte('<', 0);
  // while(1)
  // {
  //   HAL_UART_Receive(&huart1, &tmp[0], 1, HAL_MAX_DELAY);
  //   if(tmp[0] == '<') break;
  // }

  // receive header
  HAL_UART_Receive(&huart1, &tmp[0], 6, HAL_MAX_DELAY);

  // calculate size
  inFmt = tmp[0];
  *tag = tmp[1];
  length = tmp[2] | (tmp[3]<<8) | (tmp[4]<<16) | (tmp[5]<<24);
  nBytes = fmtToNbytes[inFmt-0x30]*length;

  // assert data type
  if(fmt && (fmt != inFmt)) return 0;

  // send byte to ack transfer
  tmp[0] = 'a';
  HAL_UART_Transmit(&huart1, (uint8_t*)tmp, 1, HAL_MAX_DELAY);

  // read data
  nBytes = (nBytes > maxlen) ? maxlen : nBytes;
  HAL_UART_Receive(&huart1, (uint8_t*)data, nBytes, HAL_MAX_DELAY);

  // receive CRC
  HAL_UART_Receive(&huart1, (uint8_t*)&crc_in, 2, HAL_MAX_DELAY);

  // validate crc
  crc_out = calcCcrSum(data, nBytes);
  // 
  if(crc_in != crc_out)
  {
    // printf("crc in %d crc out %d\n", crc_in, crc_out);
    // printf("CRCFAIL", crc_in, crc_out);
    tmp[0] = 'C';
    HAL_UART_Transmit(&huart1, (uint8_t*)tmp, 1, HAL_MAX_DELAY);
    return 0;
  } 

  // success, return number of elements received
  tmp[0] = '^';
  HAL_UART_Transmit(&huart1, (uint8_t*)tmp, 1, HAL_MAX_DELAY);
  return length;
}

/*------------------------------------------------------------------------------
 * Privates
 * ---------------------------------------------------------------------------*/
/**
 * @brief executes a given command
 * @details 
 * 
 * @param cmd pointer to command
 * @param args pointer to command arguments
 */
static void runCommand(const hostCommands_t* cmd, uint8_t* args)
{
  uint16_t u16;
  uint8_t u8;

  switch(cmd->cmd)
  {
    case CMD_VERSION:
      printf("%s / %s / %s / %s\n", verProgName, verVersion, verBuildDate, verGitSha);
      break;
    case CMD_MIC_SAMPLE:
      // parse arguments
      u16 = args[0]<<8 | args[1];
      // call
      micHostSampleRequest(u16);
      break;
    case CMD_MIC_SAMPLE_PREPROCESSED:
      // parse arguments
      u8 = args[0];
      u16 = args[1]<<8 | args[2];
      // call
      micHostSampleRequestPreprocessed(u16, u8);
      break;
    case CMD_MIC_SAMPLE_PREPROCESSED_MANUAL:
      micHostSampleRequestPreprocessed(10, 16);
      break;
    case CMD_MEL_ONE_BATCH:
      audioMELSingleBatch();
      break;
    default:
      // invalid command
      return;
  }
}

/**
 * @brief Send header of a datat ransmission MCU to host
 * @details 
 * 
 * @param fmt 
 * @param tag 
 * @param len 
 */
static void sendDataTransferHeader(hiDataFormat_t fmt, uint8_t tag, uint32_t len)
{
  uint8_t tmp[7];

  // assemble header
  tmp[0] = '>';
  tmp[1] = fmt;
  tmp[2] = tag;
  tmp[3] = ((len >>  0) & 0x000000ff);
  tmp[4] = ((len >>  8) & 0x000000ff);
  tmp[5] = ((len >> 16) & 0x000000ff);
  tmp[6] = ((len >> 24) & 0x000000ff);
  HAL_UART_Transmit(&huart1, tmp, 7, HAL_MAX_DELAY);
}

/**
 * @brief Calculate simple crc by summing up all elements and storing them in 16bit unsigned
 * @details 
 * 
 * @param data pointer to data
 * @param len data length in bytes
 * 
 * @return 
 */
static uint16_t calcCcrSum(void * data, uint32_t len)
{
  uint16_t crc = CRC_SEED;
  uint32_t i = 0;
  while(len--) crc = crc + (uint16_t)((uint8_t*)data)[i++];
  return crc;
}

/**
 * @brief Checks for ack byte after sending data
 * @details 
 * @return 
 */
static int8_t checkSendAck(void)
{
  uint8_t tmp[1];
  // wait for start byte
  return waitForByte('^', 1000);
}

/**
 * @brief Listen on serial port for byte
 * @details 
 * 
 * @param b listen for this byte
 * @param timeout timeout in ms
 * 
 * @return 0 on success, -1 on fail
 */
static int8_t waitForByte (uint8_t b, uint32_t timeout)
{
  uint8_t tmp;

  do
  {
    HAL_UART_Receive(&huart1, &tmp, 1, HAL_MAX_DELAY);
  } while(tmp != b);

  // if(timeout)
  // {
  //   while(timeout--)
  //   {
  //     if(HAL_UART_Receive(&huart1, &tmp, 1, 0) == HAL_OK)
  //       if(tmp == b) return 0;
  //     HAL_Delay(1);
  //   }
  //   return -1;
  // }
  // else
  // {
  //   while(true)
  //   {
  //     if(HAL_UART_Receive(&huart1, &tmp, 1, 0) == HAL_OK)
  //       if(tmp == b) return 0;
  //     HAL_Delay(1);
  //   }
  // }
}

/*------------------------------------------------------------------------------
 * Callbacks
 * ---------------------------------------------------------------------------*/
