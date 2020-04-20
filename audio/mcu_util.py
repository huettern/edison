from serial import Serial, SerialException
from time import sleep
import struct
import numpy as np

try:
  ser = Serial('/dev/tty.usbmodem1413303', 115200)  # open serial port
except SerialException:
  print("coult not open serial port")
  exit()

CRC_SEED = 0x1234

valid_fmt_bytes = [0,1,2,3,4,5]
fmt_byte_to_nbytes = [1,1,2,2,4,4]
fmt_byte_to_upack_string = ['<B', '<b', '<H', '<h', '<I', '<i']
fmt_byte_to_dtype = ['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32']


def receiveData():
  """
    Listens for incomming data streams
  """
  ret_data = None
  ret_tag = None

  # wait for start of frame
  while(1):
    sleep(0.01)
    c = ser.read(1)
    if c == b'>':
      break

  # print("> received") # DBG

  # Wait for header arrived
  while(1):
    sleep(0.01)
    if ser.in_waiting >= 6:
      buf = ser.read(6)
      break
  fmt, tag, length = struct.unpack('<BBI', buf)

  # print("fmt = %d tag = 0x%02x length = %d" % (fmt, tag, length)) # DBG

  # convert fmt byte to usable index
  fmt = fmt & (~0x30)

  # check correct format byte
  if fmt not in valid_fmt_bytes:
    print('Invalid format byte received. Aborting')
    return ret_data, ret_tag

  # receive data
  toRead = fmt_byte_to_nbytes[fmt]*length
  buf = b''
  # print('start receiving %d bytes' % (toRead)) # DBG
  while(toRead):
    sleep(0.01)
    inWaiting = ser.in_waiting
    nRead = min(inWaiting, toRead)
    buf = ser.read(nRead)
    toRead = toRead - nRead

  # print('read %d bytes' % len(buf)) # DBG
  
  # receive CRC
  while(1):
    sleep(0.01)
    if ser.in_waiting >= 2:
      crcbuf = ser.read(2)
      break
  crc_in = struct.unpack('<H', crcbuf)[0]
  
  # unpack
  data = []
  for i in range(length):
    data.append(struct.unpack(fmt_byte_to_upack_string[fmt], 
      buf[i*fmt_byte_to_nbytes[fmt]:(i+1)*fmt_byte_to_nbytes[fmt]])[0])

  # calculate crc
  crc_out = np.dtype('uint16').type(CRC_SEED)
  for i in range(len(buf)):
    crc_out = np.dtype('uint16').type(crc_out+struct.unpack('<B',buf[i:i+1])[0])

  if crc_out != crc_in:
    print('CRC mismatch! Aborting')
    return ret_data, ret_tag

  # print('Unpacked data:') # DBG
  # print(data) # DBG
  ret_data = np.asarray(data, dtype=fmt_byte_to_dtype[fmt])
  ret_tag = tag

  # print('crc_in = %d crc_out = %d' % (crc_in, crc_out)) # DBG
  
  # Ack the transfer
  ser.write(b'^')
  ser.flush()
  return ret_data, ret_tag


def sendData(data, tag):
  """
    Send data, length and type is infered from data
  """
  if data.dtype in fmt_byte_to_dtype:
    fmt_byte = fmt_byte_to_dtype.index(data.dtype) + 0x30
  else:
    print('Unsupported datatype, aborting')

  length = len(data)
  ser.flush()

  # assemble and send header
  hdr = struct.pack('<cBBL', b'<', fmt_byte, tag, length)
  ser.write(hdr)
  ser.flush()

  # send data
  crc = np.dtype('uint16').type(CRC_SEED)
  for element in data:
    # online crc calculation
    for i in range(fmt_byte_to_nbytes[fmt_byte-0x30]):
      data_byte = (element // (2**(8*i))) & 0xff
      crc = np.dtype('uint16').type(crc+data_byte)
    # actual send
    ser.write(struct.pack(fmt_byte_to_upack_string[fmt_byte-0x30], element))

  # send crc
  ser.write(struct.pack('<H', crc))

  # Read ack
  timeout = 500
  while timeout:
    sleep(0.01)
    timeout -= 1
    if(ser.in_waiting):
      ret = ser.read(1)
      if ret == b'^':
        print('Transfer acknowledged')
        return

  print('Error: Transfer not acknowledged!')
  print(ser.readline())


def pingtest():

  sleep(1)
  print('--- Transferring uint8 -----------------------')
  dat = np.array([-2,-1,0,1,2,3], dtype='uint8')
  sendData(dat, 1)
  print(ser.readline())
  print(ser.readline())

  sleep(1)
  print('--- Transferring int8 -----------------------')
  dat = np.array([-2,-1,0,1,2,3], dtype='int8')
  sendData(dat, 2)
  print(ser.readline())
  print(ser.readline())
  
  sleep(1)
  print('--- Transferring uint16 -----------------------')
  dat = np.array([-2,-1,0,1,2,3], dtype='uint16')
  sendData(dat, 3)
  print(ser.readline())
  print(ser.readline())
  
  sleep(1)
  print('--- Transferring int16 -----------------------')
  dat = np.array([-2,-1,0,1,2,3], dtype='int16')
  sendData(dat, 4)
  print(ser.readline())
  print(ser.readline())
  
  sleep(1)
  print('--- Transferring uint32 -----------------------')
  dat = np.array([-2,-1,0,1,2,3], dtype='uint32')
  sendData(dat, 5)
  print(ser.readline())
  print(ser.readline())
  
  sleep(1)
  print('--- Transferring int32 -----------------------')
  dat = np.array([-2,-1,0,1,2,3], dtype='int32')
  sendData(dat, 6)
  print(ser.readline())
  print(ser.readline())

def pongtest():
  data, tag = receiveData()
  print('Received %s type with tag 0x%x: %s' % (data.dtype, tag, data))
  data, tag = receiveData()
  print('Received %s type with tag 0x%x: %s' % (data.dtype, tag, data))
  data, tag = receiveData()
  print('Received %s type with tag 0x%x: %s' % (data.dtype, tag, data))
  data, tag = receiveData()
  print('Received %s type with tag 0x%x: %s' % (data.dtype, tag, data))
  data, tag = receiveData()
  print('Received %s type with tag 0x%x: %s' % (data.dtype, tag, data))
  data, tag = receiveData()
  print('Received %s type with tag 0x%x: %s' % (data.dtype, tag, data))

def pingpongtest():

  # while(1):
  #   print('------------------------------------')
  #   data, tag = receiveData()
  #   print(tag)
  #   print(data)
  #   print(type(data))
  #   print(data.dtype)

  sleep(1)
  print('--- Transferring uint8 -----------------------')
  dat = np.array([-2,-1,0,1,2,3], dtype='uint8')
  sendData(dat, 1)
  print(ser.readline())
  print(ser.readline())
  data, tag = receiveData()
  print(type(data))
  print(data.dtype)

  sleep(1)
  print('--- Transferring int8 -----------------------')
  dat = np.array([-2,-1,0,1,2,3], dtype='int8')
  sendData(dat, 2)
  print(ser.readline())
  print(ser.readline())
  data, tag = receiveData()
  print(type(data))
  print(data.dtype)
  
  sleep(1)
  print('--- Transferring uint16 -----------------------')
  dat = np.array([-2,-1,0,1,2,3], dtype='uint16')
  sendData(dat, 3)
  print(ser.readline())
  print(ser.readline())
  data, tag = receiveData()
  print(type(data))
  print(data.dtype)
  
  sleep(1)
  print('--- Transferring int16 -----------------------')
  dat = np.array([-2,-1,0,1,2,3], dtype='int16')
  sendData(dat, 4)
  print(ser.readline())
  print(ser.readline())
  data, tag = receiveData()
  print(type(data))
  print(data.dtype)
  
  sleep(1)
  print('--- Transferring uint32 -----------------------')
  dat = np.array([-2,-1,0,1,2,3], dtype='uint32')
  sendData(dat, 5)
  print(ser.readline())
  print(ser.readline())
  data, tag = receiveData()
  print(type(data))
  print(data.dtype)
  
  sleep(1)
  print('--- Transferring int32 -----------------------')
  dat = np.array([-2,-1,0,1,2,3], dtype='int32')
  sendData(dat, 6)
  print(ser.readline())
  print(ser.readline())
  data, tag = receiveData()
  print(type(data))
  print(data.dtype)

if __name__ == '__main__':
  import sys, os
  try:
    pingpongtest()
    # pingtest()
    # pongtest()
  except KeyboardInterrupt:
    print('Interrupted')
    ser.close()
    try:
      sys.exit(0)
    except SystemExit:
      os._exit(0)


