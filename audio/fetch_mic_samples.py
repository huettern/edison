from serial import Serial, SerialException
import numpy as np
from scipy.io.wavfile import write
from tqdm import tqdm
from time import sleep
import struct

# Set tu False to get raw sampels, to true to get preprocessed samples
preprocessed = True

try:
  ser = Serial('/dev/tty.usbmodem1413303', 115200)  # open serial port
except SerialException:
  print("coult not open serial port")
  exit()

data = []

count = 25000
fs = 5000

# request count samples
ser.flush()
if preprocessed:
  ser.write(b'\1') # command byte
else:
  ser.write(b'\0') # command byte

ser.write(count.to_bytes(2,'big')) # number of samples

# poll for command status
ret = ser.read(1)
if ret != b'\0':
  print('Command not accepted, exiting')
  exit()
print('Command accepted')

# wait for data
print('Acquiring.')
while (ser.in_waiting < 1):
  sleep(0.01)

# Read data to keep buffer from overflowing

if preprocessed:
  nBytesPerSampe = 1
else:
  nBytesPerSampe = 4

buf = b''
readCtr = 0
inWaiting = 0
while(1):
  sleep(0.01)
  inWaiting = ser.in_waiting
  nRead = min(inWaiting, count*nBytesPerSampe-readCtr)
  buf += ser.read(nRead)
  readCtr += nRead
  print("read %d, total bytes read: %d" % (nRead,readCtr))
  if readCtr >= count*nBytesPerSampe:
    break
# print(ser.readline())
ser.close()             # close port

# convert byte string of integers to int array
d = []
for i in range(int(readCtr/nBytesPerSampe)):
  inp = buf[nBytesPerSampe*i:nBytesPerSampe*i+nBytesPerSampe]
  if preprocessed:
    d.append(struct.unpack('>b', inp)[0]) # b: signed char
  else:
    d.append(struct.unpack('>i', inp)[0]) # i: int
data = np.array(d)

print("Acquired min: %d max: %d" % (np.min(data), np.max(data)))

# use only some MSB bits
# msbBits = 16
# data = data / 2**(32-msbBits)

# dump raw values to file
f=open('data_raw.txt','w')
for ele in data:
    # f.write(("%08x\n"%(ele)))
    f.write(("%d\n"%(ele)))

# strip from mean
data = data - np.mean(data)

# normalize to [-1,1]
data = data / np.max(np.abs(data))

# dump raw values to file
f=open('data_ac.txt','w')
for ele in data:
    f.write(str(ele)+'\n')

# scaled = np.int16((data-np.mean(data))/np.max(np.abs(data)) * 32767)
write('test.wav', fs, data)
