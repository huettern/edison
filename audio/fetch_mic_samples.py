from serial import Serial
import numpy as np
from scipy.io.wavfile import write
from tqdm import tqdm
from time import sleep
import struct

ser = Serial('/dev/tty.usbmodem1413303', 115200)  # open serial port
# ser.open()

data = []

count = 20000
fs = 5000

# request count samples
ser.flush()
# ser.read(-1)
ser.write(b'\0') # command byte
ser.write(count.to_bytes(2,'big')) # number of samples
ser.write(b'\0') # optional argument

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
buf = b''
readCtr = 0
inWaiting = 0
while(1):
  sleep(0.01)
  inWaiting = ser.in_waiting
  nRead = min(inWaiting, count*4-readCtr)
  buf += ser.read(nRead)
  readCtr += nRead
  print("read %d, total bytes read: %d" % (nRead,readCtr))
  if readCtr >= count*4:
    break
print(ser.readline())
ser.close()             # close port

# convert byte string of integers to int array
d = []
for i in range(int(readCtr/4)):
  inp = buf[4*i:4*i+4]
  d.append(struct.unpack('>i', inp)[0])
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

exit()


######################################################################
# ASCII continuous stream
######################################################################

# ser = Serial('/dev/tty.usbmodem1413303', 115200)  # open serial port
# # ser.open()

# data = []

# count = 5000
# for i in tqdm(range(count)):
#   try:
#     s = int(ser.readline()[:-1])
#     data.append(s)
#   except:
#     pass
#   # print(s)

# ser.close()             # close port

# data = np.array(data)

# f=open('data.txt','w')
# for ele in data:
#     f.write(str(ele)+'\n')

# scaled = np.int16((data-np.mean(data))/np.max(np.abs(data)) * 32767)
# write('test.wav', int(5000/2), scaled)
