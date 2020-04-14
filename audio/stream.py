from serial import Serial
import numpy as np
from scipy.io.wavfile import write
from tqdm import tqdm

ser = Serial('/dev/tty.usbmodem1413303', 115200)  # open serial port
# ser.open()

data = []

count = 5000
for i in tqdm(range(count)):
  try:
    s = int(ser.readline()[:-1])
    data.append(s)
  except:
    pass
  # print(s)

ser.close()             # close port

# d = []
# for item in data:
#   if item > 127:
#     item = - ( (item^0xff) + 1)
#   else:
#     item = item
#   d.append(item)
data = np.array(data)

f=open('data.txt','w')
for ele in data:
    f.write(str(ele)+'\n')

scaled = np.int16((data-np.mean(data))/np.max(np.abs(data)) * 32767)
write('test.wav', int(5000/2), scaled)

# chunk = 400
# count = 100
# for i in range(count):
#   print("reading "+str(chunk)+" bytes..")
#   s = ser.read(chunk)
#   print("done!")
#   data = data + list(s)
#   print("%d/%d" % (i+1,count))

# import matplotlib.pyplot as plt

# # plt.plot(data)
# # plt.show()

# ser.close()             # close port

# import numpy as np
# from scipy.io.wavfile import write

# # data = np.array(data) - 128
# print(data)
# d = []
# for item in data:
#   if item > 127:
#     item = - ( (item^0xff) + 1)
#   else:
#     item = item
#   d.append(item)
# data = np.array(d)

# print(data)
# scaled = np.int16(data/np.max(np.abs(data)) * 32767)
# write('test.wav', 40323, scaled)