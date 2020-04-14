import numpy as np
from time import sleep
import struct
import matplotlib.pyplot as plt


# input raw samples from MCU
# in_data = 'out/data_raw.txt'
in_data = 'data/data_raw_5k_24bit_ac_command.txt'
fs = 5000
in_bits = 24

# load file
raw = np.loadtxt(in_data)

# Stats
print("Max=%d Min=%d Mean=%d swing=%d %.1fbits" % \
  (np.max(raw), np.min(raw), np.mean(raw),
   np.max(raw) - np.min(raw), np.log2(np.max(raw) - np.min(raw))))

# generate different bit audio
data_depth = {}
print(raw)
data_depth['16bit'] = 2**(in_bits-16)*(raw / (2**(in_bits-16))).astype('int')
print(data_depth['16bit'])
data_depth['10bit'] = 2**(in_bits-10)*(raw / (2**(in_bits-10))).astype('int')
data_depth['8bit'] = 2**(in_bits-8)*(raw / (2**(in_bits-8))).astype('int')
data_depth['7bit'] = 2**(in_bits-7)*(raw / (2**(in_bits-7))).astype('int')
data_depth['6bit'] = 2**(in_bits-6)*(raw / (2**(in_bits-6))).astype('int')
data_depth['2bit'] = 2**(in_bits-2)*(raw / (2**(in_bits-2))).astype('int')

# normalize and zero mean all 
for key in data_depth:
  data_depth[key] = data_depth[key] - np.mean(data_depth[key])
  data_depth[key] = data_depth[key] / np.max(np.abs(data_depth[key]))

# write audio files
from scipy.io.wavfile import write
for key in data_depth:
  write('out/test'+key+'.wav', fs, data_depth[key])

# plot some
t = np.arange(0, len(raw)/fs, 1/fs)
fig, axs = plt.subplots(1, 1)

axs.step(t, data_depth['16bit'], label='16bit')
axs.step(t, data_depth['8bit'], label='8bit')
axs.step(t, data_depth['7bit'], label='7bit')
axs.step(t, data_depth['6bit'], label='6bit')
axs.step(t, data_depth['2bit'], label='2bit')
# axs.set_xlim(0, 6e-3)
# axs.set_ylim(-1, 1)
axs.set_xlabel('time [s]')
axs.set_ylabel('mic data')
axs.grid(True)
axs.legend()

fig.tight_layout()
plt.show()


