
# Cache directory
cache_dir = 'cache/'

# Voice data dir
speech_data_dir = 'cache/acquire/noah'

# serial port
mcu_serial_port = '/dev/tty.usbmodem141342103'

# audio and MFCC settings
sample_len_seconds = 2.0
fs = 16000
mel_mtx_scale = 128
lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 32
mel_nbins = num_mel_bins
mel_lower_hz = lower_edge_hertz
mel_upper_hz = upper_edge_hertz
frame_length = 1024
sample_size = frame_length
num_spectrogram_bins = sample_size//2+1
sample_rate = fs
first_mfcc = 0
num_mfcc = 13
nSamples = int(sample_len_seconds*fs)
sample_len = int(sample_len_seconds*fs)
frame_step = frame_length
frame_len = frame_length
frame_count = 0 # 0 for auto
fft_len = frame_length
n_frames = 1 + (nSamples - frame_length) // frame_step
mel_twiddle_scale = 128

# mel freq. constants -> https://en.wikipedia.org/wiki/Mel_scale
MEL_HIGH_FREQUENCY_Q = 1127.0
MEL_BREAK_FREQUENCY_HERTZ = 700.0

# net

# NNoM
nnom_net_input_scale = 1.0 / 1
nnom_net_input_clip_min = -128
nnom_net_input_clip_max = 127

# Cube
net_input_scale = 1.0
net_input_clip_min = -2**15
net_input_clip_max = 2**15-1