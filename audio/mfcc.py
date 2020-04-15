import numpy as np
from time import sleep
import struct
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.io import wavfile

# Input wav file to use
in_wav = 'data/heysnips_true_5k_16b.wav'

# mel freq. constants -> https://en.wikipedia.org/wiki/Mel_scale
MEL_HIGH_FREQUENCY_Q = 1127.0
MEL_BREAK_FREQUENCY_HERTZ = 700.0

def frames(data, frame_length=3, frame_step=1):
  """
  Split a data vector into (possibly overlapipng) frames

    frame_length: length of each frame
    frame_step: how many sample to advance the frame each step
  """
  n_frames = 1 + (data.shape[0] - frame_length) // frame_step
  out = np.zeros((n_frames,frame_length))
  for i in range(n_frames):
    out[i] = data[i*frame_step:i*frame_step+frame_length]
  return out


def hertz_to_mel(frequencies_hertz):
  """
  Converts frequencies in `frequencies_hertz` in Hertz to the mel scale.
  """
  return MEL_HIGH_FREQUENCY_Q * np.log(1.0 + (frequencies_hertz / MEL_BREAK_FREQUENCY_HERTZ))

def gen_mel_weight_matrix(num_mel_bins=20, num_spectrogram_bins=129, sample_rate=8000, \
    lower_edge_hertz=125.0, upper_edge_hertz=3800.0):
  """
  Generate mel weight matric from linear frequency spectrum, inspired by 
    https://www.tensorflow.org/api_docs/python/tf/signal/linear_to_mel_weight_matrix

  """
  nyquist_hertz = sample_rate / 2.0
  # excludes DC spectrogram bin
  n_bands_to_zero = 1
  linear_frequencies = np.linspace(0, nyquist_hertz, num_spectrogram_bins)[n_bands_to_zero:]
  # convert linear frequency vector to mel scale
  spectrogram_bins_mel = np.expand_dims( hertz_to_mel(linear_frequencies), 1)
  
  # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
  # center of each band is the lower and upper edge of the adjacent bands.
  # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
  # num_mel_bins + 2 pieces.
  band_edges_mel = frames(
    np.linspace(hertz_to_mel(lower_edge_hertz), hertz_to_mel(upper_edge_hertz), num_mel_bins + 2),
    frame_length=3, frame_step=1)

  # Split the triples up and reshape them into [1, num_mel_bins] vectors, one vector for
  # lower edge, one for center and one for uppers
  lower_edge_mel, center_mel, upper_edge_mel = tuple(
    np.reshape( t, [1, num_mel_bins] ) for t in np.split(band_edges_mel, 3, axis=1))
  
  # Calculate lower and upper slopes for every spectrogram bin. Line segments are 
  # linear in the mel domain, not Hertz.
  lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
      center_mel - lower_edge_mel)
  upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
      upper_edge_mel - center_mel)
  
  # Intersect the line segments with each other and zero.
  mel_weights_matrix = np.maximum(0, np.minimum(lower_slopes, upper_slopes))
  
  # Re-add the zeroed lower bins we sliced out above
  return np.pad(mel_weights_matrix, [[n_bands_to_zero, 0], [0, 0]])

def mfcc(data, \
  fs, nSamples, frame_len, frame_step, frame_count, \
  fft_len, \
  mel_nbins, mel_lower_hz, mel_upper_hz):
  """
    Runs windowed mfcc on a strem of data

      data          input data
      fs            input sample rate
      nSamples      number of samples in input
      frame_len     length of each frame
      frame_step    how many samples to advance the frame
      frame_count   how many frames to compute
      fft_len       length of FFT, ideally frame_len 
      mel_nbins     number of mel filter banks to create
      mel_lower_hz  lowest frequency of mel bank
      mel_upper_hz  highest frequency of mel bank

  """

  if frame_count == 0:
    frame_count = 1 + (nSamples - frame_len) // frame_step
  print("Running mfcc for %d frames with %d step" % (frame_count, frame_step))
  # will return a list with a dict for each frame
  output = []

  for frame_ctr in range(frame_count):
    frame = {}
    frame['t_start'] = frame_ctr*frame_step/fs
    frame['t_end'] = (frame_ctr*frame_step+frame_len)/fs

    print("frame %d start %f end %f"%(frame_ctr, frame['t_start'],frame['t_end']))

    # get chunk of data
    chunk = data[frame_ctr*frame_step : frame_ctr*frame_step+frame_len]

    # calculate FFT
    frame['fft'] = np.fft.fft(chunk)[:frame_len//2]
    
    # calcualte spectorgram
    frame['spectrogram'] = np.abs(frame['fft'])
    num_spectrogram_bins = len(frame['spectrogram'])

    # calculate mel weights
    mel_weight_matrix = gen_mel_weight_matrix(num_mel_bins=mel_nbins, 
      num_spectrogram_bins=num_spectrogram_bins, sample_rate=fs,
      lower_edge_hertz=mel_lower_hz, upper_edge_hertz=mel_upper_hz)
    frame['mel_weight_matrix'] = mel_weight_matrix

    # Add frame to output list
    output.append(frame)

  return output


######################################################################
# Main
######################################################################

# Read data
in_fs, in_data = wavfile.read(in_wav)
in_data = np.array(in_data)

# Set MFCC settings
fs = in_fs
nSamples = len(in_data)
frame_len = 1024
frame_step = 1024
frame_count = 0 # 0 for auto
fft_len = 1024
mel_nbins = 8
mel_lower_hz = 1000
mel_upper_hz = 4000

# Some info
print("Frame length in seconds = %.3fs" % (frame_len/fs))
print("Number of input samples = %d" % (nSamples))

# calculate mfcc
o_mfcc = mfcc(in_data, fs, nSamples, frame_len, frame_step, frame_count, fft_len, mel_nbins, mel_lower_hz, mel_upper_hz)

# exit()
######################################################################
# Plottery
######################################################################
plt.style.use('seaborn-bright')
t = np.linspace(0, nSamples/fs, num=nSamples)
f = np.linspace(0.0, fs/2.0, fft_len/2)
fig, axs = plt.subplots(2, 2)

# which frame to plot
frame = o_mfcc[10]

# Input
axs[0,0].plot(t, in_data, label='input')

origin = (frame['t_start'],-30000)
width = frame['t_end']-frame['t_start']
height = 60000
rect = patches.Rectangle(origin,width,height,linewidth=0,edgecolor='r',facecolor='r')
axs[0,0].add_patch(rect)

axs[0,0].set_xlabel('time [s]')
axs[0,0].set_ylabel('amplitude')
axs[0,0].set_xlim(0,nSamples/fs)
# axs[0,0].set_ylim(-25000,25000)
axs[0,0].grid(True)
axs[0,0].legend()

# Spectrum
axs[1,0].plot(f, frame['spectrogram'], label='input')
axs[1,0].set_xlabel('frequency [Hz]')
axs[1,0].set_ylabel('spectrogram')
axs[1,0].set_xlim(10,fs/2)
axs[1,0].grid(True)
axs[1,0].legend()

# Mel coefficients
spectrogram_bins_mel = np.expand_dims( hertz_to_mel(f), 1)
axs[0,1].plot(f, frame['spectrogram']/np.max(frame['spectrogram']), color='black', label='input')
axs[0,1].plot(f, frame['mel_weight_matrix'], label='mel filters')
axs[0,1].text(mel_lower_hz, 0.4, 'mel_lower_hz',rotation=90)
axs[0,1].text(mel_upper_hz, 0.4, 'mel_upper_hz',rotation=90)
rect = patches.Rectangle((mel_lower_hz,0),mel_upper_hz-mel_lower_hz,1,linewidth=0,edgecolor='r',facecolor='k',alpha=0.2)
axs[0,1].add_patch(rect)
axs[0,1].set_xlim(0,fs/2)
axs[0,1].set_xlabel('frequency [Hz]')
axs[0,1].grid(True)
axs[0,1].legend()

# show
fig.tight_layout()
plt.show()

