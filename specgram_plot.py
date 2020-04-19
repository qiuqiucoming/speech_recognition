import librosa
from IPython.display import Audio
from librosa import display
import matplotlib.pyplot as plt
import numpy as np
import sys
#将时域转化为频谱的函数
def spectrogram(samples, sample_rate, stride_ms = 10.0, 
                          window_ms = 20.0, max_freq = None, eps = 1e-14):

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, 
                                          shape = nshape, strides = nstrides)
    
    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2
    
    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    
    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    
    # Compute spectrogram feature
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[:ind, :] + eps)
    return specgram
#利用librosa解析出音频的时域峰值以及采样频率
file_path = "D:/Documents/dev-clean/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac"
#其中samples， sampling_rate都是一个列表
samples, sampling_rate = librosa.load(file_path, sr = None, mono = True, offset = 0.0, duration = None)
#由接口spectrogram可以获取音频对应的分贝频谱，其格式为一个多维数组
specgrams = spectrogram(samples, sampling_rate, max_freq = 8000)
#获取频谱之后就直接画图了，接口是librosa.display.specshow()
plt.figure(figsize=(10, 4))
librosa.display.specshow(specgrams, x_axis='time', y_axis='mel',sr=sampling_rate,fmax=8000)
plt.colorbar(format='%+2.0f db')
plt.title("mel specgram")
plt.tight_layout()
plt.show()