import librosa
from IPython.display import Audio
from librosa import display
import matplotlib.pyplot as plt
import numpy as np
import sys

#解析
file_path = "D:/Documents/dev-clean/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac"
samples, sampling_rate = librosa.load(file_path, sr = None, mono = True, offset = 0.0, duration = None)
#调用librosa中的melspectrogram接口获取频谱
specs = librosa.feature.melspectrogram(y = samples, sr = sampling_rate, n_mels = 128, fmax = 8000)
#画图
plt.figure(figsize=(10, 4))
#将获取的频谱转换成分贝形式
s_db = librosa.power_to_db(specs, ref = np.max)
librosa.display.specshow(s_db, x_axis='time', y_axis='mel', sr = sampling_rate, fmax = 8000)
plt.colorbar(format = '%+2.0f db')
plt.tight_layout()
plt.show()