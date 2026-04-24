import os
import numpy as np
import mne
from functions import (calculate_rms_in_intervals, raw_filt, extract_n_events,
                       detect_artifacts_threshold, detect_artifacts_trend,
                       detect_artifacts_diff)
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal

# Параметры сигнала
tmin = 0.0      # начало сигнала, с
tmax = 0.25      # конец сигнала, с
tmin_noise = -0.1 # начало участка для оценки шума, с
fs = 10000       # частота дискретизации, Гц


#fname  = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\ffr_da_N4000_non_filt.BDF'
#fname  = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\non_filt\ffr_da_N4000_non_filt.BDF'
fname  = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\non_filt\ffr_da_N4000_non_filt_step2.BDF'

output_dir = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\pics\step2'
os.makedirs(output_dir, exist_ok=True)


raw = mne.io.read_raw_bdf(
    fname,
    preload=True,          # Загружаем данные в память сразу
    verbose=True          # Подробный вывод процесса
)


events, event_dict = mne.events_from_annotations(raw)

selected_events_6low = extract_n_events(
    events,
    event_dict,
    label='6_low',
    n=2000,
    random_selection=True
)
selected_events_7low = extract_n_events(
    events,
    event_dict,
    label='7_low',
    n=1999,
    random_selection=True
)
combined_events = np.concatenate([selected_events_6low, selected_events_7low])
# Получаем индексы сортировки по первому столбцу (времени)
sorted_indices = np.argsort(combined_events[:, 0])
sorted_events = combined_events[sorted_indices]

epochs = mne.Epochs(
    raw.copy(),
    combined_events,
    tmin=tmin,
    tmax=tmax,   # в секундах: 300 мс = 0.3 с
    baseline=(tmin, 0),
    preload=True
)

# Расчёт FFT для всего сигнала
t = np.linspace(tmin, tmax, int(fs * (tmax - tmin)), endpoint=False)

data = epochs.get_data()
print('data shape', data.shape)
n_times = data.shape[2] #время
# Расчёт FFT по оси времени (axis=-1 или axis=2)
#fft_result = fft(data, axis=-1)  # или axis=2
# Расчёт FFT
fft_result = fft(data)
frequencies_fft = fftfreq(n_times, 1 / fs)

# Берём только положительные частоты (односторонний спектр)
positive_mask = frequencies_fft >= 0
frequencies_positive = frequencies_fft[positive_mask]
fft_positive = fft_result[positive_mask]

# Масштабирование амплитуды (для одностороннего спектра)
amplitude_spectrum = 2.0 * np.abs(fft_positive) / n_times
amplitude_spectrum[0] = np.abs(fft_positive[0]) / n_times  # DC‑компонент не удваивается

# Перевод амплитуды в микровольты (если исходные данные в вольтах)
amplitude_uV = amplitude_spectrum * 1e6  # 1 В = 1 000 000 мкВ

# Расчёт метода Welch
f_welch, Pxx_den = signal.welch(data, fs, nperseg=1024)

# Ограничение по частоте до 1200 Гц
max_freq = 1200

# Для FFT
freq_mask_fft = frequencies_positive <= max_freq
f_fft_limited = frequencies_positive[freq_mask_fft]
amplitude_uV_limited = amplitude_uV[freq_mask_fft]

# Перевод PSD из В²/Гц в мкВ²/Гц
Pxx_den_uV = Pxx_den_limited * 1e12  # 1 В² = 10¹² мкВ²

# Визуализация
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# График 1: спектр FFT (амплитуда в мкВ)
ax1.plot(f_fft_limited, amplitude_uV_limited, linewidth=1.5, color='blue', label='FFT')
ax1.set_xlabel('Частота, Гц', fontsize=12)
ax1.set_ylabel('Амплитуда, мкВ', fontsize=12)
ax1.set_title('Спектр FFT (односторонний)', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, max_freq)
ax1.legend()

plt.tight_layout()
plt.show()
