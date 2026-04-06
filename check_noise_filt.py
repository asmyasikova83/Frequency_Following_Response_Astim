import numpy as np
import os
import mne
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from functions import (calculate_rms_in_intervals,
                        plot_epochs_visualization)

import warnings
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    module='mne.*'  # или 'pandas.*' и т. д.
)


#fname  = r'C:\Users\msasha\Desktop\AStim\stim\check_noise_10mln_div_ep.BDF'
#fname  = r'C:\Users\msasha\Desktop\AStim\stim\check_noise_TS100ms_TP400ms_1_mln_div_pre50ms_post200ms_ep.BDF'
#fname  = r'C:\Users\msasha\Desktop\AStim\stim\check_noise_TS100ms_TP400ms_100_mln_div_pre50ms_post200ms_ep.BDF'
fname  = r'C:\Users\msasha\Desktop\AStim\stim\check_noise_10_mln_div_pre50ms_post200ms_ep.BDF'

#fname  = r'C:\Users\msasha\Desktop\AStim\stim\check_noise_10mln_TS100ms_TP300ms_N2000_prestim50ms_poststim200ms_ep.BDF'

output_dir = r'C:\Users\msasha\Desktop\AStim\stim'

raw = mne.io.read_raw_bdf(
    fname,
    preload=True,           # Загружаем данные в память сразу
    verbose=True          # Подробный вывод процесса
)

events, event_dict = mne.events_from_annotations(raw)
#filtered_events = events[events[:, 2] == 1]

tmin = -0.05
tmax = 0.2

epochs = mne.Epochs(
    raw,
    events,
    tmin=tmin,  # в секундах: -50 мс = -0.05 с
    tmax=tmax,   # в секундах: 200 мс = 0.2 с
    #baseline=(-0.02, 0),  # кортеж, значения в секундах
    baseline=None,
    detrend=0,
    preload=True
)

low_cutoff = 780
high_cutoff = 820

filt = True
rms_data, snr_result = calculate_rms_in_intervals(epochs, filt, low_cutoff, high_cutoff,(tmin, 0), (0, tmax))

# Выводим результаты
print(f"SNR = {snr_result:.2f} дБ")
print("RMS на престимульном интервале:")
print(f"Среднее : {np.mean( rms_data['interval1']['rms_values']):.2f} мкВ")

print("\nRMS на стимульном интервале :")
print(f"Среднее: {np.mean(rms_data['interval2']['rms_values']):.2f} мкВ")

if not filt:
    plot_epochs_visualization(epochs)