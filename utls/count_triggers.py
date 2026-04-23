import os
from pathlib import Path
import numpy as np
import mne
from scipy.io import wavfile
import matplotlib.pyplot as plt
from functions import (import_raw, select_events)

tmin = -0.05
tmax = 0.3


fname  = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\test_triggers_in_stim_G_note.BDF'

n_6low = 2000
n_7low = 2000
use_non_filt = False
non_filt = ''
label_6 = '6_low'
label_7 = '7_low'
#label_6 = '6_high'
#label_7 = '7_high'
raw = mne.io.read_raw_bdf(
    fname,
    preload=True,  # Загружаем данные в память сразу
    verbose=True  # Подробный вывод процесса
)
print(raw.ch_names)
events, event_dict = mne.events_from_annotations(raw)

fs = raw.info.get('sfreq')

sorted_events = select_events(raw, n_6low, n_7low, label_6, label_7, events, event_dict)
#sorted_events = select_events(raw, n_6low, n_7low, label_6, label_7, events, event_dict)
# Создание эпох
epochs = mne.Epochs(
    raw,
    sorted_events,
    tmin=tmin,
    tmax=tmax,  # в секундах: 300 мс = 0.3 с
    baseline=(tmin, 0),
    preload=True
)

print('-----------------------------------------')
print('Общее Число меток в файле: ', len(sorted_events))
