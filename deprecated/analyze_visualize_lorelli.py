# Decoding Speech and Music Stimuli from the Frequency Following Response
# Steven Losorelli, Blair Kaneshiro, Gabriella A. Musacchia, Nikolas H. Blevins, Matthew B. Fitzgerald
# doi: https://doi.org/10.1101/661066
# https://purl.stanford.edu/cp051gh0103

import scipy.io as sio
import numpy as np
import mne
import matplotlib.pyplot as plt

# Шаг 1 : Load .mat-файл
mat_data = sio.loadmat(r'\\MCSSERVER\DB Temp\physionet.org\FFR\cp051gh0103\losorelli_100sweep_epoched.mat')
#mat_data = sio.loadmat(r'\\MCSSERVER\DB Temp\physionet.org\FFR\cp051gh0103\losorelli_500sweep_epoched.mat')

# Извлекаем переменные
X = mat_data['X']
Y = mat_data['Y'].flatten()
P = mat_data['P'].flatten()
t = mat_data['t'].flatten()

# Находим индексы, где Y == 2
indices = np.where(Y == 2)[0]
# Фильтруем данные
X_filtered = X[indices]
t_filtered = t

"""
print('X_filtered', X_filtered.shape)
# 325 epochs per stim, 2801 time points
# X_filtered (325, 2801)

"""
print(f"Извлечено {len(indices)} эпох, где Y == 2")

# Шаг 2 : Evoked
# Electrodes were placed at the
# frontal midline (Fz) with nasion reference and ground on bilateral earlobes

ch_names = ['Fz']
sfreq = float(20000)  # частота дискретизации (Гц)

info = mne.create_info(
    ch_names=ch_names,
    sfreq=sfreq,
    ch_types='eeg'
)

print(X_filtered.shape)
evokeds = []
for i in range(X_filtered.shape[0]):  # по всем epochs
    data_epoch = X_filtered[i, np.newaxis,  ]
    # Создаём Evoked объект
    evoked = mne.EvokedArray(
        # when plotting after grand averaging, the data get multiplied in grand_average.plot()
        data=data_epoch/ 1000000,
        info=info,
        tmin=float(t_filtered[0])  # начало временного интервала
    )
    evokeds.append(evoked)

# Шаг 3: grand average
grand_average = mne.grand_average(evokeds)
# Извлекаем данные из EvokedArray
data = grand_average.data  # массив NumPy

print(np.min(data), np.max(data))
print('t_filtered', t_filtered)
print('t_filtered[0]', t_filtered[0])
print('t_filtered[-1]', t_filtered[-1])
print('t_filtered len', len(t_filtered))
  # мкВ для ЭЭГ*(.m
# Шаг 4: визуализация
fig1 = grand_average.plot(spatial_colors=False, gfp=False,  show=False)
fig1.suptitle('Grand Average FFR Da Fz', fontsize=16)
plt.show()
