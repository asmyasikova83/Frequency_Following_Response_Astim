import os
import mne
from pathlib import Path
from mne.decoding import *
#from mne.decoding import SSD, get_spatial_filter_from_estimator
import matplotlib.pyplot as plt

# Базовый путь
base_dir = Path(r'\\MCSSERVER\DB Temp\physionet.org\files\sponge_eeg')
out_dir = Path(r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\pics')

fname_open = base_dir / 'S00_wet_eyes_open.bdf'
fname_closed = base_dir / 'S00_wet_eyes_closed.bdf'

# Параметры частот
freqs_sig = 9, 12
freqs_noise = 8, 13


# Функция для обработки данных и получения PSD
def process_and_get_psd(fname, freqs_sig, freqs_noise):
    # Загрузка данных
    raw = mne.io.read_raw_bdf(
        fname,
        preload=True,
        verbose=True
    )

    # Создание и настройка SSD
    ssd = SSD(
        info=raw.info,
        reg="oas",
        sort_by_spectral_ratio=False,
        filt_params_signal=dict(
            l_freq=freqs_sig[0],
            h_freq=freqs_sig[1],
            l_trans_bandwidth=1,
            h_trans_bandwidth=1,
        ),
        filt_params_noise=dict(
            l_freq=freqs_noise[0],
            h_freq=freqs_noise[1],
            l_trans_bandwidth=1,
            h_trans_bandwidth=1,
        ),
    )

    # Обучение SSD
    ssd.fit(X=raw.get_data())

    # Трансформация данных
    ssd_sources = ssd.transform(X=raw.get_data())

    # Получение PSD
    psd, freqs = mne.time_frequency.psd_array_welch(
        ssd_sources, sfreq=raw.info["sfreq"], n_fft=4096
    )

    # Спектральное соотношение
    spec_ratio, sorter = ssd.get_spectral_ratio(ssd_sources)

    return psd, freqs, spec_ratio, raw.info["sfreq"]


# Обработка данных для открытых глаз
print("Обработка данных для открытых глаз...")
psd_open, freqs_open, spec_ratio_open, sfreq_open = process_and_get_psd(
    fname_open, freqs_sig, freqs_noise
)

# Обработка данных для закрытых глаз
print("Обработка данных для закрытых глаз...")
psd_closed, freqs_closed, spec_ratio_closed, sfreq_closed = process_and_get_psd(
    fname_closed, freqs_sig, freqs_noise
)

# Визуализация — два графика на одной фигуре
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Верхний график — открытые глаза
below50 = freqs_open < 50
bandfilt = (freqs_sig[0] <= freqs_open) & (freqs_open <= freqs_sig[1])

ax1.loglog(freqs_open[below50], psd_open[0, below50], label="max SNR")
ax1.loglog(freqs_open[below50], psd_open[-1, below50], label="min SNR")
ax1.loglog(freqs_open[below50], psd_open[:, below50].mean(axis=0), label="mean")
ax1.fill_between(freqs_open[bandfilt], 0, 1e6, color="green", alpha=0.15)
ax1.set_title("PSD — Покой: Открытые глаза")
ax1.set_xlabel("Частота, Гц")
ax1.set_ylabel("Мощность (log)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Нижний график — закрытые глаза
below50_closed = freqs_closed < 50
bandfilt_closed = (freqs_sig[0] <= freqs_closed) & (freqs_closed <= freqs_sig[1])

ax2.loglog(freqs_closed[below50_closed], psd_closed[0, below50_closed], label="max SNR")
ax2.loglog(freqs_closed[below50_closed], psd_closed[-1, below50_closed], label="min SNR")
ax2.loglog(freqs_closed[below50_closed], psd_closed[:, below50_closed].mean(axis=0), label="mean")
ax2.fill_between(freqs_closed[bandfilt_closed], 0, 1e6, color="green", alpha=0.15)
ax2.set_title("PSD — Покой: Закрытые глаза")
ax2.set_xlabel("Частота, Гц")
ax2.set_ylabel("Мощность (log)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Общая настройка
plt.tight_layout()
out_file = Path(out_dir) / 'SSD_SNR.png'
fig.savefig(out_file, dpi=300, bbox_inches='tight')
plt.show()

# Вывод спектральных соотношений
print('Спектральное соотношение (открытые глаза):', spec_ratio_open)
print('Спектральное соотношение (закрытые глаза):', spec_ratio_closed)