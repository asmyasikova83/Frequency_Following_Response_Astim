import os
from pathlib import Path
import numpy as np
import mne
from functions import (import_raw, select_events, calculate_rms_in_intervals,
                       remove_artifacts, plot_noise_PSD, plot_PSD, compute_GA, plot_GA)

subject = 'S0'
preamplifier = 'preamplifier' # ''
dummy = ''# 'dummy'
non_filt = 'non_filt' # ''

# Базовый путь
base_path = Path(r'\\MCSSERVER\DB Temp\physionet.org\FFR\data')

#fname_non_filt  = fr'\\MCSSERVER\DB Temp\physionet.org\FFR\data\{non_filt}\ffr_da_N4000_{non_filt}_{subject}_{preamplifier}.BDF'
if dummy:
    fname = base_path / non_filt / dummy / preamplifier / f'ffr_da_N4000_{dummy}{non_filt}{preamplifier}.BDF'
    output_dir = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\pics\{0}\{1}'.format(preamplifier, dummy)
else:
    fname = base_path / non_filt / dummy / preamplifier / f'ffr_da_N4000_{dummy}{non_filt}{subject}{preamplifier}.BDF'
    output_dir = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\pics\{0}\{1}\{2}'.format(preamplifier, dummy, subject)
os.makedirs(output_dir, exist_ok=True)

# Prestimulus [-0.05; 0] Stimulus [0 ; 0.25] + 0.05 to get back to baseline
tmin = -0.05
tmax = 0.4

#n_6low_list = [1999, 1000, 500, 250]
#n_7low_list = [1999, 1000, 500, 250]

if dummy:
    n_6low_list = [1999]
    n_7low_list = [1992]
else:
    n_6low_list = [1999]
    n_7low_list = [1999]

method = 'multitaper'
fmin = 40
fmax = 250
order = 100

if preamplifier:
    multiplier = 1e-3
else:
    multiplier = 1e-6

AMP_THRESHOLD = 35  # 35 мкВ в вольтах
TREND_THRESHOLD = 10  # 10 мкВ/с в вольтах/с
DIFF_THRESHOLD = 25  # 25 мкВ в вольтах

# Preprocessing 1: Import Raw
raw, raw_to_epo, events, event_dict, label_6, label_7 = import_raw(fname, non_filt, preamplifier, dummy, fmin, fmax, order)
fs = raw.info.get('sfreq')

# Цикл по разным уровням усреднения
for n_6low, n_7low in zip(n_6low_list, n_7low_list):
    print(f"\n=== Обработка для n_6low={n_6low}, n_7low={n_7low} ===")

    # Preprocessing 2: Epoching with baseline
    sorted_events = select_events(raw_to_epo, n_6low, n_7low, label_6, label_7, events, event_dict)
    # Создание эпох
    epochs = mne.Epochs(
        raw_to_epo,
        sorted_events,
        tmin=tmin,
        tmax=tmax,  # в секундах: 300 мс = 0.3 с
        baseline=(tmin, 0),
        preload=True
    )
    # Preprocessing 3: Cleaning
    epochs = remove_artifacts(epochs, sorted_events, AMP_THRESHOLD, TREND_THRESHOLD, DIFF_THRESHOLD, multiplier)

    # Preprocessing 4: Grand Average
    noise = False
    grand_average = compute_GA(epochs, fs, preamplifier, noise, tmin)
    # data_averaged = fir_bandpass_filter(grand_average.data, fmin, fmax, fs, order)

    # FFT — Power Spectral Density (PSD)
    sumeve = n_6low + n_7low
    plot_PSD(grand_average, sumeve, method, fmin, fmax, output_dir, subject)

    # SNR
    filt = False
    rms_data, snr_result = calculate_rms_in_intervals(
        epochs, filt, fmin, fmax, (tmin, 0), (0, tmax), preamplifier
    )

    # Создаём фигуру для текущего результата
    if dummy:
        output_path = os.path.join(output_dir, f'FFR_dummy_{preamplifier}_{tmin}ms_{tmax}ms_FIR offline_{fmin}_{fmax}Hz_N{sumeve}.png')
        title = f'Шум {preamplifier} {tmin}ms_{tmax}ms_FIR offline {fmin}_{fmax}Hz N{sumeve}'
    else:
        output_path = os.path.join(output_dir, f'FFR_Da_{preamplifier}_{tmin}ms_{tmax}ms_FIR_{fmin}_{fmax}Hz_{subject}_N{sumeve}.png')
        title = f'FFR {subject} Da {tmin}ms_{tmax}ms_FIR {fmin}_{fmax}Hz N{sumeve} '

    plot_GA(grand_average, output_path, snr_result, title)
    noise = True
    grand_average_noise = compute_GA(epochs, fs, preamplifier, noise, tmin)
    plot_noise_PSD(grand_average, grand_average_noise, sumeve, method, fmin, fmax, output_dir, subject)

