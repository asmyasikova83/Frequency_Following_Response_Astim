import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from functions import (calculate_rms_in_intervals,
                       fir_bandpass_filter, extract_n_events,
                       detect_artifacts_threshold, detect_artifacts_trend,
                       detect_artifacts_diff, plot_noise_PSD)

#TODO
#fname_dummy  = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\test\dummy_data.BDF'
#fname_dummy  = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\test_preamplifier\dummy_data_preamplifier.BDF'
#fname_dummy  = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\non_filt\test_preamplifier\dummy_data_preamlifier_raw.BDF'
fname_dummy  = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\non_filt\test\dummy_data_raw.BDF'

subject = 'S0'
fname_non_filt  = fr'\\MCSSERVER\DB Temp\physionet.org\FFR\data\non_filt\ffr_da_N4000_non_filtS0_step1.BDF'
output_dir = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\pics\preamplifier\{0}'.format(subject)
os.makedirs(output_dir, exist_ok=True)

preamplifier = False

tmin = -0.1
tmax = 0.3

#n_6low_list = [1999, 1000, 500, 250]
#n_7low_list = [1999, 1000, 500, 250]

n_6low_list = [1999]
n_7low_list = [1999]

label_6 = '6_low'
label_7 = '7_low'

method = 'multitaper'
fmin = 40
fmax = 850
fs = 10000
order = 100

raw = mne.io.read_raw_bdf(
fname_non_filt,
    preload=True,          # Загружаем данные в память сразу
    verbose=True          # Подробный вывод процесса
    )

#raw_selected = raw.copy().pick_channels(['Fp1-Fp2'])
raw_selected = raw.copy().pick_channels(['1'])
filtered_signal = fir_bandpass_filter(raw.get_data(), fmin, fmax, fs, order)
raw_to_epo = mne.io.RawArray(filtered_signal, raw.info)

#Preprocessing 1: Filter
# Preprocessing 2: Baseline

events, event_dict = mne.events_from_annotations(raw)

# Список для хранения результатов по разным уровням усреднения
results = []

# Цикл по разным уровням усреднения
for n_6low, n_7low in zip(n_6low_list, n_7low_list):
    print(f"\n=== Обработка для n_6low={n_6low}, n_7low={n_7low} ===")

    # Preprocessing: выбор событий
    selected_events_6low = extract_n_events(
        events,
        event_dict,
        label=label_6,
        n=n_6low,
        random_selection=True
    )
    selected_events_7low = extract_n_events(
        events,
        event_dict,
        label=label_7,
        n=n_7low,
        random_selection=True
    )

    combined_events = np.concatenate([selected_events_6low, selected_events_7low])
    # Получаем индексы сортировки по первому столбцу (времени)
    sorted_indices = np.argsort(combined_events[:, 0])
    sorted_events = combined_events[sorted_indices]


    # Создание эпох
    epochs = mne.Epochs(
        raw_to_epo.copy(),
        combined_events,
        tmin=tmin,
        tmax=tmax,  # в секундах: 300 мс = 0.3 с
        baseline=(tmin, 0),
        preload=True
    )
    # Preprocessing 3: Cleaning
    if preamplifier:
        multiplier = 1e-3
    else:
        multiplier = 1e-6

    AMP_THRESHOLD = 35  # 35 мкВ в вольтах
    TREND_THRESHOLD = 10  # 10 мкВ/с в вольтах/с
    DIFF_THRESHOLD = 25  # 25 мкВ в вольтах

    bad_epochs_amp, amp_values = detect_artifacts_threshold(epochs, AMP_THRESHOLD * multiplier)

    bad_epochs_trend, trend_values = detect_artifacts_trend(epochs, TREND_THRESHOLD * multiplier)

    bad_epochs_diff, diff_values = detect_artifacts_diff(epochs, DIFF_THRESHOLD * multiplier)

    # Объединение всех критериев: эпоха помечается как артефакт, если хотя бы один метод её выявил
    bad_epochs_combined = bad_epochs_amp | bad_epochs_trend | bad_epochs_diff
    total_bad = np.sum(bad_epochs_combined)

    bad_indices = np.where(bad_epochs_combined)[0]
    epochs.drop(bad_indices, reason='artifact_detection')

    # Preprocessing 4: Grand Average
    ch_names = ['Cz']
    sfreq = float(10000)  # частота дискретизации (Гц)

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types='eeg'
    )
    ddata = epochs.get_data()

    print('ddata shape', ddata.shape)

    evokeds = []
    for j in range(ddata.shape[0]):  # по всем epochs
        data_epoch = ddata[j, :]
        if preamplifier:
            data = data_epoch * 1e-3
        else:
            data = data_epoch
        # Создаём Evoked объект
        evoked = mne.EvokedArray(
            data=data,
            info=info,
            tmin=tmin
        )
        evokeds.append(evoked)

    evokeds_noise = []
    for j in range(ddata.shape[0]):  # по всем epochs
        data_epoch = ddata[j, :]
        data_epoch_tr = data_epoch[:, 0:1000]
        if preamplifier:
            data = data_epoch_tr * 1e-3
        else:
            data = data_epoch_tr
        # Создаём Evoked объект
        evoked = mne.EvokedArray(
            data=data,
            info=info,
            tmin=tmin
        )
        evokeds_noise.append(evoked)

    # Шаг 2: grand average
    grand_average = mne.grand_average(evokeds)
    grand_average_noise = mne.grand_average(evokeds_noise)

    sumeve = n_6low + n_7low
    # Сохраняем результаты для текущего уровня усреднения
    results.append({
        'n_6low': n_6low,
        'n_7low': n_7low,
        'sumeve': sumeve,
        'grand_average': grand_average,
        'epochs_count': len(epochs),
    })

# Визуализация и сохранение всех результатов
print("\n=== Визуализация и сохранение результатов ===")
for result in results:
    plot_noise_PSD(grand_average, grand_average_noise, sumeve, method, fmin, fmax, output_dir, subject)
