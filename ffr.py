import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from functions import (calculate_rms_in_intervals, raw_filt, extract_n_events,
                       detect_artifacts_threshold, detect_artifacts_trend,
                       detect_artifacts_diff)



#fname  = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\ffr_da_N4000_non_filt.BDF'
#TODO
fname_non_filt  = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\non_filt\ffr_da_N4000_non_filt_step2_raw.BDF'
fname  = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\ffr_da_N4000_step2_filt.BDF'
fname_dummy  = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\test\dummy_data.BDF'

output_dir = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\pics\step2\pics_to_combine'
os.makedirs(output_dir, exist_ok=True)

dummy = False
non_filt = False
if dummy:
    raw = mne.io.read_raw_bdf(
    fname_dummy,
    preload=True,          # Загружаем данные в память сразу
    verbose=True          # Подробный вывод процесса
    )
    raw_to_epo = raw
elif non_filt:
    raw = mne.io.read_raw_bdf(
    fname_non_filt,
    preload=True,          # Загружаем данные в память сразу
    verbose=True          # Подробный вывод процесса
    )
    raw.set_eeg_reference(ref_channels=['13', '19'], projection=False)
    raw_selected = raw.copy().pick_channels(['1'])
    raw_to_epo = raw_selected
else:
    raw = mne.io.read_raw_bdf(
    fname,
    preload=True,          # Загружаем данные в память сразу
    verbose=True          # Подробный вывод процесса
    )
    raw_to_epo = raw

low_cutoff = 40
high_cutoff = 250


tmin = -0.05
tmax = 0.3

n_6low_list = [2000, 1000, 500, 250]
n_7low_list = [1999, 1000, 500, 250]

label_6 = '6_low'
label_7 = '7_low'

if dummy:
    n_6low_list = [1999]
    n_7low_list = [1996]
if non_filt:
    label_6 ='In\\6'
    label_7 ='In\\7'


"""
# Preprocessing 1: Filter
raw.notch_filter(
    freqs=np.arange(50, 550, 50),  # частоты для фильтрации: 50, 100, 150, 200, 250,
    method='fir',  # метод: FIR‑фильтр
    filter_length='auto',  # автоматическая длина фильтра
    phase='zero-double',  # нулевая фаза (без сдвига сигнала)
    fir_window='hann',
    fir_design='firwin'  # дизайн FIR‑фильтра
)


iir = False
fir = True

#raw_f = raw_filt(raw, iir, fir, low_cutoff, high_cutoff)

# Preprocessing 2: Baseline
# Prestimulus [-0.05; 0] Stimulus [0 ; 0.25] + 0.05 to get back to baseline
"""

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
        raw_to_epo,
        combined_events,
        tmin=tmin,
        tmax=tmax,  # в секундах: 300 мс = 0.3 с
        baseline=(tmin, 0),
        preload=True
    )

    # Preprocessing 3: Cleaning
    AMP_THRESHOLD = 35e-6  # 35 мкВ в вольтах
    TREND_THRESHOLD = 10e-6  # 10 мкВ/с в вольтах/с
    DIFF_THRESHOLD = 25e-6  # 25 мкВ в вольтах

    bad_epochs_amp, amp_values = detect_artifacts_threshold(epochs, AMP_THRESHOLD * 1e6)
    print(f"Artifacts — amplitude: {np.sum(bad_epochs_amp)} эпох")

    bad_epochs_trend, trend_values = detect_artifacts_trend(epochs, TREND_THRESHOLD * 1e6)
    print(f"Artifacts — trend: {np.sum(bad_epochs_trend)} эпох")

    bad_epochs_diff, diff_values = detect_artifacts_diff(epochs, DIFF_THRESHOLD * 1e6)
    print(f"Artifacts — difference between 2 adjacent time points: {np.sum(bad_epochs_diff)} эпох")

    # Объединение всех критериев: эпоха помечается как артефакт, если хотя бы один метод её выявил
    bad_epochs_combined = bad_epochs_amp | bad_epochs_trend | bad_epochs_diff
    total_bad = np.sum(bad_epochs_combined)
    print(f"Total artifactual epochs (combined): {total_bad} epochs")

    # Получение индексов плохих эпох
    bad_indices = np.where(bad_epochs_combined)[0]
    # Удаление плохих эпох из объекта epochs
    epochs.drop(bad_indices, reason='artifact_detection')

    # Preprocessing 4: Grand Average
    ch_names = ['Cz']
    sfreq = float(10000)  # частота дискретизации (Гц)

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types='eeg'
    )

    data = epochs.get_data()

    evokeds = []
    for i in range(data.shape[0]):  # по всем epochs
        data_epoch = data[i, :]
        # Создаём Evoked объект
        evoked = mne.EvokedArray(
            data=data_epoch,
            info=info,
            tmin=tmin
        )
        evokeds.append(evoked)

    # Шаг 2: grand average
    grand_average = mne.grand_average(evokeds)
    data_averaged = grand_average.data

    sumeve = n_6low + n_7low

    # Расчёт SNR
    filt = False
    rms_data, snr_result = calculate_rms_in_intervals(
        epochs, filt, low_cutoff, high_cutoff, (tmin, 0), (0, tmax)
    )

    # Сохраняем результаты для текущего уровня усреднения
    results.append({
        'n_6low': n_6low,
        'n_7low': n_7low,
        'sumeve': sumeve,
        'grand_average': grand_average,
        'epochs_count': len(epochs),
        'snr': snr_result,
        'rms_data': rms_data
    })

    print(f"Обработано эпох после очистки: {len(epochs)}")
    print(f"Суммарное количество усреднений: {sumeve}")
    print(f"SNR = {snr_result:.2f} дБ")
    print("RMS на престимульном интервале:")
    print(f"Среднее: {np.mean(rms_data['interval1']['rms_values']):.2f} мкВ")
    print("\nRMS на стимульном интервале:")
    print(f"Среднее: {np.mean(rms_data['interval2']['rms_values']):.2f} мкВ")

# Визуализация и сохранение всех результатов
print("\n=== Визуализация и сохранение результатов ===")
for result in results:
    grand_avg = result['grand_average']
    sumeve = result['sumeve']
    snr_result = result['snr']

    # Создаём фигуру для текущего результата
    fig1 = grand_avg.plot(
        spatial_colors=False,
        gfp=False,
        show=False
    )

    ax = fig1.axes[0]

    ax.axvline(
        x=0,  # координата X (время = 0)
        color='red',  # красный цвет
        linestyle='--',  # пунктирная линия
        linewidth=2,  # толщина линии
        alpha=0.8,  # прозрачность
        zorder=10  # порядок отрисовки (поверх данных)
    )

    # Добавляем текст с SNR на график
    ax.text(
        0.68, 0.95,  # позиция (относительные координаты)
        f'SNR = {snr_result:.2f} дБ',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    if dummy:
        facecolor = 'white'
        output_path = os.path.join(output_dir, f'FFR_{sumeve}_test_dummy.png')
    elif non_filt:
        facecolor = 'white'
        output_path = os.path.join(output_dir, f'FFR_{sumeve}_raw_non_filt.png')
    else:
        #facecolor = 'lightgreen'
        facecolor = 'white'
        output_path = os.path.join(output_dir, f'FFR_{sumeve}.png')

    # Сохраняем с настройками качества
    fig1.savefig(
    output_path,
    dpi=300,                  # Высокое разрешение
    bbox_inches='tight',     # Убирает лишние поля
    facecolor=facecolor ,       # Белый фон
    edgecolor='none'         # Без рамки
    )

    print(f"Рисунок сохранён: {output_path}")
