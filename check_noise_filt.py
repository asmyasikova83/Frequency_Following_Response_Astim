import numpy as np
import os
import mne
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
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
    eog=None,             # Автоматическое определение EOG-каналов
    misc=None,            # Автоматическое определение вспомогательных каналов
    stim_channel='auto',  # Автоматическое определение триггерных каналов
    verbose=True          # Подробный вывод процесса
)
# Получаем общую длительность записи в секундах
total_duration = raw.times[-1]

events, event_dict = mne.events_from_annotations(raw)
#filtered_events = events[events[:, 2] == 1]

epochs = mne.Epochs(
    raw,
    events,
    tmin=-0.05,  # в секундах: -50 мс = -0.05 с
    tmax=0.2,   # в секундах: 200 мс = 0.2 с
    #baseline=(-0.02, 0),  # кортеж, значения в секундах
    baseline=None,
    detrend=0,
    preload=True
)

def plot_epochs_visualization(epochs,
                           mne_block=False,
                           mne_scalings={'eeg': 1e-6},
                           show_mne=False,
                           figsize=(12, 4),
                           ylim_flat=(-1.2, 1.2),
                           title_mne=None,
                           title_flat='sin 800 Hz TS 100 ms TP 400 ms (EP: prestim 50 ms poststim 200 ms)',
                           xlabel_flat='Время (отсчёты)',
                           ylabel_flat='Амплитуда (мкВ)',
                           output_dir=output_dir):
    """
    Визуализирует эпохи двумя способами:
    1. Через MNE (автоматическая фигура)
    2. Плоский график всех отсчётов в мкВ

    Параметры:
    - epochs: объект Epochs из MNE
    - mne_block: блокировать выполнение при показе графика MNE?
    - mne_scalings: масштабирование для MNE-графика
    - show_mne: показывать ли график MNE сразу?
    - figsize: размер фигуры для плоского графика
    - ylim_flat: пределы по оси Y для плоского графика
    - title_mne: заголовок для графика MNE (если None, используется стандартный)
    - title_flat: заголовок для плоского графика
    - xlabel_flat: подпись оси X для плоского графика
    - ylabel_flat: подпись оси Y для плоского графика
    """

    # 1. Визуализация через MNE — MNE создаст свою фигуру автоматически
    epochs.plot(
        block=mne_block,
        scalings=mne_scalings,
        show=show_mne,
        title=title_mne
    )

    plt.xlabel('Время, с')
    plt.tight_layout()
    if show_mne:
        plt.show()

    # 2. Визуализация плоского графика всех отсчётов
    data = epochs.get_data()
    # Переводим в float и в микровольты (µV)
    data_uV = data.astype(np.float64) * 1e6
    # Приводим к плоскому массиву: (epochs, channels, time) → (все отсчёты)
    data_flat = data_uV.flatten()

    # Создаём новую фигуру для плоского графика
    plt.figure(figsize=figsize)
    # Строим график — транспонирование не требуется для плоского массива
    plt.plot(data_flat)

    # Настройки плоского графика
    plt.ylim(ylim_flat)
    plt.xlabel(xlabel_flat)
    plt.ylabel(ylabel_flat)
    plt.title(title_flat)
    plt.grid(True)
    plt.tight_layout()

    fpath = os.path.join(output_dir, 'sin 800 Hz TS 100 ms TP 400 ms.png')
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.show()

#RMS
def calculate_rms_in_intervals(epochs, filt, interval_prestim=(-0.05, 0), interval_poststim=(0, 0.2)):
    """
    Рассчитывает RMS сигнала для двух временных интервалов в эпохах.

    Параметры:
    -----------
    epochs : mne.Epochs
        Объект эпох MNE.
    interval1 : tuple
        Первый временной интервал (в секундах) для расчёта RMS, например (-0.02, 0).
    interval2 : tuple
        Второй временной интервал (в секундах) для расчёта RMS, например (0, 0.05).

    Возвращает:
    -----------
    rms_results : dict
        Словарь с RMS для каждого интервала, канала и эпохи.
    """

    t_ranges = dict(
        prestim=interval_prestim,
        poststim=interval_poststim
    )

    if filt:
        signal = epochs.get_data()
        # Нормализуем частоты относительно частоты НайквистаDZVG
        nyquist = 0.5 * epochs.info['sfreq']
        low = 780 / nyquist
        high = 820 / nyquist

        # Создаём полосовой фильтр Баттерворта
        b, a = butter(2, [low, high], btype='band', analog=False)

        # Применяем фильтрацию с нулевой фазой
        filtered_signal = filtfilt(b, a, signal, axis=-1)

        # Создаём новый объект Epochs с отфильтрованными данными
        epochs_filt = mne.EpochsArray(
        filtered_signal,
        epochs.info,
        tmin=epochs.tmin,
        events=epochs.events,
        event_id=epochs.event_id
        )

        plot_epochs_visualization(epochs_filt)

    else:
        epochs_filt = epochs
    idx_prestim = epochs_filt.time_as_index(t_ranges['prestim'])
    idx_poststim = epochs_filt.time_as_index(t_ranges['poststim'])

    data = epochs_filt.get_data()
    data_prestim = data[:, :, idx_prestim[0]:idx_prestim[1]]
    data_poststim = data[:, :, idx_poststim[0]:idx_poststim[1]]


    rms_prestim = np.sqrt(np.mean(data_prestim ** 2, axis=-1))  # Ось -1 — время
    rms_poststim = np.sqrt(np.mean(data_poststim ** 2, axis=-1))

    # Формируем результат
    rms_results = {
        'interval1': {
            'time_range': t_ranges['prestim'],
            'rms_values': 1_000_000 * rms_prestim  # форма: [n_epochs, n_channels]
        },
        'interval2': {
            'time_range': t_ranges['poststim'],
            'rms_values': 1_000_000 * rms_poststim
        }
    }
    snr_db = calculate_snr(rms_poststim, rms_prestim)

    return rms_results, snr_db


def calculate_snr(rms_signal, rms_noise):
    """
    Рассчитывает SNR в дБ для каждой эпохи и канала.

    Параметры:
        rms_signal: array [n_epochs, n_channels] — RMS сигнала
        rms_noise: array [n_epochs, n_channels] — RMS шума
    Возвращает:
        snr_db: array [n_epochs, n_channels] — SNR в дБ
    """
    # Избегаем деления на ноль
    rms_noise = np.where(rms_noise == 0, np.finfo(float).eps, rms_noise)

    # Вычисляем SNR в дБ
    # https://pubmed.ncbi.nlm.nih.gov/31505395/
    snr_ratio = np.mean(rms_signal) / np.mean(rms_noise)
    #https://ib-lenhardt.com/kb/glossary/snr
    # SNR_dB = 20 * log10(V_s / V_n) and SNR_dB = 10 * log10(P_s / P_n)
    snr_db = 20 * np.log10(snr_ratio)

    return snr_db

filt = False
rms_data, snr_result = calculate_rms_in_intervals(epochs, filt, (-0.05, 0), (0, 0.2))

# Выводим результаты
print(f"SNR = {snr_result:.2f} дБ")
print("RMS на престимульном интервале:")
print(f"Среднее : {np.mean( rms_data['interval1']['rms_values']):.2f} мкВ")

print("\nRMS на стимульном интервале :")
print(f"Среднее: {np.mean(rms_data['interval2']['rms_values']):.2f} мкВ")

if not filt:
    plot_epochs_visualization(epochs)