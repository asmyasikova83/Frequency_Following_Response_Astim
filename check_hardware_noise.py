import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    module='mne.*'  # или 'pandas.*' и т. д.
)


#fname  = r'C:\Users\msasha\Desktop\AStim\stim\check_noise_10mln_div_ep.BDF'
fname  = r'C:\Users\msasha\Desktop\AStim\stim\check_noise_long50ms_200ms_ep.BDF'

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

epochs = mne.Epochs(
    raw,
    events,
    tmin=-0.05,  # в секундах: -20 мс = -0.02 с
    tmax=0.2,   # в секундах: 100 мс = 0.1 с
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
                           ylabel_flat='Амплитуда (мкВ)'):
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
    plt.show()


#RMS
def calculate_rms_in_intervals(epochs, interval_prestim=(-0.02, 0), interval_poststim=(0, 0.1)):
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

    idx_prestim = epochs.time_as_index(t_ranges['prestim'])
    idx_poststim = epochs.time_as_index(t_ranges['poststim'])

    data = epochs.get_data()
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


rms_data, snr_result = calculate_rms_in_intervals(epochs, (-0.02, 0), (0, 0.05))

# Выводим результаты
print(f"SNR = {snr_result:.2f} дБ")
print("RMS на престимульном интервале:")
print(f"Среднее : {np.mean( rms_data['interval1']['rms_values']):.2f} мкВ")

print("\nRMS на стимульном интервале :")
print(f"Среднее: {np.mean(rms_data['interval2']['rms_values']):.2f} мкВ")

plot_epochs_visualization(epochs)