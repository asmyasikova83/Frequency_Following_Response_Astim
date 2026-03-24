import numpy as np
import mne
import warnings
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    module='mne.*'  # или 'pandas.*' и т. д.
)

fname  = r'C:\Users\msasha\Desktop\AStim\stim\check_noise_ep.BDF'
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
    tmin=-0.02,  # в секундах: -200 мс = -0.2 с
    tmax=0.1,   # в секундах: 100 мс = 0.1 с
    #baseline=(-0.02, 0),  # кортеж, значения в секундах
    baseline=None,
    detrend=0,
    preload=True
)


data = epochs.get_data()
print(data.shape)



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