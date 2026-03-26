import os
import numpy as np
import re
import mne
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    module='mne.*'  # или 'pandas.*' и т. д.
)


#fname  = r'C:\Users\msasha\Desktop\AStim\stim\check_noise_10mln_div_ep.BDF'
#fname  = r'C:\Users\msasha\Desktop\AStim\stim\check_noise_10mln_div_50ms_200ms_ep.BDF'
#fname  = r'C:\Users\msasha\Desktop\AStim\stim\check_noise_100mln_div_50ms_200ms_ep.BDF'
#fname  = r'C:\Users\msasha\Desktop\AStim\stim\check_noise_TS100ms_TP400ms_10_mln_div_pre50ms_post200ms_ep.BDF'
fname  = r'C:\Users\msasha\Desktop\AStim\stim\check_noise_TS100ms_TP400ms_100_mln_div_pre50ms_post200ms_ep.BDF'


output_dir = r'C:\Users\msasha\Desktop\AStim\stim'
output_dir_pics = r'C:\Users\msasha\Desktop\AStim\stim\pics'
os.makedirs(output_dir_pics, exist_ok=True)

def parse_filename_for_title(fname):
    """
    Парсит имя файла и формирует строку title_flat в формате:
    'sin 800 Hz TS 100 ms TP 400 ms (EP: prestim 50 ms poststim 200 ms)'

    Параметры:
        fname (str): полный путь к файлу

    Возвращает:
        str: отформатированная строка title_flat
    """
    # Извлекаем имя файла из полного пути
    filename = os.path.basename(fname)

    # Ищем параметры в имени файла с помощью регулярных выражений
    ts_match = re.search(r'TS(\d+)ms', filename)
    tp_match = re.search(r'TP(\d+)ms', filename)
    ep_pre_match = re.search(r'pre(\d+)ms_.*?ep', filename)  # prestim
    ep_post_match = re.search(r'post(\d+)ms.*?ep', filename)  # poststim

    # Поиск параметра 1_mln
    mln_match = re.search(r'(\d+_mln)', filename)

    # Извлекаем значения, если найдены, иначе — None
    ts_value = ts_match.group(1) if ts_match else None
    tp_value = tp_match.group(1) if tp_match else None
    prestim_value = ep_pre_match.group(1) if ep_pre_match else None
    poststim_value = ep_post_match.group(1) if ep_post_match else None
    mln_value = mln_match.group(1) if mln_match else None


    # Формируем базовую часть строки
    title_parts = ['sin 800 Hz']

    if ts_value:
        title_parts.append(f'TS {ts_value} ms')
    if tp_value:
        title_parts.append(f'TP {tp_value} ms')

    # Формируем часть с EP, если есть prestim и poststim
    ep_parts = []
    if prestim_value:
        ep_parts.append(f'prestim {prestim_value} ms')
    if poststim_value:
        ep_parts.append(f'poststim {poststim_value} ms')

    if ep_parts:
        title_parts.append(f'(EP: {" ".join(ep_parts)})')

    tmin = float(prestim_value) / 1000
    tmax = float(poststim_value) / 1000

    return mln_value, tmin, tmax, ' '.join(title_parts)

def bandpass_zero_phase_filter(epo, lowcut, highcut, fs, order=2):
    """
    Фильтрация сигнала с нулевой фазой (bandpass) в заданном диапазоне.

    Параметры:
    - signal: входной сигнал размерности [1, 1, N]
    - lowcut: нижняя частота среза (Гц)
    - highcut: верхняя частота среза (Гц)
    - fs: частота дискретизации (Гц)
    - order: порядок фильтра
    Возвращает: отфильтрованный сигнал той же размерности
    """
    signal = epo.get_data()
    # Нормализуем частоты относительно частоты Найквиста
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Создаём полосовой фильтр Баттерворта
    b, a = butter(order, [low, high], btype='band', analog=False)

    # Применяем фильтрацию с нулевой фазой
    filtered_signal = filtfilt(b, a, signal, axis=-1)

    # Создаём новый объект Epochs с отфильтрованными данными
    epochs = mne.EpochsArray(
        filtered_signal,
        epo.info,
        tmin=epo.tmin,
        events=epo.events,
        event_id=epo.event_id
    )
    return epochs

def plot_epochs_visualization(epochs,
                           mne_block=False,
                           mne_scalings={'eeg': 1e-6},
                           show_mne=False,
                           figsize=(12, 4),
                           #ylim_flat=(1.1, 1.1),
                           ylim_flat=None,
                           title_mne=None,
                           title_flat=None,
                           xlabel_flat='Время (отсчёты)',
                           ylabel_flat='Амплитуда (мкВ)',
                           output_dir_pics=None):
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
    filepath = os.path.join(output_dir_pics, 'pic.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"График сохранён: {filepath}")

#RMS
def calculate_rms_in_intervals(epochs_, filt, lowcutoff, highcutoff, interval_prestim, interval_poststim):
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

    epochs = bandpass_zero_phase_filter(epochs_, lowcut=lowcutoff, highcut=highcutoff, fs=epochs_.info['sfreq'] , order=2)


    t_ranges = dict(
        prestim=interval_prestim,
        poststim=interval_poststim
    )

    idx_prestim = epochs.time_as_index(t_ranges['prestim'])
    idx_poststim = epochs.time_as_index(t_ranges['poststim'])

    data = epochs.get_data()
    data_prestim =   data[:, :, idx_prestim[0]:idx_prestim[1]]
    data_poststim =   data[:, :, idx_poststim[0]:idx_poststim[1]]


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

mln_value, tmin, tmax, title_flat = parse_filename_for_title(fname)
epochs = mne.Epochs(
    raw,
    events,
    #tmin=-tmin,  # в секундах: -20 мс = -0.02 с
    #tmax=tmax,   # в секундах: 100 мс = 0.1 с
    tmin = -0.02,
    tmax=0.1,
    #baseline=(-0.02, 0),  # кортеж, значения в секундах
    baseline=None,
    detrend=0,
    preload=True
)


lowcutoff= 780
highcutoff = 820
# Ожидаемый результат: 'sin 800 Hz TS 100 ms TP 400 ms (EP: prestim 50 ms poststim 200 ms)'
rms_data, snr_result = calculate_rms_in_intervals(epochs, filt=True,lowcutoff=lowcutoff, highcutoff=highcutoff, interval_prestim=(-0.02, 0), interval_poststim=(0, 0.1))

# Выводим результаты
print(f"SNR = {snr_result:.2f} дБ")
print("RMS на престимульном интервале:")
print(f"Среднее : {np.mean( rms_data['interval1']['rms_values']):.2f} мкВ")

print("\nRMS на стимульном интервале :")
print(f"Среднее: {np.mean(rms_data['interval2']['rms_values']):.2f} мкВ")

if mln_value == '1_mln' or mln_value == '10_mln':
    ylim_flat = (-1.2, 1.2)
if mln_value == '100_mln':
    ylim_flat = (-0.1, 0.1)

plot_epochs_visualization(epochs, title_flat=title_flat, ylim_flat=ylim_flat, output_dir_pics=output_dir_pics)