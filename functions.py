import numpy as np
import os
import mne
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy import stats
import warnings
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    module='mne.*'  # или 'pandas.*' и т. д.
)



def calculate_rms_in_intervals(epochs, filt, low_cutoff, high_cutoff, interval_prestim, interval_poststim):
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

    if filt:
        signal = epochs.get_data()
        # Нормализуем частоты относительно частоты НайквистаDZVG
        nyquist = 0.5 * epochs.info['sfreq']
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist

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

    t_ranges = dict(
        prestim=interval_prestim,
        poststim=interval_poststim
    )

    idx_prestim = epochs_filt.time_as_index(t_ranges['prestim'])
    idx_poststim = epochs_filt.time_as_index(t_ranges['poststim'])

    data = epochs_filt.get_data()
    data_prestim = data[:, :, idx_prestim[0]:idx_prestim[1]]
    data_poststim = data[:, :, idx_poststim[0]:idx_poststim[1]]

    rms_prestim = np.sqrt(np.mean(data_prestim ** 2, axis=-1))
    rms_poststim = np.sqrt(np.mean(data_poststim ** 2, axis=-1))

    # Формируем результат
    rms_results = {
        'interval1': {
            'time_range': t_ranges['prestim'],
            'rms_values': 1_000_000 * rms_prestim
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
    snr_db = 10 * np.log10(snr_ratio)

    return snr_db


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
                           output_dir=r'C:\Users\msasha\Desktop\AStim\stim'):
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

    title_png = title_flat.split('(')[0]
    fpath = os.path.join(output_dir, '{title_png}.png')
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.show()

def raw_filt(raw, iir, fir, low_cutoff, high_cutoff):
    """
    Filters raw
    # iir aka  filtfilt() with zero phase (default)
    # https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.filter
    """
    if iir:
        iir_params = dict(order=8, ftype="butter")
        raw_filtered = raw.copy().filter(l_freq=low_cutoff, h_freq=high_cutoff, method='iir', iir_params=None,
                                         verbose=True)
    else:
        assert (fir == True)
        fir_params = dict(
            window='hamming',  # тип оконной функции
            fir_design='firwin'  # метод проектирования фильтра
        )

        raw_filtered = raw.copy().filter(
            l_freq=low_cutoff,
            h_freq=high_cutoff,
            method='fir',
            #fir_params=fir_params,
            verbose=True
        )
    return raw_filtered

def extract_n_events(events, event_dict, label, n=500, random_selection=True):
    """
    Извлекает ровно n событий с заданной меткой.

    Parameters:
    -----------
    events : array, shape (n_events, 3)
        Массив событий из mne.events_from_annotations()
    event_dict : dict
        Словарь меток из mne.events_from_annotations()
    label : str
        Название метки для извлечения
    n : int
        Требуемое количество событий
    random_selection : bool
        Если True — случайный выбор, иначе — первые n событий

    Returns:
    --------
    selected_events : array
        Выбранные события
    """
    # Получаем ID метки
    target_id = event_dict[label]

    # Находим все события с этой меткой
    indices = np.where(events[:, 2] == target_id)[0]
    available_count = len(indices)

    print('available_count', available_count)

    # Выбираем события
    if random_selection:
        selected_indices = np.random.choice(indices, size=n, replace=False)
    else:
        selected_indices = indices[:n]

    return events[selected_indices]

def detect_artifacts_threshold(epochs, threshold_uV=80):
    """
    Обнаружение артефактов по амплитудному порогу.
    """
    threshold = threshold_uV * 1e-6  # мкВ → В
    data = epochs.get_data(copy=True)  # форма: (n_epochs, n_channels, n_times)

    # Проверяем абсолютную амплитуду по всем каналам и отсчётам
    max_amps = np.max(np.abs(data), axis=(1, 2))  # макс. амплитуда для каждой эпохи

    # Отмечаем эпохи, где макс. амплитуда > порога
    bad_epochs = max_amps > threshold
    check = max_amps[0] - threshold
    return bad_epochs, max_amps


def detect_artifacts_trend(epochs, trend_threshold_uVs=10):
    """
    Обнаружение артефактов по наклону тренда.
    """
    trend_threshold = trend_threshold_uVs * 1e-6  # мкВ/с → В/с

    data = epochs.get_data(copy=True)
    n_epochs, n_channels, n_times = data.shape
    sfreq = epochs.info['sfreq']

    # Временные метки в секундах для каждой точки в эпохе
    times = np.arange(n_times) / sfreq
    print('times', times)
    trends = np.zeros(n_epochs)

    for i in range(n_epochs):
        epoch_data = data[i]  # данные для одной эпохи
        # Вычисляем наклон тренда для каждого канала
        channel_trends = []
        for ch in range(n_channels):
            # Линейная регрессия: наклон — это коэффициент при x
            slope, _, _, _, _ = stats.linregress(times, epoch_data[ch])
            channel_trends.append(abs(slope))
        # Берём макс. наклон среди каналов
        trends[i] = np.max(channel_trends)

    # Отмечаем эпохи с наклоном > порога
    bad_epochs = trends > trend_threshold
    return bad_epochs, trends

def plot_trend_detection(epochs, epoch_idx, channel_idx=0):
    """Показывает подгонку тренда для выбранной эпохи и канала."""
    data = epochs.get_data(copy=True)
    epoch_data = data[epoch_idx, channel_idx, :]
    sfreq = epochs.info['sfreq']
    times = np.arange(len(epoch_data)) / sfreq  # время в секундах

    # Линейная регрессия
    slope, intercept, _, _, _ = stats.linregress(times, epoch_data)
    trend_line = slope * times + intercept  # уравнение прямой

    plt.figure(figsize=(10, 4))
    plt.plot(times, epoch_data * 1e6, label='Сигнал (мкВ)')  # в мкВ
    plt.plot(times, trend_line * 1e6, '--r', label=f'Тренд (наклон: {slope*1e6:.2f} мкВ/с)')
    plt.xlabel('Время, с')
    plt.ylabel('Амплитуда, мкВ')
    plt.title(f'Эпоха {epoch_idx}, канал {channel_idx}')
    plt.legend()
    plt.grid(True)
    plt.show()


def detect_artifacts_diff(epochs, diff_threshold_uV=25):
    """
    Обнаружение артефактов по разности между соседними отсчётами.
    """
    diff_threshold = diff_threshold_uV * 1e-6  # мкВ → В
    data = epochs.get_data(copy=True)
    n_epochs, n_channels, n_times = data.shape

    max_diffs = np.zeros(n_epochs)
    for i in range(n_epochs):
        epoch_data = data[i]
        # Разность между соседними отсчётами по времени
        diffs = np.diff(epoch_data, axis=1)  # форма: (n_channels, n_times-1)
        # Макс. абсолютная разность среди всех каналов и отсчётов
        max_diff = np.max(np.abs(diffs))
        max_diffs[i] = max_diff
    # Отмечаем эпохи с макс. разностью > порога
    bad_epochs = max_diffs > diff_threshold
    return bad_epochs, max_diffs

