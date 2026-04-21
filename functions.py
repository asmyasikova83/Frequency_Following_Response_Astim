import numpy as np
import os
import mne
import matplotlib.pyplot as plt
import random
from scipy import stats
from scipy.signal import butter
from scipy.signal import firwin, firwin2, filtfilt
import warnings
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    module='mne.*'  # или 'pandas.*' и т. д.
)


def import_raw(fname, non_filt, use_non_filt, preamplifier, dummy, fmin, fmax, order,transition_width):
    label_6 = '6_low'
    label_7 = '7_low'
    raw = mne.io.read_raw_bdf(
        fname,
        preload=True,  # Загружаем данные в память сразу
        verbose=True  # Подробный вывод процесса
    )
    print(raw.ch_names)

    if non_filt and not preamplifier:
        raw.set_eeg_reference(ref_channels=['13', '19'], projection=False)
        raw_selected = raw.copy().pick_channels(['1'])
        label_6 = 'In\\6'
        label_7 = 'In\\7'
    elif dummy and non_filt and preamplifier:
        raw.set_eeg_reference(ref_channels=['13', '19'], projection=False)
        raw_selected = raw.copy().pick_channels(['1'])
        label_6 = 'In\\6'
        label_7 = 'In\\7'
    else:
        raw_selected = raw

    if use_non_filt:
        raw_to_epo = raw_selected
    else:
        filtered_signal = fir_bandpass_filter(raw_selected.get_data(), fmin, fmax,  int(raw.info.get('sfreq')), order,transition_width)
        raw_to_epo = mne.io.RawArray(filtered_signal, raw_selected.info)


    events, event_dict = mne.events_from_annotations(raw)
    #raw_to_epo = raw_selected
    return raw, raw_to_epo, events, event_dict, label_6, label_7

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

def select_events(raw_to_epo,n_6low, n_7low, label_6, label_7, events, event_dict):

        # Preprocessing: выбор событий
        selected_events_6low = extract_n_events(
        events,
        event_dict,
        label=label_6,
        n=n_6low,
        random_selection=True
        )
        print('-----------------------------------------')
        print('Число меток 6 low  файле BDF', len(selected_events_6low))
        print('-----------------------------------------')
        selected_events_7low = extract_n_events(
        events,
        event_dict,
        label=label_7,
        n=n_7low,
        random_selection=True
        )
        print('-----------------------------------------')
        print('Число меток 7 low  файле BDF', len(selected_events_7low))
        print('-----------------------------------------')
        combined_events = np.concatenate([selected_events_6low, selected_events_7low])
        # Получаем индексы сортировки по первому столбцу (времени)
        sorted_indices = np.argsort(combined_events[:, 0])
        sorted_events = combined_events[sorted_indices]

        return sorted_events

def remove_artifacts(epochs, sorted_events, AMP_THRESHOLD, TREND_THRESHOLD, DIFF_THRESHOLD, multiplier):
    bad_epochs_amp, amp_values = detect_artifacts_threshold(epochs, AMP_THRESHOLD * multiplier)
    print(f"Artifacts — amplitude: {np.sum(bad_epochs_amp)} эпох")

    bad_epochs_trend, trend_values = detect_artifacts_trend(epochs, TREND_THRESHOLD * multiplier)
    print(f"Artifacts — trend: {np.sum(bad_epochs_trend)} эпох")

    bad_epochs_diff, diff_values = detect_artifacts_diff(epochs, DIFF_THRESHOLD * multiplier)
    print(f"Artifacts — difference between 2 adjacent time points: {np.sum(bad_epochs_diff)} эпох")

    # Объединение всех критериев: эпоха помечается как артефакт, если хотя бы один метод её выявил
    bad_epochs_combined = bad_epochs_amp | bad_epochs_trend | bad_epochs_diff
    total_bad = np.sum(bad_epochs_combined)
    print(f"Total artifactual epochs (combined): {total_bad} epochs")

    # Получение индексов плохих эпох
    bad_indices = np.where(bad_epochs_combined)[0]
    # Удаление плохих эпох из объекта epochs
    epochs.drop(bad_indices, reason='artifact_detection')
    return epochs, bad_indices

def plot_stim(stimulus, grand_aver,  ax, tmin, tmax, fs, ts):
    if grand_aver:
        data_stim_padded = stimulus
    else:
        # Извлекаем каждый 4‑й отсчёт (fs stim = 44100, fs data = 10000)
        """Построение стимула на заданной оси"""
        #data = stimulus[:,0].copy()
        data = stimulus.copy()
        if fs == 10000.0:
            data_stim = data[::4, 0]
        else:
            data_stim = data
        n_points_decim = len(data_stim)

        # Длительность стимула в отсчётах и секундах
        stim_duration_samples =  ts * fs
        # Количество нулей для добавления
        n_zeros_front = int(-tmin * fs)  # спереди
        n_zeros_back = int(tmax * fs - stim_duration_samples)  # сзади
        # Добавление нулей спереди и сзади
        data_stim_padded = np.concatenate([
            np.zeros(n_zeros_front),  # нули спереди
            data_stim[:int(stim_duration_samples)],  # исходные данные стимула
            np.zeros(n_zeros_back)  # нули сзади
        ])
    print('data_stim_padded', data_stim_padded)
    n_points = data_stim_padded.shape[0]
    times_stim = np.linspace(tmin, tmax, n_points, endpoint=False)

        # Строим график
    ax.plot(times_stim, data_stim_padded, color='green', linewidth=1.5, label='Стимул')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(x=ts, color='red', linestyle='--', linewidth=2, alpha=0.8)
    # Устанавливаем границы оси X
    ax.set_xlim(tmin, tmax)
    # Убираем разметку с оси Y
    ax.set_yticks([])  # убираем деления (ticks) на оси Y
    ax.set_ylabel('')  # убираем подпись оси Y
    ax.tick_params(axis='both', which='major', labelsize=10)
    #ax.set_xlabel('cек', fontsize = 10)
    ax.set_xlabel('')

def fir_bandpass_filter(data, low_cutoff, high_cutoff, fs, order, transition_width):
    """
    FIR-фильтрация с нулевой фазой (полосовой фильтр).

    Обрабатывает одно- и многоканальные данные.

    Параметры:
    - data: массив данных (n_channels, n_samples или n_samples);
    - low_cutoff: нижняя частота среза (Гц);
    - high_cutoff: верхняя частота среза (Гц);
    - fs: частота дискретизации (Гц);
    - order: порядок FIR-фильтра (по умолчанию 100).

    Возвращает: отфильтрованные данные той же формы, что и входные.
    """
    # Нормализуем частоты относительно частоты Найквиста
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    """
    # Создаём полосовой FIR-фильтр с помощью firwin
    b = firwin(
        numtaps=order + 1,
        cutoff=[low, high],
        pass_zero=False,
        window='bartlett'
    )
    """
    freq = [0, max(0, low - transition_width), low, high, min(1, high + transition_width), 1]
    gain = [0, 0, 1, 1, 0, 0]

    # Удаляем дубликаты частот
    unique_freq, unique_indices = np.unique(freq, return_index=True)
    unique_gain = [gain[idx] for idx in unique_indices]

    b = firwin2(
        numtaps=order + 1,
        freq=unique_freq,
        gain=unique_gain,
        fs=2.0
    )

    # Коэффициенты знаменателя для FIR-фильтра: a = 1
    a = 1.0

    # Применяем фильтрацию с нулевой фазой (двукратная фильтрация вперёд и назад)
    # axis=-1 — фильтрация вдоль последней оси (по времени)
    filtered_signal = filtfilt(b, a, data, axis=-1)

    return filtered_signal

def detect_artifacts_threshold(epochs, threshold_uV):
    """
    Обнаружение артефактов по амплитудному порогу.
    """
    data = epochs.get_data(copy=True)  # форма: (n_epochs, n_channels, n_times)
    # Проверяем абсолютную амплитуду по всем каналам и отсчётам
    max_amps = np.max(np.abs(data), axis=(1, 2))  # макс. амплитуда для каждой эпохи

    # Отмечаем эпохи, где макс. амплитуда > порога
    bad_epochs = max_amps > threshold_uV
    check = max_amps[0] - threshold_uV
    return bad_epochs, max_amps


def detect_artifacts_trend(epochs, trend_threshold_uVs):
    """
    Обнаружение артефактов по наклону тренда.
    """
    data = epochs.get_data(copy=True)
    n_epochs, n_channels, n_times = data.shape
    sfreq = epochs.info['sfreq']

    # Временные метки в секундах для каждой точки в эпохе
    times = np.arange(n_times) / sfreq
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
    bad_epochs = trends > trend_threshold_uVs
    return bad_epochs, trends


def detect_artifacts_diff(epochs, diff_threshold_uV):
    """
    Обнаружение артефактов по разности между соседними отсчётами.
    """
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
    bad_epochs = max_diffs > diff_threshold_uV
    return bad_epochs, max_diffs

def calculate_rms_in_intervals(epochs, filt, low_cutoff, high_cutoff, interval_prestim, interval_poststim, preamplifier):
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
        b, a = butter(order+1, [low, high], btype='band', analog=False)

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

    if preamplifier:
        rms_prestim = 1e3 * rms_prestim
        rms_poststim = 1e3 * rms_poststim
    else:
        rms_prestim = 1e6 * rms_prestim
        rms_poststim = 1e6 * rms_poststim

    # Формируем результат
    rms_results = {
        'interval1': {
            'time_range': t_ranges['prestim'],
            'rms_values': rms_prestim
        },
        'interval2': {
            'time_range': t_ranges['poststim'],
            'rms_values': rms_poststim
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

def compute_GA(epochs, fs, preamplifier, noise, tmin):
    # Preprocessing 4: Grand Average
    info = mne.create_info(
        ch_names=['Cz'],
        sfreq=fs,
        ch_types='eeg'
    )

    ddata = epochs.get_data()

    evokeds = []
    for j in range(ddata.shape[0]):  # по всем epochs
        data_epoch = ddata[j, :]
        if noise:
            prestim_interval = -tmin * fs
            data_epoch_tr = data_epoch[:, 0:int(prestim_interval)]
        else:
            data_epoch_tr = data_epoch
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
        evokeds.append(evoked)

    grand_average = mne.grand_average(evokeds)
    return grand_average

def plot_GA(dummy, grand_avg, GA, ax, ts, tmin, fs):
    """Построение Grand Average на заданной оси"""
    if GA:
        info = mne.create_info(
        ch_names=['Cz'],
        sfreq=fs,
        ch_types='eeg'
        )
        evoked = mne.EvokedArray(
            data=grand_avg,
            info=info,
            tmin=tmin
        )
        grand_avg = evoked
    grand_avg.plot(
        spatial_colors=False,
        gfp=False,
        axes=ax,
        show=False,
        verbose=None
    )
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(x=ts, color='red', linestyle='--', linewidth=2, alpha=0.8)
    if dummy:
        ax.set_ylim(-0.08, 0.08)
        ax.set_yticks([-0.08, 0.08])
    else:

        ax.set_ylim(-0.7, 0.7)
        ax.set_yticks([-0.7, 0.7])
    ax.axhline(
        y=0.00,
        color='grey',
        linestyle='--',
        linewidth=1,
        alpha=0.8,
        label='y=0 muV '  # подпись линии
    )
    ax.legend(loc='upper right')  # отображает легенду с подписью
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xlabel('Время, сек', fontsize=10, loc='left')

def zero_padding(stimulus, ga, padding_factor):
    # zero paddding для fft
    # Параметры zero‑padding
    if ga:
        original_length = stimulus.shape[1]
    else:
        original_length = len(stimulus)
    target_length = original_length * padding_factor
    # Добавляем нули в конец сигнала
    if ga:
        n_channels, original_length = stimulus.shape
        zeros_to_add = target_length - original_length
        zeros_array = np.zeros((n_channels, zeros_to_add))
        stimulus_padded = np.concatenate([stimulus, zeros_array], axis=1)

    else:
        stimulus_padded = np.pad(stimulus, (0, target_length - original_length), mode='constant', constant_values=0)
        stimulus_padded = stimulus_padded[:, np.newaxis]
    return stimulus_padded

def plot_noise_PSD(dummy, grand_average, grand_average_noise, ax, method, fmin, fmax, fs, tmin):
    # FFT анализ — Power Spectral Density (PSD)

    ga = True
    ga_data_padded = zero_padding(grand_average.get_data(),  ga, padding_factor = 4)
    info = mne.create_info(
        ch_names=['Cz'],
        sfreq=fs,
        ch_types='eeg'
    )
    evoked = mne.EvokedArray(
            data=ga_data_padded,
            info=info,
            tmin=tmin
    )
    grand_average_padded = evoked
    psd = grand_average_padded.compute_psd(
        method=method,
        fmin=fmin,
        fmax=fmax,
        verbose=False
    )

    # Визуализация PSD
    # 2. Преобразуем мощность в амплитуду: извлекаем квадратный корень
    data_psd = psd.get_data()  # получаем массив мкВ²/Гц
    data_amplitude = np.sqrt(data_psd).flatten()  # преобразуем в мкВ/√Гц
    freqs_data = psd.freqs

    ga = True
    ga_noise_data_padded = zero_padding(grand_average_noise.get_data(),  ga, padding_factor = 4)
    info = mne.create_info(
        ch_names=['Cz'],
        sfreq=fs,
        ch_types='eeg'
    )
    evoked = mne.EvokedArray(
            data=ga_noise_data_padded,
            info=info,
            tmin=tmin
    )
    grand_average_noise_padded = evoked
    psd_noise = grand_average_noise_padded.compute_psd(
        method=method,
        fmin=fmin,
        fmax=fmax,
        verbose=False
    )

    # Визуализация PSD
    # 2. Преобразуем мощность в амплитуду: извлекаем квадратный корень
    data_psd_noise = psd_noise.get_data()  # получаем массив мкВ²/Гц

    data_noise_amplitude = np.sqrt(data_psd_noise).flatten()  # преобразуем в мкВ/√Гц
    freqs_noise = psd_noise.freqs

    # 3. Создаём новый объект PSD с амплитудными значениями

    ax.plot(freqs_data, data_amplitude, 'b-', label='Сигнал (data)', linewidth=1.5)
    ax.plot(freqs_noise, data_noise_amplitude, 'r-', label='Фоновый шум', linewidth=1.5)

    ax.set_xlabel('')
    ax.set_ylabel('Амплитуда, мкВ\/√Гц', fontsize=10)
    if dummy:
        ax.set_ylim(0.0, 0.09 * 1e-6)
        ax.set_yticks([0.0, 0.09 * 1e-6])
        ax.axhline(
            y=0.045 * 1e-6,
            color='purple',
            linestyle='--',
            linewidth=1,
            alpha=0.8,
            label='y=0.045 * 1e-6 '  # подпись линии
        )
        ax.legend()  # отображает легенду с подписью
    else:
        ax.set_ylim(0.0, 1.1 * 1e-6)
        ax.set_yticks([0.0, 1.1 * 1e-6])
    if fs == 50000.0:
        ax.set_ylim(0.0, 1.0 * 1e-6)
        ax.set_yticks([0.0, 1.0 * 1e-6])
    #ax.set_title(f'Спектральная амплитуда', fontsize=14)

    # Добавляем сетку и легенду
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

def plot_stim_PSD(stimulus, sinus_tone, frequencies, ax, method, fmin, fmax, fs):
    # FFT анализ — Power Spectral Density (PSD)
    if sinus_tone:
        data_stim = stimulus
    else:
        if fs == 10000:
            data_stim = stimulus[::4, 0]
        else:
            #data_stim = stimulus[:, 0]
            data_stim = stimulus

    # zero paddding для fft
    ga = False
    data_stim_padded = zero_padding(data_stim,  ga, padding_factor = 4)

    info = mne.create_info(
        ch_names=['Cz'],
        sfreq=fs,
        ch_types='eeg'
    )

    evoked_stim = mne.EvokedArray(
            data=np.transpose(data_stim_padded),
            info=info,
            tmin=0
        )
    psd = evoked_stim.compute_psd(
        method=method,
        fmin=fmin,
        fmax=fmax,
        verbose=False
    )

    # Визуализация PSD
    # 2. Преобразуем мощность в амплитуду: извлекаем квадратный корень
    data_psd = psd.get_data()  # получаем массив мкВ²/Гц
    data_amplitude = np.sqrt(data_psd).flatten()  # преобразуем в мкВ/√Гц
    freqs_data = psd.freqs

    ax.plot(freqs_data, data_amplitude, 'g-', linewidth=2.0)  # толщина линии увеличена до 2.0

    if sinus_tone:
        colors = ['magenta', 'orange', 'blue', 'green']
        for idx, frequency in enumerate(frequencies):
            ax.axvline(
                x=frequency,
                alpha=0.3,
                color=colors[idx % len(colors)],
                label=f'F{idx}: {frequency} Гц',
                linewidth=2.5  # толщина вертикальных линий увеличена до 2.5
            )

    # Настройка осей и заголовка
    ax.set_xlabel('Stimulus Spectra, Hz', fontsize=10, loc='left')
    # ax.set_xticks([])  # убираем деления (ticks) на оси X
    ax.set_ylabel('')
    ax.set_yticks([])  # убираем деления (ticks) на оси Y

    # Добавляем сетку и легенду
    ax.grid(True, alpha=0.3)

    # Настройка легенды с размером шрифта 10
    ax.legend(loc='best', fontsize=10)

    # Дополнительно: увеличиваем размер шрифта для подписей на осях (если они есть)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.show()

def save_signal_plot(signal, filename, frequency, stimulus_duration, inter_stimulus_interval, num_repetitions):
    """
    An auxiliary function plots the stimulus
    """

    # Устанавливаем глобальные параметры шрифтов
    plt.rcParams.update({
        'font.size': 20,  # Базовый размер шрифта
        'axes.titlesize': 20,  # Размер заголовков осей
        'axes.labelsize': 20,  # Размер подписей осей
        'xtick.labelsize': 20,  # Размер меток на оси X
        'ytick.labelsize': 20,  # Размер меток на оси Y
        'legend.fontsize': 20,  # Размер шрифта легенды
        'figure.titlesize': 22  # Размер общего заголовка (чуть больше)
    })

    # Plot for the left and right channels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(40, 12))

    # Left channel (with the stimulus)
    # Plot first 1 sec
    ax1.plot(signal[:44000, 0], color='blue', linewidth=1)
    ax1.set_title('Left channel')
    ax1.set_ylabel('Amplitude: stim=np.int16(stim * 32767) ')
    ax1.grid(True)
    ax1.set_ylim(-35000, 35000)  # Фиксированный масштаб для левого подграфика

    # Right channel (with the triggers)
    # Plot first 1 sec
    ax2.plot(signal[:44000, 1], color='red', linewidth=1)
    ax2.set_title('Right channel')
    ax2.set_xlabel('Samples (K)')
    ax2.grid(True)
    ax2.set_ylim(-35000, 35000)  # Фиксированный масштаб для правого подграфика

    fig.suptitle(
        f'{frequency} Hz, TS {stimulus_duration}ms, TP {inter_stimulus_interval}ms,{num_repetitions} repetitions',
        fontsize=22, fontweight='bold')

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def make_pause(inter_stimulus_interval, sample_rate, p):
    """
    The function makes an interstimulus interval of a varying length (+- 20% of TP)
    """
    percent = inter_stimulus_interval * p
    var = random.randint(-percent, percent)
    pause = inter_stimulus_interval + var

    pause_ms = pause / 1000
    n_samples = int(sample_rate * pause_ms)
    t_silence = np.linspace(0, pause_ms, n_samples, endpoint=False)
    return t_silence


def make_ramp_window(stimulus_duration, sample_rate, rate, growth_rate):
    """
    The function makes a ramp increase of the stimulus using exp and first 10% of the stim
    growth_rate: коэффициент скорости нарастания (больше = быстрее)
    """
    stim_ms = stimulus_duration / 1000
    n_samples = int(sample_rate * stim_ms)
    t_stim = np.linspace(0, stim_ms, n_samples, endpoint=False)

    ramp_duration_samples = int(len(t_stim) * rate)
    ramp_window = np.ones_like(t_stim)

    # Exponеntial increase of a stimulus, intense at the beggining
    x = np.linspace(0, 1, ramp_duration_samples)
    exp_ramp = (np.exp(growth_rate * x) - 1) / (np.exp(growth_rate) - 1)
    ramp_window[:ramp_duration_samples] = exp_ramp

    #  Exponеntial decay of a stimulus
    x = np.linspace(0, 1, ramp_duration_samples)
    exp_decay = (np.exp(growth_rate * (1 - x)) - 1) / (np.exp(growth_rate) - 1)
    ramp_window[-ramp_duration_samples:] = exp_decay
    return ramp_window, t_stim


def add_triggers(stimulus, sinus, inv, sample_rate):
    """
    Function to make 2 channels, inserts triggers at the start and at the end of the right channel.
    Returns signal with 2 channels - first for the stimuli and the second one  for the triggers
    add 3bit commands as in https://github.com/mcsltd/AStimWavPatcher/tree/master?tab=readme-ov-file
    """

    _SILENCE = 1

    if sinus:
        # int16 format
        stimulus = np.int16(stimulus * 32767)

    # Make 2 channels
    size = len(stimulus)
    left = stimulus.copy()

    right = np.zeros(size, dtype=np.int16)

    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min

    # Add triggers
    if inv:
        # 110 - set trigger 7 LOW (HIGH (default))
        right[_SILENCE + 0] = max_int16
        right[_SILENCE + 1] = min_int16
        right[_SILENCE + 2] = max_int16
        right[_SILENCE + 3] = min_int16
        right[_SILENCE + 4] = min_int16
        right[_SILENCE + 5] = max_int16

        # 111 - set trigger 7 HIGH (default)
        right[size - 6] = max_int16
        right[size - 5] = min_int16
        right[size - 4] = max_int16
        right[size - 3] = min_int16
        right[size - 2] = max_int16
        right[size - 1] = min_int16

    else:
        # 100 - set trigger 6 LOW (HIGH (default))
        right[_SILENCE + 0] = max_int16
        right[_SILENCE + 1] = min_int16
        right[_SILENCE + 2] = min_int16
        right[_SILENCE + 3] = max_int16
        right[_SILENCE + 4] = min_int16
        right[_SILENCE + 5] = max_int16

        # 101 - set trigger 6 HIGH (default)
        right[size - 6] = max_int16
        right[size - 5] = min_int16
        right[size - 4] = min_int16
        right[size - 3] = max_int16
        right[size - 2] = max_int16
        right[size - 1] = min_int16

    return np.column_stack([left, right])


def make_inv_stimulus(stimulus):
    return (-1) * stimulus


def trim_stim(stimulus, stimulus_duration, sample_rate):
    required_length = int((stimulus_duration / 1000) * sample_rate)
    return stimulus[:required_length]