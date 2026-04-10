import numpy as np
import os
import mne
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy import stats
from scipy.signal import butter, sosfiltfilt
from scipy.signal import firwin, filtfilt
import warnings
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    module='mne.*'  # или 'pandas.*' и т. д.
)


def import_raw(fname, non_filt, preamplifier, dummy, fmin, fmax, order):
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
    filtered_signal = fir_bandpass_filter(raw_selected.get_data(), fmin, fmax,  int(raw.info.get('sfreq')), order)
    raw_to_epo = mne.io.RawArray(filtered_signal, raw_selected.info)
    events, event_dict = mne.events_from_annotations(raw)

    return raw, raw_to_epo, events, event_dict, label_6, label_7

def select_events(raw_to_epo,n_6low, n_7low, label_6, label_7, events, event_dict):

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
    return epochs

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

def plot_GA(grand_avg, output_path, snr_result, title):
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

    ax.axvline(
        x=0.25,  # координата X (время = 0)
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
    fig1.suptitle(title)
    fig1.savefig(
    output_path,
    dpi=300,                  # Высокое разрешение
    bbox_inches='tight',     # Убирает лишние поля
    facecolor='white' ,       # Белый фон
    edgecolor='none'         # Без рамки
    )
    plt.show()
    print(f"Рисунок сохранён: {output_path}")


def bw_taps_lpf(cutoff):
    """Низкочастотный фильтр Баттерворта 2‑го порядка."""
    pi = np.pi
    sqr = np.sqrt(2)
    b = np.tan(cutoff * pi)
    b2 = b ** 2
    s = 1 + sqr * b + b2

    b0 = b2 / s
    return [b0, 2 * b0, b0, 1.0, 2 * (b2 - 1) / s, (1 - sqr * b + b2) / s]

def bw_taps_hpf(cutoff):
    """Высокочастотный фильтр Баттерворта 2‑го порядка."""
    pi = np.pi
    b = np.tan(cutoff * pi)
    s = 1 + b

    b0 = b / s
    a1 = (1 - b) / s
    # Возвращаем [b0, b1, 0, a0=1, a1, 0] — b2=0, a2=0
    return [b0, -b0, 0.0, 1.0, a1, 0.0]

def create_bw_sos(lowcut, highcut, fs):
    """Создаёт SOS‑матрицу для полосового фильтра Баттерворта."""
    nyq = fs * 0.5

    # Проверка корректности частот среза
    if lowcut >= highcut:
        raise ValueError("lowcut должен быть меньше highcut")
    if lowcut <= 0 or highcut >= nyq:
        raise ValueError(f"Частоты среза должны быть в диапазоне (0, {nyq} Гц)")

    # Нормализованные частоты
    low_norm = lowcut / nyq
    high_norm = highcut / nyq

    # Получаем коэффициенты
    hpf_coeffs = bw_taps_hpf(low_norm)
    lpf_coeffs = bw_taps_lpf(high_norm)

    # Формируем SOS‑матрицу: 2 секции, 6 коэффициентов каждая
    sos = np.zeros((2, 6))

    # Каскад HPF
    sos[0, :] = hpf_coeffs
    # Каскад LPF
    sos[1, :] = lpf_coeffs

    return sos

def bandpass_butterworth_zero_phase(data, lowcut, highcut, fs):
    """
    Полосовой фильтр Баттерворта с нулевой фазой.
    Обрабатывает одно‑ и многоканальные данные.
    """
    # Преобразуем входные данные в корректный формат
    data = np.asarray(data, dtype=np.float64)

    sos = create_bw_sos(lowcut, highcut, fs)
    filtered_data = sosfiltfilt(sos, data)

    return filtered_data


def butter_zero_phase(data, low_cutoff, high_cutoff, order, fs):
    """
    Полосовой фильтр Баттерворта с нулевой фазой.
    Обрабатывает одно‑ и многоканальные данные.
    """
    # Нормализуем частоты относительно частоты НайквистаDZVG
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist

    # Создаём полосовой фильтр Баттерворта
    b, a = butter(order, [low, high], btype='band', analog=False)

    # Применяем фильтрацию с нулевой фазой
    filtered_signal = filtfilt(b, a, data, axis=-1)
    return filtered_signal

def fir_bandpass_filter(data, low_cutoff, high_cutoff, fs, order):
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

    # Создаём полосовой FIR-фильтр с помощью firwin
    b = firwin(
        numtaps=order + 1,  # numtaps = порядок + 1
        cutoff=[low, high],
        pass_zero=False,
        window='hamming'  # окно Хэмминга для снижения пульсаций
    )

    # Коэффициенты знаменателя для FIR-фильтра: a = 1
    a = 1.0

    # Применяем фильтрацию с нулевой фазой (двукратная фильтрация вперёд и назад)
    # axis=-1 — фильтрация вдоль последней оси (по времени)
    filtered_signal = filtfilt(b, a, data, axis=-1)

    return filtered_signal

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

    if preamplifier:
        rms_prestim = 1000 * rms_prestim
        rms_poststim = 1000 * rms_poststim
    else:
        rms_prestim = 1000000 * rms_prestim
        rms_poststim = 1000000 * rms_poststim

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

def plot_PSD(grand_average, sumeve, method, fmin, fmax, output_dir, subject):
    # FFT анализ — Power Spectral Density (PSD)
    psd = grand_average.compute_psd(
        method=method,
        fmin=fmin,  # минимальная частота: 40 Гц
        fmax=fmax,  # максимальная частота: 250 Гц
        verbose=False
    )

    # Визуализация PSD
    # 2. Преобразуем мощность в амплитуду: извлекаем квадратный корень
    data_psd = psd.get_data()  # получаем массив мкВ²/Гц
    data_amplitude = np.sqrt(data_psd).flatten()  # преобразуем в мкВ/√Гц

    # 3. Создаём новый объект PSD с амплитудными значениями
    # (MNE не поддерживает это напрямую, поэтому строим график «вручную»)
    freqs = psd.freqs

    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freqs, data_amplitude)
    ax.set_xlabel('Частота, Гц')
    ax.set_ylabel('Амплитуда, мкВ/√Гц')
    ax.set_title(f'Спектральная амплитуда — FFT {subject} Grand Average (n={sumeve})', fontsize=14)
    ax.grid(True)
    ax.legend()

    # Теперь fig2 — это корректный график амплитуды
    # Можно сохранить: fig2.savefig('psd_amplitude.png')

    output_path = os.path.join(output_dir, f'FFR_{sumeve}_PSD_{subject}.png')

    # Выделяем диапазон частот стимула (например, 103–125 Гц)
    #https: // pubmed.ncbi.nlm.nih.gov / 20084007 /
    ax.axvspan(103, 125, alpha=0.3, color='magenta', label='F0: 103–125 Гц')

    # Добавляем легенду
    ax.legend()
    fig2.savefig(
        output_path,
        dpi=300,  # Высокое разрешение
        bbox_inches='tight',  # Убирает лишние поля
        facecolor='white',  # Белый фон
        edgecolor='none'  # Без рамки
    )

    plt.show()


def plot_noise_PSD(grand_average, grand_average_noise,  sumeve, method, fmin, fmax, output_dir, subject):
    # FFT анализ — Power Spectral Density (PSD)
    psd = grand_average.compute_psd(
        method=method,
        fmin=fmin,  # минимальная частота: 40 Гц
        fmax=fmax,  # максимальная частота: 250 Гц
        verbose=False
    )

    # Визуализация PSD
    # 2. Преобразуем мощность в амплитуду: извлекаем квадратный корень
    data_psd = psd.get_data()  # получаем массив мкВ²/Гц
    data_amplitude = np.sqrt(data_psd).flatten()  # преобразуем в мкВ/√Гц
    freqs_data = psd.freqs

    psd_noise = grand_average_noise.compute_psd(
        method=method,
        fmin=fmin,  # минимальная частота: 40 Гц
        fmax=fmax,  # максимальная частота: 250 Гц
        verbose=False
    )
    # Визуализация PSD
    # 2. Преобразуем мощность в амплитуду: извлекаем квадратный корень
    data_psd_noise = psd_noise.get_data()  # получаем массив мкВ²/Гц

    data_noise_amplitude = np.sqrt(data_psd_noise).flatten()  # преобразуем в мкВ/√Гц
    freqs_noise = psd_noise.freqs

    # 3. Создаём новый объект PSD с амплитудными значениями
    # (MNE не поддерживает это напрямую, поэтому строим график «вручную»)


    # Создаём фигуру и оси
    fig, ax = plt.subplots(figsize=(10, 6))

    # Строим обе линии на одном графике
    ax.plot(freqs_data, data_amplitude, 'b-', label='Сигнал (data)', linewidth=1.5)
    ax.plot(freqs_noise, data_noise_amplitude, 'r-', label='Шум (noise)', linewidth=1.5)

    # Настройка осей и заголовка
    ax.set_xlabel('Частота, Гц')
    ax.set_ylabel('Амплитуда, мкВ/√Гц')
    ax.set_title(f'Спектральная амплитуда — Сигнал vs Шум\n{subject} {fmin}-{fmax} Гц Grand Average (n={sumeve})', fontsize=14)

    # Добавляем сетку и легенду
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Автоматически подбираем отступы
    plt.tight_layout()

    # Сохраняем график
    output_path = os.path.join(output_dir, f'FFR_sig_noise_{fmin}-{fmax}_Hz{sumeve}_PSD_{subject}.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')

    # Показываем график (опционально)
    plt.show()

    # Выделяем диапазон частот стимула (например, 103–125 Гц)
    #https: // pubmed.ncbi.nlm.nih.gov / 20084007 /
    #ax.axvspan(103, 125, alpha=0.3, color='magenta', label='F0: 103–125 Гц')

    plt.show()

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