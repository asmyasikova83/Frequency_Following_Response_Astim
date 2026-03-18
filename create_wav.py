import os
import argparse
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import random

# to add 3bit commands as in https://github.com/mcsltd/AStimWavPatcher/tree/master?tab=readme-ov-file


def save_signal_plot(signal, filename, frequency, stimulus_duration, inter_stimulus_interval, num_repetitions):
    """Вспомогательная функция для сохранения графика сигнала"""


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

    # График для левого и правого каналов
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(40, 12))

    # Левый канал
    ax1.plot(signal[:20000, 0], color='blue', linewidth=1)
    ax1.set_title('Левый канал')
    ax1.set_ylabel('Амплитуда 75%: stim=np.int16(stim * 32767) ')
    ax1.grid(True)
    ax1.set_ylim(-35000, 35000)  # Фиксированный масштаб для левого подграфика

    # Правый канал
    ax2.plot(signal[17500:37500, 1], color='red', linewidth=1)
    ax2.set_title('Правый канал')
    ax2.set_xlabel('Отсчёты (K)')
    ax2.grid(True)
    ax2.set_ylim(-35000, 35000)  # Фиксированный масштаб для правого подграфика

    # Общий заголовок с увеличенным шрифтом
    fig.suptitle(f'{frequency} Гц, TS {stimulus_duration}ms, TP {inter_stimulus_interval}ms,{num_repetitions} повторений', fontsize=22, fontweight='bold')

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def make_ramp_window(t_stim, growth_rate):
    """
    Вспо могательная функция с экспоненциальным нарастанием
    growth_rate: коэффициент скорости нарастания (больше = быстрее)
    """
    ramp_duration_samples = int(len(t_stim) * 0.1)
    ramp_window = np.ones_like(t_stim)

    # Экспоненциальное нарастание: очень быстрое в начале
    x = np.linspace(0, 1, ramp_duration_samples)
    exp_ramp = (np.exp(growth_rate * x) - 1) / (np.exp(growth_rate) - 1)
    ramp_window[:ramp_duration_samples] = exp_ramp

    # Экспоненциальное затухание
    x = np.linspace(0, 1, ramp_duration_samples)
    exp_decay = (np.exp(growth_rate * (1 - x)) - 1) / (np.exp(growth_rate) - 1)
    ramp_window[-ramp_duration_samples:] = exp_decay
    return ramp_window

def add_triggers(stimulus,  inv, trigger_delay, sample_rate):

    """Создает и заполняет 2 канала, вставляет метки в начало и конец стимула в правом канале,
     возвращает 2канальный сигнал"""

    _SILENCE = 1

    # zero padding for 3bit commands to enable triggers
    stimulus = np.append(np.zeros(_SILENCE), stimulus)
    # int16 format
    stimulus = np.int16(stimulus * 32767)
    # make 2 channels
    left = stimulus.copy()

    size = len(stimulus)
    right = np.zeros(size, dtype = np.int16)

    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min

    # Итеративное добавление триггеров для каждого стимула

    trigger_delay = (trigger_delay / 1000) * sample_rate
    trigger_delay = int(trigger_delay)

    # Add triggers
    if inv:
        # 110 - set trigger 7 LOW (HIGH (default))
        right[_SILENCE + trigger_delay + 0] = max_int16
        right[_SILENCE + trigger_delay + 1] = min_int16
        right[_SILENCE + trigger_delay + 2] = max_int16
        right[_SILENCE + trigger_delay + 3] = min_int16
        right[_SILENCE + trigger_delay + 4] = min_int16
        right[_SILENCE + trigger_delay + 5] = max_int16

        # 111 - set trigger 7 HIGH (default)
        right[size - 6] = max_int16
        right[size - 5] = min_int16
        right[size - 4] = max_int16
        right[size - 3] = min_int16
        right[size - 2] = max_int16
        right[size - 1] = min_int16

    else:
        # 100 - set trigger 6 LOW (HIGH (default))
        right[_SILENCE + trigger_delay + 0] = max_int16
        right[_SILENCE + trigger_delay + 1] = min_int16
        right[_SILENCE + trigger_delay + 2] = min_int16
        right[_SILENCE + trigger_delay + 3] = max_int16
        right[_SILENCE + trigger_delay + 4] = min_int16
        right[_SILENCE + trigger_delay + 5] = max_int16

        # 101 - set trigger 6 HIGH (default)
        right[size - 6] = max_int16
        right[size - 5] = min_int16
        right[size - 4] = min_int16
        right[size - 3] = max_int16
        right[size - 2] = max_int16
        right[size - 1] = min_int16

    return  np.column_stack([left, right])

def make_inv_stimulus(stimulus):
    return (-1) * stimulus

def create_repeated_sinusoidal_wav(
        dir,
        frequency,
        stimulus_duration,
        inter_stimulus_interval,
        amplitude,
        num_repetitions,
        trigger_delay,
        sample_rate
):
    """
    Создаёт WAV‑файл с повторяющимися синусоидальными тонами.

    Параметры:
    - dirname: наименование директории для сохранения wav файла;
    - frequency: частота тона в Гц;
    - stimulus_duration: длительность одного стимула в секундах;
    - inter_stimulus_interval: межстимульный интервал в секундах;
    - num_repetitions: количество повторений;
    - sample_rate: частота дискретизации;
    - amplitude: амплитуда сигнала.
    """

    # Создаём один стимул

    # into ms
    n_samples = int(sample_rate * stimulus_duration/ 1000) # или любое другое число отсчётов
    t_stim = np.arange(n_samples) / sample_rate
    # Окно нарастания/затухания
    ramp_window = make_ramp_window(t_stim, growth_rate = 3.0)
    # Синусоидальный сигнал и inv sin
    stimulus = (amplitude / 100) * ramp_window * np.sin(2 * np.pi * frequency * t_stim)
    inv_stimulus = make_inv_stimulus(stimulus)

    # Создаём паузу
    n_samples = int(sample_rate * inter_stimulus_interval / 1000)  # или любое другое число отсчётов
    t_silence = np.arange(n_samples) / sample_rate
    silence = np.zeros_like(t_silence)
    isi = np.column_stack([silence , silence ])
    isi = np.int16(isi * 32767)

    # Собираем полный сигнал: стимул + pause + inv stimulus + pause , повторяем нужное число раз
    full_signal = []
    all_stimuli = []

    # Создаём список всех стимулов (оригинальные и инвертированные)
    for _ in range(num_repetitions // 2):
        inv = False
        stim_triggers = add_triggers(stimulus, inv, trigger_delay, sample_rate)
        all_stimuli.append(stim_triggers)

        inv = True
        inv_stim_triggers = add_triggers(inv_stimulus, inv, trigger_delay, sample_rate)
        all_stimuli.append(inv_stim_triggers)

    # Перемешиваем все стимулы в случайном порядке
    random.shuffle(all_stimuli)

    # Добавляем в сигнал: стимул → пауза
    for stim in all_stimuli:
        full_signal.append(stim)
        full_signal.append(isi)

    # Объединяем все части в один массив
    full_signal = np.concatenate(full_signal)

    # Формируем имена файлов
    base_name = f'sin_{int(frequency)}Hz_TS{stimulus_duration:.1f}s_TP{inter_stimulus_interval:.1f}s_N{num_repetitions}_A{amplitude:.1f}%_TR0{trigger_delay:.1f}_2_chs'
    wav_filename = f'{base_name}.wav'
    png_filename = f'{base_name}.png'

    wav_path = os.path.join(dir, wav_filename)
    png_path = os.path.join(dir, png_filename)

    # Создаём и сохраняем график
    save_signal_plot(full_signal, png_path, frequency, stimulus_duration, inter_stimulus_interval, num_repetitions)

    #  сохраняем WAV
    write(wav_path, sample_rate, full_signal)
    print(f"WAV‑файл успешно создан: {wav_path}")

    print(f"График сигнала сохранён: {png_path}")


def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='Создание WAV‑файла с повторяющимися синусоидальными тонами',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Пример использования:
  python create_wav.py --F 800 --TS 0.1 --TP 0.3 --N 100
        """
    )

    parser.add_argument(
        '--F',
        type=float,
        required=True,
        help='Частота тона в Гц (например, 800)'
    )
    parser.add_argument(
        '--TS',
        type=float,
        required=True,
        help='Длительность одного стимула в секундах (например, 100 ms)'
    )
    parser.add_argument(
        '--TP',
        type=float,
        required=True,
        help='Межстимульный интервал в секундах (например, 300 ms)'
    )
    parser.add_argument(
        '--A',
        type=float,
        default=100,
        help='Амплитуда сигнала (до 100%, по умолчанию: 75%)'
    )
    parser.add_argument(
        '--N',
        type=int,
        required=True,
        help='Количество повторений стимула (например, 5)'
    )
    parser.add_argument(
        '--TR0',
        type=float,
        default=0,
        help='Задержка триггера (по умолчанию: 0 ms)'
    )
    parser.add_argument(
        '--SR',
        type=int,
        default=44100,
        help='Частота дискретизации в Гц (по умолчанию: 44100)'
    )
    parser.add_argument(
        '--dirname',
        type=str,
        default='M:\\DB Temp\\physionet.org\\files\\ffr_astim',
        help='Название директории для записи wav файла (по умолчанию: M:\\DB Temp\\physionet.org\\files\\ffr_astim)'
    )

    return parser.parse_args()

if __name__ == '__main__':
    # Парсим аргументы командной строки
    args = parse_arguments()

    # Выводим используемые параметры
    print("Используемые параметры:")
    print(f"  Частота: {args.F} Гц")
    print(f"  Длительность стимула: {args.TS} ms")
    print(f"  Длительность паузы: {args.TP} ms")
    print(f"  Задержка триггера: {args.TR0} ms")
    print(f"  Количество повторений: {args.N}")
    print(f"  Амплитуда: {args.A}")
    print(f"  Выходная директория: {args.dirname}")


    # Создаём WAV‑файл
    create_repeated_sinusoidal_wav(
        dir=args.dirname,
        frequency=args.F,
        stimulus_duration=args.TS,
        inter_stimulus_interval=args.TP,
        amplitude=args.A,
        num_repetitions=args.N,
        trigger_delay=args.TR0,
        sample_rate=args.SR
    )
