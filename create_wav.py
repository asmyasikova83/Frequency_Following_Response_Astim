import os
import argparse
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt


def save_signal_plot(signal, filename, frequency, num_repetitions):
    """Вспомогательная функция для сохранения графика сигнала"""
    plt.figure(figsize=(12, 4))
    plt.plot(signal)
    plt.title(f'Сигнал: {frequency} Гц, {num_repetitions} повторений')
    plt.xlabel('Отсчёты')
    plt.ylabel('Амплитуда')
    plt.grid(True)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

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
    # Into ms
    stimulus_duration = stimulus_duration / 100
    t_stim = np.linspace(0, stimulus_duration, int(sample_rate * stimulus_duration), endpoint=False)

    # Рассчитываем количество отсчётов для 5 % длительности стимула
    ramp_duration_samples = int(len(t_stim) * 0.1)

    # Создаём окно нарастания и затухания
    ramp_window = np.ones_like(t_stim)

    # Линейное нарастание от 0 до 1 в течение первых 10 %
    ramp_window[:ramp_duration_samples] = np.linspace(0, 1, ramp_duration_samples)

    # Линейное затухание от 1 до 0 в течение последних 5 %
    ramp_window[-ramp_duration_samples:] = np.linspace(1, 0, ramp_duration_samples)

    # Синусоидальный сигнал с плавным нарастанием и затуханием
    stimulus = (amplitude / 100) * ramp_window * np.sin(2 * np.pi * frequency * t_stim)

    # y(t)=A⋅sin(2πft+φ)
    #stimulus = amplitude * np.sin(2 * np.pi * frequency * t_stim)

    # Создаём паузу
    # Into ms
    inter_stimulus_interval = inter_stimulus_interval / 100
    t_silence = np.linspace(0, inter_stimulus_interval, int(sample_rate * inter_stimulus_interval), endpoint=False)
    silence = np.zeros_like(t_silence)
git
    # Собираем полный сигнал: стимул + пауза, повторяем нужное число раз
    full_signal = []
    for _ in range(num_repetitions):
        full_signal.append(stimulus)
        full_signal.append(silence)

    # Объединяем все части в один массив
    full_signal = np.concatenate(full_signal)

    # Формируем имена файлов
    base_name = f'sin_{int(frequency)}Hz_TS{stimulus_duration:.1f}s_TP{inter_stimulus_interval:.1f}s_N{num_repetitions}_A{amplitude:.1f}%'
    wav_filename = f'{base_name}.wav'
    png_filename = f'{base_name}.png'

    wav_path = os.path.join(dir, wav_filename)
    png_path = os.path.join(dir, png_filename)

    # Создаём и сохраняем график
    save_signal_plot(full_signal, png_path, frequency, num_repetitions)

    # Преобразуем в 16‑битный целочисленный формат и сохраняем WAV
    audio = np.int16(full_signal * 32767)
    #audio = full_signal
    write(wav_path, sample_rate, audio)
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
        default=50,
        help='Амплитуда сигнала (до 100%, по умолчанию: 50%)'
    )
    parser.add_argument(
        '--N',
        type=int,
        required=True,
        help='Количество повторений стимула (например, 5)'
    )
    parser.add_argument(
        '--TR',
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
    print(f"  Задержка триггера: {args.TR} ms")
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
        trigger_delay=args.TR,
        sample_rate=args.SR
    )
