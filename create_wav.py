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
        ramp_duration,
        inter_stimulus_interval,
        num_repetitions,
        sample_rate=44100,
        amplitude=0.5
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
    t_stim = np.linspace(0, stimulus_duration, int(sample_rate * stimulus_duration), endpoint=False)

    # Создаём окно нарастания (линейное)
    ramp_samples = int(sample_rate * ramp_duration)
    ramp_window = np.ones_like(t_stim)

    # Линейное нарастание от 0 до amplitude в течение ramp_duration
    ramp_window[:ramp_samples] = np.linspace(0, amplitude, ramp_samples)

    # Синусоидальный сигнал с нарастанием
    stimulus = amplitude * ramp_window * np.sin(2 * np.pi * frequency * t_stim)
    # y(t)=A⋅sin(2πft+φ)
    #stimulus = amplitude * np.sin(2 * np.pi * frequency * t_stim)

    # Создаём паузу
    t_silence = np.linspace(0, inter_stimulus_interval, int(sample_rate * inter_stimulus_interval), endpoint=False)
    silence = np.zeros_like(t_silence)

    # Собираем полный сигнал: стимул + пауза, повторяем нужное число раз
    full_signal = []
    for _ in range(num_repetitions):
        full_signal.append(stimulus)
        full_signal.append(silence)

    # Объединяем все части в один массив
    full_signal = np.concatenate(full_signal)

    # Формируем имена файлов
    base_name = f'sin_{int(frequency)}Hz_{stimulus_duration:.1f}s_isi{inter_stimulus_interval:.1f}s_{num_repetitions}reps_sf{sample_rate}Hz_A{amplitude:.1f}'
    wav_filename = f'{base_name}.wav'
    png_filename = f'{base_name}.png'

    wav_path = os.path.join(dir, wav_filename)
    png_path = os.path.join(dir, png_filename)

    # Создаём и сохраняем график
    save_signal_plot(full_signal, png_path, frequency, num_repetitions)

    # Преобразуем в 16‑битный целочисленный формат и сохраняем WAV
    audio = np.int16(full_signal * 32767)
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
  python create_wav.py --frequency 800 --num_repetitions 5  --stimulus_duration 0.1 --inter_stimulus_interval 0.3
        """
    )

    parser.add_argument(
        '--frequency',
        type=float,
        required=True,
        help='Частота тона в Гц (например, 800)'
    )
    parser.add_argument(
        '--num_repetitions',
        type=int,
        required=True,
        help='Количество повторений стимула (например, 5)'
    )
    parser.add_argument(
        '--stimulus_duration',
        type=float,
        required=True,
        help='Длительность одного стимула в секундах (например, 0.1)'
    )
    parser.add_argument(
        '--inter_stimulus_interval',
        type=float,
        required=True,
        help='Межстимульный интервал в секундах (например, 0.3)'
    )
    parser.add_argument(
        '--ramp_duration',
        type=float,
        default=None,
        help='Длительность нарастания в секундах (по умолчанию: половина stimulus_duration)'
    )
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=44100,
        help='Частота дискретизации в Гц (по умолчанию: 44100)'
    )
    parser.add_argument(
        '--amplitude',
        type=float,
        default=0.5,
        help='Амплитуда сигнала (0.0–1.0, по умолчанию: 0.5)'
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

    # Устанавливаем ramp_duration по умолчанию, если не задано
    if args.ramp_duration is None:
        args.ramp_duration = args.stimulus_duration / 2

    # Выводим используемые параметры
    print("Используемые параметры:")
    print(f"  Частота: {args.frequency} Гц")
    print(f"  Длительность стимула: {args.stimulus_duration} с")
    print(f"  Длительность нарастания: {args.ramp_duration} с")
    print(f"  Межстимульный интервал: {args.inter_stimulus_interval} с")
    print(f"  Количество повторений: {args.num_repetitions}")
    print(f"  Частота дискретизации: {args.sample_rate} Гц")
    print(f"  Амплитуда: {args.amplitude}")
    print(f"  Выходная директория: {args.dirname}")

    # Создаём WAV‑файл
    create_repeated_sinusoidal_wav(
        dir=args.dirname,
        frequency=args.frequency,
        stimulus_duration=args.stimulus_duration,
        ramp_duration=args.ramp_duration,
        inter_stimulus_interval=args.inter_stimulus_interval,
        num_repetitions=args.num_repetitions,
        sample_rate=args.sample_rate,
        amplitude=args.amplitude
    )
