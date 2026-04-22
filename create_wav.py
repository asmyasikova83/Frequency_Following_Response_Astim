import os
import argparse
import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import write
import random
import matplotlib.pyplot as plt
from functions import (trim_stim, make_inv_stimulus, add_triggers,
                       make_ramp_window, make_pause, save_signal_plot,
                       plot_stim_PSD)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def create_multiple_sin_wav(
        dir,
        frequencies,
        stimulus_duration,
        inter_stimulus_interval,
        num_repetitions,
        sample_rate,
        add_inv

):
    """
    Creates WAV file with sin tones with predefined frequencies.

    Arguments:
    - dirname: directory for saving WAV file;
    - frequency: frequency of sin tone in Hz;
    - stimulus_duration: the length of 1 stimulus in ms;
    - inter_stimulus_interval: the length of ISI in ms;
    - num_repetitions: number of stimuli in WAV;
    - sample_rate: sampling rate 44100 Hz;
    - amplitude: amplitude of a simulus.
    """

    # Create one stimulus
    # Make a ramp window
    ramp_window, t_stim = make_ramp_window(stimulus_duration, sample_rate, rate = 0.1 , growth_rate = 3.0)

    # Sin +  inv sin
    sinus = np.zeros_like(t_stim)
    amplitude_factor = [1.0, 0.7, 0.5, 0.3]
    for i in range(len(frequencies)):
        stimulus = amplitude_factor[i]  * ramp_window * np.sin(2 * np.pi * frequencies[i] * t_stim)
        sinus = sinus + stimulus
    sinus /= np.max(np.abs(sinus))
    if add_inv:
        inv_sinus = make_inv_stimulus(sinus)

    plot_stim_psd = True
    if plot_stim_psd:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        # Choose fmin and fmax for spectra visualization
        plot_stim_PSD(sinus, True, frequencies, axes, 'multitaper', 30, 1500, 44100)

    # Make a varying interstimulus interval
    t_silence = make_pause(inter_stimulus_interval, sample_rate, p=0.2)
    silence = np.zeros_like(t_silence)
    isi = np.column_stack([silence , silence])
    isi = np.int16(isi * 32767)

    # Собираем полный сигнал: стимул + pause + inv stimulus + pause , повторяем нужное число раз
    full_signal = []
    all_stimuli = []

    # Создаём список всех стимулов (оригинальные и инвертированные)
    sin = True
    if add_inv:
        for _ in range(num_repetitions // 2):
            inv = False
            stim_triggers = add_triggers(sinus, sin, inv, sample_rate)
            all_stimuli.append(stim_triggers)

            inv = True
            inv_stim_triggers = add_triggers(inv_sinus, sin, inv, sample_rate)
            all_stimuli.append(inv_stim_triggers)
    else:
        assert(add_inv == 0)
        for _ in range(num_repetitions):
            inv = False
            stim_triggers = add_triggers(sinus,  sin, inv, sample_rate)
            all_stimuli.append(stim_triggers)

    # Перемешиваем все стимулы в случайном порядке
    random.shuffle(all_stimuli)

    # Добавляем в сигнал: стимул → пауза
    for stim in all_stimuli:
        full_signal.append(stim)
        full_signal.append(isi)

    # Объединяем все части в один массив
    full_signal = np.concatenate(full_signal)

    # Формируем имена файлов
    base_name = f'sin_{frequencies}Hz_TS{stimulus_duration:.1f}s_TP{inter_stimulus_interval:.1f}s_N{num_repetitions}_INV{add_inv}'
    wav_filename = f'{base_name}.wav'
    png_filename = f'{base_name}.png'

    wav_path = os.path.join(dir, wav_filename)
    png_path = os.path.join(dir, png_filename)

    # Создаём и сохраняем график
    save_signal_plot(full_signal, png_path, frequencies, stimulus_duration, inter_stimulus_interval, num_repetitions)

    #  сохраняем WAV
    write(wav_path, sample_rate, full_signal)
    print(f"WAV‑файл успешно создан: {wav_path}")

    print(f"График сигнала сохранён: {png_path}")

def create_repeated_da_syllable_wav(
        dir,
        frequencies,
        stimulus_duration,
        inter_stimulus_interval,
        num_repetitions,
        sample_rate,
        add_inv

):
    """
    Creates WAV‑файл with syllables.
    """

    # Create one stimulus
    if stimulus_duration > 199:
        long = True
    else:
        long = True
    if long:
        fs, stimulus = wavfile.read(r'\\MCSSERVER\DB Temp\physionet.org\FFR\stim\Da_syll_250ms.wav')
    else:
        fs, stimulus = wavfile.read(r'\\MCSSERVER\DB Temp\physionet.org\FFR\stim\short\Da_syll_140ms.wav')

    stimulus = trim_stim(stimulus, stimulus_duration, sample_rate)

    sin_tone = False
    plot_PSD = True
    if plot_PSD:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        plot_stim_PSD(stimulus, sin_tone,frequencies,  axes, 'multitaper', 30, 800,fs)

    if add_inv:
        inv_stimulus = make_inv_stimulus(stimulus)

    # Make a full stimulation: stimulus + pause + inv stimulus + pause , N repetitions
    full_signal = []
    all_stimuli = []

    ramp_window, t_stim = make_ramp_window(stimulus_duration, sample_rate, rate=0.1, growth_rate=3.0)
    sin  = False
    if add_inv:
        for _ in range(num_repetitions // 2):
            inv = False
            stim_triggers = add_triggers(stimulus * ramp_window, sin, inv, sample_rate)
            all_stimuli.append(stim_triggers)

            inv = True
            inv_stim_triggers = add_triggers(inv_stimulus * ramp_window, sin, inv, sample_rate)
            all_stimuli.append(inv_stim_triggers)

    else:
        assert(add_inv == 0)
        for _ in range(num_repetitions):
            inv = False
            stim_triggers = add_triggers(stimulus, sin, inv, sample_rate)
            all_stimuli.append(stim_triggers)

    random.shuffle(all_stimuli)
    for stim in all_stimuli:
        full_signal.append(stim)
        # Add a pause with a varying length
        t_silence = make_pause(inter_stimulus_interval, sample_rate, p=0.2)
        silence = np.zeros_like(t_silence)
        isi = np.column_stack([silence, silence])
        isi = np.int16(isi * 32767)
        full_signal.append(isi)

    # Объединяем все части в один массив
    full_signal = np.concatenate(full_signal)

    base_name = f'Da_syll_TS{stimulus_duration}ms_TP{inter_stimulus_interval}ms_N{num_repetitions}_INV{add_inv}'
    wav_filename = f'{base_name}.wav'
    png_filename = f'{base_name}.png'

    wav_path = os.path.join(dir, wav_filename)
    png_path = os.path.join(dir, png_filename)

    # Создаём и сохраняем график
    save_signal_plot(full_signal, png_path, frequencies, stimulus_duration, inter_stimulus_interval, num_repetitions)

    #  сохраняем WAV
    write(wav_path, sample_rate, full_signal)
    print(f"WAV‑файл успешно создан: {wav_path}")

    print(f"График сигнала сохранён: {png_path}")


def main():
    parser = argparse.ArgumentParser(description='Audio stimuli generation')
    parser.add_argument('--function', '-f', required=True,
                            choices=['repeated_da', 'multiple_sin'],
                            help='Function to generate stimuli: repeated_da или multiple_sin')
    parser.add_argument('--dirname', type=str, default='M:\\DB Temp\\physionet.org\\files\\ffr_astim',
                        help='Директория для сохранения  wav файла (по умолчанию: M:\\DB Temp\\physionet.org\\files\\ffr_astim)')
    parser.add_argument('--F', type=int, nargs='+', default=[], help='Частоты (Гц)')
    parser.add_argument('--TS', type=float, required=True, help='Length of the stimulus (ms)')
    parser.add_argument('--TP', type=float, required=True, help='Interstimulus interval (ms)')
    parser.add_argument('--N', type=int, required=True, help='Number of repetitions of the stimulus')
    parser.add_argument('--SR', type=int, default=44100, help='Sampling rate default: 44100 Hz')
    parser.add_argument('--INV', type=int, required=True, help='Add an inverted (polar) stimuli')
    """
    Example call:  python create_wav.py  --function multiple_sin --F 440 880 --TS 100 --TP 100 --N 1 --INV 0
    """
    args = parser.parse_args()

    if args.function == 'repeated_da':
        create_repeated_da_syllable_wav(
                dir=args.dirname,
                frequencies=args.F,
                stimulus_duration=args.TS,
                inter_stimulus_interval=args.TP,
                num_repetitions=args.N,
                sample_rate=args.SR,
                add_inv=args.INV
        )
    elif args.function == 'multiple_sin':
            create_multiple_sin_wav(
                dir=args.dirname,
                frequencies=args.F,
                stimulus_duration=args.TS,
                inter_stimulus_interval=args.TP,
                num_repetitions=args.N,
                sample_rate=args.SR,
                add_inv=args.INV
        )


if __name__ == '__main__':
        main()