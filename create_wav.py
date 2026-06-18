import os
import argparse
import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import write
import random
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from functions import (trim_stim, make_inv_stimulus, add_triggers,
                       make_ramp_window, make_pause, save_signal_plot,
                       plot_stim_PSD, make_full_signal)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def create_multiple_sin_wav(
        dir,
        frequencies,
        stimulus_duration,
        inter_stimulus_interval,
        num_repetitions,
        sample_rate,
        add_inv,
        A

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
    sinus = A * sinus

    if add_inv:
        inv_sinus = make_inv_stimulus(sinus)

    sin_tone = True
    plot_stim_psd = True
    if plot_stim_psd:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        # Choose fmin and fmax for spectra visualization
        plot_stim_PSD(sinus, sin_tone, frequencies, axes, 'multitaper', 30, 1500, 32)


    # Собираем полный сигнал: стимул + pause + inv stimulus + pause , повторяем нужное число раз
    all_stimuli = []

    #Reset ASTIM: false cycle
    inv = False
    _ = add_triggers(sinus, sin_tone, inv, sample_rate)

    # Создаём список всех стимулов (оригинальные и инвертированные)
    if add_inv:
        for _ in range(num_repetitions // 2):
            inv = False
            stim_triggers = add_triggers(sinus, sin_tone, inv, sample_rate)
            all_stimuli.append(stim_triggers)

            inv = True
            inv_stim_triggers = add_triggers(inv_sinus, sin_tone, inv, sample_rate)
            all_stimuli.append(inv_stim_triggers)
    else:
        assert(add_inv == 0)
        for _ in range(num_repetitions):
            inv = False
            stim_triggers = add_triggers(sinus,  sin_tone, inv, sample_rate)
            all_stimuli.append(stim_triggers)

    # Перемешиваем все стимулы в случайном порядке
    random.shuffle(all_stimuli)

    # Добавляем в сигнал: стимул → пауза
    full_signal = make_full_signal(all_stimuli, inter_stimulus_interval, sample_rate, percent_var_pause=0.1)

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
        add_inv,
        A

):
    """
    Creates WAV‑файл with syllables.
    """

    # Create one stimulus
    fs, stimulus = wavfile.read(r'\\MCSSERVER\DB Temp\physionet.org\FFR\stim\DA-20.wav')

    sin_tone = False
    plot_PSD = False
    if plot_PSD:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        plot_stim_PSD(stimulus, sin_tone, frequencies,  axes, 'multitaper', 30, 800, padding_factor=32)

    if add_inv:
        inv_stimulus = make_inv_stimulus(stimulus)

    # Make a full stimulation: stimulus + pause + inv stimulus + pause , N repetitions
    ramp_window, t_stim = make_ramp_window(len(stimulus) / sample_rate * 1000, sample_rate, rate=0.1, growth_rate=3.0)


    #Reset ASTIM: false cycle
    inv = False
    _ = add_triggers(stimulus, sin_tone, inv, sample_rate)

    all_stimuli = []
    if add_inv:
        for _ in range(num_repetitions // 2):
            inv = False
            #stim_triggers = add_triggers(stimulus * ramp_window, sin_tone, inv, sample_rate)
            stim_triggers = add_triggers(stimulus, sin_tone, inv, sample_rate)
            all_stimuli.append(stim_triggers)

            inv = True
            #inv_stim_triggers = add_triggers(inv_stimulus * ramp_window, sin_tone, inv, sample_rate)
            inv_stim_triggers = add_triggers(inv_stimulus, sin_tone, inv, sample_rate)
            all_stimuli.append(inv_stim_triggers)

    else:
        assert(add_inv == 0)
        for _ in range(num_repetitions):
            inv = False
            #stim_triggers = add_triggers(stimulus * ramp_window, sin_tone, inv, sample_rate)
            stim_triggers = add_triggers(stimulus, sin_tone, inv, sample_rate)
            all_stimuli.append(stim_triggers)

    random.shuffle(all_stimuli)
    full_signal = make_full_signal(all_stimuli, inter_stimulus_interval, sample_rate, percent_var_pause=0.1)

    base_name = f'Da_-20_TS{stimulus_duration}ms_TP{inter_stimulus_interval}ms_N{num_repetitions}_INV{add_inv}'
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
                        help='Directory for saving WAV file (default: M:\\DB Temp\\physionet.org\\files\\ffr_astim)')
    parser.add_argument('--F', type=int, nargs='+', default=[], help='Frequencies (Hz).\
    Default number of frequencies is 0 (no frequencies). Maximum number of frequencies is 5')
    parser.add_argument('--TS', type=float, required=True, help='Length of the stimulus (ms)')
    parser.add_argument('--TP', type=float, required=True, help='Interstimulus interval (ms)')
    parser.add_argument('--N', type=int, required=True, help='Number of repetitions of the stimulus')
    parser.add_argument('--SR', type=int, default=44100, help='Sampling rate default: 44100 Hz (fixed: 44100 Hz)')
    parser.add_argument('--INV', type=int, required=True,
                            choices=[0, 1], help='Add an inverted (polar) stimuli')
    parser.add_argument('--A', type=float, default=0.8,
                            choices=[0.03, 0.5, 1.0], help='Choose amplitude of the stimuli')
    """
    Example call:  python create_wav.py  --function multiple_sin --F 440 880 --TS 100 --TP 100 --N 1 --INV 0
    """
    args = parser.parse_args()
    F = np.array(args.F)

    # Validation of parameter ranges
    if F.shape[0] > 5:
        parser.error("argument --F: value must be in range 1 - 5, got {}".format(len(args.F)))
    for f in F:
        if not 1 < f < 1000:
            parser.error(f'argument --F: values must be in range 1 - 1000 Hz, got {f}')
    if not (50 <= args.TS <= 750):
        parser.error('argument --TS: value must be in range 50–750 ms')
    if not (100 <= args.TP <= 750):
        parser.error('argument --TP: value must be in range 100–750 ms')
    if not (2 <= args.N <= 10000):
        parser.error('argument --N: value must be in range 2 to 10000')
    if args.SR != 44100:
        parser.error("Sampling rate must be exactly 44100 Hz, got {}".format(args.SR))

    if args.function == 'repeated_da':
        create_repeated_da_syllable_wav(
                dir=args.dirname,
                frequencies=args.F,
                stimulus_duration=args.TS,
                inter_stimulus_interval=args.TP,
                num_repetitions=args.N,
                sample_rate=args.SR,
                add_inv=args.INV,
                A = args.A
        )
    elif args.function == 'multiple_sin':
            create_multiple_sin_wav(
                dir=args.dirname,
                frequencies=args.F,
                stimulus_duration=args.TS,
                inter_stimulus_interval=args.TP,
                num_repetitions=args.N,
                sample_rate=args.SR,
                add_inv=args.INV,
                A = args.A
        )


if __name__ == '__main__':
        main()