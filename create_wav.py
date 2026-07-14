import os
import argparse
import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import write
import random
import matplotlib.pyplot as plt
import config as cfg
from functions import (create_multiple_sin_wav, create_repeated_da_syllable_wav)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    parser = argparse.ArgumentParser(description='Audio stimuli generation')
    parser.add_argument('--function', '-f', required=True,
                            choices=['repeated_da', 'multiple_sin'],
                            help='Function to generate stimuli: repeated_da или multiple_sin')
    parser.add_argument('--dirname', type=str, default='M:\\DB Temp\\physionet.org\\files\\ffr_astim',
                        help='Directory for saving WAV file (default: M:\\DB Temp\\physionet.org\\files\\ffr_astim)')
    parser.add_argument('--F', type=int, nargs='+', default=[80, 1500], help='Frequencies (Hz).\
    Default number of frequencies is 0 (no frequencies). Maximum number of frequencies is 5')
    parser.add_argument('--TS', type=float, required=True, help='Length of the stimulus (ms)')
    parser.add_argument('--TP', type=float, required=True, help='Interstimulus interval (ms)')
    parser.add_argument('--N', type=int, required=True, help='Number of repetitions of the stimulus')
    parser.add_argument('--SR', type=int, default=44100, help='Sampling rate default: 44100 Hz (fixed: 44100 Hz)')
    parser.add_argument('--INV', type=int, required=True,
                            choices=[0, 1], help='Add an inverted (polar) stimuli')
    parser.add_argument('--A', type=float, default=0.8,
                            choices=[0.03, 0.5, 1.0], help='Choose amplitude of the stimuli')
    parser.add_argument('--wavfname', type=str, default=' ',
                            help='Choose an examp[le of syllable')
    args = parser.parse_args()
    F = np.array(args.F)

    # Validation of parameter ranges
    if F.shape[0] > 5:
        parser.error("argument --F: value must be in range 1 - 5, got {}".format(len(args.F)))
    for f in F:
        if not 1 < f < 5000:
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
                A=args.A,
                wavfname = args.wavfname
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