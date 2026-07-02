import sys
from pathlib import Path
import re
import argparse
import threading
from datetime import datetime
import matplotlib.pyplot as plt
import config as cfg
from functions import (project_paths, process_plot_filt, save_pdf, show_progress)

def main():
    parser = argparse.ArgumentParser(description='FFR data processing')
    parser.add_argument('--subject', type=str, default='S1',
                        help='Subject identifier (default: S1). Valid: any non‑empty string')
    parser.add_argument('--preamplifier', type=str, default='False',
                        choices=['True', 'False'])
    parser.add_argument('--dummy', type=str, default='', help='Dummy mode flag. Valid: only "dummy" string',
                        choices=['dummy'])
    parser.add_argument('--short', type=str, default='',
                        help='Stimulus duration mode (default: empty). Valid: "short" or empty',
                        choices=['short', 'shortG'])
    parser.add_argument('--TS', type=int, required=True,
                        help='Stimulus duration in ms. Valid range: 50–750 ms')
    parser.add_argument('--TP', type=int, required=True,
                        help='Interstimulus interval (pause) in ms. Valid range: 100–750 ms')
    parser.add_argument('--tmin', type=int, required=True,
                        help='Start of time window in ms. Valid range: -300 to 0 ms')
    parser.add_argument('--tmax', type=int, required=True,
                        help='End of time window in ms. Valid range: 0 to 1000 ms')
    parser.add_argument('--method', type=str, default='multitaper',
                        help='Spectral analysis method (default: multitaper). Valid: "welch"',
                        choices=['welch'])
    parser.add_argument('--fmin', type=int, default=80,
                        help='Lower frequency bound in Hz (default: 80 Hz). Valid range: 1–100 Hz')
    parser.add_argument('--fmax', type=int, default=2500,
                        help='Upper frequency bound in Hz (default: 850 Hz). Valid range: 150 - 2000 Hz')
    parser.add_argument('--order', type=int, default=2,
                        help='Filter order (default: 1). Valid range: 1 - 100')
    parser.add_argument('--amp_threshold', type=int, default=35,
                        help='Amplitude threshold in µV (default: 35 µV). Valid range: 35 - 100')
    parser.add_argument('--trend_threshold', type=int, default=10,
                        help='Trend threshold in µV/s (default: 10 µV/s). Valid range: 10 - 100')
    parser.add_argument('--diff_threshold', type=int, default=25,
                        help='Difference threshold in µV (default: 25 µV). Valid range: 25 - 100')
    parser.add_argument('--average_out', type=bool, default=True,
                        help='Save Grand Average in EDF (default: True). Valid option False',
                        choices=[False])
    parser.add_argument('--N', type=int, required=True,
                        help='Number of averages. Valid range: 1-4000')
    parser.add_argument('--ftype', type=str, default='bdf',
                        help='Type of data file. Valid options: fif, bdf',
                        choices=['fif'])
    parser.add_argument('--fname_stim', type=str,
                        #default=r'\\MCSSERVER\DB Temp\physionet.org\FFR\stim\Da_syll_TS250.0ms_TP200.0ms_N4000_Amplitude_INV1.wav',
                        required=True,
                        help='Path to stimulus file (default: standard path)')
    """
    parser.add_argument('--fname_data', type=str,
                        # r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\non_filt\preamplifier\ffr_da_N4000_non_filtS1preamplifierbig.bdf'
                        required=True,
                        help='Path to data (bdf) file')
    """

    args = parser.parse_args()

    # Validation of parameter ranges
    if not (50 <= args.TS <= 750):
        parser.error('argument --TS: value must be in range 50–750 ms')
    if not (100 <= args.TP <= 750):
        parser.error('argument --TP: value must be in range 100–750 ms')
    if not (-300 <= args.tmin <= 0):
        parser.error('argument --tmin: value must be in range -300 to 0 ms')
    if not (100 <= args.tmax <= 800):
        parser.error('argument --tmax: value must be in range 100 to 800 ms')
    if not (70 <= args.fmin):
        parser.error('argument --fmin: value must be in range from 70 Hz')
    if not (150 <= args.fmax <= 3000):
        parser.error('argument --fmax: value must be in range 70 to 100 Hz')
    if not (1 <= args.order <= 100):
        parser.error('argument --order: value must be in range 1 to 100')
    if not (25 <= args.amp_threshold <= 100):
        parser.error('argument --amp_threshold: value must be in range 35 to 100 μV')
    if not (10 <= args.trend_threshold <= 100):
        parser.error('argument --trend_threshold: value must be in range 10 to 100 μV/s')
    if not (10 <= args.diff_threshold <= 100):
        parser.error('argument --diff_threshold: value must be in range 25 to 100 μV')
    if not (1 <= args.N <= 4000):
        parser.error('argument --N: value must be in range 25 to 4000')
    if Path(args.fname_stim).suffix.lower() != '.wav':
        parser.error('argument --fname_stim: not a WAV file (expected .wav extension)')
    """
    if not args.fname_stim.lower().endswith('.wav'):
        parser.error('argument --fname_stim: not a WAV file (expected .wav extension)')
    if not not args.fname_data.lower().endswith('.bdf'):
        parser.error('argument --fname_data: not a BDF file (expected .bdf extension)')
    """
    if args.short == 'short' or args.short == 'shortG':
        stim_name = Path(args.fname_stim).name
        parts = stim_name.split('_')
        ts_part = parts[2]  # 'TS90.0ms'
        match = re.search(r'TS(\d+\.?\d*)', ts_part)
        ts_value = float(match.group(1))  # 90.0
        if not (ts_value < 250):
            parser.error(f"When --short short is used, TS value must be < 250 ms), got {ts_value} ms")
        if not (args.TS < 250):
            parser.error(f"When --short short is used, TS value must be < 250 ms), got {args.TS} ms")
        if not (args.tmax - args.tmin  < 250):
            parser.error(f"When --short short is used, args.tmax - args.tmin value must be < 250 ms), got {args.tmax - args.tmin} ms")

    print(f"Parameters of FFR data processing:")
    print(f"  Stim file name: {args.fname_stim}")
    #print(f"  Data file name: {args.fname_data}")
    print(f"  Subject identifier: {args.subject}")
    print(f"  Stimulus duration: {args.TS} ms")
    print(f"  Interstimulus interval (pause): {args.TP} ms")
    print(f"  Number of averages: {args.N}")
    print(f"  Time window: {args.tmin} ms – {args.tmax} ms")
    print(f"  Frequency range: {args.fmin}–{args.fmax} Hz")
    print(f"  Filter order: {args.order}")
    print(f"  Amplitude threshold: {args.amp_threshold} microV")
    print(f"  Trend threshold: {args.trend_threshold} microV/s")
    print(f"  Difference threshold: {args.diff_threshold} microV")
    print(f"  Preamplifier: {args.preamplifier}")
    print(f"  Dummy mode: {args.dummy}")

    #base_path = Path(r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\FFR_diamond')
    #base_path = Path(r'\\MCSSERVER\DB Temp\physionet.org\FFR\data')
    #base_path = Path(r'C:\Users\msasha\Desktop\AStim\data')

    if args.ftype == 'fif':
        label_6 = cfg.LABEL_6_FIF
        label_7 = cfg.LABEL_7_FIF
    else:  # bdf
        label_6 = cfg.LABEL_6_BDF
        label_7 = cfg.LABEL_7_BDF

    if args.dummy:
        subject = 'hardware noise'
    else:
        subject = args.subject
    if args.preamplifier == 'True':
        preamplifier = 'preamplifier'
    else:
        preamplifier = ''

    if args.short or (args.fmax - args.fmin < 300):
        padding_factor = 32
    else:
        padding_factor = 4

    n_6low = [args.N // 2]
    n_7low = [args.N // 2]

    fpath_data, output_dir = project_paths(args.ftype, base_path,'non_filt', args.dummy, args.short, preamplifier, args.subject, args.N)

    fig, axes = plt.subplots(3, 2, figsize=(6, 8))

    stim_type = args.fname_stim.split('_')[0].split('\\')[-1]

    bad_indices, events, event_dict, n_epochs_clean, eeg_registration = process_plot_filt(
        axes, args.N, args.fname_stim, fpath_data, args.ftype, base_path, subject, args.short, 'non_filt', n_6low, n_7low,
        label_6, label_7,
        preamplifier, args.dummy, args.fmin, args.fmax, args.method, args.order, args.TS / 1000, args.tmin / 1000, args.tmax / 1000,
        args.amp_threshold, args.trend_threshold, args.diff_threshold, args.average_out,
        padding_factor, use_non_filt=True)


    save_pdf(fig, output_dir, args.fname_stim, stim_type, fpath_data, preamplifier, subject,
             n_6low, n_7low, label_6, label_7, n_epochs_clean, args.N, args.TS, args.TP, args.fmin, args.fmax,
             args.order, eeg_registration, events, event_dict)

if __name__ == '__main__':
    thread = threading.Thread(target=show_progress, args=(30, 0.5))
    thread.start()
    old_stdout = sys.stdout
    current_time = datetime.now().strftime("%d.%m.%y_%H.%M")
    log_filename = f"Log_{current_time}_command_line_ffr.txt"
    log_file = open(log_filename, 'w')
    sys.stdout = log_file

    main()

    thread.join()

    sys.stdout = old_stdout
    log_file.close()
