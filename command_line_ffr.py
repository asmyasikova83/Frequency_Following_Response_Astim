import sys
import re
import argparse
import threading
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from functions import (project_paths, process_plot_filt, process_plot_last_filt,save_pdf, show_progress)

def main():
    parser = argparse.ArgumentParser(description='FFR data processing')
    parser.add_argument('--subject', type=str, default='S1',
                        help='Subject identifier (default: S1). Valid: any non‑empty string')
    parser.add_argument('--preamplifier', type=str, default='True',
                        choices=['True', 'False'])
    parser.add_argument('--dummy', type=str, default='', help='Dummy mode flag. Valid: only "dummy" string',
                        choices=['dummy'])
    parser.add_argument('--short', type=str, default='',
                        help='Stimulus duration mode (default: empty). Valid: "short" or empty',
                        choices=['short', 'shortG'])
    parser.add_argument('--ts', type=int, default=250,
                        help='Stimulus duration in ms (default: 250 ms). Valid range: 50–750 ms')
    parser.add_argument('--tp', type=int, default=200,
                        help='Interstimulus interval (pause) in ms (default: 200 ms). Valid range: 100–750 ms')
    parser.add_argument('--tmin', type=int, default=-50,
                        help='Start of time window in ms (default: -50 ms). Valid range: -300 to 0 ms')
    parser.add_argument('--tmax', type=int, default=300,
                        help='End of time window in ms (default: 300 ms). Valid range: 100 to 800 ms')
    parser.add_argument('--method', type=str, default='multitaper',
                        help='Spectral analysis method (default: multitaper). Valid: "welch"',
                        choices=['welch'])
    parser.add_argument('--fmin', type=int, default=80,
                        help='Lower frequency bound in Hz (default: 80 Hz). Valid range: 1–100 Hz')
    parser.add_argument('--fmax', type=int, default=850,
                        help='Upper frequency bound in Hz (default: 850 Hz). Valid range: 150 - 2000 Hz')
    parser.add_argument('--order', type=int, default=100,
                        help='Filter order (default: 100). Valid range: 2 - 250')
    parser.add_argument('--amp_threshold', type=int, default=35,
                        help='Amplitude threshold in µV (default: 35 µV). Valid range: 35 - 100')
    parser.add_argument('--trend_threshold', type=int, default=10,
                        help='Trend threshold in µV/s (default: 10 µV/s). Valid range: 10 - 100')
    parser.add_argument('--diff_threshold', type=int, default=25,
                        help='Difference threshold in µV (default: 25 µV). Valid range: 25 - 100')
    parser.add_argument('--average_out', type=bool, default=True,
                        help='Save Grand Average in EDF (default: True). Valid option False',
                        choices=[False])
    parser.add_argument('--N', type=int, default=4000,
                        help='Number of averages (default: 4000). Valid range: 250-4000')

    parser.add_argument('--fname_stim', type=str,
                        default=r'\\MCSSERVER\DB Temp\physionet.org\FFR\stim\DA_syll_TS250ms_N4000_A100.0%_INV1.wav',
                        help='Path to stimulus file (default: standard path)')

    args = parser.parse_args()

    # Validation of parameter ranges
    if not (50 <= args.ts <= 750):
        parser.error('argument --ts: value must be in range 50–750 ms')
    if not (100 <= args.tp <= 750):
        parser.error('argument --tp: value must be in range 100–750 ms')
    if not (-300 <= args.tmin <= 0):
        parser.error('argument --tmin: value must be in range -300 to 0 ms')
    if not (100 <= args.tmax <= 800):
        parser.error('argument --tmax: value must be in range 100 to 800 ms')
    if not (70 <= args.fmin):
        parser.error('argument --fmin: value must be in range from 70 Hz')
    if not (150 <= args.fmax <= 2000):
        parser.error('argument --fmax: value must be in range 70 to 100 Hz')
    if not (1 <= args.order <= 250):
        parser.error('argument --order: value must be in range 2 to 250')
    if not (35 <= args.amp_threshold <= 100):
        parser.error('argument --amp_threshold: value must be in range 35 to 100 μV')
    if not (10 <= args.trend_threshold <= 100):
        parser.error('argument --trend_threshold: value must be in range 10 to 100 μV/s')
    if not (10 <= args.diff_threshold <= 100):
        parser.error('argument --diff_threshold: value must be in range 25 to 100 μV')
    if not (25 <= args.N <= 4000):
        parser.error('argument --N: value must be in range 25 to 4000')
    if Path(args.fname_stim).suffix.lower() != '.wav':
        parser.error('argument --fname_stim: not a WAV file (expected .wav extension)')
    if args.short == 'shortG' and Path(args.fname_stim).name.split('_')[1] != 'note':
        parser.error('argument --short shortG requires --fname_stim containing: note ')
    if args.short == 'short':
        stim_name = Path(args.fname_stim).name
        parts = stim_name.split('_')
        ts_part = parts[2]  # 'TS90.0ms'
        match = re.search(r'TS(\d+\.?\d*)', ts_part)
        ts_value = float(match.group(1))  # 90.0
        if not (ts_value < 250):
            parser.error(f"When --short short is used, TS value must be < 250 ms), got {ts_value} ms")
        if not (args.ts < 250):
            parser.error(f"When --short short is used, TS value must be < 250 ms), got {args.ts} ms")
        if not (args.tmax - args.tmin  < 250):
            parser.error(f"When --short short is used, args.tmax - args.tmin value must be < 250 ms), got {args.tmax - args.tmin} ms")

    print(f"Parameters of FFR data processing:")
    print(f"  Subject identifier: {args.subject}")
    print(f"  Stimulus duration: {args.ts} ms")
    print(f"  Interstimulus interval (pause): {args.tp} ms")
    print(f"  Number of averages: {args.N}")
    print(f"  Time window: {args.tmin} ms – {args.tmax} ms")
    print(f"  Frequency range: {args.fmin}–{args.fmax} Hz")
    print(f"  Filter order: {args.order}")
    print(f"  Amplitude threshold: {args.amp_threshold} microV")
    print(f"  Trend threshold: {args.trend_threshold} microV/s")
    print(f"  Difference threshold: {args.diff_threshold} microV")
    print(f"  Preamplifier: {args.preamplifier}")
    print()

    base_path = Path(r'\\MCSSERVER\DB Temp\physionet.org\FFR\data')

    n_6low = [args.N // 2]
    n_7low = [args.N // 2]

    if args.dummy:
        subject = 'hardware noise'
    else:
        subject = args.subject
    if args.preamplifier == 'True':
        multiplier = 1e-3
        preamplifier = 'preamplifier'
    else:
        multiplier = 1e-6
        preamplifier = ''
    if args.short or (args.fmax - args.fmin < 300):
        padding_factor = 32
    else:
        padding_factor = 4

    fname_bdf, output_dir = project_paths(base_path, 'non_filt', args.dummy, args.short, preamplifier, args.subject, args.N)

    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    stim_type = args.fname_stim.split('_')[0].split('\\')[-1]
    bad_indices = process_plot_filt(
        axes, stim_type, args.fname_stim, fname_bdf, base_path, subject, args.short, 'non_filt', n_6low, n_7low,
        preamplifier, args.dummy, args.fmin, args.fmax, args.method, args.order, args.ts / 1000, args.tmin / 1000, args.tmax / 1000, 0.05,
        args.amp_threshold, args.trend_threshold, args.diff_threshold, multiplier, args.average_out,
        padding_factor, use_non_filt=False)

    process_plot_last_filt(
        axes, bad_indices, fname_bdf,'non_filt',
        n_6low, n_7low, preamplifier, args.dummy, args.short, args.fmin, args.fmax, args.method, args.order, args.ts / 1000,
        args.tmin / 1000, args.tmax / 1000,0.05, args.amp_threshold, args.trend_threshold, args.diff_threshold, multiplier,
        padding_factor, use_non_filt=True)

    save_pdf(fig, output_dir, preamplifier, subject, args.short, n_6low, n_7low, args.fmin, args.fmax,
        args.ts, args.tmin / 1000, args.tmax / 1000)

    """  
    Example call:
    
    python command_line_ffr.py --subject S1  --short 'short' --ts 90 --tmin -50 
    --tmax 150 --fname_stim '\\MCSSERVER\DB Temp\physionet.org\FFR\stim\short\DA_syll_TS90.0ms_N4000_A100.0%_INV1.wav'
    
    python command_line_ffr.py --subject S1  --ts 250 --tmin -50 --tmax 300 
    --fname_stim '\\MCSSERVER\DB Temp\physionet.org\FFR\stim\DA_syll_TS250ms_N4000_A100.0%_INV1.wav'
  
    """
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
