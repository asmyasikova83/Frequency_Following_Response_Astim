import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from functions import (process_plot_filt, project_paths, process_plot_last_filt,save_pdf, show_progress)

def main():
    parser = argparse.ArgumentParser(description='Обработка FFR данных')
    parser.add_argument('--subject', type=str, default='S1', help='Идентификатор субъекта (по умолчанию: S1)')
    parser.add_argument('--preamplifier', type=str, default='', help='Преамплификатор (по умолчанию: пустой)')
    parser.add_argument('--dummy', type=str, default='dummy', help='Режим dummy (по умолчанию: dummy)')
    parser.add_argument('--ts', type=float, default=0.25, help='Длительность стимула ts (по умолчанию: 0.25)')
    parser.add_argument('--tmin', type=float, default=-0.05, help='Начало временного окна (в секундах, по умолчанию: -0.05 с)')
    parser.add_argument('--tmax', type=float, default=0.3, help='Конец временного окна (в секундах, по умолчанию: 0.3 с)')
    parser.add_argument('--method', type=str, default='multitaper', help='Метод спектрального анализа (по умолчанию: multitaper)')
    parser.add_argument('--fmin', type=float, default=40, help='Нижняя граница частоты (в Гц, по умолчанию: 40 Гц)')
    parser.add_argument('--fmax', type=float, default=850, help='Верхняя граница частоты (в Гц, по умолчанию: 850 Гц)')
    parser.add_argument('--order', type=int, default=100, help='Порядок фильтра (по умолчанию: 100)')
    parser.add_argument('--amp_threshold', type=float, default=35, help='Порог амплитуды (в мкВ, по умолчанию: 85 мкВ)')
    parser.add_argument('--trend_threshold', type=float, default=10, help='Порог тренда (в мкВ/с, по умолчанию: 100 мкВ/с)')
    parser.add_argument('--diff_threshold', type=float, default=25, help='Порог разности (в мкВ, по умолчанию: 25 мкВ)')
    parser.add_argument('--fname_stim', type=str, default=r'\\MCSSERVER\DB Temp\physionet.org\FFR\stim\DA_syll_TS250ms_N4000_A100.0%_INV1.wav',
             help='Путь к файлу стимула (по умолчанию: стандартный путь)')

    args = parser.parse_args()

    print(f"Параметры обработки:")
    print(f"  Субъект: {args.subject}")
    print(f"  Временное окно: {args.tmin} с – {args.tmax} с")
    print(f"  Метод: {args.method}")
    print(f"  Частотный диапазон: {args.fmin}–{args.fmax} Гц")
    print(f"  Порядок фильтра: {args.order}")
    print(f"  Порог амплитуды: {args.amp_threshold} мкВ")
    print(f"  Порог тренда: {args.trend_threshold} мкВ/с")
    print(f"  Порог разности: {args.diff_threshold} мкВ")

    base_path = Path(r'\\MCSSERVER\DB Temp\physionet.org\FFR\data')
    fname_bdf, output_dir = project_paths(base_path, 'non_filt', args.dummy, 'short', args.preamplifier, args.subject)

    if args.dummy:
        n_6low = [1999]
        n_7low = [1992]
    else:
        n_6low = [1999]
        n_7low = [1999]

    if args.preamplifier:
        multiplier = 1e-3
    else:
        multiplier = 1e-6

    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    bad_indices = process_plot_filt(
        axes, args.fname_stim, fname_bdf, output_dir, args.subject, 'short', 'non_filt', n_6low, n_7low,
        args.preamplifier, args.dummy, args.fmin, args.fmax, args.method, args.order, args.ts, args.tmin, args.tmax, 0.05,
        args.amp_threshold, args.trend_threshold, args.diff_threshold, multiplier, use_non_filt=False)

    process_plot_last_filt(
        axes, bad_indices, args.fname_stim, fname_bdf, output_dir, args.subject, 'short', 'non_filt',
        n_6low, n_7low, args.preamplifier, args.dummy, args.fmin, args.fmax, args.method, args.order, args.ts,
        args.tmin, args.tmax,0.05, args.amp_threshold, args.trend_threshold, args.diff_threshold, multiplier,
        use_non_filt=True)

    save_pdf(fig, output_dir, args.preamplifier, args.subject, 'short', n_6low, n_7low, args.fmin, args.fmax,
        args.ts, args.tmin, args.tmax)

    """  
    Example call:
    python process_ffr.py \
     --subject S0 \
    --preamplifier 'preamplifier \
    --dummy '' \
    --ts 0.2 \
    --fname_stim r'\\MCSSERVER\DB Temp\physionet.org\files\ffr_astim\DA_syll_TS250.0ms_TP200.0ms_N1_A100.0%_INV0.wav'
  
    """
if __name__ == '__main__':
    show_progress(total_steps=15, duration=0.5)

    old_stdout = sys.stdout
    log_file = open('process_ffr.log', 'w')
    sys.stdout = log_file

    main()

    sys.stdout = old_stdout
    log_file.close()