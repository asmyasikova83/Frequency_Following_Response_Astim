import sys
import argparse
import threading
from pathlib import Path
import matplotlib.pyplot as plt
from functions import (project_paths, process_plot_filt, process_plot_last_filt,save_pdf, show_progress)

def main():
    parser = argparse.ArgumentParser(description='Обработка FFR данных')
    parser.add_argument('--subject', type=str, default='S1', help='Идентификатор субъекта (по умолчанию: S1)')
    parser.add_argument('--preamplifier', type=str, default='', help='Преамплификатор (по умолчанию: пустой)')
    parser.add_argument('--dummy', type=str, default='', help='Режим dummy (по умолчанию: dummy)')
    parser.add_argument('--short', type=str, default='', help='По длительности стимула (по умолчанию: 250 мс)')
    parser.add_argument('--ts', type=int, default=250, help='Длительность стимула ts мс (по умолчанию: 250 мс)')
    parser.add_argument('--tmin', type=float, default=-50, help='Начало временного окна (в мс, по умолчанию: -50 с)')
    parser.add_argument('--tmax', type=float, default=300, help='Конец временного окна  (в мс, по умолчанию: 300 с)')
    parser.add_argument('--method', type=str, default='multitaper', help='Метод спектрального анализа (по умолчанию: multitaper)')
    parser.add_argument('--fmin', type=float, default=40, help='Нижняя граница частоты (в Гц, по умолчанию: 40 Гц)')
    parser.add_argument('--fmax', type=float, default=850, help='Верхняя граница частоты (в Гц, по умолчанию: 850 Гц)')
    parser.add_argument('--order', type=int, default=100, help='Порядок фильтра (по умолчанию: 100)')
    parser.add_argument('--amp_threshold', type=float, default=35, help='Порог амплитуды (в мкВ, по умолчанию: 85 мкВ)')
    parser.add_argument('--trend_threshold', type=float, default=10, help='Порог тренда (в мкВ/с, по умолчанию: 100 мкВ/с)')
    parser.add_argument('--diff_threshold', type=float, default=25, help='Порог разности (в мкВ, по умолчанию: 25 мкВ)')
    parser.add_argument('--save_averag_in_edf', type=bool, default=True, help='Сохранить результат усреднения (по умолчанию: True)')
    parser.add_argument('--fname_stim', type=str, default=r'\\MCSSERVER\DB Temp\physionet.org\FFR\stim\DA_syll_TS250ms_N4000_A100.0%_INV1.wav',
             help='Путь к файлу стимула (по умолчанию: стандартный путь)')

    args = parser.parse_args()

    print(f"Параметры обработки:")
    print(f"  Субъект: {args.subject}")
    print(f"  Длительность стимула: {args.ts} мс")
    print(f"  Временное окно: {args.tmin} с – {args.tmax} с")
    print(f"  Частотный диапазон: {args.fmin}–{args.fmax} Гц")
    print(f"  Порядок фильтра: {args.order}")
    print(f"  Порог амплитуды: {args.amp_threshold} мкВ")

    base_path = Path(r'\\MCSSERVER\DB Temp\physionet.org\FFR\data')
    fname_bdf, output_dir = project_paths(base_path, 'non_filt', args.dummy, args.short, args.preamplifier, args.subject)

    if args.dummy:
        n_6low = [1999]
        n_7low = [1992]
    elif args.short:
        n_6low = [1994]
        n_7low = [1994]
    else:
        n_6low = [1999]
        n_7low = [1999]

    if args.preamplifier:
        multiplier = 1e-3
    else:
        multiplier = 1e-6

    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    bad_indices = process_plot_filt(
        axes, args.fname_stim, fname_bdf, output_dir, args.subject, args.short, 'non_filt', n_6low, n_7low,
        args.preamplifier, args.dummy, args.fmin, args.fmax, args.method, args.order, args.ts / 1000, args.tmin / 1000, args.tmax / 1000, 0.05,
        args.amp_threshold, args.trend_threshold, args.diff_threshold, multiplier, args.save_averag_in_edf, use_non_filt=False)

    process_plot_last_filt(
        axes, bad_indices, fname_bdf,'non_filt',
        n_6low, n_7low, args.preamplifier, args.dummy, args.short, args.fmin, args.fmax, args.method, args.order, args.ts / 1000,
        args.tmin / 1000, args.tmax / 1000,0.05, args.amp_threshold, args.trend_threshold, args.diff_threshold, multiplier,
        use_non_filt=True)

    save_pdf(fig, output_dir, args.preamplifier, args.subject, args.short, n_6low, n_7low, args.fmin, args.fmax,
        args.ts, args.tmin / 1000, args.tmax / 1000)

    """  
    Example call:
    
    python command_line_ffr.py --subject S1 --preamplifier 'preamplifier' --short 'short' --ts 90 --tmin -50 \
    --tmax 150 --fname_stim '\\MCSSERVER\DB Temp\physionet.org\FFR\stim\short\DA_syll_TS90.0ms_N4000_A100.0%_INV1.wav'
    
    python command_line_ffr.py --subject S1 --preamplifier 'preamplifier' --ts 250 --tmin -50 --tmax 300 \
    --fname_stim '\\MCSSERVER\DB Temp\physionet.org\FFR\stim\DA_syll_TS250ms_N4000_A100.0%_INV1.wav'
  
    """
if __name__ == '__main__':
    thread = threading.Thread(target=show_progress, args=(30, 0.5))
    thread.start()

    old_stdout = sys.stdout
    log_file = open('command_line_ffr.log', 'w')
    sys.stdout = log_file

    main()

    thread.join()

    sys.stdout = old_stdout
    log_file.close()
