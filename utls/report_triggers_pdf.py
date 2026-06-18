import os
import numpy as np
import pandas as pd
import mne
from scipy.signal import correlate
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde


output_path = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\pics\test_triggers'
trim_interval = 160

def extract_n_events(events, event_dict, label):
    """
    Derives n epochs with label=label
    """
    target_id = event_dict[label]

    indices = np.where(events[:, 2] == target_id)[0]
    available_count = len(indices)

    return available_count

def count_wav_triggers_optimized(data, sequence):
    """
    Detect triggers in wav file
    """
    seq = np.array(sequence)
    corr = correlate(data, seq, mode='valid')
    # Absolute match
    target_corr = np.sum(seq ** 2)
    matches = np.where(corr == target_corr)[0]

    return len(matches), matches.tolist()

def compute_mean_std(intervals, bdf,  stim, fs, fname_bdf, fs_bdf, fname_wav):
    """
    Computes stat for intervals
    """
    #Most frequent value in intervals is stim
    stimulus= pd.Series(intervals).value_counts().index[:2]
    print(f"Most common value: {stimulus[0] } {stimulus[1] } ")

    if bdf:
        test_intervals = [stimulus[0], stimulus[1]]
        title = 'bdf '
        fs = fs_bdf
        fname = fname_bdf
    else:
        test_intervals = [stimulus[0]]
        title = 'wav '
        fs = fs_wav
        fname = fname_wav

    mask = np.isin(intervals, test_intervals)
    if stim:
        intervals_filtered = intervals[mask]
        boolean_stim = np.where(mask)[0]
        fname_indices_stim = os.path.join(output_path, f'boolean_stim_bdf_{bdf}.csv')
        np.savetxt(fname_indices_stim, boolean_stim, fmt='%d', delimiter=',')
        title = title + ' stim'
    else:
        intervals_filtered = intervals[~mask]
        title = title + ' pause'

    # Convert into seconds
    intervals_seconds = intervals_filtered / fs
    intervals_filtered = [round(x, 3) for x in intervals_filtered]
    mean_interval = round(np.mean(intervals_seconds), 3)
    std_interval = round(np.std(intervals_seconds, ddof=1), 3)  # ddof=1 for unbiased estimate

    return mean_interval, std_interval

def plot_deviations(ax, intervals_wav_s_cum_with_start, intervals_bdf_s_with_start):
    """
    Plot distribution of time jitter
    """
    # Remove uncomplete triggers
    unique_values, counts = np.unique(intervals_bdf_s_with_start, return_counts=True)
    most_common_value_bdf = unique_values[np.argmax(counts)]

    # Indices to remove "glued trigger" time interval
    indices_base = np.where(intervals_bdf_s_with_start > np.round(most_common_value_bdf, 1))[0]
    indices_base = np.unique(indices_base)

    # In wav we will have to remove both time intervals with glued trigger
    indices_next = indices_base + 1

    indices_to_remove_wav = np.concatenate([indices_base, indices_next])
    indices_to_remove_wav = np.unique(indices_to_remove_wav)

    intervals_bdf_s_with_start_corr = np.delete(intervals_bdf_s_with_start, indices_base)
    intervals_wav_s_cum_with_start_corr = np.delete(intervals_wav_s_cum_with_start, indices_to_remove_wav)

    m = min(len(intervals_wav_s_cum_with_start_corr ), len(intervals_bdf_s_with_start_corr ))

    diff_intervals = 1000 * (intervals_wav_s_cum_with_start_corr[:m] - intervals_bdf_s_with_start_corr[:m])
    data = np.round(diff_intervals, decimals=3)

    idx_min = np.argmin(data)
    idx_max = np.argmax(data)

    print('Значение разницы в минимуме:', data[idx_min])
    print('Значение разницы в максимуме:', data[idx_max])

    diff_big = len(np.where(abs(data) > 1)[0])

    mpl.rcParams['patch.linewidth'] = 2

    counts, bins, _ = ax.hist(
        data,
        bins=40,
        alpha=0.25,
        color='lightgray',
        edgecolor='black',
        label='Overall distribution'
    )


    kde = gaussian_kde(data)

    x_grid = np.linspace(data.min(), data.max(), 2000)
    y_kde = kde(x_grid)

    peaks, _ = find_peaks(
        y_kde,
        distance=50
    )

    colors = plt.cm.Set2.colors

    # Window around the mode
    window_ratio = 0.08
    window = (data.max() - data.min()) * window_ratio

    for i, peak_idx in enumerate(peaks):

        mode_x = x_grid[peak_idx]
        mode_x = round(mode_x)

        local_data = data[
            (data >= mode_x - window) &
            (data <= mode_x + window)
            ]

        if len(local_data) < 5:
            continue

        ax.axvline(
            mode_x,
            color=colors[i % len(colors)],
            linestyle='--',
            linewidth=1
        )

        ax.text(
            mode_x,
            counts.mean(),
            f'{mode_x:d} ms',
            color=colors[i % len(colors)],
            fontsize=14,
            fontweight='bold',
            ha='right'
        )

    ax.text(
        0.05, 0.95,  # upper left corner
        f' Count Time jitters > 1 ms: \n {round(diff_big / 4000 , 2)} %',
        color='red',
        fontsize=14,
        fontweight='bold',
        ha='left',  # allignment on the left
        va='top',
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8)
    )

    title = 'Distribution of deviations of bdf from wav triggers, ms'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim([0, counts.max() + 100])
    ax.set_xlabel('Time, ms', fontsize=16)
    ax.set_xlim([data.min() - 100, data.max() + 100])
    ax.set_ylabel('Counts', fontsize=18)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(axis='y', alpha=0.3)
    print(f'{title} saved!')

    return title

def prepare_intervals(intervals_bdf_s, intervals_wav_s):
    """
    Time intervals for stim and pause
    """
    intervals_bdf_with_start = np.concatenate([[0], intervals_bdf_s])
    fname_stim_bdf = os.path.join(output_path, 'boolean_stim_bdf_True.csv')
    stim_bdf = np.loadtxt(fname_stim_bdf, dtype=int)

    intervals_wav_s_cum = np.cumsum(intervals_wav_s)
    intervals_wav_s_with_start = np.concatenate([[0], intervals_wav_s])
    intervals_wav_s_cum_with_start = np.concatenate([[0], intervals_wav_s_cum])
    fname_stim_wav = os.path.join(output_path, 'boolean_stim_bdf_False.csv')
    stim_wav = np.loadtxt(fname_stim_wav, dtype=int)
    stim_wav = np.array(stim_wav)

    return intervals_wav_s_cum_with_start, intervals_wav_s_with_start, intervals_bdf_with_start, stim_wav, stim_bdf

def plot_triggers(ax2, ax3, ax4, events_bdf_s, intervals_wav_s_cum_with_start,
                  stim_wav, stim_bdf, filename_bdf, filename_wav):
    events_bdf_s_corr = events_bdf_s[:] - events_bdf_s.min()

    ax2.vlines(
    intervals_wav_s_cum_with_start[:trim_interval],
    ymin=0,
    ymax=0.4,
    colors='red',
    linewidth=3,
    alpha=0.6,
    label=f'WAV {filename_wav}'  # убираем fontsize из этой строки
    )

    ax2.legend(loc='upper left', fontsize=16)

    stim_wav_trim = stim_wav[stim_wav < trim_interval - 1]

    for xmin_i in stim_wav_trim:
        ax2.hlines(
        y=0.4,
        xmin=intervals_wav_s_cum_with_start[xmin_i],
        xmax=intervals_wav_s_cum_with_start[xmin_i + 1],
        colors='red',
        linewidth=3,
        alpha=0.6
    )

    ax2.set_ylim(0, 0.7)
    ax2.legend(loc='upper left', fontsize=16)
    ax2.grid(axis='x', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])

    ax3.vlines(events_bdf_s_corr[:trim_interval], ymin=0, ymax=0.7,
           colors='blue', linewidth=2, alpha=0.4, label=f'BDF {filename_bdf}')

    ax3.legend(loc='upper left', fontsize=16)
    stim_bdf_trim = stim_bdf[stim_bdf < trim_interval - 1]

    for xmin_i in stim_bdf_trim:
        ax3.hlines(
        y=0.7,  # уровень линии соответствует значению stim_bdf
        xmin=events_bdf_s_corr[xmin_i],
        xmax=events_bdf_s_corr[xmin_i + 1],
        colors='blue',
        linewidth=2,
        alpha=0.6
    )

    ax3.set_ylim(0, 1.0)
    ax3.set_xticklabels([])
    ax3.legend(loc='upper left', fontsize=16)
    ax3.grid(axis='x', alpha=0.3)

    ax3.tick_params(axis='x', labelsize=16, rotation=45)
    ax3.set_yticklabels([])
    ax3.tick_params(axis='y', labelsize=16)

    ax4.vlines(
    intervals_wav_s_cum_with_start[:trim_interval],
    ymin=0,
    ymax=0.4,
    colors='red',
    linewidth=3,
    alpha=0.6,
    label=f'WAV {filename_wav}'
        )
    ax4.vlines(
    events_bdf_s_corr[:trim_interval],
    ymin=0,
    ymax=0.7,
    colors='blue',
    linewidth=2,
    alpha=0.4,
    label=f'BDF {filename_bdf}'
    )

    for xmin_i in stim_wav_trim:
        ax4.hlines(
        y=0.4,
        xmin=intervals_wav_s_cum_with_start[xmin_i],
        xmax=intervals_wav_s_cum_with_start[xmin_i + 1],
        colors='red',
        linewidth=3,
        alpha=0.6
    )
    for xmin_i in stim_bdf_trim:
        ax4.hlines(
        y=0.7,
        xmin=events_bdf_s_corr[xmin_i],
        xmax=events_bdf_s_corr[xmin_i + 1],
        colors='blue',
        linewidth=2,
        alpha=0.6
    )

    # Общие настройки для ax4
    ax4.set_ylim(0, 1.0)
    ax4.legend(loc='upper left', fontsize=16)
    ax4.grid(axis='x', alpha=0.3)
    ax4.tick_params(axis='x', labelsize=16, rotation=45)
    ax4.tick_params(axis='y', labelsize=16)
    ax4.set_ylim(0, 1.4)
    ax4.set_xlabel('Time (s)', fontsize=18)
    ax4.legend(loc='upper left', fontsize=16)
    ax4.grid(axis='x', alpha=0.3)
    ax4.tick_params(axis='x', labelsize=16, rotation=45)
    ax4.set_yticklabels([])  # убираем метки оси Y


def plot_intervals(dummy,  events_bdf_tm, intervals_bdf_s, intervals_wav_s, filename_bdf, filename_wav) -> None:

    """
    Plots intervals_bdf_ms, intervals_wav_ms on one scale
    """

    (intervals_wav_s_cum_with_start, intervals_wav_s_with_start,
     intervals_bdf_s_with_start, stim_wav, stim_bdf) = prepare_intervals(intervals_bdf_s, intervals_wav_s)

    fig = plt.figure(figsize=(40, 12))

    gs = GridSpec(3, 2, width_ratios=[1, 3], wspace=0.3, hspace=0.4)

    ax1 = fig.add_subplot(gs[0, 0])  # 1-я строка, 1-й столбец
    ax2 = fig.add_subplot(gs[0, 1])  # 1-я строка, 2-й столбец
    ax3 = fig.add_subplot(gs[1, 1])  # 2-я строка, 1-й столбец
    ax4 = fig.add_subplot(gs[2, 1])  # 3-я строка, 1-й столбец

    # Новая ось в первом столбце, третьей строке
    ax_stats = fig.add_subplot(gs[2, 0])
    ax_stats.axis('off')  # скрываем оси — они не нужны для текста

    title = plot_deviations(ax1, intervals_wav_s_with_start, intervals_bdf_s_with_start)
    ax1.set_title(title, fontsize=16)

    plot_triggers(ax2, ax3, ax4, events_bdf_tm, intervals_wav_s_cum_with_start,
                  stim_wav, stim_bdf, fname_bdf, fname_wav)
    ax2.set_title('Timing of triggers', fontsize=18)

    # Statistics
    if dummy:
        event_counts = {
        'In\\6': extract_n_events(events_bdf, event_dict, label='In\\ 6'),
        'In/6': extract_n_events(events_bdf, event_dict, label='In/ 6'),
        'In\\7': extract_n_events(events_bdf, event_dict, label='In\\ 7'),
        'In/7': extract_n_events(events_bdf, event_dict, label='In/ 7'),
        'N wav': filename_wav.split('_')[-3]
        }
    else:
        event_counts = {
        'In\\6': extract_n_events(events_bdf, event_dict, label='6_low'),
        'In/6': extract_n_events(events_bdf, event_dict, label='6_high'),
        'In\\7': extract_n_events(events_bdf, event_dict, label='7_low'),
        'In/7': extract_n_events(events_bdf, event_dict, label='7_high'),
        'N wav': filename_wav.split('_')[-3]
        #'N wav': filename_wav.split('_')[-2]
        }

    # Формируем текст для отображения
    text_str = "Triggers statistics\n" + "-" * 20 + "\n"
    for label, count in event_counts.items():
        text_str += f"{label}: {count}\n"

    ax_stats.text(0.5, 0.5, text_str,
                  transform=ax_stats.transAxes,
                  fontsize=14,
                  ha='center',
                  va='center',
                  bbox=dict(boxstyle="round,pad=1", facecolor="lightblue", alpha=0.7),
                  fontfamily='monospace')

    # Сохранение в высоком разрешении
    filename_bdf = filename_bdf.split('.')[0] + filename_wav.split('_')[-3] + '.pdf'
    full_path_pdf = os.path.join(output_path, filename_bdf)

    fig.suptitle(filename_bdf, fontsize=18, fontweight='bold', y=0.98)
    fig.savefig(full_path_pdf, format='pdf', dpi=600, bbox_inches='tight', pad_inches=0.2)
    os.startfile(full_path_pdf)


def compute_interval_stat(dummy, events_wav, events_bdf, stat, fs_bdf, fs_wav, fname_bdf, fname_wav):

    """
    Computes statistics for stim, for pause
    """

    # Compute diff betw adjacent indices
    intervals_bdf = np.diff(events_bdf[:, 0])

    intervals_bdf_s = intervals_bdf / fs_bdf

    intervals_wav = np.diff(events_wav)

    intervals_wav_s = intervals_wav / fs_wav


    if stat:
        # Compute stat for stim, for pause
        bdf = True
        stim = True
        mean_stim, std_stim = compute_mean_std(intervals_bdf, bdf, stim, fs_bdf, fname_bdf, fs_bdf, fname_wav)
        stim = False
        mean_pause, std_pause = compute_mean_std(intervals_bdf, bdf, stim, fs_bdf, fname_bdf, fs_bdf, fname_wav)
        print('mean_stim', mean_stim, 'std_stim', std_stim, 'mean_pause', mean_pause, 'std_pause', std_pause)

        bdf = False
        stim = True
        mean_stim, std_stim = compute_mean_std(intervals_wav, bdf, stim, fs_bdf, fname_bdf, fs_bdf, fname_wav)
        stim = False
        mean_pause, std_pause = compute_mean_std(intervals_wav, bdf, stim, fs_bdf, fname_bdf, fs_bdf, fname_wav)
        print('mean_stim', mean_stim, 'std_stim', std_stim, 'mean_pause', mean_pause, 'std_pause', std_pause)

        plot_intervals(dummy, events_bdf[:, 0] / fs_bdf, intervals_bdf_s, intervals_wav_s, fname_bdf, fname_wav)


#fname_bdf  =  r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\test_triggers\test_DA_June26.bdf' #10000Hz
#fname_bdf  =  r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\test_triggers\test_DA-_June26.bdf' #5000Hz
fname_bdf  =  r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\test_triggers\test_DA+_June26.bdf'

#fname_wav  = r'\\MCSSERVER\DB Temp\physionet.org\FFR\stim\Da_syll_TS250.0ms_TP200.0ms_N4000_Amplitude_INV1.wav'
#fname_wav  = r'\\MCSSERVER\DB Temp\physionet.org\FFR\stim\Da_syll_TS250.0ms_TP150.0ms_N4000_INV1_June26.wav'
#fname_wav  = r'\\MCSSERVER\DB Temp\physionet.org\FFR\stim\Da_-20_TS250.0ms_TP150.0ms_N4000_INV1_June26.wav'
fname_wav  = r'\\MCSSERVER\DB Temp\physionet.org\FFR\stim\Da_+20_TS250.0ms_TP150.0ms_N4000_INV1_June26.wav'
#################################wav constants##################################
_SILENCE = 1
max_int16 = np.iinfo(np.int16).max
min_int16 = np.iinfo(np.int16).min
# 100
seq_6low = [max_int16, min_int16, min_int16, max_int16, min_int16, max_int16]
# 110
seq_7low = [max_int16, min_int16, max_int16, min_int16, min_int16, max_int16]
# 101
seq_6high = [max_int16, min_int16, min_int16, max_int16, max_int16, min_int16]
# 111
seq_7high = [max_int16, min_int16, max_int16, min_int16, max_int16, min_int16]

#####################################bdf######################################


raw = mne.io.read_raw_bdf(
        fname_bdf,
        preload=True,  # Загружаем данные в память сразу
        verbose=True  # Подробный вывод процесса
)

events_bdf, event_dict = mne.events_from_annotations(raw)
fs_bdf = int(raw.info.get('sfreq'))

fs_wav, stim = wavfile.read(fname_wav)
count_opt_6low, indices_opt_6low = count_wav_triggers_optimized(stim[:, 1], seq_6low)
count_opt_7low, indices_opt_7low = count_wav_triggers_optimized(stim[:, 1], seq_7low)

count_opt_6high, indices_opt_6high = count_wav_triggers_optimized(stim[:, 1], seq_6high)
count_opt_7high, indices_opt_7high = count_wav_triggers_optimized(stim[:, 1], seq_7high)
events_wav = np.concatenate([
        indices_opt_6low,
        indices_opt_6high,
        indices_opt_7low,
        indices_opt_7high,
])

events_wav_sorted = np.sort(events_wav)
filename_bdf = fname_bdf.split('\\')[-1]

filename_wav = fname_wav.split('\\')[-1]
stat = True
dummy = False

compute_interval_stat(dummy, events_wav_sorted, events_bdf, stat, fs_bdf, fs_wav, filename_bdf, filename_wav)
