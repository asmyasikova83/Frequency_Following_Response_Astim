import os
import csv
from pathlib import Path
from scipy.io import wavfile
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.ticker import FuncFormatter
import config as cfg
from functions import select_events, average_and_filter_epochs, zero_padding, find_peaks, plot_spectra_with_freq_vals

def compute_GA(epochs, tmin):
    """
    Preprocessing 4: Grand Average
    """
    grand_average = average_and_filter_epochs(epochs.get_data(), tmin)

    return grand_average

def extract_and_process(fname, dummy):

    raw_dummy = mne.io.read_raw_bdf(
    fname,
    preload=True,
    verbose=True
    )
    if dummy:
        raw_dummy_selected = raw_dummy.copy().pick_channels(ch_name)
    else:
        raw_dummy_selected = raw_dummy.copy()

    events, event_dict = mne.events_from_annotations(raw_dummy)
    available_6low, available_7low, adjusted_events_6low, adjusted_events_7low, sorted_events = select_events(n_6low, n_7low, LABEL_6_BDF, LABEL_7_BDF, events, event_dict)

    # Data segmentation (epoching)
    tmin = -0.1
    tmax = 0.3
    sound_delay = 0.00076

    epochs = mne.Epochs(
    raw_dummy_selected,
    sorted_events,
    tmin=tmin,
    tmax=tmax,
    baseline=(tmin, 0 + sound_delay),
    preload=True
    )

    grand_average = compute_GA(epochs, tmin)

    to_GA = True
    ga_data_padded = zero_padding(grand_average.get_data(), to_GA, 8)
    evoked = mne.EvokedArray(
    data=ga_data_padded,
    info=cfg.info1ch,
    tmin=0
    )

    psd = evoked.compute_psd(
    method='welch',
    fmin=fmin,
    fmax=fmax,
    n_fft=cfg.n_fft,  # без zero-padding
    n_per_seg=cfg.n_per_seg,  # длиннее сегмент → лучше разрешение
    n_overlap=cfg.n_overlap,  # 50% перекрытия
    verbose=False
    )

    data_psd = psd.get_data()  # V²/Hz
    #psd.plot()
    #plt.show(block=False)

    data_amplitude = np.sqrt(data_psd).flatten() * 1e9  # nV/√Hz
    #data_amplitude = np.log10(data_psd.flatten())
    freqs_data = psd.freqs

    #mean_spec_amp = mean_without_outliers_iqr(data_amplitude, k=3.0)
    idx_to_exclude = 3

    # Создаём булеву маску: True везде, кроме индекса idx_to_exclude
    exclude_mask = np.zeros_like(data_amplitude, dtype=bool)
    exclude_mask[idx_to_exclude] = True

    #mean_spec_amp = np.sum(data_amplitude[~exclude_mask])

    sum_spec_amp = np.sum(data_amplitude)

    return grand_average, data_amplitude, sum_spec_amp, freqs_data


def filter_peaks_by_distance(peak_indices, freq_slice, min_distance_hz):
    """
    Оставляет только пики, между которыми не менее min_distance_hz Гц.
    Возвращает отфильтрованные индексы.
    """
    if len(peak_indices) == 0:
        return []

    sorted_idx = sorted(peak_indices, key=lambda i: freq_slice[i])

    filtered = [sorted_idx[0]]

    for i in sorted_idx[1:]:
        last_freq = freq_slice[filtered[-1]]
        curr_freq = freq_slice[i]
        if (curr_freq - last_freq) >= min_distance_hz:
            filtered.append(i)

    return filtered

def find_final_peaks_to_plot(freq_slice, data_amp_pre):
    delta_f = freq_slice[1] - freq_slice[0]

    distance_bins = max(1, int(np.ceil(cfg.min_freq_gap / delta_f)))

    peaks_idx, _ = find_peaks(data_amp_pre, distance=distance_bins, prominence=None)

    sorted_order = np.argsort(data_amp_pre[peaks_idx])[::-1]
    top_peaks_indices = peaks_idx[sorted_order[:cfg.n_peaks]]

    sort_by_freq_order = np.argsort(freq_slice[top_peaks_indices])
    final_peaks_to_plot = top_peaks_indices[sort_by_freq_order]

    return final_peaks_to_plot

def mean_without_outliers_iqr(data, k):
    data = np.asarray(data)
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1

    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    mask = (data >= lower_bound) & (data <= upper_bound)
    clean_data = data[mask]

    return np.sum(clean_data)

def plot_final_peaks(ax, freq_slice, final_peaks_amp_to_plot, final_peaks_amp_pre_to_plot, y_top, dummy, stim):
    """
    Отрисовка пиков:
      - final_peaks_amp_to_plot: зелёные пунктирные линии (color='green', linestyle='--')
      - final_peaks_amp_pre_to_plot: фиолетовые пунктирные линии (color='purple', linestyle='--')

    Параметры:
      ax: matplotlib Axes
      freq_slice: массив частот для среза
      final_peaks_amp_to_plot: индексы/значения пиков основного сигнала
      final_peaks_amp_pre_to_plot: индексы/значения пиков сигнала с предусилителем
      data_amp, data_amp_pre: массивы амплитуд (нужны для получения значений на пиках)
      y_top: верхняя граница оси Y для размещения подписей
    """

    if stim:
        freq_to_save = []
        for i, idx in enumerate(final_peaks_amp_to_plot):
            freq_val = freq_slice[idx]
            freq_to_save.append(freq_val)

            ax.axvline(
            x=freq_val,
            alpha=0.8,
            linewidth=1.0,
            linestyle='--',
            color='green'
            )
            ax.axvspan(
                xmin=freq_val - 10,
                xmax=freq_val + 10,
                color='teal',
                alpha=0.2,
                linewidth=0
            )
        fname = r"\\MCSSERVER\DB Temp\physionet.org\FFR\data\dummy\freq_values.csv"
        df_save = pd.DataFrame({'freq_Hz': freq_to_save})
        df_save.to_csv(fname, index=False)

    else:
        if not dummy:
            fname = r"\\MCSSERVER\DB Temp\physionet.org\FFR\data\dummy\freq_values.csv"
            final_peaks_stim = pd.read_csv(fname, delimiter=',')
            final_peaks_stim_np= final_peaks_stim['freq_Hz'].to_numpy()
            for i, idx in enumerate(final_peaks_stim_np):
                freq_val = final_peaks_stim_np[i]
                ax.axvspan(
                xmin=freq_val - 10,
                xmax=freq_val+ 10,
                color='teal',
                alpha=0.2,
                linewidth=0
                )
            for i, idx in enumerate(final_peaks_amp_to_plot):
                freq_val = freq_slice[idx]
                freq_display = np.round(freq_val)

                ax.axvline(
                    x=freq_val,
                    alpha=0.8,
                    # label=f'{i + 1}: {freq_display} Hz',
                    linewidth=1.0,
                    linestyle='--',
                    color='blue'
                )
                ax.axvline(
                    x=freq_val,
                    alpha=0.8,
                    # label=f'{i + 1}: {freq_display} Hz',
                    linewidth=1.0,
                    linestyle='--',
                    # color='#40E0D0'
                    color='blue'
                )

            for i, idx in enumerate(final_peaks_amp_pre_to_plot):
                freq_val = freq_slice[idx]
                freq_display = np.round(freq_val)

                ax.axvline(
                    x=freq_val,
                    alpha=0.8,
                    # label=f'{i + 1}: {freq_display} Hz',
                    linewidth=1.0,
                    linestyle='--',
                    color='blue'
                )
                ax.axvline(
                    x=freq_val,
                    alpha=0.8,
                    # label=f'{i + 1}: {freq_display} Hz',
                    linewidth=1.0,
                    linestyle='--',
                    # color='#40E0D0'
                    color='purple'
                )

    def add_hz(x, pos):
        return f'{int(x)} Hz'

    ax.xaxis.set_major_formatter(FuncFormatter(add_hz))
    ax.set_ylabel('nV/√Hz', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.tick_params(axis='both', which='major', labelsize=6)

    ax.legend( loc='upper right',
                ncol=2,  fontsize=6)

    return ax

def plot_spectra_with_freq_pre_vs_nonpre(ax, freq_slice, data_amp, data_amp_pre, dummy):
    """
    Plots spectra with frequency labels
    """

    y_max = max(data_amp)
    if dummy:
        y_top = y_max * 1.7
    else:
        y_top = y_max * 1.1

    final_peaks_amp = filter_peaks_by_distance(
        find_final_peaks_to_plot(freq_slice, data_amp),
        freq_slice,
        min_distance_hz=100.0
    )

    final_peaks_amp_pre = filter_peaks_by_distance(
        find_final_peaks_to_plot(freq_slice, data_amp_pre),
        freq_slice,
        min_distance_hz=100.0
    )
    stim = 0
    ax =  plot_final_peaks(ax, freq_slice, final_peaks_amp, final_peaks_amp_pre, y_top, dummy, stim)

    return ax

def plot_stim_PSD(ax, stimulus, fmin, fmax, padding_factor):
    """
    Plot Spectral Amplitude of the stimulus
    """
    data_stim = stimulus[:, 0]

    to_GA = False
    data_stim_padded = zero_padding(data_stim, to_GA, padding_factor=padding_factor)
    evoked_stim = mne.EvokedArray(
        data=np.transpose(data_stim_padded),
        info=cfg.info_wav,
        tmin=0
    )
    evoked_stim_resampled = evoked_stim.resample(cfg.fs, npad="auto")

    psd = evoked_stim_resampled.compute_psd(
        method='welch',
        fmin=fmin,
        fmax=fmax,
        n_fft=cfg.n_fft,
        n_per_seg=cfg.n_per_seg,
        n_overlap=cfg.n_overlap,
        verbose=False
    )
    # Frequency resolution: 10_000 / 1024 ≈ 9.77 Гц

    data_psd = psd.get_data()
    data_amplitude = np.sqrt(data_psd).flatten()
    freqs_data = psd.freqs

    y_max = data_amplitude.max()
    y_top = y_max * 1.1 # 50%
    y_bottom = 0

    ax.plot(freqs_data , data_amplitude, 'g-', linewidth=1.5)
    final_peaks = filter_peaks_by_distance(
        find_final_peaks_to_plot(freqs_data, data_amplitude),
        freqs_data,
        min_distance_hz=100.0
    )
    dummy = 0
    stim = 1
    ax =  plot_final_peaks(ax, freqs_data, final_peaks, [], y_top, dummy, stim)
    ax.set_ylim(y_bottom, y_top)

ch_name = ['Cz-(A1+A2)/2']
ref_chs = ['A1A2']
LABEL_6_BDF = '6_low'
LABEL_7_BDF = '7_low'
n_6low = [4000 // 2]
n_7low = [4000 // 2]
fmin = 10
fmax = 1500

# Верхний подграфик: grand_average и grand_average_preamplifier
frequencies = []
wavfname = Path(r"\\MCSSERVER\DB Temp\physionet.org\FFR\stim\DA_syll_TS250ms_N4000_A100.0%_INV1.wav")
fs, stimulus = wavfile.read(wavfname)

fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=False)  # 3 подграфика, общая ось X
plot_stim_PSD(axs[0], stimulus, fmin=fmin, fmax=fmax, padding_factor=8)
fig.suptitle(
    'Сравнение NVX36 vs NVX36 + внешний предусилитель MNSENS-ACP',
    fontsize=14,
    fontweight='bold',
    y=0.98  # чуть ниже верхнего края, чтобы не наезжало на поля
)
#axs[0].set_yscale('log')
axs[0].set_title('Спектральная амплитуда аудиостимула (слог Да)')
axs[0].set_yticks([])
axs[0].tick_params(axis='both', which='major', labelsize=10)

dummy = 0
fname_preamplifier = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\non_filt\preamplifier\ffr_da_N4000_non_filtS0preamplifier.bdf'
fname_dummy = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\ffr_da_N4000_S0_step1.bdf'
grand_average, data_amplitude, sum_spec_amp, freqs_data = extract_and_process(fname_dummy, dummy=dummy)
grand_average_preamplifier, data_amplitude_pre, sum_spec_amp_pre, freqs_data_preamplifier = extract_and_process(fname_preamplifier, dummy=dummy)

# Нижний подграфик: data_amplitude и data_amplitude_preamplifier (PSD /
axs[1].plot(freqs_data , data_amplitude, 'purple', linewidth=1.5, label='NVX36 (3999 усреднений)')
axs[1].plot(freqs_data , data_amplitude_pre, 'b-', linewidth=1.5, label='NVX36 + MNSENS-ACP (4002 усреднений)')

plot_spectra_with_freq_pre_vs_nonpre(axs[1], freqs_data, data_amplitude, data_amplitude_pre, dummy)
axs[1].set_title('Спектральная амплитуда ВП (FFR) испытуемого')
axs[1].legend(loc='upper right', fontsize=12)
axs[1].tick_params(axis='both', which='major', labelsize=10)

dummy = 1
fname_dummy = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\dummy\ffr_da_N4000_dummy.bdf'
fname_preamplifier = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\dummy\preamplifier\ffr_da_N4000_dummypreamplifier.bdf'
grand_average, data_amplitude, sum_spec_amp, freqs_data = extract_and_process(fname_dummy, dummy=dummy)
grand_average_preamplifier, data_amplitude_pre, sum_spec_amp_pre, freqs_data_preamplifier = extract_and_process(fname_preamplifier, dummy=dummy)

axs[2].plot(freqs_data , data_amplitude, 'purple', linewidth=1.5, label='NVX36 (3995 усреднений)')
axs[2].plot(freqs_data , data_amplitude_pre, 'b-', linewidth=1.5, label='NVX36 + MNSENS-ACP (3991 усреднений)')
plot_spectra_with_freq_pre_vs_nonpre(axs[2], freqs_data, data_amplitude, data_amplitude_pre, dummy)
axs[2].set_title('Спектральная амплитуда собственного шума (на закоротке входов оборудования)')
axs[2].tick_params(axis='both', which='major', labelsize=10)
axs[2].legend(loc='upper right', fontsize=12)
plt.tight_layout()

out_path =r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\dummy'
full_path = os.path.join(out_path, "Preamplifier Experiment")
pdf = 0
if pdf:
    plt.savefig(
    full_path,
    format='pdf',
    dpi=300,
    bbox_inches='tight',
    transparent=False
    )
else:
    base, _ = os.path.splitext(full_path)
    svg_path = base + '.svg'

    plt.savefig(
    svg_path,
    format='svg',
    bbox_inches='tight',
    transparent=True
    )

print(f"График сохранён: {os.path.abspath(out_path)}")
plt.show()