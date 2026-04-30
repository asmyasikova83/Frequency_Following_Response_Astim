import numpy as np
import os
import mne
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from mne.decoding import *
import random
from scipy import stats
from scipy.signal import firwin, firwin2, filtfilt
from scipy.io import wavfile
import warnings
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    module='mne.*'  # или 'pandas.*' и т. д.
)
warnings.filterwarnings("ignore", message=".*EDF format requires equal-length data blocks.*")

def show_progress(steps, delay, width=20):
    print('Parsed the args, starting... ', end='', flush=True)
    for _ in range(steps):
        print('.', end='', flush=True)
        time.sleep(delay)
    print('Done!')

def project_paths(base_path, non_filt, dummy, short, preamplifier, subject, N ):
    """
    Returns fname_bdf, output_dir
    """
    if dummy:
        fname_bdf = base_path / non_filt / dummy / preamplifier / f'ffr_da_N4000_{dummy}{non_filt}{preamplifier}{short}.BDF'
        output_dir = base_path.joinpath('pics', preamplifier, dummy)
        subject = 'Hardware noise'
    else:
        fname_bdf = base_path / non_filt / dummy / preamplifier / f'ffr_da_N4000_{dummy}{non_filt}{subject}{preamplifier}{short}.BDF'
        output_dir = base_path / 'pics' / f'{preamplifier}/{dummy}/{subject}'

    os.makedirs(output_dir, exist_ok=True)

    return fname_bdf, output_dir

def save_ga_in_edf(grand_average, output_dir, subject, preamplifier, short, tmin, tmax, fmin, fmax, n_6low, n_7low):

    # Convert into Raw
    raw_conv = mne.io.RawArray(grand_average.data, grand_average.info)

    out_ga = os.path.join(output_dir, 'grand_average_data')
    os.makedirs(out_ga, exist_ok=True)

    out = os.path.join(out_ga,
                               f'Grand_Average_{subject}{preamplifier}{short}_{tmin}ms_{tmax}ms_FIR_{fmin}_{fmax}Hz_N{n_6low + n_7low}.edf')
    # Save as  EDF
    raw_conv.export(
    fname=out,
    fmt='edf',
    overwrite=True
    )
    print(f"Data are successfully saved in : {out}")

def import_raw(fname, non_filt, use_non_filt, preamplifier, dummy, fmin, fmax, order,transition_width):
    """
    Imports and filters data if needed
    """
    label_6 = '6_low'
    label_7 = '7_low'
    raw = mne.io.read_raw_bdf(
        fname,
        preload=True,  # Загружаем данные в память сразу
        verbose=True  # Подробный вывод процесса
    )

    if non_filt and not preamplifier:
        raw.set_eeg_reference(ref_channels=['13', '19'], projection=False)
        raw_selected = raw.copy().pick_channels(['1'])
        label_6 = 'In\\6'
        label_7 = 'In\\7'
    elif dummy and non_filt and preamplifier:
        raw.set_eeg_reference(ref_channels=['13', '19'], projection=False)
        raw_selected = raw.copy().pick_channels(['1'])
        label_6 = 'In\\6'
        label_7 = 'In\\7'
    else:
        raw_selected = raw

    if use_non_filt:
        raw_to_epo = raw_selected
    else:
        filtered_signal = fir_bandpass_filter(raw_selected.get_data(), fmin, fmax,  int(raw.info.get('sfreq')), order,transition_width)
        raw_to_epo = mne.io.RawArray(filtered_signal, raw_selected.info)

    events, event_dict = mne.events_from_annotations(raw)

    return raw, raw_to_epo, events, event_dict, label_6, label_7

def extract_n_events(events, event_dict, label, n, random_selection=True):
    """
    Derives n epochs with label=label
    """
    target_id = event_dict[label]

    indices = np.where(events[:, 2] == target_id)[0]
    available_count = len(indices)

    if available_count < n[0]:
        print(f'Error: there are less events than {n[0]}, take {available_count}')
        n = available_count

    if random_selection:
        selected_indices = np.random.choice(indices, size=n, replace=False)
    else:
        selected_indices = indices[:n]

    return available_count, events[selected_indices], selected_indices

def select_events(n_6low, n_7low, label_6, label_7, events, event_dict):
        """
        Preprocessing: pick events
        """

        available_6low, selected_events_6low, selected_indices_6low = extract_n_events(
        events,
        event_dict,
        label=label_6,
        n=n_6low,
        random_selection=True
        )
        available_7low, selected_events_7low, selected_indices_7low = extract_n_events(
        events,
        event_dict,
        label=label_7,
        n=n_7low,
        random_selection=True
        )

        if available_6low != available_7low:
            m = np.min([available_6low, available_7low])
            adjusted_events_6low = selected_events_6low[:m]
            adjusted_events_7low = selected_events_7low[:m]
        else:
            adjusted_events_6low = selected_events_6low
            adjusted_events_7low = selected_events_7low

        print('len adjusted_events_6low', len(adjusted_events_6low))
        print('len adjusted_events_7low', len(adjusted_events_7low))

        combined_events = np.concatenate([adjusted_events_6low, adjusted_events_7low])
        # Derive indices according to the first col (times)
        sorted_indices = np.argsort(combined_events[:, 0])
        sorted_events = combined_events[sorted_indices]

        return available_6low, available_7low, sorted_events

def remove_artifacts(epochs, AMP_THRESHOLD, TREND_THRESHOLD, DIFF_THRESHOLD, multiplier):
    bad_epochs_amp, amp_values = detect_artifacts_threshold(epochs, AMP_THRESHOLD * multiplier)
    print(f"Artifacts — amplitude: {np.sum(bad_epochs_amp)} epochs")

    bad_epochs_trend, trend_values = detect_artifacts_trend(epochs, TREND_THRESHOLD * multiplier)
    print(f"Artifacts — trend: {np.sum(bad_epochs_trend)} epochs")

    bad_epochs_diff, diff_values = detect_artifacts_diff(epochs, DIFF_THRESHOLD * multiplier)
    print(f"Artifacts — difference between 2 adjacent time points: {np.sum(bad_epochs_diff)} epochs")

    # Объединение всех критериев: эпоха помечается как артефакт, если хотя бы один метод её выявил
    bad_epochs_combined = bad_epochs_amp | bad_epochs_trend | bad_epochs_diff
    total_bad = np.sum(bad_epochs_combined)
    print(f"Total artifactual epochs (combined): {total_bad} epochs")

    bad_indices = np.where(bad_epochs_combined)[0]
    epochs.drop(bad_indices, reason='artifact_detection')

    return epochs, bad_indices

def fir_bandpass_filter(data, low_cutoff, high_cutoff, fs, order, transition_width):
    """
    FIR filter using firwin and firwin2
    """
    # Нормализуем частоты относительно частоты Найквиста
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    """
    # FIR-filter using firwin
    b = firwin(
        numtaps=order + 1,
        cutoff=[low, high],
        pass_zero=False,
        window='bartlett'
    )
    """
    freq = [0, max(0, low - transition_width), low, high, min(1, high + transition_width), 1]
    gain = [0, 0, 1, 1, 0, 0]

    unique_freq, unique_indices = np.unique(freq, return_index=True)
    unique_gain = [gain[idx] for idx in unique_indices]

    b = firwin2(
        numtaps=order + 1,
        freq=unique_freq,
        gain=unique_gain,
        fs=2.0
    )

    a = 1.0

    # zero-phase filtering (double pass)
    # axis=-1 — times
    filtered_signal = filtfilt(b, a, data, axis=-1)

    return filtered_signal

def detect_artifacts_threshold(epochs, threshold_uV):
    """
    Use amplitude threshold for artifact detection
    """
    data = epochs.get_data(copy=True)  # форма: (n_epochs, n_channels, n_times)
    # Go through max amplitude in each ch and epoch
    # Max amplitude in each epoch
    max_amps = np.max(np.abs(data), axis=(1, 2))

    bad_epochs = max_amps > threshold_uV

    return bad_epochs, max_amps

def detect_artifacts_trend(epochs, trend_threshold_uVs):
    """
    Use slope for artifact detection
    """
    data = epochs.get_data(copy=True)
    n_epochs, n_channels, n_times = data.shape
    sfreq = epochs.info['sfreq']

    times = np.arange(n_times) / sfreq
    trends = np.zeros(n_epochs)

    for i in range(n_epochs):
        epoch_data = data[i]
        # Compute the slope in each ch
        channel_trends = []
        for ch in range(n_channels):
            slope, _, _, _, _ = stats.linregress(times, epoch_data[ch])
            channel_trends.append(abs(slope))
        trends[i] = np.max(channel_trends)
    bad_epochs = trends > trend_threshold_uVs

    return bad_epochs, trends


def detect_artifacts_diff(epochs, diff_threshold_uV):
    """
    Use the diff betw adjacent epochs for artifact detection
    """
    data = epochs.get_data(copy=True)
    n_epochs, n_channels, n_times = data.shape

    max_diffs = np.zeros(n_epochs)
    for i in range(n_epochs):
        epoch_data = data[i]
        diffs = np.diff(epoch_data, axis=1)  # (n_channels, n_times-1)
        max_diff = np.max(np.abs(diffs))
        max_diffs[i] = max_diff
    bad_epochs = max_diffs > diff_threshold_uV
    return bad_epochs, max_diffs

def calculate_rms_in_intervals(epochs, low_cutoff, high_cutoff, interval_prestim, interval_poststim, preamplifier):
    """
    Computes RMS of the signal in interval_prestim, interval_poststim
    """

    t_ranges = dict(
        prestim=interval_prestim,
        poststim=interval_poststim
    )

    idx_prestim = epochs.time_as_index(t_ranges['prestim'])
    idx_poststim = epochs.time_as_index(t_ranges['poststim'])

    data = epochs.get_data()
    data_prestim = data[:, :, idx_prestim[0]:idx_prestim[1]]
    data_poststim = data[:, :, idx_poststim[0]:idx_poststim[1]]

    rms_prestim = np.sqrt(np.mean(data_prestim ** 2, axis=-1))
    rms_poststim = np.sqrt(np.mean(data_poststim ** 2, axis=-1))

    if preamplifier:
        rms_prestim = 1e3 * rms_prestim
        rms_poststim = 1e3 * rms_poststim
    else:
        rms_prestim = 1e6 * rms_prestim
        rms_poststim = 1e6 * rms_poststim

    # Формируем результат
    rms_results = {
        'interval1': {
            'time_range': t_ranges['prestim'],
            'rms_values': rms_prestim
        },
        'interval2': {
            'time_range': t_ranges['poststim'],
            'rms_values': rms_poststim
        }
    }
    snr_db = calculate_snr(rms_poststim, rms_prestim)

    return rms_results, snr_db


def calculate_snr(rms_signal, rms_noise):
    """
    Computes SNR in dB
    """
    # Avoid division by zero
    rms_noise = np.where(rms_noise == 0, np.finfo(float).eps, rms_noise)

    # https://pubmed.ncbi.nlm.nih.gov/31505395/
    snr_ratio = np.mean(rms_signal) / np.mean(rms_noise)
    #https://ib-lenhardt.com/kb/glossary/snr
    # SNR_dB = 20 * log10(V_s / V_n) and SNR_dB = 10 * log10(P_s / P_n)
    snr_db = 10 * np.log10(snr_ratio)

    return snr_db

def trim_freq(freqs_data, cutoff_freq=50):
    """
    Trims the left part of psd with noise
    """
    index = next(i for i, value in enumerate(freqs_data) if value >= cutoff_freq)

    return index

def compute_GA(epochs, fs, preamplifier, noise, tmin):
    """
    Preprocessing 4: Grand Average
    """
    ddata = epochs.get_data()

    info = mne.create_info(
        ch_names=['Cz'],
        sfreq=fs,
        ch_types='eeg'
    )

    evokeds = []
    for j in range(ddata.shape[0]):  # по всем epochs
        data_epoch = ddata[j, :]
        if noise:
            # Take -100 0 ms
            prestim_interval = 0.1 * fs
            data_epoch_tr = data_epoch[:, 0:int(prestim_interval)]
        else:
            data_epoch_tr = data_epoch
        if preamplifier:
            data = data_epoch_tr * 1e-3
        else:
            data = data_epoch_tr

        evoked = mne.EvokedArray(
            data=data,
            info=info,
            tmin=tmin
        )
        evokeds.append(evoked)
    grand_average = mne.grand_average(evokeds)

    return grand_average

def plot_stim(stimulus, ax, tmin, tmax, fs_stim, ts):
    """Plot stim"""
    data = stimulus.copy()
    data_stim = data[:, 0]

    # Adjust stim to the timing of ffr epoch
    stim_duration_samples = ts * fs_stim
    n_zeros_front = int(-tmin * fs_stim)
    n_zeros_back = int(tmax * fs_stim - stim_duration_samples)

    data_stim_padded = np.concatenate([
    np.zeros(n_zeros_front),
         data_stim[:int(stim_duration_samples)],
         np.zeros(n_zeros_back)
        ])
    n_points = data_stim_padded.shape[0]
    times_stim = np.linspace(tmin, tmax, n_points, endpoint=False)

    ax.plot(times_stim, data_stim_padded, color='green', linewidth=1.5, label='Stimulus')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(x=ts, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax.set_xlim(tmin, tmax)
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xlabel('Time, s', loc='left', fontsize=10)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))

def plot_GA(dummy, short, grand_avg, to_GA, ax, ts, tmin, fs):
    """plot Grand Average"""
    if to_GA:
        info = mne.create_info(
        ch_names=['Cz'],
        sfreq=fs,
        ch_types='eeg'
        )
        evoked = mne.EvokedArray(
            data=grand_avg,
            info=info,
            tmin=tmin
        )
        grand_avg = evoked
    grand_avg.plot(
        spatial_colors=False,
        gfp=False,
        axes=ax,
        show=False,
        verbose=None
    )
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(x=ts, color='red', linestyle='--', linewidth=2, alpha=0.8)

    if dummy:
        ax.set_ylim(-0.08, 0.08)
        ax.set_yticks([-0.08, 0.08])
    elif short:
        ax.set_ylim(-0.3, 0.3)
        ax.set_yticks([-0.3, 0.3])
    else:
        ax.set_ylim(-0.7, 0.7)
        ax.set_yticks([-0.7, 0.7])
    ax.axhline(
        y=0.00,
        color='grey',
        linestyle='--',
        linewidth=1,
        alpha=0.8
    )
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xlabel('Time, s', fontsize=10, loc='left')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))

def zero_padding(stimulus, ga, padding_factor):
    """
    Zero padding for fft
    """
    if ga:
        original_length = stimulus.shape[1]
    else:
        original_length = len(stimulus)
    target_length = original_length * padding_factor

    # Add zeros at the end of the signal
    if ga:
        n_channels, original_length = stimulus.shape
        zeros_to_add = target_length - original_length
        zeros_array = np.zeros((n_channels, zeros_to_add))
        stimulus_padded = np.concatenate([stimulus, zeros_array], axis=1)

    else:
        stimulus_padded = np.pad(stimulus, (0, target_length - original_length), mode='constant', constant_values=0)
        stimulus_padded = stimulus_padded[:, np.newaxis]

    return stimulus_padded
"""
def max_snr_psd(raw.info, fmin, fmax, fs, tmin):

    SNR

    freqs_sig = fmin, fmax
    freqs_noise = fmin-1, fmax+1


    ssd = SSD(
            info=raw.info,
            reg="oas",
            sort_by_spectral_ratio=False,
            filt_params_signal=dict(
                l_freq=freqs_sig[0],
                h_freq=freqs_sig[1],
                l_trans_bandwidth=1,
                h_trans_bandwidth=1,
        ),
            filt_params_noise=dict(
                l_freq=freqs_noise[0],
                h_freq=freqs_noise[1],
                l_trans_bandwidth=1,
                h_trans_bandwidth=1,
            ),
        )

    ssd.fit(X=raw.get_data())

    ssd_sources = ssd.transform(X=raw.get_data())

    psd, freqs = mne.time_frequency.psd_array_welch(
            ssd_sources, sfreq=raw.info["sfreq"], n_fft=4096
    )

    spec_ratio, sorter = ssd.get_spectral_ratio(ssd_sources)

    return psd, freqs, spec_ratio
"""
def plot_noise_PSD(dummy, short, grand_average, grand_average_noise, ax, method, fmin, fmax, fs, padding_factor, tmin):
    """
    Plot Spectral Amplitude of the FFR
    """
    to_GA = True
    ga_data_padded = zero_padding(grand_average.get_data(),  to_GA, padding_factor)

    info = mne.create_info(
        ch_names=['Cz'],
        sfreq=fs,
        ch_types='eeg'
    )
    evoked = mne.EvokedArray(
            data=ga_data_padded,
            info=info,
            tmin=tmin
    )
    psd = evoked.compute_psd(
        method=method,
        fmin=fmin,
        fmax=fmax,
        verbose=False
    )

    # Compute Spectral Amplitude
    # Convert PSD into Spectral Amplitude
    data_psd = psd.get_data()  # muV²/Hz
    data_amplitude = np.sqrt(data_psd).flatten()  * 1e6 # muV²/√Hz
    freqs_data = psd.freqs

    ga_noise_data_padded = zero_padding(grand_average_noise.get_data(), to_GA, padding_factor)
    info = mne.create_info(
        ch_names=['Cz'],
        sfreq=fs,
        ch_types='eeg'
    )
    evoked = mne.EvokedArray(
            data=ga_noise_data_padded,
            info=info,
            tmin=tmin
    )
    psd_noise = evoked.compute_psd(
        method=method,
        fmin=fmin,
        fmax=fmax,
        verbose=False
    )

    # Compute and plot Noise Spectral Amplitude
    data_psd_noise = psd_noise.get_data()

    data_noise_amplitude = np.sqrt(data_psd_noise).flatten() * 1e6
    freqs_noise = psd_noise.freqs

    # Plot Spectral Amplitude
    trim_index_data = trim_freq(freqs_data)
    trim_index_noise = trim_freq(freqs_noise)

    ax.plot(freqs_data[trim_index_data:], data_amplitude[trim_index_data:], 'b-', label='FFR', linewidth=1.5)
    ax.plot(freqs_noise[trim_index_noise:], data_noise_amplitude[trim_index_noise:], 'r-', label='Noise  -100 0 ms', linewidth=1.5)
    ax.legend(loc='upper right')

    ax.set_xlabel('Hz', loc = 'right')
    ax.set_ylabel('μV', fontsize=10, labelpad=-10)

    formatter = ticker.ScalarFormatter(useOffset=False, useMathText=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)

    if dummy:
        ax.set_ylim(0.0, 0.09)
        ax.set_yticks([0.0, 0.09])
        ax.axhline(
            y=0.045,
            color='purple',
            linestyle='--',
            linewidth=1,
            alpha=0.8
        )
    elif short:
        ax.set_ylim(0.0, 0.3)
        ax.set_yticks([0.0, 0.3])
    else:
        ax.set_ylim(0.0, 1.2)
        ax.set_yticks([0.0, 1.2])

    ax.grid(True, alpha=0.3)

def plot_stim_PSD(stimulus, sinus_tone, frequencies, ax, method, fmin, fmax, fs_stim, padding_factor):
    """
    Plot Spectral Amplitude of the stimulus
    """
    if  sinus_tone:
        data_stim = stimulus
    else:
        data_stim = stimulus[:, 0]

    to_GA = False
    data_stim_padded = zero_padding(data_stim, to_GA, padding_factor)


    info = mne.create_info(
        ch_names=['Cz'],
        sfreq=fs_stim,
        ch_types='eeg'
    )

    evoked_stim = mne.EvokedArray(
            data=np.transpose(data_stim_padded),
            info=info,
            tmin=0
        )
    psd = evoked_stim.compute_psd(
        method=method,
        fmin=fmin,
        fmax=fmax,
        verbose=False
    )

    data_psd = psd.get_data()
    data_amplitude = np.sqrt(data_psd).flatten()
    freqs_data = psd.freqs

    trim_index = trim_freq(freqs_data)

    ax.plot(freqs_data[trim_index:], data_amplitude[trim_index:], 'g-', linewidth=2.0)

    if sinus_tone:
        colors = ['magenta', 'orange', 'blue', 'green']
        for idx, frequency in enumerate(frequencies):
            ax.axvline(
                x=frequency,
                alpha=0.3,
                color=colors[idx % len(colors)],
                label=f'F{idx}: {frequency} Гц',
                linewidth=2.5
            )
        ax.set_xlabel('Stimulus Spectra, Hz', fontsize=10, loc='left')
    ax.set_xlabel('Hz', loc='right', fontsize=10)
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=10)
    #if sinus_tone:
    #    plt.show()

def import_and_epoch(fname_bdf, non_filt, use_non_filt, n_6low, n_7low, preamplifier, dummy, fmin, fmax, order, transition_width,
                     tmin, tmax,
                     AMP_THRESHOLD, TREND_THRESHOLD, DIFF_THRESHOLD,
                     multiplier):

    _, raw_to_epo, events, event_dict, label_6, label_7 = import_raw(fname_bdf, non_filt, use_non_filt, preamplifier, dummy, fmin, fmax, order, transition_width)
    fs = raw_to_epo.info.get('sfreq')

    # Preprocessing 2: Epoching with baseline
    available_6low, available_7low, sorted_events = select_events(n_6low, n_7low, label_6, label_7, events, event_dict)

    # Data segmentation (epoching)
    epochs = mne.Epochs(
            raw_to_epo,
            sorted_events,
            tmin=tmin,
            tmax=tmax,
            baseline=(tmin, 0),
            preload=True
    )
    # Preprocessing 3: Cleaning
    if use_non_filt:
        return epochs, [], fs
    else:
        epochs_clean, bad_indices = remove_artifacts(epochs, AMP_THRESHOLD, TREND_THRESHOLD, DIFF_THRESHOLD,
                                               multiplier)
        return epochs_clean, bad_indices, fs

def process_plot_filt(axes, stim_type, fname_stim, fname_bdf, base_path, subject, short, non_filt, n_6low, n_7low, preamplifier, dummy, fmin, fmax, method,  order, ts, tmin, tmax, transition_width,
                          AMP_THRESHOLD, TREND_THRESHOLD, DIFF_THRESHOLD, multiplier, average_out, padding_factor, use_non_filt):
    # Preprocessing 1: Import Raw
    epochs, bad_indices, fs = import_and_epoch(fname_bdf, non_filt, use_non_filt, n_6low, n_7low, preamplifier, dummy, fmin, fmax, order,
                                            transition_width,
                                            tmin, tmax,
                                            AMP_THRESHOLD, TREND_THRESHOLD, DIFF_THRESHOLD,
                                            multiplier)

    sumeve = n_6low + n_7low
    sampl_freq_stim, stimulus = wavfile.read(fname_stim)


    stimulus_corr = trim_stim(stimulus, ts, sampl_freq_stim)

    # 1st row, 1st col — Stimulus
    ax1 = axes[0, 0]
    plot_stim(stimulus_corr, ax1, tmin, tmax, sampl_freq_stim, ts)

    ax1.set_title(f'Stimulus {stim_type}', fontsize=12)

    # 1st row, 2d col — Spectral Amplitude of the Stimulus
    ax2 = axes[0, 1]
    sin_tone = False

    plot_stim_PSD(stimulus_corr, sin_tone, [], ax2, method, fmin, fmax, sampl_freq_stim, padding_factor)
    ax2.set_title(f'Spectra', fontsize=12)

    # 2d row, 1st col — Grand Average FFR
    ax3 = axes[1, 0]
    noise = False
    grand_average = compute_GA(epochs, fs, preamplifier, noise, tmin)
    if average_out:
        save_ga_in_edf(grand_average, base_path, subject, preamplifier, short, tmin, tmax, fmin, fmax, n_6low, n_7low)

    to_GA = False
    plot_GA(dummy, short, grand_average, to_GA, ax3, ts, tmin, fs)
    ax3.set_title(f'FFR {int(fmin)}-{int(fmax)} Hz', fontsize=12)

    # 2d row, 2d col — Spectral Amplitude FFR + Noise
    ax4 = axes[1, 1]
    noise = True
    grand_average_noise = compute_GA(epochs, fs, preamplifier, noise, tmin)
    plot_noise_PSD(dummy, short, grand_average, grand_average_noise, ax4, method, fmin, fmax, fs, padding_factor, tmin)
    ax4.set_title(f'Spectra', fontsize=12)

    return bad_indices

def process_plot_last_filt(axes, bad_indices, fname_bdf, non_filt, n_6low, n_7low, preamplifier, dummy, short, fmin, fmax, method,  order, ts, tmin, tmax, transition_width,
                          AMP_THRESHOLD, TREND_THRESHOLD, DIFF_THRESHOLD, multiplier, padding_factor, use_non_filt):

    epochs_nf, _, fs = import_and_epoch(fname_bdf, non_filt, use_non_filt, n_6low, n_7low, preamplifier, dummy, fmin, fmax, order,
                                            transition_width,
                                            tmin, tmax,
                                            AMP_THRESHOLD, TREND_THRESHOLD, DIFF_THRESHOLD,
                                            multiplier)

    epochs_nf.drop(bad_indices)

    noise = False
    grand_average_nf = compute_GA(epochs_nf, fs, preamplifier, noise, tmin)

    show_non_filt = True
    ax5 = axes[2, 0]
    if show_non_filt:
        to_GA = False
        plot_GA(dummy, short, grand_average_nf, to_GA, ax5, ts, tmin, fs)
        ax5.set_title('FFR non-filtered', fontsize=12)
        # 3d row, 2d col — Spectral Amplitude non-filt FFR (with noise)
        ax6 = axes[2, 1]
        noise = True
        grand_average_nf_noise = compute_GA(epochs_nf, fs, preamplifier, noise, tmin)
        plot_noise_PSD(dummy, short, grand_average_nf, grand_average_nf_noise, ax6, method, fmin, fmax, fs, padding_factor, tmin)
        ax6.set_title(f'Spectra', fontsize=12)
    else:
        #TODO
        data_filt_averaged = fir_bandpass_filter(grand_average_nf.data, fmin, fmax, fs, order, transition_width)
        #plot_stim(data_filt_averaged, ax5, tmin, tmax, fs, ts)
        to_GA = True
        plot_GA(dummy, short, data_filt_averaged,  to_GA, ax5, ts, tmin, fs)
        ax5.set_title('FFR filtered', fontsize=12, loc='left')

    plt.subplots_adjust(hspace=0.7, top=0.93, bottom=0.07)

def save_pdf(fig, output_dir, preamplifier, subject, short, n_6low, n_7low, fmin, fmax, ts, tmin, tmax):
    if preamplifier:
        fig.suptitle(f'FFR : {subject} with preamplifier MNSENS-ACP', fontsize=16, y=1.0)
    else:
        fig.suptitle(f'FFR : {subject} w/o preamplifier', fontsize=16, y=1.0)

    total_n = n_6low[0] + n_7low[0]
    output_path = os.path.join(output_dir,
                                   f'FFR_{subject}_{preamplifier}_{short}_{tmin}ms_{tmax}ms_FIR_{fmin}_{fmax}Hz_N{total_n}.pdf')
    # A4 in inches
    fig.set_size_inches(8.27, 11.69)
    fig.savefig(
        output_path,
        dpi=900,
        bbox_inches='tight',
        pad_inches=0.5,
        facecolor='white',
        edgecolor='none',
        format='pdf'  # Явно указываем формат
    )
    os.startfile(output_path)


def save_signal_plot(signal, filename, frequency, stimulus_duration, inter_stimulus_interval, num_repetitions):
    """
    An auxiliary function plots the stimulus
    """

    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.titlesize': 22
    })

    # Plot for the left and right channels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(40, 12))

    # Left channel (with the stimulus)
    # Plot first 1 sec
    ax1.plot(signal[:44000, 0], color='blue', linewidth=1)
    ax1.set_title('Left channel')
    ax1.set_ylabel('Amplitude: stim=np.int16(stim * 32767) ')
    ax1.grid(True)
    ax1.set_ylim(-35000, 35000)  # Фиксированный масштаб для левого подграфика

    # Right channel (with the triggers)
    # Plot first 1 sec
    ax2.plot(signal[:44000, 1], color='red', linewidth=1)
    ax2.set_title('Right channel')
    ax2.set_xlabel('Samples (K)')
    ax2.grid(True)
    ax2.set_ylim(-35000, 35000)  # Фиксированный масштаб для правого подграфика

    fig.suptitle(
        f'{frequency} Hz, TS {stimulus_duration}ms, TP {inter_stimulus_interval}ms,{num_repetitions} repetitions',
        fontsize=22, fontweight='bold')

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def make_pause(inter_stimulus_interval, sample_rate, p):
    """
    The function makes an interstimulus interval of a varying length (+- 20% of TP)
    """
    percent = inter_stimulus_interval * p
    var = random.randint(-percent, percent)
    pause = inter_stimulus_interval + var

    pause_ms = pause / 1000
    n_samples = int(sample_rate * pause_ms)
    t_silence = np.linspace(0, pause_ms, n_samples, endpoint=False)
    return t_silence


def make_ramp_window(stimulus_duration, sample_rate, rate, growth_rate):
    """
    The function makes a ramp increase of the stimulus using exp and first 10% of the stim
    growth_rate: coefficient of ramp rate
    """
    stim_ms = stimulus_duration / 1000
    n_samples = int(sample_rate * stim_ms)
    t_stim = np.linspace(0, stim_ms, n_samples, endpoint=False)

    ramp_duration_samples = int(len(t_stim) * rate)
    ramp_window = np.ones_like(t_stim)

    # Exponеntial increase of a stimulus, intense at the beggining
    x = np.linspace(0, 1, ramp_duration_samples)
    exp_ramp = (np.exp(growth_rate * x) - 1) / (np.exp(growth_rate) - 1)
    ramp_window[:ramp_duration_samples] = exp_ramp

    #  Exponеntial decay of a stimulus
    x = np.linspace(0, 1, ramp_duration_samples)
    exp_decay = (np.exp(growth_rate * (1 - x)) - 1) / (np.exp(growth_rate) - 1)
    ramp_window[-ramp_duration_samples:] = exp_decay
    return ramp_window, t_stim


def add_triggers(stimulus, sinus, inv, sample_rate):
    """
    Function to make 2 channels, inserts triggers at the start and at the end of the right channel.
    Returns signal with 2 channels - first for the stimuli and the second one  for the triggers
    add 3bit commands as in https://github.com/mcsltd/AStimWavPatcher/tree/master?tab=readme-ov-file
    """

    _SILENCE = 1

    if sinus:
        # int16 format
        stimulus = np.int16(stimulus * 32767)

    # Make 2 channels
    size = len(stimulus)
    left = stimulus.copy()

    right = np.zeros(size, dtype=np.int16)

    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min

    # Add triggers
    if inv:
        # 110 - set trigger 7 LOW (HIGH (default))
        right[_SILENCE + 0] = max_int16
        right[_SILENCE + 1] = min_int16
        right[_SILENCE + 2] = max_int16
        right[_SILENCE + 3] = min_int16
        right[_SILENCE + 4] = min_int16
        right[_SILENCE + 5] = max_int16

        # 111 - set trigger 7 HIGH (default)
        right[size - 6] = max_int16
        right[size - 5] = min_int16
        right[size - 4] = max_int16
        right[size - 3] = min_int16
        right[size - 2] = max_int16
        right[size - 1] = min_int16

    else:
        # 100 - set trigger 6 LOW (HIGH (default))
        right[_SILENCE + 0] = max_int16
        right[_SILENCE + 1] = min_int16
        right[_SILENCE + 2] = min_int16
        right[_SILENCE + 3] = max_int16
        right[_SILENCE + 4] = min_int16
        right[_SILENCE + 5] = max_int16

        # 101 - set trigger 6 HIGH (default)
        right[size - 6] = max_int16
        right[size - 5] = min_int16
        right[size - 4] = min_int16
        right[size - 3] = max_int16
        right[size - 2] = max_int16
        right[size - 1] = min_int16

    return np.column_stack([left, right])

def check_triggers(stimulus, stim_triggers, inv):
    """
    Checks if stim contains triggers
    """

    _SILENCE = 1
    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min

    size = len(stimulus)

    # Шаблоны для разных режимов
    patterns = {
        True: {  # inv == 1
            'start': [max_int16, min_int16, max_int16, min_int16, min_int16, max_int16],  # 110
            'end':   [max_int16, min_int16, max_int16, min_int16, max_int16, min_int16]   # 111
        },
        False: { # inv == 0
            'start': [max_int16, min_int16, min_int16, max_int16, min_int16, max_int16],  # 100
            'end':   [max_int16, min_int16, min_int16, max_int16, max_int16, min_int16]   # 101
        }
    }

    # Получаем шаблон для текущего режима
    pattern = patterns[bool(inv)]

    # Проверяем начальную последовательность
    start_indices = range(_SILENCE, _SILENCE + 6)
    start_sequence = [stim_triggers[i] for i in start_indices]
    if start_sequence != pattern['start']:
        return False

    # Проверяем конечную последовательность
    end_indices = range(size - 6, size)
    end_sequence = [stim_triggers[i] for i in end_indices]
    if end_sequence != pattern['end']:
        return False

    return True

def make_inv_stimulus(stimulus):
    return (-1) * stimulus


def trim_stim(stimulus, stimulus_duration, sample_rate):
    required_length = int((stimulus_duration) * sample_rate)
    return stimulus[:required_length]