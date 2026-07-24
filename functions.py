import config as cfg
import numpy as np
import os
import mne
import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.table import Table
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from pypdf import PdfReader, PdfWriter
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from mne.decoding import *
import random
from scipy import stats
from scipy.stats import pearsonr
from scipy.signal import correlate
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
from scipy.io import wavfile
from scipy.io.wavfile import write
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import re
import warnings
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    module='mne.*'  # или 'pandas.*' и т. д.
)
warnings.filterwarnings("ignore", message=".*EDF format requires equal-length data blocks.*")

info = cfg.info
info_wav = cfg.info_wav
info1ch = cfg.info1ch
fs_wav = cfg.fs_wav
fs = cfg.fs
ref_chs = cfg.ref_chs
sound_delay = cfg.sound_delay
n_fft = cfg.n_fft
n_per_seg = cfg.n_per_seg
n_overlap = cfg.n_overlap
freq_res = cfg.freq_res

def add_triggers(stimulus, sin_tone, inv, sample_rate):
    """
    Function to make 2 channels, inserts triggers at the start and at the end of the right channel.
    Returns signal with 2 channels - first for the stimuli and the second one  for the triggers
    add 3bit commands as in https://github.com/mcsltd/AStimWavPatcher/tree/master?tab=readme-ov-file
    """
    _SILENCE = cfg._SILENCE
    max_int16 = cfg.max_int16
    min_int16 = cfg.min_int16

    if sin_tone:
        # int16 format
        stimulus = np.int16(stimulus * 32767)

    # Make 2 channels
    size = len(stimulus)
    left = stimulus.copy()

    right = np.zeros(size, dtype=np.int16)

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

def average_and_filter_epochs(data_clean, fmin, fmax, tmin, order):
    """
    Function to average epochs across channels anbd filter
    """
    # Mean over epochs, axis=0
    mean_data = np.mean(data_clean, axis=0)
    filtered_data = butter_bandpass_filter(mean_data, fmin, fmax, order=order)
    # Average over chans
    filtered_data_m = np.mean(filtered_data, axis=0)

    grand_average = mne.EvokedArray(
        data=filtered_data_m[np.newaxis, :],
        info=info1ch,
        tmin=tmin
    )

    return grand_average

def butter_bandpass_filter(data, lowcut, highcut, order):
    """
    Butterworth filter for the data as in  doi: 10.1016/j.heares.2019.107779
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

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

def clean_epochs(epochs, tmin):
    """
    Function to clean epochs by amplitude (minus cfg.trim_epo_share of epochs with max amps)
    """
    data_stack = epochs.get_data()
    # Max_amps: in each epoch - over time points ansd chans
    max_amps = np.max(np.abs(data_stack), axis=(1, 2))
    if cfg.hexagone:
        # Number of epochs to drop
        n_drop = int(np.ceil(cfg.trim_epo_share * data_stack.shape[0]))
        # Epochs with largest amps
        drop_idx = np.argsort(max_amps)[-n_drop:]
        # Remove noisy epochs
        keep_mask = np.ones(data_stack.shape[0], dtype=bool)
        keep_mask[drop_idx] = False
    else:
        drop_mask = max_amps > cfg.amp_threshold
        drop_idx = np.where(drop_mask)[0]
        keep_mask = ~drop_mask

    data_clean = data_stack[keep_mask]

    print(f"Number of removed epochs: {drop_idx.size} from {data_stack.shape[0]} (amp treshold : {cfg.amp_threshold}) V")

    if cfg.hexagone:
        epochs_clean = mne.EpochsArray(
        #over chans
        data=data_clean,
        info=info1ch,
        tmin = tmin,
        event_id=None,
        verbose=False
        )
    else:
        epochs_clean = mne.EpochsArray(
        #over chans
        data=data_clean,
        info=info,
        tmin = tmin,
        event_id=None,
        verbose=False
        )
    return data_clean, epochs_clean

def compute_GA(epochs, tmin, fmin, fmax, order):
    """
    Preprocessing 4: Grand Average
    """
    data_clean, epochs_clean = clean_epochs(epochs, tmin)
    grand_average = average_and_filter_epochs(data_clean, fmin, fmax, tmin, order)

    return grand_average, epochs_clean

def count_wav_triggers_optimized(wav_fname):
    """
    Counts triggers in a wav
    """

    _, stim_triggers = wavfile.read(wav_fname)
    trigger_signal = stim_triggers[:, 1]

    sequences = cfg.sequences

    results = {}
    for seq_name, seq_data in sequences.items():

        seq = np.array(seq_data)
        corr = correlate(trigger_signal, seq, mode='valid')

        # Absolute match
        target_corr = np.sum(seq ** 2)

        matches = np.where(corr == target_corr)[0]

        results[seq_name] = {
            'count': len(matches),
            'matches': matches.tolist()
        }

    return results

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

    # Make a ramp window
    # Create one original stimulus and an inverted stimulus
    ramp_window, t_stim = make_ramp_window(stimulus_duration, sample_rate, rate = 0.1 , growth_rate = 3.0)
    sin_tone = True
    sinus, inv_sinus = make_stimulus(t_stim, sin_tone, [], add_inv, ramp_window, frequencies, A)

    plot_stim_psd = False
    if plot_stim_psd:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        spectra_corr = 0
        plot_stim_PSD(axes, cfg.base_path,  spectra_corr, sinus, sin_tone, frequencies, fmin=min(frequencies), fmax=max(frequencies), padding_factor=32)

    # Make full signal: stimulus + pause + inv stimulus + pause , N reps
    full_signal = make_full_signal(inter_stimulus_interval, sinus, inv_sinus, sin_tone, add_inv, num_repetitions, sample_rate, cfg.percent_var_pause)

    # Save the WAV and the plot of stimuli
    base_name = f'sin_{frequencies}Hz_TS{stimulus_duration:.1f}s_TP{inter_stimulus_interval:.1f}s_N{num_repetitions}_INV{add_inv}'
    save_wav_output(base_name, dir, full_signal, sample_rate, frequencies, stimulus_duration, inter_stimulus_interval, num_repetitions)

def create_repeated_da_syllable_wav(
        dir,
        frequencies,
        stimulus_duration,
        inter_stimulus_interval,
        num_repetitions,
        sample_rate,
        add_inv,
        A,
        wavfname

    ):
    """
    Creates WAV‑файл with syllables.
    """

    # Create one stimulus
    fs, syllable = wavfile.read(wavfname)

    plot_PSD = False
    if plot_PSD:
        sin_tone = True
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        spectra_corr = 0
        plot_stim_PSD(axes, cfg.base_path,  spectra_corr, syllable, sin_tone, frequencies, fmin=min(frequencies), fmax=max(frequencies), padding_factor=32)
    sin_tone = False
    # Make a full stimulation: stimulus + pause + inv stimulus + pause , N repetitions
    ramp_window, t_stim = make_ramp_window(len(syllable) / sample_rate * 1000, sample_rate, rate=0.1, growth_rate=3.0)
    stimulus, inv_stimulus = make_stimulus(t_stim, sin_tone, syllable, add_inv, ramp_window, frequencies, A)

    # Make full signal: stimulus + pause + inv stimulus + pause , N reps
    full_signal = make_full_signal(inter_stimulus_interval, stimulus, inv_stimulus, sin_tone, add_inv, num_repetitions, sample_rate, cfg.percent_var_pause)
    base_name = f'Da_{frequencies}Hz_TS{stimulus_duration:.1f}s_TP{inter_stimulus_interval:.1f}s_N{num_repetitions}_INV{add_inv}'
    save_wav_output(base_name, dir, full_signal, sample_rate, frequencies, stimulus_duration, inter_stimulus_interval, num_repetitions)

def create_section_table(header_text, rows_data, styles, colWidths):
    """
    Creates a table for pdf report.

    :param header_text: Header string
    :param rows_data: [("Label", "Value"), ...]
    :param styles: ReportLab (styles = getSampleStyleSheet())
    :param colWidths: example [0.4 * usable_width, 0.6 * usable_width]
    """
    data = []
    data.append([Paragraph(f"<b>{header_text}</b>", styles['Normal'])])

    for label, value in rows_data:
        p_label = Paragraph(f"<b>{label}:</b>", styles['Normal'])
        p_value = Paragraph(str(value), styles['Normal'])
        data.append([p_label, p_value])

    t = Table(data, colWidths=colWidths)

    ts = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('LEADING', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('GRID', (0, 1), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ])
    t.setStyle(ts)
    return t

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
        if isinstance(n, (int, np.integer, float)):
            n_int = int(n)
        else:
            n_int = int(n[0])
        selected_indices = indices[:n_int]

    return available_count, events[selected_indices], selected_indices

def find_nearest_freq(amp_stim, freq_stim, amp_ffr, freq_ffr):
    """
    Find nearest stim frequency to ffr frequency with largest amplitude
    """
    pairs = []
    used_ffr_indices = set()

    # Sort indices using amps
    sorted_stim_indices = np.argsort(amp_stim)

    for idx in sorted_stim_indices:
        stim_val = freq_stim[idx]
        stim_amp = amp_stim[idx]

        # Find nearest FFR freq
        diffs = np.abs(freq_ffr - stim_val)
        best_idx = np.argmin(diffs)
        min_diff = diffs[best_idx]

        #2 * freq_res = 20 Hz
        if min_diff <= 2 * freq_res:
            if best_idx in used_ffr_indices:
                continue

            used_ffr_indices.add(best_idx)

            pair = {
                'stim_freqs': stim_val,
                'stim_amps': stim_amp,
                'ffr_freqs': freq_ffr[best_idx],
                'ffr_amps': amp_ffr[best_idx],
                'diff': min_diff
            }
            pairs.append(pair)
            print(
                f"Stimulus {stim_val:.1f} Hz (Amp: {stim_amp:.5f}) -> Nearest FFR {freq_ffr[best_idx]:.1f} Hz {amp_ffr[best_idx]} ({min_diff:.5f} Hz)")
        else:
            print(f"For stimulus {stim_val:.1f} Hz there is no available spectral peak in FFR (min. difference {min_diff:.2f} > the theashold)")

    return pairs

def find_harmonics(freq_ffr, freq_stim_to_corr):
    """
    Find nearest stim frequency to the first 3 harmonics of the first 5 frequencies in freq_to_corr.

    Parameters:
    freq_stim (array-like): Array of stimulus frequencies.
    freq_to_corr (array-like): Array of FFR frequencies to check (will take first 5, then first 3 harmonics).
    max_diff (float): Maximum allowed difference in Hz.

    Returns:
    list: List of dictionaries with 'stim_freq', 'ffr_freq', 'diff'.

    For the first 5 base frequencies from freq_to_corr, harmonics (1×, 2×, 3×) are generated.
    For each harmonic, the closest stimulus frequency from freq_stim is found, with a difference ≤ max_diff.
    Each stimulus frequency can be used only once.

    Returns a list of pairs: {stim_freq, ffr_freq, diff, base_freq, harmonic_num}.
    """

    pairs = []
    used_ffr_indices = set()  # индексы в freq_stim, которые уже использованы

    # 1. First 5 basic freqs
    base_freqs = np.array(freq_stim_to_corr)[:5]

    # 2. 1x, 2x, 3x
    harmonics_to_check = []
    for base in base_freqs:
        harmonics_to_check.extend([base * 1.0, base * 2.0, base * 3.0])
    harmonics_to_check = np.array(harmonics_to_check)

    freq_ffr = np.asarray(freq_ffr)

    for h_idx, h_val in enumerate(harmonics_to_check):
        best_ffr_idx = None
        min_diff = np.inf

        for f_idx, f_val in enumerate(freq_ffr):
            if f_idx in used_ffr_indices:
                continue

            diff = abs(f_val - h_val)
            if diff < min_diff:
                min_diff = diff
                best_ffr_idx = f_idx

        if best_ffr_idx is not None and min_diff <= freq_res:
            used_ffr_indices.add(best_ffr_idx)

            base_idx = h_idx // 3
            harmonic_num = (h_idx % 3) + 1

            pair = {
                'stim_freq': float(freq_ffr[best_ffr_idx]),
                'ffr_freq': float(h_val),
                'ffr_idx': int(best_ffr_idx),
                'diff': float(min_diff),
                'base_freq': float(base_freqs[base_idx]),
                'harmonic_num': harmonic_num
            }
            pairs.append(pair)

            print(f"[OK] Гармоника {harmonic_num}x ({h_val:.1f} Гц) <- FFR {freq_ffr[best_ffr_idx]:.1f} Гц (разница {min_diff:.2f} Гц)")
        else:
            base_idx = h_idx // 3
            harmonic_num = (h_idx % 3) + 1
            status = f"(Best stimulus {freq_ffr[best_ffr_idx]:.1f} Hz, difference {min_diff:.2f} Hz)" if best_ffr_idx is not None else "(no available stimuli)"
            print(f"[FAIL] Harmonic {harmonic_num}x ({h_val:.1f} Hz): o available stimuli {freq_res} Гц {status}")

    return pairs

def import_and_epoch(fname, ftype,  ch_name, non_filt, use_non_filt, n_6low, n_7low, label_6, label_7, base_path, dummy, fmin, fmax, order,
                     tmin, tmax,
                     AMP_THRESHOLD, TREND_THRESHOLD, DIFF_THRESHOLD):

    _, raw_to_epo, events, event_dict, eeg_registration = import_raw(fname, ftype, ch_name, non_filt, use_non_filt,
                                                                     base_path, dummy, fmin, fmax, order)

    # Preprocessing 2: Epoching with baseline
    available_6low, available_7low, adjusted_events_6low, adjusted_events_7low, sorted_events = select_events(n_6low, n_7low, label_6, label_7, events, event_dict)
    if cfg.substraction:
        epochs_6low = mne.Epochs(
            raw_to_epo,
            adjusted_events_6low,
            tmin=tmin,
            tmax=tmax,
            baseline=(tmin, 0 + sound_delay),
            preload=True
        )
        epochs_7low = mne.Epochs(
            raw_to_epo,
            adjusted_events_7low,
            tmin=tmin,
            tmax=tmax,
            baseline=(tmin, 0 + sound_delay),
            preload=True
        )
        epochs_6low_data = epochs_6low.get_data()
        epochs_7low_data = epochs_7low.get_data()
        data_sub = epochs_6low_data - epochs_7low_data
        if cfg.hexagone:
            epochs = mne.EpochsArray(
            data=data_sub,
            info=info1ch,
            tmin=tmin,
            verbose=False
            )
        else:
            epochs = mne.EpochsArray(
            data=data_sub,
            info=info,
            tmin=tmin,
            verbose=False
            )
    else:
        # Data segmentation (epoching)
        epochs = mne.Epochs(
            raw_to_epo,
            sorted_events,
            tmin=tmin,
            tmax=tmax,
            baseline=(tmin, 0 + sound_delay),
            preload=True
        )
    # Preprocessing 3: Cleaning
    if use_non_filt:
        return epochs, [], events, event_dict, eeg_registration
    else:
        epochs_clean, bad_indices = remove_artifacts(epochs, AMP_THRESHOLD, TREND_THRESHOLD, DIFF_THRESHOLD)

        return epochs_clean, bad_indices, events, event_dict, eeg_registration

def import_fif(fname, ch_name):
    """
    Import fif file
    """
    raw = mne.io.read_raw_fif(fname)
    raw.load_data()
    if cfg.hexagone:
        raw_ref = raw.copy().pick_channels(cfg.ref_chs)
        raw_ch = raw.copy().pick_channels(cfg.ch_name)
        raw_ch_data = raw_ch.get_data()
        raw_ref_data = raw_ref.get_data()
        raw_ch_data_av = np.mean(raw_ch_data, axis =0)
        raw_ch_data_minus_k3 = raw_ch_data_av[np.newaxis, ] - raw_ref_data
        raw_selected  = mne.io.RawArray(raw_ch_data_minus_k3, raw_ref.info)
    else:
        raw_selected = raw.copy().pick_channels(ch_name)
        raw_selected.set_eeg_reference(ref_channels=ref_chs, projection=False)

    return raw_selected , raw

def import_raw(fname, ftype, ch_name, non_filt, use_non_filt, base_path, dummy, fmin, fmax, order):
    """
    Imports and filters data if needed
    """
    if ftype == '.fif':
        raw_selected, raw = import_fif(fname, ch_name)
    else:
        assert(ftype == '.bdf')
        raw = mne.io.read_raw_bdf(
            fname,
            include=ch_name,
            preload=True,
            verbose=True
        )
        raw_selected = raw

    ctime = os.path.getctime(fname)
    creation_time = datetime.fromtimestamp(ctime)
    eeg_registration = creation_time.strftime('%Y-%m-%d %H:%M')

    if use_non_filt:
        raw_to_epo = raw_selected
    else:
        #filtered_signal = butter_bandpass_filter(raw.get_data(), fmin, fmax, order=order)
        #raw_to_epo = mne.io.RawArray(filtered_signal, raw.info)
        #filtered_signal = butter_bandpass_filter(raw_selected.get_data(), fmin, fmax, order=order)
        #raw_to_epo = mne.io.RawArray(filtered_signal, raw_selected.info)
        #filtered_signal = butter_bandpass_filter(raw_selected.get_data(), fmin, fmax, order=order)
        raw_to_epo = mne.io.RawArray(raw_selected.get_data(), raw.info)

    events, event_dict = mne.events_from_annotations(raw)

    return raw_selected, raw_to_epo, events, event_dict, eeg_registration

def load_raw_bdf(base_path):
    """
    Choose BDF/FIF from base_path.
    Returns file_path or None.
    """
    root = tk.Tk()
    root.withdraw()

    initial_dir = str(base_path)

    file_path_str = filedialog.askopenfilename(
        initialdir=initial_dir,
        title="Choose BDF/FIF file",
        filetypes=[
            ("BrainVision BDF files", "*.bdf"),
            ("MNE FIF files", "*.fif")
        ]
    )

    if not file_path_str:
        print("File selection cancelled")
        return None

    file_path = Path(file_path_str)
    ftype = file_path.suffix.lower()

    s = file_path.stem
    m_subj = re.search(r'S(\d+)', s)

    if m_subj:
        subject = f"S{m_subj.group(1)}"
    else:
        subject = 'S_not_defined'

    if 'preamplifier' in s:
        preamplifier = 'preamplifier'
    else:
        preamplifier = ' '
    if 'non_filt' in s:
        non_filt = 'non_filt'
    else:
        non_filt = ' '
    if 'short' in s:
        short = 'short'
    else:
        short = ' '
    if 'dummy' in s:
        dummy = 'dummy'
        subject = 'hardware noise'
    else:
        dummy = ' '

    print(f"Loaded: {subject}")
    output_dir = base_path.joinpath('pics', subject)
    os.makedirs(output_dir, exist_ok=True)

    return ftype, subject, preamplifier, dummy, non_filt, short, file_path, output_dir

def load_stim(base_path):
    """
    Choose WAV from base_path.
    Returns file_path or None.
    """
    root = tk.Tk()
    root.withdraw()

    initial_dir = str(base_path)

    file_path_str = filedialog.askopenfilename(
        initialdir=initial_dir,
        title="Choose WAV file",
        filetypes=[
            ("Audio ", "*.WAV")
        ]
    )

    if not file_path_str:
        print("File selection cancelled")
        return None

    stim_path = Path(file_path_str)

    return stim_path

def make_amps_z_score(amp_stim, amp_ffr):

    eps = 1e-12

    # Z‑score
    s_norm = (amp_stim - np.mean(amp_stim)) / (np.std(amp_stim) + eps)
    f_norm = (amp_ffr - np.mean(amp_ffr)) / (np.std(amp_ffr) + eps)

    return s_norm, f_norm

def make_inv_stimulus(stimulus):
    return (-1) * stimulus

def make_full_signal(inter_stimulus_interval, sinus, inv_sinus, sin_tone, add_inv,num_repetitions, sample_rate, percent_var_pause):
    """
    Concatenates stimuli with pauses in a full signal
    """
    all_stimuli = []

    # Reset ASTIM: false cycle
    inv = False
    _ = add_triggers(sinus, sin_tone, inv, sample_rate)

    # List of all stimuli : original and inverted
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

    random.shuffle(all_stimuli)

    full_signal = []
    for stim in all_stimuli:
        full_signal.append(stim)
        # Add a pause with a varying length
        t_silence = make_pause(inter_stimulus_interval, sample_rate, percent_var_pause)
        silence = np.zeros_like(t_silence)
        isi = np.column_stack([silence, silence])
        isi = np.int16(isi * 32767)
        full_signal.append(isi)
    full_signal = np.concatenate(full_signal)
    return full_signal

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

def make_prestim_poststim(tmin):
    """
    Function to make prestim and postim intervals
    """
    prestim_interval = (-tmin + sound_delay) * fs
    poststim_interval = round((0.05 - sound_delay) * fs)

    return round(prestim_interval), round(poststim_interval)

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

def make_stimulus(t_stim, sin_tone, syllable, add_inv, ramp_window, frequencies, A):
    """
    Function to make and original and inverted sinusoidal signals
    """

    if sin_tone:
        sinus = np.zeros_like(t_stim)
        amplitude_factor = [1.0, 0.7, 0.5, 0.3]
        for i in range(len(frequencies)):
            stimulus = amplitude_factor[i]  * ramp_window * np.sin(2 * np.pi * frequencies[i] * t_stim)
            sinus = sinus + stimulus
        sinus /= np.max(np.abs(sinus))
        sinus = A * sinus
    else:
        sinus = syllable
    if add_inv:
        inv_sinus = make_inv_stimulus(sinus)
    else:
        inv_sinus = []

    return sinus, inv_sinus

def make_stim_epochs(stim_padded, tmin,fmin, fmax, padding_factor, epochs_ffr):
    """
    Function to make stim epochs for coherence with epochs_ffr
    """
    evoked_stim = mne.EvokedArray(
        # (n_channels, n_samples)
        data=stim_padded[np.newaxis, :],
        info=info_wav,
        tmin=tmin
    )
    evoked_stim_resampled = evoked_stim.resample(fs, npad="auto")
    da_stim = evoked_stim_resampled.get_data()
    n_epochs = len(epochs_ffr)
    e_stim = []
    for n in range(n_epochs):
        e_stim.append(da_stim)

    e_stim_filtered = butter_bandpass_filter(e_stim, fmin, fmax, order=2)

    epochs_stim = mne.EpochsArray(
        data=e_stim_filtered,
        #info=info,
        info=info1ch,
        tmin=tmin,
        verbose=False
    )

    return epochs_stim

def morlet_psd_epochs(
        base_path,
        epochs_stim,
        epochs_ffr,
        r_amps,
        tmin
):
    """
    Computes morlet wavelets, R amps stim/ffr or relative spectral power of ffr
    """
    #amps_ffr_to_corr, freqs_ffr_to_corr = read_amps_freqs(base_path, tp='ffr')
    #amps_stim_to_corr, freqs_stim_to_corr = read_amps_freqs(base_path, tp='stim')

    file_path = os.path.join(base_path, "ffr_freqs_and_amps.txt")
    data = np.loadtxt(file_path, comments='#')
    freqs_ffr_to_corr = data[:, 0]
    amps_ffr_to_corr = data[:, 1]

    file_path = os.path.join(base_path, "stim_freqs_and_amps.txt")
    data = np.loadtxt(file_path, comments='#')
    freqs_stim_to_corr = data[:, 0]
    amps_stim_to_corr = data[:, 1]

    pairs = find_nearest_freq(amps_stim_to_corr, freqs_stim_to_corr, amps_ffr_to_corr, freqs_ffr_to_corr)
    data = [(p['stim_freqs'], p['stim_amps'], p['ffr_freqs'], p['ffr_amps']) for p in pairs]

    stim_freqs = np.array([x[0] for x in data])
    ffr_freqs = np.array([x[2] for x in data])

    # For each freq [f-10, f, f+10]
    step = freq_res
    ffr_step = np.concatenate([ffr_freqs - step, ffr_freqs, ffr_freqs + step])
    ffr_grid = np.unique(ffr_step)

    stim_step = np.concatenate([stim_freqs - step, stim_freqs, stim_freqs + step])
    stim_grid = np.unique(stim_step)

    n_cycles = np.clip(ffr_grid * cfg.dt_target, 4, 80)

    tfr_ffr = mne.time_frequency.tfr_morlet(
        epochs_ffr,
        freqs=ffr_grid,
        n_cycles=n_cycles,
        return_itc=False,  # Instantaneous phase coupling (если нужно)
        average=False,  # усреднить по эпохам
        decim=1,  # прореживать, если данных много
        use_fft=False,
        picks=None  # каналы
    )

    if r_amps:
        tfr_stim = mne.time_frequency.tfr_morlet(
        epochs_stim,
        freqs=stim_grid,
        n_cycles=n_cycles,
        return_itc=False,  # Instantaneous phase coupling (если нужно)
        average=False,  # усреднить по эпохам
        decim=1,  # прореживать, если данных много
        use_fft=False,
        picks=None  # каналы
        )
        amp_mean_stim = np.sum(tfr_stim.data, axis=2)
        amp_mean_ffr = np.sum(tfr_ffr.data, axis=2)
        s_norm, f_norm = make_amps_z_score(amp_mean_stim, amp_mean_ffr)
        f_norm_resh = f_norm.reshape(-1, 1)
        s_norm_resh = s_norm.reshape(-1, 1)
        corr_coeff, p_val = pearsonr(f_norm_resh, s_norm_resh)

        return np.mean(corr_coeff), p_val
    else:
        # Relative spectral power
        # tfr_stim.data (510, 1, 12, 4001), axis=2 - freqs

        power_ffr = tfr_ffr.data
        power_sum_ffr  = power_ffr.sum(axis=-2)

        t_baseline_mask = (tfr_ffr.times >= tmin) & (tfr_ffr.times < 0)
        baseline_power_ffr  = power_sum_ffr[:, :, t_baseline_mask].mean(axis=-1, keepdims=True)  # усредняем по baseline
        ffr_power_db = 10 * np.log10(power_sum_ffr  / baseline_power_ffr )  # относительные dB

        return np.mean(ffr_power_db), []

def plot_GA(grand_avg, to_GA, ax, ts, tmin):
    """plot Grand Average"""
    if to_GA:
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
    # Remove text with 'Nave' (или 'N$_{\mathrm{ave}}$')
    for txt in ax.texts[:]:  # копия списка, чтобы безопасно удалять
        if 'Nave' in txt.get_text() or 'N$_{\\mathrm{ave}}' in txt.get_text():
            txt.remove()
    #ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    #ax.axvline(x=ts, color='red', linestyle='--', linewidth=2, alpha=0.8)
    # ==========================================
    # Autoresize of Y axis
    # ==========================================

    data = grand_avg.get_data()
    max_val = np.max(np.abs(data))
    # max_val = 0.5
    # Keep the graph away from the edges
    y_limit = max_val * 1.5 * 1e6
    ax.set_ylim(-y_limit, y_limit)

    y_tick_val = round(y_limit, 1)
    ax.set_yticks([-y_tick_val, y_tick_val])

    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xlabel('Time, ms', loc='left', fontsize=10)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x * 1000):d}'))
    ax.grid(True, alpha=0.3)

def plot_noise_PSD(ax, base_path, spectra_corr, grand_average, fmin, fmax, padding_factor, tmin):
    """
    Plot Spectral Amplitude of the FFR
    """
    trimmed_ga = trim_ga(grand_average.get_data(), tmin)

    to_GA = True
    ga_data_padded = zero_padding(trimmed_ga, to_GA, padding_factor)
    evoked = mne.EvokedArray(
        data=ga_data_padded,
        info=info1ch,
        # info=info,
        tmin=0
    )

    psd = evoked.compute_psd(
        method='welch',
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,  # без zero-padding
        n_per_seg=n_per_seg,  # длиннее сегмент → лучше разрешение
        n_overlap=n_overlap,  # 50% перекрытия
        verbose=False
    )

    # Frequency resolution: 10_000 / 1024 ≈ 9.77 Гц

    # Compute Spectral Amplitude
    # Convert PSD into Spectral Amplitude
    data_psd = psd.get_data()  # V²/Hz

    data_amplitude = np.sqrt(data_psd).flatten() * 1e6  # muV/√Hz
    freqs_data = psd.freqs

    # Plot Spectral Amplitude
    trim_index_data = trim_freq(freqs_data)

    data_slice = data_amplitude[trim_index_data:]
    freq_slice = freqs_data[trim_index_data:]

    if spectra_corr:
        ax.plot(freq_slice, data_slice, 'b-', linewidth=1.5)
        ax.set_ylabel('muV/√Hz', fontsize=8, labelpad=1)
    y_max = data_slice.max()
    amp_ffr_to_corr, freqs_ffr_to_corr = plot_spectra_with_freq_vals(ax, spectra_corr, y_max, freq_slice, data_slice)

    filename = os.path.join(base_path, "ffr_freqs_and_amps.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# FFR amplitudes and frequencies for correlation\n")
        f.write("# Format: frequency (Hz) | amplitude\n\n")

        for freq, amp in zip(freqs_ffr_to_corr, amp_ffr_to_corr):
            f.write(f"{freq:.3f} {amp:.3f}\n")
    #tp = 'ffr'
    #write_amps_freqs(freqs_ffr_to_corr, amp_ffr_to_corr, tp, base_path)

    return data_slice, freq_slice

def plot_spectral_correlation(ax, r, pval, r_amps, N):
    """
    Plot correlation of stim and response in the time domain
    """

    r = np.abs(r)
    if len(pval):
        if pval < 0.05:
            # p-val is significant
            ax.plot(N, r, color='green', marker='o', markersize=10)
        else:
            ax.plot(N, r, color='red', marker='o', markersize=10)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10,
                   label='R significant, p-val<0.05'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='R insignificant'),
        ]
        ax.legend(handles=legend_elements, loc='best')
    else:
        ax.plot(N, r, color='darkgrey', marker='o', markersize=10)
    ax.set_xlabel('N averages')
    if r_amps:
        ax.set_ylabel('R amps z-scored')
    else:
        ax.set_ylabel('PSD baselined [-100 0] ms, dB')
    ax.set_xticks([0, 250, 500, 1000, 2000, 3000, 4000])
    #ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])

    ax.grid(True, which='both', linestyle='--', alpha=0.5)

def plot_spectra_with_freq_vals(ax, spectra_corr,  y_top, freq_slice, data_slice):
    """
    Plots spectra with frequency labels
    """
    delta_f = freq_slice[1] - freq_slice[0]

    distance_bins = max(1, int(np.ceil(cfg.min_freq_gap / delta_f)))

    peaks_idx, _ = find_peaks(data_slice, distance=distance_bins, prominence=None)

    sorted_order = np.argsort(data_slice[peaks_idx])[::-1]
    top_peaks_indices = peaks_idx[sorted_order[:cfg.n_peaks]]

    sort_by_freq_order = np.argsort(freq_slice[top_peaks_indices])
    final_peaks_to_plot = top_peaks_indices[sort_by_freq_order]

    y_max = max(data_slice)
    y_top = y_max * 1.1
    amp_to_corr = []
    freq_to_corr = []
    for i, idx in enumerate(final_peaks_to_plot):
        freq_val = freq_slice[idx]
        freq_display = np.round(freq_val)
        freq_to_corr.append(freq_display)

        amp_val = data_slice[idx]
        amp_to_corr.append(amp_val)

        if spectra_corr:
            ax.axvline(
            x=freq_val,
            alpha=0.8,
            label=f'{i + 1}: {freq_display} Hz',
            linewidth=1.0,
            linestyle='--'
            )

            ax.text(
            x=freq_val,
            y=y_top,
            s=f'{i + 1}',
            ha='center',
            va='bottom',
            fontsize=6,
            color='blue'
            )

        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.set_xlabel('Stimulus Spectra, Hz', fontsize=10, loc='left')
        ax.legend(fontsize=6)
        ax.set_xlabel('Hz', loc='right', fontsize=10)
        ax.set_yticks([])
        ax.set_ylim(0, y_top)
        label_val = round(y_top, 3)
        ax.set_yticks([0, label_val])
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=10)

    return amp_to_corr, freq_to_corr

def plot_stim(stimulus, ax, tmin, tmax, ts):
    """Plot stim"""
    data = stimulus.copy()
    data_stim = data[:, 0]

    # Adjust stim to the timing of ffr epoch
    stim_duration_samples = ts * fs_wav
    n_zeros_front = int(-tmin * fs_wav)
    # more points to be consistent with epochs_ffr
    n_zeros_back = int(tmax * fs_wav - stim_duration_samples + 4)

    data_stim_padded = np.concatenate([
        np.zeros(n_zeros_front),
        data_stim[:int(stim_duration_samples)],
        np.zeros(n_zeros_back)
    ])
    n_points = data_stim_padded.shape[0]
    times_stim = np.linspace(tmin, tmax, n_points, endpoint=False)

    ax.plot(times_stim, data_stim_padded, color='green', linewidth=1.5, )
    #ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    #ax.axvline(x=ts, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax.set_xlim(tmin, tmax)
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xlabel('Time, ms', loc='left', fontsize=10)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x * 1000):d}'))

    return data_stim_padded

def plot_stim_PSD(ax, base_path, spectra_corr, stimulus, sinus_tone, frequencies, fmin, fmax, padding_factor):
    """
    Plot Spectral Amplitude of the stimulus
    """
    if sinus_tone:
        data_stim = stimulus
    else:
        data_stim = stimulus[:, 0]

    to_GA = False
    data_stim_padded = zero_padding(data_stim, to_GA, padding_factor)

    evoked_stim = mne.EvokedArray(
        data=np.transpose(data_stim_padded),
        info=info_wav,
        tmin=0
    )
    evoked_stim_resampled = evoked_stim.resample(fs, npad="auto")

    psd = evoked_stim_resampled.compute_psd(
        method='welch',
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        verbose=False
    )
    # Frequency resolution: 10_000 / 1024 ≈ 9.77 Гц

    data_psd = psd.get_data()
    data_amplitude = np.sqrt(data_psd).flatten()
    freqs_data = psd.freqs

    trim_index = trim_freq(freqs_data)
    data_slice = data_amplitude[trim_index:]
    freq_slice = freqs_data[trim_index:]

    y_max = data_slice.max()
    y_top = y_max * 1.50  # 50%
    y_bottom = 0

    ax.plot(freq_slice, data_slice, 'g-', linewidth=1.5)
    ax.set_ylim(y_bottom, y_top)

    if sinus_tone:
        colors = ['magenta', 'orange', 'blue', 'green']
        for idx, frequency in enumerate(frequencies):
            ax.axvline(
                x=frequency,
                alpha=0.3,
                color=colors[idx % len(colors)],
                label=f'F{idx}: {frequency} H',
                linewidth=2.5
            )
        ax.legend()
        plt.show()

        return []

    #TODO params 12, 30.0
    amps_stim_to_corr, freqs_stim_to_corr = plot_spectra_with_freq_vals(ax, spectra_corr, y_top,  freq_slice, data_slice)
    ax.set_yticks([])

    filename = os.path.join(base_path, "stim_freqs_and_amps.txt")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Stim amplitudes and frequencies for correlation\n")
        f.write("# Format: frequency (Hz) | amplitude\n\n")
        for freq, amp in zip(freqs_stim_to_corr, amps_stim_to_corr):
            f.write(f"{freq:.3f} {amp:.3f}\n")
    #tp = 'stim'
    #write_amps_freqs(amps_stim_to_corr, freqs_stim_to_corr, tp, base_path)

    return data_slice, freq_slice, freqs_stim_to_corr

def plot_waveform_correlation(ax, results, N):
    """
    Plot correleation of stim and response in the time domain
    """

    match = next((item for item in results if item[0] == cfg.lag_target_ms), None)
    if match[2] < 0.05:
        # p-val is significant
        ax.plot(N, match[1], color='green', marker='o', markersize=10)
    else:
        ax.plot(N, match[1], color='red', marker='o', markersize=10)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10,
               label='R significant, p-val<0.05'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='R insignificant'),
    ]
    ax.legend(handles=legend_elements, loc='best')
    ax.set_xlabel('N averages')
    ax.set_ylabel('R value')
    ax.set_xticks([0, 250, 500, 1000, 2000, 3000, 4000])
    ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])

    ax.grid(True, which='both', linestyle='--', alpha=0.5)

def prepare_stim_resp_arrays(resp, stim, tmin):
    """
    Function makes arrays of stim and responses for waveform correlation
    """
    prestim_interval, poststim_interval = make_prestim_poststim(tmin)
    resp_data = resp[:, round(prestim_interval):-poststim_interval - 1]

    stim_channel = stim[:, 0]
    stim_channel = stim_channel[np.newaxis,:]
    stim_conv = mne.io.RawArray(stim_channel, info_wav)
    stim_resample = stim_conv.resample(sfreq=fs)

    lag_ms = 5
    lag_samples = int(lag_ms * fs / 1000)

    stim = stim_resample.get_data().flatten()  # (N_stim,)
    resp = resp_data.flatten()  # (N_resp,)

    n_stim = len(stim)
    n_resp = len(resp)

    return n_stim, stim, n_resp, resp, lag_samples


def process_plot_filt(axes, N, fname_stim, fname_data, ftype, ch_name, base_path, non_filt, n_6low, n_7low,
                      label_6, label_7, preamplifier, dummy, fmin, fmax, order, ts, tmin, tmax,  AMP_THRESHOLD,
                                                                                 TREND_THRESHOLD, DIFF_THRESHOLD,
                      padding_factor, use_non_filt):

    # Preprocessing 1: Import Raw
    epochs, bad_indices, events, event_dict, eeg_registration = import_and_epoch(fname_data, ftype, ch_name, non_filt,
                                                                                 use_non_filt, n_6low, n_7low, label_6, label_7,
                                                                                 base_path, dummy, fmin, fmax, order,
                                                                                 tmin, tmax, AMP_THRESHOLD,
                                                                                 TREND_THRESHOLD, DIFF_THRESHOLD)
    sampl_freq_stim, stimulus = wavfile.read(fname_stim)

    stimulus_corr = trim_stim(stimulus, ts, sampl_freq_stim)

    # 1st row, 1st col — Stimulus
    ax1 = axes[0, 0]
    stim_padded = plot_stim(stimulus_corr, ax1, tmin, tmax, ts)

    ax1.set_title('Stimulus waveform', fontsize=12)

    # 1st row, 2d col — Spectral Amplitude of the Stimulus
    ax2 = axes[0, 1]
    sin_tone = False
    spectra_corr = 1
    _, _, _ = plot_stim_PSD(ax2, base_path, spectra_corr, stimulus_corr, sin_tone, [], fmin, fmax,
                                                            padding_factor)
    ax2.set_title(f'Stimulus spectra ', fontsize=12)

    # 2d row, 1st col — Grand Average FFR
    ax3 = axes[1, 0]
    noise = False
    grand_average, epochs_ffr = compute_GA(epochs, tmin, fmin, fmax, order)

    to_GA = False
    plot_GA(grand_average, to_GA, ax3, ts, tmin)
    ax3.set_title(f'FFR averaged {ch_name}', fontsize=12)

    # 2d row, 2d col — Spectral Amplitude FFR + Noise
    ax4 = axes[1, 1]
    spectra_corr = 1
    _, _ = plot_noise_PSD(ax4, base_path, spectra_corr, grand_average, fmin, fmax, padding_factor, tmin)
    ax4.set_title(f'FFR Spectra', fontsize=12)

    ax5 = axes[2, 0]
    ax6 = axes[2, 1]

    step = cfg.step
    if step > N:
        step = N
    averages = np.arange(cfg.start, N + step, step)

    for n in averages:
        n_6low_, n_7low_ = [n // 2], [n // 2]
        epochs, bad_indices, events, event_dict, eeg_registration = import_and_epoch(fname_data, ftype,
                                                                                     ch_name,
                                                                                     non_filt,
                                                                                     use_non_filt,
                                                                                     n_6low_, n_7low_,
                                                                                     label_6, label_7,
                                                                                     preamplifier,
                                                                                     dummy, fmin,
                                                                                     fmax, order,
                                                                                     tmin, tmax,
                                                                                     AMP_THRESHOLD,
                                                                                     TREND_THRESHOLD,
                                                                                     DIFF_THRESHOLD)

        noise = False
        grand_average, epochs_ffr = compute_GA(epochs, tmin, fmin, fmax, order)
        wcorr_results = waveform_correlation(stimulus_corr, grand_average, n, tmin, tmax)
        plot_waveform_correlation(ax5, wcorr_results, n)

        spectra_corr = 0
        _, _= plot_noise_PSD(ax4, base_path, spectra_corr, grand_average, fmin, fmax, padding_factor,
                                            tmin)
        epochs_stim = make_stim_epochs(stim_padded, tmin, fmin, fmax, padding_factor, epochs_ffr)
        r_amps = False
        r, pval = morlet_psd_epochs(base_path, epochs_stim, epochs_ffr, r_amps, tmin)
        plot_spectral_correlation(ax6, r, [], r_amps, n)

    ax5.set_title(f'Time domain correlation stim/FFR: lag = 10 ms', fontsize=12)
    if r_amps:
        ax6.set_title(f'FFR spectral amplitude in best to stim freqs', fontsize=12)
    else:
        ax6.set_title(f'FFR spectral power in best to stim freqs', fontsize=12)
    plt.subplots_adjust(hspace=0.7, top=0.93, bottom=0.07)

    return bad_indices, events, event_dict, len(epochs_ffr), eeg_registration

def read_amps_freqs(base_path, tp):
    """
    Function to read peak amps and freqs
    """
    file_path = os.path.join(base_path, f"{tp}_amps_freqs.txt")
    data = np.loadtxt(file_path, comments='#')
    freqs_to_corr = data[:, 0]
    amps_to_corr = data[:, 1]

    return amps_to_corr, freqs_to_corr

def remove_artifacts(epochs, AMP_THRESHOLD, TREND_THRESHOLD, DIFF_THRESHOLD):
    """
    Removes artifactual epochs
    """
    multiplier = cfg.multiplier

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

def save_pdf(fig, output_dir, fname_stim, stim_type, fpath_data, ch_name, preamplifier, subject,
             n_6low, n_7low, label_6, label_7, n_epochs_clean, N, TS, TP, fmin, fmax, order, eeg_registration, events, event_dict):

    os.makedirs(output_dir, exist_ok=True)

    # 1. Prepare data
    available_6low, available_7low, _, _, _= select_events(n_6low, n_7low, label_6, label_7, events, event_dict)
    total_n = available_6low + available_7low

    #match = re.search(r'N(\d+)', fname_stim)

    wav = Path(fname_stim).name
    bdf = Path(fpath_data).name

    wav_triggers = count_wav_triggers_optimized(fname_stim)

    trigger_rows = []
    grand_total = 0

    for key, data in wav_triggers.items():
        count = data.get('count', 0)
        if '_low' in key or 'low' in key:
            trigger_rows.append((f"'{key}'", int(count)))
            grand_total += count
    trigger_rows.append(("Total N", int(grand_total)))
    stimuli_str = ", ".join([f"{key} {value}" for key, value in trigger_rows])

    min_jitter, max_jitter = time_jitter(events, fname_stim)

    report_data = {
        "Data": [
            ("Data file", bdf),
            ("Date of EEG recording", eeg_registration),
            ("Total number of events", f'6low {available_6low}, 7low {available_7low}, Total {total_n}'),
            ("Number of epochs in analysis", f'6low , 7low , Total {n_epochs_clean}'),
            ("Channel name", f"{ch_name}, GND, ref = {cfg.ref_chs}"),
            ("Mean event time jitter", f'Minimum time jitter {min_jitter} ms, maximum time jitter, ms {max_jitter}  ms' ),
        ],
        "Equipment and software": [
            ("EEG Amplifier", "NVX136"),
            ("Audio stimulator", "AStim"),
            #("Earphones", "Anti-radiation 3.5mm Air Acoustic Tube Earpiece Headset"),
            ("Earphones", "Nicolet Reusable Tubal Insert Phones 300 Ohm (or TIP300)"),
            ("Audio delay", "0,76 ms"),
            #("Preamplifier", "MNSENS-ACP, Gain 500, 16.....3000 Hz"),
            ("EEG recording software", "NeoRec 1.6"),
            ("Report generator", "Project FFR 1 https://github.com/asmyasikova83/Frequency_Following_Response_Astim/tree/main")
        ],
        "Stimulus": [
            ("Stimulus file", wav),
            ("Stimuli", f'{stimuli_str}'),
            ("Stimulus latency", f"{TS} ms"),
            ("Pause latency", f"{TP} ms"),
    ],
        "Processing": [
            ("Number of averages", n_epochs_clean),
            ("Filtering", f"Butterworth {fmin} - {fmax} Hz, order {order}")
        ]
    }

    output_filename = f'FFR_{stim_type}_{subject}_{preamplifier}_N{N}_{ch_name}_{ref_chs}.pdf'
    output_path = os.path.join(output_dir, output_filename)

    temp_table_pdf = os.path.join(output_dir, "temp_table_page.pdf")
    temp_plot_pdf = os.path.join(output_dir, "temp_plot_page.pdf")

    try:
        # ==========================================
        # Synchronize general settings
        # ==========================================
        page_width, page_height = landscape(A4)
        left_margin = 0.5 * inch
        right_margin = 0.5 * inch
        top_margin = 0.4 * inch
        bottom_margin = 0.4 * inch

        usable_width = page_width - left_margin - right_margin
        pad_inches_val = 0.3

        target_width_inches = (usable_width / 72.0) + (2 * pad_inches_val)
        target_height_inches = page_height / 72.0

        # ==========================================
        # PDF: 1 page (TABLE)
        # ==========================================
        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(
            temp_table_pdf,
            pagesize=landscape(A4),
            leftMargin=left_margin,
            rightMargin=right_margin,
            topMargin=top_margin,
            bottomMargin=bottom_margin,
        )
        elements = []

        title_style = styles['Heading1']
        title_style.alignment = 1
        title_style.fontSize = 14
        elements.append(Paragraph("Frequency Following Response /FFR/ Report", title_style))

        company_style = title_style.clone('CompanyStyle')
        company_style.fontSize = 10
        company_style.alignment = 1
        elements.append(Paragraph("Medical Computer Systems Ltd.", company_style))
        elements.append(Spacer(1, 0.2 * inch))

        date_now = datetime.now().strftime("%Y-%m-%d")

        info_block_data = [
            ["Subject", subject],
            ["Report Date", date_now],
        ]
        col_width_info = [0.4 * usable_width, 0.6 * usable_width]

        info_table = Table(info_block_data, colWidths=col_width_info)
        info_ts = TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('LEADING', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ])
        info_table.setStyle(info_ts)
        elements.append(info_table)
        elements.append(Spacer(1, 0.1 * inch))

        section_col_widths = [0.4 * usable_width, 0.6 * usable_width]
        for section_title, rows in report_data.items():
            section_table = create_section_table(
                header_text=section_title,
                rows_data=rows,
                styles=styles,
                colWidths=section_col_widths
            )
            elements.append(section_table)
            elements.append(Spacer(1, 0.1 * inch))  # Уменьшен отступ

        doc.build(elements)

        # ==========================================
        # PDF: 2 page (PLOT)
        # ==========================================
        fig.set_size_inches(target_width_inches, target_height_inches)
        plt.tight_layout(pad=0.1)

        fig.savefig(
            temp_plot_pdf,
            dpi=300,
            bbox_inches='tight',
            pad_inches=pad_inches_val,
            facecolor='white',
            edgecolor='none',
            format='pdf'
        )
        plt.close(fig)

        # ==========================================
        # Merge 1 + 2 PDFs
        # ==========================================
        writer = PdfWriter()
        reader_table = PdfReader(temp_table_pdf)
        writer.append_pages_from_reader(reader_table)

        reader_plots = PdfReader(temp_plot_pdf)
        writer.append_pages_from_reader(reader_plots)

        with open(output_path, "wb") as out:
            writer.write(out)

        print(f"Report created: {output_path}")

    finally:
        for temp_file in [temp_table_pdf, temp_plot_pdf]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    os.startfile(output_path)

def save_wav_output(base_name, dir, full_signal, sample_rate, frequencies, stimulus_duration, inter_stimulus_interval, num_repetitions):
    """
    Function to save full_signal, pic
    """
    wav_filename = f'{base_name}.wav'
    png_filename = f'{base_name}.png'

    wav_path = os.path.join(dir, wav_filename)
    png_path = os.path.join(dir, png_filename)

    save_signal_plot(full_signal, png_path, frequencies, stimulus_duration, inter_stimulus_interval, num_repetitions)

    write(wav_path, sample_rate, full_signal)
    print(f"WAV‑file created: {wav_path}")
    print(f"Plot of the signal saved: {png_path}")

def select_events(n_6low, n_7low,  label_6, label_7, events, event_dict):
        """
        Preprocessing: pick events
        """
        available_6low, selected_events_6low, selected_indices_6low = extract_n_events(
        events,
        event_dict,
        label=label_6,
        n=n_6low,
        random_selection=False
        )
        available_7low, selected_events_7low, selected_indices_7low = extract_n_events(
        events,
        event_dict,
        label=label_7,
        n=n_7low,
        random_selection=False
        )

        if available_6low != available_7low:
            m = np.min([available_6low, available_7low])
            adjusted_events_6low = selected_events_6low[:m]
            adjusted_events_7low = selected_events_7low[:m]
        else:
            adjusted_events_6low = selected_events_6low
            adjusted_events_7low = selected_events_7low

        combined_events = np.concatenate([adjusted_events_6low, adjusted_events_7low])
        # Derive indices according to the first col (times)
        sorted_indices = np.argsort(combined_events[:, 0])
        sorted_events = combined_events[sorted_indices]
        #TEST sin TODO
        #sorted_events = selected_events_6low
        #available_7low = []

        return available_6low, available_7low, adjusted_events_6low, adjusted_events_7low, sorted_events

def show_progress(steps, delay, width=20):
    """
    Visualize the string
    """
    print('Parsed the args, starting... ', end='', flush=True)
    for _ in range(steps):
        print('.', end='', flush=True)
        time.sleep(delay)
    print('Done!')

def spectral_correlation_ssd(ffr_epochs, stim_epochs, amp_stim, amp_ffr, freq_ffr, freq_stim_to_corr):
    """
    Correlate spectral amplitudes between stim and ffr

    Parameters:
    - amp_stim, amp_ffr: specrtral amplitudes from psd.data)
    """

    pairs = find_nearest_freq(amp_stim, freq_stim_to_corr, amp_ffr, freq_ffr)
    data = [(p['stim_freqs'], p['stim_amps'], p['ffr_freqs'], p['ffr_amps']) for p in pairs]

    data.sort(key=lambda x: x[3])
    data_trim = data

    stim_freqs = np.array([x[0] for x in data_trim])
    ffr_freqs = np.array([x[2] for x in data_trim])


    freqs_sig = np.stack([ffr_freqs - 10, ffr_freqs + 10], axis=1)  # shape: (N, 2)
    freqs_noise = np.stack([ffr_freqs - 20, ffr_freqs + 20], axis=1)

    psds_ffr = []
    for fr_sig, fr_noise in list(zip(freqs_sig, freqs_noise)):
        psd, freqs = SSD_GA(ffr_epochs, fr_sig, fr_noise)
        psds_ffr.append(psd)

    psds_ffr_max_snr = np.stack(psds_ffr)
    psds_ffr_max_snr_data = psds_ffr_max_snr.flatten()


    freqs_sig_stim = np.stack([stim_freqs - 10, ffr_freqs + 10], axis=1)  # shape: (N, 2)
    freqs_noise_stim = np.stack([stim_freqs - 20, ffr_freqs + 20], axis=1)

    psds_stim = []
    for fr_sig, fr_noise in list(zip(freqs_sig_stim, freqs_noise_stim)):
        psd, freqs = SSD_GA(stim_epochs, fr_sig, fr_noise)
        psds_stim.append(psd)

    psds_stim_max_snr = np.stack(psds_stim)
    psds_stim_max_snr_data = psds_stim_max_snr.flatten()

    eps = 1e-12

    # Z‑score только по этим точкам
    s_norm = (psds_stim_max_snr_data - np.mean(psds_stim_max_snr_data)) / (np.std(psds_stim_max_snr_data) + 1e-9)
    f_norm = (psds_ffr_max_snr_data - np.mean(psds_ffr_max_snr_data)) / (np.std(psds_ffr_max_snr_data) + 1e-9)

    corr_coeff, p_val = pearsonr(s_norm, f_norm)

    return corr_coeff, p_val


def SSD_GA(grand_average, freqs_sig, freqs_noise):
    """
    Maximizes SNR, narrowband frequency ranges
    """
    # freqs_sig = 150, 160
    # freqs_noise = 140, 170

    ssd = SSD(
        info=grand_average.info,
        reg="oas",
        sort_by_spectral_ratio=False,
        filt_params_signal=dict(
            l_freq=freqs_sig[0],
            h_freq=freqs_sig[1],
            l_trans_bandwidth=10,
            h_trans_bandwidth=10,
        ),
        filt_params_noise=dict(
            l_freq=freqs_noise[0],
            h_freq=freqs_noise[1],
            l_trans_bandwidth=10,
            h_trans_bandwidth=10,
        ),
    )
    # Learning SSD
    ssd.fit(X=grand_average.get_data())

    ssd_sources = ssd.transform(X=grand_average.get_data())

    psd, freqs = mne.time_frequency.psd_array_welch(
        ssd_sources, sfreq=fs, n_per_seg=n_per_seg, n_fft=4096
    )
    spec_ratio, sorter = ssd.get_spectral_ratio(ssd_sources)

    below50 = freqs < 500
    # for highlighting the freq. band of interest
    bandfilt = (freqs_sig[0] <= freqs) & (freqs <= freqs_sig[1])

    plot = False
    if plot:
        fig, ax = plt.subplots(1)
        ax.loglog(freqs[below50], psd[0, below50], label="max SNR")
        ax.loglog(freqs[below50], psd[-1, below50], label="min SNR")
        ax.loglog(freqs[below50], psd[:, below50].mean(axis=0), label="mean")
        ax.fill_between(freqs[bandfilt], 0, 10000, color="green", alpha=0.15)
        ax.set_title("GA FFR: PSD SSD")
        ax.set_xlabel("log(frequency)")
        ax.set_ylabel("log(power)")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return psd, freqs

def time_jitter(events_bdf, fname_stim):
    """
    Function to compute event time jitter in data file
    """

    def compute_deviations(intervals_wav_s_cum_with_start, intervals_bdf_s_with_start):
        """
        Removes unpaired triggers
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
        indices_to_remove_wav_corr = indices_to_remove_wav[indices_to_remove_wav < len(intervals_wav_s_cum_with_start)]
        intervals_wav_s_cum_with_start_corr = np.delete(intervals_wav_s_cum_with_start, indices_to_remove_wav_corr)

        m = min(len(intervals_wav_s_cum_with_start_corr), len(intervals_bdf_s_with_start_corr))

        diff_intervals = 1000 * (intervals_wav_s_cum_with_start_corr[:m] - intervals_bdf_s_with_start_corr[:m])
        data = np.round(diff_intervals, decimals=3)

        idx_min = np.argmin(data)
        idx_max = np.argmax(data)

        print('Minimum time jitter, ms:', data[idx_min])
        print('Maximum time jitter, ms:', data[idx_max])

        return data[idx_min],  data[idx_max]

    def prepare_intervals(fname_stim, events_bdf):
        """
        Time intervals for stim and pause
        """
        wav_triggers = count_wav_triggers_optimized(fname_stim)

        indices_opt_6low = []
        indices_opt_6high = []
        indices_opt_7low = []
        indices_opt_7high = []
        for key, data in wav_triggers.items():
            matches = data.get('matches', 0)
            if '6_low' in key or '6low' in key:
                indices_opt_6low.append(matches)
            elif '6_high' in key or '6high' in key:
                indices_opt_6high.append(matches)
            elif '7_low' in key or '7low' in key:
                indices_opt_7low.append(matches)
            else:
                indices_opt_7high.append(matches)

        events_wav = np.concatenate([
            indices_opt_6low,
            indices_opt_6high,
            indices_opt_7low,
            indices_opt_7high,
        ])
        events_wav_flat = np.array(events_wav).flatten()
        events_wav_sorted = np.sort(events_wav_flat)

        # Compute diff betw adjacent indices
        intervals_bdf = np.diff(events_bdf[:, 0])
        intervals_bdf_s = intervals_bdf / fs
        intervals_wav = np.diff(events_wav_sorted)
        intervals_wav_s = intervals_wav / fs_wav

        intervals_bdf_with_start = np.concatenate([[0], intervals_bdf_s])
        intervals_wav_s_with_start = np.concatenate([[0], intervals_wav_s])

        return intervals_wav_s_with_start, intervals_bdf_with_start

    (intervals_wav_s_with_start,
     intervals_bdf_s_with_start) = prepare_intervals(fname_stim, events_bdf)

    min_jitter, max_jitter = compute_deviations(intervals_wav_s_with_start, intervals_bdf_s_with_start)

    return min_jitter, max_jitter

def trim_freq(freqs_data, cutoff_freq=50):
    """
    Trims the left part of psd with noise
    """
    index = next(i for i, value in enumerate(freqs_data) if value >= cutoff_freq)

    return index

def trim_ga(resp, tmin):
    """
    Function to cut out the stim interval from the grand average
    """

    prestim_interval, poststim_interval = make_prestim_poststim(tmin)
    resp_trimmed = resp[:, prestim_interval:-poststim_interval - 1]

    return resp_trimmed

def trim_stim(stimulus, stimulus_duration, sample_rate):
    """
    Function to trim stimulus to the required accurate latency
    """
    required_length = int((stimulus_duration) * sample_rate)
    return stimulus[:required_length]


def waveform_correlation(stim, grand_average, n, tmin, tmax):
    """
    Computes pearson's corrrelation between stim and ffr waveform in time domain
    """
    # W — FFR waveform (1D array)
    # S — stimulus waveform (1D array)

    n_stim, stim, n_resp, resp, lag_samples = prepare_stim_resp_arrays(grand_average.get_data(), stim, tmin)

    max_lag_ms = 50
    max_lag_samples = int(max_lag_ms * fs / 1000)

    results = []
    for lag in range(0, max_lag_samples + 1, lag_samples):
        L = min(n_stim - lag, n_resp)

        stim_shifted = stim[lag: lag + L]
        resp_window = resp[:L]

        if np.std(stim_shifted) == 0 or np.std(resp_window) == 0:
            r, p_val = np.nan, np.nan
        else:
            r, p_val = pearsonr(stim_shifted, resp_window)

        results.append((lag / fs * 1000, abs(round(r, 2)), round(p_val, 5)))
        print(f"Lag = {lag / fs * 1000:.1f} ms | Pearson r = {r:.4f}, p-value = {p_val:.4e}")

    best_lag, best_r, best_p = max(results, key=lambda x: abs(x[1]))
    print("\nBest lag:", best_lag / fs * 1000, "ms, r =", best_r)

    return results

def write_amps_freqs(freqs_to_corr, amps_to_corr, tp, base_path):
    """
    Function to write peak amps and freqs
    """
    filename = os.path.join(base_path, f"{tp}_amps_freqs.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# {tp} amplitudes and frequencies for correlation\n")
        f.write("# Format: frequency (Hz) | amplitude\n\n")

        for freq, amp in zip(freqs_to_corr, amps_to_corr):
            f.write(f"{freq:.3f} {amp:.3f}\n")

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