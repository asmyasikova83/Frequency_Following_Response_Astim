import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import config as cfg
from functions import select_events, average_and_filter_epochs, zero_padding, butter_bandpass_filter

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
    ga_data_padded = zero_padding(grand_average.get_data(), to_GA, 32)
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


    data_amplitude = np.sqrt(data_psd).flatten() * 1e9  # nV/√Hz
    freqs_data = psd.freqs

    return grand_average, data_amplitude, freqs_data
dummy = 0
if dummy:
    fname_dummy =  r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\dummy\ffr_da_N4000_dummy.bdf'
    fname_preamplifier = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\dummy\preamplifier\ffr_da_N4000_dummypreamplifier.bdf'
    filename = "Dummy_w_wout_preamplifier_comparison_time_freq.pdf"
    label = 'Dummy w/out preamp'
    label_preamp = 'Dummy with preamp'
    title = 'Grand Average: Dummy vs Dummy Preamplifier (Time Domain)'
    ylabel = 'nV'
    spectral_title ='Spectral Amplitude: Dummy vs Dummy Preamplifier (Frequency Domain)'
    multiplier = 1e9
else:
    fname_preamplifier =  r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\non_filt\preamplifier\ffr_da_N4000_non_filtS0preamplifier.bdf'
    fname_dummy = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\ffr_da_N4000_S0_step1.bdf'
    filename = "S0_w_wout_preamplifier_comparison_time_freq.pdf"
    label = 'S0 without preamplifier'
    label_preamp = 'S0 with preamplifier'
    title = 'Grand Average: S0 vs S0 Preamplifier (Time Domain)'
    ylabel = 'muV'
    spectral_title = 'Spectral Amplitude: S0 vs S0 Preamplifier(Frequency Domain)'
    multiplier = 1e9

out_path =r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\dummy'

full_path = os.path.join(out_path, filename)
ch_name = ['Cz-(A1+A2)/2']
ref_chs = ['A1A2']
LABEL_6_BDF = '6_low'
LABEL_7_BDF = '7_low'
n_6low = [4000 // 2]
n_7low = [4000 // 2]
fmin = 10
fmax = 2500
order = 2


grand_average, data_amplitude, freqs_data = extract_and_process(fname_dummy, dummy=dummy)
grand_average_preamplifier, data_amplitude_preamplifier, freqs_data_preamplifier = extract_and_process(fname_preamplifier, dummy=dummy)

fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=False)  # 2 подграфика, общая ось X

ga_dummy = grand_average.data.mean(axis=0)  # форма: (n_times,)
ga_preamp = grand_average_preamplifier.data.mean(axis=0)

times = grand_average.times

# Верхний подграфик: grand_average и grand_average_preamplifier
axs[0].plot(times, ga_dummy * multiplier, 'b-', linewidth=0.5, label=label)
axs[0].plot(times,ga_preamp * multiplier, 'r-', linewidth=0.5, label=label_preamp)
axs[0].set_ylabel(ylabel, fontsize=9)
axs[0].set_title(title, fontsize=10)
if dummy:
    axs[0].set_ylim(-50, 50)  # 50 нВ = 0.05 мкВ
    axs[0].set_yticks([-50, 0, 50])
axs[0].legend(loc='upper right', fontsize=8, frameon=True, fancybox=True)
axs[0].tick_params(axis='both', which='major', labelsize=10)
axs[0].set_xlabel('Time, ms', loc='left', fontsize=10)

#axs[0].xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(round(x * 1000)):d}'))
axs[0].grid(True, linestyle='--', alpha=0.3)

# Нижний подграфик: data_amplitude и data_amplitude_preamplifier (PSD / спектральная плотность)
axs[1].plot(freqs_data, data_amplitude, 'b-', linewidth=0.5, label=label)
axs[1].plot(freqs_data, data_amplitude_preamplifier, 'r-', linewidth=0.5, label=label_preamp)
axs[1].set_xlabel('Frequency (Hz)', fontsize=9)
axs[1].set_ylabel('nV/√Hz', fontsize=9)
axs[1].set_title(spectral_title, fontsize=10)
axs[1].legend(loc='upper right', fontsize=8, frameon=True, fancybox=True)
axs[1].grid(True, linestyle='--', alpha=0.3)

if dummy:
    axs[1].set_ylim(0, 4e-1)
    axs[1].set_yticks([0, 4e-1])

plt.tight_layout()
plt.savefig(
    full_path,
    format='pdf',
    dpi=300,
    bbox_inches='tight',
    transparent=False
)

print(f"График сохранён: {os.path.abspath(full_path)}")
plt.show()