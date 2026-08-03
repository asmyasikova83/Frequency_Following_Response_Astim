import numpy as np
import mne
from pathlib import Path

# create_wav
percent_var_pause = 0.1
_SILENCE = 1
max_int16 = np.iinfo(np.int16).max
min_int16 = np.iinfo(np.int16).min
fs_wav = 44100
fs = 10000

# command_line_ffr
base_path = Path(r'C:\Users\msasha\Desktop\AStim\data')

sequences = {
    '6low': [max_int16, min_int16, min_int16, max_int16, min_int16, max_int16],  # 100
    '7low': [max_int16, min_int16, max_int16, min_int16, min_int16, max_int16],  # 110
    '6high': [max_int16, min_int16, min_int16, max_int16, max_int16, min_int16],  # 101
    '7high': [max_int16, min_int16, max_int16, min_int16, max_int16, min_int16]  # 111
}
LABEL_6_FIF = r'In\6'
LABEL_7_FIF = r'In\7'

LABEL_6_BDF = '6_low'
LABEL_7_BDF = '7_low'

# setup
#mse
mse = 0
hexagone = 0
substraction = 0
raw = 0

if hexagone:
    ch_name = ['10','11', '12', '13', '16', '18']
    ref_chs = ['17']
    #ch_name = ['27', '46', '47', '29', '16', '15']
    #ref_chs = ['28']
else:
    ref_chs = ['4', '7']
    ch_name = ['17']
    #ch_name = ['8']
    #ch_name = ['28']

info = mne.create_info(
    ch_names=ch_name,
    sfreq=fs,
    ch_types='eeg'
)
info1ch = mne.create_info(
    ch_names=['Cz'],
    sfreq=fs,
    ch_types='eeg'
)
info_wav = mne.create_info(
    ch_names=['Cz'],
    sfreq=fs_wav,
    ch_types='eeg'
)
# cleaning, psd, plotting and monitoring params
non_filt_epo = 1
early_filt = 1
if early_filt:
    amp_threshold = 40e-06
else:
    amp_threshold = 80e-06
multiplier = 1e-6
trim_epo_share = 0.15
sound_delay = 0.00076
freq_res = 10
n_fft = 1024
n_per_seg = 1024
n_overlap = 512
step = 500
start = 100
n_peaks = 10
min_freq_gap = 30
dt_target = 0.035

# waveform correlation
#settings for from Skow, Kraus, 2010
filter_stim = 1
min_lag_ms = 7
max_lag_ms = 10
step_ms = 1
start_transition_stim = 0.0195
end_transition_stim = 0.0442