import numpy as np
import mne
from pathlib import Path

_SILENCE = 1
max_int16 = np.iinfo(np.int16).max
min_int16 = np.iinfo(np.int16).min
fs_wav = 44100
fs = 10000
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

multiplier = 1e-6
trim_epo = 0.15

# For info
ch_name = ['8']

"""
ch_name = ['F7 (3)-CVII (70)']
ch_name = ['Cz (8)-CVII (70)']
ch_name =  ['Fp1 (2)-CVII (70', 'F7 (3)-CVII (70)', 'Fp2 (5)-CVII (70', 'F8 (6)-CVII (70)']
ch_name = ['Cz (8)']
ch_name = ['1']
ch_name = ['8', '9', '10', '11', '12', '13', '14']
ch_name = ['8', '4', '7']
ch_name = ['4',   '7', '8', '9',  '10', '11', '12', '13', '14',
           '15', '16', '17', '18', '19', '20', '21', '22', '23',
           '24', '25', '26', '27', '28', '29', '30', '31', '32',
           '33', '34', '35', '36', '37', '38', '39', '40', '41',
           '42', '43', '44', '45', '46', '47', '48', '49', '50',
           '51', '52', '53', '54', '55', '56', '57', '58', '59',
           '60', '61', '62', '63', '64', '65', '66', '67', '68']
ch_name = ['4',   '7', '8', '9',  '10', '11', '12', '13', '14',
           '15', '16', '17', '18', '19', '20', '21', '22', '23',
           '24', '25', '26']

ch_name = ['4',   '7', '8', '9', '15', '27', '45'] # axis Cz+ I1 K1 L1 E1
"""
ref_chs = ['4', '7']
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
multiplier = 1e-6
sound_delay = 0.00076
lag_target_ms = 10
freq_res = 10
n_fft = 1024
n_per_seg = 1024
n_overlap = 512
n_per_seg_noise = 512
n_overlap_noise = 256
trim_epo_share = 0.1
step = 500
start = 100
n_peaks = 10
min_freq_gap = 30