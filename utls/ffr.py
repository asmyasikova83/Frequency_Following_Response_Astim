import os
from pathlib import Path
import mne
import matplotlib.pyplot as plt
from functions import (process_plot_filt, project_paths, process_plot_last_filt,save_pdf)

subject = 'S1'
preamplifier ='preamplifier'   # ''
dummy = 'dummy'  # ''
non_filt = 'non_filt'   # ''
short = ''# 'shortG'
ts = 0.25
fname_stim = r'\\MCSSERVER\DB Temp\physionet.org\files\ffr_astim\DA_syll_TS250.0ms_TP200.0ms_N1_A100.0%_INV0.wav'
# sampl_freq_stim, stimulus = wavfile.read(r'\\MCSSERVER\DB Temp\physionet.org\FFR\stim\short\G.wav')
# sampl_freq_stim, stimulus = wavfile.read(r'\\MCSSERVER\DB Temp\physionet.org\files\ffr_astim\DA_stim2_TS90.0ms_N1_A100.0%_INV0.wav'

# Базовый путь
base_path = Path(r'\\MCSSERVER\DB Temp\physionet.org\FFR\data')
fname_bdf, output_dir = project_paths(base_path, non_filt, dummy, short, preamplifier, subject  )

# Prestimulus [-0.05; 0] Stimulus [0 ; 0.25] + 0.05 to get back to baseline
tmin = -0.05
tmax = 0.3

if dummy:
    n_6low = [1999]
    n_7low = [1992]
else:
    n_6low = [1999]
    n_7low = [1999]

method = 'multitaper'
fmin = 40
fmax = 850
order = 100
transition_width = 0.05

if preamplifier:
    multiplier = 1e-3
else:
    multiplier = 1e-6

AMP_THRESHOLD = 85  # 35 мкВ в вольтах
TREND_THRESHOLD = 100  # 10 мкВ/с в вольтах/с
DIFF_THRESHOLD = 25  # 25 мкВ в вольтах

# Make a Fig with  subplots: 3 rows, 2 cols
fig, axes = plt.subplots(3, 2, figsize=(10, 12))

bad_indices = process_plot_filt(axes, fname_stim, fname_bdf, output_dir, subject, short, non_filt, n_6low, n_7low, preamplifier, dummy, fmin, fmax, method, order, ts,
                      tmin, tmax, transition_width, AMP_THRESHOLD, TREND_THRESHOLD, DIFF_THRESHOLD, multiplier, use_non_filt=False)

# 3d row, 1st col — Grand Average FFR filtered on the final step/not filtered
process_plot_last_filt(axes, bad_indices, fname_stim, fname_bdf, output_dir, subject, short, non_filt, n_6low,
                           n_7low, preamplifier, dummy, fmin, fmax, method, order, ts, tmin, tmax, transition_width,
                           AMP_THRESHOLD, TREND_THRESHOLD, DIFF_THRESHOLD, multiplier, use_non_filt=True)

save_pdf(fig, output_dir, preamplifier, subject, short, n_6low, n_7low, fmin, fmax, ts, tmin, tmax)