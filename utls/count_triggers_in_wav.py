import numpy as np
from scipy.signal import correlate
from scipy.io import wavfile



#fname  = r'\\MCSSERVER\DB Temp\physionet.org\files\ffr_astim\Da_syll_TS250.0ms_TP200.0ms_N2000_INV1.wav'
fname  = r'\\MCSSERVER\DB Temp\physionet.org\files\ffr_astim\Da_syll_TS250.0ms_TP200.0ms_N3000_INV1.wav'
#fname  = r'\\MCSSERVER\DB Temp\physionet.org\FFR\stim\DA_syll_TS250ms_N4000_A100.0%_INV1.wav'
#fname  = r'\\MCSSERVER\DB Temp\physionet.org\FFR\stim\short\G_note_TS100ms_TP100ms_N4000_A100%_INV1.wav'
#fname  =  r'\\MCSSERVER\DB Temp\physionet.org\FFR\stim\short\DA_syll_TS90ms_N4000_A100%_INV1.wav'
_SILENCE = 1
max_int16 = np.iinfo(np.int16).max
min_int16 = np.iinfo(np.int16).min

##################################count_triggers#############################
# 100
seq_6low = [max_int16, min_int16, min_int16, max_int16, min_int16, max_int16]
# 110
seq_7low = [max_int16, min_int16, max_int16, min_int16, min_int16, max_int16]
# 101
seq_6high = [max_int16, min_int16, min_int16, max_int16, max_int16, min_int16]
# 111
seq_7high = [max_int16, min_int16, max_int16, min_int16, max_int16, min_int16]

sampl_freq_stim, stim = wavfile.read(fname)

def count_wav_triggers_optimized(data, sequence):
    """"""
    seq = np.array(sequence)
    # Нормализованная корреляция
    corr = correlate(data, seq, mode='valid')
    # Absolute match
    target_corr = np.sum(seq ** 2)
    matches = np.where(corr == target_corr)[0]
    return len(matches), matches.tolist()

# Использование оптимизированной версии
count_opt_6low, indices_opt_6low = count_wav_triggers_optimized(stim[:, 1], seq_6low)
count_opt_7low, indices_opt_7low = count_wav_triggers_optimized(stim[:, 1], seq_7low)

count_opt_6high, indices_opt_6high = count_wav_triggers_optimized(stim[:, 1], seq_6high)
count_opt_7high, indices_opt_7high = count_wav_triggers_optimized(stim[:, 1], seq_7high)
print(f"6low - {count_opt_6low} вхождений")
print(f"7low - {count_opt_7low} вхождений")


print(f"6high - {count_opt_6high} вхождений")
print(f"7high - {count_opt_7high} вхождений")
"""
print(f"6low - {indices_opt_6low} ")
print(f"7low - {indices_opt_7low} ")
print(f"6high - {indices_opt_6high} ")
print(f"7high - {indices_opt_7high} ")
"""

def compute_mean_std(intervals, stim, fs):
    """
    Computes stat for intervals
    """
    print('-------intervals', intervals[:10])
    if stim:
        intervals_filtered = intervals[intervals == 11018]
        print('len stims ', len(intervals_filtered))
    else:
        # Exclude stimuli if not stim (pause)
        intervals_filtered = intervals[intervals != 11018]
        print('len pauses ', len(intervals_filtered))
    # Convert into seconds
    intervals_seconds = intervals_filtered / fs
    print('intervals_seconds ', intervals_seconds[:10])

    # Считаем среднее и стандартное отклонение
    mean_interval = round(np.mean(intervals_seconds), 3)
    std_interval = round(np.std(intervals_seconds, ddof=0), 3)  # ddof=1 for unbiased estimate
    return mean_interval, std_interval

#########################compute_isi###########################################
def compute_interval_stat(indices_opt_6low, indices_opt_6high, indices_opt_7low, indices_opt_7high, fs=44100):
    """
    Computes statistics for stim, for pause
    """

    all_indices = np.concatenate([
        indices_opt_6low,
        indices_opt_6high,
        indices_opt_7low,
        indices_opt_7high,
    ])
    all_indices_sorted = np.sort(all_indices)

    # Compute diff betw adjacent indices
    intervals = np.diff(all_indices_sorted)

    # Compute stat for stim, for pause
    stim = True
    mean_stim, std_stim = compute_mean_std(intervals, stim, fs)
    stim = False
    mean_pause, std_pause = compute_mean_std(intervals, stim, fs)
    return mean_stim, std_stim, mean_pause, std_pause

mean_stim, std_stim, mean_pause, std_pause = compute_interval_stat(indices_opt_6low, indices_opt_6high, indices_opt_7low, indices_opt_7high, fs=44100)
print(' Mean stim interval: ', mean_stim, '\n', 'Std stim interval: ',  std_stim, '\n', 'Mean pause interval: ', mean_pause, '\n', 'Std pause interval: ',  std_pause)