import numpy as np
from scipy.signal import correlate
from scipy.io import wavfile



#fname  = r'\\MCSSERVER\DB Temp\physionet.org\files\ffr_astim\Da_syll_TS250.0ms_TP200.0ms_N2000_INV1.wav'
#fname  = r'\\MCSSERVER\DB Temp\physionet.org\files\ffr_astim\Da_syll_TS250.0ms_TP200.0ms_N3000_INV1.wav'
fname  = r'\\MCSSERVER\DB Temp\physionet.org\FFR\stim\Da_syll_TS250.0ms_TP200.0ms_N4000_Amplitude_INV1.wav'

#fname  = r'\\MCSSERVER\DB Temp\physionet.org\FFR\stim\DA_syll_TS250ms_N4000_A100.0%_INV1.wav'
#fname  = r'\\MCSSERVER\DB Temp\physionet.org\FFR\stim\short\G_note_TS100ms_TP100ms_N4000_A100%_INV1.wav'
#fname  =  r'\\MCSSERVER\DB Temp\physionet.org\FFR\stim\short\DA_syll_TS90ms_N4000_A100%_INV1.wav'
_SILENCE = 1
max_int16 = np.iinfo(np.int16).max
min_int16 = np.iinfo(np.int16).min

print(f'Processing {fname}....')
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


count_opt_6low, indices_opt_6low = count_wav_triggers_optimized(stim[:, 1], seq_6low)
count_opt_7low, indices_opt_7low = count_wav_triggers_optimized(stim[:, 1], seq_7low)

count_opt_6high, indices_opt_6high = count_wav_triggers_optimized(stim[:, 1], seq_6high)
count_opt_7high, indices_opt_7high = count_wav_triggers_optimized(stim[:, 1], seq_7high)
print(f"6low - {count_opt_6low} вхождений")
print(f"7low - {count_opt_7low} вхождений")


print(f"6high - {count_opt_6high} вхождений")
print(f"7high - {count_opt_7high} вхождений")