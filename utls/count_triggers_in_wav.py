import numpy as np
from scipy.signal import correlate
from scipy.io import wavfile


fname  = r'\\MCSSERVER\DB Temp\physionet.org\FFR\stim\DA_syll_TS250ms_N4000_A100.0%_INV1.wav'
_SILENCE = 1
max_int16 = np.iinfo(np.int16).max
min_int16 = np.iinfo(np.int16).min

##################################count_triggers#############################
# 100
seq_6low = [max_int16, min_int16, min_int16, max_int16, min_int16, max_int16]
# 110
seq_7low = [max_int16, min_int16, max_int16, min_int16, min_int16, max_int16]

sampl_freq_stim, stim = wavfile.read(fname)

print('stim[:1] shape', stim.shape)
print('seq_6low shape', len(seq_6low))

def count_sequence_optimized(data, sequence):
    """Оптимизированная версия с использованием корреляции."""
    seq = np.array(sequence)
    # Нормализованная корреляция
    corr = correlate(data, seq, mode='valid')
    # Точное совпадение — когда корреляция равна сумме квадратов элементов последовательности
    target_corr = np.sum(seq ** 2)
    matches = np.where(corr == target_corr)[0]
    return len(matches), matches.tolist()

# Использование оптимизированной версии
count_opt_6low, indices_opt_6low = count_sequence_optimized(stim[:, 1], seq_6low)
count_opt_7low, indices_opt_7low = count_sequence_optimized(stim[:, 1], seq_7low)
print(f"6low - {count_opt_6low} вхождений")
print(f"7low - {count_opt_7low} вхождений")
print(f"6low - {indices_opt_6low} ")
print(f"7low - {indices_opt_7low} ")

#########################compute_isi###########################################
def compute_isi(indices_opt_6, indices_opt_7, fs=44100):
    # Сортируем индексы на случай, если они не упорядочены

    general = np.column_stack([indices_opt_6, indices_opt_7])
    general_sorted = np.sort(general)
    # Рассчитываем разницы между соседними индексами (в отсчётах)
    isi_samples = np.diff(general_sorted)

    # Переводим в секунды: делим количество отсчётов на частоту дискретизации
    isi_seconds = isi_samples / fs

    # Считаем среднее и стандартное отклонение
    mean_isi = np.mean(isi_seconds)
    std_isi = np.std(isi_seconds, ddof=1)  # ddof=1 для несмещённой оценки (как в статистике выборки)

    return mean_isi, std_isi

mean_isi_6low, std_isi_6low = compute_isi(count_opt_6low, count_opt_7low, fs=44100)

print(f"6low - {mean_isi_6low} , {std_isi_6low } ")