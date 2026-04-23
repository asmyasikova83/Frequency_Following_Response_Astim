import os
import numpy as np
import mne
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks

def extract_seconds(timestamp):
    # Преобразуем Timestamp в строку
    timestamp_str = str(timestamp)
    dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
    minutes = dt.minute
    seconds = dt.second
    fractional = dt.microsecond / 1_000_000
    return [minutes, seconds + fractional]

fname  = r'C:\Users\msasha\Desktop\AStim\stim\check_triggers_26_min.BDF'
output_dir = r'C:\Users\msasha\Desktop\AStim\stim'

raw = mne.io.read_raw_bdf(
    fname,
    preload=True,           # Загружаем данные в память сразу
    eog=None,             # Автоматическое определение EOG-каналов
    misc=None,            # Автоматическое определение вспомогательных каналов
    stim_channel='auto',  # Автоматическое определение триггерных каналов
    verbose=True          # Подробный вывод процесса
)

# Базовая информация о данных
print("Информация о файле:")
print(f"Каналы: {raw.ch_names}")
print(f"Частота дискретизации: {raw.info['sfreq']} Гц")
print(f"Длительность: {raw.times[-1]:.2f} сек")

# Получаем общую длительность записи в секундах
total_duration = raw.times[-1]

# Отрезаем 7 с в начале и 33 с в конце, так как на 40 сек длиннее, чем стимульная запись
#Длительность: 1640.00 сек
#Оригинальная длительность: 1640.00 с

raw.crop(
    tmin=7.0,                    # Начало: 7 секунд от старта
    tmax=total_duration - 33.0   # Конец: общая длительность минус 33 секунды
)

print(f"Оригинальная длительность: {total_duration:.2f} с")
print(f"Новая длительность: {raw.times[-1]:.2f} с")

events, event_dict = mne.events_from_annotations(raw)

# Извлечение аннотаций
annotations = raw.annotations
df = annotations.to_data_frame()

df['onset'] = pd.to_datetime(df['onset'], format='%H:%M:%S.%f')
result = df['description'].value_counts()
print(result)

timestamps_seconds = (df['onset'] - df['onset'].iloc[0]).dt.total_seconds()
# Шаг 1: вычисляем интервалы между последовательными метками
intervals = np.diff(timestamps_seconds)


# Находим самые частые значения (моды) с помощью гистограммы
hist_counts, hist_bins = np.histogram(intervals, bins=50)
# Получаем индексы для сортировки в убывающем порядке
sorted_indices = np.argsort(hist_counts)[::-1]
# Сортированные значения
sorted_peaks = hist_counts[sorted_indices]

# Первые два пика и их индексы в отсортированном массиве
first_peak_count = sorted_peaks[0]
second_peak_count = sorted_peaks[1]
first_peak_idx_sorted = sorted_indices[0]
second_peak_idx_sorted = sorted_indices[1]

# Находим границы бинов для этих пиков
first_bin_left = hist_bins[first_peak_idx_sorted]
first_bin_right = hist_bins[first_peak_idx_sorted + 1]
second_bin_left = hist_bins[second_peak_idx_sorted]
second_bin_right = hist_bins[second_peak_idx_sorted + 1]

# Создаём фигуру и оси
fig, ax = plt.subplots(figsize=(12, 8))

# Строим основную гистограмму
n, bins, patches = ax.hist(
    intervals,
    bins=50,
    alpha=0.7,
    color='skyblue',
    edgecolor='black',
    linewidth=0.5,
    label='Распределение интервалов'
)

# Разбиваем на бины и получаем метки
num_bins = 50
bin_labels = pd.cut(intervals, bins=num_bins, labels=False)
bin_intervals = pd.cut(intervals, bins=num_bins)  # для получения интервалов

# Находим индексы значений в первом бине (метка 0)
bin1_indices = np.where(bin_labels == 0)[0]
bin1_values = intervals[bin1_indices]
bin2_indices = np.where(bin_labels == 48)[0]
bin2_values = intervals[bin2_indices]

print(bin2_values)
print("Индексы значений в первом бине:", bin1_indices)
print("Min Значения длины стимула в ms:", np.min(bin1_values), "Max Значения длины стимула в ms:", np.max(bin1_values))
print("Min Значения длины паузы в ms:", np.min(bin2_values), "Max Значения паузы в ms:", np.max(bin2_values))


plt.figure(figsize=(12, 8))

# Нормализованная гистограмма (плотность вероятности)
fig, ax = plt.subplots(figsize=(12, 8))  # Создаём фигуру с явным указанием размера

n, bins, patches = ax.hist(
    bin1_values,
    bins=25,
    density=True,  # Нормализация на единицу площади
    alpha=0.6,
    color='red',
    edgecolor='red',
    linewidth=1
)

ax.set_title('Плотность распределения длительности стимула', fontsize=16, pad=20)
ax.set_xlabel('Длительность, s', fontsize=20)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)

ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Сохраняем фигуру в файл
filepath = os.path.join(output_dir, 'Плотность распределения длительности стимула.png')
plt.savefig(filepath, dpi=300, bbox_inches='tight')  # Высокое разрешение, обрезаем лишние поля
plt.close(fig)  # Закрываем фигуру, чтобы освободить память

print(f"График сохранён: {filepath}")


plt.figure(figsize=(12, 8))

# Нормализованная гистограмма (плотность вероятности)
fig, ax = plt.subplots(figsize=(12, 8))  # Создаём фигуру с явным указанием размера

n, bins, patches = ax.hist(
    bin2_values,
    bins=25,
    density=True,  # Нормализация на единицу площади
    alpha=0.6,
    color='lightgreen',
    edgecolor='darkgreen',
    linewidth=1
)

ax.set_title('Плотность распределения длительности паузы', fontsize=16, pad=20)
ax.set_xlabel('Длительность, s', fontsize=20)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)

ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Сохраняем фигуру в файл
filepath = os.path.join(output_dir, 'Плотность распределения длительности паузы.png')
plt.savefig(filepath, dpi=300, bbox_inches='tight')  # Высокое разрешение, обрезаем лишние поля
plt.close(fig)  # Закрываем фигуру, чтобы освободить память

print(f"График сохранён: {filepath}")