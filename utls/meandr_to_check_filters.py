import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from functions import fir_bandpass_filter

# Параметры
duration = 2
frequencies = [10, 50, 100]  # список частот
sampling_rate = 10000
method = 'multitaper'
fmin = 5
fmax = 200
order = 100

# Время
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Создаём фигуру с 6 строками (3 частоты × 2 графика для каждой) и 1 колонкой
fig, axes = plt.subplots(6, 1, figsize=(14, 18))
fig.suptitle('Сравнение меандров разных частот до и после фильтрации', fontsize=16, y=0.98)

# Цикл по частотам
for i, freq in enumerate(frequencies):
    # Генерируем меандр для текущей частоты
    square_wave = signal.square(2 * np.pi * freq * t)
    # Фильтруем сигнал
    filtered_signal = fir_bandpass_filter(square_wave, fmin, fmax, sampling_rate, order, transition_width=0.02)

    # График 1: исходный меандр (строка 2*i)
    ax1 = axes[2 * i]
    ax1.plot(t, square_wave, 'b-', linewidth=1.5)
    ax1.set_title(f'Меандр: {freq} Гц (исходный)', fontsize=12)
    #ax1.set_xlabel('Время, с', fontsize=10)
    ax1.set_ylabel('Амплитуда', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.4, 1.4)

    # График 2: отфильтрованный меандр (строка 2*i + 1)
    ax2 = axes[2 * i + 1]
    ax2.plot(t, filtered_signal, 'r-', linewidth=1.5)
    ax2.set_title(f'Меандр:FIR {freq} Гц bandpass ({fmin}–{fmax} Гц, порядок {order})', fontsize=12)
    #ax2.set_xlabel('Время, с', fontsize=10)
    ax2.set_ylabel('Амплитуда', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.4, 1.4)

# Улучшаем расположение графиков
plt.tight_layout(pad=3.0, h_pad=2.0)

# Сохраняем
output_dir = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\pics'
# Создаём директорию, если её нет
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f'square_wave_filter_comparison_{order}.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()