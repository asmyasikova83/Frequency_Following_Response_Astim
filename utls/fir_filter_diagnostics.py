import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

output_dir =  r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\pics'

# Параметры фильтра
low_cutoff = 40      # нижняя частота среза, Гц
high_cutoff = 850   # верхняя частота среза, Гц
orders = [50, 100, 150, 200, 250]  # порядки фильтров для сравнения
fs = 10000           # частота дискретизации, Гц
nyquist = 0.5 * fs  # частота Найквиста

# Нормализованные частоты среза
low = low_cutoff / nyquist
high = high_cutoff / nyquist

# Цвета для разных кривых
colors = ['blue', 'red', 'green', 'purple']

def design_fir_filter(order, method='firwin', transition_width=0.05):
    """
    Проектирование FIR‑фильтра разными методами.

    Параметры:
    - order: порядок фильтра;
    - method: 'firwin' или 'firwin2';
    - transition_width: ширина переходной полосы (для firwin2).

    Возвращает: коэффициенты фильтра b.
    """
    numtaps = order + 1

    if method == 'firwin':
        b = signal.firwin(
            numtaps=numtaps,
            cutoff=[low, high],
            pass_zero=False,
            window='bartlett'
        )
    elif method == 'firwin2':
        freq = [0, max(0, low - transition_width), low, high, min(1, high + transition_width), 1]
        gain = [0, 0, 1, 1, 0, 0]

        # Удаляем дубликаты частот
        unique_freq, unique_indices = np.unique(freq, return_index=True)
        unique_gain = [gain[idx] for idx in unique_indices]

        b = signal.firwin2(
            numtaps=numtaps,
            freq=unique_freq,
            gain=unique_gain,
            fs=2.0
        )
    return b

def analyze_filter_characteristics(b, fs, low_cutoff, high_cutoff):
    """
    Анализ характеристик фильтра.

    Возвращает: словарь с АЧХ, ФЧХ, групповой задержкой и пульсациями.
    """
    # АЧХ и ФЧХ
    w, h = signal.freqz(b, worN=4096, fs=fs)
    mag_db = 20 * np.log10(np.abs(h) + 1e-15)
    phase_rad = np.unwrap(np.angle(h))

    # Групповая задержка
    w_gd, gd = signal.group_delay((b, 1), w=4096)
    gd_seconds = np.array(gd) / fs

    # Пульсации в полосе пропускания
    mask_pass = (w >= low_cutoff) & (w <= high_cutoff)
    ripple_db = np.max(mag_db[mask_pass]) - np.min(mag_db[mask_pass]) if np.any(mask_pass) else np.nan

    return {
        'freqs': w,
        'mag_db': mag_db,
        'phase_rad': phase_rad,
        'group_delay_mean': np.mean(gd_seconds),
        'group_delay_std': np.std(gd_seconds),
        'ripple_db': ripple_db
    }


def plot_filter_responses(results, method, orders, colors, low_cutoff, high_cutoff, output_dir):
    """Построение АЧХ и ФЧХ для всех фильтров с указанием типа фильтра.


    Параметры:
    - results: список словарей с характеристиками фильтров;
    - orders: список порядков фильтров;
    - colors: список цветов для линий;
    - low_cutoff: нижняя частота среза (Гц);
    - high_cutoff: верхняя частота среза (Гц);
    - filter_type: тип фильтра ('firwin' или 'firwin2').
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    for i, (order, result) in enumerate(zip(orders, results)):
        color = colors[i % len(colors)]
        label = f'Порядок {order} '  # Добавляем тип фильтра в подпись

        ax1.plot(result['freqs'], result['mag_db'], color=color, linewidth=2, label=label)
        ax2.plot(result['freqs'], result['phase_rad'], color=color, linewidth=2, label=label)

    # Настройка АЧХ
    ax1.set_xlabel('Частота, Гц')
    ax1.set_ylabel('Амплитуда, дБ')
    ax1.set_title(f'АЧХ — {method}')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(low_cutoff, color='k', linestyle='--', alpha=0.7, label='Нижняя частота среза')
    ax1.axvline(high_cutoff, color='k', linestyle='--', alpha=0.7, label='Верхняя частота среза')
    ax1.legend()
    ax1.set_xlim(0, 2000)
    ax1.set_ylim(-80, 5)

    # Настройка ФЧХ
    ax2.set_xlabel('Частота, Гц')
    ax2.set_ylabel('Фаза, рад')
    ax2.set_title(f'ФЧХ — {method}')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(low_cutoff, color='k', linestyle='--', alpha=0.7)
    ax2.axvline(high_cutoff, color='k', linestyle='--', alpha=0.7)
    ax2.legend()
    ax2.set_xlim(0, 2000)

    output_path = os.path.join(output_dir, f'filter_response_{method}_{orders}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def plot_impulse_response(b, method, order):
    """Построение импульсной характеристики."""
    plt.figure(figsize=(10, 4))
    plt.stem(b)
    plt.title(f'Импульсная хар-ка (коэффициенты b) - {method} порядок {order}')
    plt.xlabel('Отсчёт')
    plt.ylabel('Амплитуда')
    plt.grid(True, alpha=0.3)
    plt.show()

# Основной цикл: проектирование и анализ фильтров
results_firwin = []
results_firwin2 = []

print("Анализ фильтров FIRWIN:")
for order in orders:
    b = design_fir_filter(order, method='firwin')
    result = analyze_filter_characteristics(b, fs, low_cutoff, high_cutoff)
    results_firwin.append(result)
    print(f"Порядок {order}:")
    print(f"  Групповая задержка: {result['group_delay_mean']:.4f} ± {result['group_delay_std']:.6f} с")
    print(f"  Пульсации в полосе пропускания: {result['ripple_db']:.4f} дБ")
    print("-" * 40)

print("\nАнализ фильтров FIRWIN2:")
for order in orders:
    b = design_fir_filter(order, method='firwin2')
    result = analyze_filter_characteristics(b, fs, low_cutoff, high_cutoff)
    results_firwin2.append(result)
    print(f"Порядок {order}:")
    print(f"  Групповая задержка: {result['group_delay_mean']:.4f} ± {result['group_delay_std']:.6f} с")
    print(f"  Пульсации в полосе пропускания: {result['ripple_db']:.4f} дБ")
    print("-" * 40)

# Построение графиков для FIRWIN
method = 'FIRWIN (только  свертка с окном)'
plot_filter_responses(results_firwin, method, orders, colors, low_cutoff, high_cutoff,output_dir)

# Пример импульсной характеристики для последнего фильтра FIRWIN
b_example = design_fir_filter(orders[-1], method='firwin')
plot_impulse_response(b_example,  method='firwin', order = orders[-1])

# Построение графиков для FIRWIN2
method = 'FIRWIN2 (gain + ifft)'
plot_filter_responses(results_firwin2, method,  orders, colors, low_cutoff, high_cutoff,output_dir)
