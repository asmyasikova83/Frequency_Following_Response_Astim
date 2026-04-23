
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

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

def plot_response(w, h, title):
    "Utility function to plot response functions"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(w, 20*np.log10(np.abs(h)))
    ax.set_ylim(-40, 5)
    ax.grid(True)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain (dB)')
    ax.set_title(title)

fs = 10000
band = [40, 850]  # Desired pass band, Hz
trans_width = 30    # Width of transition from pass to stop, Hz
orders = [50, 100, 150, 200, 250]         # Size of the FIR filter.
edges = [0, band[0] - trans_width, band[0], band[1],
         band[1] + trans_width, 0.5*fs]

results = []
print("\nАнализ фильтров eaquiripple:")
for order in orders:
    taps = signal.remez(order, edges, [0, 1, 0], fs=fs)
    w, h = signal.freqz(taps, [1], worN=8000, fs=fs)
    plot_response(w, h, "Equiripple Band-pass Filter")
    plt.show()
    result = analyze_filter_characteristics(taps, fs, band[0], band[1])
    results.append(result)
    print(f"Порядок {order}:")
    print(f"  Групповая задержка: {result['group_delay_mean']:.4f} ± {result['group_delay_std']:.6f} с")
    print(f"  Пульсации в полосе пропускания: {result['ripple_db']:.4f} дБ")
    print("-" * 40)

