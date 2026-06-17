import mne
import numpy as np

fname_bdf = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\test_triggers\test_DA_June26.bdf'

raw = mne.io.read_raw_bdf(
    fname_bdf,
    preload=True,
    verbose=True
)

events_bdf, event_dict = mne.events_from_annotations(raw)
events_to_save = events_bdf

onset_sec = events_to_save[:, 0] / raw.info['sfreq']
duration = np.zeros_like(onset_sec)

code_to_old_label = {v: k for k, v in event_dict.items()}

old_to_new = {
    'In/ 6': '6_high',
    'In/ 7': '7_high',
    'In\\ 6': '6_low',
    'In\\ 7': '7_low',
}

description = []
for code in events_to_save[:, 2]:
    old_label = code_to_old_label.get(code, '')
    new_label = old_to_new.get(old_label, f'Unknown_{code}')
    description.append(new_label)

def truncate_to_3_decimals(value: float) -> float:
    """Отбрасывает всё после 3 знаков после запятой (без округления)."""
    factor = 1000.0
    return int(value * factor) / factor

def sec_to_hms_ms_truncated(total_seconds: float) -> str:
    total_seconds = float(total_seconds)
    # Обрезаем до 3 знаков
    total_seconds = truncate_to_3_decimals(total_seconds)

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60

    # Форматируем: секунды с ровно 3 знаками, без округления (уже обрезаны)
    return f"{hours}:{minutes:02d}:{seconds:06.3f}"

fname_txt = fname_bdf.replace('.bdf', '_annotations.txt')
with open(fname_txt, 'w', encoding='utf-8') as f:
    f.write("onset\tduration\tdescription\n")
    for onset, _, desc in zip(onset_sec, duration, description):
        onset_str = sec_to_hms_ms_truncated(onset)
        # duration оставляем пустым
        f.write(f"{onset_str}{desc}\n")

print(f"Аннотации сохранены в: {fname_txt}")
