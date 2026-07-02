import mne
import numpy as np
import os

#fname_bdf = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\test_triggers\test_DA_June26.bdf'
fname_bdf = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\FFR_diamond\non_filt\ffr_da_N4000_non_filtS0.bdf'
out =  r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\FFR_diamond\triggers'
raw = mne.io.read_raw_bdf(
    fname_bdf,
    include=['F7 (3)-CVII (70)'],
    preload=True,  # Загружаем данные в память сразу
    verbose=True  # Подробный вывод процесса
)

events_bdf, event_dict = mne.events_from_annotations(raw)

# onset в секундах
onset_sec = events_bdf[:, 0] / raw.info['sfreq']

# description берём из raw.annotations.description — там уже '6_high', '6_low' и т.д.
description = raw.annotations.description.tolist()

onset_by_label = {}
for onset, label in zip(onset_sec, description):
    onset_by_label.setdefault(label, []).append(onset)

base_name = os.path.basename(fname_bdf)
subject_id = base_name.rsplit('.', 1)[0]
# Сохраняем каждый тип в отдельный файл
for label, onsets in onset_by_label.items():
    # Имя файла: <2символа>_<тип_метки>.txt
    out_fname = os.path.join(out, f"{subject_id}_{label}.txt")
    with open(out_fname, 'w', encoding='utf-8') as f:
        # Пишем onset как есть (полная точность float64)
        for onset in onsets:
            f.write(f"{onset}\n")
    print(f"Сохранено: {out_fname} ({len(onsets)} событий)")

print("Готово.")

