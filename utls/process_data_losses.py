import os
import numpy as np
import pandas as pd
import mne
from datetime import datetime

max_int16 = np.iinfo(np.int16).max
min_int16 = np.iinfo(np.int16).min

fname_step1 = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\non_filt\ffr_da_N4000_non_filtS0_step1.bdf'
fname_step2 = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\non_filt\ffr_da_N4000_non_filt_step2_raw.bdf'

raw = mne.io.read_raw_bdf(
    fname_step2,
    preload=True,  # Загружаем данные в память сразу
    verbose=True  # Подробный вывод процесса
)
raw_selected = raw.copy().pick_channels(['1'])
ctime = os.path.getctime(fname_step1)
creation_time = datetime.fromtimestamp(ctime)
short_time = creation_time.strftime('%Y-%m-%d %H:%M')
print('creation_time', short_time )

data = raw_selected.get_data()



def find_all_gaps(data, zero_threshold=1e-20):
    """
    Ищет разные типы пропусков:
    - NaN
    - бесконечности
    - нулевые значения (ниже порога)
    """
    gaps = {}

    # NaN
    nan_mask = np.isnan(data)
    gaps['NaN'] = nan_mask

    # Бесконечности
    inf_mask = np.isinf(data)
    gaps['inf'] = inf_mask

    # Нулевые значения
    zero_mask = (np.abs(data) < zero_threshold)
    gaps['zero'] = zero_mask

    return gaps

# Поиск всех типов пропусков
gaps = find_all_gaps(data)

for gap_type, mask in gaps.items():
    count = mask.sum()
    if count > 0:
        print(f"{gap_type}: {count} пропусков")
        # Первые 5 индексов
        indices = np.where(mask)
        print("  Первые индексы:", list(zip(indices[0][:5], indices[1][:5])))
    else:
        print(f"{gap_type}: пропусков нет")

