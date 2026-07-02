import mne
import matplotlib.pyplot as plt
from functions import extract_n_events, select_events, compute_GA,plot_GA

#fname = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\test_triggers\NeoRec_2026-06-18_13-41-42.bdf'

ch_name =  ['Fp1 (2)-CVII (70', 'F7 (3)-CVII (70)', 'Fp2 (5)-CVII (70', 'F8 (6)-CVII (70)']
fname = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\FFR_diamond\non_filt\Da_base.bdf'
out = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\FFR_diamond\raw_fif_all_chs.fif'
label_6 = '6_low'
label_7 = '7_low'
n_6low = [500]
n_7low = [500]
tmin = -0.1


raw = mne.io.read_raw_bdf(
    fname,
    preload=True,  # Загружаем данные в память сразу
    verbose=True  # Подробный вывод процесса
)

events, event_dict = mne.events_from_annotations(raw)

available_6low, selected_events_6low, selected_indices_6low = extract_n_events(
    events,
    event_dict,
    label=label_6,
    n=n_6low,
    random_selection=True
)
available_7low, selected_events_7low, selected_indices_7low = extract_n_events(
    events,
    event_dict,
    label=label_7,
    n=n_7low,
    random_selection=True
)
# Preprocessing 2: Epoching with baseline
available_6low, available_7low, sorted_events = select_events(n_6low, n_7low, label_6, label_7, events, event_dict)

# Data segmentation (epoching)
print(raw.info['ch_names'])
#raw_selected = raw.copy().pick_channels(['Fpz (1)-CVII (70'])
raw_selected = raw.copy().pick_channels(['Fp1 (2)-CVII (70', 'F7 (3)-CVII (70)', 'Fp2 (5)-CVII (70', 'F8 (6)-CVII (70)'])
epochs = mne.Epochs(
    raw_selected,
    sorted_events,
    tmin=-0.1,
    tmax=0.3,
    baseline=(-0.1, 0),
    preload=True
)
noise = False
grand_average = compute_GA(epochs, noise, tmin)
print(grand_average)
grand_average.plot(
    spatial_colors=False,
    gfp=False,
    show=False,
    verbose=None
)
#plt.title('Grand Average:Fpz (1)-CVII (70')
plt.title('Grand Average: Fp1 (2)-CVII , F7 (3)-CVII, M1 (4)-CVII , Fp2 (5)-CVII , F8 (6)-CVII, M2 (7)-CVII', fontsize=14)
plt.show()
"""