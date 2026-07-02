import mne
import matplotlib.pyplot as plt
from functions import import_fif, extract_n_events, select_events, compute_GA,plot_GA

ch_name = ['4',   '7', '8', '9', '15', '27', '45'] # axis Cz+ I1 K1 L1 E1
#ch_name = ['10']
ref_chs = ['4', '7']
n_6low = [2000]
n_7low = [2000]
fs = 10000
tmin = -0.1

info = mne.create_info(
    ch_names=ch_name,
    sfreq=fs,
    ch_types='eeg'
)


fname = fr'\\MCSSERVER\DB Temp\physionet.org\FFR\data\FFR_diamond\raw_fif_{ch_name}_raw.fif'
label_6, label_7, raw_selected, raw = import_fif(ch_name, ref_chs, fname)
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

plt.title(f'{ch_name}', fontsize=14)
plt.show()