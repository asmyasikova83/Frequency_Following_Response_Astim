import os
import mne


#ch_name =  ['Fp1 (2)-CVII (70', 'F7 (3)-CVII (70)', 'Fp2 (5)-CVII (70', 'F8 (6)-CVII (70)']
#ch_name =  ['Cz (8)-CVII (70)']
ch_name = ['8', '4', '7'] #Cz, M1, M2
#ch_name = ['1', '4', '7']  #Fpz, M1, M2
#ch_name = ['4', '7', '8', '9', '10', '11', '12', '13', '14'] #Cz + inner circle

"""
ch_name = ['4',   '7', '8', '9',  '10', '11', '12', '13', '14',
           '15', '16', '17', '18', '19', '20', '21', '22', '23',
           '24', '25', '26', '27', '28', '29', '30', '31', '32',
           '33', '34', '35', '36', '37', '38', '39', '40', '41',
           '42', '43', '44', '45', '46', '47', '48', '49', '50',
           '51', '52', '53', '54', '55', '56', '57', '58', '59',
           '60', '61', '62', '63', '64', '65', '66', '67', '68']


ch_name = ['4',   '7', '8', '9',  '10', '11', '12', '13', '14',
           '15', '16', '17', '18', '19', '20', '21', '22', '23',
           '24', '25', '26'] #Cz + inner I and K circles

ch_name = ['4',   '7', '8', '9', '15', '27', '45'] # axis Cz+ I1 K1 L1 E1

"""
#'8' - Cz
#'4', '7'  - M1 M2

#fname = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\FFR_diamond\non_filt\Da_base_raw.bdf'
fname = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\FFR_diamond\non_filt\Da_20+_raw.bdf'
#out = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\FFR_diamond\raw_fif_all_chs_raw.fif'
#out = (r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\FFR_diamond\non_filt\raw_fif_4ch.fif')
#out = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\FFR_diamond\raw_fif_[Cz, M1, M2].fif'
#out = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\FFR_diamond\raw_fif_[Fpz, M1, M2].fif'
#out = r'\\MCSSERVER\DB Temp\physionet.org\FFR\data\FFR_diamond\raw_fif_all_chs_raw.fif'
#out = fr'\\MCSSERVER\DB Temp\physionet.org\FFR\data\FFR_diamond\raw_fif_8_68_raw.fif'
#out = fr'\\MCSSERVER\DB Temp\physionet.org\FFR\data\FFR_diamond\raw_fif_8_68_Da+_raw.fif'
out = fr'\\MCSSERVER\DB Temp\physionet.org\FFR\data\FFR_diamond\raw_fif_{ch_name}_Da+_raw.fif'

#Step 1

raw = mne.io.read_raw_bdf(
    fname,
    include=ch_name,
    preload=True,  # Загружаем данные в память сразу
    verbose=True  # Подробный вывод процесса
)

raw.save(out, overwrite=True)
"""


#Step 2
raw_fif = mne.io.read_raw_fif(out)
#raw_fif.plot(block=True)
raw_fif.load_data()
data = raw_fif.get_data()
print(" raw_selected", data)
"""