from scipy.io import wavfile
import matplotlib.pyplot as plt

fs, data = wavfile.read(r'C:\Users\msasha\Desktop\AStim\audio_with_triggers\sin_800Hz_0.1s_isi0.3s_10reps_sf44100Hz_A0.5_triggers.wav')


right_channel = data  # правый канал

plt.figure(figsize=(12, 4))
plt.plot(right_channel[:30000])
plt.show()