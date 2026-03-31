from scipy.io import wavfile
import matplotlib.pyplot as plt

fs, data = wavfile.read(r'C:\Users\msasha\Desktop\AStim\stim\da_syllable_fixed.wav')

print(fs)
print(data)
plt.figure(figsize=(12, 4))
plt.plot(data)
plt.show()