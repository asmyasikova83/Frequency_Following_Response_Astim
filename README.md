# Frequency_Following_Response_Astim
Frequency_Following_Response https://en.wikipedia.org/wiki/Frequency_following_response

AStim  https://docs.mks.ru/ru/file/682f7130953d8#to-docs

triggers https://github.com/mcsltd/AStimWavPatcher/tree/master


Audio stimuli generation


creates WAV with audio stimuli - Da syllable or sinusoidal toned with a predefined range of frequencies


Example call:  python create_wav.py  --function multiple_sin --F 440 880 --TS 100 --TP 100 --N 1 --INV 0
               python create_wav.py  --function repeated_da --F 440 880 --TS 100 --TP 100 --N 100 --INV 1


ffr.py : preprocessing and visualization of DA syllable EP response (FFR)