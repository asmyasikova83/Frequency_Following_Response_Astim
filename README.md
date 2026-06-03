# Frequency_Following_Response_Astim
Frequency_Following_Response https://en.wikipedia.org/wiki/Frequency_following_response

AStim  https://docs.mks.ru/ru/file/682f7130953d8#to-docs

triggers https://github.com/mcsltd/AStimWavPatcher/tree/master


Audio stimuli generation


creates WAV with audio stimuli Da syllable or sinusoidal tones with a predefined range of frequencies


Example call:  

               python create_wav.py  --function multiple_sin --F 110 220 440 880 --TS 100 --TP 100 --N 2 --INV 0

               python create_wav.py  --function repeated_da  --TS 100 --TP 100 --N 100 --INV 1

--F frequency


--TS length of 1 stimulus


--TP length of interstimulus interval (pause)


--N number of stimulus repetitions


--INV add polar (inverted) stimulus

Preprocessing and visualization of FFR

creates PDF with a picture of stimuli, its spectra, grand average of FFR after filtering and without filtering 
and FFR spectra,respectively. Puts PDF in the pics catalogue which is created in the data directory.

Example call:  

Analysis of FFR of DA syllable

            python command_line_ffr.py --subject S1 --short 'short' --TS 90 --tmin -50 
            --tmax 150 --fname_stim '\\MCSSERVER\DB Temp\physionet.org\FFR\stim\short\DA_syll_TS90.0ms_N4000_A100.0%_INV1.wav'

            python command_line_ffr.py --subject S1 --TS 250 --tmin -50 --tmax 300 
            --fname_stim '\\MCSSERVER\DB Temp\physionet.org\FFR\stim\Da_syll_TS250.0ms_TP200.0ms_N4000_Amplitude_INV1.wav'

Analysis of FFR of G note (sin):

            python command_line_ffr.py --subject S1 --short 'shortG' --TS 100 --TP 100 --tmin -50
            --tmax 150 --fname_stim '\\MCSSERVER\DB Temp\physionet.org\FFR\stim\short\G_note_TS100ms_TP100ms_N4000_A100%_INV1.wav'

Analysis of FFR dummy  data:

            python command_line_ffr.py --dummy 'dummy'--TS 250 --tmin -50 --tmax 300
            --fname_stim '\\MCSSERVER\DB Temp\physionet.org\FFR\stim\DA_syll_TS250ms_N4000_A100.0%_INV1.wav'

            python command_line_ffr.py --preamplifier 'False' --dummy 'dummy'--TS 250 --tmin -50 --tmax 300
            --fname_stim '\\MCSSERVER\DB Temp\physionet.org\FFR\stim\DA_syll_TS250ms_N4000_A100.0%_INV1.wav'

Test sin 150 Hz:

            python command_line_ffr.py --subject S1 --short 'short' --TS 100 --TP 100 --tmin -50 --tmax 150
            --fname_stim '\\MCSSERVER\DB Temp\physionet.org\FFR\stim\short\sin_[150]Hz_TS100.0s_TP100.0s_N4000_INV0.wav' 

pip install -r requirements.txt