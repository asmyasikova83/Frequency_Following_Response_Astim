# Frequency_Following_Response_Astim Version 1
Frequency_Following_Response https://en.wikipedia.org/wiki/Frequency_following_response

________________________________________________________________________________________________________________________
Equipment and software

AStim https://mks.ru/ru/products/ep-erp https://docs.mks.ru/ru/file/682f7130953d8#to-docs

Triggers https://github.com/mcsltd/AStimWavPatcher/tree/master

NVX 36 https://mks.ru/en/products/nvx

MCScap https://mks.ru/en/products/mcscap

Electrodes MCScap-CS22 https://mcscap.ru/catalog/tes-elektrody-dlya-stimulyatsii/mcscap-cs22/

NeoRec 1.6 https://files.mks.ru/download/LLOyHWRfcqqhssudBIj8

Detailed description and instruction for Frequency_Following_Response_Astim Version 1 
https://files.mks.ru/download/k5nRHWAYzK3n0BQGwOmO

________________________________________________________________________________________________________________________



Audio stimuli generation


creates WAV with audio stimuli Da syllable or sinusoidal tones with a predefined range of frequencies


Example call:  

               python create_wav.py  --function multiple_sin --F 110 220 440 880 --TS 100 --TP 100 --N 2 --INV 0

               python create_wav.py  --function repeated_da  --TS 100 --TP 100 --N 100 --INV 1

--F frequency


--TS stimulus latency


--TP interstimulus interval (pause) latency 


--N number of stimulus repetitions


--INV add polar (inverted) stimulus


Preprocessing and visualization of FFR

creates PDF with a picture of stimuli, its spectra, grand average of FFR after filtering and without filtering 
and FFR spectra,respectively. Puts PDF in the pics catalogue which is created in the data directory.

Example call:  

           python command_line_ffr.py --TS 250 --TP 200 --fmin 80 --fmax 1500 --tmin -100 --tmax 300 --N 500

--TS: stimulus latency


--TP: interstimulus interval (pause) latency 


--fmin: lower cutoff frequency for filtering


--fmax: upper cutoff frequency for filtering


--tmin: lower boundary of the time window (ms)


--tmax: upper boundary of the time window (ms)


--N: number of stimulus repetitions


pip install -r requirements.txt