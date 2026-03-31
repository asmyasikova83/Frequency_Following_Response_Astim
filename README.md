# Frequency_Following_Response_Astim
Frequency_Following_Response https://en.wikipedia.org/wiki/Frequency_following_response

AStim  https://docs.mks.ru/ru/file/682f7130953d8#to-docs

triggers https://github.com/mcsltd/AStimWavPatcher/tree/master

create_wav.py : creates stimulus in audio format wav with triggers and inversion in predefined dir


creates sin


def create_repeated_sinusoidal_wav(
        dir,
        frequency,
        stimulus_duration,
        inter_stimulus_interval,
        amplitude,
        num_repetitions,
        trigger_delay,
        sample_rate,
        add_inv

)
creates stim with triggers and inversion using wav with DA syllable with varying pause (200 ms +- 20%)


def create_repeated_da_syllable_wav(
        dir,
        frequency,
        stimulus_duration,
        inter_stimulus_interval,
        amplitude,
        num_repetitions,
        trigger_delay,
        sample_rate,
        add_inv

)
usage :  python create_wav.py --F 800 --TS 100 --TP 300 --N 100 
         python create_wav.py --N 100 --INV 0
help:    python create_wav.py -h

analyze_triggers.py : explores time jitter during synch with amplifier

check_noise_filt.py : explores noise from hardware 

analyze_visualize_lorelli.py : visualizes FFR from https://purl.stanford.edu/cp051gh0103 dataset

ffr.py : preprocessing and visualization of DA syllable EP response (FFR)