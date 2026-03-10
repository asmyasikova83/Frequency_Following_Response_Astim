# Frequency_Following_Response_Astim
Frequency_Following_Response https://en.wikipedia.org/wiki/Frequency_following_response

AStim  https://docs.mks.ru/ru/file/682f7130953d8#to-docs

create_wav.py : creates sin in audio format wav with predefined dir,
        frequency,
        stimulus_duration,
        ramp_duration,
        inter_stimulus_interval,
        num_repetitions,
        sample_rate=44100,
        amplitude=0.5

usage :  python create_wav.py --frequency 800 --num_repetitions 100 --stimulus_duration 0.1  --inter_stimulus_interval 0.3 
         
help:    python create_wav.py -h