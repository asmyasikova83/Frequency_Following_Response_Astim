# Frequency_Following_Response_Astim
Frequency_Following_Response https://en.wikipedia.org/wiki/Frequency_following_response

AStim  https://docs.mks.ru/ru/file/682f7130953d8#to-docs

triggers https://github.com/mcsltd/AStimWavPatcher/tree/master

create_wav.py : creates sin in audio format wav with triggers in predefined dir,
        dir=args.dirname,
        frequency=args.F,
        stimulus_duration=args.TS,
        inter_stimulus_interval=args.TP,
        amplitude=args.A,
        num_repetitions=args.N,
        trigger_delay=args.TR0,
        sample_rate=args.SR
    )

usage :   python create_wav.py --F 800 --TS 100 --TP 300 --N 100 
         
help:    python create_wav.py -h