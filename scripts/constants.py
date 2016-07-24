import os.path

TORCH_RNN_DIR = os.path.expanduser('~/torch-rnn') # NOTE: point this as necessary

BACHBOT_DIR = os.path.abspath(os.path.join(os.path.realpath(__file__), '..', '..'))
SCRATCH_DIR = BACHBOT_DIR + '/scratch'
OUT_DIR = BACHBOT_DIR + '/out'

START_DELIM = unichr(1111)
PADDING = unichr(1100)
END_DELIM = unichr(1115)

CHORD_BOUNDARY_DELIM = '|||'

NOTE_START_SYM = '@'

REST_SYM = 'REST'
FERMATA_SYM = '(.)'

BLANK_MASK_TXT = '??'
BLANK_MASK_UTF = unichr(1130)

# quarterLength's in JCB chorales:
# [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 0.25, 8.0, 12.0, 0.125, 0.75, 1.5]
# Multiplying by 8 turns all durations into integers
# 4 => quantize to 16th notes
# TODO: quantize to 8th notes i.e. set to 2
FRAMES_PER_CROTCHET = 4
