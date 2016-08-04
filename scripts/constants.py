import os.path

TORCH_RNN_DIR = os.path.expanduser('~/torch-rnn') # NOTE: point this as necessary

BACHBOT_DIR = os.path.abspath(os.path.join(os.path.realpath(__file__), '..', '..'))
SCRATCH_DIR = BACHBOT_DIR + '/scratch'
OUT_DIR = BACHBOT_DIR + '/out'

CHORD_BOUNDARY_DELIM = '|||'
FERMATA_SYM = '(.)'

START_DELIM = unichr(1111)
END_DELIM = unichr(1115)

BLANK_MASK_TXT = '??' # for masking out tokens in harmonizations
BLANK_MASK_UTF = unichr(1130)

FRAMES_PER_CROTCHET = 4 # min resolution: 16th notes
#FRAMES_PER_CROTCHET = 2 # min resolution: 8th notes
