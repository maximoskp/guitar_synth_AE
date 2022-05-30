import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle
import data_utils
import os
import numpy as np
import data_utils

with open('data/guitar_and_synth_audio.pickle', 'rb') as handle:
    audios = pickle.load(handle)

size_in_samples = 1600

# XY matrices

