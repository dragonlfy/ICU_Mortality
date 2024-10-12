import os
import wfdb
from scripts.window_segment_reader import read_wave

folder = 'data/mimic3waveform/mimic3wdb-matched'
path = 'p00/p000079/p000079-2175-09-26-01-25n'

wave_path = os.path.join(folder, path)
# read_wave(fpath)
raw_signals, fields = wfdb.rdsamp(wave_path,  return_res=32)
idx = fields['sig_name'].index('ABPMean')
print(idx)
print(raw_signals[:,idx])
