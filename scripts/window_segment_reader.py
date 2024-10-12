import wfdb
import numpy as np
from numpy import ndarray


def _convert_nume_sig_names(sig_names):
    new_sig_names = []
    for sig_name in sig_names:
        sig_name = sig_name.lower().replace(' ', '')
        if sig_name == r'%spo2':
            new_sig_names.append('spo2')
        else:
            new_sig_names.append(sig_name)

    return new_sig_names


def read_nume(nume_path: str, nume_feats: list[str]) \
        -> tuple[dict[str, int], ndarray]:
    raw_signals, fields = wfdb.rdsamp(nume_path,  return_res=32)
    raw_signals = raw_signals.transpose()
    sig_names = _convert_nume_sig_names(fields['sig_name'])
    selected_feat_idxs = [sig_names.index(feat) for feat in nume_feats]
    signals: ndarray = raw_signals[np.array(selected_feat_idxs)]
    np.nan_to_num(signals, False, )
    fs = fields['fs']
    if fs > 1. / 60. + 1e-5:
        assert 1. - 1e-5 < fs < 1. + 1e-5, f"unexcepted fs ({fs:.2f})"
        signals = signals[:, :int(signals.shape[1] // 60 * 60)]
        signals = signals.reshape((len(nume_feats), -1, 60))
        signals = np.mean(signals, 2)
        fields['sig_len'] = signals.shape[1]

    fields['fs'] = 1. / 60.
    del fields['base_date']
    del fields['base_time']
    del fields['n_sig']
    time_len_hour = fields['sig_len'] / fields['fs'] / 3600.
    fields['time_len_hour'] = f"{time_len_hour:.1f}"
    return fields, signals


def _convert_wave_sig_names(sig_names):
    new_sig_names = [
        sig_name.lower().replace(' ', '')
        for sig_name in sig_names
    ]
    return new_sig_names


def read_wave(wave_path: str, wave_feats: list[str]) \
        -> tuple[dict[str, int], ndarray]:
    raw_signals, fields = wfdb.rdsamp(wave_path,  return_res=32)
    raw_signals = raw_signals.transpose()
    sig_names = _convert_wave_sig_names(fields['sig_name'])
    selected_feat_idxs = [sig_names.index(feat) for feat in wave_feats]
    signals: ndarray = raw_signals[np.array(selected_feat_idxs)]
    np.nan_to_num(signals, False, )
    del fields['base_date']
    del fields['base_time']
    del fields['n_sig']
    time_len_hour = fields['sig_len'] / fields['fs'] / 3600.
    fields['time_len_hour'] = f"{time_len_hour:.1f}"
    return fields, signals
