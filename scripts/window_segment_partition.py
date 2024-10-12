import numpy as np
from datetime import timedelta
from copy import deepcopy
from task_config import Task_Config
import pickle as pkl
from .window_segment_output_desc import output_segment_desc
from .simple_hcv import split_hcv


def partition_wn_record(
        hcv_wave_nume_dict, hcv, nume_sig, wave_sig,
        mortality_inunit, desc_dict_list, wfile, task_config: Task_Config):

    minimal_time_gap = task_config.minimal_time_gap
    segment_length = task_config.segment_length
    mortality_window_length = task_config.mortality_window_length
    survival_window_length = task_config.survival_window_length
    sampling_interval = task_config.sampling_interval
    data_dir = task_config.data_dir

    intime = hcv_wave_nume_dict['INTIME']
    los2 = hcv_wave_nume_dict['LOS2']
    end_time = intime + timedelta(hours=los2)
    wave_base_time = hcv_wave_nume_dict['WAVE_BASE_DATETIME']
    wave_time_len = hcv_wave_nume_dict['WAVE_TIME_LEN']
    wave_end_time = wave_base_time + timedelta(hours=wave_time_len)
    nume_base_time = hcv_wave_nume_dict['NUME_BASE_DATETIME']
    nume_time_len = hcv_wave_nume_dict['NUME_TIME_LEN']
    nume_end_time = nume_base_time + timedelta(hours=nume_time_len)

    lastest_pred_time = end_time - timedelta(hours=(minimal_time_gap))
    if nume_sig.shape[0] > 0 and wave_sig.shape[0] > 0:
        earliest_pred_time = max(wave_base_time, nume_base_time)
        lastest_pred_time = min(lastest_pred_time, wave_end_time,
                                nume_end_time)
    elif nume_sig.shape[0] > 0:
        earliest_pred_time = nume_base_time
        lastest_pred_time = min(lastest_pred_time, nume_end_time)
    elif wave_sig.shape[0] > 0:
        earliest_pred_time = wave_base_time
        lastest_pred_time = min(lastest_pred_time, wave_end_time)
    else:
        return
    wn_end_los2 = (lastest_pred_time - intime).total_seconds() / 3600.

    if mortality_inunit == 1:
        pred_time = end_time - timedelta(hours=mortality_window_length)
        earliest_pred_time = max(earliest_pred_time, pred_time)
    else:
        if survival_window_length < 0:
            earliest_pred_time = max(earliest_pred_time, intime)
        else:
            pred_time = end_time - timedelta(hours=survival_window_length)
            earliest_pred_time = max(earliest_pred_time, pred_time)

    wn_base_los2 = (earliest_pred_time - intime).total_seconds() / 3600.
    len_time_window_to_split = wn_end_los2 - wn_base_los2
    wn_desc_dict = {**deepcopy(hcv_wave_nume_dict),
                    'seg_base_los2': wn_base_los2,
                    'seg_end_los2': wn_end_los2, }

    if len_time_window_to_split < segment_length:
        # Insufficient Length
        useable_hcv_length = \
            (end_time - timedelta(hours=(minimal_time_gap))
             - earliest_pred_time).total_seconds() / 3600.
        useable_wave_length = \
            (wave_end_time - earliest_pred_time).total_seconds() / 3600.
        useable_nume_length = \
            (nume_end_time - earliest_pred_time).total_seconds() / 3600.

        wn_desc_dict['Time_Window'] = {
            'ErrorMsg': 'Insufficient Length',
            'wn_base_los2': wn_base_los2,
            'wn_end_los2': wn_end_los2,
            'useable_hcv_length': useable_hcv_length,
            'useable_wave_length': useable_wave_length,
            'useable_nume_length': useable_nume_length, }
        output_segment_desc(wn_desc_dict, wfile)
        return

    seg_end_los2 = wn_end_los2
    while (wn_base_los2 + segment_length) < seg_end_los2 <= wn_end_los2:
        # seg_end_los2: [segment_length + wn_base_los2, wn_end_los2]
        seg_base_los2 = seg_end_los2 - segment_length
        desc_dict = {**wn_desc_dict,
                     'seg_base_los2': seg_base_los2,
                     'seg_end_los2': seg_end_los2, }
        seg_hcv_flag, seg_desc_dict, seg_data_dict = split_hcv(
            hcv, seg_base_los2, seg_end_los2)

        if not seg_hcv_flag:
            seg_end_los2 -= 1
            desc_dict['hcv_desc'] = seg_desc_dict
            output_segment_desc(desc_dict, wfile)
            continue

        desc_dict['hcv_flag'] = True
        seg_begin_datetime = intime + timedelta(hours=seg_base_los2)

        nume_begin_seconds = (seg_begin_datetime -
                              nume_base_time).total_seconds()
        nume_fs = hcv_wave_nume_dict['nume_desc']['fs']
        nume_len = int(segment_length * 3600. * nume_fs)
        if nume_sig.shape[0] > 0:
            nume_begin_idx = int(nume_begin_seconds * nume_fs)
            nume_end_idx = nume_begin_idx + nume_len

            desc_dict['nume_desc'].update(
                nume_begin_idx=nume_begin_idx,
                nume_end_idx=nume_end_idx,
            )
            nume_begin_idx_cor, nume_end_idx_cor = nume_begin_idx, nume_end_idx
            # corrected
            if nume_begin_idx < 0:
                nume_begin_idx_cor = 0
                nume_end_idx_cor = nume_len

            elif nume_end_idx > nume_sig.shape[1]:
                nume_end_idx_cor = nume_sig.shape[1]
                nume_begin_idx_cor = nume_end_idx_cor - nume_len

            if nume_end_idx_cor > nume_sig.shape[1]:
                desc_dict['nume_desc']['ErrorMsg'] = \
                    "corrected end index out of range " +\
                    f"({nume_end_idx_cor} > {nume_sig.shape[1]})"
                output_segment_desc(desc_dict, wfile)
                seg_end_los2 -= 1
                continue
            if nume_begin_idx_cor < 0:
                desc_dict['nume_desc']['ErrorMsg'] = \
                    f"corrected begin index < 0 ({nume_begin_idx_cor})"
                output_segment_desc(desc_dict, wfile)
                seg_end_los2 -= 1
                continue

            nume_seg = nume_sig[:, nume_begin_idx_cor: nume_end_idx_cor]
            count_nonzero = np.count_nonzero(np.abs(nume_seg) < -1e-3)
            ratio_zeros = count_nonzero / nume_sig.size
            desc_dict['nume_desc']['ratio_zeros'] = ratio_zeros
            desc_dict['nume_flag'] = True
            seg_data_dict['nume_seg'] = nume_seg
        else:
            desc_dict['nume_desc']['ratio_zeros'] = 1.
            desc_dict['nume_flag'] = True
            seg_data_dict['nume_seg'] = np.empty((0, nume_len))
            # numeric ends

        wave_fs = hcv_wave_nume_dict['wave_desc']['fs']
        wave_len = int(segment_length * 3600. * wave_fs)
        if wave_sig.shape[0] > 0:
            wave_begin_seconds = (seg_begin_datetime -
                                  wave_base_time).total_seconds()
            wave_begin_idx = int(wave_begin_seconds * wave_fs)
            wave_end_idx = wave_begin_idx + wave_len
            desc_dict['wave_desc'].update(
                wave_begin_idx=wave_begin_idx,
                wave_end_idx=wave_end_idx
            )
            wave_begin_idx_cor, wave_end_idx_cor = wave_begin_idx, wave_end_idx
            # corrected
            if wave_begin_idx < 0:
                wave_begin_idx_cor = 0
                wave_end_idx_cor = wave_len

            elif wave_end_idx > wave_sig.shape[1]:
                wave_end_idx_cor = wave_sig.shape[1]
                wave_begin_idx_cor = wave_end_idx_cor - wave_len

            if wave_end_idx_cor > wave_sig.shape[1]:
                desc_dict['wave_desc']['ErrorMsg'] = \
                    "end index out of range " + \
                    f"({wave_end_idx_cor} > {wave_sig.shape[1]})"
                output_segment_desc(desc_dict, wfile)
                seg_end_los2 -= 1
                continue
            if wave_begin_idx_cor < 0:
                desc_dict['wave_desc']['ErrorMsg'] = \
                    f"corrected begin index < 0 ({wave_begin_idx_cor})"
                output_segment_desc(desc_dict, wfile)
                seg_end_los2 -= 1
                continue

            wave_seg = wave_sig[:, wave_begin_idx_cor: wave_end_idx_cor]
            count_nonzero = np.count_nonzero(np.abs(wave_seg) < -1e-3)
            ratio_zeros = count_nonzero / wave_seg.size
            desc_dict['wave_desc']['ratio_zeros'] = ratio_zeros
            if ratio_zeros > 0.2:
                desc_dict['wave_desc']['ErrorMsg'] = \
                    f"{ratio_zeros*100.:.1f}% elements are zeros"
                output_segment_desc(desc_dict, wfile)
                seg_end_los2 -= 1
                continue

            desc_dict['wave_flag'] = True
            seg_data_dict['wave_seg'] = wave_seg
        else:
            desc_dict['nume_desc']['ratio_zeros'] = 1.
            desc_dict['nume_flag'] = True
            seg_data_dict['wave_seg'] = np.empty((0, wave_len))
            # wave ends

        gender = 1. if hcv_wave_nume_dict['GENDER'] == 'M' else 0.
        seg_data_dict.update(AGE=hcv_wave_nume_dict['AGE'], GENDER=gender)

        output_segment_desc(desc_dict, wfile)
        desc_dict_list.append(desc_dict)
        saving_path = f"{data_dir}/" + \
            f"{mortality_inunit}_{hcv_wave_nume_dict['SUBJECT_ID']}_" + \
            f"{hcv_wave_nume_dict['ICUSTAY_ID']}_" + \
            f"{seg_base_los2:.0f}_{seg_end_los2:.0f}.pkl"

        with open(saving_path, 'wb') as wbfile:
            pkl.dump(seg_data_dict, wbfile)

        seg_end_los2 -= sampling_interval
