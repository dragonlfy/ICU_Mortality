import argparse
import os
import numpy as np
import pandas as pd
from task_config import Task_Config
from data_config import wnhcv_desc_path, mimic3stays_dir, matched_path
from typing import Any, Dict, Generator, Tuple
from pandas import DataFrame
from tqdm import tqdm
import os.path as osp
import pickle as pkl

from .simple_hcv import read_simple_hcv_record
from .window_segment_output_desc import output_segment_desc
from .window_segment_reader import read_nume, read_wave
from .window_segment_partition import partition_wn_record


def delimitate_window_and_sample_segments():
    args = parse_args()
    task_config = Task_Config(args.config_name)
    data_dir = task_config.data_dir
    print('Task name', task_config)
    os.makedirs(data_dir, exist_ok=args.force_reload)
    wfile = open(f"{data_dir}.segment.log", 'w')
    print(task_config, file=wfile)

    nume_features = task_config.nume_features
    wave_features = task_config.wave_features
    minimal_time_gap = task_config.minimal_time_gap
    segment_length = task_config.segment_length
    mortality_window_length = task_config.mortality_window_length
    survival_window_length = task_config.survival_window_length
    sampling_interval = task_config.sampling_interval
    print('nume_features:', ", ".join(nume_features), file=wfile)
    print('wave_features:', ", ".join(wave_features), flush=True, file=wfile)
    print('minimal_time_gap:', minimal_time_gap, file=wfile)
    print('segment_length:', segment_length, file=wfile)
    print('mortality_window_length:', mortality_window_length, file=wfile)
    print('survival_window_length:', survival_window_length, file=wfile)
    print('sampling_interval:', sampling_interval, file=wfile)

    dtype = {
        'SUBJECT_ID': int, 'ICUSTAY_ID': int, 'LOS2': float, 'WAVEFORMS': str,
        'WAVE_TIME_LEN': float, 'NUMERICS': str, 'NUME_TIME_LEN': float,
        'GENDER': str, 'MORTALITY_INUNIT': int, 'AGE': float,
    }
    parse_dates = ['INTIME', 'WAVE_BASE_DATETIME', 'NUME_BASE_DATETIME']
    wnhcv_desc = pd.read_csv(wnhcv_desc_path, dtype=dtype,
                             parse_dates=parse_dates)
    condition = True
    for feat in task_config.selected_features:
        condition = condition & (wnhcv_desc[feat] == 1)
    selected_wnhcv = wnhcv_desc[condition]
    wnhcv_icustay_grp = selected_wnhcv.groupby('ICUSTAY_ID')
    mortality_wnhcv_desc = selected_wnhcv[selected_wnhcv.MORTALITY_INUNIT == 1]
    mortality_desc_grp = mortality_wnhcv_desc.groupby('ICUSTAY_ID')
    print("number of avaliable icustays: ", len(wnhcv_icustay_grp), file=wfile)
    print("number of avaliable mortality icustays: ",
          len(mortality_desc_grp), file=wfile)
    print('', file=wfile, flush=True)

    wnhcv_icustay_grp_indices = wnhcv_icustay_grp.indices
    last_icustay_id = None
    desc_dict_list = None
    icustay__desc_dict_list = []
    icustay__label_list = []
    for hcv_wave_nume_dict, hcv, nume_sig, wave_sig in \
        _hcv_wave_nume_dict_generator(
            selected_wnhcv, wnhcv_icustay_grp_indices,
            nume_features, wave_features, wfile):

        icustay_id = hcv_wave_nume_dict['ICUSTAY_ID']
        mortality_inunit = hcv_wave_nume_dict['MORTALITY_INUNIT']
        if icustay_id != last_icustay_id:
            desc_dict_list = []
            icustay__desc_dict_list.append(desc_dict_list)
            icustay__label_list.append(mortality_inunit)
            last_icustay_id = icustay_id

        partition_wn_record(
            hcv_wave_nume_dict, hcv, nume_sig, wave_sig,
            mortality_inunit, desc_dict_list, wfile, task_config)
    # end looping _hcv_wave_nume_dict_generator

    icustay__desc_dict_list_2 = []
    icustay__label_list_2 = []
    for desces, label in zip(icustay__desc_dict_list, icustay__label_list):
        if len(desces) == 0:
            continue
        icustay__desc_dict_list_2.append(desces)
        icustay__label_list_2.append(label)

    num_mortality, num_subject_mortality = 0, 0
    for idx in range(len(icustay__desc_dict_list_2)):
        if icustay__label_list_2[idx] == 1:
            num_subject_mortality += 1
            num_mortality += len(icustay__desc_dict_list_2[idx])

    print(f"total {len(icustay__desc_dict_list_2)} " +
          f"/ mortality {num_mortality} /" +
          f" num_subject_mortality {num_subject_mortality}", file=wfile)
    wfile.close()
    with open(f"{data_dir}/icustay__desc_dict_list.pkl", 'wb') as wbfile:
        pkl.dump(icustay__desc_dict_list_2, wbfile)
    with open(f"{data_dir}/icustay__label_list.pkl", 'wb') as wbfile:
        pkl.dump(icustay__label_list_2, wbfile)


def _hcv_wave_nume_dict_generator(
    selected_wnhcv: pd.DataFrame, grp_indices: dict,
    nume_feats: list, wave_feats: list, wfile) -> \
        Generator[Tuple[Dict[str, Any], DataFrame], None, None]:
    """HCV-Wave-Nume dict generator

    Args:
        selected_wnhcv (pd.DataFrame): selected hcv-wave-nume description
        grp_indices (dict): hcv-wave-nume group indices
        wfile (_type_): wfile

    Yields:
        DataFrame, dict[str, Any]: hcv_wave_nume_dict, hcv
    """

    icustay_cols = ['SUBJECT_ID', 'GENDER', 'AGE',
                    'LOS2', 'INTIME', 'MORTALITY_INUNIT']
    wave_nume_cols = ['WAVEFORMS', 'WAVE_BASE_DATETIME', 'WAVE_TIME_LEN',
                      'NUMERICS', 'NUME_BASE_DATETIME', 'NUME_TIME_LEN']

    for icustay_id, indices in tqdm(grp_indices.items()):
        icustay_dict = selected_wnhcv.iloc[indices[0]][icustay_cols].to_dict()
        subject_id = icustay_dict['SUBJECT_ID']
        hcv_fname = f"episode_{icustay_id}_timeseries.csv"
        hcv_path = f"{mimic3stays_dir}/{subject_id}/{hcv_fname}"

        hcv_flag, hcv_desc_dict, hcv = read_simple_hcv_record(hcv_path)
        if not hcv_flag:
            desc_dict = {'ICUSTAY_ID': icustay_id, **icustay_dict,
                         'HCV ErrorMsg': hcv_desc_dict['ErrorMsg']}
            output_segment_desc(desc_dict, wfile)
            continue
        for idx in reversed(indices):
            wave_nume_dict = selected_wnhcv.iloc[idx][wave_nume_cols].to_dict()
            hcv_wave_nume_dict = {'ICUSTAY_ID': icustay_id,
                                  **icustay_dict, **wave_nume_dict}
            hcv_wave_nume_dict['hcv_flag'] = False
            hcv_wave_nume_dict['hcv_desc'] = dict(**hcv_desc_dict)

            if len(nume_feats) > 0 and \
                    isinstance(wave_nume_dict['NUMERICS'], str):
                nume_path = osp.join(matched_path, wave_nume_dict['NUMERICS'])
                nume_desc, nume_sig = read_nume(nume_path, nume_feats)
            else:
                nume_desc = {'fs': 0, 'time_len_hour': 0,
                             'nume_feats': nume_feats,
                             'nume': wave_nume_dict['NUMERICS']}
                nume_sig = np.empty((len(nume_feats), 0))
            hcv_wave_nume_dict['nume_flag'] = False
            hcv_wave_nume_dict['nume_desc'] = nume_desc

            if len(wave_feats) > 0 and \
                    isinstance(wave_nume_dict['WAVEFORMS'], str):
                wave_path = osp.join(matched_path, wave_nume_dict['WAVEFORMS'])
                wave_desc, wave_sig = read_wave(wave_path, wave_feats)
            else:
                wave_desc = {'fs': 0, 'time_len_hour': 0,
                             'wave_feats': wave_feats,
                             'wave': wave_nume_dict['WAVEFORMS']}
                nume_sig = np.empty((len(wave_feats), 0))
            hcv_wave_nume_dict['wave_flag'] = False
            hcv_wave_nume_dict['wave_desc'] = wave_desc

            yield hcv_wave_nume_dict, hcv, nume_sig, wave_sig


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config_name', type=str,
                        default='combination4_option3')
    parser.add_argument('--force_reload', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    delimitate_window_and_sample_segments()
