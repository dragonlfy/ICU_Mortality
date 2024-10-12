import numpy as np
import pandas as pd
from data_config import (HCV_Var_Converters, HCV_Is_Categorical_Channel,
                         HCV_Var_Impute_Value, HCV_Num_Possible_Values)
from typing import Tuple, Dict
from scipy.interpolate import interp1d


def read_simple_hcv_record(filepath):
    try:
        hcv_flag = True
        hcv = pd.read_csv(filepath, converters=HCV_Var_Converters)
        hcv_desc_dict = {'last_hour': f"{hcv.Hours.to_numpy()[-1]:.1f}"}
    except Exception as e:
        hcv_flag = False
        hcv_desc_dict = {'ErrorMsg': e.args}
        hcv = None

    return hcv_flag, hcv_desc_dict, hcv


def split_hcv(hcv, begin_hour, end_hour) -> Tuple[bool, Dict, Dict]:

    seg_desc_dict = {}
    if begin_hour < 0:
        error_msg = f"Begin hour ({begin_hour:.1f}) < 0."
        seg_desc_dict['ErrorMsg'] = error_msg
        return False, seg_desc_dict, None

    num_hours = int(end_hour - begin_hour)
    cate_ts_list = []
    nume_ts_list = []

    visible_hcv = hcv[hcv['Hours'] < end_hour].copy()
    # visible time series, whose hours < end_hour
    visible_hours = visible_hcv['Hours'].to_numpy()
    seg_all_hcv = visible_hcv[visible_hcv['Hours'] >= begin_hour].copy()
    # segment time series, whose begin_hour <= hours < end_hour
    hour_vals = np.arange(begin_hour + 0.5, end_hour, 1)
    # hours of seg_all_hcv for interpolating
    max_count_nonzero = 0
    for feat_name, is_cate in HCV_Is_Categorical_Channel.items():
        seg_vals = seg_all_hcv[feat_name].to_numpy()
        notnan_mask = ~np.isnan(seg_vals)
        count_nonzero = np.count_nonzero(notnan_mask)
        seg_desc_dict[feat_name] = count_nonzero
        max_count_nonzero = max(max_count_nonzero, count_nonzero)
        if count_nonzero == 0:  # all values are none, using impute value
            impute_value = HCV_Var_Impute_Value[feat_name]
            if is_cate:
                impute_ts = impute_value * np.ones((num_hours), dtype=np.int64)
                cate_ts_list.append(impute_ts)
            else:
                impute_ts = impute_value * \
                    np.ones((num_hours), dtype=np.float32)
                nume_ts_list.append(impute_ts)
        elif count_nonzero < 3:  # using mean value
            notnan_vals = seg_vals[notnan_mask]
            val_mean = np.mean(notnan_vals)
            if is_cate:
                mean_ts = val_mean.astype(
                    int) * np.ones((num_hours), dtype=np.int64)
                cate_ts_list.append(mean_ts)
            else:
                mean_ts = val_mean * np.ones((num_hours), dtype=np.float32)
                nume_ts_list.append(mean_ts)
        else:  # > 3, interpolating time series
            visible_vals = visible_hcv[feat_name].to_numpy()
            notnan_visible_mask = ~np.isnan(visible_vals)
            notnan_visible_hours = visible_hours[notnan_visible_mask]
            notnan_visible_vals = visible_vals[notnan_visible_mask]
            interpolate_func = interp1d(notnan_visible_hours,
                                        notnan_visible_vals, 'nearest',
                                        bounds_error=False,
                                        fill_value='extrapolate')
            interped_val = interpolate_func(hour_vals)
            # interpe time series for $hour_vals based on $notnan_vals 
            # ranging from 0 to end_hour
            if is_cate:
                interped_val = np.around(interped_val, 0).astype(np.int64)
                num_possible = HCV_Num_Possible_Values[feat_name]
                interped_val[interped_val > (
                    num_possible - 1)] = (num_possible - 1)  # max value
                interped_val[interped_val < 0] = 0  # min value
                cate_ts_list.append(interped_val)
            else:
                nume_ts_list.append(interped_val.astype(np.float32))

    if max_count_nonzero < 3:
        error_msg = "no enough observations from "
        error_msg += f"{begin_hour:.1f}h to {end_hour:.1f}h."
        seg_desc_dict['ErrorMsg'] = error_msg
        return False, seg_desc_dict, None

    cate_ts_arr = np.stack(cate_ts_list)
    nume_ts_arr = np.stack(nume_ts_list)
    data_dict = {'category': cate_ts_arr, 'numeric': nume_ts_arr}
    return True, seg_desc_dict, data_dict
