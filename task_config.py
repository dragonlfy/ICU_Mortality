from copy import copy

_avaliable_feats = {
    'w_ii',
    'w_v',
    'w_pleth',
    'w_resp',
    'w_abp',
    'n_hr',
    'n_spo2',
    'n_resp',
    'n_pulse',
    'n_nbpsys',
    'n_nbpmean',
    'n_nbpdias',
    'n_abpmean',
    'n_abpsys',
    'n_abpdias',
}


class Task_Config:
    def __init__(self, name):
        print('config name', name)
        self._name = name
        if name is None:
            self._set_ii_abp_gap1_seg4_mort07_surv25_samp4()
        # elif name == 'ii_abp_gap1_seg4_mort07_surv25_samp4':
        #     self._set_ii_abp_gap1_seg4_mort07_surv25_samp4()
        elif name == 'combination1_option1':
            self._set_combination1_option1()
        elif name == 'combination1_option2':
            self._set_combination1_option2()
        elif name == 'combination1_option3':
            self._set_combination1_option3()
        elif name == 'combination2_option1':
            self._set_combination2_option1()
        elif name == 'combination2_option2':
            self._set_combination2_option2()
        elif name == 'combination2_option3':
            self._set_combination2_option3()
        elif name == 'combination3_option1':
            self._set_combination3_option1()
        elif name == 'combination3_option2':
            self._set_combination3_option2()
        elif name == 'combination3_option3':
            self._set_combination3_option3()
        elif name == 'combination4_option1':
            self._set_combination4_option1()
        elif name == 'combination4_option2':
            self._set_combination4_option2()
        elif name == 'combination4_option3':
            self._set_combination4_option3()
        elif name == 'combination5_option1':
            self._set_combination5_option1()
        elif name == 'combination5_option2':
            self._set_combination5_option2()
        elif name == 'combination5_option3':
            self._set_combination5_option3()
        elif name == 'combination5_option4':
            self._set_combination5_option4()
        elif name == 'ii_abp_gap1_seg4_mort09_surv13_samp2':
            self._set_ii_abp_gap1_seg4_mort09_surv13_samp2()
        elif name == 'ii_abp_gap2_seg4_mort10_surv14_samp2':
            self._set_ii_abp_gap2_seg4_mort10_surv14_samp2()
        elif name == 'ii_abp_gap3_seg4_mort11_surv15_samp2':
            self._set_ii_abp_gap3_seg4_mort11_surv15_samp2()
        else:
            raise KeyError(f"Error name {name}")

        self._selected_features = ['w_'+feat
                                   for feat in self._wave_features]
        self._selected_features += {'n_'+feat
                                    for feat in self._nume_features}
        invalid_features = set(self._selected_features) - _avaliable_feats
        assert len(invalid_features) == 0, \
            f"Invalid features {invalid_features}"
        self._features_abbr = 'w_' + ('_'.join(self._wave_features))
        self._features_abbr += '.n_' + '_'.join(sorted(
            {feat[:3] for feat in self._nume_features}))

    def _set_combination1_option1(self):
        self._wave_features = ['ii', 'v', 'pleth', 'resp']
        self._nume_features = ['hr', 'nbpmean', 'nbpsys', 'nbpdias']
        self._minimal_time_gap = 2
        self._segment_length = 4
        self._mortality_window_length = 14  # deteriorating length
        self._survival_window_length = 14  # survivable length
        self._sampling_interval = 4  # stride

    def _set_combination1_option2(self):
        self._wave_features = ['ii', 'v', 'pleth', 'resp']
        self._nume_features = ['hr', 'abpmean', 'abpsys', 'abpdias']
        self._minimal_time_gap = 2
        self._segment_length = 4
        self._mortality_window_length = 14  # deteriorating length
        self._survival_window_length = 26  # survivable length
        self._sampling_interval = 4  # stride

    def _set_combination1_option3(self):
        self._wave_features = ['ii', 'v', 'pleth', 'resp']
        self._nume_features = ['hr', 'abpmean', 'abpsys', 'abpdias']
        self._minimal_time_gap = 2
        self._segment_length = 4
        self._mortality_window_length = 26  # deteriorating length
        self._survival_window_length = 14  # survivable length
        self._sampling_interval = 4  # stride

    def _set_combination2_option1(self):
        self._wave_features = ['ii', 'v']
        self._nume_features = ['hr', 'abpmean', 'abpsys', 'abpdias']
        self._minimal_time_gap = 2
        self._segment_length = 4
        self._mortality_window_length = 14  # deteriorating length
        self._survival_window_length = 14  # survivable length
        self._sampling_interval = 4  # stride

    def _set_combination2_option2(self):
        self._wave_features = ['ii', 'v']
        self._nume_features = ['hr', 'abpmean', 'abpsys', 'abpdias']
        self._minimal_time_gap = 2
        self._segment_length = 4
        self._mortality_window_length = 14  # deteriorating length
        self._survival_window_length = 26  # survivable length
        self._sampling_interval = 4  # stride

    def _set_combination2_option3(self):
        self._wave_features = ['ii', 'v']
        self._nume_features = ['hr', 'abpmean', 'abpsys', 'abpdias']
        self._minimal_time_gap = 2
        self._segment_length = 4
        self._mortality_window_length = 26  # deteriorating length
        self._survival_window_length = 14  # survivable length
        self._sampling_interval = 4  # stride

    def _set_combination3_option1(self):
        self._wave_features = ['ii', 'v', 'pleth', 'resp']
        self._nume_features = ['hr', 'nbpmean', 'nbpsys', 'nbpdias']
        self._minimal_time_gap = 2
        self._segment_length = 4
        self._mortality_window_length = 14  # deteriorating length
        self._survival_window_length = 14  # survivable length
        self._sampling_interval = 4  # stride

    def _set_combination3_option2(self):
        self._wave_features = ['ii', 'v', 'pleth', 'resp']
        self._nume_features = ['hr', 'nbpmean', 'nbpsys', 'nbpdias']
        self._minimal_time_gap = 2
        self._segment_length = 4
        self._mortality_window_length = 14  # deteriorating length
        self._survival_window_length = 26  # survivable length
        self._sampling_interval = 4  # stride

    def _set_combination3_option3(self):
        self._wave_features = ['ii', 'v', 'pleth', 'resp']
        self._nume_features = ['hr', 'nbpmean', 'nbpsys', 'nbpdias']
        self._minimal_time_gap = 2
        self._segment_length = 4
        self._mortality_window_length = 26  # deteriorating length
        self._survival_window_length = 14  # survivable length
        self._sampling_interval = 4  # stride

    def _set_combination4_option1(self):
        self._wave_features = ['ii', 'v']
        self._nume_features = ['hr', 'nbpmean', 'nbpsys', 'nbpdias']
        self._minimal_time_gap = 2
        self._segment_length = 4
        self._mortality_window_length = 14  # deteriorating length
        self._survival_window_length = 14  # survivable length
        self._sampling_interval = 4  # stride

    def _set_combination4_option2(self):
        self._wave_features = ['ii', 'v']
        self._nume_features = ['hr', 'nbpmean', 'nbpsys', 'nbpdias']
        self._minimal_time_gap = 2
        self._segment_length = 4
        self._mortality_window_length = 14  # deteriorating length
        self._survival_window_length = 26  # survivable length
        self._sampling_interval = 4  # stride

    def _set_combination4_option3(self):
        self._wave_features = ['ii', 'v']
        self._nume_features = ['hr', 'nbpmean', 'nbpsys', 'nbpdias']
        self._minimal_time_gap = 2
        self._segment_length = 4
        self._mortality_window_length = 26  # deteriorating length
        self._survival_window_length = 14  # survivable length
        self._sampling_interval = 4  # stride
        
    def _set_combination5_option1(self):
        self._wave_features = ['ii']
        self._nume_features = ['hr', 'abpmean', 'abpsys', 'abpdias']
        self._minimal_time_gap = 12
        self._segment_length = 4
        self._mortality_window_length = 24  # deteriorating length
        self._survival_window_length = 24  # survivable length
        self._sampling_interval = 4  # stride
        
    def _set_combination5_option2(self):
        self._wave_features = ['ii', 'v', 'pleth', 'resp']
        self._nume_features = ['hr', 'abpmean', 'abpsys', 'abpdias']
        self._minimal_time_gap = 12
        self._segment_length = 4
        self._mortality_window_length = 24  # deteriorating length
        self._survival_window_length = 24  # survivable length
        self._sampling_interval = 4  # stride
        
    def _set_combination5_option3(self):
        self._wave_features = ['ii']
        self._nume_features = ['hr', 'nbpmean', 'nbpsys', 'nbpdias']
        self._minimal_time_gap = 12
        self._segment_length = 4
        self._mortality_window_length = 24  # deteriorating length
        self._survival_window_length = 24  # survivable length
        self._sampling_interval = 4  # stride   

    def _set_combination5_option4(self):
        self._wave_features = ['ii', 'v', 'pleth', 'resp']
        self._nume_features = ['hr', 'nbpmean', 'nbpsys', 'nbpdias']
        self._minimal_time_gap = 12
        self._segment_length = 4
        self._mortality_window_length = 24  # deteriorating length
        self._survival_window_length = 24  # survivable length
        self._sampling_interval = 4  # stride

    def _set_ii_abp_gap1_seg4_mort07_surv25_samp4(self,):
        self._wave_features = ['ii', 'v', 'pleth', 'resp']
        self._nume_features = ['abpmean', 'abpsys', 'abpdias']
        self._minimal_time_gap = 1
        self._segment_length = 4
        self._mortality_window_length = 14  # deteriorating length
        self._survival_window_length = 14  # survivable length
        self._sampling_interval = 4  # stride

    def _set_ii_abp_gap1_seg4_mort09_surv13_samp2(self,):
        self._wave_features = ['ii', 'v', 'pleth', 'resp']
        self._nume_features = ['abpmean', 'abpsys', 'abpdias']
        self._minimal_time_gap = 2
        self._segment_length = 4
        self._mortality_window_length = 14
        self._survival_window_length = 26
        self._sampling_interval = 2

    def _set_ii_abp_gap2_seg4_mort10_surv14_samp2(self,):
        self._wave_features = ['ii']
        self._nume_features = ['nbpmean', 'nbpsys', 'nbpdias']
        self._minimal_time_gap = 2
        self._segment_length = 4
        self._mortality_window_length = 10
        self._survival_window_length = 14
        self._sampling_interval = 2

    def _set_ii_abp_gap3_seg4_mort11_surv15_samp2(self,):
        self._wave_features = ['ii']
        self._nume_features = ['abpmean', 'abpsys', 'abpdias']
        self._minimal_time_gap = 2  # minimal time interval
        self._segment_length = 4
        self._mortality_window_length = 11  # deteriorating state length
        self._survival_window_length = 15  # good state length
        self._sampling_interval = 2  # sampling stride

    @property
    def nume_features(self):
        return copy(self._nume_features)

    @property
    def wave_features(self):
        return copy(self._wave_features)

    @property
    def selected_features(self):
        return copy(self._selected_features)

    @property
    def minimal_time_gap(self):
        return self._minimal_time_gap

    @property
    def segment_length(self):
        return self._segment_length

    @property
    def mortality_window_length(self):
        return self._mortality_window_length

    @property
    def survival_window_length(self):
        # let it survival_window_length be -1
        # if full time is used
        return self._survival_window_length

    @property
    def sampling_interval(self):
        return self._sampling_interval

    def __repr__(self) -> str:
        repr_str = self._features_abbr
        repr_str += f".gap{self.minimal_time_gap}"
        repr_str += f".seg{self.segment_length}"
        repr_str += f".mort{self.mortality_window_length}"
        repr_str += f".surv{self._survival_window_length}"
        repr_str += f".samp{self._sampling_interval}"
        return repr_str

    @property
    def data_dir(self):
        if self._name in {
                          "combination1_option4",
                          "combination2_option4",
                          "combination3_option4",
                          "combination4_option4",
                          }:
            return f"data/tasks/{self}"

        return f"data/tasks/{self}"


if __name__ == '__main__':
    task_config = Task_Config('test')
    print(task_config)
    print(task_config.data_dir)
