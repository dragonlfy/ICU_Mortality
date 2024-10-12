## experiments running 

C1O2
w_ii_v_pleth_resp.n_hr_abp.gap2.seg4.mort14.surv26.samp4

C1O3
w_ii_v_pleth_resp.n_hr_abp.gap2.seg4.mort26.surv26.samp4


## segmenting

C1O1
w_ii_v_pleth_resp.n_hr_abp.gap2.seg4.mort14.surv14.samp4

## 分割数据
1. 更改 `task_config.py`
```python

elif name == 'combination1_option1':
    self._set_combination1_option1()

def _set_combination1_option1(self):
    self._wave_features = ['ii', 'v', 'pleth', 'resp']
    self._nume_features = ['hr', 'abpmean', 'abpsys', 'abpdias']
    self._minimal_time_gap = 2
    self._segment_length = 4
    self._mortality_window_length = 14  # deteriorating length
    self._survival_window_length = 14  # survivable length
    self._sampling_interval = 4  # stride

```
2. 改 `scripts/delimitate_window_sample_segments.py` 里的 config_name
就是上面的`name`

3. 
source ~/anaconda3/bin/activate physionet
python -m scripts.delimitate_window_sample_segments



## 跑实验步骤

0. 更改 CUDA
export CUDA_VISIBLE_DEVICES=2

source ~/anaconda3/bin/activate physionet

1. 改模型参数
在`models/attentive_inception_time.py`文件里。
要改的参数
ni_w=4, nf_1=16, ni_n=4, nf_2=32, 
nf_3=408, pred_nf=408,

ni_w: wave的变量的数量
ni_n: nume的变量的数量
nf_3 = (ni_w+ni_n+43) * 8

要求:
nf_1 % ni_w == 0
nf_2 % (ni_w+ni_n) == 0
nf_3 % ni_3 == 0

2. run task
```
python run_task_attentive_classfication.py
```
