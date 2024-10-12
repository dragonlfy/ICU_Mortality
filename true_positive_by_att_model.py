from tqdm import tqdm
import random
import numpy as np
import torch as th
from models import HCV_NUME_WAVE_Attentive_InceptionTime
from task_config import Task_Config
from datamodule import MIMIC3Data, MIMIC3DataModule


if __name__ == '__main__':
    seed = 31415
    random.seed(seed)
    np.random.seed(seed)
    th.random.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    config_name = 'ii_abp_gap1_seg4_mort07_surv25_samp4'
    task_config = Task_Config(config_name)
    dropout_p = 0.5
    device = th.device('cuda:1')
    model = HCV_NUME_WAVE_Attentive_InceptionTime(
        ni_w=len(task_config.wave_features),
        ni_n=len(task_config.nume_features),
        dropout_p=dropout_p)

    ckpt_path = 'logs/00/838d22a/att_ii_abp_gap1_seg4_mort07_surv25_samp4/'
    ckpt_path += '20220922T192730/models/0/74.ckpt'
    model = model.to(device)
    model.load_state_dict(th.load(ckpt_path))
    model.eval()

    fold_idx = 0
    n_splits = 5
    multiple_negative = 3
    sample_last_k = 1
    batch_size = 8
    num_workers = 8
    datamodule = MIMIC3DataModule(
        task_config, fold_idx, multiple_negative,
        batch_size, num_workers, n_splits, sample_last_k)
    test_loader = datamodule.test_loader()
    gid = 0
    with open('xxx.csv', 'w') as wfile:
        print('id, fpath, label, prediction', file=wfile)
        for batch in tqdm(test_loader):
            batch: MIMIC3Data = batch.move_to(device)
            fpath = batch.fpath
            labels = batch.mortality.flatten().detach().cpu().numpy().tolist()
            out = model(batch).flatten().detach().cpu().numpy().tolist()
            for idx in range(len(fpath)):
                print(f"{gid}, {fpath[idx]}, {labels[idx]}, {out[idx]}", file=wfile)
                gid += 1
    print()
    