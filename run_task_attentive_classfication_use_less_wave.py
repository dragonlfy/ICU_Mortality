import random
import torch as th
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from datamodule import MIMIC3Data, MIMIC3DataModule
from models import HCV_NUME_WAVE_Attentive_InceptionTime
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import os
import subprocess
import datetime
from task_config import Task_Config


def train(loader: DataLoader, exp: str, model: nn.Module, criterion,
          optimizer: Optimizer, device, num_epoches, epoch_idx):
    model.train()
    tqdm_bar = tqdm(loader, desc=f"Training {exp} {epoch_idx}/{num_epoches}")
    loss_list = []
    for batch in tqdm_bar:
        batch: MIMIC3Data = batch.move_to(device)
        batch_Y = batch.mortality
        logits = model(batch)
        loss: Tensor = criterion(logits, batch_Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        loss_list.append(loss_val)
        tqdm_bar.set_postfix(loss=loss_val)

    return np.mean(loss_list)


def evaluation(loader: DataLoader, exp: str, model: nn.Module, criterion,
               metrics_fn: nn.Module, device, num_epoches, epoch_idx,
               maxLastK=12):
    model.eval()
    tqdm_bar = tqdm(loader, desc=f"Evaluating {exp} {epoch_idx}/{num_epoches}")
    loss_list = []
    y_score_list = []
    Y_list = []
    icustay_list = []
    for batch in tqdm_bar:
        batch: MIMIC3Data = batch.move_to(device)
        icustay_list.extend(batch.icustay_id)
        batch_Y = batch.mortality
        logits = model(batch)
        y_score = th.sigmoid(logits)
        loss: Tensor = criterion(y_score, batch_Y)
        loss_list.append(loss.item())
        y_score_list.append(y_score.detach().cpu().numpy())
        Y_list.append(batch_Y.cpu().numpy())

    y_socre_all = np.vstack(y_score_list)
    Y_all = np.vstack(Y_list)
    macro_metrics_dict = metrics_fn(y_socre_all, Y_all)
    icu__score_label = {icustay: (list(), list())
                        for icustay in set(icustay_list)}

    y_socre_all = y_socre_all.flatten().tolist()
    Y_all = Y_all.flatten().tolist()
    for idx in range(len(icustay_list)):
        icustay = icustay_list[idx]
        y_socre = y_socre_all[idx]
        y_label = Y_all[idx]
        icu__score_label[icustay][0].append(y_socre)
        icu__score_label[icustay][1].append(y_label)

    label_list = []
    score_whole_all = []  # all prediction should be correct,
    # so maximal value prediction for survival patient is kept and
    # mimimal value prediction for mortal  patient is kept
    score_whole_avg = []
    score_lastK_all_dict = {
        lastK: []
        for lastK in range(1, maxLastK + 1, 1)
    }
    score_lastK_avg_list = {
        lastK: []
        for lastK in range(1, maxLastK + 1, 1)
    }

    # score_max_summary, score_avg_summary = [], []
    # score_last, score_last_2_mean = [], []
    for scores, labels in icu__score_label.values():
        assert len(set(labels)) == 1
        label = labels[0]
        label_list.append(label)
        score_whole_all.append(max(scores)
                               if label == 1 else min(scores))
        score_whole_avg.append(np.mean(scores))
        for lastK in range(1, maxLastK + 1, 1):
            last_scores = scores[-lastK:]
            score_last_all = score_lastK_all_dict[lastK]
            score_last_all.append(max(last_scores)
                                  if label == 1 else min(last_scores))
            score_last_avg = score_lastK_avg_list[lastK]
            score_last_avg.append(np.mean(last_scores))

    label_arr = np.array(label_list)
    scores_whole_all = np.array(score_whole_all)
    scores_whole_avg = np.array(score_whole_avg)
    metrics_whole_all = metrics_fn(scores_whole_all, label_arr)
    metrics_whole_avg = metrics_fn(scores_whole_avg, label_arr)
    metrics_lastK_all_dict = {}
    metrics_lastK_avg_dict = {}
    for lastK in range(1, maxLastK + 1, 1):
        scores_last_all = np.array(score_lastK_all_dict[lastK])
        metrics_last_all = metrics_fn(scores_last_all, label_arr)
        metrics_lastK_all_dict[lastK] = metrics_last_all
        scores_last_avg = np.array(score_lastK_all_dict[lastK])
        metrics_last_avg = metrics_fn(scores_last_avg, label_arr)
        metrics_lastK_avg_dict[lastK] = metrics_last_avg

    return macro_metrics_dict, metrics_whole_all, metrics_whole_avg, \
        metrics_lastK_all_dict, metrics_lastK_avg_dict


def metrics_fn(y_score, ys):
    y_pred = y_score > 0.5
    accuracy = metrics.accuracy_score(ys, y_pred)
    f1 = metrics.f1_score(ys, y_pred)
    precision = metrics.precision_score(ys, y_pred)
    recall = metrics.recall_score(ys, y_pred)  # true positive rate
    roc_auc = metrics.roc_auc_score(ys, y_score)
    tn, fp, fn, tp = metrics.confusion_matrix(ys, y_pred).ravel()
    specificity = tn / (tn + fp)
    metrics_dict = {'f1': f"{f1:.3f}", 'recall': f"{recall:.3f}",
                    'specificity': f"{specificity:.3f}",
                    'precision': f"{precision:.3f}",
                    'accuracy': f"{accuracy:.3f}",
                    'roc_auc': f"{roc_auc:.3f}",
                    'tn': f"{tn}", 'fp': f"{fp}",
                    'fn': f"{fn}", 'tp': f"{tp}",
                    }
    return metrics_dict


def run():
    multiple_negative = 3  # 3, 7
    learning_rate = 1e-3  # 1e-2, 1e-3, 1e-4
    batch_size = 32  # 16, 32, 64 # number of postive samples in a batch
    dropout_p = 0.5  # 0.5
    device = th.device('cuda:1')
    # device = torch.device('cpu')
    num_epoches = 150
    seed = 31415
    n_splits = 5
    num_workers = 4
    maxLastK = 6
    last_k = 6
    random.seed(seed)
    np.random.seed(seed)
    th.random.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    criterion = nn.BCEWithLogitsLoss(
        # pos_weight=th.FloatTensor([multiple_negative]).to(device)
    )
    sha = subprocess.check_output(
        ["git", "describe", "--always"]).strip().decode()
    now = datetime.datetime.now()
    timestamp_str = now.strftime(r'%Y%m%dT%H%M%S')
    config_name = 'combination1_option2'
    log_dir = f"logs/00/{sha}/att_{config_name}_last_{last_k}/{timestamp_str}"
    task_config = Task_Config(config_name)
    print(sha)
    for fold_idx in range(0, n_splits):
        ckpt_dir = f"{log_dir}/models/{fold_idx}"
        os.makedirs(ckpt_dir, exist_ok=True)
        datamodule = MIMIC3DataModule(
            task_config, fold_idx, multiple_negative,
            batch_size, num_workers, n_splits, last_k,
            use_less_wave_var=True)
        model = HCV_NUME_WAVE_Attentive_InceptionTime(
            ni_w=2,
            ni_n=len(task_config.nume_features),
            dropout_p=dropout_p)
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)

        with open(f'{log_dir}/mimic3.fold_{fold_idx}.log', 'w') as wfile:
            print('model:\n', model, file=wfile)
            print('criterion:', criterion, file=wfile)
            print('total_params', total_params, file=wfile)
            print('multiple_negative', multiple_negative, file=wfile)
            print('learning_rate', learning_rate, file=wfile)
            print('batch_size', batch_size, file=wfile)
            print('seed', seed, file=wfile)
            print('num_samples', datamodule.num_samples, file=wfile)
            print('num_total_test_samples',
                  datamodule.num_total_test_samples, file=wfile)
            print('num_mortality_samples',
                  datamodule.num_mortality_samples, file=wfile)
            print('num_test_mortality_samples',
                  datamodule.num_test_mortality_samples, file=wfile)

        metric_names = ['f1', 'recall', 'specificity',
                        'precision', 'accuracy', 'roc_auc',
                        'tn', 'fp', 'fn', 'tp']
        train_names = ['train_' + name for name in metric_names]
        test_names = ['test_' + name for name in metric_names]

        macro_wfile = open(f'{log_dir}/mimic3.fold_{fold_idx}_macro.csv', 'w')
        print('epoch idx', 'loss', *test_names,
              *train_names, sep=',', file=macro_wfile, flush=True)
        all_wfile = open(f'{log_dir}/mimic3.fold_{fold_idx}_all.csv', 'w')
        print('epoch idx', 'loss', *test_names,
              *train_names, sep=',', file=all_wfile, flush=True)
        avg_wfile = open(f'{log_dir}/mimic3.fold_{fold_idx}_avg.csv', 'w')
        print('epoch idx', 'loss', *test_names,
              *train_names, sep=',', file=avg_wfile, flush=True)
        lastK_all_wfiles = {
            lastK: open(f'{log_dir}/mimic3.fold_{fold_idx}_l{lastK}_all.csv',
                        'w')
            for lastK in range(1, maxLastK + 1, 1)
        }
        lastK_avg_wfiles = {
            lastK: open(f'{log_dir}/mimic3.fold_{fold_idx}_l{lastK}avg.csv',
                        'w')
            for lastK in range(1, maxLastK + 1, 1)
        }
        for lastK in range(1, maxLastK + 1, 1):
            print('epoch idx', 'loss', *test_names,
                  *train_names, sep=',',
                  file=lastK_all_wfiles[lastK], flush=True)
            print('epoch idx', 'loss', *test_names,
                  *train_names, sep=',',
                  file=lastK_avg_wfiles[lastK], flush=True)

        for epoch_idx in range(num_epoches):
            train_loader = datamodule.train_loader()
            loss_mean = train(train_loader, config_name, model, criterion,
                              optimizer, device, num_epoches, epoch_idx)
            # train set
            macro_metrics_dict, metrics_whole_all, metrics_whole_avg, \
                metrics_lastK_all_dict, metrics_lastK_avg_dict = \
                evaluation(train_loader, config_name, model,
                           criterion, metrics_fn, device,
                           num_epoches, epoch_idx, maxLastK)
            test_loader = datamodule.test_loader()
            train_macro_list = [macro_metrics_dict[name]
                                for name in metric_names]
            train_whole_all_list = [metrics_whole_all[name]
                                    for name in metric_names]
            train_whole_avg_list = [metrics_whole_avg[name]
                                    for name in metric_names]
            train_lastK_all_list = {
                lastK: [metrics_lastK_all_dict[lastK][name]
                        for name in metric_names]
                for lastK in range(1, maxLastK + 1, 1)}
            train_lastK_avg_list = {
                lastK: [metrics_lastK_avg_dict[lastK][name]
                        for name in metric_names]
                for lastK in range(1, maxLastK + 1, 1)}
            # test set
            macro_metrics_dict, metrics_whole_all, metrics_whole_avg, \
                metrics_lastK_all_dict, metrics_lastK_avg_dict = \
                evaluation(test_loader, config_name, model,
                           criterion, metrics_fn, device,
                           num_epoches, epoch_idx, maxLastK)
            test_macro_list = [macro_metrics_dict[name]
                               for name in metric_names]
            test_whole_all_list = [metrics_whole_all[name]
                                   for name in metric_names]
            test_whole_avg_list = [metrics_whole_avg[name]
                                   for name in metric_names]
            test_lastK_all_list = {
                lastK: [metrics_lastK_all_dict[lastK][name]
                        for name in metric_names]
                for lastK in range(1, maxLastK + 1, 1)}
            test_lastK_avg_list = {
                lastK: [metrics_lastK_avg_dict[lastK][name]
                        for name in metric_names]
                for lastK in range(1, maxLastK + 1, 1)}

            print(epoch_idx, loss_mean, *test_macro_list, *train_macro_list,
                  sep=',', file=macro_wfile, flush=True)
            print(epoch_idx, loss_mean, *test_whole_all_list,
                  *train_whole_all_list, sep=',',
                  file=all_wfile, flush=True)
            print(epoch_idx, loss_mean, *test_whole_avg_list,
                  *train_whole_avg_list, sep=',',
                  file=avg_wfile, flush=True)
            for lastK in range(1, maxLastK + 1, 1):
                print(epoch_idx, loss_mean, *test_lastK_all_list[lastK],
                      *train_lastK_all_list[lastK], sep=',',
                      file=lastK_all_wfiles[lastK], flush=True)
                print(epoch_idx, loss_mean, *test_lastK_avg_list[lastK],
                      *train_lastK_avg_list[lastK], sep=',',
                      file=lastK_avg_wfiles[lastK], flush=True)

            ckpt_path = f"{ckpt_dir}/{epoch_idx}.ckpt"
            th.save(model.state_dict(), ckpt_path)

        macro_wfile.close()
        all_wfile.close()
        avg_wfile.close()

        for lastK in range(1, maxLastK + 1, 1):
            lastK_all_wfiles[lastK].close()
            lastK_avg_wfiles[lastK].close()


if __name__ == "__main__":
    run()
