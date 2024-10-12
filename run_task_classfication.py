import random
import torch as th
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from datamodule import MIMIC3Data, MIMIC3DataModule
from models import HCV_NUME_WAVE_InceptionTime
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
               metrics_fn: nn.Module, device, num_epoches, epoch_idx):

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
    metrics_dict = metrics_fn(y_socre_all, Y_all)
    icustay_summary = {icustay: (list(), list())
                       for icustay in set(icustay_list)}

    y_socre_all = y_socre_all.flatten().tolist()
    Y_all = Y_all.flatten().tolist()
    for idx in range(len(icustay_list)):
        icustay = icustay_list[idx]
        y_socre = y_socre_all[idx]
        y_label = Y_all[idx]
        icustay_summary[icustay][0].append(y_socre)
        icustay_summary[icustay][1].append(y_label)

    label_summary = []
    score_max_summary, score_avg_summary = [], []
    score_last, score_last_2_mean = [], []
    for scores, labels in icustay_summary.values():
        assert len(set(labels)) == 1
        label = labels[0]
        label_summary.append(label)
        score_max_summary.append(max(scores))
        score_avg_summary.append(sum(scores) / len(scores))
        score_last.append(scores[0])
        last_2 = scores[:2]
        score_last_2_mean.append(sum(last_2) / len(last_2))

    label_s = np.array(label_summary)
    score_max_s = np.array(score_max_summary)
    score_avg_s = np.array(score_avg_summary)
    score_last = np.array(score_last)
    score_last_2_mean = np.array(score_last_2_mean)

    max_metrics = metrics_fn(score_max_s, label_s)
    avg_metrics = metrics_fn(score_avg_s, label_s)
    last_metrics = metrics_fn(score_last, label_s)
    last2_metrics = metrics_fn(score_last_2_mean, label_s)
    return metrics_dict, max_metrics, avg_metrics, last_metrics, last2_metrics


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
    sample_last_k = 3
    learning_rate = 1e-3  # 1e-2, 1e-3, 1e-4
    batch_size = 32  # 16, 32, 64 # number of postive samples in a batch
    dropout_p = 0.5  # 0.5
    device = th.device('cuda:1')
    # device = torch.device('cpu')
    num_epoches = 500
    seed = 31415
    n_splits = 5
    num_workers = 8
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
    config_name = 'ii_abp_gap1_seg4_mort07_surv25_samp4'
    log_dir = f"logs/00/{sha}/att_{config_name}/{timestamp_str}"
    task_config = Task_Config(config_name)
    print(sha)
    for fold_idx in range(n_splits):
        ckpt_dir = f"{log_dir}/models/{fold_idx}"
        os.makedirs(ckpt_dir, exist_ok=True)
        datamodule = MIMIC3DataModule(
            task_config, fold_idx, multiple_negative,
            batch_size, num_workers, n_splits, sample_last_k)
        model = HCV_NUME_WAVE_InceptionTime1(
            ni_w=len(task_config.wave_features),
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

        max_wfile = open(f'{log_dir}/mimic3.fold_{fold_idx}_max.csv', 'w')
        avg_wfile = open(f'{log_dir}/mimic3.fold_{fold_idx}_avg.csv', 'w')
        last_wfile = open(f'{log_dir}/mimic3.fold_{fold_idx}_last.csv', 'w')
        last2_wfile = open(f'{log_dir}/mimic3.fold_{fold_idx}_last2.csv', 'w')
        with open(f'{log_dir}/mimic3.fold_{fold_idx}_sep.csv', 'w') as wfile:
            metric_names = ['f1', 'recall', 'specificity',
                            'precision', 'accuracy', 'roc_auc',
                            'tn', 'fp', 'fn', 'tp']
            train_names = ['train_' + name for name in metric_names]
            test_names = ['test_' + name for name in metric_names]
            print('epoch idx', 'loss', *test_names,
                  *train_names, sep=',', file=wfile)
            print('epoch idx', 'loss', *test_names,
                  *train_names, sep=',', file=max_wfile)
            print('epoch idx', 'loss', *test_names,
                  *train_names, sep=',', file=avg_wfile)
            print('epoch idx', 'loss', *test_names,
                  *train_names, sep=',', file=last_wfile)
            print('epoch idx', 'loss', *test_names,
                  *train_names, sep=',', file=last2_wfile)

            for epoch_idx in range(num_epoches):
                train_loader = datamodule.train_loader()
                loss_mean = train(train_loader, config_name, model, criterion,
                                  optimizer, device, num_epoches, epoch_idx)
                train_metrics, train_max_metrics, train_avg_metrics,\
                    trainlast_metrics, last2_metrics = \
                    evaluation(train_loader, config_name, model,
                               criterion, metrics_fn, device,
                               num_epoches, epoch_idx)
                test_loader = datamodule.test_loader()
                test_metrics, test_max_metrics, test_avg_metrics,\
                    test_last_metrics, test_last2_metrics = \
                    evaluation(test_loader, config_name, model,
                               criterion, metrics_fn, device,
                               num_epoches, epoch_idx)

                ckpt_path = f"{ckpt_dir}/{epoch_idx}.ckpt"
                th.save(model.state_dict(), ckpt_path)
                train_list = [train_metrics[name] for name in metric_names]
                test_list = [test_metrics[name] for name in metric_names]
                print(epoch_idx, loss_mean, *test_list, *train_list,
                      sep=',', file=wfile, flush=True)

                train_list = [train_max_metrics[name] for name in metric_names]
                test_list = [test_max_metrics[name] for name in metric_names]
                print(epoch_idx, loss_mean, *test_list, *train_list,
                      sep=',', file=max_wfile, flush=True)

                train_list = [train_avg_metrics[name] for name in metric_names]
                test_list = [test_avg_metrics[name] for name in metric_names]
                print(epoch_idx, loss_mean, *test_list, *train_list,
                      sep=',', file=avg_wfile, flush=True)

                train_list = [trainlast_metrics[name] for name in metric_names]
                test_list = [test_last_metrics[name] for name in metric_names]
                print(epoch_idx, loss_mean, *test_list, *train_list,
                      sep=',', file=last_wfile, flush=True)

                train_list = [last2_metrics[name] for name in metric_names]
                test_list = [test_last2_metrics[name] for name in metric_names]
                print(epoch_idx, loss_mean, *test_list, *train_list,
                      sep=',', file=last2_wfile, flush=True)

        max_wfile.close()
        avg_wfile.close()
        last_wfile.close()
        last2_wfile.close()


if __name__ == "__main__":
    run()
