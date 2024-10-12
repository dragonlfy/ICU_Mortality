import pickle as pkl
import torch as th
import numpy as np
import torch.nn.functional as F
from task_config import Task_Config
from models import HCV_NUME_WAVE_Attentive_InceptionTime
from datamodule import MIMIC3Data
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"


def get_example_params():
    sample_path = "data/tasks/w_ii_v_pleth_resp.n_abp_hr.gap2.seg4.mort14.surv14.samp4/"
    sample_path += "1_96879_288154_106_110.pkl"
    with open(sample_path, 'rb') as rb:
        datadict = pkl.load(rb)
    datadict['MORTALITY_INUNIT'] = 1
    sample = MIMIC3Data.new_obj_from_datadict(
        '1_96879_288154_106_110', sample_path, datadict)
    target = 0
    sample_id = sample_path.rsplit('/', 1)[1][:-4]

    config_name = 'combination1_option1'
    task_config = Task_Config(config_name)
    dropout_p = 0.5
    model = HCV_NUME_WAVE_Attentive_InceptionTime(
        ni_w=len(task_config.wave_features),
        ni_n=len(task_config.nume_features),
        dropout_p=dropout_p)

    print(len(task_config.wave_features))
    print(len(task_config.nume_features))

    ckpt_path = "logs/00/e890b8d/att_combination1_option1_last_6/20221109T223404/models/0/24.ckpt"
    model.load_state_dict(th.load(ckpt_path))
    model.eval()
    return sample, target, sample_id, model


class GradCam():
    """
        Produces class activation map
    """

    def __init__(self, model: HCV_NUME_WAVE_Attentive_InceptionTime):
        self.model = model
        self.model.eval()
        # Define extractor

    def generate_cam(self, sample: MIMIC3Data, target_class, sample_id):
        out, ret_dict = self.model.forward_cam(sample)
        # ret_dict['chan_att_1']        # B*NH,ni_w,1
        # ret_dict['time_att_1']        # B*NH,1,L2
        # ret_dict['wave_att']          # B*NH,ni_w,L2
        # ret_dict['waver_att']         # B*NH,ni_w,L2
        # ret_dict['chan_att_2']        # B,ni_2,1
        # ret_dict['time_att_2']        # B,1,NH
        # ret_dict['out2_att']          # B,ni_2,NH
        # ret_dict['chan_att_3']        # B,ni_3,1
        # ret_dict['output3_att']       # B,ni_3,1
        # self.waver_att_gradients      # B*NH,ni_w,L2
        # self.output2_att_gradients    # B,ni_2,NH
        # self.output3_att_gradients    # B,ni_3,1
        # assert B == 1

        # Target for backprop
        one_hot_output = th.FloatTensor(1, out.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.zero_grad()
        # Backward pass with specified target
        out.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients

        # figure = plt.figure(figsize=(8, 6))
        num_vars = 3
        fig, axs = plt.subplots(num_vars, 1, figsize=(6, 4),
                                gridspec_kw={'height_ratios': [1, 1, 3]})
        for ax in axs:
            ax.yaxis.set_label_text('#elements')

        wave = sample.wave
        B, NI_w, L_w = wave.shape
        wave_att_m = ret_dict['wave_att'].permute(1, 0, 2).reshape(B, NI_w, -1)
        wave_att = wave_att_m.detach().flatten().numpy()
        ax0 = axs[0]
        # ax0.yaxis.set
        ax0.set_title('Channel & Time Attentions on WAVE')
        ax0.title.set_size(12)
        # ax0.set_xlim(0., 1.)
        sns.distplot(wave_att,
                     bins=40, hist=True, kde=False, rug=False, ax=ax0,
                     color='#045275')

        # waver_att_gradients = self.model.waver_att_gradients
        # wave_tam = ret_dict['waver_att'] * waver_att_gradients
        # wave_tam = F.relu(wave_tam)
        # wave_tam = wave_tam.permute(1, 0, 2).reshape(B, NI_w, -1)
        # wave_tam_arr = wave_tam.detach().flatten().numpy()
        # ax1 = figure.add_subplot(num_vars, 1, 2)
        # ax1.set_title('wave_tam')
        # sns.distplot(wave_tam_arr,
        #              bins=20, hist=True, kde=False, rug=False, ax=ax1)

        nume = sample.nume
        B, NI_n, L_n = nume.shape
        out2_att_m = ret_dict['out2_att'].permute(
            1, 0, 2).reshape(B, NI_n+NI_w, -1)
        out2_att = out2_att_m.detach().flatten().numpy()
        ax1 = axs[1]
        ax1.set_title('Channel & Time Attentions on NUME')
        ax1.title.set_size(12)
        # ax1.set_xlim(0., 1.)
        sns.distplot(out2_att[4:],
                     bins=10, hist=True, kde=False, rug=False, ax=ax1, color='#045275')

        # output2_att_gradients = self.model.output2_att_gradients
        # out2_tam = ret_dict['out2_att'] * output2_att_gradients
        # out2_tam = F.relu(out2_tam)
        # out2_tam = out2_tam.permute(1, 0, 2).reshape(B, NI_n+NI_w, -1)
        # out2_tam_arr = out2_tam.detach().flatten().numpy()
        # ax1 = figure.add_subplot(num_vars, 1, 4)
        # ax1.set_title('out2_tam')
        # sns.distplot(out2_tam_arr,
        #              bins=20, hist=True, kde=False, rug=False, ax=ax1)

        hcv = sample.hcv
        B, NI_hcv, L_hcv = hcv.shape
        chan_att_3 = ret_dict['chan_att_3'].permute(
            1, 0, 2).reshape(B, NI_w+NI_n+NI_hcv, -1)
        chan_att_3 = F.relu(chan_att_3)
        chan_att_3 = chan_att_3.detach().flatten().numpy()
        # ax2 = axs[2]
        # ax2.set_title('Channel Attention on WAVE, NUME & HCV')
        # ax2.title.set_size(12)
        # ax2.set_xlim(0., 1.)
        # sns.distplot(chan_att_3,
        #              bins=10, hist=True, kde=False, rug=False, ax=ax2)

        output3_att_gradients = self.model.output3_att_gradients
        out3_tam = ret_dict['output3_att'] * output3_att_gradients
        out3_tam = F.relu(out3_tam)
        out3_tam = out3_tam.reshape(B, NI_w+NI_n+NI_hcv)
        out3_tam_arr = out3_tam.detach().flatten().numpy()
        ax3 = axs[2]
        # ax3 = axs[3]
        ax3.set_title('Grad-CAM on HCV')
        ax3.title.set_size(12)
        sns.distplot(out3_tam_arr[8:],
                     bins=10, hist=True, kde=False, rug=False, ax=ax3, color='#045275')

        plt.tight_layout(pad=0.8)
        plt.savefig(f"{sample_id}.2.pdf")
        plt.close()

        _, axs = plt.subplots(9, 1, figsize=(6, 7),
                              gridspec_kw={
            'height_ratios': [1, 1, 1, 1, 1, 1, 1, 1, 5]})
        wave_signals = wave[0].numpy()
        wave_x = np.arange(0, 4, 1 / 60 / 60 / 125)[:-1]
        wave_wx = np.arange(0, 4, 1 / 60)
        titles = ["II", "V", "PLETH", "RESP"]
        wave_att_m_w = wave_att_m[0].detach().numpy()
        for wave_idx in range(4):
            wave_sig = wave_signals[wave_idx, :]
            wave_w = wave_att_m_w[wave_idx, :]
            wave_w = (wave_w - np.min(wave_w)) / (np.max(wave_w) - np.min(wave_w))
            mean, std = np.mean(wave_sig), np.std(wave_sig)
            wave_sig = wave_sig - mean
            wave_sig = np.clip(wave_sig, -2, 2)
            y = wave_sig + wave_idx
            w = wave_w * 2 + 0.5
            axs[wave_idx].set_yticks([])
            axs[wave_idx].plot(wave_x, y)
            axs[wave_idx].fill_between(wave_wx, w, 0, color='red', alpha=0.2)
            axs[wave_idx].yaxis.set_label_text(titles[wave_idx])

        nume_x = np.arange(0, 4, 1 / 60)
        nume_wx = np.array([0, 1.5, 3, 4])
        nume_signals = nume[0].numpy()
        nume_offset = 0
        titles = ['HR', 'ABP\nmean', 'ABP\nsys', 'ABP\ndias']
        out2_att_m_w = out2_att_m[0].detach().numpy()
        for nume_idx in range(4):
            nume_sig = nume_signals[nume_idx, :]
            nume_w = out2_att_m_w[4+nume_idx, :]
            nume_w = (nume_w - np.min(nume_w)) / (np.max(nume_w) - np.min(nume_w))
            mean, std = np.mean(nume_sig), np.std(nume_sig)
            if std > 0:
                nume_sig = (nume_sig - mean) / std
            nume_sig = np.clip(nume_sig, -2, 2)
            y = nume_sig + nume_offset + nume_idx
            w = nume_w * 2 + 0.5
            axs[nume_idx+4].plot(nume_x, y)
            axs[nume_idx+4].fill_between(nume_wx, w, 0, color='red', alpha=0.2)
            axs[nume_idx+4].set_yticks([])
            axs[nume_idx+4].yaxis.set_label_text(titles[nume_idx])

        hcv_signals = hcv[0].numpy()
        hcv_x = np.arange(0.5, 4, 1)
        axs[-1].set_xticks([0, 1, 2, 3, 4])
        axs[-1].set_xlim(0, 4)
        axs[-1].yaxis.set_label_text("HCV")
        axs[-1].set_yticks([])
        out3_tam_w = out3_tam_arr[8:] + 0.5
        out3_tam_w = out3_tam_w / out3_tam_w.max()

        print('max out3_tam_w', np.max(out3_tam_w))
        print('min out3_tam_w', np.min(out3_tam_w))
        hcv_signals_sum = np.sum(hcv_signals, axis=1)
        hcv_signals_list = [hcv_signals[idx, :] for idx in range(hcv_signals.shape[0]) if hcv_signals_sum[idx] > 0]
        out3_tam_w_list = [out3_tam_w[idx] for idx in range(hcv_signals.shape[0]) if hcv_signals_sum[idx] > 0]
        for idx in range(len(hcv_signals_list)):
            hcv_sig = hcv_signals_list[idx]
            if (np.max(hcv_sig) - np.min(hcv_sig)) > 0:
                print('np.max(hcv_sig)', np.max(hcv_sig), np.mean(hcv_sig))
                hcv_sig = (hcv_sig - np.min(hcv_sig)) / (np.max(hcv_sig) - np.min(hcv_sig)) * 0.8 + 0.2
            else:
                hcv_sig = np.ones_like(hcv_sig)

            # axs[-1].scatter(hcv_x, idx + hcv_sig, alpha=out3_tam_w_list[idx])
            w = out3_tam_w_list[idx]*hcv_sig
            axs[-1].scatter(hcv_x, idx * np.ones_like(hcv_x), alpha=w)

        plt.xlabel("Time/hour")
        plt.tight_layout(pad=1)
        plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0)
        plt.savefig(f"inter_{sample_id}.2.pdf")
        # wave # 1, 4, 1800000
        # wave_att_m # 4, 240
        # out2_att_m # 8, 4
        # nume # 1, 4, 240
        # chan_att_3 #


if __name__ == '__main__':
    # sample_path = 'data/tasks/w_ii.n_abp.gap1.seg4.mort7.surv25.samp4/1_86158_223563_58_62.pkl'
    sample, target, sample_id, pretrained_model = get_example_params()
    gcv2 = GradCam(pretrained_model)
    cam = gcv2.generate_cam(sample, target, sample_id)
