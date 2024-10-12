from typing import Dict, Tuple

import torch as th
from torch import Tensor, nn

from datamodule import MIMIC3Data


class HCV_NUME_WAVE_Attentive_InceptionTime(nn.Module):
    def __init__(self, ni_w=4, nf_1=None, ni_n=4, nf_2=None, ni_hcv=37,
                 num_hours=4, nf_3=None, pred_nf=None, num_classes=1,
                 dropout_p=0.5):
        # ni_w: the number of types of waveform variables
        # ni_n: the number of types of numerics variables
        # ni_hcv: the number of types of HCV variables

        super().__init__()
        if nf_1 is None:
            nf_1 = ni_w * 4
        if nf_2 is None:
            nf_2 = (ni_w + ni_n) * 4
        if nf_3 is None:
            nf_3 = (ni_w + ni_n + ni_hcv) * 8
        if pred_nf is None:
            assert nf_3 is not None
            pred_nf = nf_3

        self.ni_w = ni_w
        self.ni_n = ni_n
        self.ni_hcv = ni_hcv

        ni_2 = ni_w + ni_n
        ni_3 = ni_2 + ni_hcv
        print(ni_w, nf_1,
              f"\n{ni_n}", ni_2, nf_2,
              f"\n{ni_hcv}", ni_3, nf_3)
        assert nf_1 % ni_w == 0, (nf_1, ni_w)
        assert nf_2 % ni_2 == 0, (nf_2, ni_2)
        assert nf_3 % ni_3 == 0, (nf_3, ni_3)
        self.num_hours = num_hours
        self._extra_repr = f"ni_w:{ni_w}, nf_1: {nf_1}, ni_n: {ni_n}, " \
            + f"nf_2: {nf_2}, ni_hcv: {ni_hcv}, nf_3: {nf_3}," \
            + f" pred_nf: {pred_nf}, num_classes: {num_classes}," \
            + f" dropout_p: {dropout_p}"

        num_layers_1 = 4
        k_list_1 = [5, 9, 17, 33]
        stride_1 = 4
        self.net_1 = InceptionNetwork(num_layers_1, ni_w, nf_1,
                                      k_list_1, stride_1, dropout_p)
        # self.net_1 = InceptionNetwork1(num_layers_1, ni_w, nf_1,
        #                               stride_1, dropout_p)
        self.linear_reducer_1 = nn.Sequential(
            nn.Conv1d(self.net_1.outc, ni_w, 1, groups=ni_w, bias=False),
            nn.ReLU())
        self.chan_att_1 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(self.net_1.outc, ni_w, 1, groups=ni_w, bias=False),
            nn.Sigmoid())
        self.time_att_1 = nn.Sequential(
            nn.Conv1d(self.net_1.outc, 1, 1, bias=False),
            nn.Sigmoid())

        num_layers_2 = 2
        k_list_2 = [3, 5, 9]
        stride_2 = 4
        self.net_2 = InceptionNetwork(
            num_layers_2, ni_2, nf_2,
            k_list_2, stride_2, dropout_p)
        # self.net_2 = InceptionNetwork1(
        #     num_layers_2, ni_2, nf_2,
        #     stride_2, dropout_p)
        self.linear_reducer_2 = nn.Sequential(
            nn.Conv1d(self.net_2.outc, ni_2, 1, groups=ni_2, bias=False),
            nn.ReLU())
        self.chan_att_2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(self.net_2.outc, ni_2, 1, groups=ni_2, bias=False),
            nn.Sigmoid())
        self.time_att_2 = nn.Conv1d(self.net_2.outc, 1, 1, bias=False)

        self.multi_conv1d = nn.Sequential(
            nn.Conv1d(ni_3, nf_3, num_hours, groups=ni_3),
            nn.ReLU(),
            nn.Conv1d(nf_3, ni_3, 1, groups=ni_3),
            nn.BatchNorm1d(ni_3),
            nn.ReLU(),
        )
        self.chan_att_3 = nn.Sequential(
            nn.Conv1d(ni_3, nf_3, num_hours, groups=ni_3),
            nn.ReLU(),
            nn.Conv1d(nf_3, ni_3, 1, groups=ni_3),
            nn.Sigmoid())

        self.classfier = nn.Sequential(
            nn.Linear(ni_3, ni_3),
            nn.BatchNorm1d(ni_3),
            nn.ReLU(),
            nn.Linear(ni_3, num_classes),
        )

        self.bilstm = nn.LSTM(
            input_size=ni_3,
            hidden_size=ni_3,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

    def extra_repr(self) -> str:
        return self._extra_repr

    def forward(self, batch: MIMIC3Data):
        wave = batch.wave
        # wave = wave[:, :2, :]
        B = wave.shape[0]
        NH = self.num_hours
        L1, L2 = 60 * 125, 60
        # wave = th.index_select(wave, 1, th.IntTensor(data=(0,1,3)).to(wave.device))
        wavei = wave.reshape((B, self.ni_w, NH * L2, L1))
        wavei = wavei.permute((0, 2, 1, 3)).reshape((-1, self.ni_w, L1))
        # wavei = wave.reshape((B, 2, NH * L2, L1))
        # wavei = wavei.permute((0, 2, 1, 3)).reshape((-1, 2, L1))
        # reshape to B*NH*L2, ni_w, L1

        wavef = self.net_1(wavei)  # shape: B*NH*L2, self.net_1.outc, 1
        waves = wavef.reshape((B * NH, L2, self.net_1.outc))
        waves: Tensor = waves.permute((0, 2, 1))  # B*NH,net_1.outc,L2

        waver: Tensor = self.linear_reducer_1(waves)  # B*NH,ni_w,L2
        chan_att_1 = self.chan_att_1(waves)  # B*NH,ni_w,1
        time_att_1 = self.time_att_1(waves)  # B*NH,1,L2
        waver_att: Tensor = waver * chan_att_1 * time_att_1
        # B*NH,ni_w,L2

        nume = batch.nume  # B,ni_n,NH*L2
        # nume = th.index_select(nume, 1, th.IntTensor(data=(0,)).to(nume.device))
        numei = nume.reshape((B, -1, NH, L2))
        numei = numei.permute((0, 2, 1, 3)).reshape((B*NH, self.ni_n, L2))

        # ni_2 = self.ni_n + self.ni_w
        input2 = th.concat((waver_att, numei), 1)  # B*NH,ni_2,L2
        output2: Tensor = self.net_2(input2)  # B*NH,net_2.outc,1
        outs = output2.reshape((B, NH, self.net_2.outc))
        outs: Tensor = outs.permute((0, 2, 1))  # B,net_2.outc,NH

        outr = self.linear_reducer_2(outs)  # B,ni_2,NH
        chan_att_2 = self.chan_att_2(outs)  # B,ni_2,1
        time_att_2 = th.sigmoid(self.time_att_2(outs))  # B,1,NH
        output2_att: Tensor = outr * chan_att_2 * time_att_2

        # ni_3 = self.ni_hcv + self.ni_n + self.ni_w
        hcv = batch.hcv  # B,ni_hcv,NH
        input3 = th.concat((output2_att, hcv), 1)
        out3 = self.multi_conv1d(input3)  # B,ni_3,1
        chan_att_3 = self.chan_att_3(input3)
        output3_att = (out3 * chan_att_3).squeeze(2)

        out3, _ = self.bilstm(output3_att)
        out3_1 = th.mean(out3.view(-1, 45, 2), dim=2)
        out = self.classfier(out3_1)

        # out = self.classfier(output3_att)
        return out

    def save_waver_att_gradients(self, grad):
        self.waver_att_gradients = grad

    def save_out2_att_gradients(self, grad):
        self.output2_att_gradients = grad

    def save_out3_att_gradients(self, grad):
        self.output3_att_gradients = grad

    def forward_cam(self, batch: MIMIC3Data) \
            -> Tuple[Tensor, Dict[str, Tensor]]:
        ret_dict = {}

        wave = batch.wave
        wave = wave[:, :2, :]
        B = wave.shape[0]
        assert B == 1
        NH = self.num_hours
        L1, L2 = 60 * 125, 60
        wavei = wave.reshape((B, self.ni_w, NH * L2, L1))
        wavei = wavei.permute((0, 2, 1, 3)).reshape((-1, self.ni_w, L1))
        # reshape to B*NH*L2, ni_w, L1

        wavef = self.net_1(wavei)  # shape: B*NH*L2, self.net_1.outc, 1
        waves = wavef.reshape((B * NH, L2, self.net_1.outc))
        waves: Tensor = waves.permute((0, 2, 1))  # B*NH,net_1.outc,L2

        waver: Tensor = self.linear_reducer_1(waves)  # B*NH,ni_w,L2
        chan_att_1 = self.chan_att_1(waves)  # B*NH,ni_w,1
        time_att_1 = self.time_att_1(waves)  # B*NH,1,L2
        wave_att = chan_att_1 * time_att_1
        waver_att: Tensor = waver * wave_att  # B*NH,ni_w,L2
        # B*NH,ni_w,L2

        ret_dict['chan_att_1'] = chan_att_1
        ret_dict['time_att_1'] = time_att_1
        ret_dict['wave_att'] = wave_att
        ret_dict['waver_att'] = waver_att
        waver_att.register_hook(self.save_waver_att_gradients)

        nume = batch.nume  # B,ni_n,NH*L2
        numei = nume.reshape((B, -1, NH, L2))
        numei = numei.permute((0, 2, 1, 3)).reshape((B*NH, self.ni_n, L2))

        # ni_2 = self.ni_n + self.ni_w
        input2 = th.concat((waver_att, numei), 1)  # B*NH,ni_2,L2
        output2: Tensor = self.net_2(input2)  # B*NH,net_2.outc,1
        outs = output2.reshape((B, NH, self.net_2.outc))
        outs: Tensor = outs.permute((0, 2, 1))  # B,net_2.outc,NH

        outr = self.linear_reducer_2(outs)  # B,ni_2,NH
        chan_att_2 = self.chan_att_2(outs)  # B,ni_2,1
        time_att_2 = th.sigmoid(self.time_att_2(outs))  # B,1,NH
        out2_att = chan_att_2 * time_att_2
        output2_att: Tensor = outr * out2_att  # B,ni_2,NH

        ret_dict['chan_att_2'] = chan_att_2
        ret_dict['time_att_2'] = time_att_2
        ret_dict['out2_att'] = out2_att
        output2_att.register_hook(self.save_out2_att_gradients)

        # ni_3 = self.ni_hcv + self.ni_n + self.ni_w
        hcv = batch.hcv  # B,ni_hcv,NH
        input3 = th.concat((output2_att, hcv), 1)
        out3 = self.multi_conv1d(input3)  # B,ni_3,1
        chan_att_3 = self.chan_att_3(input3)
        output3_att = (out3 * chan_att_3).squeeze(2)
        output3_att.register_hook(self.save_out3_att_gradients)

        ret_dict['chan_att_3'] = chan_att_3
        ret_dict['output3_att'] = output3_att

        out = self.classfier(output3_att)
        return out, ret_dict


class InceptionNetwork(nn.Module):
    def __init__(self, num_layers, ni, nf, k_list: list,
                 stride=3, dropout_p=0.0):
        super(InceptionNetwork, self).__init__()
        self.num_layers = num_layers
        self.outc = len(k_list) * nf
        module_list = [InceptionModule(ni if idx == 0 else self.outc, nf,
                                       ni, k_list, stride, dropout_p)
                       for idx in range(num_layers)]
        self.module_list = nn.ModuleList(module_list)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: Tensor):
        h = x
        for module in self.module_list:
            h = module(h)
        h = self.pool(h)
        return h


class InceptionModule(nn.Module):
    def __init__(self, ni, nf, ng, k_list: list, stride, dropout_p):
        super(InceptionModule, self).__init__()
        # print(f'ng: {ng}, ni: {ni}, nf: {nf}')
        assert nf % ng == 0
        self.ng = ng
        self.nf = nf
        k_list = [int(k // 2 * 2 + 1) for k in k_list]
        # ensure odd k_list
        if stride > 1:
            stride = stride // 2 * 2  # ensure even stride
        # self.bottleneck = nn.Conv1d(ni, nf, 1, bias=False, groups=ng)
        self.bottleneck = nn.Conv1d(ni, nf, 1, bias=False, groups=ng)

        self.convs = nn.ModuleList([
            nn.Conv1d(nf, nf, k, stride, padding=k//2, groups=ng)
            for k in k_list])
        self.outc = nf * len(k_list)
        self.bn = nn.BatchNorm1d(self.outc)
        self.relu = nn.ReLU()
        self.dropout = DropoutTimeSeriesChannels(dropout_p)

    def forward(self, x: Tensor):
        B = x.shape[0]
        f = self.bottleneck(x)
        f_list = [conv(f).reshape(B, self.ng, self.nf // self.ng, -1)
                  for conv in self.convs]
        h0 = th.concat(f_list, dim=2).reshape(B, self.outc, -1)
        print(h0.shape)
        h1 = self.bn(h0)
        h2 = self.relu(h1)
        y: Tensor = self.dropout(h2)
        print(y.shape)
        # B, self.ni * self.nf // self.ni, L
        return y


class DropoutTimeSeriesChannels(nn.Module):
    # Randomly zero out entire channels (a channel is a 1D feature map,
    # e.g., the j-th channel of the i-th sample
    #   in the batched input is a 1D tensor \text{input}[i, j]).
    # Each channel will be zeroed out independently on every forward call
    #   with probability p using samples from a Bernoulli distribution.

    def __init__(self, p=0.5) -> None:
        super().__init__()
        self._p = p
        self.dropout2d = nn.Dropout2d(p)

    def forward(self, inputs: Tensor):
        inputs = inputs.unsqueeze(2)
        outputs: Tensor = self.dropout2d(inputs)
        outputs = outputs.squeeze(2)
        return outputs

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self._p})'


class InceptionNetwork1(nn.Module):
    def __init__(self, num_layers, ni, nf, stride=3, dropout_p=0.0):
        super(InceptionNetwork1, self).__init__()
        self.num_layers = num_layers
        self.outc = 4 * nf
        module_list = [
            InceptionModule1(ni if idx == 0 else self.outc, nf, ni, stride, dropout_p)
            for idx in range(num_layers)
        ]
        self.module_list = nn.ModuleList(module_list)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: Tensor):
        h = x
        for module in self.module_list:
            h = module(h)
        h = self.pool(h)
        return h


class InceptionModule1(nn.Module):
    def __init__(self, ni, nf, ng, stride, dropout_p):
        super(InceptionModule1, self).__init__()
        # print(f'ng: {ng}, ni: {ni}, nf: {nf}')
        assert nf % ng == 0
        self.ng = ng
        self.nf = nf
        # ensure odd k_list
        if stride > 1:
            stride = stride // 2 * 2  # ensure even stride
        self.bottleneck = nn.Conv1d(ni, nf, 1, bias=False, groups=ng)

        self.convs = nn.Conv1d(nf, nf * 4, 3, stride, padding=3 // 2, groups=ng)
        self.outc = nf * 4
        self.bn = nn.BatchNorm1d(self.outc)
        self.relu = nn.ReLU()
        self.dropout = DropoutTimeSeriesChannels(dropout_p)

    def forward(self, x: Tensor):
        B = x.shape[0]
        f = self.bottleneck(x)
        f: Tensor = self.convs(f).reshape(B, self.ng, self.nf // self.ng * 4, -1)
        h0 = f.reshape(B, self.outc, -1)
        print(h0.shape)
        h1 = self.bn(h0)
        h2 = self.relu(h1)
        y: Tensor = self.dropout(h2)
        print(y.shape)
        # B, self.ni * self.nf // self.ni, L
        return y


if __name__ == '__main__':
    batch = MIMIC3Data()
    batch.wave = th.FloatTensor(size=(3, 1, 4 * 3600 * 125))
    batch.nume = th.FloatTensor(size=(3, 2, 4 * 60))
    batch.hcv = th.FloatTensor(size=(3, 43, 4))
    batch.demography = th.FloatTensor(size=(3, 2))
    model = HCV_NUME_WAVE_Attentive_InceptionTime()
    x = model(batch)
    print(x.shape)
    print()


