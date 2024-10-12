from typing import Dict, Tuple
import torch as th
from torch import nn, Tensor
from datamodule import MIMIC3Data
import torch.nn.functional as F


class HCV_NUME_WAVE_InceptionTime(nn.Module):
    def __init__(self, ni_w=1, nf_1=16, ni_n=3, nf_2=32, ni_hcv=43,
                 num_hours=4, nf_3=376, pred_nf=376, num_classes=1,
                 dropout_p=0.5, jump_knowledge=True):
        super().__init__()
        self.ni_w = ni_w
        self.ni_n = ni_n
        self.ni_hcv = ni_hcv
        self.jump_knowledge = jump_knowledge
        ni_2 = ni_w + ni_n
        ni_3 = ni_2 + ni_hcv
        print(ni_w, nf_1,
              f"\n{ni_n}", ni_2, nf_2,
              f"\n{ni_hcv}", ni_3, nf_3)
        assert nf_1 % ni_w == 0
        assert nf_2 % ni_2 == 0
        assert nf_3 % ni_3 == 0
        self.num_hours = num_hours
        self._extra_repre = f"ni_w:{ni_w}, nf_1: {nf_1}, ni_n: {ni_n}, " \
            + f"nf_2: {nf_2}, ni_hcv: {ni_hcv}, nf_3: {nf_3}," \
            + f" pred_nf: {pred_nf}, num_classes: {num_classes}," \
            + f" dropout_p: {dropout_p}"

        num_layers_1 = 4
        k_list_1 = [5, 9, 17, 33]
        stride_1 = 4
        self.module_1 = Nested_InceptionModule(
            num_layers_1, ni_w, nf_1,
            k_list_1, stride_1, dropout_p)
        self.reduce_linear_1 = nn.Conv1d(self.module_1.outc, ni_w, 1,
                                         groups=ni_w)

        num_layers_2 = 2
        k_list_2 = [3, 5, 9, ]
        stride_2 = 4
        self.module_2 = Nested_InceptionModule(
            num_layers_2, ni_2, nf_2,
            k_list_2, stride_2, dropout_p)
        ni_2 = ni_w + ni_n
        self.reduce_linear_2 = nn.Conv1d(self.module_2.outc, ni_2, 1,
                                         groups=ni_2)

        self.module_3 = nn.Sequential(
            nn.Conv1d(ni_3, nf_3, num_hours, groups=ni_3),
            nn.ReLU(),
            nn.Conv1d(nf_3, ni_3, 1, groups=ni_3),
            nn.BatchNorm1d(ni_3),
            nn.ReLU(),
        )

        if jump_knowledge:
            last_in = (ni_w + ni_2 + ni_3)
        else:
            last_in = ni_3

        self.classfier = nn.Sequential(
            nn.Linear(last_in, last_in),
            nn.BatchNorm1d(last_in),
            nn.ReLU(),
            nn.Linear(last_in, num_classes),
        )

    def forward(self, batch: MIMIC3Data):
        wave = batch.wave
        B = wave.shape[0]
        H1, L1 = self.num_hours * 60, 60 * 125
        wavei = wave.view((-1, self.ni_w, H1, L1))
        wavei = wavei.permute((0, 2, 1, 3)).view((-1, self.ni_w, L1))
        wavef = self.module_1(wavei)
        waver: Tensor = self.reduce_linear_1(wavef)  # -1, ni_w, L1s
        waver: Tensor = F.relu(waver)  # -1, nf_w, L1s
        waver_s = waver.mean(2)
        input2p = waver_s.view((B, H1, self.ni_w))
        input2r = input2p.permute((0, 2, 1))  # B, self.ni_w, H1
        H2, L2 = self.num_hours, 60
        nume = batch.nume
        input2 = th.concat((input2r, nume), 1)
        n_input2 = self.ni_w + self.ni_n
        input2 = input2.view((-1, n_input2, H2, L2)).permute((0, 2, 1, 3))
        input2 = input2.reshape((-1, n_input2, L2))

        output2: Tensor = self.module_2(input2)
        output2r: Tensor = self.reduce_linear_2(output2)
        output2r = F.relu(output2r)
        output2r_s = output2r.mean(2)
        input3p = output2r_s.view((B, H2, n_input2))
        input3r = input3p.permute((0, 2, 1))

        hcv = batch.hcv
        # n_input3 = n_input2 + self.ni_hcv
        input3 = th.concat((input3r, hcv), 1)
        output3 = self.module_3(input3).squeeze(2)

        if self.jump_knowledge:
            jp_1 = input2r.mean(dim=2)
            jp_2 = input3r.mean(dim=2)
            jp_concat = th.concat((jp_1, jp_2, output3), dim=1)
            out = self.classfier(jp_concat)
        else:
            out = self.classfier(output3)
        return out

    def save_waver_gradients(self, grad):
        self.waver_gradients = grad

    def save_output2r_gradients(self, grad):
        self.output2r_gradients = grad

    def save_output3_gradients(self, grad):
        self.output3_gradients = grad

    def forward_cam(self, batch: MIMIC3Data) \
            -> Tuple[Tensor, Dict[str, Tensor]]:
        ret_dict = {}
        wave = batch.wave
        B = wave.shape[0]
        H1, L1 = self.num_hours * 60, 60 * 125
        wavei = wave.view((-1, self.ni_w, H1, L1))
        wavei = wavei.permute((0, 2, 1, 3)).view((-1, self.ni_w, L1))
        wavef = self.module_1(wavei)
        waver: Tensor = self.reduce_linear_1(wavef)  # -1, ni_w, L1s
        waver: Tensor = F.relu(waver)  # -1, nf_w, L1s
        ret_dict['waver'] = waver
        waver.register_hook(self.save_waver_gradients)

        waver_s = waver.mean(2)
        input2p = waver_s.view((B, H1, self.ni_w))
        input2r = input2p.permute((0, 2, 1))  # B, self.ni_w, H1
        H2, L2 = self.num_hours, 60
        nume = batch.nume
        input2 = th.concat((input2r, nume), 1)
        n_input2 = self.ni_w + self.ni_n
        input2 = input2.view((-1, n_input2, H2, L2)).permute((0, 2, 1, 3))
        input2 = input2.reshape((-1, n_input2, L2))

        output2: Tensor = self.module_2(input2)
        output2r: Tensor = self.reduce_linear_2(output2)
        output2r = F.relu(output2r)
        ret_dict['output2r'] = output2r
        output2r.register_hook(self.save_output2r_gradients)

        output2r_s = output2r.mean(2)
        input3p = output2r_s.view((B, H2, n_input2))
        input3r = input3p.permute((0, 2, 1))

        hcv = batch.hcv
        input3 = th.concat((input3r, hcv), 1)
        output3: Tensor = self.module_3(input3).squeeze(2)
        ret_dict['output3'] = output3
        output3.register_hook(self.save_output3_gradients)

        if self.jump_knowledge:
            jp_1 = input2r.mean(dim=2)
            jp_2 = input3r.mean(dim=2)
            jp_concat = th.concat((jp_1, jp_2, output3), dim=1)
            out = self.classfier(jp_concat)
        else:
            out = self.classfier(output3)
        return out, ret_dict


class Nested_InceptionModule(nn.Module):
    def __init__(self, num_layers, ni, nf, k_list: list,
                 stride=3, dropout_p=0.0):
        super(Nested_InceptionModule, self).__init__()
        self.num_layers = num_layers
        self.outc = len(k_list) * nf
        module_list = [InceptionModule(ni if idx == 0 else self.outc, nf,
                                       ni, k_list, stride, dropout_p)
                       for idx in range(num_layers)]
        self.module_list = nn.ModuleList(module_list)

    def forward(self, x: Tensor):
        h = x
        for module in self.module_list:
            h = module(h)

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
        f_list = [conv(f).view(B, self.ng, self.nf // self.ng, -1)
                  for conv in self.convs]
        h0 = th.concat(f_list, dim=2).view(B, self.outc, -1)
        h1 = self.bn(h0)
        h2 = self.relu(h1)
        y: Tensor = self.dropout(h2)
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


if __name__ == '__main__':
    batch = MIMIC3Data()
    batch.wave = th.FloatTensor(size=(3, 1, 4 * 3600 * 125))
    batch.nume = th.FloatTensor(size=(3, 2, 4 * 60))
    batch.hcv = th.FloatTensor(size=(3, 43, 4))
    batch.demography = th.FloatTensor(size=(3, 2))
    model = HCV_NUME_WAVE_InceptionTime()
    x = model(batch)
    print(x.shape)
    print()
