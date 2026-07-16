"""Custom implementation of the RWKV, from the RWKV paper published 2023
The CUDA kernel was implemented with the help of AI and from the official repository listed in the readme.
"""
import math

import torch
import torch.nn as nn

from mini_vlm.models import wkv_cuda_kernel

class RWKVTimeMix(nn.Module):

    def __init__(self, n_embd, layer_num, tot_layer):
        super().__init__()
        self.R = nn.Linear(n_embd, n_embd, bias=False)
        self.K = nn.Linear(n_embd, n_embd, bias=False)

        self.V = nn.Linear(n_embd, n_embd, bias=False)
        self.O = nn.Linear(n_embd, n_embd, bias=False)
        # zero initializatoin per paper -- K/R/O start as a no-op, V keeps
        # its default orthogonal init and learns normally from the start
        self.K.scale_init = 0
        self.R.scale_init = 0
        self.O.scale_init = 0

        with torch.no_grad(): # learned token shifting vectors initiliasations
            x = torch.arange(n_embd)/n_embd
            layer_ratio = 1 - layer_num/tot_layer
            ratio_0_to_1 = layer_num / max(tot_layer - 1, 1)
            self.mu_r = nn.Parameter(x.pow(layer_ratio).unsqueeze(0).unsqueeze(0))
            self.mu_k = nn.Parameter(0.5*x.pow(layer_ratio).unsqueeze(0).unsqueeze(0))
            self.mu_v = nn.Parameter((x.pow(layer_ratio)+0.3*ratio_0_to_1).unsqueeze(0).unsqueeze(0))

            l = torch.arange(n_embd)/max(n_embd-1, 1)
            self.w = nn.Parameter(-5 + 8*l.pow(0.7+1.3*ratio_0_to_1))

            self.u = nn.Parameter(0.5*torch.tensor([(i+1)%3 - 1 for i in range(n_embd)], dtype=torch.float32) + math.log(0.3))

        self.tokenShift = nn.ZeroPad2d((0,0,1,-1))

    def forward(self, x):
        B, T, C = x.shape
        x_shift = self.tokenShift(x)

        r = self.R((self.mu_r*x+(1-self.mu_r)*x_shift))
        k = self.K((self.mu_k*x+(1-self.mu_k)*x_shift))
        v = self.V((self.mu_v*x+(1-self.mu_v)*x_shift))

        if wkv_cuda_kernel.available(k):
            wkv = wkv_cuda_kernel.run_cuda_wkv(self.w, self.u, k, v) # custom cuda kernel adapted from github
        else:
            w = -torch.exp(self.w)  # per-channel decay, always <= 0
            u = self.u

            aa = torch.zeros(B, C, device=x.device, dtype=x.dtype)
            bb = torch.zeros(B, C, device=x.device, dtype=x.dtype)
            pp = torch.full((B, C), -1e30, device=x.device, dtype=x.dtype)
            wkv = torch.empty(B, T, C, device=x.device, dtype=x.dtype)

            for t in range(T):
                kt = k[:, t]
                vt = v[:, t]
                ww = u + kt
                q = torch.maximum(pp, ww)
                e1 = torch.exp(pp - q)
                e2 = torch.exp(ww - q)
                wkv[:, t] = (e1 * aa + e2 * vt) / (e1 * bb + e2 + 1e-8)

                ww2 = pp + w
                q2 = torch.maximum(ww2, kt)
                e1b = torch.exp(ww2 - q2)
                e2b = torch.exp(kt - q2)
                aa = e1b * aa + e2b * vt
                bb = e1b * bb + e2b
                pp = q2

        return self.O(torch.sigmoid(r)*wkv)




class RWKVChannelMix(nn.Module):

    def __init__(self, n_embd, layer_num, tot_layer, hidden_mult=4):
        super().__init__()
        n_hidden = n_embd*hidden_mult
        self.R  = nn.Linear(n_embd, n_embd, bias=False)
        self.K = nn.Linear(n_embd, n_hidden, bias=False)
        self.V = nn.Linear(n_hidden, n_embd, bias=False)

        # initializations -- V/R zero-init per paper, K keeps default init
        self.V.scale_init = 0
        self.R.scale_init = 0


        with torch.no_grad():
            x = torch.arange(n_embd)/n_embd
            layer_ratio = 1-layer_num/tot_layer
            self.mu_r = nn.Parameter(x.pow(layer_ratio).unsqueeze(0).unsqueeze(0))
            self.mu_k = nn.Parameter(x.pow(layer_ratio).unsqueeze(0).unsqueeze(0))

        self.tokenShift = nn.ZeroPad2d((0,0,1,-1))

    def forward(self, x):
        x_shift = self.tokenShift(x) - x

        r = self.R(x + x_shift*self.mu_r)
        k = self.K(x + x_shift*self.mu_k)

        return torch.sigmoid(r) * self.V(torch.square(torch.relu(k)))


class RWKVBlock(nn.Module):

    def __init__(self, n_embd, layer_num, tot_layer, dropout=0.1):
        super().__init__()
        self.timeMix = RWKVTimeMix(n_embd, layer_num, tot_layer)
        self.ln1 = nn.LayerNorm(n_embd)
        self.channelMix = RWKVChannelMix(n_embd, layer_num, tot_layer)
        self.ln2 = nn.LayerNorm(n_embd)
        # dropped on each sublayer's output before the residual add, same
        # placement nn.TransformerEncoderLayer uses -- the baseline gets this
        # for free from its PyTorch default (dropout=0.1); the RWKV stack had
        # no regularization at all until this, which was letting it overfit
        # harder than the baseline despite being the same rough size
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)


    def forward(self, x):
        x = x + self.drop1(self.timeMix(self.ln1(x))) # residual connections as per paper
        x = x + self.drop2(self.channelMix(self.ln2(x)))
        return x


class RWKVStack(nn.Module):

    def __init__(self, n_embd, n_layer, dropout=0.1):
        super().__init__()
        self.modelSeq = nn.Sequential(
            *[RWKVBlock(n_embd, i, n_layer, dropout=dropout) for i in range(n_layer)]
        )


    def forward(self, x):
        return self.modelSeq(x)
