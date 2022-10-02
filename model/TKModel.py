import numpy as np
from scipy.optimize import least_squares as ls
from scipy.optimize import lsq_linear

from utils.config import get_config

config = get_config()
r1 = config.protocol.r1
TR = config.protocol.TR
alpha = config.protocol.alpha / 180 * np.pi
deltt = config.protocol.deltt / 60


def eTofts_model(ktrans, vp, ve, T10, cp):
    ce = np.zeros_like(cp, dtype=np.float)
    cp_length = np.size(cp)
    R10 = 1 / T10
    for t in range(cp_length):
        ce[t] = np.sum(cp[:t + 1] * np.exp(ktrans / ve * (np.arange(t + 1) - t) * deltt)) * deltt
    # print('fit',ce)
    ce = ce * ktrans
    ct = vp * cp + ce
    R1 = R10 + r1 * ct
    s = (1 - np.exp(-TR * R1)) * np.sin(alpha) / (1 - np.exp(-TR * R1) * np.cos(alpha))
    return s


def s(T10):
    R1 = 1 / T10
    s = (1 - np.exp(-TR * R1)) * np.sin(alpha) / (1 - np.exp(-TR * R1) * np.cos(alpha))
    return s


def target_func(x, T10, cp, signal):
    ktrans, vp, ve = x
    s = eTofts_model(ktrans, vp, ve, T10, cp)
    s0 = np.mean(signal[:6]) / np.mean(s[:6])
    return s * s0 - signal


def fit_eTofts(T10, cp, signal):
    k_trans0 = 0.01
    vb0 = 0.02
    vo0 = 0.2
    x0 = (k_trans0, vb0, vo0)
    bounds = [(1e-5, 0.0005, 0.04), (1, 0.1, 0.6)]
    result = ls(target_func, x0, bounds=bounds, args=(T10, cp, signal))
    return result.x, result.fun + signal


def full_eTofts(ktrans, vp, ve, T10, cp, signal):
    s = eTofts_model(ktrans, vp, ve, T10, cp)
    s0 = np.mean(signal[:6]) / np.mean(s[:6])
    return s * s0


def NLSQ(T10, cp, signal):
    R10 = 1 / T10
    s0 = (1 - np.exp(-TR * R10)) * np.sin(alpha) / (1 - np.exp(-TR * R10) * np.cos(alpha))
    M = np.mean(signal[:6]) / s0
    R1t = -np.log((signal - M * np.sin(alpha)) / (signal * np.cos(alpha) - M * np.sin(alpha))) / TR
    ctis = (R1t - R10) / r1
    cp_intergral = np.zeros_like(cp, dtype=np.float)
    ctis_intergral = np.zeros_like(cp, dtype=np.float)
    cp_length = np.size(cp)
    for t in range(cp_length):
        cp_intergral[t] = np.sum(cp[:t + 1]) * deltt
    for t in range(cp_length):
        ctis_intergral[t] = np.sum(ctis[:t + 1]) * deltt
    matrixA = np.concatenate((cp_intergral.reshape((cp_length, 1)), -ctis_intergral.reshape((cp_length, 1)),
                              cp.reshape((cp_length, 1))), axis=1)
    matrixC = ctis
    bounds = [(-np.inf, -np.inf, 0.0005), (np.inf, np.inf, 0.1)]
    matrixB = lsq_linear(matrixA, matrixC, bounds=bounds).x
    # matrixB = lsq_linear(matrixA, matrixC).x
    vp = matrixB[2]
    k2 = matrixB[1]
    ktrans = matrixB[0] - k2 * vp
    ve = ktrans / k2
    return np.array([ktrans, vp, ve])


import torch
from torch import nn


class eTofts_torch(nn.Module):
    def __init__(self, ):
        super(eTofts_torch, self).__init__()
        self.deltt = deltt
        self.range = torch.arange(0, 112, dtype=torch.float, requires_grad=False)
        self.kt_sin = torch.sin(torch.tensor([alpha, ]))
        self.kt_cos = torch.cos(torch.tensor([alpha, ]))

    def forward(self, ktrans, vp, ve, T10, CAp):
        ce = self.CAe(CAp, ktrans, ve)
        # plt.plot(ce[0, ...].cpu().numpy(), '--', label='torch ce')
        # plt.plot(signal_, '*', label='eTofts')
        ct = vp * CAp + ce
        R1 = 1 / T10 + r1 * ct
        s = (1 - torch.exp(-TR * R1)) * self.kt_sin / (1 - torch.exp(-TR * R1) * self.kt_cos)
        return s * 20

    def cuda(self):
        # self.range = self.range.cuda()
        self.kt_sin = self.kt_sin.cuda()
        self.kt_cos = self.kt_cos.cuda()
        return self

    def CAe(self, CAp, ktrans, vo):
        '''
        :param CAp: [batch, N]
        :param ktrans: [batch, 1]
        :param vo: [batch, 1]
        :return:
        '''
        batch, N = CAp.size()
        range_matrix = torch.exp(-ktrans / vo * torch.arange(start=N - 1, end=-1, step=-1,
                                                             device=CAp.device).float() * self.deltt)  # N x 1 [batch, N]
        ce_x = torch.zeros((batch, N, N), device=range_matrix.device, dtype=CAp.dtype)
        for i in range(N):
            ce_x[:, :i + 1, i] = range_matrix[:, -i - 1:]
        return torch.mul(ktrans * self.deltt, torch.bmm(CAp.unsqueeze(1), ce_x).squeeze(1))
