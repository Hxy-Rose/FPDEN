import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn


def nets():
    return nn.Sequential(nn.Linear(3, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 3),
                         nn.Tanh())


def nett():
    return nn.Sequential(nn.Linear(3, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 3))


class RealNVP(nn.Module):
    def __init__(self):
        super(RealNVP, self).__init__()

        self.prior = distributions.MultivariateNormal(torch.zeros(3), torch.eye(3))
        mask = torch.from_numpy(np.array([[1, 1, 0],
                                          [0, 1, 1],
                                          [1, 0, 1]
                                          ] * 2).astype(np.float32))
        self.register_buffer('mask', mask)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])

    def cuda(self, device=None):
        self.prior.loc = self.prior.loc.cuda(device)
        self.prior.scale_tril = self.prior.scale_tril.cuda(device)
        self.prior._unbroadcasted_scale_tril = self.prior._unbroadcasted_scale_tril.cuda(device)
        self.prior.covariance_matrix = self.prior.covariance_matrix.cuda(device)
        self.prior.precision_matrix = self.prior.precision_matrix.cuda(device)
        super().cuda(device)
        return self

    def forward_p(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def backward_p(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x):
        z, logp = self.backward_p(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize, device=None):
        z = self.prior.sample((batchSize,))
        x = self.forward_p(z)
        return x

    def forward(self, x):
        return self.log_prob(x)
