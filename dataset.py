import os

import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
import torch
from model.TKModel import eTofts_model, fit_eTofts
from utils.config import get_config

config = get_config()
r1 = config.protocol.r1
TR = config.protocol.TR
alpha = config.protocol.alpha / 180 * np.pi
deltt = config.protocol.deltt / 60


class SimulatePatient(Dataset):
    def __init__(self, root_list, length=10000):
        self.root_list = root_list
        self.cps = self.read_cps()
        self.len = length

    def __getitem__(self, item):
        T10, ct, noise_signal, cp, param = self.signal_fited_raw(item)
        return T10, torch.tensor(ct).unsqueeze(0), torch.tensor(noise_signal).unsqueeze(0), \
               torch.tensor(cp).unsqueeze(0), param

    def generate_signal(self, item):
        np.random.seed()
        T10 = self.generate_t1()
        ktrans = self.generate_ktrans()
        vp = self.generate_vp()
        ve = self.generate_ve()
        cp = self.generate_cp()
        # s0 = 20
        # exit(0)
        signal = eTofts_model(ktrans, vp, ve, T10, cp)  # * s0
        level = np.random.random() * 0.1  # np.random.random()*0.02+0.01
        # 0.0 44.83
        # 0.01 36.98
        # 0.03 29.8
        # 0.05 26.15
        # 0.07 23.73
        # 0.09 21.84
        # random0.1 27.8
        noise_signal = add_rician_noise(signal, level)
        exp_tr_r1 = (1 - noise_signal / np.sin(alpha)) / (1 - noise_signal * np.cos(alpha) / np.sin(alpha))
        R1 = -np.log(exp_tr_r1) / TR
        ct = (R1 - 1 / T10) / r1
        param = np.array([ktrans, vp, ve])
        return T10, ct, noise_signal, cp, param
        # return np.array([T10, ]).astype(np.float32), fited.astype(np.float32), data.astype(np.float32), param.astype(np.float32)

    def signal_fited(self, item):
        T10, ct, noise_signal, cp, param = self.generate_signal(item)
        # sample = NLSQ(T10, cp, noise_signal)
        sample, _s = fit_eTofts(T10, cp, noise_signal)
        # sample = param
        np.clip(sample, a_min=np.array([1e-5, 0.0005, 0.04]), a_max=np.array([1, 0.1, 0.6]))
        return np.array([T10, ]).astype(np.float32), \
               ct.astype(np.float32), \
               noise_signal.astype(np.float32), \
               cp.astype(np.float32), \
               param.astype(np.float32), \
               sample.astype(np.float32)

    def signal_fited_raw(self, item):
        T10, ct, noise_signal, cp, param = self.generate_signal(item)
        # sample = NLSQ(T10, cp, noise_signal)
        # sample, _s = fit_eTofts(T10, cp, noise_signal)
        # np.clip(sample, a_min=np.array([1e-5, 0.0005, 0.04]), a_max=np.array([1, 0.1, 0.6]))
        return np.array([T10, ]).astype(np.float32), \
               ct.astype(np.float32), \
               noise_signal.astype(np.float32), \
               cp.astype(np.float32), \
               param.astype(np.float32)

    def __len__(self):
        # return 1
        return self.len

    def read_cps(self):
        cps = []
        for root in self.root_list:
            for cp_path in os.listdir(root):
                info_file = os.path.join(root, cp_path)
                cp = np.load(info_file, allow_pickle=True)[:, 0]
                cpm = np.argmax(cp)
                to_cp = cp[cpm - 12:cpm + 100]
                if to_cp.shape == (112,):
                    cps.append(cp[cpm - 12:cpm + 100])
                else:
                    print(cp_path, to_cp.shape)
            return cps

    def generate_cp(self):
        cp1id, cp2id = np.random.choice(len(self.cps), 2)
        cp1, cp2 = self.cps[cp1id], self.cps[cp2id]
        lam = np.random.random()
        cp = lam * cp1 + (1 - lam) * cp2
        return cp

    def generate_ktrans(self):
        log_or_line = np.random.random() >= 0.5 or True
        lam = np.random.random()
        if log_or_line:  # line
            return lam * 0.00001 + (1 - lam) * 1
        else:
            log_trans = lam * (-5) + (1 - lam) * np.log10(1)
            return np.power(10, log_trans)

    def generate_vp(self):
        lam = np.random.random()
        return lam * 0.0005 + (1 - lam) * 0.1

    def generate_ve(self):
        lam = np.random.random()
        return lam * 0.04 + (1 - lam) * 0.6

    def generate_t1(self):
        lam = np.random.random()
        return lam * 0.8 + (1 - lam) * 3.5

    def snr(self, item):
        T10, ct, signal, cp, param = self.generate_signal(item)
        R1 = 1 / T10 + r1 * ct
        re_signal = (1 - np.exp(-TR * R1)) * np.sin(alpha) / (1 - np.exp(-TR * R1) * np.cos(alpha))
        # import matplotlib.pyplot as plt
        # plt.plot(signal, label='clean')
        # plt.plot(re_signal, label='re_signal')
        # plt.title(str(item))
        # plt.legend()
        # plt.pause(0.1)
        # plt.cla()
        noise = np.mean(signal[:6], keepdims=True) - signal[:6]
        signal_power = np.sum(np.square(signal[:6]))
        noise_power = np.sum(np.square(noise[:6]))
        level = np.sqrt(noise_power / signal_power)
        snr = 10 * np.log10(signal_power / noise_power)
        return level, snr

    def snr_full(self, item):
        t10, fited, data, par = self.__getitem__(item)
        noise = fited - data[0, :]
        signal_power = np.sum(np.square(fited))
        noise_power = np.sum(np.square(noise))
        level = np.sqrt(noise_power / signal_power)
        snr = 10 * np.log10(signal_power / noise_power)
        return level, snr


class FastSimulatePatient():
    def __init__(self, root):
        self.root = root
        print(f'loading datset {root}')
        self.data = torch.load('{}'.format(root))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        T10, ct, noise_signal, cp, param, sample = self.data[item]
        return T10, ct.unsqueeze(0), noise_signal.unsqueeze(0), cp.unsqueeze(0), param, sample
        # return T10, fited, noise_signal, cp, param, sample


def add_gaussian_noise(signal, noise_per):
    rms = np.sqrt(np.mean(np.square(signal[:6])))
    n_std = rms * noise_per
    noise = np.random.randn(*signal.shape) * n_std
    return signal + noise


def add_rician_noise(signal, noise_per):
    rms = np.sqrt(np.mean(np.square(signal[:6])))
    n_std = rms * noise_per
    noise_real = np.random.randn(*signal.shape) * n_std
    noise_img = np.random.randn(*signal.shape) * n_std
    return np.sqrt((signal + noise_real) ** 2 + noise_img ** 2)


def train_valid_test_dataset():
    train_data = SimulatePatient([f'/ext/fk/data/cps/dataset{i}' for i in range(1, 5)], length=400000)
    test_data = SimulatePatient([f'/ext/fk/data/cps/dataset{i}' for i in range(5, 6)], length=100000)
    training = train_data
    train_set, val_set = random_split(training, [int(training.__len__() * 0.8),
                                                 training.__len__() - int(training.__len__() * 0.8)])
    return train_set, val_set, test_data

