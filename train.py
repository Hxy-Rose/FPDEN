import argparse
import datetime
import logging
import os
import time

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from model.RealNVP import RealNVP
from model.network import CALSTMVarAvg
from utils.config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=1, type=int)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--name', default='flow', type=str)
args = parser.parse_args()
# merge config
default_config = get_config()
train_config = default_config.train
train_config.update(args.__dict__)
# train_config.max_epoch = 10
# apply config
gpu = torch.device('cuda:{}'.format(train_config.gpu))
torch.cuda.set_device(gpu)
from model.TKModel import eTofts_torch
from dataset import train_valid_test_dataset

torch.multiprocessing.set_sharing_strategy('file_system')
lr = train_config.lr

max_epoch = train_config.max_epoch
batch = train_config.batch

str_time = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))

reg_model = CALSTMVarAvg(levels=2, channel_num=128).cuda()
flow_model = RealNVP().cuda()
eTofts_m = eTofts_torch().cuda()
model_path = os.path.join(f'experiment/{train_config.name}')


def init_logging(log_root_dir):
    os.makedirs(log_root_dir, exist_ok=True)
    import sys

    log_root = logging.getLogger()
    log_root.setLevel(logging.INFO)
    formatter = logging.Formatter("MRI %(asctime)s-%(message)s")
    handler_file = logging.FileHandler(os.path.join(log_root_dir, "training.log"))
    handler_stream = logging.StreamHandler(sys.stdout)
    handler_file.setFormatter(formatter)
    handler_stream.setFormatter(formatter)
    log_root.addHandler(handler_file)
    log_root.addHandler(handler_stream)


init_logging(model_path)


def get_signal(ktrans, vp, ve, t1, cp):
    ktrans, vp, ve = ktrans / 2 + 0.5, vp / 20 + 0.05, ve / 3 + 0.3
    ktrans, vp, ve = torch.clamp(ktrans, 1e-5, 1), torch.clamp(vp, 0.0005, 0.1), torch.clamp(ve, 0.04, 0.6)
    signal = eTofts_m(ktrans.detach(), vp.detach(), ve.detach(), t1, cp.squeeze(1)).unsqueeze(1)
    return signal


def normalize_parameters(ktrans, vp, ve):
    ktrans, vp, ve = ktrans * 2 - 1, \
                     vp * 20 - 1, \
                     ve * 3 - 0.9
    return ktrans, vp, ve


def train(reg_model, optimizer, name='uncertainty'):
    '''
    :param train_set:.,
    :param valid_set:
    :param ssm_model:
    :param optimizer:
    :param name:
    :return:
    '''

    reg_model.train()
    min_loss = torch.FloatTensor([float('inf'), ])

    for epoch in range(max_epoch):
        train_l, valid_l = 0, 0
        c_train, c_valid = 0, 0
        losses = []
        dataloader = DataLoader(dataset=train_set, batch_size=batch, shuffle=True, num_workers=2)
        for count, data in enumerate(dataloader):
            T10, ct, noise_signal, cp, sample = data
            T10, ct, noise_signal, cp, sample = T10.cuda(), ct.cuda(), noise_signal.cuda() * 20, cp.cuda(), sample.cuda()
            ktrans_p, vp_p, ve_p, ktrans_p_var, vp_p_var, ve_p_var = reg_model(ct, cp / 3)

            ktrans, vp, ve = sample[:, 0:1], sample[:, 1:2], sample[:, 2:3]  # , par[:, 3:4], par[:, 4:5]
            ktrans_normalize_sample, vp_normalize_sample, ve_normalize_sample = normalize_parameters(ktrans, vp, ve)

            target = torch.cat((ktrans_p, vp_p, ve_p), dim=1)

            log_var = torch.cat((ktrans_p_var, vp_p_var, ve_p_var), dim=1)
            samples = torch.cat((ktrans_normalize_sample, vp_normalize_sample, ve_normalize_sample), dim=1)
            loss = torch.mean(flow_log_loss(target, log_var, samples))
            losses.append(loss.detach().cpu())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            if count % 10 == 0:
                logging.info(f'train epoch:{epoch},batch:{count}/{len(dataloader)},loss:{loss:.8f}')
        if epoch % 5 != 4:
            continue

        with torch.no_grad():
            valid_load = DataLoader(dataset=valid_set, batch_size=batch, shuffle=False, num_workers=2)
            for count, data in enumerate(valid_load):
                T10, ct, noise_signal, cp, sample = data
                T10, ct, noise_signal, cp, sample = T10.cuda(), ct.cuda(), noise_signal.cuda() * 20, cp.cuda(), sample.cuda()
                ktrans_p, vp_p, ve_p, ktrans_p_var, vp_p_var, ve_p_var = reg_model(ct, cp / 3)

                ktrans, vp, ve = sample[:, 0:1], sample[:, 1:2], sample[:, 2:3]  # , par[:, 3:4], par[:, 4:5]
                ktrans_normalize_sample, vp_normalize_sample, ve_normalize_sample = normalize_parameters(ktrans, vp, ve)

                target = torch.cat((ktrans_p, vp_p, ve_p), dim=1)

                log_var = torch.cat((ktrans_p_var, vp_p_var, ve_p_var), dim=1)
                samples = torch.cat((ktrans_normalize_sample, vp_normalize_sample, ve_normalize_sample), dim=1)
                loss = torch.mean(flow_log_loss(target, log_var, samples))

                valid_l = valid_l + loss * noise_signal.size(0)
                c_valid = c_valid + noise_signal.size(0)
                logging.info('valid epoch:{epoch},batch:{batch}/{v_batch},loss:{loss}'.format(
                    epoch=epoch, batch=count, v_batch=int((valid_set.__len__() - 1) / batch + 1),
                    loss=valid_l.item() / c_valid))
                with open(os.path.join(model_path, 'valid_result_{}.txt'.format(name)), 'a+') as f:
                    f.write('valid epoch:{epoch},batch:{batch}/{v_batch},loss:{loss}\n'.format(
                        epoch=epoch, batch=count, v_batch=int((valid_set.__len__() - 1) / batch + 1),
                        loss=valid_l.item() / c_valid))

        if min_loss > valid_l.cpu():
            torch.save(reg_model.state_dict(), os.path.join(model_path, 'checkpoint.tar'.format(name)))
            torch.save(flow_model.state_dict(), os.path.join(model_path, 'RealNVP.tar'.format(name)))
            min_loss = valid_l.cpu()
        logging.info(
            f'*** valid loss{valid_l.item() / c_valid} min loss:{min_loss.item() / c_valid}')
        return epoch


def test(dataset, reg_model, name):
    mse_record = [0] * 6
    residual_record = 0
    count_record = 0
    mae_record = [0] * 6
    with torch.no_grad():
        data_load = DataLoader(dataset=dataset, batch_size=get_config().test.batch, shuffle=False, num_workers=4)
        for count, data in enumerate(data_load):
            T10, ct, noise_signal, cp, sample = data
            T10, ct, noise_signal, cp, sample = T10.cuda(), ct.cuda(), noise_signal.cuda() * 20, cp.cuda(), sample.cuda()
            ktrans_p, vp_p, ve_p, ktrans_p_var, vp_p_var, ve_p_var = reg_model(ct, cp / 3)
            recon_signal = get_signal(ktrans_p, vp_p, ve_p, T10, cp)

            ktrans, vp, ve = ktrans_p / 2 + 0.5, vp_p / 20 + 0.05, ve_p / 3 + 0.3
            ktrans_gt, vp_gt, ve_gt = sample[:, 0:1], sample[:, 1:2], sample[:, 2:3]

            mse_record[0] = mse_record[0] + torch.sum((ktrans - ktrans_gt) ** 2).item()
            mse_record[1] = mse_record[1] + torch.sum((vp - vp_gt) ** 2).item()
            mse_record[2] = mse_record[2] + torch.sum((ve - ve_gt) ** 2).item()

            mae_record[0] = mae_record[0] + torch.sum(torch.abs(ktrans - ktrans_gt)).item()
            mae_record[1] = mae_record[1] + torch.sum(torch.abs(vp - vp_gt)).item()
            mae_record[2] = mae_record[2] + torch.sum(torch.abs(ve - ve_gt)).item()

            residual_record = residual_record + torch.sum(torch.sum((recon_signal - noise_signal) ** 2, dim=-1)).item()
            count_record = count_record + noise_signal.size(0)

            logging.info('test batch:{batch}/{total_batch}'.format(batch=count,
                                                                   total_batch=int((dataset.__len__() - 1) / (
                                                                       get_config().test.batch) + 1)))

    loss_info = '\nMSE\nk_trans,{},\nvp,{},\nve,{},\nres,{}\n'.format(
        mse_record[0] / count_record, mse_record[1] / count_record,
        mse_record[2] / count_record, residual_record / count_record,
    )
    MAE_info = '\nMAE\nk_trans,{},\nvp,{},\nve,{},\nres,{}\n'.format(
        mae_record[0] / count_record, mae_record[1] / count_record, mae_record[2] / count_record,
        residual_record / count_record,
    )

    logging.info(loss_info)
    logging.info(MAE_info)
    with open(os.path.join(model_path, 'Result_all_{}.txt'.format(name)), 'a+') as f:
        f.write(loss_info)
        f.write(MAE_info)


if not os.path.exists(model_path):
    os.makedirs(model_path)
loss_l1 = nn.L1Loss()
loss_l2 = nn.MSELoss()


def flow_log_loss(mu, log_var, sample):
    sigma = torch.exp(log_var)
    bar_mu = (sample - mu) / sigma  # [batch, 5]
    log_phi = flow_model.log_prob(bar_mu)
    return log_var.sum(1) - log_phi


optimizer = Adam([{'params': reg_model.parameters()}, {'params': flow_model.parameters()}], lr=lr)

if not os.path.exists(model_path):
    os.makedirs(model_path)

with open(os.path.join(model_path, 'config.dict'), 'w') as f:
    import pprint

    f.write(pprint.pformat(train_config))

train_time = str(datetime.datetime.now())
train_set, valid_set, test_set = train_valid_test_dataset()
logging.info(f'Dataset: train:{len(train_set)}, valid:{len(valid_set)}, test:{len(test_set)}')
s_epoch = train(reg_model, optimizer, name='flow')
logging.info(F'Train start at {train_time},total {s_epoch} epoch')
best_model = torch.load(os.path.join(model_path, 'checkpoint.tar'), map_location=gpu)
reg_model.load_state_dict(best_model)

test(test_set, reg_model, f'test')



logging.info(f'End at {str(datetime.datetime.now())}, result path:{model_path}')
