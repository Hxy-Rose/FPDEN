import torch
from torch import nn


class CALSTMVarAvg(nn.Module):
    def __init__(self, channel_num=64, levels=1):
        super(CALSTMVarAvg, self).__init__()
        self.channel_nun = channel_num
        self.levels = levels
        self.embeding = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=channel_num, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.encoder = nn.LSTM(input_size=channel_num, hidden_size=channel_num, num_layers=levels, bidirectional=True)
        self.pre = nn.Sequential(
            nn.Linear(in_features=channel_num * 2, out_features=3),
        )
        self.logvar = nn.Sequential(
            nn.Linear(in_features=channel_num * 2, out_features=3),
        )

    def forward(self, cp_en, signal_en):
        param = torch.cat((signal_en, cp_en), dim=1)
        param_emb = self.embeding(param)
        in_feature = param_emb  # batch, ch, L
        output, (h, c) = self.encoder(in_feature.permute((2, 0, 1)))  # [L, batch, ch]
        feature = output
        feature = torch.mean(feature, dim=0)
        return torch.split(self.pre(feature), 1, dim=1) + torch.split(self.logvar(feature), 1, dim=1)
