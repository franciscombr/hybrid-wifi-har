import torch
import torch.nn.functional as F
import torch.nn as nn


class TemporalEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalEncoder,self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.activation(x)
        x = x.mean(dim=2)
        return x


class CSI2HARModel(nn.Module):
    def __init__(self, embedding_dim, num_antennas=3, num_subcarriers=30, num_time_slices)