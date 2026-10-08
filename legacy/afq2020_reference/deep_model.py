from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch import nn, optim



import torch
import torch.nn as nn
import torch.nn.functional as F

class Trans(nn.Module):
    def __init__(self):
        super(Trans, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=102, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.linear = nn.Sequential(nn.Flatten(),
                                    nn.Linear(102*20, 1024),
                                    nn.Linear(1024, 2),
                                    nn.Dropout(0.1),
                                    nn.Softmax())
    def forward(self,x):
        x=self.transformer_encoder(x)
        x=self.linear(x)
        return x


class TextCNN(nn.Module):

    def __init__(self, CONFIG):
        super(TextCNN, self).__init__()
        self.conv = nn.Conv2d(1, self.num_filters, (self.kernal_size, self.embedding_dim))
        self.linear = nn.Sequential(nn.Flatten(),
                                    nn.Linear(102 * 20, 1024),
                                    nn.Linear(1024, 2),
                                    nn.Dropout(0.1),
                                    nn.Softmax())

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.max_pool1d(x, x.size(2))
        x = self.linear(x)
        return x


class lstm(nn.Module):

    def __init__(self):
        super(lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=102, hidden_size=256, batch_first=True,bidirectional=True)
        self.linear = nn.Sequential(nn.Flatten(),
                                    nn.Linear(10240, 1024),
                                    nn.Linear(1024, 2),
                                    nn.Dropout(0.1),
                                    nn.Softmax())
    def forward(self, x):
        x,_=self.lstm(x)
        x = self.linear(x)
        return x
