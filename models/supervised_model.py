import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import utils
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import pearsonr
import torch.nn.functional as F
from arc2vec.model import Encoder_rl
from arc2vec.configs import configs
from arc2vec.util import preprocessing


class SSL_Stage(nn.Module):  # encoder+predictor

    def __init__(self, args):
        super(SSL_Stage, self).__init__()
        self.input_dim = len(args.gate_type) + args.num_qubits + 2
        self.encoder = Encoder_rl(self.input_dim, args.hidden_dim, args.gmp_dim, args.dim, args.hops, 3)
        self.linear = nn.Linear(args.dim, self.input_dim)
        #  stage2
        self.linear1 = nn.Linear(128, 30)
        self.linear2 = nn.Linear(30, 1)
        self.BN = nn.BatchNorm1d(30)
        self.cfg = configs[args.cfg]
        self.args = args

    def forward(self, adj, ops):
        adj, ops, prep_reverse = preprocessing(adj, ops, **self.cfg['prep'])
        x = self.encoder(ops, adj)
        # x = torch.relu(self.linear(x))
        x = torch.sum(x, dim=1)
        feature = x
        x = self.linear1(x)
        x = torch.relu(self.BN(x))
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = x.mean(-1)

        return x, feature





