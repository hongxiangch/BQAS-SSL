import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP

class Encoder_rl(nn.Module):
    def __init__(self, input_dim, hidden_dim, gmp_dim, latent_dim, num_hops, num_mlp_layers):
        super(Encoder_rl, self).__init__()
        self.num_layers = num_hops
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gmp_dim = gmp_dim
        self.latent_dim = latent_dim
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.gmp = nn.AdaptiveAvgPool1d(gmp_dim)
        self.fc = nn.Linear(gmp_dim, self.latent_dim)

    def forward(self, ops, adj):
        batch_size, node_num, opt_num = ops.shape
        x = ops
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj.float(), x)
            agg = (1 + self.eps[l]) * x.view(batch_size * node_num, -1) \
                  + neighbor.view(batch_size * node_num, -1)
            x = F.relu(self.batch_norms[l](self.mlps[l](agg)).view(batch_size, node_num, -1))
        out = self.gmp(x)
        out = self.fc(out)

        return out

class Model_rl(nn.Module):
    def __init__(self, input_dim, hidden_dim, gmp_dim, latent_dim, num_hops, num_mlp_layers,
                 dropout, args, **kwargs):
        super(Model_rl, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gmp_dim = gmp_dim
        self.latent_dim = latent_dim
        self.num_layers = num_hops
        self.batch = args.bs
        self._encoder = Encoder_rl(input_dim, hidden_dim, gmp_dim, latent_dim, num_hops, num_mlp_layers)
        self.fc = nn.Linear(latent_dim, 1)

    def forward(self, ops, adj):
        out = self._encoder(ops, adj)
        out = torch.sum(out, dim=-2)
        out = torch.relu(out)
        out = self.fc(out)  # 32*1
        out = torch.flatten(torch.sigmoid(out))

        return out