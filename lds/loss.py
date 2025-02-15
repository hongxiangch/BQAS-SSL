import math

import torch
import torch.nn.functional as F
import torch.nn as nn


class weighted_mse_loss:
    def __init__(self):
        print('loss function:weighted mse loss')

    def compute_loss(self, inputs, targets, weights=None):
        loss = F.mse_loss(inputs, targets, reduction='none')
        if weights is not None:
            loss *= weights
        loss = torch.mean(loss)
        return loss

class weighted_bce_loss:
    def __init__(self):
        print('loss function:weighted bce loss')

    def compute_loss(self, inputs, targets, weights=None):
        loss_func = nn.BCELoss(reduction='none')
        loss = loss_func(inputs, targets)
        if weights is not None:
            # loss *= weights.expand_as(loss)
            loss *= weights
        loss = torch.mean(loss)
        return loss

class BPRLoss(torch.nn.Module):

    def __init__(self, exp_weighted=False):
        super(BPRLoss, self).__init__()
        self.exp_weighted = exp_weighted

    def compute_loss(self, input, target):
        N = input.size(0)
        total_loss = 0
        for i in range(N):
            indices = (target>target[i])
            x = torch.log(1 + torch.exp(-(input[indices] - input[i])))

            if self.exp_weighted:
                x = (torch.exp(target[i]) - 1) * (torch.exp(target[indices]) - 1) * x
            else:
                x = x
            total_loss += torch.sum(x)
        if self.exp_weighted:
            return 2 / (N * (math.e - 1))**2 * total_loss
        else:
            return 2 / N**2 * total_loss