#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    criteria.py
# @Project:     GuideNet
# @Author:      jie
# @Time:        2021/3/14 7:51 PM

import torch
import torch.nn as nn
from SCI_loss import LossFunction

__all__ = [
    'RMSE',
    'MSE',
]


class RMSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        val_pixels = (target > 1e-3).float().cuda()
        err = (target * val_pixels - outputs * val_pixels) ** 2
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.sqrt(loss / cnt)


class MSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        val_pixels = (target > 1e-3).float().cuda()
        loss = target * val_pixels - outputs * val_pixels
        loss = (loss ** 2).mean()
        return loss


class MSE_and_SCI(nn.Module):

    def __init__(self):
        super().__init__()
        self.pred_loss = MSE()
        self.SCI_loss = LossFunction()

    def forward(self, outputs, target, inlist, ilist, *args):
        loss1 = self.pred_loss(outputs, target)
        loss2 = 0
        for i in range(3):
            loss2 += self.SCI_loss(inlist[i], ilist[i])
        loss = loss1 + 0.3 * loss2
        return loss