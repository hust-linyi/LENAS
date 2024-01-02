# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='mean')

    def forward(self, pred, gt):

        pred_A = pred[0]
        gt_dose = gt[0]
        possible_dose_mask = gt[1]

        pred_A = pred_A[possible_dose_mask > 0]
        
        gt_dose = gt_dose[possible_dose_mask > 0]

        pred_A = pred_A.cuda()
        gt_dose = gt_dose.cuda()

        L1_loss = self.L1_loss_func(pred_A, gt_dose)
        
        return L1_loss


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, gt):

        pred = pred.to('cuda')
        gt = gt.to('cuda')

        log_softmax_outputs = F.log_softmax(pred, dim=1)
        softmax_targets = F.softmax(gt, dim=1)

        ce_loss = -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()
        # print(len(ce_loss))
        
        return ce_loss

class betweenLoss(nn.Module):
    def __init__(self, gamma=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], loss=nn.MSELoss()):
        super(betweenLoss, self).__init__()
        self.gamma = gamma
        self.loss = loss

    def forward(self, pred, gt):
        pred_ = pred[1]
        gt_ = gt[2]
        #print(len(pred_), len(gt_))
        assert len(pred_)
        assert len(pred_) == len(gt_)
        length = len(pred_)

        for i in range(len(pred_)):
            pred_[i] = pred_[i].cuda()
            gt_[i] = gt_[i].cuda()

        res = sum([self.gamma[i] * self.loss(pred_[i], gt_[i]) for i in range(length)])
        return res



class discriminatorLoss(nn.Module):
    def __init__(self, models, eta=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], loss=nn.BCELoss()):
        super(discriminatorLoss, self).__init__()
        self.models = models
        self.models = self.models.to('cuda')

        self.eta = eta
        self.loss = loss

    def forward(self, pred, gt):
        pred_ = pred[1]
        gt_ = gt[2]
        inputs = [torch.cat((i,j),0) for i, j in zip(pred_, gt_)]
        batch_size = inputs[0].size(0)
        target = torch.FloatTensor([[1, 0] for _ in range(batch_size//2)] + [[0, 1] for _ in range(batch_size//2)])
        for input_ in inputs:
            input_ = input_.to('cuda')
        target = target.to('cuda')
        
        outputs = self.models(inputs)
        
        for i, output in enumerate(outputs):
            self.loss(output, target)
        res = sum([self.eta[i] * self.loss(output, target) for i, output in enumerate(outputs)])
        return res


class discriminatorFakeLoss(nn.Module):
    def forward(self, outputs, targets):
        res = (0*outputs[0]).sum()
        return res