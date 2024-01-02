import torch
import torch.nn.functional as F
import torch.nn as nn
import pdb

class WeightedDiceLoss(nn.Module):
    def __init__(self, axis=(-1,-2,-3), smooth=1e-6):
        super().__init__()
        self.axis = axis
        self.smooth = smooth
        
    def forward(self, y_pred, y_truth):
        return 1 - torch.mean((2 * torch.sum(y_pred * y_truth, dim = self.axis) + self.smooth)/
                         (torch.sum(y_pred, dim = self.axis) + torch.sum(y_truth, dim = self.axis) + self.smooth))

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='mean')

    def forward(self, pred, gt):
        pred_A = pred
        gt_dose = gt[0]
        possible_dose_mask = gt[1]


        pred_A = pred_A[possible_dose_mask > 0]
        
        gt_dose = gt_dose[possible_dose_mask > 0]

        L1_loss = self.L1_loss_func(pred_A, gt_dose)
        return L1_loss

class Diversityloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='mean')

    def forward(self, pred, ref):
        
        div_loss = max(0, 0.2 - self.L1_loss_func(pred, ref)*2/(pred + ref).mean())
        return div_loss