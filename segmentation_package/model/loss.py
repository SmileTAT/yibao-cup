from __future__ import print_function

import torch
from torch import nn
from torch.nn import functional as F

def dice_coef(y_true, y_pred_logits, smooth=1.0):
  y_pred_prob = F.sigmoid(y_pred_logits)
  y_true_f = y_true.view(-1)
  y_pred_prob_f = y_pred_prob.view(-1)
  intersection = (y_true_f * y_pred_prob_f).sum()
  return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_prob_f.sum() + smooth)


def bce_loss(y_true, y_pred_logits):
  y_pred_prob = F.sigmoid(y_pred_logits)
  y_true_f = y_true.view(-1)
  y_pred_prob_f = y_pred_prob.view(-1).clamp(min=1e-7, max=1-1e-7)
  return -(y_true_f * y_pred_prob_f.log() + (1. - y_true_f) * (1 - y_pred_prob_f).log()).mean()


def dice_loss(y_true, y_pred_logits, smooth=1.):
  return 0 - dice_coef(y_true, y_pred_logits, smooth)


class weighted_bce_dice_loss(nn.Module):
  def __init__(self, weight=1e-3):
    super(weighted_bce_dice_loss, self).__init__()

    self.weight = weight
    self.bce_loss = bce_loss
    self.dice_loss = dice_loss

  def forward(self, y_true, y_pred_logits):
    return self.weight*self.bce_loss(y_true, y_pred_logits) + self.dice_loss(y_true, y_pred_logits)


class match_loss(nn.Module):
  def __init__(self, weight=1e-3):
    super(match_loss, self).__init__()

    self.weighted_bce_dice_loss = weighted_bce_dice_loss(weight=weight)

  def compute_match_loss(self, y_true, y_pred):
    return self.weighted_bce_dice_loss(y_true, y_pred)

  def compute_iou(self, y_true, y_pred_p, smooth=1, threshold=0.5):
    """
    :param y_true: C x H x W
    :param y_pred_p: same size
    :return: iou
    """
    y_true_f = y_true.view(-1)
    y_pred_p_f = y_pred_p.view(-1)
    y_pred_p_f = torch.where(y_pred_p_f>threshold, torch.ones_like(y_pred_p_f), torch.zeros_like(y_pred_p_f))
    Intersection =  (y_true_f * y_pred_p_f).sum()
    Union = (y_true_f + y_pred_p_f).sum() - Intersection
    return (Intersection + smooth) / (Union + smooth)

  def forward(self, y_true, y_pred_logic):
    y_pred_p = y_pred_logic.sigmoid()
    match_loss_return = 0
    size = y_true.size()[0]

    for i in range(size):
      miou_0 = self.compute_iou(y_true=y_true[i, 0], y_pred_p=y_pred_p[i, 0]) + \
               self.compute_iou(y_true=y_true[i, 1], y_pred_p=y_pred_p[i, 1])
      miou_1 = self.compute_iou(y_true=y_true[i, 0], y_pred_p=y_pred_p[i, 1]) + \
               self.compute_iou(y_true=y_true[i, 1], y_pred_p=y_pred_p[i, 0])

      if miou_0 > miou_1:
        match_loss_return += self.compute_match_loss(y_true[i, 0], y_pred_logic[i, 0]) + \
                             self.compute_match_loss(y_true[i, 1], y_pred_logic[i, 1])
      else:
        match_loss_return += self.compute_match_loss(y_true[i, 0], y_pred_logic[i, 1]) + \
                             self.compute_match_loss(y_true[i, 1], y_pred_logic[i, 0])

    return match_loss_return / size / 2
