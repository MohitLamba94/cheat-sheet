from __future__ import annotations

import math
from copy import deepcopy
from collections import namedtuple
from typing import Tuple, List, Literal, Callable

import torch
from torch import Tensor
from torch import nn, pi, from_numpy
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from torchdiffeq import odeint

import torchvision
from torchvision.utils import save_image
from torchvision.models import VGG16_Weights

import einx
from einops import einsum, reduce, rearrange, repeat
from einops.layers.torch import Rearrange

from scipy.optimize import linear_sum_assignment


# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

# tensor helpers

def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))


# normalizing helpers

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# noise schedules

def cosmap(t):
    # Algorithm 21 in https://arxiv.org/abs/2403.03206
    return 1. - (1. / (torch.tan(pi / 2 * t) + 1))

# losses


class LPIPSLoss_MSE(Module):
    def __init__(self):
        super().__init__()
        self.loss_fn_vgg = LPIPSLoss()
    def forward(self, pred_data, data):
        return self.loss_fn_vgg(pred_data, data), F.mse_loss(pred_data, data)

class VGGLoss_MSE(Module):
    def __init__(self):
        super().__init__()
        self.loss_fn_vgg = VGGLoss()
    def forward(self, pred_flow, flow,  z, padded_times, data):
        return self.loss_fn_vgg(pred_flow, flow), F.mse_loss(pred_flow, flow)
    
class VGGLossonData_MSEonFlow(Module):
    def __init__(self):
        super().__init__()
        self.loss_fn_vgg = VGGLoss()

    def forward(self, pred_flow, flow, z, padded_times, data):
        pred_data = z + (pred_flow*(1. - padded_times))
        '''
        pred_data = tX + (1-t)N + (X-N)(1-t)
                  = tX + (1-t)(X)
                  = X
        '''
        return self.loss_fn_vgg(pred_data, data), F.mse_loss(pred_flow, flow)
    
class VGGLossonData_MSEonFlow2(Module):
    def __init__(self):
        super().__init__()
        self.loss_fn_vgg = VGGLoss()

    def forward(self, pred_flow, flow, z, padded_times, data):
        pred_data = z + (pred_flow*(1. - padded_times))
        return (0.2*self.loss_fn_vgg(pred_data, data)) + (0.8*F.l1_loss(pred_data, data)), F.mse_loss(pred_flow, flow)

class LPIPSLoss(Module):
    def __init__(
        self,
        vgg: Module | None = None,
        vgg_weights: VGG16_Weights = VGG16_Weights.DEFAULT,
    ):
        super().__init__()

        if not exists(vgg):
            vgg = torchvision.models.vgg16(weights = vgg_weights)
            vgg.requires_grad_(False)
            vgg.classifier = nn.Sequential(*vgg.classifier[:-2])

        self.vgg = [vgg]

    def forward(self, pred_data, data, reduction = 'mean'):
        vgg, = self.vgg
        vgg = vgg.to(data.device)

        pred_embed = vgg(pred_data)
        with torch.no_grad():
            embed = vgg(data)

        loss = F.mse_loss(embed, pred_embed, reduction = reduction)

        if reduction == 'none':
            loss = reduce(loss, 'b ... -> b', 'mean')

        return loss
    
class VGGLoss(Module):
    def __init__(
        self,
        vgg: Module | None = None,
        vgg_weights: VGG16_Weights = VGG16_Weights.DEFAULT,
    ):
        super().__init__()

        if not exists(vgg):
            vgg = torchvision.models.vgg16(weights = vgg_weights).features[:10]
            vgg.requires_grad_(False)

        self.vgg = [vgg]

    def forward(self, pred_data, data, reduction = 'mean'):
        vgg, = self.vgg
        vgg = vgg.eval()
        vgg = vgg.to(data.device) # Accelerator should automatically handle this

        bb,cc,hh,ww = data.shape
        if cc==1:
            pred_data = pred_data.repeat(1,3,1,1)
            data = data.repeat(1,3,1,1)

        pred_embed = vgg(pred_data)
        with torch.no_grad():
            embed = vgg(data)

        loss = F.mse_loss(embed, pred_embed, reduction = reduction)

        if reduction == 'none':
            loss = reduce(loss, 'b ... -> b', 'mean')

        return loss

class PseudoHuberLoss(Module):
    def __init__(self, data_dim: int = 3):
        super().__init__()
        self.data_dim = data_dim

    def forward(self, pred, target, reduction = 'mean', **kwargs):
        data_dim = default(self.data_dim, kwargs.pop('data_dim', None))

        c = .00054 * self.data_dim
        loss = (F.mse_loss(pred, target, reduction = reduction) + c * c).sqrt() - c

        if reduction == 'none':
            loss = reduce(loss, 'b ... -> b', 'mean')

        return loss

class PseudoHuberLossWithLPIPS(Module):
    def __init__(self, data_dim: int = 3, lpips_kwargs: dict = dict()):
        super().__init__()
        self.pseudo_huber = PseudoHuberLoss(data_dim)
        self.lpips = LPIPSLoss(**lpips_kwargs)

    def forward(self, pred_flow, target_flow, *, pred_data, times, data):
        huber_loss = self.pseudo_huber(pred_flow, target_flow, reduction = 'none')
        lpips_loss = self.lpips(data, pred_data, reduction = 'none')

        time_weighted_loss = huber_loss * (1 - times) + lpips_loss * (1. / times.clamp(min = 1e-1))
        return time_weighted_loss.mean()

class MSELoss(Module):
    def forward(self, pred, target, **kwargs):
        return F.mse_loss(pred, target)

class MSEAndDirectionLoss(Module):
    """
    Figure 7 - https://arxiv.org/abs/2410.10356
    """

    def __init__(self, cosine_sim_dim: int = 1):
        super().__init__()
        assert cosine_sim_dim > 0, 'cannot be batch dimension'
        self.cosine_sim_dim = cosine_sim_dim

    def forward(self, pred, target, **kwargs):
        mse_loss = F.mse_loss(pred, target)

        direction_loss = (1. - F.cosine_similarity(pred, target, dim = self.cosine_sim_dim)).mean()

        return mse_loss + direction_loss
