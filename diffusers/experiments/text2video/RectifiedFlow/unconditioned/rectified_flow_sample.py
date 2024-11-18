from __future__ import annotations

import math

import torch
from torch.nn import Module
from torchvision.utils import save_image
from pathlib import Path

from helper_functions_and_lossess import *
from rectified_flow import *

from ema_pytorch import EMA


class Trainer(Module):
    def __init__(
        self,
        exp_folder: str,
        ckpt:int,
        num_samples: int = 16,
        ode_sample_steps: int = 20,
        unet_dim: int = 64,
        use_ema = True,
        ema_update_after_step: int=100,
        ema_update_every: int=10,
        ema_beta = 0.90,
        clip_during_sampling = True,
        clip_flow_during_sampling = False,
        clip_flow_values = (-3.,3.),
        loss_fn: str = "VGGLoss_MSE",
        data_shape = (3,32,32),
        forward_time_sampling="logit_normal"
    ):
        super().__init__()

        ### non core, housekeeping

        exp_folder = Path(exp_folder)
        self.inference_folder = exp_folder / "inference" / f"{ckpt}"
        self.inference_folder.mkdir(exist_ok = True, parents = True)
        assert self.inference_folder.is_dir()

        self.num_sample_rows = int(math.sqrt(num_samples))
        assert (self.num_sample_rows ** 2) == num_samples, f'{num_samples} must be a square'
        self.num_samples = num_samples

        self.ode_sample_steps = ode_sample_steps


        ### CORE
        
        self.model = RectifiedFlow(dict(dim = unet_dim), clip_during_sampling, clip_flow_during_sampling, clip_flow_values, loss_fn, data_shape, forward_time_sampling).cuda()

        self.use_ema = use_ema
        self.ema_model = None
        if self.use_ema:
            self.ema_model = EMA(
                self.model,
                forward_method_names = ('sample',),
                beta = ema_beta,              # exponential moving average factor
                update_after_step = ema_update_after_step,    # only after this number of .update() calls will it start updating
                update_every = ema_update_every,          # how often to actually update, to save on compute (updates every 10th .update() call)
            )
            self.ema_model.cuda()

        self.load(exp_folder / "checkpoints" / f"{ckpt}.pt")

    def load(self, path):
        
        load_package = torch.load(path,map_location=torch.device('cuda:0'))        
        self.model.load_state_dict(load_package["model"])
        if self.use_ema:
            self.ema_model.load_state_dict(load_package["ema_model"])

    def sample(self, fname):
        eval_model = default(self.ema_model, self.model)
        with torch.no_grad():
            # eval_model.eval()
            sampled = eval_model.sample(batch_size=self.num_samples, steps=self.ode_sample_steps)
      
        sampled = rearrange(sampled, '(row col) c h w -> c (row h) (col w)', row = self.num_sample_rows)
        sampled.clamp_(0., 1.)

        save_image(sampled, fname)
        return sampled

    def forward(self):

        with torch.no_grad():
            self.sample(fname=str(self.inference_folder / f"steps_{self.ode_sample_steps}-ema_{self.use_ema}.png"))

if __name__ == "__main__":

    train_dict = dict(
        exp_folder= "rectified_flow_2022/mnist_exp1",
        ckpt=38_000,
        num_samples = 64,
        ode_sample_steps= 100,
        unet_dim= 64,
        use_ema = False,
        ema_update_after_step=100,
        ema_update_every=10,
        ema_beta = 0.90,
        clip_during_sampling = True,
        clip_flow_during_sampling = False,
        clip_flow_values = (-3.,3.),
        loss_fn= "VGGLoss_MSE",
        data_shape = (1,28,28),
        forward_time_sampling="logit_normal"
    )

    trainer = Trainer(**train_dict)
    trainer()
