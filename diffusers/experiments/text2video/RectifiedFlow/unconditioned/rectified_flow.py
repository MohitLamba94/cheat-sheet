from __future__ import annotations

import math

import torch
from torch.nn import Module
from torchdiffeq import odeint
from torchvision.utils import save_image
from einops import rearrange, repeat
from pathlib import Path
from tqdm import tqdm

from helper_functions_and_lossess import *
from model import Unet
from dataset import trainset


class RectifiedFlow(Module):
    def __init__(self, 
                 model:dict|Module,
                 clip_during_sampling,
                 clip_flow_during_sampling,
                 clip_flow_values

                 ):
        super().__init__()

        if isinstance(model, dict):
            model = Unet(**model)

        self.model = model
        self.loss_fn = LPIPSLoss_MSE()#MSELoss()

        # sampling
        self.odeint_kwargs = dict(atol = 1e-5, rtol = 1e-5, method = 'midpoint')
        self.data_shape = (3,32,32)

        # clipping for epsilon prediction
        self.clip_during_sampling = clip_during_sampling
        self.clip_flow_during_sampling = clip_flow_during_sampling # this seems to help a lot when training with predict epsilon, at least for me

        self.clip_values = (-1., 1.)
        self.clip_flow_values = clip_flow_values

        # normalizing fn
        self.data_normalize_fn = normalize_to_neg_one_to_one
        self.data_unnormalize_fn = unnormalize_to_zero_to_one

    @property
    def device(self):
        return next(self.model.parameters()).device

    def predict_flow(self, z, times):
        times = rearrange(times, '... -> (...)')
        if times.numel() == 1:
            times = repeat(times, '1 -> b', b = z.shape[0])
        return self.model(z, times)

    @torch.no_grad()
    def sample(self, batch_size = 1, steps = 16):

        def ode_fn(t, z):
            z = maybe_clip(z)
            flow = self.predict_flow(z, t)
            flow = maybe_clip_flow(flow)
            return flow
        
        ### Non core 
        was_training = self.training
        self.eval()
        maybe_clip = (lambda t: t.clamp_(*self.clip_values)) if self.clip_during_sampling else identity
        maybe_clip_flow = (lambda t: t.clamp_(*self.clip_flow_values)) if self.clip_flow_during_sampling else identity
        
        ### core
        z0 = torch.randn((batch_size, *self.data_shape), device = self.device)
        times = torch.linspace(0., 1., steps, device = self.device)
        trajectory = odeint(ode_fn, z0, times, **self.odeint_kwargs)
        sampled_data = trajectory[-1]

        self.train(was_training)
        return self.data_unnormalize_fn(sampled_data)

    def forward(self,data):
        
        data = self.data_normalize_fn(data)
        z0 = torch.randn_like(data)
        flow = data - z0

        times = torch.rand(data.shape[0], device = self.device)
        padded_times = append_dims(times, data.ndim - 1)

        z = padded_times * data + (1. - padded_times) * z0

        pred_flow = self.predict_flow(z, times)

        lpips_loss, mse_loss = self.loss_fn(pred_flow, flow)
        main_loss = (lpips_loss+mse_loss)/2
        return main_loss, {"lpips_loss":lpips_loss.item(), "mse_loss":mse_loss.item(), "main_loss": main_loss.item(), "gt_flow_max": flow.max().item(), "gt_flow_min": flow.min().item()}



# trainer

from torch.optim import Adam
from accelerate import Accelerator
from torch.utils.data import DataLoader
from ema_pytorch import EMA

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

class Trainer(Module):
    def __init__(
        self,
        exp_folder: str,
        num_train_steps = 70_000,
        batch_size = 16,
        save_results_every: int = 100,
        checkpoint_every: int = 1000,
        num_samples: int = 16,
        ode_sample_steps: int = 20,
        unet_dim: int = 64,
        use_ema = True,
        ema_update_after_step: int=100,
        ema_update_every: int=10,
        ema_beta = 0.90,
        lr = 3e-4,
        adam_weight_decay = 0,
        clip_during_sampling = True,
        clip_flow_during_sampling = False,
        clip_flow_values = (-3.,3.)
    ):
        super().__init__()

        ### non core, housekeeping

        parameters = locals().copy()
        parameters.pop("self")
        parameters.pop("__class__")

        exp_folder = Path(exp_folder)

        self.checkpoints_folder = exp_folder / "checkpoints"
        self.results_folder = exp_folder / "results"
        # log_folder = exp_folder / "logs"

        self.checkpoints_folder.mkdir(exist_ok = True, parents = True)
        self.results_folder.mkdir(exist_ok = True, parents = True)
        # log_folder.mkdir(exist_ok = True, parents = True)

        self.checkpoint_every = checkpoint_every
        self.save_results_every = save_results_every

        self.num_sample_rows = int(math.sqrt(num_samples))
        assert (self.num_sample_rows ** 2) == num_samples, f'{num_samples} must be a square'
        self.num_samples = num_samples

        assert self.checkpoints_folder.is_dir()
        assert self.results_folder.is_dir()

        self.ode_sample_steps = ode_sample_steps
        self.num_train_steps = num_train_steps

        with open(exp_folder / 'parameters.txt', 'w') as df:
            for k,v in parameters.items():
                df.write(f"{k}: {v}\n")


        ### CORE
        
        self.accelerator = Accelerator(log_with="tensorboard", project_dir=str(exp_folder))
        self.accelerator.init_trackers(f"logs")

        self.model = RectifiedFlow(dict(dim = unet_dim), clip_during_sampling, clip_flow_during_sampling, clip_flow_values)

        self.use_ema = use_ema
        self.ema_model = None
        if self.is_main and self.use_ema:
            self.ema_model = EMA(
                self.model,
                forward_method_names = ('sample',),
                beta = ema_beta,              # exponential moving average factor
                update_after_step = ema_update_after_step,    # only after this number of .update() calls will it start updating
                update_every = ema_update_every,          # how often to actually update, to save on compute (updates every 10th .update() call)
            )
            self.ema_model.to(self.accelerator.device)

        self.optimizer = Adam(self.model.model.parameters(), lr = lr, weight_decay=adam_weight_decay)
        self.dl = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
        self.model, self.optimizer, self.dl = self.accelerator.prepare(self.model, self.optimizer, self.dl)       
            

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save(self, path):
        if not self.is_main:
            return

        save_package = dict(
            model = self.accelerator.unwrap_model(self.model).state_dict(),
            ema_model = self.ema_model.state_dict(),
            optimizer = self.accelerator.unwrap_model(self.optimizer).state_dict(),
        )
        torch.save(save_package, str(self.checkpoints_folder / path))

    def load(self, path):
        if not self.is_main:
            return
        
        load_package = torch.load(path)        
        self.model.load_state_dict(load_package["model"])
        self.ema_model.load_state_dict(load_package["ema_model"])
        self.optimizer.load_state_dict(load_package["optimizer"])

    def log(self, *args, **kwargs):
        return self.accelerator.log(*args, **kwargs)

    def log_images(self, *args, **kwargs):
        return self.accelerator.log(*args, **kwargs)

    def sample(self, fname):
        eval_model = default(self.ema_model, self.model)
        with torch.no_grad():
            sampled = eval_model.sample(batch_size=self.num_samples, steps=self.ode_sample_steps)
      
        sampled = rearrange(sampled, '(row col) c h w -> c (row h) (col w)', row = self.num_sample_rows)
        sampled.clamp_(0., 1.)

        save_image(sampled, fname)
        return sampled

    def forward(self):

        dl = cycle(self.dl)

        progress_bar = tqdm(range(self.num_train_steps), desc="Steps", disable=not self.is_main)

        for step in range(1, self.num_train_steps+1):

            self.model.train()

            data = next(dl)
            loss, log_dict = self.model(data[0])

            self.accelerator.backward(loss)
            progress_bar.update(1)
            if self.is_main:
                progress_bar.set_postfix({"mse": log_dict['mse_loss'], "vgg": log_dict["lpips_loss"]})
                if step%100==0:
                    self.log(log_dict, step = step)


            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.is_main and self.use_ema:
                self.ema_model.update()

            self.accelerator.wait_for_everyone()

            if self.is_main:
                if step%self.save_results_every==0:
                    sampled = self.sample(fname=str(self.results_folder / f'{step}.png'))
                    # self.log_images(sampled, step = step)

                if step%self.checkpoint_every==0:
                    self.save(f'{step}.pt')

            self.accelerator.wait_for_everyone()

        print('training complete')

if __name__ == "__main__":

    train_dict = dict(
        exp_folder= "rectified_flow_2022/exp2-LPIPS_MSE",
        num_train_steps = 1000_000,
        batch_size = 256,
        save_results_every = 1000,
        checkpoint_every = 10000,
        num_samples = 16,
        ode_sample_steps = 100,
        unet_dim = 64,
        use_ema = True,
        ema_update_after_step = 100,
        ema_update_every = 10,
        ema_beta = 0.999,
        lr = 2e-4,
        adam_weight_decay = 0,
        clip_during_sampling = False,
        clip_flow_during_sampling = False,
        clip_flow_values = (-3.,3.)
    )

    trainer = Trainer(**train_dict)
    trainer()
