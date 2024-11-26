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
from dataset import *
from lr_schedulers import *
from torch.optim.lr_scheduler import ConstantLR

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class RectifiedFlow(Module):
    def __init__(self, 
                 model:dict|Module,
                 clip_values,
                 loss_fn, 
                 data_shape,
                 forward_time_sampling,
                 use_VAE,
                 vae_latent_norm_factor,
                 data_loss_factor,
                 flow_loss_factor,
                 image_loss_factor,
                 ):
        super().__init__()

        if isinstance(model, dict):
            cc,hh,ww = data_shape
            dim_mults = [1]
            for res in [2,4,8]:
                if hh%res==0 and ww%res==0:
                    dim_mults.append(res)
                else:
                    break
            model["dim_mults"] = tuple(dim_mults)
            if cc not in [1,3]:
                model["channels"] = cc
            model = Unet(**model)

        self.model = model
        if use_VAE:
            self.vae_latent_norm_factor = vae_latent_norm_factor
            from diffusers import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True)
            self.vae.requires_grad_(False)
            # self.vae.enable_tiling()
        else:
            self.vae = None

        self.noise_predict = False
        if "noise" in loss_fn.lower():
            self.noise_predict = True

        if loss_fn=="VGGLoss_MSE":
            self.loss_fn = VGGLoss_MSE()
        elif loss_fn=="VGGLossonData_MSEonFlow2":
            self.loss_fn = VGGLossonData_MSEonFlow2()
        elif loss_fn=="MSEData_MSEFlow_VAE":
            self.loss_fn = MSEData_MSEFlow_VAE()
        elif loss_fn=="MSEData_MSEFlow_L1img_VAE":
            self.loss_fn = MSEData_MSEFlow_L1img_VAE()
        elif loss_fn=="MSEData_MSEFlow_L1imgFewer_VAE":
            self.loss_fn = MSEData_MSEFlow_L1imgFewer_VAE()
        elif loss_fn=="MSEData_MSEFlow_L1imgFewerTiled_VAE":
            self.loss_fn = MSEData_MSEFlow_L1imgFewerTiled_VAE()
        elif loss_fn=="MSEData_MSENoise_VAE":
            self.loss_fn = MSEData_MSENoise_VAE()
        else:
            self.loss_fn = MyMSE()

        self.data_loss_factor = data_loss_factor
        self.flow_loss_factor = flow_loss_factor
        self.image_loss_factor = image_loss_factor

        # sampling
        self.odeint_kwargs = dict(atol = 1e-5, rtol = 1e-5, method = 'dopri5')
        self.data_shape = data_shape
        self.forward_time_sampling = forward_time_sampling

        # clipping for epsilon prediction
        self.clip_values = clip_values

        # normalizing fn
        self.data_normalize_fn = normalize_to_neg_one_to_one
        self.data_unnormalize_fn = unnormalize_to_zero_to_one

    @property
    def device(self):
        return next(self.model.parameters()).device

    def predict_flow(self, z, times, clamp=True):
        times = rearrange(times, '... -> (...)')
        if times.numel() == 1:
            times = repeat(times, '1 -> b', b = z.shape[0])        
        model_op = self.model(z, times)

        if not self.noise_predict:
            pred_flow = model_op
        else:
            times = append_dims(times,ndims=len(z.shape)-len(times.shape))
            if self.clip_values is not None:
                if clamp:
                    model_op = model_op.clamp(*self.clip_values)
            pred_flow = (z - model_op)/(times+1e-8)

        return pred_flow, model_op

    @torch.no_grad()
    def sample(self, batch_size = 1, steps = 16):

        def ode_fn(t, z):
            z = maybe_clip(z)
            flow, _ = self.predict_flow(z, t)
            flow = maybe_clip_flow(flow)
            return flow
        
        ### Non core 
        was_training = self.training
        self.eval()
        maybe_clip = identity
        maybe_clip_flow = identity
        
        ### core
        z0 = torch.randn((batch_size, *self.data_shape), device = self.device)
        times = torch.linspace(0., 1., steps, device = self.device)
        trajectory = odeint(ode_fn, z0, times, **self.odeint_kwargs)
        sampled_data = trajectory[-1]
        if self.vae is not None:
            with torch.no_grad():
                self.vae.eval()
                sampled_data = self.vae.decode(sampled_data/self.vae_latent_norm_factor).sample

        self.train(was_training)
        return self.data_unnormalize_fn(sampled_data).clamp(0,1)

    def forward(self,data):
        
        data = self.data_normalize_fn(data)
        if self.vae is not None:
            with torch.no_grad():
                self.vae.eval()
                gt_img = data.clone().detach()
                data = self.vae_latent_norm_factor * self.vae.encode(data).latent_dist.sample()

        z0 = torch.randn_like(data)
        flow = data - z0

        if self.forward_time_sampling=="logit_normal":
            times = torch.randn(data.shape[0], device = self.device)
            times = 1 / (1 + torch.exp(-times))
        else:
            times = torch.rand(data.shape[0], device = self.device)
        padded_times = append_dims(times, data.ndim - 1)

        z = padded_times * data + (1. - padded_times) * z0

        pred_flow, pred_noise = self.predict_flow(z, times, clamp=False)

        if self.vae:
            if self.noise_predict:
                data_loss, flow_loss, image_loss = self.loss_fn(pred_flow, flow, z, padded_times, data, vae_latent_norm_factor=self.vae_latent_norm_factor, pred_noise=pred_noise, gt_noise=z0)
            else:
                data_loss, flow_loss, image_loss = self.loss_fn(pred_flow, flow, z, padded_times, data, vae_latent_norm_factor=self.vae_latent_norm_factor, vae=self.vae, gt_img=gt_img)
        else:
            data_loss, flow_loss, image_loss = self.loss_fn(pred_flow, flow, z, padded_times, data)
        
        main_loss = self.data_loss_factor*data_loss + self.flow_loss_factor*flow_loss + self.image_loss_factor*image_loss
        log_dict = dict(
            data_loss = data_loss.item(),
            flow_loss = flow_loss.item(),
            image_loss = image_loss.item(),
            main_loss = main_loss.item(),
            gt_flow_max = flow.max().item(),
            gt_flow_min = flow.min().item()
        )
        return main_loss, log_dict



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
        clip_values = (-3.,3.),
        loss_fn: str = "VGGLossonData_MSEonFlow",
        data_shape = (3,32,32),
        dataset="mnist",
        forward_time_sampling="logit_normal",
        lr_scheduler = None,
        increase_iters=1000, 
        base_lr=1e-8, 
        max_lr=5e-4,
        use_VAE = True,
        vae_latent_norm_factor = 0.18,
        data_loss_factor = 0.5,
        flow_loss_factor = 0.5,
        image_loss_factor = 0,
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

        self.model = RectifiedFlow(dict(dim = unet_dim), clip_values, loss_fn, data_shape, forward_time_sampling, use_VAE, vae_latent_norm_factor, data_loss_factor, flow_loss_factor, image_loss_factor)

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

        if lr_scheduler is None:
            lr = lr
        else:
            lr = base_lr

        self.optimizer = Adam(self.model.model.parameters(), lr=lr, weight_decay=adam_weight_decay)
        
        if dataset == "cifar10":
            self.dl = DataLoader(cifar10_trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
        elif dataset == "mnist":
            self.dl = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
        elif dataset == "fashion_mnist":
            self.dl = DataLoader(fashion_mnist_trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
        elif dataset == "cifar10_grayscale":
            self.dl = DataLoader(cifar10_grayscale_trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
        elif dataset == "celebA":
            self.dl = DataLoader(celeba_trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
        
        if lr_scheduler == "LinearDecayWithWarmup":        
            self.scheduler = LinearDecayWithWarmup(optimizer=self.optimizer, total_iters=num_train_steps, increase_iters=increase_iters, base_lr=base_lr, max_lr=max_lr, last_epoch=-1)  
        else:
            self.scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=num_train_steps)
         
        self.model, self.optimizer, self.dl, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, self.dl, self.scheduler)       
            

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
        
        load_package = torch.load(self.checkpoints_folder / path)        
        self.model.load_state_dict(load_package["model"])
        if self.use_ema:
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

            torch.cuda.empty_cache()
            self.model.train()

            data = next(dl)
            loss, log_dict = self.model(data[0])

            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            progress_bar.update(1)
            if self.is_main:
                last_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({"data_loss": log_dict['data_loss'], "flow_loss": log_dict["flow_loss"], "image_loss": log_dict["image_loss"], "lr": last_lr})
                if step%100==0:
                    log_dict['lr'] = last_lr
                    self.log(log_dict, step = step)            

            if self.is_main and self.use_ema:
                self.ema_model.update()

            self.accelerator.wait_for_everyone()

            if self.is_main:
                if step%self.save_results_every==0:
                    sampled = self.sample(fname=str(self.results_folder / f'{step}.jpg'))
                    # self.log_images(sampled, step = step)

                if step%self.checkpoint_every==0:
                    self.save(f'{step}.pt')

            self.accelerator.wait_for_everyone()

        print('training complete')

if __name__ == "__main__":

    train_dict = dict(
        exp_folder= "rectified_flow_2022/Celebexp1-MSEData_MSENoise_VAE-logit_normal-lighter_model-VAE-LatentNorm0_18-LinearDecayWithWarmup-400k",
        num_train_steps = 400_000,
        batch_size = 256,
        save_results_every = 1000,
        checkpoint_every = 10_000,
        num_samples = 16,
        ode_sample_steps = 200,
        unet_dim = 64,
        use_ema = True,
        ema_update_after_step = 5000,
        ema_update_every = 10,
        ema_beta = 0.999,
        lr = 1e-4,
        adam_weight_decay = 0,
        clip_values = (-3.0, 3.0),
        loss_fn = "MSEData_MSENoise_VAE",
        data_shape = (4, 32, 32),
        dataset = "celebA",
        forward_time_sampling = "logit_normal",
        lr_scheduler = "LinearDecayWithWarmup",
        increase_iters = 40_000,
        base_lr = 1e-05,
        max_lr = 3e-4,
        use_VAE = True,
        vae_latent_norm_factor = 0.18,
        data_loss_factor=0.3,
        flow_loss_factor=0.7,
        image_loss_factor=0.0,
    )

    trainer = Trainer(**train_dict)
    trainer()
