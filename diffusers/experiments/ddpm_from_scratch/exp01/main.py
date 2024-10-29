from network import Unet
from dataloader import CustomMnistDataset

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

class noise_schduler:
    def __init__(self, 
                 num_time_steps = 1000, 
                 beta_start = 1e-4, 
                 beta_end = 0.02,
                 device = "cuda"
                ):
        
        self.betas = torch.linspace(beta_start, beta_end, num_time_steps).to(device) #torch.Size([1000]
        self.alphas = 1 - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)

class DiffusionForwardProcess(noise_schduler):

    def __init__(self,):
        super(DiffusionForwardProcess, self).__init__()
        
    def add_noise(self, original, noise, t):
        
        r""" Adds noise to a batch of original images at time-step t.
        
        :param original: Input Image Tensor -> B x C x H x W
        :param noise: Random Noise Tensor sampled from Normal Dist N(0, 1) -> B x C x H x W
        :param t: timestep of the forward process of shape -> (B, )
        
        Note: time-step t may differ for each image inside the batch.
        
        """
        
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t]
        
        sqrt_alpha_bar_t = sqrt_alpha_bar_t[:, None, None, None]
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t[:, None, None, None]

        xt = (sqrt_alpha_bar_t * original) + (sqrt_one_minus_alpha_bar_t * noise)
        return xt
# print("Forward Testing ------")
# x0 = torch.randn(4, 1, 28, 28).cuda()
# noise = torch.randn(4, 1, 28, 28).cuda()
# t_steps = torch.randint(0, 1000, (4,)).cuda() 
# dfp = DiffusionForwardProcess()
# # print(dir(dfp))
# xt = dfp.add_noise(x0, noise, t_steps)
# print("xt.shape=",xt.shape)


    
class DiffusionReverseProcess(noise_schduler):

    def __init__(self,):
        super(DiffusionReverseProcess, self).__init__()
        
    def sample_prev_timestep(self, xt, noise_pred, t):
        
        r""" Sample x_(t-1) given x_t and noise predicted
             by model.
             
             :param xt: Image tensor at timestep t of shape -> 1 x C x H x W
             :param noise_pred: Noise Predicted by model of shape -> 1 x C x H x W
             :param t: Current time step

        """
        
        x0 = (xt - self.sqrt_one_minus_alpha_bars[t]*noise_pred)/self.sqrt_alpha_bars[t]
        x0 = torch.clamp(x0, -1., 1.) 
        
        # mean of x_(t-1)
        mean = (xt - (self.betas[t]/self.sqrt_one_minus_alpha_bars[t])*noise_pred)/self.sqrt_alphas[t] 
        
        if t == 0:
            return mean, x0        
        else:
            variance =  ((1 - self.alpha_bars[t-1]) / (1 - self.alpha_bars[t])) * self.betas[t]
            sigma = variance**0.5
            z = torch.randn(xt.shape).to(xt.device)            
            return mean + sigma * z, x0

# print("Reverse Testing -----")
# xt = torch.randn(1, 1, 28, 28).cuda()
# noise_pred = torch.randn(1, 1, 28, 28).cuda()
# t = torch.randint(0, 1000, (1,)).cuda() 
# drp = DiffusionReverseProcess()
# xt_1, x0 = drp.sample_prev_timestep(xt, noise_pred, t)
# print("x[t-1].shape=",xt_1.shape, "x0.shape=",x0.shape)

# print(f"Model Testing -----")
# model = Unet().cuda()
# x = torch.randn(4, 1, 32, 32).cuda()
# t = torch.randint(0, 10, (4,)).cuda()
# print(f"output.shape={model(x, t).shape}, input.shape={x.shape}, time.shape={t.shape}")


class CONFIG:
    model_path = 'ddpm_unet.pth'
    train_csv_path = '/path/to/MNIST/train.csv'
    test_csv_path = '/path/to/MNIST/test.csv'
    generated_csv_path = 'mnist_generated_data.csv'
    num_epochs = 50
    lr = 1e-4
    num_timesteps = 1000
    batch_size = 128
    img_size = 28
    in_channels = 1
    num_img_to_generate = 16

def train(cfg):
    
    # Dataset and Dataloader
    mnist_ds = CustomMnistDataset(cfg.train_csv_path)
    mnist_dl = DataLoader(mnist_ds, cfg.batch_size, shuffle=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')
    
    # Initiate Model
    model = Unet().to(device)
    
    # Initialize Optimizer and Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.MSELoss()
    
    # Diffusion Forward Process to add noise
    dfp = DiffusionForwardProcess()
    
    # Best Loss
    best_eval_loss = float('inf')
    
    # Train
    for epoch in range(cfg.num_epochs):
        
        # For Loss Tracking
        losses = []
        
        # Set model to train mode
        model.train()
        
        # Loop over dataloader
        for imgs in tqdm(mnist_dl):
            
            imgs = imgs.to(device)
            
            # Generate noise and timestamps
            noise = torch.randn_like(imgs).to(device)
            t = torch.randint(0, cfg.num_timesteps, (imgs.shape[0],)).to(device)
            
            # Add noise to the images using Forward Process
            noisy_imgs = dfp.add_noise(imgs, noise, t)
            
            # Avoid Gradient Accumulation
            optimizer.zero_grad()
            
            # Predict noise using U-net Model
            noise_pred = model(noisy_imgs, t)
            
            # Calculate Loss
            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            
            # Backprop + Update model params
            loss.backward()
            optimizer.step()
        
        # Mean Loss
        mean_epoch_loss = np.mean(losses)
        
        # Display
        print('Epoch:{} | Loss : {:.4f}'.format(
            epoch + 1,
            mean_epoch_loss,
        ))
        
        # Save based on train-loss
        if mean_epoch_loss < best_eval_loss:
            best_eval_loss = mean_epoch_loss
            torch.save(model, cfg.model_path)
            
    print(f'Done training.....')


if __name__ == "__main__":
    cfg = CONFIG()
    train(cfg)
