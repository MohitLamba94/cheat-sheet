from main import CONFIG, DiffusionReverseProcess

import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math

def generate(cfg):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    drp = DiffusionReverseProcess()
    
    model = torch.load(cfg.model_path).to(device)
    model.eval()
    
    xt = torch.randn(1, cfg.in_channels, cfg.img_size, cfg.img_size).to(device)
    
    # Denoise step by step by going backward.
    with torch.no_grad():
        for t in reversed(range(cfg.num_timesteps)):
            noise_pred = model(xt, torch.as_tensor(t).unsqueeze(0).to(device))
            xt, x0 = drp.sample_prev_timestep(xt, noise_pred, torch.as_tensor(t).to(device))

    xt = torch.clamp(xt, -1., 1.).detach().cpu()
    xt = (xt + 1) / 2
    
    return xt

cfg = CONFIG()

generated_imgs = []
for i in tqdm(range(cfg.num_img_to_generate)):
    xt = generate(cfg)
    xt = 255 * xt[0][0].cpu().numpy()
    generated_imgs.append(xt.astype(np.uint8))

def create_image_grid(images, grid_size):
    """
    Create a grid of images and save as a single image file.
    
    Parameters:
    images (list of np.array): List of grayscale images in numpy array format.
    grid_size (tuple): Tuple specifying the grid size (rows, cols).
    
    Returns:
    None
    """
    rows, cols = grid_size
    assert len(images) == rows * cols, "Number of images should match the grid size"
    
    # Get the shape of the individual images
    img_height, img_width = images[0].shape
    
    # Create an empty array to hold the grid
    grid_image = np.zeros((rows * img_height, cols * img_width), dtype=np.uint8)
    
    # Populate the grid with images
    for i in range(rows):
        for j in range(cols):
            grid_image[i*img_height:(i+1)*img_height, j*img_width:(j+1)*img_width] = images[i*cols + j]
    
    # Save the grid image
    plt.imsave('image_grid.png', grid_image, cmap='gray')

create_image_grid(generated_imgs,(int(math.sqrt(cfg.num_img_to_generate)), int(math.sqrt(cfg.num_img_to_generate))))