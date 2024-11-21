import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image
import os

import einx
from einops import einsum, reduce, rearrange, repeat
from torchvision.utils import save_image

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

class GrayscaleCIFAR10(CIFAR10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)        
        return img[0:1,:,:], target


class CelebA(Dataset):
    def __init__(self, transform):
        self.image_dir = "datasets/celeba/data/img_align_celeba/img_align_celeba"
        self.transform = transform
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        image = image.resize((256,256),Image.LANCZOS)
        image = self.transform(image)
        return image, image

cifar10_trainset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/data', train=True, download=False, transform=transform)
cifar10_grayscale_trainset = GrayscaleCIFAR10(root='./datasets/cifar10/data', train=True, download=False, transform=transform)
mnist_trainset = torchvision.datasets.MNIST(root='./datasets/mnist/data', train=True, download=False, transform=transform)
fashion_mnist_trainset = torchvision.datasets.FashionMNIST(root='./datasets/fashion_mnist/data', train=True, download=False, transform=transform)
celeba_trainset = CelebA(transform)

if __name__ == "__main__":
    dataiter = iter(DataLoader(celeba_trainset, batch_size=16, shuffle=True, num_workers=0))
    images = next(dataiter)[0]

    print(f'Batch of images: {images.shape}, {images.dtype}, {images.max()}, {images.min()}')
    # print(f'Batch of labels shape: {labels.shape}')

    images2 = rearrange(images, '(row col) c h w -> c (row h) (col w)', row = 4)
    save_image(images2, "celeb_gt.jpg")

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True).cuda()

    image_tensor = images*2 - 1
    with torch.no_grad():
        latents = vae.encode(image_tensor.cuda()).latent_dist.sample()
        print(latents.shape)
        decoded_image = vae.decode(latents).sample
        decoded_image = ((decoded_image+1)/2).clamp(0,1)
        decoded_image = rearrange(decoded_image, '(row col) c h w -> c (row h) (col w)', row = 4)
        save_image(decoded_image, "celeb_decoded.jpg")



    # import torch
    # from PIL import Image
    # from torchvision.transforms import ToTensor, ToPILImage

    # # Load an image
    # image = Image.open("datasets/celeba/data/img_align_celeba/img_align_celeba/000001.jpg")
    # image = image.resize((256,256),Image.LANCZOS)
    # image.save("gt.png")
    # image_tensor = ToTensor()(image).unsqueeze(0).cuda()  # Add batch dimension
    # image_tensor = image_tensor*2 - 1

    # with torch.no_grad():
    #     latents = vae.encode(image_tensor).latent_dist.sample()
    #     print(latents.shape)
    #     decoded_image = vae.decode(latents).sample
    #     decoded_image = ((decoded_image+1)/2).clamp(0,1)

    # # Convert tensor to PIL image
    # decoded_image_pil = ToPILImage()(decoded_image.squeeze(0))
    # decoded_image_pil.save("decoded.png")




