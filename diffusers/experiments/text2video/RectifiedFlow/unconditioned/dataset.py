import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

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

cifar10_trainset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/data', train=True, download=False, transform=transform)
cifar10_grayscale_trainset = GrayscaleCIFAR10(root='./datasets/cifar10/data', train=True, download=False, transform=transform)
mnist_trainset = torchvision.datasets.MNIST(root='./datasets/mnist/data', train=True, download=False, transform=transform)
fashion_mnist_trainset = torchvision.datasets.FashionMNIST(root='./datasets/fashion_mnist/data', train=True, download=False, transform=transform)

if __name__ == "__main__":
    dataiter = iter(DataLoader(cifar10_trainset, batch_size=16, shuffle=True, num_workers=0))
    images, labels = next(dataiter)

    print(f'Batch of images: {images.shape}, {images.dtype}, {images.max()}, {images.min()}')
    print(f'Batch of labels shape: {labels.shape}')

    images = rearrange(images, '(row col) c h w -> c (row h) (col w)', row = 4)
    save_image(images, "cifar10_gt.png")



