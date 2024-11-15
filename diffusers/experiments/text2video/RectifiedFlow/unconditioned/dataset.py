import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/data', train=True, download=False, transform=transform)

if __name__ == "__main__":
    dataiter = iter(DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0))
    images, labels = next(dataiter)

    print(f'Batch of images shape: {images.shape}')
    print(f'Batch of labels shape: {labels.shape}')

