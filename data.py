from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import config
from math import log2



def get_loader(image_size, root_path):
    transform = transforms.Compose([     
        transforms.Resize((image_size, image_size)),                                                                              
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(config.CHANNELS_IMG)], [0.5 for _ in range(config.CHANNELS_IMG)],),]
    )
    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root_path, transform=transform)

    dataset = datasets.CIFAR10(root='data', train=True,
                                    download=True, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    return loader, dataset

if __name__ == '__main__':
    loader, dataset = get_loader(4 * 2 ** 1)