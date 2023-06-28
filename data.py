from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms


# TRANSFORMS
transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# DATALOADER

