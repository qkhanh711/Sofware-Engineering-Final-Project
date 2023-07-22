import argparse
import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

try: 
    import config
    from utils import *
    from VAE import VAE
except: 
    from model.Generative_models.simpleVAE import config
    from model.Generative_models.simpleVAE.utils import *
    from model.Generative_models.simpleVAE.VAE import *
from torch import nn, optim

# Create argument parser
parser = argparse.ArgumentParser(description='Simple VAE Training')
parser.add_argument('--in_dims', type=int, default=784, help='Input dimensions')
parser.add_argument('--h_dims', type=int, default=200, help='Hidden dimensions')
parser.add_argument('--z_dims', type=int, default=20, help='Latent dimensions')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
args = parser.parse_args()

# Load data MNIST
dataset = datasets.MNIST(root="../datasets/mnist", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

model = VAE(args.in_dims, args.h_dims, args.z_dims).to(config.device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.BCELoss(reduction="sum")

for epoch in range(args.epochs):
    for i, (x, _) in tqdm(enumerate(train_loader)):
        x = x.to(config.device).view(x.shape[0], args.in_dims)
        x_reconstructed, mu, sigma = model(x)

        # Loss
        reconstruction_loss = loss_fn(x_reconstructed, x)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

        loss = reconstruction_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    save_checkpoint(model, optimizer, filename=config.CHECKPOINT)
    print("epoch:", epoch + 1)
    print("loss:", loss)
