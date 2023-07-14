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
    from Generative_models.simpleVAE import config
    from Generative_models.simpleVAE.utils import *
    from Generative_models.simpleVAE.VAE import *
from torch import nn, optim

# Load data MNIST
dataset = datasets.MNIST(root="../datasets/mnist", train=True, transform=transforms.ToTensor(), download = True)
train_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle = True)

model = VAE(config.in_dims, config.h_dims, config.z_dims).to(config.device)
optimizer = optim.Adam(model.parameters(), lr=config.lr)
loss_fn = nn.BCELoss(reduction="sum")

# for i,(x, _) in enumerate(train_loader):
#     x = x.to(config.device).view(x.shape[0], config.in_dims)
# print(x.shape)
# print(x.type)

for epoch in range(config.epochs):
    for i, (x, _) in tqdm(enumerate(train_loader)):
    # for i, (x, _) in enumerate(train_loader):
        x = x.to(config.device).view(x.shape[0], config.in_dims)
        x_reconstructed, mu, sigma = model(x)

        # Loss
        reconstruction_loss = loss_fn(x_reconstructed, x)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

        loss = reconstruction_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    save_checkpoint(model, optimizer, filename=config.CHECKPOINT)
    # tqdm(enumerate(train_loader)).set_postfix(loss=loss.item())
    print("epoch: " + str(epoch+1))
    print("loss: ", loss)
