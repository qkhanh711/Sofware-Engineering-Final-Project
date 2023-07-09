import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import config
from VAE import VAE
from torch import nn, optim
from torchvision.utils import save_image

# Load data MNIST
dataset = datasets.MNIST(root="/home/nyanmaruk/Uni/datasets/mnist", train=True, transform=transforms.ToTensor(), download = True)
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
    # tqdm(enumerate(train_loader)).set_postfix(loss=loss.item())
    print("epoch: " + str(epoch+1))
    print("loss: ", loss)
        

model = model.to("cpu")
def inference(digit, num_examples=1):
    """
    Generates (num_examples) of a particular digit.
    Specifically we extract an example of each digit,
    then after we have the mu, sigma representation for
    each digit we can sample from that.

    After we sample we can run the decoder part of the VAE
    and generate examples.
    """
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"Sofware-Engineering-Final_Project/Generate_images/VAE/{digit}_ex{example}.png")

for idx in range(10):
    inference(idx, num_examples=1)
