import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_dims = 784
h_dims = 200
z_dims = 20
epochs = 10
batch_size = 32
lr = 0.005
CHECKPOINT = "../weightVAE/VAE.pth"
