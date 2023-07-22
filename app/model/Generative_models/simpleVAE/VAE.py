import torch
from torch import nn



class VAE(nn.Module):
    def __init__(self, in_dims, hd_dims , noise ):
        super().__init__()
        # EC
        self.img2hd = nn.Linear(in_dims, hd_dims)
        self.hd2mean = nn.Linear(hd_dims, noise)
        self.hd2std = nn.Linear(hd_dims, noise)
        #DC
        self.ns2hd = nn.Linear(noise, hd_dims)
        self.hd2img = nn.Linear(hd_dims, in_dims)
        self.relu = nn.ReLU()
        
    def encode(self, x):
        # h = nn.ReLU(self.img2hd(x))
        h = self.relu((self.img2hd(x)))
        mean = self.hd2mean(h)
        std = self.hd2std(h)
        return mean, std
    
    def decode(self, noise):
        # h = nn.ReLU(self.ns2hd(noise))
        h = self.relu((self.ns2hd(noise)))
        return torch.sigmoid(self.hd2img(h))
    
    
    def forward(self, x):
        mean,std = self.encode(x)
        eps = torch.randn_like(std)
        new_noise = mean + std * eps
        x_new = self.decode(new_noise)
        return x_new, mean, std
    