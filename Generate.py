
import torch.optim as optim
from Generative_models.ProgressiveGAN import config as config_progressive
from Generative_models.ProgressiveGAN.utils import *
from Generative_models.ProgressiveGAN.model import *
from Generative_models.ProgressiveGAN.train import *

from Generative_models.simpleVAE import config as config_VAE
from Generative_models.simpleVAE.utils import *
from Generative_models.simpleVAE.VAE import *

def generate_with_model(model, path, number = 1, idx = None):
    if model == "ProGAN":
        gen = Generator(config_progressive.Z_DIM, config_progressive.IN_CHANNELS, img_channels=config_progressive.CHANNELS_IMG).to(config_progressive.DEVICE)
        opt_gen = optim.Adam(gen.parameters(), lr=config_progressive.LEARNING_RATE, betas=(0.0, 0.99))

        load_checkpoint('/home/nyanmaruk/Uni/weightCelebA/generator.pth', gen, opt_gen, config_progressive.LEARNING_RATE)
        generate_examples(gen, 3, root_path= path, n = number)
    else:
        dataset = datasets.MNIST(root="../datasets/mnist", train=True, transform=transforms.ToTensor(), download = True)
        model = VAE(config_VAE.in_dims, config_VAE.h_dims, config_VAE.z_dims).to(config_VAE.device)
        optimizer = optim.Adam(model.parameters(), lr=config_VAE.lr)
        
        load_checkpoint('/home/nyanmaruk/Uni/weightVAE/VAE.pth', model, optimizer, config_VAE.lr)
        inference(dataset, model, idx, num_examples=number)
    print(f"Results saved to {path}")

generate_with_model("ProGAN", path = '/home/nyanmaruk/Uni/Sofware-Engineering-Final_Project/Generate_images')