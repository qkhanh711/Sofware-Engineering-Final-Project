from typing import Union

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

import torch.optim as optim

try:
    from Generative_models.ProgressiveGAN import config
    from Generative_models.ProgressiveGAN.utils import *
    from Generative_models.ProgressiveGAN.model import *
    from Generative_models.ProgressiveGAN.train import *
except:
    import config
    from utils import *
    from model import *
    from train import *
from scipy.stats import truncnorm
import torch

app = FastAPI()

gen = Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
dis = Discriminator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)

# initialize optimizers and scalers for FP16 training
opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))

load_checkpoint('/home/nyanmaruk/Uni/weightCelebA/critic.pth', gen, opt_gen, config.LEARNING_RATE)
generate_examples(gen, 32, root_path= 'Generate_images', n = 200)

class MyImage(BaseModel):
    image: str


@app.post("/generateImage")
def generate():
    images = generate_examples(gen, 3, root_path= 'Generat_images', n = 10)     

    # ham luu anh 

    # ham luu anh vao database n
    image = MyImage(
        image=str(images)
    )
    return {"generated_images": jsonable_encoder(image)}

# @app.get("/home")
# async def main():
#     return {"messagge": "Hello world"}

@app.get("/uploadImage")
def uploadImage(upload_image: MyImage):
    print(upload_image.image)
    return {"uploadImage": jsonable_encoder(upload_image)}


