from typing import Union

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

import torch.optim as optim
from Generative_models.ProgressiveGAN import config
from Generative_models.ProgressiveGAN.utils import *
from Generative_models.ProgressiveGAN.model import *
from Generative_models.ProgressiveGAN.train import *


app = FastAPI()

gen = Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))

load_checkpoint('/home/nyanmaruk/Uni/weightCelebA/generator.pth', gen, opt_gen, config.LEARNING_RATE)

class MyImage(BaseModel):
    image: str


@app.post("/generateImage")
def generate():
    image = generate_examples(gen, 3, root_path= '/home/nyanmaruk/Uni/Sofware-Engineering-Final_Project/Generate_images', n = 1)     
    img = "/home/nyanmaruk/Uni/Sofware-Engineering-Final_Project/Generate_images/ProGAN/img_0.png"

    # ham luu anh vao database n
    images = MyImage(
        image=str(img)
    )
    return {"generated_images": jsonable_encoder(images)}

# @app.get("/home")
# async def main():
#     return {"messagge": "Hello world"}

@app.get("/uploadImage")
def uploadImage(upload_image: MyImage):
    print(upload_image.image)
    return {"uploadImage": jsonable_encoder(upload_image)}


