from typing import Union

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

app = FastAPI()

class MyImage(BaseModel):
    image: str


@app.post("/generateImage")
def generate():
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


