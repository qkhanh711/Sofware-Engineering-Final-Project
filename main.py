from typing import Union

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from Generate import generate_with_pretrained_model, generate_with_scratch_model, generate
from utils import convert2_
app = FastAPI()

class MyImage(BaseModel):
    image: str

class Caption(BaseModel):
    cap : str


@app.post("/generateImage/{models}")
def gen(models: str):
    model,result = generate(str(convert2_(models)),number=1, idx = 1
                        ,url = "../Sofware-Engineering-Final_Project/pretrained/GFPGAN/inputs/upload/deptry.jpg")
    images = MyImage(
        image=str(result)
    )
    caption = Caption(
        cap = str(result)
    )
    if model == "nlpconnect/vit-gpt2-image-captioning":
        return {"generated_texts": jsonable_encoder(caption)}
    else:
        return {"generated_images": jsonable_encoder(images)}

@app.get("/uploadImage")
def uploadImage(upload_image: MyImage):
    print(upload_image.image)
    return {"uploadImage": jsonable_encoder(upload_image)}


