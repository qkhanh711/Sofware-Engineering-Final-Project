from typing import Union

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from Generate import generate_with_pretrained_model, generate_with_scratch_model, generate

app = FastAPI()

class MyImage(BaseModel):
    image: str

class Caption(BaseModel):
    cap : str


@app.post("/generateImage/{models}")
def gen(models: str):
    model,result = generate(models,number=1, idx = 1)
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


