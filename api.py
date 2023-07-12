from typing import Union

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from Generate import generate
from utils import convert2_

app = FastAPI()

class MyImage(BaseModel):
    image: str

class Caption(BaseModel):
    cap : str

generated_result = None

@app.post("/generateImage")
def gen(models: str, path: str = None, prompt: str = None):
    model,result = generate(
        convert2_(models),
        number=1, 
        idx = 1, 
        prompt = prompt,
        url = convert2_(path)
    )
    
    images = MyImage(
        image=str(result)
    )

    caption = Caption(
        cap=str(result)
    )

    if model == "nlpconnect/vit-gpt2-image-captioning":
        generated_result = jsonable_encoder(caption)
    else:
        generated_result = jsonable_encoder(images)
        
    if generated_result:
        return {"success": True}
    else:
        return {"success": False}


@app.get("/getGeneratedResult")
def getGeneratedResult():
    global generated_result
    if generated_result:
        return {"generated_result": generated_result}
    else:
        return {"message": "No generated result available"}
    
# http://127.0.0.1:8000/generateImage?models=nlpconnect_vit-gpt2-image-captioning&path=_home_nyanmaruk_Uni_Sofware-Engineering-Final-Project_pretrained_GFPGAN_inputs_upload_deptry.jpg&prompt=some_prompt
