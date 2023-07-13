from typing import Union
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from Generate import generate
from utils import convert2_
import os
from sqlalchemy.ext.declarative import declarative_base


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/pretrained", StaticFiles(directory="pretrained"), name="pretrained")
templates = Jinja2Templates(directory="templates")

class ImageGenerationRequest(BaseModel):
    model: str 
    path: str = None
    prompt: str = None
    number: int = 1
    idx: int = 1

class MyImage(BaseModel):
    image: str

class Caption(BaseModel):
    cap: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generateImage", response_class=JSONResponse)
async def generate_image(request: ImageGenerationRequest):
    model, result = generate(
        convert2_(request.model),
        number=request.number,
        idx=request.idx,
        prompt=request.prompt,
        url=convert2_(request.path),
    )

    if model == "nlpconnect/vit-gpt2-image-captioning":
        generated_result = Caption(cap=result)
    else:
        generated_result = MyImage(image=result)

    if generated_result:
        return {"generated_result": generated_result}
    else:
        return {"message": "No generated result available"}

@app.get("/getImage")
async def get_image(path: str):
    image_path = os.path.join("static", path)
    if os.path.exists(image_path):
        return HTMLResponse(content=f'<img src="{image_path}" alt="Generated Image">')
    else:
        return HTMLResponse(content="Image not found", status_code=404)

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app)