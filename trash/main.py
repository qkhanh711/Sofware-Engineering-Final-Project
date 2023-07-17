from typing import Union
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import FastAPI, Request
from pydantic import BaseModel
from Generate import generate
from utils import convert2_
import os


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/style", StaticFiles(directory="style"), name="style")
app.mount("/Input_images", StaticFiles(directory="Input_images"), name="Input_images")
app.mount("/Generate_images", StaticFiles(directory="Generate_images"), name="Generate_images")
app.mount("/pretrained", StaticFiles(directory="pretrained"), name="pretrained")
templates = Jinja2Templates(directory="templates")

class ImageGenerationRequest(BaseModel):
    model: str 
    path_type: str
    path: str = None
    get_url: str = None
    prompt: str = None
    number: int = 1
    idx: int = 1

class MyImage(BaseModel):
    image: str

class Caption(BaseModel):
    cap: Union[str, None] = None

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/index", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/generateImage", response_class=JSONResponse)
async def generate_image(request: ImageGenerationRequest):
    if request.path_type == "local":
        get_path = request.path
    elif request.path_type == "url":
        get_path = request.get_url
    else:
        return {"message": "Invalid path type"}

    model, result = generate(
        convert2_(request.model),
        number=request.number,
        idx=request.idx,
        prompt=request.prompt,
        url=convert2_(get_path),
    )

    if model == "nlpconnect/vit-gpt2-image-captioning":
        generated_result = Caption(cap= " ".join(result))
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
    # http://127.0.0.1:8000/generateImage?models=nlpconnect_vit-gpt2-image-captioning&path=_home_nyanmaruk_Uni_Sofware-Engineering-Final-Project_pretrained_GFPGAN_inputs_upload_deptry.jpg&prompt=some_prompt