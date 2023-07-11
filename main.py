from typing import Union
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder

from pydantic import BaseModel

from Generate import generate
from utils import convert2_


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
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
    print(request)
    model, result = generate(
        convert2_(request.model),
        number=request.number,
        idx=request.idx,
        prompt=request.prompt,
        url=convert2_(request.path),
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
        return {"generated_result": generated_result}
    else:
        return {"message": "No generated result available"}

# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app)
