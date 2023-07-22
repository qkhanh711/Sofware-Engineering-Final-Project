from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import FastAPI, Request, Depends, Form, status
from Generate import generate
from utils import convert2_
import os

from modules.schemas import ImageGenerationRequest, Caption, MyImage
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from sqlalchemy.orm import Session

from modules import crud, models, schemas
from modules.database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/style", StaticFiles(directory="style"), name="style")
# app.mount("/model", StaticFiles(directory="model"), name="model")
app.mount("/Input_images", StaticFiles(directory="model/Input_images"), name="Input_images")
# app.mount("/model/Generate_images", StaticFiles(directory="model/Generate_images"), name="Generate_images")
# app.mount("/model/pretrained", StaticFiles(directory="model/pretrained"), name="pretrained")
templates = Jinja2Templates(directory="templates")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/index", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
def register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def register(request: Request,
                   username: str = Form(...), email: str = Form(...), hashed_password: str = Form(...), confirm_password: str = Form(...),
                   db: Session = Depends(get_db)):
    user = models.User(username=username, email=email, hashed_password=hashed_password)

    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    db_user = crud.get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    crud.create_user(db, user)

    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request,
                   username: str = Form(...), password: str = Form(...),
                   db: Session = Depends(get_db)):
    user = models.User(username=username, hashed_password=password)
    u = crud.get_user_by_username(db, username=user.username)
    if u != None:
        return templates.TemplateResponse("index.html", {"request": request, "user":u})
    else:
        return templates.TemplateResponse("login.html", {"request": request, "message": "Login failed! Try to correct username or password"})

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
    uvicorn.run(app)
    # http://127.0.0.1:8000/generateImage?models=nlpconnect_vit-gpt2-image-captioning&path=_home_nyanmaruk_Uni_Sofware-Engineering-Final-Project_pretrained_GFPGAN_inputs_upload_deptry.jpg&prompt=some_prompt