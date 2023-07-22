from pydantic import BaseModel
from typing import Union


class ItemBase(BaseModel):
    title: str
    description: str | None = None


class ItemCreate(ItemBase):
    pass


class Item(ItemBase):
    id: int
    owner_id: int

    class Config:
        orm_mode = True


class UserBase(BaseModel):
    email: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int
    is_active: bool
    items: list[Item] = []

    class Config:
        orm_mode = True

class ImageGenerationRequest(BaseModel):
    model: str
    path: str = None
    prompt: str = None
    number: int = 1
    idx: int = 1

class MyImage(BaseModel):
    image: str

class Caption(BaseModel):
    cap: Union[str, None] = None