from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

import PIL
import requests
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

# print(predict_step(['/home/nyanmaruk/Uni/Sofware-Engineering-Final_Project/pretrained/GFPGAN/inputs/upload/deptry.jpg']))

def convert2_(path):
    converted_path = path.replace("_", "/")
    return str(converted_path)

def download_image(url, name):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    if url.startswith("https://"):
        path = save_input(image, name)
        print(f"Save image to {path}")
    return image

def save_input(image, name):
    image.save(f"model/Input_images/{name}/input.png")
    return f"model/Input_images/{name}/input.png"

def CheckthenDown(url,name):
    if url.startswith("https://"):
        input_img = download_image(url, name)
    else:
        input_img = PIL.Image.open(url)
        input_img.save(f"model/Input_images/{name}/input.png")
    return input_img