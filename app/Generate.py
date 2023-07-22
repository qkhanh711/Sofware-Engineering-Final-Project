from model.Generative_models.ProgressiveGAN import config as config_progressive
from model.Generative_models.ProgressiveGAN.utils import *
from model.Generative_models.ProgressiveGAN.model import *
from model.Generative_models.ProgressiveGAN.train import *

from model.Generative_models.simpleVAE import config as config_VAE
from model.Generative_models.simpleVAE.utils import *
from model.Generative_models.simpleVAE.VAE import *

import torch
from torch import autocast
import torch.optim as optim
from torchvision import transforms

from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from diffusers import StableDiffusionInstructPix2PixPipeline as StableDiffusionPipelineP2P

from utils import predict_step, download_image, CheckthenDown
import subprocess
import argparse
import PIL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_with_scratch_model(model_name, path, number = 1, idx = 0):
    if model_name == "ProGAN":
        gen = Generator(config_progressive.Z_DIM, config_progressive.IN_CHANNELS, img_channels=config_progressive.CHANNELS_IMG).to(config_progressive.DEVICE)
        opt_gen = optim.Adam(gen.parameters(), lr=config_progressive.LEARNING_RATE, betas=(0.0, 0.99))

        load_checkpoint('../../weightCelebA/generator.pth', gen, opt_gen, config_progressive.LEARNING_RATE)
        tensor, path_save = generate_examples(gen, 3, root_path= path, n = number)
        print(path_save)
    else:
        dataset = datasets.MNIST(root="../../datasets/mnist", train=True, transform=transforms.ToTensor(), download = True)
        model = VAE(config_VAE.in_dims, config_VAE.h_dims, config_VAE.z_dims).to(config_VAE.device)
        optimizer = optim.Adam(model.parameters(), lr=config_VAE.lr)
        
        load_checkpoint('../../weightVAE/VAE.pth', model, optimizer, config_VAE.lr)
        tensor, path_save = inference(dataset, model, idx, num_examples=number)
    print(f"Results saved to {path_save}")
    return path_save


def generate_with_pretrained_model(name, prompt, url = None):
    """
       url: URL links or path of image

       Name models : 'hakurei/waifu-diffusion'
       promt       :"1kid, aqua eyes, baseball cap, red hair, closed mouth, earrings, green background 
       , hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, blue shirt" 
    
       Name models : "runwayml/stable-diffusion-v1-5"
       promt       : "a photo of an astronaut riding a horse on mars"

       Name models : "timbrooks/instruct-pix2pix"
       promt       : "turn him into cyborg"
       """
    
    
    if name == "nlpconnect/vit-gpt2-image-captioning":
        if url.startswith("https://"):
            input_img = download_image(url, name)
            return predict_step(["model/Input_images/nlpconnect/vit-gpt2-image-captioning/input.png"])
        else:
            input_img = PIL.Image.open(url)
            input_img.save(f"model/Input_images/{name}/input.png")
            return predict_step([url])
    elif name == "GFPGAN":
        CheckthenDown(url,name)
        command = [
            "python",
            "pretrained/GFPGAN/inference_gfpgan.py",
            "-i",
            "Input_images/GFPGAN",
            "-o",
            "Generate_images/GFPGAN",
            "-v",
            "1.3",
            "-s",
            "2",
            "--bg_upsampler",
            "realesrgan"
        ]
        # subprocess.run(["rm", "-rf", "pretrained/GFPGAN/results"], cwd="Sofware-Engineering-Final-Project")
        subprocess.run(command, cwd="../app/model")
        # subprocess.run(["ls", "pretrained/GFPGAN/results/cmp"], cwd="../Sofware-Engineering-Final-Project/")
        path = "/model/Generate_images/GFPGAN/gfp.png"
        # path = "/Generate_images/GFPGAN/gfp_00.png"
        return path 
    else:
        if name == "timbrooks/instruct-pix2pix":
            df = StableDiffusionPipelineP2P
        else:
            df = StableDiffusionPipeline
        pipe = df.from_pretrained(
        name,
        torch_dtype=torch.float32,
        safety_checker=None
        ).to(device)
        with autocast("cuda"):
            if name == 'hakurei/waifu-diffusion':
                gen_image = pipe(prompt, guidance_scale=6)["images"][0]
            elif name == "runwayml/stable-diffusion-v1-5":
                gen_image = pipe(prompt).images[0] 
            elif name == "timbrooks/instruct-pix2pix":
                pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
                input_img = CheckthenDown(url,name)
                gen_image = pipe(prompt, image=input_img, num_inference_steps=10, image_guidance_scale=1).images[0]
            path = f"/model/Generative_images/{name}/pdf.png"
            gen_image = gen_image.resize((512,512))
            gen_image.save(f"model/Generate_images/{name}/pdf.png")
            return path 


def generate(model, number = 1, idx = 1, prompt = None, url = None):
    if model == "VAE" or model == "ProGAN":
        result = generate_with_scratch_model(model,path = '../app/model/Generate_images', number = number, idx = idx)
    else:
        result = generate_with_pretrained_model(model, prompt = prompt, url = url)
    return model,result
    

if __name__ == '__main__':
    # url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
    # model, result   = generate("ProGAN", number = 1, idx = 1, prompt = None, url = None)
    # model,result = generate(convert2_("nlpconnect_vit-gpt2-image-captioning"),number=1, idx = 1
    #                     ,url = convert2_("_home_nyanmaruk_Uni_Sofware-Engineering-Final-Project_pretrained_GFPGAN_inputs_upload_deptry.jpg"))
    parser = argparse.ArgumentParser()

    parser.add_argument("--models", type=str,default= "ProGAN",help="Model name")
    parser.add_argument("--number", type=int, default=1, help="Number of images")
    parser.add_argument("--idx", type=int, default=0, help="Index")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt")
    parser.add_argument("--url", type=str, default=None, help="URL")

    args = parser.parse_args()

    model = args.models
    number = args.number
    idx = args.idx
    prompt = args.prompt
    url = args.url

    name, result = generate(model, number=number, idx=idx, prompt=prompt, url=url)
    print(result)

