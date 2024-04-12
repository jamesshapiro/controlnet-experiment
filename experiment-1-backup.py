#!/usr/bin/env python
import argparse
import json
import os
import torch
import ulid
import subprocess
import sys
import time
import random
from torch import autocast
# from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import torch

import cv2
from PIL import Image
import numpy as np
#from IPython.display import display
WEIGHTS_DIR = "/home/huggingface/weights/zwx/800"
# WEIGHTS_DIR = "/home/huggingface/content/stable_diffusion_weights/zwx/800/"
# import run_trainer

model_path = WEIGHTS_DIR             # If you want to use previously trained model saved in gdrive, replace this with the full path of model in gdrive

# pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float16).to("cuda")
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# ulid_prefix = str(int(time.time()))
#ulid_prefix = ulid_prefix[:8]
#pipe.enable_xformers_memory_efficient_attention()
# g_cuda = None

#'pixarStyleModel_lora128.safetensors'

# g_cuda = torch.Generator(device='cuda')

# seed = torch.random.seed()
#seed = 52362 #@param {type:"number"}
# g_cuda.manual_seed(seed)

def get_ulid():
    return str(ulid.new())

def main():
    pass
    #prompt = "photo of zwx person as a cartoon" #@param {type:"string"}
    # prompt = args.prompt0
    #negative_prompt = "asian, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face, blurry, draft, grainy, facial hair, stubble, multiple faces" #@param {type:"string"}
    # negative_prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face, blurry, draft, grainy, facial hair, stubble, multiple faces" #@param {type:"string"}
    # num_samples = args.iters #@param {type:"number"}
    # guidance_scale = 7.5 #@param {type:"number"}
    # num_inference_steps = args.steps #@param {type:"number"}
    # height = args.height #@param {type:"number"}
    # width = args.width #@param {type:"number"}

    # prefix = prompt.replace(' ', '_')[:200]

    # with autocast("cuda"), torch.inference_mode():
    #     images = pipe(
    #         prompt,
    #         height=height,
    #         width=width,
    #         negative_prompt=negative_prompt,
    #         num_images_per_prompt=num_samples,
    #         num_inference_steps=num_inference_steps,
    #         guidance_scale=guidance_scale,
    #         generator=g_cuda
    #     ).images

    # for idx, img in enumerate(images):
    #     img.save(f'output/{prefix}_{ulid_prefix}_{idx+1}.png')


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        print(f'generating image #{i}')
        new_ulid = get_ulid()
        img.save(f'output/{new_ulid}.png')
        grid.paste(img, box=(i % cols * w, i // cols * h))
    grid.save(f'output/output_grid.png')
    return grid


if __name__ == "__main__":
    image = load_image(
        # "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
        # "https://stabilitygenius.com/photos/IMG_3423.png"
        # "https://stabilitygenius.com/photos/closeup_portrait_of_zwx_person_as_a_paladin,_wearing_brilliant_white_armor_and_a_crown,_fantasy_concept_art,_artstation_trending,_highly_detailed,_beautiful_landscape_in_the_background,_art_by_wlop,_g_fb6ad8e3_4.png"
        # "https://stabilitygenius.com/photos/IMG_1080.png"
        "https://pbs.twimg.com/profile_images/1267830833940815872/gN21yPbi_400x400.jpg"
    )
    print('loaded image')
    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    canny_image.save(f'output/canny_image.png')
    controlnet = ControlNetModel.from_pretrained(
        # "thibaud/controlnet-canny-sd21"
        "lllyasviel/sd-controlnet-canny", 
        # torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        # "stabilityai/stable-diffusion-2",
        "runwayml/stable-diffusion-v1-5",
        safety_checker=None,
        controlnet=controlnet,
        # torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    prompt = ", best quality, ultra high res, (photorealistic:1.4)"
    #prompt = [t + prompt for t in ["asian", "anime", "gorilla", "frenchman"]]
    prompt = [t + prompt for t in ["blue clouds pixel art"]*25]
    #prompt = [t + prompt for t in ["George Washington", "Thomas Jefferson", "Theodore Roosevelt", "Abraham Lincoln"]]
    #generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(len(prompt))]
    #my_random_int = random.randint(0,1_000_000)
    generator = [torch.Generator(device="cpu").manual_seed(random.randint(0,1_000_000)) for i in range(len(prompt))]
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    output = pipe(
        prompt,
        canny_image,
        negative_prompt=["paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans"] * len(prompt),
        generator=generator,
        num_inference_steps=28,
    )
    image_grid(output.images, 5, 5)
    #image_grid(output.images, 2, 2)
    main()