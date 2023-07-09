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
from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import torch

import cv2
from PIL import Image
import numpy as np

def get_ulid():
    return str(ulid.new())

def main():
    pass

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
        "lllyasviel/sd-controlnet-canny", 
        # torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        safety_checker=None,
        controlnet=controlnet,
        # torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    prompt = ", best quality, ultra high res, (photorealistic:1.4)"
    prompt = [t + prompt for t in ["blue clouds"]*25]
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