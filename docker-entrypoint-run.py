#!/usr/bin/env python
import argparse
import json
import os
import torch
import ulid
import subprocess
import sys
import time
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
#from IPython.display import display
WEIGHTS_DIR = "/home/huggingface/weights/zwx/800"
# WEIGHTS_DIR = "/home/huggingface/content/stable_diffusion_weights/zwx/800/"
# import run_trainer

model_path = WEIGHTS_DIR             # If you want to use previously trained model saved in gdrive, replace this with the full path of model in gdrive

pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
ulid_prefix = str(int(time.time()))
#ulid_prefix = ulid_prefix[:8]
#pipe.enable_xformers_memory_efficient_attention()
g_cuda = None

#'pixarStyleModel_lora128.safetensors'

g_cuda = torch.Generator(device='cuda')

seed = torch.random.seed()
#seed = 52362 #@param {type:"number"}
g_cuda.manual_seed(seed)

def main():
    #prompt = "photo of zwx person as a cartoon" #@param {type:"string"}
    prompt = args.prompt0
    #negative_prompt = "asian, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face, blurry, draft, grainy, facial hair, stubble, multiple faces" #@param {type:"string"}
    negative_prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face, blurry, draft, grainy, facial hair, stubble, multiple faces" #@param {type:"string"}
    num_samples = args.iters #@param {type:"number"}
    guidance_scale = 7.5 #@param {type:"number"}
    num_inference_steps = args.steps #@param {type:"number"}
    height = args.height #@param {type:"number"}
    width = args.width #@param {type:"number"}

    prefix = prompt.replace(' ', '_')[:200]

    with autocast("cuda"), torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=g_cuda
        ).images

    for idx, img in enumerate(images):
        img.save(f'output/{prefix}_{ulid_prefix}_{idx+1}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create images from a text prompt.")
    parser.add_argument(
        "prompt0",
        metavar="PROMPT",
        type=str,
        nargs="?",
        help="The prompt to render into an image",
    )
    parser.add_argument(
        "--height", type=int, nargs="?", default=512, help="Image height in pixels"
    )
    parser.add_argument(
        "--width", type=int, nargs="?", default=512, help="Image width in pixels"
    )
    parser.add_argument(
        "--iters",
        type=int,
        nargs="?",
        default=1,
        help="Number of times to run pipeline",
    )
    parser.add_argument(
        "--steps", type=int, nargs="?", default=50, help="Number of sampling steps"
    )
    args = parser.parse_args()
    print(f'{args=}')
    if args.prompt0 is not None:
        prompt = args.prompt0
        print(prompt)
    main()