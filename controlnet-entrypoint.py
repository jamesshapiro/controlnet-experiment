#!/usr/bin/env python

from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image
import ulid

ulid_prefix = str(int(time.time()))
ulid_prefix = ulid_prefix[:8]

image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)

import cv2
from PIL import Image
import numpy as np

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)
img.save(f'output/canny_image.png')