from PIL import Image
import os

# Make sure to adjust this to the path where your images are located
image_folder = 'output'

# Make sure the images are sorted in the order you want them to appear
image_files = sorted([image_file for image_file in os.listdir(image_folder) if image_file.endswith(".png") and 'grid' not in image_file])

# Open all the images
images = [Image.open(os.path.join(image_folder, image_file)) for image_file in image_files]

# Create the GIF
images[0].save('output.gif', save_all=True, append_images=images[1:], loop=0, duration=10)