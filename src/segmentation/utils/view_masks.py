import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

mask_dir = "src/segmentation/data/train/masks"  # Replace with the directory where your class label masks are saved

# Get the list of mask files
mask_files = os.listdir(mask_dir)

# Loop over the mask files and display them
for mask_file in mask_files:
    mask_path = os.path.join(mask_dir, mask_file)

    # Load the mask image as a numpy array
    mask = np.array(Image.open(mask_path))


    # Display the mask image
    plt.imshow(mask, cmap="jet")  # You can change the colormap as needed
    plt.title(mask_file)
    plt.colorbar()
    plt.savefig(f"{mask_file}_saved.png")
