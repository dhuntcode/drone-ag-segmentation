import os
import shutil
from sklearn.model_selection import train_test_split

# Set the directories
image_dir = "src/segmentation/data/images"
mask_dir = "src/segmentation/data/masks"

# Get a list of all the file names
images = sorted(os.listdir(image_dir))
masks = sorted(os.listdir(mask_dir))

# Make sure that the image and mask lists contain the same files
common_files = set(images).intersection(set(masks))
images = sorted(list(common_files))
masks = sorted(list(common_files))

# Split the data into training and testing
train_images, test_images, train_masks, test_masks = train_test_split(
    images, masks, test_size=0.2, random_state=42
)

# Set the output directories
train_image_dir = "src/segmentation/data/train/images"
train_mask_dir = "src/segmentation/data/train/masks"
test_image_dir = "src/segmentation/data/test/images"
test_mask_dir = "src/segmentation/data/test/masks"

# Create the directories if they don't exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_mask_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(test_mask_dir, exist_ok=True)

# Move the files into the appropriate directories
for file in train_images:
    shutil.move(os.path.join(image_dir, file), os.path.join(train_image_dir, file))
for file in train_masks:
    shutil.move(os.path.join(mask_dir, file), os.path.join(train_mask_dir, file))
for file in test_images:
    shutil.move(os.path.join(image_dir, file), os.path.join(test_image_dir, file))
for file in test_masks:
    shutil.move(os.path.join(mask_dir, file), os.path.join(test_mask_dir, file))
