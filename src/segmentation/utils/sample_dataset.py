import os
import shutil

original_image_dir = "src/segmentation/data/raw_data/scout_point_images"
original_mask_dir = "src/segmentation/data/raw_data/agrohix"
new_image_dir = "src/segmentation/data/images"
new_mask_dir = "src/segmentation/data/masks"

os.makedirs(new_image_dir, exist_ok=True)
os.makedirs(new_mask_dir, exist_ok=True)

image_files = os.listdir(original_image_dir)
mask_files = os.listdir(original_mask_dir)

copied_files = 0

for mask_file in mask_files:
    # Extract the common part in the mask filename
    common_part = mask_file.split("_")[-1].split(".")[0]

    # Define the image file based on the common part in the filename
    image_file = f"scout_point_image_{common_part}.jpe"

    # Check if this image file exists in the image_files list
    if image_file in image_files:
        # Get the extension of the mask file
        mask_extension = os.path.splitext(mask_file)[1]

        # Generate a new filename for both the image and mask
        new_filename = f"image_mask_pair_{copied_files}{mask_extension}"

        # Copy the mask file and rename it
        new_mask_path = os.path.join(new_mask_dir, new_filename)
        shutil.copy(os.path.join(original_mask_dir, mask_file), new_mask_path)

        # Copy the image file and rename it
        new_image_path = os.path.join(new_image_dir, new_filename)
        shutil.copy(os.path.join(original_image_dir, image_file), new_image_path)

        # Increment the copied files counter
        copied_files += 1

    if copied_files >= 500:
        break

print(f"Copied {copied_files} image-mask pairs.")
