import torch
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from PIL import Image
from segmentation.models.unet import UNet
import os
import matplotlib.pyplot as plt


def transform_image(image_path):
    """
    Transform the image using the specified transformations.

    Args:
        image_path (str): Path to the input image.

    Returns:
        torch.Tensor: Transformed image tensor.
    """

    image_transform = Compose(
        [
            Resize((256, 256)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path)
    return image_transform(image).unsqueeze(0)  # add an extra dimension


def predict(image_path, model_path):
    """
    Perform prediction on the input image using the trained model.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the trained model checkpoint.

    Returns:
        torch.Tensor: Predicted output tensor.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 5  # number of classes
    model = UNet(n_class=num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    image = transform_image(image_path)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)

    return output


image_directory = "src/segmentation/data/test/images"

# Load the model
model_path = "src/segmentation/models/saved_checkpoints/best_model.pth"

os.makedirs("src/segmentation/data/predict", exist_ok=True)

# Loop through the images in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(".png"):  # add any other image types you need
        image_path = os.path.join(image_directory, filename)

        # Make a prediction
        output = predict(image_path, model_path)

        # Process the output tensor to create a segmentation map.
        # The following line assumes that the output tensor is 4D and takes the argmax over the channel dimension.
        segmentation_map = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

        # Display the original image
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(Image.open(image_path))
        plt.title("Original Image")

        # Display the segmentation map
        plt.subplot(1, 2, 2)
        plt.imshow(segmentation_map)
        plt.title("Segmentation Map")

        # Save the figure
        plt.savefig(f"src/segmentation/data/predict/{filename}_output.png")
