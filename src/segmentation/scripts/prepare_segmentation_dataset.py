import os
from PIL import Image

from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        images,
        masks,
        image_transform=None,
        mask_transform=None,
    ):
        """
        Custom dataset for image segmentation.

        Args:
            image_dir (str): Directory path of the input images.
            mask_dir (str): Directory path of the corresponding mask images.
            images (list): List of image filenames.
            masks (list): List of mask filenames.
            image_transform (callable, optional): Transformations to be applied to the input image. Default is None.
            mask_transform (callable, optional): Transformations to be applied to the mask image. Default is None.
        """

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Get the image and mask at the specified index.

        Args:
            index (int): Index of the sample.

        Returns:
            torch.Tensor: Transformed image.
            torch.Tensor: Transformed mask.
        """
        
        image_file = self.images[index]
        mask_file = self.masks[index]

        image_path = os.path.join(self.image_dir, image_file)
        mask_path = os.path.join(self.mask_dir, mask_file)

        # Open and resize the image and mask
        image = Image.open(image_path).resize((256, 256))
        mask = Image.open(mask_path).resize((256, 256))

        # Perform other necessary transformations
        if self.image_transform is not None:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)

        return image, mask
