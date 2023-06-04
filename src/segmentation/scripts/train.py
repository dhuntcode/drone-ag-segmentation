import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from sklearn.model_selection import train_test_split
from segmentation.scripts.prepare_segmentation_dataset import SegmentationDataset 
from segmentation.models.unet import UNet  
from segmentation.utils.load_params import load_yaml

params = load_yaml('params.yaml', 'train')

images = sorted(os.listdir(params.image_dir))
masks = sorted(os.listdir(params.mask_dir))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
train_images, val_images, train_masks, val_masks = train_test_split(images, masks, test_size=0.2, random_state=42)

image_transform = Compose([
    Resize((256, 256)),  
    ToTensor(),  
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = Compose([
    ToTensor()
])


train_dataset = SegmentationDataset(
    image_dir=params.image_dir, 
    mask_dir=params.mask_dir, 
    images=train_images, 
    masks=train_masks, 
    image_transform=image_transform,
    mask_transform=mask_transform

)


val_dataset = SegmentationDataset(params.image_dir, params.mask_dir, val_images, val_masks, image_transform=image_transform, mask_transform=mask_transform)

train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False)

num_classes = 5  
model = UNet(n_class=num_classes)

model = model.to(device)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

best_val_loss = float('inf') 

# Training loop
for epoch in range(params.num_epochs):
    model.train()

    for images, masks in train_loader:
        images = images.to(device)

        # Change the shape of the masks
        masks = masks.to(device)
        masks = torch.argmax(masks, dim=1)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

    model.eval()

    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            masks = torch.argmax(masks, dim=1)

            outputs = model(images)
            val_loss += criterion(outputs, masks).item()  # accumulate the loss

    val_loss /= len(val_loader)  # get the average loss

    print(f'Epoch {epoch+1}, Train Loss: {loss.item()}, Val Loss: {val_loss}')


    # Save model weights if validation loss has decreased
    if val_loss < best_val_loss:
        print(f'Validation loss decreased ({best_val_loss} --> {val_loss}).  Saving model ...')
        torch.save(model.state_dict(), 'src/segmentation/models/saved_checkpoints/best_model.pth')
        best_val_loss = val_loss
