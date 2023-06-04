""" Evaluation Script"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from segmentation.scripts.prepare_segmentation_dataset import SegmentationDataset 
from segmentation.models.unet import UNet  
from segmentation.utils.load_params import load_yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

params = load_yaml('params.yaml', 'test')

images = sorted(os.listdir(params.image_dir))
masks = sorted(os.listdir(params.mask_dir))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_transform = Compose([
    Resize((256, 256)),  
    ToTensor(),  
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = Compose([
    ToTensor()
])

dataset = SegmentationDataset(
    image_dir=params.image_dir, 
    mask_dir=params.mask_dir, 
    images=images, 
    masks=masks, 
    image_transform=image_transform,
    mask_transform=mask_transform
)

loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False)

num_classes = 5  
model = UNet(n_class=num_classes)
model.load_state_dict(torch.load(params.checkpoint_path))
model = model.to(device)

criterion = nn.CrossEntropyLoss()

model.eval()

total_loss = 0
total_samples = 0
y_true = []
y_pred = []

with torch.no_grad():
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        masks = torch.argmax(masks, dim=1)

        outputs = model(images)
        loss = criterion(outputs, masks)

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        y_true.extend(masks.view(-1).detach().cpu().numpy())
        y_pred.extend(torch.argmax(outputs, dim=1).view(-1).detach().cpu().numpy())


average_loss = total_loss / total_samples
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f'Average Loss: {average_loss:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
