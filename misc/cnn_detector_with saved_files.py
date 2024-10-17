import os
import shutil
from sklearn.model_selection import train_test_split

# Paths to the directories containing the defect and non-defect images
defect_dir = r"D:\Rishabh Uikey\onward solar\onward 40mw\Clipped_defected_panels\combined_Data\tif_images\combined\defect"
non_defect_dir = r"D:\Rishabh Uikey\onward solar\onward 40mw\Clipped_defected_panels\combined_Data\tif_images\combined\no defect"

# Output directories for saving the split datasets
output_dir = r"D:\Rishabh Uikey\onward solar\onward 40mw\Clipped_defected_panels\combined_Data\tif_images\dataset"
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")

# Create the directories if they do not exist
for dir_path in [train_dir, val_dir, test_dir]:
    for class_name in ['defect', 'no defect']:
        os.makedirs(os.path.join(dir_path, class_name), exist_ok=True)

# Get all the image paths and their labels
defect_images = [(os.path.join(defect_dir, img), 'defect') for img in os.listdir(defect_dir) if img.endswith('.tif')]
non_defect_images = [(os.path.join(non_defect_dir, img), 'no defect') for img in os.listdir(non_defect_dir) if img.endswith('.tif')]

all_images = defect_images + non_defect_images

# Shuffle and split the data into train, validation, and test sets
train_val_images, test_images = train_test_split(all_images, test_size=0.1, stratify=[label for _, label in all_images], random_state=42)
train_images, val_images = train_test_split(train_val_images, test_size=0.1111, stratify=[label for _, label in train_val_images], random_state=42)  # 0.1111 to get 10% of the original dataset for validation

# Function to copy images to their respective directories
def copy_images(image_list, output_folder):
    for image_path, label in image_list:
        class_folder = os.path.join(output_folder, label)
        shutil.copy(image_path, class_folder)

# Copy images to the corresponding directories
copy_images(train_images, train_dir)
copy_images(val_images, val_dir)
copy_images(test_images, test_dir)

print("Data split completed successfully!")


# import os
# import torch
# from torch import nn, optim
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# from torch.cuda.amp import GradScaler, autocast
# from PIL import Image
# from sklearn.metrics import confusion_matrix, classification_report

# # Ensure the model is on the GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Enable cuDNN auto-tuner for potential speed-up
# torch.backends.cudnn.benchmark = True

# # Define custom dataset
# class PanelDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.image_paths = []
#         self.labels = []
#         self.transform = transform

#         # Traverse the directories and collect image paths and labels
#         for label_dir in os.listdir(root_dir):
#             label_path = os.path.join(root_dir, label_dir)
#             if os.path.isdir(label_path):
#                 label = 1 if label_dir == 'defect' else 0
#                 for img_name in os.listdir(label_path):
#                     if img_name.endswith(('.png', '.jpg', '.jpeg', '.tif')):
#                         self.image_paths.append(os.path.join(label_path, img_name))
#                         self.labels.append(label)

#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         label = self.labels[idx]
#         image = Image.open(img_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         return image, label

# # Define transforms for the dataset
# transform = transforms.Compose([
#     transforms.RandomResizedCrop(64, scale=(0.9, 1.0)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Define paths for the split data
# train_dir = r"D:\Rishabh Uikey\onward solar\onward 40mw\Clipped_defected_panels\dataset\train"
# val_dir = r"D:\Rishabh Uikey\onward solar\onward 40mw\Clipped_defected_panels\dataset\val"
# test_dir = r"D:\Rishabh Uikey\onward solar\onward 40mw\Clipped_defected_panels\dataset\test"

# # Create datasets and dataloaders
# train_dataset = PanelDataset(train_dir, transform=transform)
# val_dataset = PanelDataset(val_dir, transform=transform)
# test_dataset = PanelDataset(test_dir, transform=transform)

# # Print the number of images in each dataset once
# print(f"Number of training images: {len(train_dataset)}")
# print(f"Number of validation images: {len(val_dataset)}")
# print(f"Number of testing images: {len(test_dataset)}")

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

# # Define a simple 2-3 layer CNN model
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(128 * 8 * 8, 256)  # Assuming input images are 64x64
#         self.fc2 = nn.Linear(256, 2)
#         self.dropout = nn.Dropout(0.5)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = self.pool(self.relu(self.conv3(x)))
#         x = x.view(-1, 128 * 8 * 8)  # Flattening the tensor for the fully connected layer
#         x = self.dropout(self.relu(self.fc1(x)))
#         x = self.fc2(x)
#         return x

# model = SimpleCNN().to(device)

# # Define Focal Loss
# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2.0, alpha=0.5):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
    
#     def forward(self, inputs, targets):
#         BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
#         return F_loss.mean()

# criterion = FocalLoss()

# # Optimizer with weight decay for better generalization
# optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# # Learning rate scheduler to reduce learning rate on plateau
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# # Mixed Precision Training
# scaler = GradScaler()

# # Gradient Accumulation
# accumulation_steps = 4

# # Training loop with validation and performance tracking
# def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50):
#     best_val_accuracy = 0

#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
        
#         optimizer.zero_grad()
#         for i, (images, labels) in enumerate(train_loader):
#             images, labels = images.to(device), labels.to(device)
            
#             with autocast():
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
            
#             loss = loss / accumulation_steps
#             scaler.scale(loss).backward()
            
#             if (i + 1) % accumulation_steps == 0:
#                 scaler.step(optimizer)
#                 scaler.update()
#                 optimizer.zero_grad()

#             running_loss += loss.item()
            
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#         # Validation phase
#         model.eval()
#         val_correct = 0
#         val_total = 0
#         val_loss = 0.0
#         val_labels = []
#         val_preds = []
        
#         with torch.no_grad():
#             for images, labels in val_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs.data, 1)
                
#                 val_total += labels.size(0)
#                 val_correct += (predicted == labels).sum().item()
                
#                 val_labels.extend(labels.cpu().numpy())
#                 val_preds.extend(predicted.cpu().numpy())

#                 val_loss += criterion(outputs, labels).item()

#         val_accuracy = 100 * val_correct / val_total
#         val_loss /= len(val_loader)
#         scheduler.step(val_loss)
        
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {100 * correct / total:.2f}%, Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.4f}')

#         # Save the best model
#         if val_accuracy > best_val_accuracy:
#             best_val_accuracy = val_accuracy
#             torch.save(model.state_dict(), 'best_solar_panel_defect_classifier.pth')

#     return val_labels, val_preds

# if __name__ == "__main__":
#     # Train the model
#     val_labels, val_preds = train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50)

#     # Compute and print final confusion matrix and classification report
#     cm = confusion_matrix(val_labels, val_preds)
#     cr = classification_report(val_labels, val_preds, target_names=['Non-Defect', 'Defect'])
    
#     print("Final Confusion Matrix:")
#     print(cm)
#     print("Final Classification Report:")
#     print(cr)


# import torch
# from torch import nn
# from torchvision import transforms
# from torch.utils.data import DataLoader, Dataset
# from PIL import Image
# import os
# import json
# from sklearn.metrics import confusion_matrix, classification_report

# # Ensure the model is on the GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define custom dataset
# class PanelDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.image_paths = []
#         self.labels = []
#         self.transform = transform

#         # Traverse the root directory and store image paths and labels
#         for label, class_name in enumerate(['no defect', 'defect']):  # Ensure correct order
#             class_dir = os.path.join(root_dir, class_name)
#             for img_name in os.listdir(class_dir):
#                 img_path = os.path.join(class_dir, img_name)
#                 if os.path.isfile(img_path):  # Ensure it's a file, not a directory
#                     self.image_paths.append(img_path)
#                     self.labels.append(label)

#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         label = self.labels[idx]
#         image = Image.open(img_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         return image, label

# # Define the transform used during training (without random augmentations)
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),  # Ensure consistent resizing
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Path to validation set and model
# val_dir = r"D:\Rishabh Uikey\onward solar\onward 40mw\Clipped_defected_panels\dataset\val"
# model_path = r"C:\Users\LEGION\Downloads\best_solar_panel_defect_classifier_RESNET.pth"

# # Create validation dataset and dataloader
# val_dataset = PanelDataset(val_dir, transform=transform)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

# # Define the model (Make sure this matches exactly with the one used during training)
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(128 * 8 * 8, 256)  # Assuming input images are 64x64
#         self.fc2 = nn.Linear(256, 2)
#         self.dropout = nn.Dropout(0.5)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = self.pool(self.relu(self.conv3(x)))
#         x = x.view(-1, 128 * 8 * 8)  # Flattening the tensor for the fully connected layer
#         x = self.dropout(self.relu(self.fc1(x)))
#         x = self.fc2(x)
#         return x

# if __name__ == "__main__":
#     # Load the trained model
#     model = SimpleCNN().to(device)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()  # Set the model to evaluation mode

#     # Perform inference on the validation set
#     val_labels = []
#     val_preds = []

#     with torch.no_grad():  # Ensure gradients are not calculated
#         for images, labels in val_loader:
#             images = images.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             val_labels.extend(labels.cpu().numpy())
#             val_preds.extend(predicted.cpu().numpy())

#     # Compute and print final confusion matrix and classification report
#     cm = confusion_matrix(val_labels, val_preds)
#     cr = classification_report(val_labels, val_preds, target_names=['Non-Defect', 'Defect'])

#     print("Final Confusion Matrix:")
#     print(cm)
#     print("Final Classification Report:")
#     print(cr)

#     # Optionally save the predictions
#     output_results = []
#     for i, path in enumerate(val_dataset.image_paths):
#         result = {
#             "image_path": path,
#             "predicted_label": "Defect" if val_preds[i] == 1 else "Non-Defect",
#             "actual_label": "Defect" if val_labels[i] == 1 else "Non-Defect"
#         }
#         output_results.append(result)

#     with open("validation_results.json", "w") as f:
#         json.dump(output_results, f, indent=4)

#     print(f"Inference completed and results saved to validation_results.json.")


##clipping panel
# import os
# import geopandas as gpd
# import rasterio
# from rasterio.mask import mask
# from PIL import Image

# def clip_panels_and_save(ortho_path, geojson_path, output_dir):
#     # Ensure output directory exists
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     # Open the orthomosaic and clip panels using the geometries from GeoJSON
#     annotations = gpd.read_file(geojson_path)
#     with rasterio.open(ortho_path) as src:
#         for i, row in annotations.iterrows():
#             geom = row.geometry
#             if geom.is_valid and not geom.is_empty:
#                 out_image, out_transform = mask(src, [geom], crop=True)
#                 img_path = os.path.join(output_dir, f"panel_{i}.tif")
#                 Image.fromarray(out_image.squeeze()).save(img_path)
#                 print(f"Saved clipped panel {i} to {img_path}")

# if __name__ == "__main__":
#     ortho_path = r"D:\Rishabh Uikey\onward solar\onward 20mw\ortho\kml1\Thermal_Ortho_KML1Clipped.tif"
#     geojson_path = r"D:\Rishabh Uikey\onward solar\onward 40mw\new_model-3\Thermal_Ortho_KML1Clipped\Thermal_Ortho_KML1Clipped_annotations.geojson"
#     output_dir = r"D:\Rishabh Uikey\onward solar\clipped_panels"
    
#     # Clip panels and save to directory
#     clip_panels_and_save(ortho_path, geojson_path, output_dir)


## main working code 
# import torch
# from torch import nn
# from torchvision import transforms, models
# from torch.utils.data import DataLoader, Dataset
# from torchvision.models import ResNet18_Weights
# from PIL import Image
# import os
# from sklearn.metrics import confusion_matrix, classification_report
# import numpy as np
# import random

# # Set random seeds for reproducibility
# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# set_seed(42)


# # Ensure the model is on the GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define custom dataset
# class PanelDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.image_paths = []
#         self.labels = []
#         self.transform = transform

#         # Traverse the directories and collect image paths and labels
#         for label_dir in os.listdir(root_dir):
#             label_path = os.path.join(root_dir, label_dir)
#             if os.path.isdir(label_path):
#                 label = 1 if label_dir == 'defect' else 0
#                 for img_name in os.listdir(label_path):
#                     if img_name.endswith(('.png', '.jpg', '.jpeg', '.tif')):
#                         self.image_paths.append(os.path.join(label_path, img_name))
#                         self.labels.append(label)

#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         label = self.labels[idx]
#         image = Image.open(img_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         return image, label

# # Define transforms for the validation dataset
# val_transform = transforms.Compose([
#     transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # Resized to 224x224 to fit ResNet18
#     # transforms.RandomHorizontalFlip(),
#     # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])


# # Path to validation set and model
# val_dir = r"D:\Rishabh Uikey\onward solar\onward 40mw\new_model-3\Thermal_Ortho_KML1Clipped\clipped_panels"
# model_path = r"C:\Users\LEGION\Downloads\best_solar_panel_defect_classifier_RESNET.pth"

# # Create the validation dataset and dataloader
# val_dataset = PanelDataset(val_dir, transform=val_transform)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

# # Load the pre-trained ResNet18 model
# model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# # Replace the final fully connected layer with a custom layer
# model.fc = nn.Sequential(
#     nn.Linear(model.fc.in_features, 256),
#     nn.ReLU(),
#     nn.Dropout(0.5),
#     nn.Linear(256, 2)  # 2 output classes: 'Non-Defect' and 'Defect'
# )

# model = model.to(device)

# # Load the trained model weights
# model.load_state_dict(torch.load(model_path))
# model.eval()  # Set the model to evaluation mode

# # Define the loss criterion
# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2.0, alpha=3.0):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
    
#     def forward(self, inputs, targets):
#         BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
#         return F_loss.mean()

# criterion = FocalLoss()

# # Function to evaluate the model on the validation set
# def evaluate(model, val_loader, criterion):
#     model.eval()
#     val_correct = 0
#     val_total = 0
#     val_loss = 0.0
#     val_labels = []
#     val_preds = []

#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)

#             val_total += labels.size(0)
#             val_correct += (predicted == labels).sum().item()

#             val_labels.extend(labels.cpu().numpy())
#             val_preds.extend(predicted.cpu().numpy())

#             val_loss += criterion(outputs, labels).item()

#     val_accuracy = 100 * val_correct / val_total
#     val_loss /= len(val_loader)

#     return val_accuracy, val_loss, val_labels, val_preds

# if __name__ == "__main__":
#     # Evaluate the model
#     val_accuracy, val_loss, val_labels, val_preds = evaluate(model, val_loader, criterion)

#     # Compute and print the final confusion matrix and classification report
#     cm = confusion_matrix(val_labels, val_preds)
#     cr = classification_report(val_labels, val_preds, target_names=['Non-Defect', 'Defect'])

#     print("Final Confusion Matrix:")
#     print(cm)
#     print("Final Classification Report:")
#     print(cr)
#     print(f"Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.4f}")


# import torch
# from torch import nn
# from torchvision import transforms, models
# from torch.utils.data import DataLoader, Dataset
# from torchvision.models import ResNet18_Weights
# from PIL import Image
# import os
# import rasterio
# import json
# import numpy as np
# import random
# from geojson import Feature, FeatureCollection, dump
# from shapely.geometry import shape, mapping
# import shutil  # For deleting the temporary directory

# # Set random seeds for reproducibility
# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# set_seed(42)

# # Ensure the model is on the GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Focal Loss implementation
# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2.0, alpha=3.0):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
    
#     def forward(self, inputs, targets):
#         BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
#         return F_loss.mean()

# # Function to clip panels from the orthomosaic and save them temporarily with polygon info
# def clip_and_save_panels(ortho_path, geojson_path, temp_dir):
#     no_defect_dir = os.path.join(temp_dir, "no_defect")
#     if not os.path.exists(no_defect_dir):
#         os.makedirs(no_defect_dir)

#     annotations = []
#     with rasterio.open(ortho_path) as src:
#         with open(geojson_path) as f:
#             geojson = json.load(f)
#             crs = geojson.get("crs")  # Extract CRS from the GeoJSON if available
#             for i, feature in enumerate(geojson['features']):
#                 geom = shape(feature['geometry'])
#                 if geom.is_valid:
#                     coords = geom.bounds  # Get bounding box of the polygon
#                     window = rasterio.windows.from_bounds(*coords, transform=src.transform)
#                     clip = src.read(window=window, indexes=1)  # Read the first (and only) band

#                     # Define the path for the TIFF image
#                     img_path = os.path.join(no_defect_dir, f"panel_{i}.tif")

#                     # Save the clipped image as a GeoTIFF
#                     profile = src.profile
#                     profile.update({
#                         "driver": "GTiff",
#                         "height": clip.shape[0],
#                         "width": clip.shape[1],
#                         "transform": rasterio.windows.transform(window, src.transform)
#                     })

#                     with rasterio.open(img_path, "w", **profile) as dst:
#                         dst.write(clip, 1)

#                     # Save the image path and the original geometry
#                     annotations.append((img_path, geom))

#     return annotations, crs

# # Define custom dataset that includes polygon geometries
# class PanelDataset(Dataset):
#     def __init__(self, annotations, transform=None):
#         self.annotations = annotations
#         self.transform = transform

#     def __len__(self):
#         return len(self.annotations)
    
#     def __getitem__(self, idx):
#         img_path, geom = self.annotations[idx]
#         with rasterio.open(img_path) as src:
#             image = src.read(1)  # Read the first band
#             image = Image.fromarray(image).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         return image, geom, img_path

# # Define transforms for the validation dataset
# val_transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resized to 224x224 to fit ResNet18
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Function to run inference and generate predictions sequentially with thresholding
# def evaluate_and_predict(model, dataset, threshold):
#     model.eval()
#     predictions = []
#     img_paths = []
#     geoms = []

#     with torch.no_grad():
#         for i in range(len(dataset)):
#             image, geom, img_path = dataset[i]
#             image = image.unsqueeze(0).to(device)  # Add batch dimension
#             output = model(image)
#             probabilities = nn.functional.softmax(output, dim=1)
#             defect_prob = probabilities[0][1].item()

#             # Apply custom threshold for defect classification
#             if defect_prob >= threshold:
#                 predicted = 1  # Defect
#             else:
#                 predicted = 0  # Non-Defect

#             predictions.append(predicted)
#             img_paths.append(img_path)
#             geoms.append(geom)

#     return predictions, img_paths, geoms

# # Function to create GeoJSON features based on predictions
# def create_geojson_features(img_paths, predictions, geoms):
#     defect_features = []
#     non_defect_features = []
#     for img_path, pred, geom in zip(img_paths, predictions, geoms):
#         label = "Defect" if pred == 1 else "Non-Defect"
#         feature = Feature(geometry=mapping(geom), properties={"prediction": label, "image": img_path})
#         if label == "Defect":
#             defect_features.append(feature)
#         else:
#             non_defect_features.append(feature)
#     return defect_features, non_defect_features

# # Function to save predictions to GeoJSON
# def save_predictions_to_geojson(features, output_path, crs=None):
#     feature_collection = FeatureCollection(features)
#     if crs:
#         feature_collection['crs'] = crs  # Include CRS information in the GeoJSON
#     with open(output_path, 'w') as f:
#         dump(feature_collection, f)

# if __name__ == "__main__":
#     # Paths
#     ortho_path = r"D:\Rishabh Uikey\onward solar\onward 40mw\new_model-3\KML6_THERMAL1_32643\KML6_THERMAL1_32643_reproj_raster.tif"
#     geojson_path = r"D:\Rishabh Uikey\onward solar\onward 40mw\new_model-3\KML6_THERMAL1_32643\KML6_THERMAL1_32643_annotations.geojson"
#     temp_dir = r"D:\Rishabh Uikey\onward solar\temp_clipped_panels"
#     defect_output_geojson_path = r"D:\Rishabh Uikey\onward solar\onward 40mw\new_model-3\KML6_THERMAL1_32643\KML6_THERMAL1_32643_defect_predictions.geojson"
#     non_defect_output_geojson_path = r"D:\Rishabh Uikey\onward solar\onward 40mw\new_model-3\KML6_THERMAL1_32643\KML6_THERMAL1_32643_non_defect_predictions.geojson"
    
#     # Clip and save panels from the orthomosaic to the "no_defect" folder
#     annotations, crs = clip_and_save_panels(ortho_path, geojson_path, temp_dir)

#     # Create the validation dataset
#     val_dataset = PanelDataset(annotations, transform=val_transform)

#     # Load the pre-trained ResNet18 model
#     model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

#     # Replace the final fully connected layer with a custom layer
#     model.fc = nn.Sequential(
#         nn.Linear(model.fc.in_features, 256),
#         nn.ReLU(),
#         nn.Dropout(0.5),
#         nn.Linear(256, 2)  # 2 output classes: 'Non-Defect' and 'Defect'
#     )

#     model = model.to(device)

#     # Load the trained model weights
#     model_path = r"D:\Rishabh Uikey\best_solar_panel_defect_classifier_RESNET.pth"
#     model.load_state_dict(torch.load(model_path))
#     model.eval()  # Set the model to evaluation mode

#     # Run inference and predict with a custom threshold
#     threshold = 0.1 # Adjust this threshold as needed
#     val_preds, val_img_paths, val_geoms = evaluate_and_predict(model, val_dataset, threshold=threshold)

#     # Create GeoJSON features
#     defect_features, non_defect_features = create_geojson_features(val_img_paths, val_preds, val_geoms)

#     # Save the features to two separate GeoJSON files with CRS
#     save_predictions_to_geojson(defect_features, defect_output_geojson_path, crs)
#     save_predictions_to_geojson(non_defect_features, non_defect_output_geojson_path, crs)

#     # Clean up: Delete the temporary directory
#     shutil.rmtree(temp_dir)

#     print(f"Defect predictions saved to {defect_output_geojson_path}")
#     print(f"Non-defect predictions saved to {non_defect_output_geojson_path}")
#     print(f"Temporary directory {temp_dir} deleted.")
