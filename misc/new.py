# # ## Training the model

import os
import torch
import random
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Ensure the model is on the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable cuDNN auto-tuner for potential speed-up
torch.backends.cudnn.benchmark = True

# Define custom dataset
class PanelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Traverse the directories and collect image paths and labels
        for label_dir in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_dir)
            if os.path.isdir(label_path):
                label = 1 if label_dir == 'defect' else 0
                for img_name in os.listdir(label_path):
                    if img_name.endswith(('.png', '.jpg', '.jpeg', '.tif')):
                        self.image_paths.append(os.path.join(label_path, img_name))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transforms for the dataset
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # Resized to 224x224 to fit ResNet18
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define paths for the split data
train_dir = r"D:\Rishabh Uikey\onward solar\onward 40mw\Clipped_defected_panels\combined_Data\tif_images\dataset\train"
val_dir = r"D:\Rishabh Uikey\onward solar\onward 40mw\Clipped_defected_panels\combined_Data\tif_images\dataset\val"
test_dir = r"D:\Rishabh Uikey\onward solar\onward 40mw\Clipped_defected_panels\combined_Data\tif_images\dataset\test"

# Create datasets and dataloaders
train_dataset = PanelDataset(train_dir, transform=transform)
val_dataset = PanelDataset(val_dir, transform=transform)
test_dataset = PanelDataset(test_dir, transform=transform)

# Print the number of images in each dataset
print(f"Number of training images: {len(train_dataset)}")
print(f"Number of validation images: {len(val_dataset)}")
print(f"Number of testing images: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

# Load pre-trained ResNet18 model with the updated method
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Replace the final fully connected layer with a custom layer
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2)  # 2 output classes: 'Non-Defect' and 'Defect'
)

model = model.to(device)

# Define Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=3.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

criterion = FocalLoss()

# Optimizer with weight decay for better generalization
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Learning rate scheduler to reduce learning rate on plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Mixed Precision Training
scaler = torch.amp.GradScaler()

# Gradient Accumulation
accumulation_steps = 4

# Training loop with validation and performance tracking
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100):
    best_val_accuracy = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        optimizer.zero_grad()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        val_labels = []
        val_preds = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(predicted.cpu().numpy())

                val_loss += criterion(outputs, labels).item()

        val_accuracy = 100 * val_correct / val_total
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {100 * correct / total:.2f}%, Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.4f}')

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_solar_panel_defect_classifier_RESNET.pth')

    return val_labels, val_preds

if __name__ == "__main__":
    # Train the model
    val_labels, val_preds = train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100)

    # Compute and print final confusion matrix and classification report
    cm = confusion_matrix(val_labels, val_preds)
    cr = classification_report(val_labels, val_preds, target_names=['Non-Defect', 'Defect'])
    
    print("Final Confusion Matrix:")
    print(cm)
    print("Final Classification Report:")
    print(cr)

## evaluation on validation set 
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
# val_dir = r"D:\Rishabh Uikey\onward solar\onward 40mw\Clipped_defected_panels\dataset\val"
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



## save one geojson file
# import torch
# from torch import nn
# from torchvision import transforms, models
# from torch.utils.data import DataLoader, Dataset
# from torchvision.models import ResNet18_Weights
# from PIL import Image
# import os
# import numpy as np
# import random
# import rasterio
# import json
# import shutil
# from shapely.geometry import shape, mapping
# from geojson import Feature, FeatureCollection, dump

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

# # Function to clip panels from the orthomosaic and save them temporarily
# def clip_and_save_panels(ortho_path, geojson_path, temp_dir):
#     annotations = []
#     if not os.path.exists(temp_dir):
#         os.makedirs(temp_dir)

#     with rasterio.open(ortho_path) as src:
#         with open(geojson_path) as f:
#             geojson = json.load(f)
#             for i, feature in enumerate(geojson['features']):
#                 geom = shape(feature['geometry'])
#                 if geom.is_valid:
#                     coords = geom.bounds  # Get bounding box of the polygon
#                     window = rasterio.windows.from_bounds(*coords, transform=src.transform)
#                     clip = src.read(window=window, indexes=1)  # Read the first (and only) band
                    
#                     # Convert to image and save temporarily
#                     img = Image.fromarray(clip)
#                     img = img.convert("L")  # Convert to 8-bit pixels, grayscale
#                     img_path = os.path.join(temp_dir, f"panel_{i}.png")
#                     img.save(img_path)

#                     annotations.append(geom)  # Save the geometry for later

#     return annotations

# # Define custom dataset for inference on a mixed set
# class MixedPanelDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.image_paths = []
#         self.transform = transform

#         # Traverse the directory and collect image paths
#         for img_name in os.listdir(root_dir):
#             if img_name.endswith(('.png', '.jpg', '.jpeg', '.tif')):
#                 self.image_paths.append(os.path.join(root_dir, img_name))

#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         return image, img_path  # Returning image path for later use

# # Define transforms for the mixed dataset
# mixed_transform = transforms.Compose([
#     transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # Resize to 224x224 to fit ResNet18
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Define the Focal Loss criterion
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
# model_path = r"C:\Users\LEGION\Downloads\best_solar_panel_defect_classifier_RESNET.pth"
# model.load_state_dict(torch.load(model_path))
# model.eval()  # Set the model to evaluation mode

# # Function to run inference on the clipped panels
# def run_inference(model, clipped_loader, criterion=None):
#     model.eval()
#     predictions = []
#     total_loss = 0.0

#     with torch.no_grad():
#         for images, img_paths in clipped_loader:  # img_paths is a list of strings (file paths)
#             images = images.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)

#             if criterion is not None:
#                 targets = torch.zeros_like(predicted).to(device)
#                 loss = criterion(outputs, targets)
#                 total_loss += loss.item()

#             for img_path, pred in zip(img_paths, predicted.cpu().numpy()):
#                 predictions.append((img_path, 'Defect' if pred == 1 else 'Non-Defect'))

#     if criterion is not None:
#         avg_loss = total_loss / len(clipped_loader)
#         print(f"Avg Loss during inference: {avg_loss:.4f}")

#     return predictions

# # Function to save predictions to a GeoJSON file
# def save_predictions_to_geojson(predictions, annotations, output_geojson_path):
#     features = []
#     for (img_path, label), geom in zip(predictions, annotations):
#         feature = Feature(geometry=mapping(geom), properties={"prediction": label, "image": img_path})
#         features.append(feature)
    
#     feature_collection = FeatureCollection(features)
    
#     with open(output_geojson_path, 'w') as f:
#         dump(feature_collection, f)


# if __name__ == "__main__":
#     # Paths to the orthomosaic and GeoJSON file
#     ortho_path = r"D:\Rishabh Uikey\onward solar\onward 40mw\new_model-3\kml3_thermal_ortho_32643\kml3_thermal_ortho_32643_reproj_raster.tif"
#     geojson_path = r"D:\Rishabh Uikey\onward solar\onward 40mw\new_model-3\kml3_thermal_ortho_32643\kml3_thermal_ortho_32643_annotations.geojson"
#     temp_dir = r"D:\Rishabh Uikey\onward solar\temp_clipped_panels"
#     output_geojson_path = r"D:\Rishabh Uikey\onward solar\predictions.geojson"

#     # Clip and save panels from the orthomosaic
#     annotations = clip_and_save_panels(ortho_path, geojson_path, temp_dir)

#     # Create the dataset and dataloader for clipped panels
#     clipped_dataset = MixedPanelDataset(temp_dir, transform=mixed_transform)
#     clipped_loader = DataLoader(clipped_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

#     # Run inference on the clipped panels
#     predictions = run_inference(model, clipped_loader, criterion=criterion)

#     # Save the predictions to a GeoJSON file
#     save_predictions_to_geojson(predictions, annotations, output_geojson_path)

#     # Output predictions
#     for img_path, label in predictions:
#         print(f"Image: {img_path} - Predicted: {label}")

#     # Delete the temporary clipped images
#     shutil.rmtree(temp_dir)


## working code 

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

# # Function to run inference and generate predictions sequentially
# def evaluate_and_predict(model, dataset):
#     model.eval()
#     predictions = []
#     img_paths = []
#     geoms = []

#     with torch.no_grad():
#         for i in range(len(dataset)):
#             image, geom, img_path = dataset[i]
#             image = image.unsqueeze(0).to(device)  # Add batch dimension
#             output = model(image)
#             _, predicted = torch.max(output.data, 1)

#             predictions.append(predicted.item())
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
#     ortho_path = r"D:\Rishabh Uikey\onward solar\onward 40mw\new_model-3\Thermal_Ortho_KML1Clipped\Thermal_Ortho_KML1Clipped_reproj_raster.tif"
#     geojson_path = r"D:\Rishabh Uikey\onward solar\onward 40mw\new_model-3\Thermal_Ortho_KML1Clipped\Thermal_Ortho_KML1Clipped_annotations.geojson"
#     temp_dir = r"D:\Rishabh Uikey\onward solar\temp_clipped_panels"
#     defect_output_geojson_path = r"D:\Rishabh Uikey\onward solar\onward 40mw\new_model-3\Thermal_Ortho_KML1Clipped\Thermal_Ortho_KML1Clipped_defect_predictions.geojson"
#     non_defect_output_geojson_path = r"D:\Rishabh Uikey\onward solar\onward 40mw\new_model-3\Thermal_Ortho_KML1Clipped\Thermal_Ortho_KML1Clipped_non_defect_predictions.geojson"
    
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
#     model_path = r"C:\Users\LEGION\Downloads\best_solar_panel_defect_classifier_RESNET.pth"
#     model.load_state_dict(torch.load(model_path))
#     model.eval()  # Set the model to evaluation mode

#     # Run inference and predict
#     val_preds, val_img_paths, val_geoms = evaluate_and_predict(model, val_dataset)

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


##new try 

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
#     ortho_path = r"D:\Rishabh Uikey\onward solar\onward 40mw\new_model-3\kml3_thermal_ortho_32643\kml3_thermal_ortho_32643_reproj_raster.tif"
#     geojson_path = r"D:\Rishabh Uikey\onward solar\onward 40mw\new_model-3\kml3_thermal_ortho_32643\kml3_thermal_ortho_32643_annotations.geojson"
#     temp_dir = r"D:\Rishabh Uikey\onward solar\temp_clipped_panels_1"
#     defect_output_geojson_path = r"D:\Rishabh Uikey\onward solar\onward 40mw\new_model-3\kml3_thermal_ortho_32643\kml3_thermal_ortho_32643_annotations_defect.geojson"
#     non_defect_output_geojson_path = r"D:\Rishabh Uikey\onward solar\onward 40mw\new_model-3\kml3_thermal_ortho_32643\kml3_thermal_ortho_32643_annotations_no_defect.geojson"
    
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
#     threshold = 0.2  # Adjust this threshold as needed
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

# # Function to run inference and generate predictions with thresholding
# def evaluate_and_predict(model, dataset, low_threshold, high_threshold):
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

#             # Apply refined threshold for defect classification
#             if defect_prob >= high_threshold:
#                 predicted = 1  # High confidence Defect
#             elif defect_prob < low_threshold:
#                 predicted = 0  # Clear Non-Defect
#             else:
#                 predicted = 0  # Treat the intermediate range as non-defect or handle separately

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
#     ortho_path = r"D:\Rishabh Uikey\onward solar\onward 40mw\new_model-3\kml3_thermal_ortho_32643\kml3_thermal_ortho_32643_reproj_raster.tif"
#     geojson_path = r"D:\Rishabh Uikey\onward solar\onward 40mw\new_model-3\kml3_thermal_ortho_32643\kml3_thermal_ortho_32643_annotations.geojson"
#     temp_dir = r"D:\Rishabh Uikey\onward solar\temp_clipped_panels_1"
#     defect_output_geojson_path = r"D:\Rishabh Uikey\onward solar\onward 40mw\new_model-3\kml3_thermal_ortho_32643\kml3_thermal_ortho_32643_annotations_defect.geojson"
#     non_defect_output_geojson_path = r"D:\Rishabh Uikey\onward solar\onward 40mw\new_model-3\kml3_thermal_ortho_32643\kml3_thermal_ortho_32643_annotations_no_defect.geojson"
    
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

#     # Define thresholds
#     low_threshold = 0.2  # Adjust this threshold as needed
#     high_threshold = 0.8  # Adjust for more confident defect classification

#     # Run inference with refined thresholds
#     val_preds, val_img_paths, val_geoms = evaluate_and_predict(model, val_dataset, low_threshold=low_threshold, high_threshold=high_threshold)

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
