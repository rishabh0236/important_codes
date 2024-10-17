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