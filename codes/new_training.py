import os
import torch
from ultralytics import YOLO

def train_yolov8_model(data_path, epochs, batch_size, img_size, weights):
    """
    Train a YOLOv8 model.

    Parameters:
        data_path (str): Path to the dataset in YOLO format.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        img_size (int): Image size for training.
        weights (str): Path to the pre-trained weights file.
    """
    # Check if CUDA is available and use it if possible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the YOLO model
    model = YOLO(weights).to(device)

    # Train the model with correct argument format
    model.train(data=data_path, epochs=epochs, batch=batch_size, imgsz=img_size, device=device)

    # Save the trained model
    model.save(os.path.join(output1, 'trained_model.pt'))
    print("Training complete. Model saved.")

if __name__ == "__main__":
    data_path = r"D:\new_data\yolo_split.yaml" # Update with your dataset path
    epochs = 100
    batch_size = 8
    img_size = 640
    weights = "yolov8s-obb.pt"  # Path to your pre-trained weights
    output1=r"D:\combined_data\training"

    train_yolov8_model(data_path, epochs, batch_size, img_size, weights)
