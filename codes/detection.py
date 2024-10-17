import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights
from PIL import Image
from rasterio.mask import mask
import os
import cv2
import rasterio
import json
import numpy as np
import random
from geojson import Feature, FeatureCollection, dump
from shapely.geometry import shape, mapping
import shutil

ORTHO_PATH = os.getenv('ORTHO_PATH', './ortho/ortho.tif')  # Default to './ortho/ortho.tif'
GEOJSON_PATH = os.getenv('GEOJSON_PATH', './output/inference_output.geojson')  # Default to './output/inference_output.geojson'
TEMP_DIR = os.getenv('TEMP_DIR', './temp_clipped_panels')  # Default to './temp_clipped_panels'
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './output')  # Default to './output'
MODEL_PATH = os.getenv('MODEL_PATH', './model/best_solar_panel_defect_classifier_RESNET.pth')  # Model path

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

# Focal Loss implementation
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

# Function to clip panels from the orthomosaic and save them temporarily with polygon info
def clip_and_save_panels(ortho_path, geojson_path, temp_dir):
    no_defect_dir = os.path.join(temp_dir, "no_defect")
    if not os.path.exists(no_defect_dir):
        os.makedirs(no_defect_dir)

    annotations = []
    with rasterio.open(ortho_path) as src:
        with open(geojson_path) as f:
            geojson = json.load(f)
            crs = geojson.get("crs")  # Extract CRS from the GeoJSON if available
            for i, feature in enumerate(geojson['features']):
                geom = shape(feature['geometry'])
                if geom.is_valid:
                    coords = geom.bounds  # Get bounding box of the polygon
                    window = rasterio.windows.from_bounds(*coords, transform=src.transform)
                    clip = src.read(window=window, indexes=1) 

                    # Define the path for the TIFF image
                    img_path = os.path.join(no_defect_dir, f"panel_{i}.tif")

                    # Save the clipped image as a GeoTIFF
                    profile = src.profile
                    profile.update({
                        "driver": "GTiff",
                        "height": clip.shape[0],
                        "width": clip.shape[1],
                        "transform": rasterio.windows.transform(window, src.transform)
                    })

                    with rasterio.open(img_path, "w", **profile) as dst:
                        dst.write(clip, 1)

                    # Save the image path and the original geometry
                    annotations.append((img_path, geom))

    return annotations, crs

# Define custom dataset that includes polygon geometries
class PanelDataset(Dataset):
    def __init__(self, annotations, transform=None):
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_path, geom = self.annotations[idx]
        with rasterio.open(img_path) as src:
            image = src.read(1) 
            image = Image.fromarray(image).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, geom, img_path

# Define transforms for the validation dataset
val_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to run inference and generate predictions with thresholding to reduce false negatives
def evaluate_and_predict_with_threshold(model, dataset, threshold):
    model.eval()
    predictions = []
    img_paths = []
    geoms = []

    with torch.no_grad():
        for i in range(len(dataset)):
            image, geom, img_path = dataset[i]
            image = image.unsqueeze(0).to(device) 
            output = model(image)
            probs = nn.functional.softmax(output, dim=1)  # Get probabilities

            # Get the defect probability 
            defect_prob = probs[0, 1].item()

            # Apply threshold
            if defect_prob > threshold:
                predicted = 1  
            else:
                predicted = 0  

            predictions.append(predicted)
            img_paths.append(img_path)
            geoms.append(geom)

    return predictions, img_paths, geoms


# Function to create GeoJSON features based on predictions
def create_geojson_features(img_paths, predictions, geoms):
    defect_features = []
    non_defect_features = []
    for img_path, pred, geom in zip(img_paths, predictions, geoms):
        label = "Defect" if pred == 1 else "Non Defect"
        feature = Feature(geometry=mapping(geom), properties={"prediction": label, "image": img_path})
        if label == "Defect":
            defect_features.append(feature)
        else:
            non_defect_features.append(feature)
    return defect_features, non_defect_features

# Function to save predictions to GeoJSON
def save_predictions_to_geojson(features, output_path, crs=None):
    feature_collection = FeatureCollection(features)
    if crs:
        feature_collection['crs'] = crs  
    with open(output_path, 'w') as f:
        dump(feature_collection, f)

def load_clip_and_blur_tiff_image(file_path, min_temp, max_temp):
    clip_pixels = 2  
    blur_ksize = (5, 5)
    blur_sigma = 1

    with rasterio.open(file_path) as src:
        image = src.read(1)
    
    # Clipping 2 pixels from all sides
    clipped_image = image[clip_pixels:-clip_pixels, clip_pixels:-clip_pixels]
    
    # Normalize the clipped image
    def normalize_image(image, min_temp, max_temp):
        normalized_image = ((image - min_temp) / (max_temp - min_temp)) * 255
        return np.clip(normalized_image, 0, 255).astype(np.uint8)
    
    normalized_clipped_image = normalize_image(clipped_image, min_temp, max_temp)
    
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(normalized_clipped_image, blur_ksize, blur_sigma)

    # Calculate threshold values based on the original pixel values
    min_val = np.min(normalized_clipped_image)
    max_val = np.max(normalized_clipped_image)
    lower_threshold = min_val + 0.2 * (max_val - min_val)
    upper_threshold = min_val + 0.68 * (max_val - min_val)
    
    # Apply thresholding to create a binary mask for defects using the upper threshold
    _, thresholded_image_upper = cv2.threshold(normalized_clipped_image, upper_threshold, 255, cv2.THRESH_BINARY)

    # Apply morphological closing
    kernel = np.ones((5, 5), np.uint8)
    closed_image = cv2.morphologyEx(thresholded_image_upper, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Calculate panel area
    panel_area = clipped_image.shape[0] * clipped_image.shape[1]
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(closed_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    large_contours = [c for c in contours if cv2.contourArea(c) >=0.2 * panel_area]
    
    # Calculate panel area
    panel_area = clipped_image.shape[0] * clipped_image.shape[1]
    
    # Check the defect type based on the number of contours and their areas
    defect_type = "No Defect"
    
    if len(contours) == 1:
        contour_area = cv2.contourArea(contours[0])
        x, y, w, h = cv2.boundingRect(contours[0])
        aspect_ratio = w / h

        if 0.1 * panel_area <= contour_area < 0.33 * panel_area and aspect_ratio < 0.4:
            defect_type = "bypass diode"
        elif contour_area < 0.1 * panel_area:  
            defect_type = "single cell hotspot"

    elif len(contours) > 1:
        larger_contours = [c for c in contours if cv2.contourArea(c) > 0.1 * panel_area]
        
        if len(larger_contours) == 0:
            defect_type = "multi cell hotspot"
        elif len(larger_contours) == 1:
            contour_area = cv2.contourArea(larger_contours[0])
            x, y, w, h = cv2.boundingRect(larger_contours[0])
            aspect_ratio = w / h

            if contour_area < 0.33 * panel_area and aspect_ratio < 0.33:
                defect_type = "bypass diode"
        elif len(larger_contours) > 1:
            defect_type = "No Defect"

    return defect_type


# Correct function to clip and classify panels
def clip_and_classify_panels(geojson_path, ortho_path, output_dir):
    with rasterio.open(ortho_path) as src: 
        ortho_image = src.read(1)
        min_temp = np.min(ortho_image)
        max_temp = np.max(ortho_image)
        with open(geojson_path) as f:
            geojson_data = json.load(f)
        
        for feature in geojson_data['features']:
            geom = shape(feature['geometry'])
            coords = [mapping(geom)]
            out_image, out_transform = mask(src, coords, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            # Save the clipped image
            panel_id = feature['properties'].get('id', 'panel')
            panel_path = os.path.join(output_dir, f"{panel_id}.tif")
            if os.path.exists(panel_path):
                os.remove(panel_path)
            with rasterio.open(panel_path, "w", **out_meta) as dest:
                dest.write(out_image)

            # Load the clipped image for defect detection
            defect_name = load_clip_and_blur_tiff_image(panel_path, min_temp, max_temp)
            
            # Update the GeoJSON with defect information
            feature['properties']['defect'] = defect_name
            
            print(f"Panel ID: {panel_id}, Defect: {defect_name}")

        # Save the updated GeoJSON
        updated_geojson_path = os.path.join(output_dir, 'updated_annotations.geojson')
        with open(updated_geojson_path, 'w') as f:
            json.dump(geojson_data, f)
        print(f"Updated GeoJSON saved to {updated_geojson_path}")
    
    return updated_geojson_path


# Function to merge two GeoJSON files (updated annotations and non-defect), ensuring NULL values are replaced with "No-Defect"
def merge_geojson_files(updated_geojson_path, non_defect_geojson_path, output_geojson_path, crs=None):
    # Load the updated GeoJSON (defects)
    with open(updated_geojson_path, 'r') as updated_file:
        updated_data = json.load(updated_file)
    
    # Load the non-defect GeoJSON
    with open(non_defect_geojson_path, 'r') as non_defect_file:
        non_defect_data = json.load(non_defect_file)

    # Merge the features
    merged_features = []

    # Go through the updated features and ensure that "defect" column has "No-Defect" where it's NULL
    for feature in updated_data['features']:
        if 'defect' not in feature['properties'] or feature['properties']['defect'] is None:
            feature['properties']['defect'] = "No Defect"
        merged_features.append(feature)

    # Add the non-defect features (we assume these already have "No-Defect" in the "defect" field)
    for feature in non_defect_data['features']:
        if 'defect' not in feature['properties'] or feature['properties']['defect'] is None:
            feature['properties']['defect'] = "No Defect"
        merged_features.append(feature)

    # Create the final merged GeoJSON
    merged_geojson = FeatureCollection(merged_features)

    # Add CRS if provided
    if crs:
        merged_geojson['crs'] = crs

    # Save the merged GeoJSON to a new file
    with open(output_geojson_path, 'w') as output_file:
        dump(merged_geojson, output_file)

    print(f"Merged GeoJSON saved to {output_geojson_path}")


if __name__ == "__main__":
    # Replace hardcoded paths with environment variables
    defect_output_geojson_path = os.getenv('DEFECT_OUTPUT_GEOJSON', './output/defect_predictions.geojson')
    non_defect_output_geojson_path = os.getenv('NON_DEFECT_OUTPUT_GEOJSON', './output/non_defect_predictions.geojson')
    merged_output_geojson_path = os.getenv('MERGED_OUTPUT_GEOJSON', './output/final_predictions.geojson')

    # Clip and save panels
    annotations, crs = clip_and_save_panels(ORTHO_PATH, GEOJSON_PATH, TEMP_DIR)

    # Create the validation dataset
    val_dataset = PanelDataset(annotations, transform=val_transform)

    # Load the pre-trained ResNet18 model
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)  
    )

    model = model.to(device)

    # Load the trained model weights from environment variable
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()  

    # Run inference and predict
    val_preds, val_img_paths, val_geoms = evaluate_and_predict_with_threshold(model, val_dataset, threshold=0.3)

    # Create GeoJSON features
    defect_features, non_defect_features = create_geojson_features(val_img_paths, val_preds, val_geoms)

    # Save predictions
    save_predictions_to_geojson(defect_features, defect_output_geojson_path, crs)
    save_predictions_to_geojson(non_defect_features, non_defect_output_geojson_path, crs)

    # Clean up: Delete the temporary directory
    shutil.rmtree(TEMP_DIR)

    # Merge updated and non-defect GeoJSONs
    updated_geojson_path = clip_and_classify_panels(defect_output_geojson_path, ORTHO_PATH, OUTPUT_DIR)
    merge_geojson_files(updated_geojson_path, non_defect_output_geojson_path, merged_output_geojson_path, crs)

    print(f"Merged GeoJSON saved to {merged_output_geojson_path}")