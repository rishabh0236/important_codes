import cv2
import os 
import json 
import utm
import random
import rasterio
import shutil
import torch
import geopandas as gpd
import numpy as np
import supervision as sv
import matplotlib.pyplot as plt 
from PIL import Image
from tqdm import tqdm
from rasterio.windows import Window
from rasterio.crs import CRS
from rasterio.mask import mask 
from rasterio.warp import calculate_default_transform, reproject, Resampling
from geojson import Feature, FeatureCollection, dump
from shapely.geometry import Polygon, mapping, shape
from shapely.affinity import scale, translate
from rtree import index
from ultralytics import YOLO
from torch import nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights

# Adjust these paths according to your environment
OUTPUT_PATH = os.getenv('OUTPUT_DIR', './output')  # Default to './output'
YOLO_MODEL_PATH = os.getenv('MODEL_PATH', './model/best_v4.pt')  # Model path
ORTHOMOSAIC_PATH = os.getenv('ORTHO_PATH', './ortho/ortho.tif')  # Default to './ortho/ortho.tif'
model_path = os.getenv('MODEL_PATH', './model/best_solar_panel_defect_classifier_RESNET.pth')  # Model path
temp_dir = os.getenv('TEMP_DIR', './temp_clipped_panels')  # Default to './temp_clipped_panels'
defect_output_geojson_path = os.getenv('DEFECT_OUTPUT_GEOJSON', './output/defect_predictions.geojson')
non_defect_output_geojson_path = os.getenv('NON_DEFECT_OUTPUT_GEOJSON', './output/non_defect_predictions.geojson')
merged_output_geojson_path = os.getenv('MERGED_OUTPUT_GEOJSON', './output/final_predictions.geojson')

count=0

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
            crs = geojson.get("crs")  
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
        label = "Defect" if pred == 1 else "Non-Defect"
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

    # Calculate panel area
    panel_area = clipped_image.shape[0] * clipped_image.shape[1]

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(closed_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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


def reproject_raster(input_raster, output_raster, dst_crs):
    with rasterio.open(input_raster) as src:
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        with rasterio.open(output_raster, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=rasterio.warp.Resampling.nearest)
    print(f"Reprojected raster saved at: {output_raster}")

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE to an image."""
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def split_orthomosaic_and_annotations(model, ortho_path, tile_size=640, overlap=100):
    global count
    bboxes = []

    with rasterio.open(ortho_path) as src:
        ortho_name = os.path.splitext(os.path.basename(ortho_path))[0]
        transform = src.transform
        resolution_x = transform[0]
        resolution_y = -transform[4]
        num_cols = len(range(0, src.width, tile_size - overlap))
        num_rows = len(range(0, src.height, tile_size - overlap))
        progress_bar = tqdm(total=num_cols * num_rows, desc=f'Processing {ortho_name}')

        for col_off in range(0, src.width, tile_size - overlap):
            for row_off in range(0, src.height, tile_size - overlap):
                window = Window(col_off, row_off, tile_size, tile_size)
                tile_data = src.read(window=window)

                if not np.any(tile_data):
                    progress_bar.update(1)
                    continue  # Skip empty tiles

                tile_x = col_off
                tile_y = row_off

                tile_data = tile_data[0]
                # Apply CLAHE
                tile_data_clahe = apply_clahe(tile_data)

                # Convert to image and apply model
                tile_image = Image.fromarray(tile_data_clahe)
                
                # Convert tile_image to numpy array for supervision
                tile_image_np = np.array(tile_image)

                results = model.predict(tile_image, conf=0.2)

                tile_bboxes = []

                try:
                    detections = sv.Detections.from_ultralytics(results[0])
                    for obb in detections.data['xyxyxyxy']:
                        class_id = 0  
                        polygon = obb.tolist()
                        adjusted_polygon = [(tile_x + x, tile_y + y) for x, y in polygon]
                        poly = Polygon(adjusted_polygon)
                        count += 1

                        if poly.is_valid:  # Check if the polygon is valid
                            tile_bboxes.append([class_id, list(poly.exterior.coords)])
                            bboxes.append([class_id, list(poly.exterior.coords)])
                            print(f"Extracted OBB: {list(poly.exterior.coords)}")

                    # Plot inference results for the current tile
                    # plot_tile_inference(tile_image_np, detections, tile_x, tile_y)

                except Exception as e:
                    print(f"Error in extracting OBB boxes: {e}")

                progress_bar.update(1)

    return bboxes

def coco_to_geojson_polygon(polygon_coordinates, transform):
    transformed_polygon = []
    for x, y in polygon_coordinates:
        x_transformed, y_transformed = transform * (x, y)
        transformed_polygon.append((x_transformed, y_transformed))
    
    return Polygon(transformed_polygon)

def adjust_polygon_size(polygon, width=1.1, height=2.1):
    centroid = polygon.centroid
    current_width = polygon.bounds[2] - polygon.bounds[0]
    current_height = polygon.bounds[3] - polygon.bounds[1]
    scale_x = width / current_width
    scale_y = height / current_height
    scaled_polygon = scale(polygon, xfact=scale_x, yfact=scale_y, origin=centroid)
    return scaled_polygon

def remove_overlapping_polygons(features, threshold):
    """
    Removes the second overlapping polygon that has more than the specified overlap percentage,
    keeping the first polygon from overlapping sets.

    Parameters:
        features (list): List of geojson features.
        threshold (float): Percentage of overlap above which polygons are removed.

    Returns:
        list: Filtered list of geojson features with reduced overlaps.
    """
    filtered_features = []
    polygons_to_remove = set()

    # Create an R-tree index
    idx = index.Index()
    
    # Add polygons to the index
    for i, feature in enumerate(features):
        polygon = Polygon(feature['geometry']['coordinates'][0])
        idx.insert(i, polygon.bounds)

    for i, feature in enumerate(features):
        if i in polygons_to_remove:
            continue
        
        polygon_i = Polygon(feature['geometry']['coordinates'][0])
        overlap_found = False

        # Get potential overlapping polygons using the index
        potential_overlaps = list(idx.intersection(polygon_i.bounds))

        for j in potential_overlaps:
            if i != j and j not in polygons_to_remove:
                polygon_j = Polygon(features[j]['geometry']['coordinates'][0])
                intersection_area = polygon_i.intersection(polygon_j).area
                if intersection_area / polygon_i.area > threshold or intersection_area / polygon_j.area > threshold:
                    polygons_to_remove.add(j)  
                    overlap_found = True
        
        if not overlap_found or i not in polygons_to_remove:
            filtered_features.append(feature)

    return filtered_features

def convert_annotations_to_geojson(geotiff_path, annotations_txt_path, output_geojson_path):
    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        crs = src.crs

    features = []

    with open(annotations_txt_path, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc='Converting annotations'):
        line = line.strip().split()
        class_label = line[0]
        coordinates = list(map(float, line[1:]))
        polygon_coordinates = [(coordinates[i], coordinates[i + 1]) for i in range(0, len(coordinates), 2)]
        
        if len(polygon_coordinates) >= 4:  # Ensure we have 4 points for a valid polygon
            polygon = coco_to_geojson_polygon(polygon_coordinates, transform)
            
            # Adjust polygon size
            polygon = adjust_polygon_size(polygon)

            feature = {
                'type': 'Feature',
                'geometry': mapping(polygon),
                'properties': {
                    'class_name': class_label
                }
            }
            features.append(feature)

    # Remove overlapping polygons
    filtered_features = remove_overlapping_polygons(features, threshold=0.2)

    geojson_data = {
        'type': 'FeatureCollection',
        'features': filtered_features,
        'crs': {
            'type': 'name',
            'properties': {
                'name': crs.to_string()
            }
        }
    }

    with open(output_geojson_path, 'w') as f:
        json.dump(geojson_data, f, indent=2)
        print(f"GeoJSON data: {geojson_data}")

def process_ortho(model, ortho_file):
    ortho_dir = os.path.dirname(os.path.abspath(ortho_file))
    filename, _ = os.path.splitext(os.path.basename(ortho_file))
    output_dir = os.path.join(OUTPUT_PATH, filename)
    os.makedirs(output_dir, exist_ok=True)

    reproj_raster = os.path.join(output_dir, f"{filename}_reproj_raster.tif")
    annotations_txt_path = os.path.join(output_dir, f"{filename}_annotations.txt")
    output_geojson_path = os.path.join(output_dir, f"{filename}_annotations.geojson")


    with rasterio.open(ortho_file) as src:
        if src.crs.to_epsg() == 4326:
            lon, lat = src.transform[0], src.transform[3]
            utm_zone = utm.from_latlon(lat, lon)[2]
            dst_crs = CRS.from_epsg(32600 + utm_zone)
        else:
            dst_crs = src.crs

        reproject_raster(ortho_file, reproj_raster, dst_crs)

    bboxes = split_orthomosaic_and_annotations(model, reproj_raster)

    print("boxes are", bboxes)  # Debugging print to check bboxes

    with open(annotations_txt_path, 'w') as file:
        for annotation in bboxes:
            label = int(annotation[0])
            coordinates = annotation[1]
            coordinates_list = [f"{x:.4f} {y:.4f}" for x, y in coordinates]
            line = f"{label} " + " ".join(coordinates_list)
            file.write(line + '\n')

    convert_annotations_to_geojson(reproj_raster, annotations_txt_path, output_geojson_path)
    print(f"GeoJSON file saved at: {output_geojson_path}")

    return output_geojson_path

if __name__ == "__main__":
    print("Loading model...")
    model = YOLO(YOLO_MODEL_PATH)

    print("Starting inference...")
    geojson_path = process_ortho(model, ORTHOMOSAIC_PATH)

    print("Inference completed.")


    # Clip and save panels from the orthomosaic to the "no_defect" folder
    annotations, crs = clip_and_save_panels(ORTHOMOSAIC_PATH, geojson_path, temp_dir)

    # Create the validation dataset
    val_dataset = PanelDataset(annotations, transform=val_transform)

    # Load the pre-trained ResNet18 model
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Replace the final fully connected layer with a custom layer
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)  
    )

    model = model.to(device)

    # Load the trained model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # Run inference and predict
    # Set a custom threshold to reduce false negatives (e.g., 0.3 for higher sensitivity)
    val_preds, val_img_paths, val_geoms = evaluate_and_predict_with_threshold(model, val_dataset, threshold=0.25)


    # Create GeoJSON features
    defect_features, non_defect_features = create_geojson_features(val_img_paths, val_preds, val_geoms)

    # Save the features to two separate GeoJSON files with CRS
    save_predictions_to_geojson(defect_features, defect_output_geojson_path, crs)
    save_predictions_to_geojson(non_defect_features, non_defect_output_geojson_path, crs)

    # Clean up: Delete the temporary directory
    shutil.rmtree(temp_dir)

    print(f"Defect predictions saved to {defect_output_geojson_path}")
    print(f"Non-defect predictions saved to {non_defect_output_geojson_path}")
    print(f"Temporary directory {temp_dir} deleted.")

    # Now clip and classify the panels, passing the correct GeoJSON file path
    updated_geojson_path = clip_and_classify_panels(defect_output_geojson_path, ORTHOMOSAIC_PATH, OUTPUT_PATH)

    # Now merge the updated annotations and non-defect GeoJSONs into one
    merge_geojson_files(updated_geojson_path, non_defect_output_geojson_path, merged_output_geojson_path, crs)

    print(f"Merged GeoJSON saved to {merged_output_geojson_path}")

    

