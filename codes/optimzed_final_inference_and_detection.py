import cv2
import os 
import json 
import utm
import time
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

# Paths

# Get paths from environment variables or use default values if not set
OUTPUT_PATH = os.getenv('OUTPUT_PATH', './output')  # Default to './output'
YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', './models/best_v4.pt')  # Default to './models/best_v4.pt'
ORTHOMOSAIC_PATH = os.getenv('ORTHOMOSAIC_PATH', './data/ortho/ortho.tif')  # Default to './data/ortho/ortho.tif'
model_path = os.getenv('MODEL_PATH', './models/best_solar_panel_defect_classifier_RESNET.pth')  # Default to './models/best_solar_panel_defect_classifier_RESNET.pth'
temp_dir = os.getenv('TEMP_DIR', './temp_clipped_panels')  # Default to './temp_clipped_panels'
defect_output_geojson_path = os.getenv('DEFECT_OUTPUT_GEOJSON', './output/defect_predictions.geojson')
non_defect_output_geojson_path = os.getenv('NON_DEFECT_OUTPUT_GEOJSON', './output/non_defect_predictions.geojson')
merged_output_geojson_path = os.getenv('MERGED_OUTPUT_GEOJSON', './output/merged_predictions.geojson')

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


# Transform for ResNet classifier
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Panel Dataset class for defect detection using ResNet
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


# Function to load and blur the image for defect type classification
def load_clip_and_blur_tiff_image(file_path, min_temp, max_temp):
    clip_pixels = 2
    blur_ksize = (5, 5)
    blur_sigma = 1

    with rasterio.open(file_path) as src:
        image = src.read(1)

    clipped_image = image[clip_pixels:-clip_pixels, clip_pixels:-clip_pixels]

    def normalize_image(image, min_temp, max_temp):
        normalized_image = ((image - min_temp) / (max_temp - min_temp)) * 255
        return np.clip(normalized_image, 0, 255).astype(np.uint8)

    normalized_clipped_image = normalize_image(clipped_image, min_temp, max_temp)

    blurred_image = cv2.GaussianBlur(normalized_clipped_image, blur_ksize, blur_sigma)

    min_val = np.min(normalized_clipped_image)
    max_val = np.max(normalized_clipped_image)
    upper_threshold = min_val + 0.68 * (max_val - min_val)

    _, thresholded_image_upper = cv2.threshold(normalized_clipped_image, upper_threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    closed_image = cv2.morphologyEx(thresholded_image_upper, cv2.MORPH_CLOSE, kernel, iterations=2)

    panel_area = clipped_image.shape[0] * clipped_image.shape[1]

    contours, _ = cv2.findContours(closed_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    large_contours = [c for c in contours if cv2.contourArea(c) > 0.2 * panel_area]

    defect_type = "No Defect"
    # Check the defect type based on the updated conditions
    if len(contours) == 1:
        contour_area = cv2.contourArea(contours[0])
        x, y, w, h = cv2.boundingRect(contours[0])
        aspect_ratio = w / h

        if 0.1 * panel_area <= contour_area < 0.33 * panel_area and aspect_ratio < 0.33:
            defect_type = "bypass diode"
        elif contour_area < 0.07 * panel_area:  # Reduced threshold for single cell
            defect_type = "single cell hotspot"
        else:
            defect_type = "No Defect"
            
    elif len(contours) > 1:
        larger_contours = [c for c in large_contours if cv2.contourArea(c) > 0.2 * panel_area]
        print("Total large contours:", len(large_contours))
            
        if len(larger_contours) == 0:
            defect_type = "multi cell hotspot"
        elif len(larger_contours) == 1:
            contour_area = cv2.contourArea(larger_contours[0])
            x, y, w, h = cv2.boundingRect(larger_contours[0])
            aspect_ratio = w / h

            if contour_area < 0.33 * panel_area and aspect_ratio < 0.33:
                defect_type = "bypass diode"
            else:
                defect_type = "No Defect"
        elif len(larger_contours) > 1:
            defect_type = "No Defect"
    
    return defect_type


def clip_and_classify_panels(ortho_path, geojson_path, temp_dir, resnet_model, threshold, output_dir):
    """
    Clips the panels from the orthomosaic based on GeoJSON, classifies them as defect/no-defect using ResNet,
    and further classifies the defect type if a defect is found. Clipping happens only once.

    Parameters:
    - ortho_path (str): Path to the orthomosaic TIFF file.
    - geojson_path (str): Path to the GeoJSON file containing the panel polygons.
    - temp_dir (str): Directory to temporarily save the clipped panels.
    - resnet_model (torch.nn.Module): The pre-trained ResNet model for defect/no-defect classification.
    - threshold (float): Probability threshold for defect/no-defect classification.
    - output_dir (str): Directory to save the updated GeoJSON and panel images.

    Returns:
    - updated_geojson_path (str): Path to the updated GeoJSON with classification results.
    """

    # Create a temporary directory for saving clipped panels if it doesn't exist
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Create the output directory for saving results if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    annotations = []  # To store clipped image paths and geometries

    # Open the orthomosaic
    with rasterio.open(ortho_path) as src:

        ortho_image = src.read(1)
        min_temp = np.min(ortho_image)
        max_temp = np.max(ortho_image)

        with open(geojson_path) as f:
            geojson_data = json.load(f)
            crs = geojson_data.get("crs")  # Get CRS from GeoJSON

            # Loop through each feature (panel) in the GeoJSON
            for i, feature in enumerate(geojson_data['features']):
                geom = shape(feature['geometry'])

                # Check if the geometry is valid before processing
                if geom.is_valid:
                    coords = geom.bounds  # Get bounding box for clipping
                    window = rasterio.windows.from_bounds(*coords, transform=src.transform)  # Create clipping window
                    clip = src.read(window=window)  # Clip the panel image

                    # Define the path for saving the clipped panel as a GeoTIFF
                    panel_id = feature['properties'].get('id', f"panel_{i}")
                    img_path = os.path.join(temp_dir, f"{panel_id}.tif")

                    # Update the profile for the clipped image
                    profile = src.profile
                    profile.update({
                        "driver": "GTiff",
                        "height": clip.shape[1],  # For multi-band support
                        "width": clip.shape[2],
                        "transform": rasterio.windows.transform(window, src.transform)
                    })

                    # Save the clipped panel as a GeoTIFF
                    with rasterio.open(img_path, "w", **profile) as dst:
                        dst.write(clip)

                    # Append the image path and geometry to annotations
                    annotations.append((img_path, geom))

                    # Prepare the image for ResNet classification (defect/no-defect)
                    panel_image = Image.fromarray(clip[0])  # Assuming single-band, convert to RGB if needed
                    panel_image = panel_image.convert("RGB")  # Convert to RGB for ResNet compatibility
                    panel_tensor = val_transform(panel_image).unsqueeze(0).to(device)

                    # Run ResNet inference for defect/no-defect classification
                    resnet_model.eval()
                    with torch.no_grad():
                        output = resnet_model(panel_tensor)
                        probs = nn.functional.softmax(output, dim=1)
                        defect_prob = probs[0, 1].item()  # Probability of defect

                    # If the panel is classified as defect, further classify the defect type
                    if defect_prob > threshold:
                        defect_type = load_clip_and_blur_tiff_image(img_path,min_temp, max_temp)
                        feature['properties']['prediction'] = "Defect"
                        feature['properties']['defect_type'] = defect_type
                    else:
                        feature['properties']['prediction'] = "Non-Defect"
                        feature['properties']['defect_type'] = 'No Defect'

                    print(f"Panel ID: {panel_id}, Defect: {feature['properties']['prediction']}, Defect Type: {feature['properties']['defect_type']}")

    # Save the updated GeoJSON with classification results
    updated_geojson_path = os.path.join(output_dir, 'updated_annotations.geojson')
    with open(updated_geojson_path, 'w') as f:
        json.dump(geojson_data, f)
    
    print(f"Updated GeoJSON saved to {updated_geojson_path}")
    
    return updated_geojson_path

# Function to create GeoJSON features based on predictions and defect types
def create_geojson_features(img_paths, predictions, defect_types, geoms):
    defect_features = []
    non_defect_features = []
    for img_path, pred, defect_type, geom in zip(img_paths, predictions, defect_types, geoms):
        label = "Defect" if pred == 1 else "Non-Defect"
        feature = Feature(geometry=mapping(geom), properties={"prediction": label, "defect_type": defect_type, "image": img_path})
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

import geopandas as gpd
from geojson import FeatureCollection

import geopandas as gpd
from geojson import FeatureCollection

import geopandas as gpd
from geojson import FeatureCollection, dump

def save_predictions_to_geojson_and_shapefile(features, output_geojson_path, output_shapefile_path, crs=None):
    # Save as GeoJSON
    feature_collection = FeatureCollection(features)
    if crs:
        feature_collection['crs'] = crs  # Set the CRS in GeoJSON if provided
    
    with open(output_geojson_path, 'w') as f:
        dump(feature_collection, f)
    
    # Convert to GeoDataFrame for shapefile saving
    gdf = gpd.GeoDataFrame.from_features(features)

    # Set the CRS if not already set
    if crs:
        epsg_code = crs['properties']['name'].split(':')[-1]  # Extract EPSG code from the GeoJSON's CRS
        gdf.set_crs(epsg=int(epsg_code), inplace=True)
    else:
        print("Warning: No CRS provided or detected, using default EPSG:4326.")
        gdf.set_crs("EPSG:4326", inplace=True)

    # Reproject the GeoDataFrame to EPSG:32643
    gdf = gdf.to_crs(epsg=32643)
    
    # Fill 'defect_typ' with 'No Defect' where it is missing (NULL)
    gdf['defect_type'] = gdf['defect_type'].fillna('No Defect')
    
    # Save the reprojected GeoDataFrame as a shapefile
    gdf.to_file(output_shapefile_path, driver='ESRI Shapefile')
    print(f"Reprojected shapefile saved to {output_shapefile_path} in EPSG:32643")


# Functionto apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE to an image."""
    if image.dtype != np.uint8:
        # Normalize the image to an 8-bit range
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Create CLAHE object and apply to the image
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

# def reproject_raster(input_raster, output_raster, dst_crs):
#     with rasterio.open(input_raster) as src:
#         transform, width, height = rasterio.warp.calculate_default_transform(
#             src.crs, dst_crs, src.width, src.height, *src.bounds)
#         kwargs = src.meta.copy()
#         kwargs.update({
#             'crs': dst_crs,
#             'transform': transform,
#             'width': width,
#             'height': height
#         })
#         with rasterio.open(output_raster, 'w', **kwargs) as dst:
#             for i in range(1, src.count + 1):
#                 rasterio.warp.reproject(
#                     source=rasterio.band(src, i),
#                     destination=rasterio.band(dst, i),
#                     src_transform=src.transform,
#                     src_crs=src.crs,
#                     dst_transform=transform,
#                     dst_crs=dst_crs,
#                     resampling=rasterio.warp.Resampling.nearest)
#     print(f"Reprojected raster saved at: {output_raster}")

def reproject_raster(input_raster, output_raster, dst_crs, resolution=0.5):
    with rasterio.open(input_raster) as src:
        # Calculate the transform and new dimensions
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=(resolution, resolution)
        )
        
        # Update metadata with new CRS, transform, and resolution
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # Reproject and save the raster
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

# Function to split orthomosaic, apply CLAHE, and run inference using a model
def split_orthomosaic_and_annotations(model, ortho_path, tile_size=640, overlap=100):
    global count  # Assuming you are using this for tracking
    bboxes = []

    with rasterio.open(ortho_path) as src:
        ortho_name = os.path.splitext(os.path.basename(ortho_path))[0]
        transform = src.transform
        num_cols = len(range(0, src.width, tile_size - overlap))
        num_rows = len(range(0, src.height, tile_size - overlap))
        progress_bar = tqdm(total=num_cols * num_rows, desc=f'Processing {ortho_name}')

        # Loop through the orthomosaic in tile-size chunks
        for col_off in range(0, src.width, tile_size - overlap):
            for row_off in range(0, src.height, tile_size - overlap):
                window = rasterio.windows.Window(col_off, row_off, tile_size, tile_size)
                tile_data = src.read(window=window)

                # Skip empty tiles
                if not np.any(tile_data):
                    progress_bar.update(1)
                    continue

                tile_x, tile_y = col_off, row_off

                # Apply CLAHE
                tile_data_clahe = apply_clahe(tile_data[0])

                # Convert tile_data to an image for inference
                tile_image = Image.fromarray(tile_data_clahe)

                # Run YOLO model for object detection
                results = model.predict(tile_image, conf=0.15)

                try:
                    detections = sv.Detections.from_ultralytics(results[0])
                    for obb in detections.data['xyxyxyxy']:
                        class_id = 0  # Assuming a single class
                        polygon = obb.tolist()
                        adjusted_polygon = [(tile_x + x, tile_y + y) for x, y in polygon]
                        poly = Polygon(adjusted_polygon)
                        count += 1

                        # Ensure the polygon is valid before adding
                        if poly.is_valid:
                            bboxes.append([class_id, list(poly.exterior.coords)])
                            print(f"Extracted OBB: {list(poly.exterior.coords)}")

                except Exception as e:
                    print(f"Error in extracting OBB boxes: {e}")

                progress_bar.update(1)

    return bboxes



def coco_to_geojson_polygon(polygon_coordinates, transform):
    """
    Convert COCO-style polygon coordinates to a GeoJSON polygon using the provided transformation.

    Parameters:
        polygon_coordinates (list of tuples): List of (x, y) coordinates representing the polygon.
        transform (Affine): Transformation to be applied to the coordinates.

    Returns:
        Polygon: A Shapely Polygon object with transformed coordinates.
    """
    transformed_polygon = []
    for x, y in polygon_coordinates:
        # Apply the transformation to each coordinate
        x_transformed, y_transformed = transform * (x, y)
        transformed_polygon.append((x_transformed, y_transformed))
    
    # Return the polygon with transformed coordinates
    return Polygon(transformed_polygon)

def adjust_polygon_size(polygon, width=1.1, height=2.1):
    """
    Adjust the size of the given polygon by scaling it based on the specified width and height factors.

    Parameters:
        polygon (Polygon): A Shapely Polygon object to be resized.
        width (float): Scaling factor for the width of the polygon.
        height (float): Scaling factor for the height of the polygon.

    Returns:
        Polygon: A new scaled Shapely Polygon object.
    """
    if not polygon.is_valid:
        raise ValueError("Invalid polygon provided for resizing.")

    centroid = polygon.centroid
    current_width = polygon.bounds[2] - polygon.bounds[0]
    current_height = polygon.bounds[3] - polygon.bounds[1]
    
    if current_width == 0 or current_height == 0:
        raise ValueError("Polygon has zero width or height, cannot scale.")

    # Calculate scaling factors
    scale_x = width / current_width
    scale_y = height / current_height
    
    # Apply scaling to the polygon
    scaled_polygon = scale(polygon, xfact=scale_x, yfact=scale_y, origin=centroid)

    return scaled_polygon

def remove_overlapping_polygons(features, threshold):
    """
    Remove polygons from the feature list that have more than the specified overlap percentage.
    The first polygon from overlapping sets is retained, and subsequent polygons are removed.

    Parameters:
        features (list): List of GeoJSON features containing polygon geometries.
        threshold (float): Percentage of overlap above which polygons are considered redundant.

    Returns:
        list: A list of filtered GeoJSON features with reduced overlaps.
    """
    filtered_features = []
    polygons_to_remove = set()

    # Create an R-tree spatial index for efficient overlap checking
    idx = index.Index()

    # Insert polygons into the index based on their bounding box
    for i, feature in enumerate(features):
        polygon = Polygon(feature['geometry']['coordinates'][0])
        if polygon.is_valid:
            idx.insert(i, polygon.bounds)

    # Iterate through the features and remove overlapping polygons
    for i, feature in enumerate(features):
        if i in polygons_to_remove:
            continue
        
        polygon_i = Polygon(feature['geometry']['coordinates'][0])
        overlap_found = False

        # Get the bounding box of potential overlapping polygons from the index
        potential_overlaps = list(idx.intersection(polygon_i.bounds))

        for j in potential_overlaps:
            if i != j and j not in polygons_to_remove:
                polygon_j = Polygon(features[j]['geometry']['coordinates'][0])
                if polygon_j.is_valid:
                    intersection_area = polygon_i.intersection(polygon_j).area
                    # Remove the polygon if overlap exceeds the threshold
                    if intersection_area / polygon_i.area > threshold or intersection_area / polygon_j.area > threshold:
                        polygons_to_remove.add(j)
                        overlap_found = True
        
        if not overlap_found or i not in polygons_to_remove:
            filtered_features.append(feature)

    return filtered_features

def convert_annotations_to_geojson(geotiff_path, annotations_txt_path, output_geojson_path):
    """
    Convert COCO-style annotations from a text file into a GeoJSON file. Adjusts polygon size and removes overlapping polygons.

    Parameters:
        geotiff_path (str): Path to the GeoTIFF file to get the coordinate reference system (CRS) and transformation.
        annotations_txt_path (str): Path to the text file containing annotations in COCO format.
        output_geojson_path (str): Path to save the output GeoJSON file.
    """
    # Read the GeoTIFF to get the transform and CRS
    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        crs = src.crs

    features = []

    # Read the annotations from the text file
    with open(annotations_txt_path, 'r') as f:
        lines = f.readlines()

    # Process each line of the annotations
    for line in tqdm(lines, desc='Converting annotations'):
        line = line.strip().split()
        class_label = line[0]
        coordinates = list(map(float, line[1:]))
        polygon_coordinates = [(coordinates[i], coordinates[i + 1]) for i in range(0, len(coordinates), 2)]

        # Ensure we have enough points for a valid polygon
        if len(polygon_coordinates) >= 4:
            # Convert COCO polygon to GeoJSON polygon
            polygon = coco_to_geojson_polygon(polygon_coordinates, transform)
            
            # Adjust the polygon size
            polygon = adjust_polygon_size(polygon)

            # Create the feature for the GeoJSON
            feature = {
                'type': 'Feature',
                'geometry': mapping(polygon),
                'properties': {
                    'class_name': class_label
                }
            }
            features.append(feature)

    # Remove overlapping polygons
    filtered_features = remove_overlapping_polygons(features, threshold=0.35)

    # Create the final GeoJSON data
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

    # Save the GeoJSON to the output path
    with open(output_geojson_path, 'w') as f:
        json.dump(geojson_data, f, indent=2)

    print(f"GeoJSON data saved to {output_geojson_path}")

def process_ortho(model, ortho_file, output_dir):
    """
    Processes the orthomosaic, reprojects it if needed, runs inference using YOLO,
    and saves the annotations in a GeoJSON format.

    Parameters:
        model (YOLO): The YOLO model used for object detection.
        ortho_file (str): Path to the orthomosaic file.
        output_dir (str): Directory where results will be saved.

    Returns:
        str: Path to the saved GeoJSON file containing annotations.
    """
    # Set up directories and file names
    filename, _ = os.path.splitext(os.path.basename(ortho_file))
    output_dir = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)

    reproj_raster = os.path.join(output_dir, f"{filename}_reproj_raster.tif")
    annotations_txt_path = os.path.join(output_dir, f"{filename}_annotations.txt")
    output_geojson_path = os.path.join(output_dir, f"{filename}_annotations_new.geojson")

    # Reproject the raster if needed (assuming CRS reprojecting logic)
    with rasterio.open(ortho_file) as src:
        if src.crs.to_epsg() == 4326:
            lon, lat = src.transform[0], src.transform[3]
            utm_zone = utm.from_latlon(lat, lon)[2]
            dst_crs = CRS.from_epsg(32600 + utm_zone)
        else:
            dst_crs = src.crs
        reproject_raster(ortho_file, reproj_raster, dst_crs)

    # Run YOLO inference on orthomosaic tiles and get bounding boxes
    bboxes = split_orthomosaic_and_annotations(model, reproj_raster)

    # Save bounding boxes to a text file for later conversion to GeoJSON
    with open(annotations_txt_path, 'w') as file:
        for annotation in bboxes:
            label = int(annotation[0])
            coordinates = annotation[1]
            coordinates_list = [f"{x:.4f} {y:.4f}" for x, y in coordinates]
            line = f"{label} " + " ".join(coordinates_list)
            file.write(line + '\n')

    # Convert the annotations into a GeoJSON file
    convert_annotations_to_geojson(reproj_raster, annotations_txt_path, output_geojson_path)
    print(f"GeoJSON file saved at: {output_geojson_path}")

    return output_geojson_path

if __name__ == "__main__":

    print(device)
    inference_start=time.time()

    print("Loading YOLO modelll...")
    model = YOLO(YOLO_MODEL_PATH)

    print("Starting inference on orthomosaic...")
    geojson_path = process_ortho(model, ORTHOMOSAIC_PATH, OUTPUT_PATH)
    print("Inference completed.")

    inference_end=time.time()
    inference_time=inference_end-inference_start

    classification_start= time.time()

    # Load the pre-trained ResNet18 model
    resnet_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Modify the final fully connected layer to fit the binary classification (defect/no defect)
    resnet_model.fc = nn.Sequential(
        nn.Linear(resnet_model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)  # 2 classes: defect and no defect
    )

    # Move the model to GPU if available
    resnet_model = resnet_model.to(device)
    resnet_model.load_state_dict(torch.load(model_path))
    resnet_model.eval()  # Set the model to evaluation mode


    # Clip panels and classify them in one step using 'clip_and_classify_panels'
    print("Clipping panels and running classification...")
    updated_geojson_path = clip_and_classify_panels(
        ortho_path=ORTHOMOSAIC_PATH,
        geojson_path=geojson_path,
        temp_dir=temp_dir,
        resnet_model=resnet_model,
        threshold=0.65,
        output_dir=OUTPUT_PATH
    )

    # Load the updated GeoJSON with classification results
    with open(updated_geojson_path, 'r') as f:
        updated_geojson_data = json.load(f)
        crs = updated_geojson_data.get('crs')
        

    # Split the features into defect and non-defect based on the 'prediction' property
    defect_features = []
    non_defect_features = []
    for feature in updated_geojson_data['features']:
        if feature['properties']['prediction'] == 'Defect':
            defect_features.append(feature)
        else:
            non_defect_features.append(feature)

    # Save the features to two separate GeoJSON files with CRS
    save_predictions_to_geojson_and_shapefile(defect_features, defect_output_geojson_path, defect_output_geojson_path.replace('.geojson', '.shp'), crs)
    save_predictions_to_geojson_and_shapefile(non_defect_features, non_defect_output_geojson_path, non_defect_output_geojson_path.replace('.geojson', '.shp'), crs)

    # Clean up temporary files after the predictions
    shutil.rmtree(temp_dir)
    print(f"Defect predictions saved to {defect_output_geojson_path}")
    print(f"Non-defect predictions saved to {non_defect_output_geojson_path}")
    print(f"Temporary directory {temp_dir} deleted.")

    classification_end= time.time()
    classification_time=classification_end-classification_start


    print('totat time taken for inference is:',inference_time)
    print('total time taken for classifcation is:',classification_time)