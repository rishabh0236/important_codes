import os
import json
from PIL import Image
import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon, mapping
from shapely.affinity import scale, translate
from rasterio.warp import calculate_default_transform, reproject, Resampling
from ultralytics import YOLO
import geopandas as gpd
import supervision as sv
import utm
from rasterio.crs import CRS
import cv2
import matplotlib.pyplot as plt
from rtree import index  # Import the rtree index for spatial indexing

# Use environment variables for paths
OUTPUT_PATH = os.getenv('OUTPUT_PATH', './output')  # Default to ./output
YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', './model/best.pt')  # Default to ./model/best.pt
ORTHOMOSAIC_PATH = os.getenv('ORTHOMOSAIC_PATH', './ortho/ortho.tif')  # Default to ./ortho/ortho.tif

count = 0

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
    global count  # Declare count as a global variable
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
                        class_id = 0  # Assuming all panels are class 0
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
                    polygons_to_remove.add(j)  # Mark the second polygon for removal
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

if __name__ == "__main__":
    print("Loading model...")
    model = YOLO(YOLO_MODEL_PATH)

    print("Starting inference...")
    process_ortho(model, ORTHOMOSAIC_PATH)

    print("Inference completed.")
