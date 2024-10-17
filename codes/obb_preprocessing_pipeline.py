import os
from PIL import Image
from tqdm import tqdm
import random
import shutil
import json
from pyproj import Proj, transform
import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, shape, box
from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import Transformer, CRS as pyprojCRS
import utm
import matplotlib.pyplot as plt

# Global variable to store class mappings
type_mapping = {}
next_type_id = 0

def get_filename_from_path(filepath):
    base = os.path.basename(filepath)
    filename, ext = os.path.splitext(base)
    return filename, ext

def calculate_grid_dimensions(grid):
    minx, miny, maxx, maxy = grid.total_bounds
    cell_width_approx = grid.iloc[0].geometry.bounds[2] - grid.iloc[0].geometry.bounds[0]
    cell_height_approx = grid.iloc[0].geometry.bounds[3] - grid.iloc[0].geometry.bounds[1]

    num_columns = round((maxx - minx) / cell_width_approx)
    num_rows = round((maxy - miny) / cell_height_approx)
    
    return num_rows, num_columns

def crop_raster_by_geojson_grid(raster_path, grid_geojson_path, output_folder):
    import numpy as np

    # Load the grid
    grid = gpd.read_file(grid_geojson_path)
    grid = grid.to_crs("EPSG:4326")  # Ensure grid is in EPSG:4326

    # Calculate grid dimensions
    num_rows, num_columns = calculate_grid_dimensions(grid)
    
    # Make sure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the raster
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        
        # Ensure the grid is in the same CRS as the raster
        if str(grid.crs) != raster_crs:
            grid = grid.to_crs(crs=raster_crs)

        raster_name = os.path.splitext(os.path.basename(raster_path))[0]

        # Process each grid cell
        for idx, cell in enumerate(grid.geometry):
            # Adjusted calculation for row and column numbers based on the specified order
            col_number = idx // num_rows + 1
            row_number = (idx % num_rows) + 1
            
            # Crop the raster with the current cell
            try:
                out_image, out_transform = mask(dataset=src, shapes=[cell], crop=True)
                
                # Convert data to uint8 or uint16
                if out_image.dtype == 'float32':
                    out_image = ((out_image - out_image.min()) / (out_image.max() - out_image.min()) * 255).astype(np.uint8)
                elif out_image.dtype == 'float64':
                    out_image = ((out_image - out_image.min()) / (out_image.max() - out_image.min()) * 255).astype(np.uint8)
                else:
                    out_image = out_image.astype(np.uint16)

                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "PNG",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "dtype": out_image.dtype
                })
                
                output_path = os.path.join(output_folder, f"{raster_name}_tile_{(num_rows - row_number)*500}_{(col_number-1)*500}.png")
                
                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(out_image)
                
                print(f"Saved tile: {output_path}")
            
            except ValueError as e:
                print(f"Skipping cell {idx} due to error: {e}")

def create_geojson(coordinates):
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [coordinates]},
            }
        ],
    }
    return geojson_data

def get_raster_crs(raster_path):
    r = rasterio.open(raster_path)
    return f"EPSG:{r.crs.to_epsg()}"

def calculate_extent(coordinates):
    min_x = min(coord[0] for coord in coordinates)
    min_y = min(coord[1] for coord in coordinates)
    max_x = max(coord[0] for coord in coordinates)
    max_y = max(coord[1] for coord in coordinates)
    return box(min_x, min_y, max_x, max_y)

def create_grid_from_geojson_and_raster(geojson_input, raster_input, output_path, 
                                        cell_size_m=50, overlap_m=0):
    output_extent_path = output_path.split('.geojson')[0] + '_extent.geojson'
    
    # Load the raster to get its CRS and extent
    with rasterio.open(raster_input) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds
    
    # Load the GeoJSON file
    vectors_gdf = gpd.read_file(geojson_input)
    total_bounds = vectors_gdf.total_bounds
    min_x, min_y, max_x, max_y = total_bounds
    
    # Ensure the vectors GeoDataFrame is in the same CRS as the raster
    if vectors_gdf.crs != raster_crs:
        vectors_gdf = vectors_gdf.to_crs(raster_crs)

    # Use raster bounds to ensure the grid covers the raster area
    minx, miny, maxx, maxy = raster_bounds

    # Initialize list for grid cells
    grid_cells = []

    # Calculate grid dimensions based on the raster CRS unit (assumed to be meters for projected CRS)
    num_cells_x = int((maxx - minx) / (cell_size_m - overlap_m))
    num_cells_y = int((maxy - miny) / (cell_size_m - overlap_m))

    # Generate grid cells
    for i in range(num_cells_x + 1):  # Cover the edge case of the last partial cell
        for j in range(num_cells_y + 1):  # Cover the edge case of the last partial cell
            left = minx + (cell_size_m - overlap_m) * i
            bottom = miny + (cell_size_m - overlap_m) * j
            right = left + cell_size_m
            top = bottom + cell_size_m
            
            grid_cells.append(box(left, bottom, right, top))

    # Create a GeoDataFrame for the grid
    grid_gdf = gpd.GeoDataFrame({'geometry': grid_cells}, crs=raster_crs)
    
    # Optionally, transform grid to EPSG:4326 if needed for output
    grid_gdf_4326 = grid_gdf.to_crs(epsg=4326)
    
    # Export the grid to GeoJSON
    grid_gdf_4326.to_file(output_path, driver='GeoJSON')

    # Also create and save the extent of the grid as a GeoJSON
    grid_extent = box(min_x, min_y, max_x, max_y)
    grid_extent_gdf = gpd.GeoDataFrame({'geometry': [grid_extent]}, crs='EPSG:4326')
    grid_extent_gdf.to_file(output_extent_path, driver='GeoJSON')

def clip_vectors_by_grid(vectors_geojson, grid_shapefile, output_geojson):
    """
    Clips vector geometries by a grid and saves the intersections as a new GeoJSON file.

    :param vectors_geojson: Path to the input GeoJSON file containing vector geometries.
    :param grid_shapefile: Path to the input shapefile containing grid geometries.
    :param output_geojson: Path to the output GeoJSON file for the resulting geometries.
    """
    # Load the vector data and grid data into GeoDataFrames
    vectors_gdf = gpd.read_file(vectors_geojson)
    grid_gdf = gpd.read_file(grid_shapefile)

    print("Initial Vector CRS is:", vectors_gdf.crs)
    print("Initial Grid CRS is:", grid_gdf.crs)
    
    # Ensure both GeoDataFrames use the same coordinate reference system (CRS)
    if vectors_gdf.crs != grid_gdf.crs:
        grid_gdf = grid_gdf.to_crs(vectors_gdf.crs)
        print("Transformed Grid CRS to match Vector CRS:", grid_gdf.crs)
    
    # Debugging: Print the bounds of vectors and grid
    print("Vector Bounds:", vectors_gdf.total_bounds)
    print("Grid Bounds:", grid_gdf.total_bounds)
    
    # Perform the intersection operation
    intersected_gdf = gpd.overlay(vectors_gdf, grid_gdf, how='intersection')
    
    # Debugging: Print the bounds of the intersection result
    print("Intersected Bounds:", intersected_gdf.total_bounds)

    # Save the intersected geometries to a new GeoJSON file
    intersected_gdf.to_file(output_geojson, driver='GeoJSON')
    intersected_gdf.to_crs(epsg=4326)
    print(f"Saved intersected vectors to: {output_geojson}")

def get_epsg_from_tif(tif_path):
    with rasterio.open(tif_path) as src:
        crs = src.crs
        if crs:
            return CRS.from_string(src.crs.to_string())
        else:
            return None

def wgs84_to_epsg(longitude, latitude, raster_crs):
    # Create a Point geometry with the WGS84 coordinate
    wgs84_coordinate = Point(longitude, latitude)
 
    # Create a GeoDataFrame with a single row containing the WGS84 coordinate
    gdf = gpd.GeoDataFrame(geometry=[wgs84_coordinate], crs="EPSG:4326")

    # Convert the coordinate to the CRS of the raster
    gdf_raster_crs = gdf.to_crs(raster_crs)

    # Access the converted coordinate
    raster_crs_coordinates = [(point.x, point.y) for point in gdf_raster_crs.geometry]

    return raster_crs_coordinates[0]

def remapping_json(raster_data, mapping):
    """
    Parameters:
        file (str): Path to the input GeoJSON file.
        mapping (str): Path to the class mapping text file.
    Returns:
        geojson_data (dict): Updated GeoJSON data with remapped properties.
    Description:
        This function reads a GeoJSON file, remaps properties, saves the remapping information to a text file, and then saves the updated GeoJSON data to a new file.
    """

    # Path to the class mapping text file
    class_mapping_path = mapping

    global type_mapping
    global next_type_id

    # Initialize an empty list to store the mapping information
    mapping_info = []
    # # Iterate through features in the GeoJSON
    for feature in raster_data['features']:
        feature_properties = feature["properties"]["properties"]
        # del(feature_properties['path'])
        feature_type = feature_properties["class_name"]

        if feature_type is not None:
            if feature_type not in type_mapping:
                # Assign the next available integer ID to the feature type
                type_mapping[feature_type] = next_type_id
                next_type_id += 1

            # Update the property "Type" with the assigned integer ID
            feature_properties["class"] = type_mapping[feature_type]
        

    # Create the class mapping text
    for feature_type, type_id in type_mapping.items():
        mapping_info.append(f"{type_id} : {feature_type}")

    # Write the class mapping to a text file
    with open(class_mapping_path, "w") as mapping_file:
        mapping_file.write("\n".join(mapping_info))

    print("Class mapping created and saved as 'class_mapping.txt'")
    return raster_data

def polygon_to_yolo(polygon_coordinates, input_tif, extent):
    """
    Parameters:
        polygon_coordinates (list of tuples): List of (x, y) coordinates representing a polygon.
        input_tif (str): Path to the input TIFF file.
        extent (list): contains the extent of TIFF file.
    Returns:
        Tuple containing the oriented bounding box (OBB) format: class,x1,y1,x2,y2,x3,y3,x4,y4.
    Description:
        Converts polygon coordinates to YOLO OBB format relative to a TIFF file's extent.
    """
   
    with rasterio.open(input_tif) as dataset:
        # Extract the resolution of the dataset
        resolution = dataset.res[0]
         
    minx, miny, maxx, maxy = extent

    tif_extent = {
        "x_min": minx,  # Replace with the actual extent values
        "x_max": maxx,
        "y_min": miny,
        "y_max": maxy,
        "resolution": resolution,
    }

    # Subtract the TIFF extent to get coordinates relative to the TIFF
    obb_coords = []
    for coord in polygon_coordinates:
        x, y = coord
        x_rel = (x - tif_extent["x_min"]) / tif_extent["resolution"]
        y_rel = (tif_extent["y_max"] - y) / tif_extent["resolution"]
        obb_coords.extend([x_rel, y_rel])

    # Remove the duplicate coordinate if it is the same as the first one
    if obb_coords[0] == obb_coords[-1]:
        obb_coords.pop()

    return obb_coords

def process_annotations(annotations_file, geojson, input_tif, mapping, extent):
    """
    Parameters:
        annotations_file (str): Path to the output YOLO annotation file.
        geojson_file (str): Path to the input GeoJSON file.
        input_tif (str): Path to the input TIFF file.
        mapping (str): Path to the class mapping text file.
    Description:
        Processes GeoJSON annotations, converts them to YOLO OBB format, and writes them to a file.
    """

    raster_data = remapping_json(geojson, mapping)
    # Process and convert to YOLO format
    with open(annotations_file, "w") as yolo_file:
        
        for feature in raster_data['features']:

            polygon = []
            polygon_coordinates = feature["geometry"]["coordinates"][0]
            
            for polygon_coordinate in polygon_coordinates:
                # Extract longitude and latitude from polygon coordinate
                longitude, latitude, z = polygon_coordinate
                polygon.append(wgs84_to_epsg(longitude, latitude, get_epsg_from_tif(input_tif)))
                
            # Calculate YOLO OBB coordinates
            obb_coords = polygon_to_yolo(polygon, input_tif, extent)

            # Extract the class label from properties
            class_label = feature["properties"]["properties"]["class"]

            # Write YOLO annotation to the text file
            yolo_file.write(f"{class_label} " + " ".join(map(str, obb_coords)) + "\n")

def split_existing_tiles_with_annotations(
    tile_dir,
    annotations_path,
    output_tile_dir,
    output_annotation_dir,
    false_positive_prob=0.2  # Probability of adding a false positive annotation
):
    """Parameters:
        tile_dir (str): Directory containing existing tile images.
        annotations_path (str): Path to the annotations file.
        output_tile_dir (str): Directory to save the output tile images.
        output_annotation_dir (str): Directory to save the output annotations.
        false_positive_prob (float): Probability of adding a false positive annotation.
    """
    classes = []
    tile_files = [f for f in os.listdir(tile_dir) if f.endswith('.png') and os.path.isfile(os.path.join(tile_dir, f))]
    for tile_file in tqdm(tile_files, desc="Processing Tiles"):
        tile_path = os.path.join(tile_dir, tile_file)
        tile_name, _ = os.path.splitext(os.path.basename(tile_file))

        annotation_file = os.path.join(output_annotation_dir, f"{tile_name}.txt")

        # Extract tile position from filename
        parts = tile_name.split('_')
        tile_y, tile_x = int(parts[-2]), int(parts[-1].split('.png')[0])

        # Load tile image
        tile_image = Image.open(tile_path)

        # Check if tile image is empty
        if np.array(tile_image).sum() == 0:
            continue  # Skip empty tiles

        # Save tile image
        out_tile_file = os.path.join(output_tile_dir, tile_file)
        tile_image.save(out_tile_file)

        # Read annotations from the annotations file
        with open(annotations_path, "r") as annotations_file:
            tile_annotations = []

            for line in annotations_file:
                parts = line.strip().split()
                class_name = parts[0]
                coords = list(map(float, parts[1:]))

                # Check if annotation falls within the boundaries of the current tile
                if all((tile_x <= coords[i] < tile_x + tile_image.size[0]) and (tile_y <= coords[i + 1] < tile_y + tile_image.size[1]) for i in range(0, len(coords), 2)):
                    classes.append(class_name)
                    adjusted_coords = [coords[i] - tile_x if i % 2 == 0 else coords[i] - tile_y for i in range(len(coords))]
                    tile_annotations.append(f"{class_name} " + " ".join(map(str, adjusted_coords)))

            # Add false positive annotation with a certain probability if no annotations found
            if not tile_annotations and random.random() < false_positive_prob:
                tile_annotations.append(" ")

            if tile_annotations:
                with open(annotation_file, "w") as tile_annotations_file:
                    tile_annotations_file.write("\n".join(tile_annotations))
                print(f"Saved annotation: {annotation_file}")

        if not tile_annotations:
            os.remove(out_tile_file)

def normalisation(output_annotation_dir, root_dir, image_width, image_height):
    """
    Parameters:
        output_annotation_dir (str): Directory containing annotation files.
        root_dir (str): Root directory.
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.
    Description:
        Normalizes the annotation files to match the image dimensions.
    """
    # List of input label files
    input_files = os.listdir(output_annotation_dir)  # Add your file names

    for input_file in input_files:
        img_path = os.path.join(root_dir, "Images", input_file.replace('.txt', '.png'))
        if not os.path.exists(img_path):
            print(f"Image file not found: {img_path}")
            continue
        
        output_file = os.path.join(root_dir, "norm_output_annotations", input_file)
        
        img = Image.open(img_path)
        image_width, image_height = img.size
        
        # Read the labels in xywh format from the input file
        with open(os.path.join(output_annotation_dir, input_file), "r") as file:
            labels = file.readlines()

        # Convert labels to normalized format
        normalized_labels = []

        for label in labels:
            parts = label.split()
            if len(parts) == 0:
                continue
            class_name = parts[0]
            coords = list(map(float, parts[1:]))
            normalized_coords = [coords[i] / image_width if i % 2 == 0 else coords[i] / image_height for i in range(len(coords))]
            normalized_label = f"{class_name} " + " ".join(map(str, normalized_coords)) + "\n"
            normalized_labels.append(normalized_label)

        # Write the normalized labels to the output file
        with open(output_file, "w") as file:
            file.writelines(normalized_labels)

def merge_images_and_labels(src_folder, dest_folder):
    """
    Parameters:
        src_folder (str): Source folder containing images and labels.
        dest_folder (str): Destination folder to merge images and labels.
    Description:
        Merges images and labels from a source folder into a destination folder.
    """
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Iterate through the source folder and its subfolders
    for root, dirs, files in os.walk(src_folder):
        rel_path = os.path.relpath(root, src_folder)

        # Create corresponding destination subfolders
        dest_images_folder = os.path.join(dest_folder, "images")
        dest_labels_folder = os.path.join(dest_folder, "labels")

        os.makedirs(dest_images_folder, exist_ok=True)
        os.makedirs(dest_labels_folder, exist_ok=True)

        for file in files:
            # Create the source and destination paths
            src_path = os.path.join(root, file)

            if file.endswith(".png") and "Images" in rel_path:
                dest_path = os.path.join(dest_images_folder, file)
                try:
                    shutil.copy2(src_path, dest_path)
                except:
                    print(file)
            elif file.endswith(".txt") and "norm_output_annotations" in rel_path:
                dest_path = os.path.join(dest_labels_folder, file)
                try:
                    shutil.copy2(src_path, dest_path)
                except:
                    print(file)

def train_val_test_split(
    image_label_folder,
    output_folder,
    train_ratio=0.7,
    val_ratio=0.3,
    test_ratio=0,
):
    """
    Parameters:
        image_folder (str): Folder containing images.
        label_folder (str): Folder containing labels.
        output_folder (str): Output folder to save train, validation, and test sets.
        train_ratio (float): Ratio of data for training.
        val_ratio (float): Ratio of data for validation.
        test_ratio (float): Ratio of data for testing.
    Description:
        Splits the dataset into train, validation, and test sets based on the specified ratios.
    """
    # Create output folders for train, validation, and test sets
    train_output_folder = os.path.join(output_folder, "train")
    val_output_folder = os.path.join(output_folder, "val")
    test_output_folder = os.path.join(output_folder, "test")

    os.makedirs(train_output_folder, exist_ok=True)
    os.makedirs(val_output_folder, exist_ok=True)
    os.makedirs(test_output_folder, exist_ok=True)
    image_folder = os.path.join(image_label_folder, 'images')
    label_folder = os.path.join(image_label_folder, 'labels')
    # Get the list of image files
    image_files = [file for file in os.listdir(image_folder) if file.endswith(".png")]

    # Shuffle the image files randomly
    random.shuffle(image_files)

    # Calculate the number of samples for each set
    num_samples = len(image_files)
    num_train = int(train_ratio * num_samples)
    num_val = int(val_ratio * num_samples)
    print(num_train)
    # Split the image files into train, validation, and test sets
    train_files = image_files[:num_train]
    val_files = image_files[num_train: num_train + num_val]
    test_files = image_files[num_train + num_val:]

    # Move image files and corresponding label files to the respective output directories
    move_files(train_files, image_folder, label_folder, train_output_folder)
    move_files(val_files, image_folder, label_folder, val_output_folder)
    move_files(test_files, image_folder, label_folder, test_output_folder)

    # Display the number of instances for each class in each split
    print("Train Set:")
    display_class_instances(os.path.join(train_output_folder, "labels"))
    print("\nValidation Set:")
    display_class_instances(os.path.join(val_output_folder, "labels"))
    print("\nTest Set:")
    display_class_instances(os.path.join(test_output_folder, "labels"))

    print("\nTrain-Validation-Test split completed successfully!")

def move_files(files, image_folder, label_folder, output_folder):
    """
    Parameters:
        files (list): List of files to move.
        image_folder (str): Folder containing images.
        label_folder (str): Folder containing labels.
        output_folder (str): Destination folder to move files.
    Description:
        Moves files from source folders to destination folders.
    """
    images_output_folder = os.path.join(output_folder, "images")
    labels_output_folder = os.path.join(output_folder, "labels")
    os.makedirs(images_output_folder, exist_ok=True)
    os.makedirs(labels_output_folder, exist_ok=True)

    for file in files:
        image_file_path = os.path.join(image_folder, file)
        label_file_path = os.path.join(label_folder, file.replace(".png", ".txt"))

        output_image_file_path = os.path.join(images_output_folder, file)
        output_label_file_path = os.path.join(
            labels_output_folder, file.replace(".png", ".txt")
        )
        print(image_file_path, output_image_file_path)
        print(label_file_path, output_label_file_path)
        shutil.copy(image_file_path, output_image_file_path)
        shutil.copy(label_file_path, output_label_file_path)

def count_class_instances(label_folder):
    """
    Parameters:
        label_folder (str): Folder containing label files.
    Returns:
        instances (dict): Dictionary containing class index and instance count.
    Description:
        Counts the number of instances for each class in label files.
    """
    instances = {}
    for file in os.listdir(label_folder):
        label_file_path = os.path.join(label_folder, file)
        with open(label_file_path, "r") as label_file:
            lines = label_file.readlines()

        for line in lines:
            class_index, *_ = line.split()
            if class_index not in instances:
                instances[class_index] = 0
            instances[class_index] += 1

    return instances

def display_class_instances(label_folder):
    """
    Parameters:
        label_folder (str): Folder containing label files.
    Description:
        Displays the number of instances for each class in label files.
    """
    instances = count_class_instances(label_folder)
    for class_index, count in instances.items():
        print(f"Class Index: {class_index}\tInstances: {count}")

def convert_geojson(intersected_vectors, converted_geojson):
    with open(intersected_vectors, 'r') as fp:
        new_payload = json.load(fp)

    # Extract the 'features' list from the GeoJSON data
    features_list = new_payload.get("features", [])

    # Iterate through each feature and convert properties to a list of dictionaries
    raster_features = []
    for feature in features_list:
        properties_dict = feature.get("properties", {})
        raster_path = properties_dict['raster_path']
        properties_list = [{"key": prop_name, "value": prop_value} for prop_name, prop_value in properties_dict.items()]
        properties_dict.clear()  # Clear existing properties dictionary
        for prop in properties_list:
            properties_dict[prop["key"]] = prop["value"]
        raster_features.append({
            "type": "Feature",
            "properties": properties_dict,
            "geometry": feature["geometry"]
        })

    # Create a new GeoJSON-like structure with raster_features
    converted_geojson_data = {
        "type": "FeatureCollection",
        "crs": new_payload.get("crs", {}),
        "raster_features": [{"raster_path": raster_path, "features": raster_features}]
    }

    # Save the modified GeoJSON data
    with open(converted_geojson, 'w') as outfile:
        json.dump(converted_geojson_data, outfile)

    return converted_geojson_data

def create_yaml(class_mapping_file, output_yaml_file, output_path):
    """
    Parameters:
        class_mapping_file (str): Path to the class mapping file.
        output_yaml_file (str): Path to the output YAML file.
        output_path (str): Output path for the YAML file.
    Description:
        Creates a YAML file containing class mapping information and paths for YOLO format.
    """
    # Read class mapping file and generate dictionary
    class_mapping = {}
    with open(class_mapping_file, "r") as f:
        for line in f:
            parts = line.strip().split(" : ")
            if len(parts) == 2:
                class_id, class_name = parts
                class_mapping[int(class_id)] = class_name
    # Write YAML content
    yaml_content = f"names:\n"
    for idx, name in class_mapping.items():
        yaml_content += f"  {idx} : '{name}'\n"
    yaml_content += f"path: {output_path}/yolo_split\n"
    yaml_content += f"test: test/images\n"
    yaml_content += f"train: train/images\n"
    yaml_content += f"val: val/images\n"

    # Write YAML to file
    with open(output_yaml_file, "w") as f:
        f.write(yaml_content)

def reproject_raster(src_path, dst_path, dst_crs, resolution=0.1):
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=resolution
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'driver': 'GTiff'
        })

        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )

def preprocess_data(raster_path, geojson_path, output_dir):
    extent = {}
    extent_v2 = {}

    CLASS_MAPPING_FOLDER = os.path.join(output_dir, "class-mapping")
    geojsons = os.path.join(output_dir, "geojsons")

    os.makedirs(geojsons, exist_ok=True)

    converted_geojson_data_template = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": " "}},
        'raster_features': []
    }

    converted_geojson_data = converted_geojson_data_template.copy()
    converted_geojson_data_v2 = converted_geojson_data_template.copy()

    extent_coordinates = []

    # Create a list for the GeoDataFrame
    features = []

    # Extract annotations and add them to the feature list
    with open(geojson_path, 'r') as f:
        annotations = json.load(f)
    for annotation in annotations['features']:
        feature = {
            "type": "Feature",
            "properties": annotation['properties'],
            "geometry": shape(annotation['geometry'])
        }
        features.append(feature)

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(features, crs='EPSG:4326')

    gdf['raster_path'] = raster_path

    # Save the GeoDataFrame to a GeoJSON file explicitly mentioning the CRS as EPSG:4326
    payload_v2 = os.path.join(geojsons, f"payload_annotations#{os.path.basename(raster_path)}.geojson")
    gdf.to_file(payload_v2, driver='GeoJSON')
    
    straight_grid_geojson = os.path.join(geojsons, f"straight_grid#{os.path.basename(raster_path)}.geojson")
    intersected_vectors = os.path.join(geojsons, f"intersected_vectors_new#{os.path.basename(raster_path)}.geojson")
    converted_geojson = os.path.join(geojsons, "final_geojson.geojson")
    converted_geojson_v2 = os.path.join(geojsons, "final_geojson_v2.geojson")
    
    create_grid_from_geojson_and_raster(
        payload_v2, raster_path, straight_grid_geojson
    )

    clip_vectors_by_grid(
        payload_v2, straight_grid_geojson, intersected_vectors
    )

    with open(straight_grid_geojson.split('.geojson')[0] + '_extent.geojson') as f:
        grid_geojson_data = json.load(f)
     
    for raster_infos in grid_geojson_data['features']:
        extent_coordinates.append(raster_infos['geometry']['coordinates'][0])
    extent[raster_path] = extent_coordinates[0]
    
    if not converted_geojson_data['raster_features']:
        converted_geojson_data = convert_geojson(intersected_vectors, converted_geojson)
    else:
        converted_data = convert_geojson(intersected_vectors, converted_geojson)
        
        converted_geojson_data['raster_features'].append(converted_data['raster_features'][0])

    with open(converted_geojson, 'w') as outfile:
        json.dump(converted_geojson_data, outfile)

    raster_info = converted_geojson_data['raster_features'][0]
    input_tif = raster_info['raster_path']
    raster_name, raster_ext = get_filename_from_path(input_tif)

    print("Processing raster:", f"{raster_name}{raster_ext}")

    root_dir = os.path.join(CLASS_MAPPING_FOLDER, raster_name)
    annotations_file = os.path.join(root_dir, "annotations.txt")
    inference = os.path.join(root_dir, "inference")
    normal_annotations = os.path.join(root_dir, "norm_output_annotations")
    output_tile_dir = os.path.join(root_dir, "Images")
    output_annotation_dir = os.path.join(root_dir, "annotations")
    reproj_raster = os.path.join(root_dir, f"{raster_name}_reproj_raster{raster_ext}")

    output_tiles = os.path.join(root_dir, "tiles")

    class_mapping_file = os.path.join(root_dir, "class_mapping.txt")

    os.makedirs(output_tile_dir, exist_ok=True)
    os.makedirs(output_annotation_dir, exist_ok=True)
    os.makedirs(normal_annotations, exist_ok=True)
    os.makedirs(inference, exist_ok=True)
    os.makedirs(output_tiles, exist_ok=True)

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    if raster_crs.to_epsg() == 4326:
        lon, lat = src.bounds.left, src.bounds.top
        utm_zone = utm.from_latlon(lat, lon)[2]
        dst_crs = f'EPSG:326{utm_zone}'
    else:
        dst_crs = raster_crs

    reproject_raster(raster_path, reproj_raster, dst_crs)

    if extent[input_tif]:
        selected_area = extent[input_tif]
        print("Clipping Ortho...")
        raster_crs = get_raster_crs(reproj_raster)
        geojson_data = create_geojson(selected_area)
        output_file = os.path.join(geojsons, "mask.geojson")
        with open(output_file, "w") as file:
            json.dump(geojson_data, file, sort_keys=True)

        polygon_coords = geojson_data["features"][0]["geometry"]["coordinates"][0]
        in_proj = pyprojCRS.from_epsg(4326)
        out_proj = pyprojCRS.from_string(raster_crs)

        transformer = Transformer.from_crs(in_proj, out_proj, always_xy=True)
        transformed_coords = [transformer.transform(lon, lat) for lon, lat in polygon_coords]
        
        # Create a Shapely polygon
        polygon = Polygon(transformed_coords)

        # Calculate the area in square meters
        total_area = polygon.area / 1000000
        reproj_file_name, reproj_file_ext = get_filename_from_path(reproj_raster)
        clip_raster = os.path.join(inference, f"{reproj_file_name}_clip{reproj_file_ext}")

        with rasterio.open(reproj_raster) as src:
            profile = src.profile
            profile.update({'driver': 'GTiff'})
            with rasterio.open(clip_raster, 'w', **profile) as dst:
                out_image, out_transform = mask(src, [polygon], crop=True)
                dst.write(out_image)
                dst.transform = out_transform

    else:
        clip_raster = reproj_raster
        with rasterio.open(clip_raster) as src:
            resolution_x = src.res[0]
            resolution_y = src.res[1]

            num_rows = src.height
            num_cols = src.width

            total_area = num_rows * num_cols * resolution_x * resolution_y / 1000000

    r = rasterio.open(clip_raster)
    count = r.count
    r.close()  # Ensure the file is closed before attempting to overwrite

    # if 4 band raster then convert to 3 band
    if count == 4:
        clip_file_name, clip_file_ext = get_filename_from_path(clip_raster)
        three_band_raster = os.path.join(inference, f"{clip_file_name}_3band{clip_file_ext}")
       
        with rasterio.open(clip_raster) as src:
            profile = src.profile
            profile.update(count=3)

            with rasterio.open(three_band_raster, 'w', **profile) as dst:
                for i in range(1, 4):
                    dst.write(src.read(i), i)
    elif count == 3:
        three_band_raster = clip_raster
    elif count == 1:
        one_band_raster = clip_raster
    else:
        raise ValueError("Raster doesn't have 3 bands or can't be converted to 3 bands")

    payload_v2 = f"{payload_v2.split('#')[0]}#{os.path.basename(input_tif)}.geojson"
    straight_grid_geojson = f"{straight_grid_geojson.split('#')[0]}#{os.path.basename(input_tif)}.geojson"
    intersected_vectors = f"{intersected_vectors.split('#')[0]}#{os.path.basename(input_tif)}.geojson"
    output_extent_path = straight_grid_geojson.split('.geojson')[0] + '_extent.geojson'

    create_grid_from_geojson_and_raster(payload_v2, one_band_raster, straight_grid_geojson)
    
    extent_gdf = gpd.read_file(straight_grid_geojson)
    total_bounds = extent_gdf.total_bounds
    min_x, min_y, max_x, max_y = total_bounds
    
    grid_extent = box(min_x, min_y, max_x, max_y)
    grid_extent_gdf = gpd.GeoDataFrame({'geometry': [grid_extent]}, crs='EPSG:4326')
    grid_extent_gdf.to_file(output_extent_path, driver='GeoJSON')

    raster_info = converted_geojson_data['raster_features'][0]
    input_tif = raster_info['raster_path']
    raster_name, raster_ext = get_filename_from_path(input_tif)

    print("Processing raster:", f"{raster_name}{raster_ext}")

    # Download raster for processing
    raster_local_path = raster_path

    root_dir = os.path.join(CLASS_MAPPING_FOLDER, raster_name)
    annotations_file = os.path.join(root_dir, "annotations.txt")
    inference = os.path.join(root_dir, "inference")
    normal_annotations = os.path.join(root_dir, "norm_output_annotations")
    output_tile_dir = os.path.join(root_dir, "Images")
    output_annotation_dir = os.path.join(root_dir, "annotations")
    reproj_raster = os.path.join(root_dir, f"{raster_name}_reproj_raster{raster_ext}")

    output_tiles = os.path.join(root_dir, "tiles")

    class_mapping_file = os.path.join(root_dir, "class_mapping.txt")
    
    tif_crs = rasterio.open(raster_local_path)

    if tif_crs.crs.to_epsg() == 4326:
        lon, lat = tif_crs.bounds.left, tif_crs.bounds.top
        utm_zone = utm.from_latlon(lat, lon)[2]
        dst_crs = f'EPSG:326{utm_zone}'
    else:
        dst_crs = tif_crs.crs

    reproject_raster(raster_local_path, reproj_raster, dst_crs)

    straight_grid_geojson = f"{straight_grid_geojson.split('#')[0]}#{os.path.basename(input_tif)}.geojson"

    with open(straight_grid_geojson.split('.geojson')[0] + '_extent.geojson') as f:
        grid_geojson_datas = json.load(f)

    for extent_info in grid_geojson_datas['features']:
        extent_coordinates.append(extent_info['geometry']['coordinates'][0])
    
    extent_v2[input_tif] = extent_coordinates[0]

    if extent_v2[input_tif]:
        selected_area = extent_v2[input_tif]
        print("Clipping Ortho...")
        raster_crs = get_raster_crs(reproj_raster)
        geojson_data = create_geojson(selected_area)
        output_file = os.path.join(geojsons, "mask.geojson")
        with open(output_file, "w") as file:
            json.dump(geojson_data, file)

        polygon_coords = geojson_data["features"][0]["geometry"]["coordinates"][0]
        in_proj = pyprojCRS.from_epsg(4326)
        out_proj = pyprojCRS.from_string(raster_crs)

        transformer = Transformer.from_crs(in_proj, out_proj, always_xy=True)
        transformed_coords = [transformer.transform(lon, lat) for lon, lat in polygon_coords]
        
        # Create a Shapely polygon
        polygon = Polygon(transformed_coords)

        # Calculate the area in square meters
        total_area = polygon.area / 1000000
        reproj_file_name, reproj_file_ext = get_filename_from_path(reproj_raster)
        print(reproj_file_name)
        clip_raster = os.path.join(inference, f"{reproj_file_name}_clip{reproj_file_ext}")

        with rasterio.open(reproj_raster) as src:
            profile = src.profile
            profile.update({'driver': 'GTiff'})
            with rasterio.open(clip_raster, 'w', **profile) as dst:
                out_image, out_transform = mask(src, [polygon], crop=True)
                dst.write(out_image)
                dst.transform = out_transform

    else:
        clip_raster = reproj_raster
        with rasterio.open(clip_raster) as src:
            resolution_x = src.res[0]
            resolution_y = src.res[1]

            num_rows = src.height
            num_cols = src.width

            total_area = num_rows * num_cols * resolution_x * resolution_y / 1000000


    r = rasterio.open(clip_raster)
    count = r.count
    r.close()  # Ensure the file is closed before attempting to overwrite

    # if 4 band raster then convert to 3 band
    if count == 4:
        clip_file_name, clip_file_ext = get_filename_from_path(clip_raster)
        three_band_raster = os.path.join(inference, f"{clip_file_name}_3band{clip_file_ext}")
       
        with rasterio.open(clip_raster) as src:
            profile = src.profile
            profile.update(count=3)

            with rasterio.open(three_band_raster, 'w', **profile) as dst:
                for i in range(1, 3):
                    dst.write(src.read(i), i)
    elif count == 3:
        three_band_raster = clip_raster
    elif count == 1:
        one_band_raster = clip_raster
    else:
        raise ValueError("Raster doesn't have 3 bands or can't be converted to 3 bands")
    
    payload_v2 = f"{payload_v2.split('#')[0]}#{os.path.basename(input_tif)}.geojson"
    intersected_vectors = f"{intersected_vectors.split('#')[0]}#{os.path.basename(input_tif)}.geojson"

    crop_raster_by_geojson_grid(reproj_raster, straight_grid_geojson, output_tiles)

    clip_vectors_by_grid(
        payload_v2, straight_grid_geojson, intersected_vectors
    )

    if not converted_geojson_data_v2['raster_features']:
        converted_geojson_data_v2 = convert_geojson(intersected_vectors, converted_geojson_v2)
    else:
        converted_data = convert_geojson(intersected_vectors, converted_geojson_v2)
        
        converted_geojson_data_v2['raster_features'].append(converted_data['raster_features'][0])

    with open(converted_geojson_v2, 'w') as outfile:
        json.dump(converted_geojson_data_v2, outfile)

    for raster_info in converted_geojson_data_v2['raster_features']:
        input_tif = raster_info['raster_path']
        raster_name, raster_ext = get_filename_from_path(input_tif)


        print("Processing raster:", f"{raster_name}{raster_ext}")

        root_dir = os.path.join(CLASS_MAPPING_FOLDER, raster_name)
        annotations_file = os.path.join(root_dir, "annotations.txt")
        inference = os.path.join(root_dir, "inference")
        normal_annotations = os.path.join(root_dir, "norm_output_annotations")
        output_tile_dir = os.path.join(root_dir, "Images")
        output_annotation_dir = os.path.join(root_dir, "annotations")
        one_band_raster = os.path.join(inference, f"{raster_name}_reproj_raster_clip{raster_ext}")

        output_tiles = os.path.join(root_dir, "tiles")

        class_mapping_file = os.path.join(root_dir, "class_mapping.txt")

        with rasterio.open(one_band_raster) as src:
            crs = src.crs
            bound = src.bounds
            print(bound)

        straight_grid_geojson = f"{straight_grid_geojson.split('#')[0]}#{os.path.basename(input_tif)}.geojson"
        gdf = gpd.read_file(straight_grid_geojson)
        gdf = gdf.to_crs(crs)
        gdf.to_file(straight_grid_geojson, driver='GeoJSON')

        total_bounds = gdf.total_bounds

        print("total bound is", total_bounds)
    
        process_annotations(annotations_file, raster_info, reproj_raster, class_mapping_file, total_bounds)
    
        split_existing_tiles_with_annotations(output_tiles, annotations_file, output_tile_dir, output_annotation_dir)
        normalisation(output_annotation_dir, root_dir, 500, 500)

        merge_images_and_labels(root_dir, output_dir)

    # Specify the output folder
    output_folder_path = os.path.join(output_dir, "yolo_split")

    # Perform train-validation-test split
    train_val_test_split(output_dir, output_folder_path)

    output_yaml_file = os.path.join(output_dir, "yolo_split.yaml")

    create_yaml(class_mapping_file, output_yaml_file, output_dir)

    return output_yaml_file

# Parameters
raster_path = r"D:\Rishabh Uikey\kochi solar\ortho\Thermal Orthomosaic_32643.tif"
geojson_path = r"D:\Rishabh Uikey\kochi solar\Kochi solar\panel1.geojson"
output_dir = r"D:\Rishabh Uikey\kochi solar\clipped"

# Preprocess data
preprocess_data(raster_path, geojson_path, output_dir)
