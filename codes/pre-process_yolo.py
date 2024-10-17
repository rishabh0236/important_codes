import os
import cv2
from tqdm import tqdm

def apply_clahe(image, clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to a grayscale image.
    
    Parameters:
        image (numpy.ndarray): The input grayscale image.
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (tuple): Size of grid for histogram equalization.
        
    Returns:
        numpy.ndarray: The image after CLAHE.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def apply_hist_equalization(image):
    """
    Applies Histogram Equalization to a grayscale image.
    
    Parameters:
        image (numpy.ndarray): The input grayscale image.
        
    Returns:
        numpy.ndarray: The image after histogram equalization.
    """
    return cv2.equalizeHist(image)

def preprocess_images(image_dir, output_dir, method='clahe', clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Preprocess grayscale images to enhance contrast and save them to the output directory.
    
    Parameters:
        image_dir (str): Directory containing the input images.
        output_dir (str): Directory to save the processed images.
        method (str): The method to use for contrast enhancement ('clahe' or 'hist_eq').
        clip_limit (float): Threshold for contrast limiting (for CLAHE).
        tile_grid_size (tuple): Size of grid for histogram equalization (for CLAHE).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in tqdm(image_files, desc=f"Processing Images in {image_dir}"):
        image_path = os.path.join(image_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        
        # Read the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load image: {image_file}")
            continue
        
        # Enhance contrast using the specified method
        if method == 'clahe':
            processed_image = apply_clahe(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        elif method == 'hist_eq':
            processed_image = apply_hist_equalization(image)
        else:
            raise ValueError("Unsupported method. Use 'clahe' or 'hist_eq'.")
        
        # Save the processed image
        cv2.imwrite(output_path, processed_image)

def process_yolo_split(base_dir, output_base_dir, method='clahe', clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Process images in YOLO split directories (train, test, val) and their respective image folders.
    
    Parameters:
        base_dir (str): The base directory containing the YOLO split directories.
        output_base_dir (str): The base directory to save the processed images.
        method (str): The method to use for contrast enhancement ('clahe' or 'hist_eq').
        clip_limit (float): Threshold for contrast limiting (for CLAHE).
        tile_grid_size (tuple): Size of grid for histogram equalization (for CLAHE).
    """
    for split in ['train', 'test', 'val']:
        image_dir = os.path.join(base_dir, split, 'images')
        output_dir = os.path.join(output_base_dir, split, 'images')
        
        if os.path.exists(image_dir):
            preprocess_images(image_dir, output_dir, method=method, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        else:
            print(f"Directory not found: {image_dir}")

if __name__ == "__main__":
    base_dir = r"D:\new_data\100 cell\split"
    output_base_dir = r"D:\new_data\100 cell\yolo_split-preprocess"
    
    # Choose the method: 'clahe' or 'hist_eq'
    method = 'clahe'  # or 'hist_eq'
    
    process_yolo_split(base_dir, output_base_dir, method=method)
