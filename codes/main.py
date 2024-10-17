import os
import json
import shutil
from ultralytics import YOLO
from inferencing import process_ortho  # Import your inference function
from Defect_detection import clip_and_classify_panels  # Import your defect detection function

# Define paths
YOLO_MODEL_PATH = r"D:\Rishabh Uikey\onward solar\onward solar\code\Yolo_v8_training\runs\obb\train13_imp\weights\best.pt"
ORTHOMOSAIC_PATH = r"D:\Rishabh Uikey\onward solar\onward 40mw\ortho\KML6_THERMAL1_32643.tif"
OUTPUT_PATH = r"D:\Rishabh Uikey\onward solar\onward 40mw\new_model-2"

# Define paths for intermediate and final outputs
annotations_txt_path = os.path.join(OUTPUT_PATH, "annotations.txt")
geojson_path = os.path.join(OUTPUT_PATH, "annotations.geojson")
final_geojson_path = os.path.join(OUTPUT_PATH, "final_defect_detection.geojson")

def run_pipeline():
    # Step 1: Run inference to generate annotations
    print("Starting inference...")
    model = YOLO(YOLO_MODEL_PATH)
    process_ortho(model, ORTHOMOSAIC_PATH)
    print("Inference completed.")
    
    # Step 2: Run defect detection using the generated annotations
    print("Starting defect detection...")
    clip_and_classify_panels(geojson_path, final_geojson_path)
    print("Defect detection completed.")

    # Optional: Move or handle final output as needed
    # shutil.move(final_geojson_path, desired_location)

if __name__ == "__main__":
    run_pipeline()
