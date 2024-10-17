# Use official Python image as the base image
FROM python:3.8.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container from the requirements folder
COPY requirements/requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container from the codes folder
COPY codes/ ./codes/

# Copy the model files into the container from the models folder
COPY models/ ./models/

# Set environment variables for paths in the script (these can be overridden when running the container)
ENV OUTPUT_PATH=/app/output
ENV YOLO_MODEL_PATH=/app/models/best_v4.pt
ENV ORTHOMOSAIC_PATH=/app/data/ortho/ortho.tif
ENV TEMP_DIR=/app/temp_clipped_panels
ENV MODEL_PATH=/app/models/best_solar_panel_defect_classifier_RESNET.pth
ENV DEFECT_OUTPUT_GEOJSON=/app/output/defect_predictions.geojson
ENV NON_DEFECT_OUTPUT_GEOJSON=/app/output/non_defect_predictions.geojson
ENV MERGED_OUTPUT_GEOJSON=/app/output/merged_predictions.geojson

# Create directories that may be used by the container
RUN mkdir -p /app/output /app/temp_clipped_panels /app/data/ortho

# Ensure that Rasterio has access to GDAL dependencies (for handling GeoTIFF files)
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the environment variable for GDAL
ENV GDAL_DATA=/usr/share/gdal

# Install other system dependencies if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Define the default command to run your final inference and detection script
CMD ["python", "./codes/optimzed_final_inference_and_detection.py"]

# To run another script, you can override the command while running the Docker container:
# docker run <image_name> python ./codes/other_script.py
