import os
import cv2
import numpy as np

# Define paths to image and label folders
image_folder = "D:\\preprocessing\\train\\images"
label_folder = "D:\\preprocessing\\train\\labels"

# Define output folders for preprocessed images and labels
output_image_folder = "D:\\preprocessing\\preprocessed\\images"
output_label_folder = "D:\\preprocessing\\preprocessed\\labels"

# Create output folders if they don't exist
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

# Define target image size for resizing
target_size = (800, 800)  # YOLO input size

# Define normalization parameters (optional)
# You may need to adjust these based on your dataset
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Loop through each image file in the image folder
# Loop through each image file in the image folder
# Loop through each image file in the image folder
for image_file in os.listdir(image_folder):
    if image_file.endswith(".jpg") or image_file.endswith(".png"):
        # Read image
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        
        # Resize image
        resized_image = cv2.resize(image, target_size)
        
        # Save preprocessed image
        output_image_path = os.path.join(output_image_folder, image_file)
        cv2.imwrite(output_image_path, resized_image)
        
        # Load corresponding label file
        label_file = image_file.replace(".jpg", ".txt").replace(".png", ".txt")
        label_path = os.path.join(label_folder, label_file)
        print("Label Path:", label_path)
        # Attempt to open the label file
        try:
            with open(label_path, "r") as f:
                lines = f.readlines()
        except OSError as e:
            print(f"Error opening label file '{label_path}': {e}")
            continue  # Skip to the next iteration of the loop if there's an error

        
        # Preprocess and save label file
        output_label_path = os.path.join(output_label_folder, label_file)
        with open(output_label_path, "w") as f:
            for line in lines:
                try:
                    # Parse line and preprocess bounding box coordinates
                    # (Assuming Darknet YOLO format with normalized coordinates)
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    x_center *= target_size[0]  # Scale to image width
                    y_center *= target_size[1]  # Scale to image height
                    width *= target_size[0]      # Scale to image width
                    height *= target_size[1]     # Scale to image height
                    f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")
                except ValueError:
                    print(f"Ignoring line in label file '{label_file}': {line.strip()} (Invalid format)")
