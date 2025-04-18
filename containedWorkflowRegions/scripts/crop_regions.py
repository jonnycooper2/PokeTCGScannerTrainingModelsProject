import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO
from pathlib import Path

class RegionCropper:
    def __init__(self, model_path=None):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Define class names based on the training configuration
        self.class_names = {0: "name", 1: "attack", 2: "number"}
        
        # Set model path (using a different default name)
        if model_path is None:
            model_path = os.path.join(self.base_dir, "models", "yolov8n_region_detector.pt")
        
        # Check if model exists
        if not os.path.exists(model_path):
            # Try finding the best.pt from a potential training run
            potential_model_path = os.path.join(self.base_dir, "models", "runs", "detect", "train", "weights", "best.pt")
            if os.path.exists(potential_model_path):
                model_path = potential_model_path
                print(f"Using model from latest training run: {model_path}")
            else:
                raise FileNotFoundError(f"Model not found at {model_path} or in default training output. Please train the model first!")
        
        # Load model
        self.model = YOLO(model_path)
        print(f"Loaded model from {model_path}")
        
        # Create output directory if it doesn't exist
        self.crops_dir = os.path.join(self.base_dir, "crops")
        os.makedirs(self.crops_dir, exist_ok=True)
    
    def detect_and_crop(self, image_path, conf_threshold=0.25, save_visualization=False):
        """Detect regions in image and crop them"""
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return []
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image: {image_path}")
            return []
        
        # Run inference
        results = self.model(image, conf=conf_threshold, verbose=False)
        
        # Process detections
        crops_info = [] # Store tuples of (crop_path, class_name, confidence)
        img_name = Path(image_path).stem
        vis_image = image.copy() if save_visualization else None
        
        # Check if there are any detections
        if not results or len(results[0].boxes) == 0:
            print(f"No regions detected in {image_path}")
            return []
        
        # Process each detection
        detection_counts = {name: 0 for name in self.class_names.values()} # Keep track of counts per class
        for box in results[0].boxes:
            # Get coordinates (convert to int for cropping)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.class_names.get(class_id, f"unknown_id_{class_id}")
            
            # Optional padding (can be adjusted or removed)
            padding = 2
            h, w = image.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Crop image
            crop = image[y1:y2, x1:x2]
            
            # Generate crop filename (include class name and detection index)
            detection_counts[class_name] += 1
            crop_filename = f"{img_name}_{class_name}_{detection_counts[class_name]}.jpg"
            crop_path = os.path.join(self.crops_dir, crop_filename)
            
            # Save crop
            if crop.size > 0: # Ensure crop is not empty
                cv2.imwrite(crop_path, crop)
                crops_info.append((crop_path, class_name, conf))
            else:
                print(f"Warning: Empty crop generated for {class_name} in {image_path}")
                detection_counts[class_name] -= 1 # Decrement count if crop failed
            
            # Draw on visualization image
            if vis_image is not None:
                # Draw rectangle (use different colors per class if desired)
                color = (0, 255, 0) # Default green
                # Example colors: name=blue, attack=green, number=red
                if class_id == 0: color = (255, 0, 0)
                elif class_id == 1: color = (0, 255, 0)
                elif class_id == 2: color = (0, 0, 255)
                    
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label and confidence
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(vis_image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save visualization if needed
        if save_visualization and vis_image is not None:
            vis_path = os.path.join(self.crops_dir, f"{img_name}_detections.jpg")
            cv2.imwrite(vis_path, vis_image)
            print(f"Saved detection visualization to {vis_path}")
        
        print(f"Detected {len(crops_info)} regions in {image_path}")
        return crops_info
    
    def process_file_or_directory(self, path, conf_threshold=0.25, save_visualization=False):
        """Process a single file or directory of images"""
        if os.path.isfile(path):
            # Process single file
            return self.detect_and_crop(path, conf_threshold, save_visualization)
        
        elif os.path.isdir(path):
            # Process directory
            all_crops_info = []
            for ext in ['.jpg', '.jpeg', '.png']:
                for img_path in Path(path).glob(f"*{ext}"):
                    crops_info = self.detect_and_crop(str(img_path), conf_threshold, save_visualization)
                    all_crops_info.extend(crops_info)
            return all_crops_info
        
        else:
            print(f"Path not found: {path}")
            return []

def main():
    parser = argparse.ArgumentParser(description="Detect and crop specific regions from images")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("--model", help="Path to YOLOv8 region detection model (default: models/yolov8n_region_detector.pt or latest training run)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--vis", action="store_true", help="Save visualization images with bounding boxes")
    args = parser.parse_args()
    
    try:
        # Initialize cropper
        cropper = RegionCropper(args.model)
        
        # Process images
        all_crops_info = cropper.process_file_or_directory(args.input, args.conf, args.vis)
        
        # Print results
        if all_crops_info:
            print(f"\nSuccessfully cropped {len(all_crops_info)} regions.")
            # Summarize counts per class
            counts = {}
            for _, class_name, _ in all_crops_info:
                counts[class_name] = counts.get(class_name, 0) + 1
            print("Crop counts per class:")
            for name, count in counts.items():
                print(f"  - {name}: {count}")
            print(f"Crops saved to {cropper.crops_dir}/")
        else:
            print("\nNo regions were detected or there was an error.")
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 