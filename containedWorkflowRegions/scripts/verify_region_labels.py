import os
import cv2
import random
import glob
import numpy as np
import yaml # To read class names from data.yaml

class RegionVerifier:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.training_dir = os.path.join(self.base_dir, "training")
        self.images_train_dir = os.path.join(self.training_dir, "images", "train")
        self.labels_train_dir = os.path.join(self.training_dir, "labels", "train")
        self.data_yaml_path = os.path.join(self.training_dir, "data.yaml")
        
        # Verify training directories
        if not os.path.exists(self.images_train_dir) or not os.path.exists(self.labels_train_dir):
            raise ValueError("Training directories do not exist. Please run the annotation tool first!")

        # Load class names from data.yaml
        self.class_names = self._load_class_names()
        self.class_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] # BGR: Name(Blue), Attack(Green), Number(Red)

    def _load_class_names(self):
        """Load class names from data.yaml"""
        if not os.path.exists(self.data_yaml_path):
             print(f"Warning: data.yaml not found at {self.data_yaml_path}. Using default class IDs.")
             return {0: "name", 1: "attack", 2: "number"} # Fallback
        try:
            with open(self.data_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            if 'names' in data and isinstance(data['names'], list):
                return {i: name for i, name in enumerate(data['names'])}
            else:
                 print(f"Warning: 'names' key not found or invalid in {self.data_yaml_path}. Using default class IDs.")
                 return {0: "name", 1: "attack", 2: "number"}
        except Exception as e:
             print(f"Error reading {self.data_yaml_path}: {e}. Using default class IDs.")
             return {0: "name", 1: "attack", 2: "number"}
    
    def get_image_label_pairs(self):
        """Get list of (image_path, label_path) pairs for verification."""
        image_files = glob.glob(os.path.join(self.images_train_dir, "*.jpg"))
        
        pairs = []
        for image_path in image_files:
            base_name = os.path.basename(image_path)
            label_name = os.path.splitext(base_name)[0] + ".txt"
            label_path = os.path.join(self.labels_train_dir, label_name)
            
            if os.path.exists(label_path):
                pairs.append((image_path, label_path))
            else:
                print(f"Warning: Label file not found for {base_name}")
        
        return pairs
    
    def parse_yolo_labels(self, label_path, img_width, img_height):
        """Parse YOLO format label file which may contain multiple lines (boxes)"""
        boxes = []
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    print(f"Invalid label format in {label_path}: '{line}'")
                    continue
                
                class_id, x_center, y_center, width, height = map(float, parts)
                
                # Convert normalized coordinates to pixel coordinates
                x_center_px = x_center * img_width
                y_center_px = y_center * img_height
                width_px = width * img_width
                height_px = height * img_height
                
                # Calculate box corners
                x1 = int(x_center_px - width_px / 2)
                y1 = int(y_center_px - height_px / 2)
                x2 = int(x_center_px + width_px / 2)
                y2 = int(y_center_px + height_px / 2)
                
                boxes.append((int(class_id), x1, y1, x2, y2))
        except Exception as e:
             print(f"Error reading or parsing {label_path}: {e}")
        return boxes
    
    def display_random_annotation(self):
        """Selects a random image and displays it with its annotations."""
        pairs = self.get_image_label_pairs()
        
        if not pairs:
            print("No annotated images found!")
            return
        
        # Select a random image-label pair
        image_path, label_path = random.choice(pairs)
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            return
        
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        
        # Parse all labels for this image
        label_data_list = self.parse_yolo_labels(label_path, img_width, img_height)
        
        if label_data_list:
            # Draw each bounding box
            for label_data in label_data_list:
                class_id, x1, y1, x2, y2 = label_data
                
                # Get class name and color
                class_name = self.class_names.get(class_id, f"ID_{class_id}")
                color = self.class_colors[class_id % len(self.class_colors)]
                
                # Draw the bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Add class label text
                cv2.putText(image, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Resize if image is too large for display
            max_display_size = 1200
            height, width = image.shape[:2]
            if max(height, width) > max_display_size:
                scale = max_display_size / max(height, width)
                image = cv2.resize(image, (int(width * scale), int(height * scale)))
            
            # Display the image with bounding boxes
            window_name = "YOLO Region Annotation Verifier"
            cv2.namedWindow(window_name)
            cv2.imshow(window_name, image)
            
            print(f"Displaying annotations for: {os.path.basename(image_path)}")
            print(f"Found {len(label_data_list)} annotations.")
            print("Press any key to close the window and load another random image (or 'q' to quit).")
            
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            
            if key == ord('q'):
                 return False # Signal to quit
        else:
            print(f"No valid labels found or failed to parse labels from {label_path}")
        
        return True # Continue
    
    def run(self):
        """Main loop for the verifier tool."""
        print("=== YOLO Region Annotation Verifier ===")
        print("Press any key to view the next random annotated image.")
        print("Press 'q' in the image window to quit.")
        
        while True:
             if not self.display_random_annotation():
                  break # Exit loop if user pressed 'q'

if __name__ == "__main__":
    try:
        verifier = RegionVerifier()
        verifier.run()
    except Exception as e:
         print(f"An error occurred: {e}") 