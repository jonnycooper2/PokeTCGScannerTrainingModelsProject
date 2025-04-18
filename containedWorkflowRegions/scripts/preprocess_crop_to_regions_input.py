import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import glob
import torch
from ultralytics.nn.tasks import DetectionModel

class CardPreprocessor:
    def __init__(self, model_path, card_class_id=0, conf_threshold=0.3, padding=5):
        # Load card detector model
        if not os.path.exists(model_path):
            # Try resolving relative to script dir if not absolute
            script_dir = os.path.dirname(os.path.abspath(__file__))
            potential_path = os.path.abspath(os.path.join(script_dir, model_path))
            if os.path.exists(potential_path):
                 model_path = potential_path
            else:
                 raise FileNotFoundError(f"Card detector model not found at {model_path} or {potential_path}")
        
        # --- PyTorch 2.6+ Safe Loading Fix --- 
        try:
            with torch.serialization.safe_globals([DetectionModel]): # Allow DetectionModel temporarily
                self.model = YOLO(model_path)
        except ImportError:
             print("Warning: Could not import DetectionModel. Attempting standard YOLO load.")
             self.model = YOLO(model_path)
        except Exception as e:
             print(f"Error loading YOLO model: {e}")
             raise e # Re-raise the error if loading failed
        # -------------------------------------

        self.card_class_id = card_class_id
        self.conf_threshold = conf_threshold
        self.padding = padding
        print(f"Loaded card detector model from: {model_path}")
        print(f"Target Card Class ID: {self.card_class_id}")
        print(f"Confidence Threshold: {self.conf_threshold}")
        print(f"Padding: {self.padding} pixels")

    def find_best_card_box(self, image_path):
        """Finds the bounding box of the highest confidence card detection."""
        try:
            results = self.model(image_path, verbose=False) # Run detector
            
            best_box = None
            max_conf = 0.0

            if results and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    # Check if it's the target card class and has higher confidence
                    if cls_id == self.card_class_id and conf > max_conf:
                        max_conf = conf
                        best_box = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            
            if best_box and max_conf >= self.conf_threshold:
                print(f"  -> Detected card (conf: {max_conf:.2f}) in {os.path.basename(image_path)}")
                return best_box
            else:
                print(f"  -> No card detected above threshold {self.conf_threshold} (max_conf: {max_conf:.2f}) in {os.path.basename(image_path)}")
                return None

        except Exception as e:
            print(f"Error detecting card in {os.path.basename(image_path)}: {e}")
            return None

    def crop_and_save(self, input_image_path, output_dir):
        """Detects the best card, crops it, and saves to the output directory."""
        # Find the best bounding box for the card
        best_box = self.find_best_card_box(input_image_path)
        
        if not best_box:
            return False # Skip if no card detected
        
        try:
            # Read the full image
            full_image = cv2.imread(input_image_path)
            if full_image is None:
                 print(f"Error reading image {input_image_path}")
                 return False
                 
            h_full, w_full = full_image.shape[:2]
            x1, y1, x2, y2 = map(int, best_box)
            
            # Apply padding
            x1 = max(0, x1 - self.padding)
            y1 = max(0, y1 - self.padding)
            x2 = min(w_full, x2 + self.padding)
            y2 = min(h_full, y2 + self.padding)
            
            # Crop the image
            cropped_image = full_image[y1:y2, x1:x2]
            
            if cropped_image.size == 0:
                print(f"Warning: Generated empty crop for {os.path.basename(input_image_path)}. Skipping.")
                return False
                
            # Create output filename (same as input)
            output_filename = os.path.basename(input_image_path)
            output_path = os.path.join(output_dir, output_filename)
            
            # Save the cropped image
            cv2.imwrite(output_path, cropped_image)
            # print(f"   Saved cropped image to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error cropping/saving {os.path.basename(input_image_path)}: {e}")
            return False

    def process_directory(self, input_dir, output_dir):
        """Processes all images in the input directory and saves crops to output dir."""
        print(f"\nProcessing images from: {input_dir}")
        print(f"Saving cropped cards to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        processed_count = 0
        skipped_count = 0
        image_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg"))) # Process in order
        
        if not image_paths:
             print(f"No .jpg images found in {input_dir}")
             return

        for img_path in image_paths:
            if self.crop_and_save(img_path, output_dir):
                processed_count += 1
            else:
                skipped_count += 1
        
        print(f"\nProcessing complete.")
        print(f"Successfully processed and cropped: {processed_count} images.")
        print(f"Skipped (no confident card detected): {skipped_count} images.")

def main():
    # --- Argument Parsing --- 
    parser = argparse.ArgumentParser(description="Preprocess images by detecting and cropping the main trading card.")
    
    # Default paths relative to this script's parent directory
    script_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_input = os.path.join(script_parent_dir, "..", "containedWorkflow", "jpgs")
    default_output = os.path.join(script_parent_dir, "jpgs")
    default_model = os.path.join(script_parent_dir, "..", "containedWorkflow", "models", "training_run", "weights", "best.pt")

    parser.add_argument("--input-dir", default=default_input, 
                        help=f"Directory containing original full card images (default: {default_input})")
    parser.add_argument("--output-dir", default=default_output,
                         help=f"Directory to save the cropped card images (default: {default_output})")
    parser.add_argument("--model", default=default_model,
                        help=f"Path to the trained card detector YOLO model (default: {default_model})")
    parser.add_argument("--card-id", type=int, default=0,
                         help="Class ID of the 'trading_card' in the detector model (default: 0)")
    parser.add_argument("--conf", type=float, default=0.3, 
                        help="Confidence threshold for card detection (default: 0.3)")
    parser.add_argument("--padding", type=int, default=5,
                         help="Pixel padding to add around the detected card box (default: 5)")
    
    args = parser.parse_args()
    # --- End Argument Parsing --- 

    try:
        # Ensure input directory exists
        if not os.path.isdir(args.input_dir):
             raise NotADirectoryError(f"Input directory not found: {args.input_dir}")
             
        # Initialize preprocessor
        preprocessor = CardPreprocessor(model_path=args.model, 
                                      card_class_id=args.card_id, 
                                      conf_threshold=args.conf, 
                                      padding=args.padding)
        
        # Process the directory
        preprocessor.process_directory(args.input_dir, args.output_dir)
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 