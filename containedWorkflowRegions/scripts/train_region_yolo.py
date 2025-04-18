import os
import torch
from ultralytics import YOLO
import yaml # Needed to read class names from data.yaml
import shutil # Added for copying the best model

class RegionTrainer:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.training_dir = os.path.join(self.base_dir, "training")
        self.models_dir = os.path.join(self.base_dir, "models")
        self.data_yaml = os.path.join(self.training_dir, "data.yaml")
        # Default name for the final trained model
        self.trained_model_path = os.path.join(self.models_dir, "yolov8n_region_detector.pt") 
        self.num_classes = 0 # Will be read from data.yaml
        
        # Ensure directories exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Check if CUDA is available
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        
        if self.device == 'cpu':
            print("WARNING: CUDA not available, training will be slow on CPU!")
        else:
            print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
    
    def read_num_classes(self):
        """Reads the number of classes from the data.yaml file."""
        try:
            with open(self.data_yaml, 'r') as f:
                data = yaml.safe_load(f)
            if 'names' in data and isinstance(data['names'], list):
                self.num_classes = len(data['names'])
                print(f"Read {self.num_classes} classes from {self.data_yaml}: {data['names']}")
            else:
                raise ValueError("'names' key not found or is not a list in data.yaml")
        except FileNotFoundError:
             raise FileNotFoundError(f"data.yaml not found at {self.data_yaml}. Cannot determine number of classes.")
        except Exception as e:
             raise ValueError(f"Error reading or parsing {self.data_yaml}: {e}")

    def check_prerequisites(self):
        """Checks if necessary files and directories exist before training."""
        # Read number of classes first, as it's a critical prerequisite
        self.read_num_classes()

        # Check if data.yaml exists (redundant due to read_num_classes, but good practice)
        if not os.path.exists(self.data_yaml):
            raise ValueError(f"data.yaml not found at {self.data_yaml}. Please run the annotation tool first!")
        
        # Check if there are training images and labels
        images_train_dir = os.path.join(self.training_dir, "images", "train")
        labels_train_dir = os.path.join(self.training_dir, "labels", "train")
        
        if not os.path.exists(images_train_dir) or not os.path.exists(labels_train_dir):
            raise ValueError("Training image/label directories do not exist. Please run the annotation tool first!")
        
        image_count = len([f for f in os.listdir(images_train_dir) if f.endswith('.jpg')])
        label_count = len([f for f in os.listdir(labels_train_dir) if f.endswith('.txt')])
        
        if image_count == 0 or label_count == 0:
            raise ValueError("No training data (images/labels) found. Please create annotations first!")
        
        if image_count != label_count:
            print(f"Warning: Found {image_count} images and {label_count} labels. Counts should match.")
        else:
             print(f"Found {image_count} images and labels for training.")
        return True
    
    def train(self, epochs=50, image_size=640, batch_size=16, base_model='yolov8n.pt'):
        """Train YOLOv8 model on the region dataset"""
        try:
            # Check prerequisites (this also reads self.num_classes)
            self.check_prerequisites()
            
            # Load base YOLOv8 nano model
            print(f"Loading base model: {base_model}")
            model = YOLO(base_model)
            
            # Modify the model head for the correct number of classes if necessary
            # (YOLOv8 usually handles this automatically based on data.yaml, but explicit check can be added)
            # Example (Conceptual - YOLO might do this internally):
            # if model.model.yaml['nc'] != self.num_classes:
            #    print(f"Adjusting model head from {model.model.yaml['nc']} to {self.num_classes} classes.")
            #    # model.model.yaml['nc'] = self.num_classes # This might not be the correct way for YOLOv8
            #    # Re-initialization might be needed, consult YOLOv8 docs if automatic adjustment fails.

            print(f"Starting training for {self.num_classes} classes...")
            # Set training parameters
            results = model.train(
                data=self.data_yaml,
                epochs=epochs,
                imgsz=image_size,
                batch=batch_size,
                device=self.device,
                project=self.models_dir, # Save runs within the models directory
                name='region_training_run', # Specific name for this run
                exist_ok=True, # Allow overwriting previous runs with the same name
                verbose=True
            )
            
            # The best model is usually saved automatically as best.pt in the run directory
            # Let's find the path to the best model from the results
            best_model_path = results.save_dir / 'weights' / 'best.pt'
            print(f"Best model saved during training: {best_model_path}")

            # Optionally, copy the best model to the predefined path
            if os.path.exists(best_model_path):
                os.makedirs(os.path.dirname(self.trained_model_path), exist_ok=True)
                shutil.copy2(best_model_path, self.trained_model_path)
                print(f"Copied best model to: {self.trained_model_path}")
                # Optionally export to ONNX format
                try:
                    print("Exporting model to ONNX...")
                    onnx_model = YOLO(self.trained_model_path)
                    onnx_model.export(format='onnx') 
                    print(f"ONNX model saved in {os.path.dirname(self.trained_model_path)}")
                except Exception as export_e:
                    print(f"Warning: Failed to export model to ONNX: {export_e}")
            else:
                print(f"Warning: Could not find best.pt at {best_model_path}. Final model might not be saved to {self.trained_model_path}.")

            print(f"\nTraining completed successfully!")
            print(f"Training logs and results saved in: {results.save_dir}")
            
            return True
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self):
        print("=== YOLOv8 Region Detector Training ===")
        print("This script will train a YOLOv8 model to detect specific regions (name, attack, number).")
        print("Make sure you have created annotation data using the annotate_region_gui.py tool.")
        
        try:
            # Get training parameters
            print("\nTraining Parameters (press Enter for defaults):")
            
            epochs_input = input("Number of epochs [e.g., 50-100]: ").strip()
            epochs = int(epochs_input) if epochs_input else 50
            
            image_size_input = input("Image size (e.g., 640): ").strip()
            image_size = int(image_size_input) if image_size_input else 640
            
            batch_size_input = input(f"Batch size (adjust based on GPU memory, e.g., 8, 16) [{16 if self.device == 0 else 8}]: ").strip()
            default_batch = 16 if self.device == 0 else 8 # Smaller default for CPU
            batch_size = int(batch_size_input) if batch_size_input else default_batch
            
            # --- Define the path to your existing model --- 
            default_model_path = r"C:\Users\jonat\PokeTCGScannerTrainingModelsProject\approvedModels\Regionsbest.pt"
            # Check if the default model exists
            if not os.path.exists(default_model_path):
                 print(f"Warning: Default model not found at {default_model_path}. Falling back to yolov8n.pt.")
                 default_model_path = 'yolov8n.pt' # Fallback if specific model not found
                 
            base_model_input = input(f"Model to fine-tune (e.g., yolov8n.pt): [{default_model_path}] ").strip()
            base_model = base_model_input if base_model_input else default_model_path

            print(f"\nStarting training with:")
            print(f"- Starting Model: {base_model}")
            print(f"- Epochs: {epochs}")
            print(f"- Image size: {image_size}")
            print(f"- Batch size: {batch_size}")
            print(f"- Device: {'GPU' if self.device == 0 else 'CPU'}")
            
            # Confirm and start training
            confirm = input("\nStart training? [y/N]: ").strip().lower()
            if confirm == 'y':
                self.train(epochs, image_size, batch_size, base_model)
            else:
                print("Training cancelled.")
                
        except KeyboardInterrupt:
            print("\nTraining interrupted.")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    trainer = RegionTrainer()
    trainer.run() 