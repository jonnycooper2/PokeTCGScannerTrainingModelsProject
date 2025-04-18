import os
import cv2
import numpy as np
import random
import glob
import shutil
import tkinter as tk # Added for screen size detection

class RegionAnnotator:
    FIXED_WINDOW_NAME = "Region Annotator" # Define a constant window name

    def __init__(self):
        # Get screen dimensions using tkinter
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.withdraw() # Hide the tk root window

        # Calculate max window dimensions (e.g., 80% of screen)
        self.max_win_width = int(screen_width * 0.8)
        self.max_win_height = int(screen_height * 0.8)
        print(f"Screen Resolution: {screen_width}x{screen_height} -> Max Window: {self.max_win_width}x{self.max_win_height}")

        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.scripts_dir = os.path.join(self.base_dir, "scripts")
        self.jpgs_dir = os.path.join(self.base_dir, "jpgs")
        self.training_dir = os.path.join(self.base_dir, "training")
        self.images_train_dir = os.path.join(self.training_dir, "images", "train")
        self.labels_train_dir = os.path.join(self.training_dir, "labels", "train")
        
        # Ensure directories exist
        os.makedirs(self.images_train_dir, exist_ok=True)
        os.makedirs(self.labels_train_dir, exist_ok=True)
        os.makedirs(self.jpgs_dir, exist_ok=True) # Ensure jpgs dir exists

        # Annotation state
        self.all_image_paths = sorted(glob.glob(os.path.join(self.jpgs_dir, "*.jpg"))) # Load sorted list
        self.current_image_index = -1 # Start before the first image
        self.image_path = None # Store path of the current image
        self.image = None # This will hold the full image for annotation
        self.clone = None
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.current_box = None 
        self.annotations = [] 
        self.img_width = 0 # Store ORIGINAL dimensions of the image
        self.img_height = 0
        self.display_image = None # Image potentially resized for display
        self.display_width = 0 # Display dimensions
        self.display_height = 0
        self.scale_factor = 1.0 # Scale factor from original to display
        
        # Class definitions for regions
        self.classes = {"name": 0, "attack": 1, "number": 2}
        self.class_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] # BGR
        self.current_class_id = 0 
        
        # Window setup - Use WINDOW_NORMAL for resizability
        cv2.namedWindow(self.FIXED_WINDOW_NAME, cv2.WINDOW_NORMAL) 
        cv2.setMouseCallback(self.FIXED_WINDOW_NAME, self.mouse_callback)
        
        # Create or update data.yaml
        self.update_data_yaml()
        
        # Initial load
        self.load_next_image()
    
    def get_class_name(self, class_id):
        for name, id_ in self.classes.items():
            if id_ == class_id:
                return name
        return "Unknown"

    def update_class_status(self):
        """Prints status and triggers redraw when class changes"""
        class_name = self.get_class_name(self.current_class_id)
        print(f"Switched to class: {class_name} (ID: {self.current_class_id})")
        self.redraw_annotations()

    def update_data_yaml(self):
        yaml_path = os.path.join(self.training_dir, "data.yaml")
        abs_training_path = self.training_dir.replace('\\', '/')
        class_names_list = [name for name, id_ in sorted(self.classes.items(), key=lambda item: item[1])]
        yaml_content = f"""path: {abs_training_path}
train: images/train
val: images/train
names: {class_names_list}
"""
        with open(yaml_path, 'w') as yaml_file:
            yaml_file.write(yaml_content)
        print(f"Updated {yaml_path} with classes: {class_names_list}")
    
    def get_next_training_id(self):
        existing_images = glob.glob(os.path.join(self.images_train_dir, "training_*.jpg"))
        if not existing_images:
            return 1
        ids = [int(os.path.basename(img).split('_')[1].split('.')[0]) for img in existing_images]
        return max(ids) + 1 if ids else 1

    def load_next_image(self):
        """Loads the next image from the sorted list and prepares for annotation."""
        if not self.all_image_paths:
             print("No images found in the jpgs directory!")
             self.image = None # Ensure no image is displayed
             self.redraw_annotations()
             return False
             
        self.current_image_index += 1
        if self.current_image_index >= len(self.all_image_paths):
            print("Reached end of image list.")
            self.current_image_index = 0 # Wrap around to the beginning
            # Or optionally: 
            # print("Reached end of image list. No more images.")
            # return False 
            
        self.image_path = self.all_image_paths[self.current_image_index]
        
        # Load the full image directly
        try:
            self.image = cv2.imread(self.image_path)
            if self.image is None or self.image.size == 0:
                print(f"Failed to load image: {self.image_path}. Skipping.")
                self.image = None
                self.redraw_annotations()
                return self.load_next_image() # Try the next one
                
            # Store the original image dimensions
            self.img_height, self.img_width = self.image.shape[:2]
            
            # Calculate scale factor to fit within max window size
            scale_w = self.max_win_width / self.img_width
            scale_h = self.max_win_height / self.img_height
            self.scale_factor = min(scale_w, scale_h, 1.0) # Don't scale up beyond 1.0

            # Resize image for display if needed
            if self.scale_factor < 1.0:
                self.display_width = int(self.img_width * self.scale_factor)
                self.display_height = int(self.img_height * self.scale_factor)
                self.display_image = cv2.resize(self.image, (self.display_width, self.display_height), interpolation=cv2.INTER_AREA)
                print(f"Resized image for display to: {self.display_width}x{self.display_height} (Scale: {self.scale_factor:.2f})")
            else:
                self.display_image = self.image.copy() # Use original if it fits
                self.display_width = self.img_width
                self.display_height = self.img_height
                self.scale_factor = 1.0 # Explicitly set scale to 1 if no resize
            
            self.clone = self.display_image.copy() # Clone the potentially resized image
            self.annotations = [] # Reset annotations
            self.current_box = None
            self.redraw_annotations() # Display the image
            
            print(f"Loaded image {self.current_image_index + 1}/{len(self.all_image_paths)}: {os.path.basename(self.image_path)} (Size: {self.img_width}x{self.img_height})")
            # Resize window after loading image and determining display size
            if self.display_image is not None:
                 cv2.resizeWindow(self.FIXED_WINDOW_NAME, self.display_width, self.display_height)
            return True
        except Exception as e:
            print(f"Error loading image {self.image_path}: {e}")
            self.image = None
            self.redraw_annotations()
            return self.load_next_image() # Try the next one

    def redraw_annotations(self):
        """Redraw the image with current annotations and status text"""
        if self.display_image is None: return # Don't draw if no image loaded

        self.clone = self.display_image.copy() # Use the potentially resized display image
        
        # Draw existing boxes (Scale coordinates to display size)
        for class_id, box in self.annotations:
            x1, y1, x2, y2 = box # These are original image coordinates
            disp_x1 = int(x1 * self.scale_factor)
            disp_y1 = int(y1 * self.scale_factor)
            disp_x2 = int(x2 * self.scale_factor)
            disp_y2 = int(y2 * self.scale_factor)
            
            color = self.class_colors[class_id % len(self.class_colors)]
            cv2.rectangle(self.clone, (disp_x1, disp_y1), (disp_x2, disp_y2), color, 2)
            cv2.putText(self.clone, self.get_class_name(class_id), (disp_x1, disp_y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Draw current box being drawn (Scale coordinates to display size)
        if self.drawing and self.current_box:
             # current_box stores display coordinates directly from mouse events
             x1_disp, y1_disp, x2_disp, y2_disp = self.current_box 
             color = self.class_colors[self.current_class_id % len(self.class_colors)]
             cv2.rectangle(self.clone, (x1_disp, y1_disp), (x2_disp, y2_disp), color, 2)

        # Add status text
        status_text = f"Class: {self.get_class_name(self.current_class_id)} ({len(self.annotations)} boxes) | Img: {self.current_image_index + 1}/{len(self.all_image_paths)}"
        cv2.putText(self.clone, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow(self.FIXED_WINDOW_NAME, self.clone)


    def mouse_callback(self, event, x, y, flags, param):
        if self.display_image is None: return # Don't draw if no image loaded

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.current_box = [self.ix, self.iy, self.ix, self.iy] 
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_box[2], self.current_box[3] = x, y
                self.redraw_annotations() 
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                # Get final box corners in display coordinates
                disp_x1, disp_y1 = min(self.ix, x), min(self.iy, y)
                disp_x2, disp_y2 = max(self.ix, x), max(self.iy, y)

                # Scale back to ORIGINAL image coordinates
                # Add a small epsilon to prevent division by zero if scale_factor is somehow zero
                orig_x1 = int(disp_x1 / (self.scale_factor + 1e-6))
                orig_y1 = int(disp_y1 / (self.scale_factor + 1e-6))
                orig_x2 = int(disp_x2 / (self.scale_factor + 1e-6))
                orig_y2 = int(disp_y2 / (self.scale_factor + 1e-6))
                
                # Clamp coordinates to the ORIGINAL image boundaries
                orig_x1 = max(0, orig_x1)
                orig_y1 = max(0, orig_y1)
                orig_x2 = min(self.img_width, orig_x2) # Use original width
                orig_y2 = min(self.img_height, orig_y2) # Use original height

                # Add box using original coordinates only if it has non-zero area
                if orig_x1 < orig_x2 and orig_y1 < orig_y2:
                    self.annotations.append((self.current_class_id, [orig_x1, orig_y1, orig_x2, orig_y2]))
                    print(f"Added box for class '{self.get_class_name(self.current_class_id)}' (Original Coords): {[orig_x1, orig_y1, orig_x2, orig_y2]}")
                else:
                    print("Box has zero area in original coordinates, ignoring.")
                
                self.current_box = None 
                self.redraw_annotations() 
    
    def undo_last_box(self):
        """Remove the last added annotation"""
        if self.annotations:
            removed_annotation = self.annotations.pop()
            class_id, box = removed_annotation
            print(f"Removed last box (Class: {self.get_class_name(class_id)}, Box: {box})")
            self.redraw_annotations()
            return True
        else:
            print("No boxes to undo")
            return False
    
    def save_annotations(self):
        if not self.annotations:
            print("No annotations drawn! Please draw at least one box.")
            return False
        if self.image is None:
             print("Cannot save, no image loaded.")
             return False
        
        training_id = self.get_next_training_id()
        training_image_name = f"training_{training_id}.jpg"
        training_label_name = f"training_{training_id}.txt"
        
        # Save the ORIGINAL full image
        training_image_path = os.path.join(self.images_train_dir, training_image_name)
        cv2.imwrite(training_image_path, self.image) 
        
        # Prepare label content using ORIGINAL image dimensions
        label_lines = []
        # Use original dimensions stored in self.img_height, self.img_width
        img_height, img_width = self.img_height, self.img_width

        for class_id, box in self.annotations: # Annotations store original coords
            x1, y1, x2, y2 = box
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = abs(x2 - x1) / img_width
            height = abs(y2 - y1) / img_height
            
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        training_label_path = os.path.join(self.labels_train_dir, training_label_name)
        with open(training_label_path, 'w') as label_file:
            label_file.write("\n".join(label_lines))
        
        print(f"Saved image as {training_image_name} and annotations ({len(label_lines)} boxes)")
        return True
    
    def run(self):
        print("\nInstructions:")
        print(" - Draw boxes for the current class on the full card image.")
        print(" - Keys 1, 2, 3: Switch to class Name(1), Attack(2), Number(3)")
        print(" - 's': Save annotations for this image and load next")
        print(" - 'n': Skip to next image without saving")
        print(" - 'z': Undo the last drawn box")
        print(" - 'q': Quit")
            
        while True: 
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'): 
                if self.save_annotations():
                    self.load_next_image() # Load next after saving
            
            elif key == ord('n'): 
                print("Skipping image...")
                self.load_next_image() # Load next
            
            elif key == ord('z'): 
                self.undo_last_box()
            
            elif key >= ord('1') and key <= ord('3'): 
                new_class_id = key - ord('1') 
                if new_class_id != self.current_class_id:
                    self.current_class_id = new_class_id
                    self.update_class_status() 
            
            elif key == ord('q'): 
                print("Quitting...")
                cv2.destroyAllWindows()
                break # Exit main loop

if __name__ == "__main__":
    try:
        annotator = RegionAnnotator()
        # The run loop now uses waitKey, initial load happens in init
        annotator.run() 
    except FileNotFoundError as e:
         print(f"Error: {e}. Please ensure the required directories and model exist.")
    except Exception as e:
         print(f"An unexpected error occurred: {e}")
         import traceback
         traceback.print_exc() 