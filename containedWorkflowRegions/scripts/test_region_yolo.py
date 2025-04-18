import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import torch
import glob
from pathlib import Path
from ultralytics import YOLO

class RegionTester:
    def __init__(self, root):
        self.root = root
        self.root.title("Region Detector - Model Testing Tool")
        self.root.geometry("1400x800")
        
        # Setup paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.base_dir, "models")
        self.class_names = {0: "name", 1: "attack", 2: "number"} # Define expected classes
        
        # Variables
        self.model = None
        self.current_image = None
        self.current_image_path = None
        self.confidence = tk.DoubleVar(value=0.25)
        self.crop_photos = []  # Store crop references
        
        # Create the UI
        self._create_ui()
        
        # Load available models
        self._refresh_model_list()
        
        # Set up drag and drop
        try:
            import tkinterdnd2
            self.root.drop_target_register(tkinterdnd2.DND_FILES)
            self.root.dnd_bind('<<Drop>>', self._on_drop)
        except (ImportError, AttributeError):
            self.status_var.set("Drag and drop not available. Install tkinterdnd2 package.")
    
    def _create_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Model selection
        ttk.Label(control_frame, text="Select model:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_combo = ttk.Combobox(control_frame, width=40, state="readonly")
        self.model_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)
        
        refresh_btn = ttk.Button(control_frame, text="🔄 Refresh", command=self._refresh_model_list)
        refresh_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Confidence threshold
        ttk.Label(control_frame, text="Confidence:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        confidence_scale = ttk.Scale(control_frame, from_=0.01, to=1.0, variable=self.confidence, orient=tk.HORIZONTAL, length=300)
        confidence_scale.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.conf_display = tk.StringVar(value="0.25")
        confidence_scale.configure(command=self._update_confidence_display)
        ttk.Label(control_frame, textvariable=self.conf_display).grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        
        # File selection buttons
        file_frame = ttk.Frame(control_frame)
        file_frame.grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        ttk.Button(file_frame, text="Open Image", command=self._open_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Open Folder", command=self._open_folder).pack(side=tk.LEFT, padx=5)
        
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas_frame = ttk.LabelFrame(paned_window, text="Image Preview (Drag & Drop images here)", padding="10")
        paned_window.add(self.canvas_frame, weight=2)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="lightgray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Right side - Detected regions display
        self.regions_frame = ttk.LabelFrame(paned_window, text="Detected Regions", padding="10")
        paned_window.add(self.regions_frame, weight=1)
        
        self.regions_scrollframe = ttk.Frame(self.regions_frame)
        self.regions_scrollframe.pack(fill=tk.BOTH, expand=True)
        
        regions_scrollbar = ttk.Scrollbar(self.regions_scrollframe, orient=tk.VERTICAL)
        regions_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.regions_canvas = tk.Canvas(self.regions_scrollframe, bg="white", yscrollcommand=regions_scrollbar.set)
        self.regions_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        regions_scrollbar.config(command=self.regions_canvas.yview)
        
        self.regions_container = ttk.Frame(self.regions_canvas)
        self.regions_canvas_window = self.regions_canvas.create_window((0, 0), window=self.regions_container, anchor=tk.NW)
        
        self.regions_container.bind("<Configure>", self._configure_regions_canvas)
        self.regions_canvas.bind("<Configure>", self._configure_regions_canvas_window)
        
        self.status_var = tk.StringVar(value="Ready. Drag and drop an image or select a model to begin.")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, padx=5, pady=5)
    
    def _configure_regions_canvas(self, event):
        self.regions_canvas.configure(scrollregion=self.regions_canvas.bbox("all"))
    
    def _configure_regions_canvas_window(self, event):
        self.regions_canvas.itemconfig(self.regions_canvas_window, width=event.width)
    
    def _update_confidence_display(self, value):
        self.conf_display.set(f"{float(value):.2f}")
        if self.current_image_path and self.model:
            self._process_image(self.current_image_path)

    def _refresh_model_list(self):
        self.status_var.set("Refreshing model list...")
        self.root.update_idletasks()
        
        model_files = []
        # Look in models dir and the default training output dir
        search_dirs = [self.models_dir, os.path.join(self.models_dir, "runs", "detect")]
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for ext in ['.pt', '.onnx']:
                    model_files.extend(glob.glob(os.path.join(search_dir, f"**/*{ext}"), recursive=True))
        
        # Deduplicate and sort
        model_files = sorted(list(set(model_files)))

        if model_files:
            model_names = [os.path.basename(f) for f in model_files]
            self.model_combo['values'] = model_names
            self.model_paths = {name: path for name, path in zip(model_names, model_files)}
            self.status_var.set(f"Found {len(model_files)} models")
            
            # Select a likely default model if available
            target_models = ["yolov8n_region_detector.pt", "best.pt"]
            selected = False
            for target_model in target_models:
                if target_model in model_names:
                    self.model_combo.set(target_model)
                    self._on_model_selected(None)
                    selected = True
                    break
            if not selected:
                self.model_combo.current(0)
                self._on_model_selected(None)
        else:
            self.status_var.set("No models found in the models directory or default training paths")
            self.model_combo['values'] = ["No models found"]
            self.model_combo.current(0)
            self.model = None
    
    def _on_model_selected(self, event):
        selected = self.model_combo.get()
        if selected and selected != "No models found":
            model_path = self.model_paths[selected]
            self.status_var.set(f"Loading model: {selected}...")
            self.root.update_idletasks()
            
            try:
                self.model = YOLO(model_path)
                self.status_var.set(f"Model loaded: {selected}")
                if self.current_image_path:
                    self._process_image(self.current_image_path)
            except Exception as e:
                self.status_var.set(f"Error loading model: {str(e)}")
                messagebox.showerror("Model Error", f"Failed to load model: {str(e)}")
                self.model = None
    
    def _open_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self._process_image(file_path)
    
    def _open_folder(self):
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        if folder_path:
            image_paths = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_paths.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
                image_paths.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
            
            if image_paths:
                self._create_image_browser(image_paths)
            else:
                messagebox.showinfo("No Images", "No image files found in the selected folder")
    
    def _create_image_browser(self, image_paths):
        browser = tk.Toplevel(self.root)
        browser.title("Image Browser")
        browser.geometry("400x500")
        
        ttk.Label(browser, text="Select an image to test:").pack(padx=10, pady=5, anchor=tk.W)
        
        listbox_frame = ttk.Frame(browser)
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = ttk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(listbox_frame)
        listbox.pack(fill=tk.BOTH, expand=True)
        
        listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=listbox.yview)
        
        for img_path in image_paths:
            listbox.insert(tk.END, os.path.basename(img_path))
        
        listbox.image_paths = image_paths
        
        def on_select(event):
            idx = listbox.curselection()
            if idx:
                selected_path = listbox.image_paths[idx[0]]
                self._process_image(selected_path)
                browser.destroy()
        
        listbox.bind("<Double-Button-1>", on_select)
        
        select_btn = ttk.Button(browser, text="Select", command=lambda: on_select(None))
        select_btn.pack(pady=10)
    
    def _on_drop(self, event):
        # Get the file path from the dropped data
        # Note: tkinterdnd2 returns a string with potential curly braces
        file_path = event.data.strip('{}')
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                self._process_image(file_path)
            elif os.path.isdir(file_path):
                messagebox.showinfo("Folder Drop", "Folder dropping not supported. Please use 'Open Folder' button.")
        else:
            messagebox.showerror("Error", f"Invalid file path dropped: {file_path}")

    def _clear_regions(self):
        """Clear the detected regions display"""
        for widget in self.regions_container.winfo_children():
            widget.destroy()
        self.crop_photos = [] # Clear references
        self.regions_canvas.yview_moveto(0) # Scroll back to top
        
    def _add_region_display(self, region_img, class_name, confidence):
        """Add a detected region to the scrollable display"""
        try:
            # Convert OpenCV image (BGR) to PIL image (RGB)
            region_img_rgb = cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(region_img_rgb)
            
            # Resize for display
            max_height = 100
            width, height = pil_img.size
            aspect_ratio = width / height
            new_height = min(height, max_height)
            new_width = int(new_height * aspect_ratio)
            pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_img)
            self.crop_photos.append(photo)  # Keep a reference
            
            # Create a frame for the region info
            region_frame = ttk.Frame(self.regions_container, borderwidth=1, relief=tk.SOLID)
            region_frame.pack(fill=tk.X, padx=5, pady=3)
            
            # Display the image
            img_label = ttk.Label(region_frame, image=photo)
            img_label.pack(side=tk.LEFT, padx=5, pady=5)
            
            # Display class and confidence
            info_label = ttk.Label(region_frame, text=f"Class: {class_name}\nConf: {confidence:.2f}")
            info_label.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
            
        except Exception as e:
            print(f"Error adding region display: {e}")
            # Add a placeholder if image processing fails
            region_frame = ttk.Frame(self.regions_container, borderwidth=1, relief=tk.SOLID)
            region_frame.pack(fill=tk.X, padx=5, pady=3)
            ttk.Label(region_frame, text=f"Error displaying region\n{class_name} ({confidence:.2f})").pack(padx=5, pady=5)
    
    def _process_image(self, image_path):
        """Load, process image, and display results"""
        if not self.model:
            self.status_var.set("No model loaded. Please select a model.")
            # messagebox.showwarning("No Model", "Please select a model first.")
            return
        
        if not os.path.exists(image_path):
            self.status_var.set(f"Image not found: {image_path}")
            messagebox.showerror("File Error", f"Image not found: {image_path}")
            return
            
        self.current_image_path = image_path
        self.status_var.set(f"Processing: {os.path.basename(image_path)}...")
        self.root.update_idletasks()
        
        try:
            # Load image with OpenCV
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                raise ValueError("Could not read image file")
            
            # Perform inference
            conf = self.confidence.get()
            results = self.model(self.current_image, conf=conf, verbose=False)
            
            # Process results
            processed_image = self.current_image.copy()
            self._clear_regions() # Clear previous results
            num_detections = 0
            
            if results and len(results[0].boxes) > 0:
                num_detections = len(results[0].boxes)
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.class_names.get(class_id, f"Unknown ID {class_id}")
                    
                    # Draw bounding box
                    color = (0, 255, 0) # Default Green
                    if class_id == 0: color = (255, 0, 0) # Name = Blue
                    elif class_id == 1: color = (0, 255, 0) # Attack = Green
                    elif class_id == 2: color = (0, 0, 255) # Number = Red
                    cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(processed_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Extract crop
                    crop_img = self.current_image[y1:y2, x1:x2]
                    if crop_img.size > 0:
                        self._add_region_display(crop_img, class_name, confidence)
            
            # Display the processed image
            self._display_image(processed_image)
            self.status_var.set(f"Processed: {os.path.basename(image_path)} | Found {num_detections} regions (Conf > {conf:.2f})")
            
        except Exception as e:
            self.status_var.set(f"Error processing image: {str(e)}")
            messagebox.showerror("Processing Error", f"Failed to process image: {str(e)}")
            # Display original image if processing failed
            if self.current_image is not None:
                self._display_image(self.current_image)
            else:
                self.canvas.delete("all") # Clear canvas if image load failed
    
    def _display_image(self, cv_image):
        """Display an OpenCV image on the Tkinter canvas"""
        try:
            # Convert image to RGB
            img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Resize image to fit canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1: # Canvas not rendered yet
                # Estimate size or wait
                self.root.after(100, lambda: self._display_image(cv_image))
                return
                
            img_width, img_height = pil_img.size
            aspect_ratio = img_width / img_height
            
            new_width = canvas_width
            new_height = int(new_width / aspect_ratio)
            if new_height > canvas_height:
                new_height = canvas_height
                new_width = int(new_height * aspect_ratio)
                
            resized_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.photo_image = ImageTk.PhotoImage(resized_img)
            
            # Display on canvas
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=self.photo_image)
            
        except Exception as e:
            print(f"Error displaying image: {e}")
            self.status_var.set(f"Error displaying image: {e}")
            self.canvas.delete("all") # Clear canvas on error

if __name__ == "__main__":
    try:
        import tkinterdnd2
        root = tkinterdnd2.Tk()
    except ImportError:
        root = tk.Tk()
        print("tkinterdnd2 not found. Drag and drop functionality will be disabled.")
        
    app = RegionTester(root)
    root.mainloop() 