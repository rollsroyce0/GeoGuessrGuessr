import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import sys
from pathlib import Path
import threading
import time
from typing import Tuple

# Add the parent directory to the path so we can import from the main module
sys.path.append(str(Path(__file__).resolve().parents[3]))

from Roy.V2.GUI.progress_tracker import ProgressTracker
from Roy.V2.GUI.embedded_map import EmbeddedMapDisplay
from Roy.Helper_Functions.project_utils import (
    get_s2_index_path,
    get_test_image_path,
    parse_test_image as parse_test_image_name,
)
from Roy.V2.ProtoNet_Test import SigLIPEncoder, load_index, evaluate, get_real_coordinates

class ProtoNetGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ProtoNet GeoGuessr Test - GUI Mode")
        self.root.geometry("1200x800")

        # Store results from evaluation
        self.evaluation_results = None
        self.test_types = []
        self.encoder = None
        self.index = None

        # Create main container
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create loading screen
        self.create_loading_screen()

        # Start the evaluation in a separate thread
        self.start_evaluation_thread()

    def create_loading_screen(self):
        """Create the loading screen with progress bar"""
        self.loading_frame = tk.Frame(self.main_frame)
        self.loading_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = tk.Label(
            self.loading_frame,
            text="ProtoNet GeoGuessr Test",
            font=('Arial', 18, 'bold')
        )
        title_label.pack(pady=20)

        # Description
        desc_label = tk.Label(
            self.loading_frame,
            text="Running evaluation with progress tracking...",
            font=('Arial', 12)
        )
        desc_label.pack(pady=10)

        # Progress tracker
        self.progress_tracker = ProgressTracker(self.root)

    def start_evaluation_thread(self):
        """Start the evaluation in a separate thread"""
        def evaluation_worker():
            try:
                # Initialize encoder and index
                self.progress_tracker.set_status("Loading encoder...")
                self.encoder = SigLIPEncoder()
                self.progress_tracker.update_progress(20, "Encoder loaded successfully")

                self.progress_tracker.set_status("Loading S2 index...")
                INDEX_PATH = get_s2_index_path()

                if INDEX_PATH.exists():
                    self.index = load_index(INDEX_PATH)
                    self.progress_tracker.update_progress(30, "Index loaded from cache")
                else:
                    messagebox.showerror(
                        "Error",
                        f"Index file not found at {INDEX_PATH}. Please run without GUI first to build the index."
                    )
                    self.root.after(0, self.root.destroy)
                    return

                # Run evaluation with progress callback
                self.progress_tracker.set_status("Running evaluation...")
                progress_callback = self.progress_tracker.create_progress_callback()
                self.evaluation_results = evaluate(self.encoder, self.index, progress_callback)

                # Close progress window and show results
                self.progress_tracker.complete()
                time.sleep(1)  # Let the user see the completion
                self.progress_tracker.close()

                # Switch to results screen
                self.root.after(0, self.create_results_screen)

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during evaluation: {str(e)}")
                self.progress_tracker.close()
                self.root.after(0, self.root.destroy)

        # Start the thread
        threading.Thread(target=evaluation_worker, daemon=True).start()

    def create_results_screen(self):
        """Create the results screen with dropdowns and map display"""
        # Clear loading screen
        self.loading_frame.destroy()

        # Main results frame
        self.results_frame = tk.Frame(self.main_frame)
        self.results_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = tk.Label(
            self.results_frame,
            text="ProtoNet GeoGuessr Test Results",
            font=('Arial', 16, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # Controls frame
        controls_frame = tk.Frame(self.results_frame)
        controls_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")

        # Test type dropdown
        test_type_label = tk.Label(controls_frame, text="Test Image Series:")
        test_type_label.grid(row=0, column=0, padx=5)

        self.test_type_var = tk.StringVar()
        self.test_type_dropdown = ttk.Combobox(
            controls_frame,
            textvariable=self.test_type_var,
            state="readonly",
            width=30
        )
        self.test_type_dropdown.grid(row=0, column=1, padx=5)

        # Image number dropdown
        image_num_label = tk.Label(controls_frame, text="Image Number:")
        image_num_label.grid(row=0, column=2, padx=5)

        self.image_num_var = tk.StringVar()
        self.image_num_dropdown = ttk.Combobox(
            controls_frame,
            textvariable=self.image_num_var,
            state="readonly",
            width=10
        )
        self.image_num_dropdown.grid(row=0, column=3, padx=5)

        # Confirm button
        confirm_button = tk.Button(
            controls_frame,
            text="Show Results",
            command=self.show_selected_results,
            bg="#4CAF50",
            fg="white",
            font=('Arial', 10, 'bold')
        )
        confirm_button.grid(row=0, column=4, padx=10)

        # Main display area
        display_frame = tk.Frame(self.results_frame)
        display_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=10)

        # Configure grid weights
        self.results_frame.grid_rowconfigure(2, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)
        self.results_frame.grid_columnconfigure(1, weight=1)

        # Map display (left) - Use the new EmbeddedMapDisplay component
        self.map_display = EmbeddedMapDisplay(display_frame)
        self.map_display.map_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Image display (right)
        self.image_frame = tk.Frame(display_frame, bg="#f0f0f0", bd=2, relief=tk.GROOVE)
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        image_title = tk.Label(
            self.image_frame,
            text="Test Image",
            font=('Arial', 12, 'bold')
        )
        image_title.pack(pady=5)

        self.image_label = tk.Label(self.image_frame, text="Select a test image to view")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Populate dropdowns
        self.populate_dropdowns()

    def populate_dropdowns(self):
        """Populate the dropdowns with available test types and image numbers"""
        # Get unique test types from results
        if self.evaluation_results is not None and not self.evaluation_results.empty:
            test_types = set()
            for img_name in self.evaluation_results['img']:
                test_type, _ = parse_test_image_name(img_name)
                test_types.add(test_type)
            test_types = sorted(list(test_types))

            self.test_types = sorted(list(test_types))

            # Set test type dropdown values
            self.test_type_dropdown['values'] = self.test_types
            if self.test_types:
                self.test_type_dropdown.set(self.test_types[0])

            # Set image number dropdown values
            self.image_num_dropdown['values'] = ['1', '2', '3', '4', '5']
            self.image_num_dropdown.set('1')

    def show_selected_results(self):
        """Show the selected test image results on the map and image display"""
        test_type = self.test_type_var.get()
        image_num = int(self.image_num_var.get()) - 1  # Convert to 0-based index

        if not test_type:
            messagebox.showwarning("Selection Required", "Please select a test image series")
            return

        try:
            # Find the corresponding result
            result_row = None
            for _, row in self.evaluation_results.iterrows():
                current_type, current_idx = parse_test_image_name(row['img'])
                if current_type == test_type and current_idx == image_num:
                    result_row = row
                    break

            if result_row is None:
                messagebox.showwarning("Not Found", f"No results found for {test_type} image {image_num + 1}")
                return

            # Get coordinates
            test_type, idx = parse_test_image_name(result_row['img'])
            real_coords = get_real_coordinates(test_type)[idx]
            pred_lat, pred_lon = result_row['pred_lat'], result_row['pred_lon']

            # Show image
            self.show_test_image(test_type, idx)

            # Create and show map
            self.display_map_in_gui(real_coords, (pred_lat, pred_lon), test_type, idx)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to show results: {str(e)}")

    def show_test_image(self, test_type: str, image_idx: int):
        """Display the selected test image"""
        try:
            image_path = get_test_image_path(test_type, image_idx)
            if image_path is None:
                self.image_label.config(text=f"No image found: {test_type}_Test{image_idx + 1}")
                return

            img = Image.open(image_path)

            # Resize image to fit the frame
            max_size = (400, 400)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)

            # Update the label
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep reference

            # Add image info
            info_text = f"{test_type} - Image {image_idx + 1}\nSize: {img.size[0]}x{img.size[1]}"
            self.image_label.config(text=info_text, compound=tk.TOP)

        except Exception as e:
            self.image_label.config(text=f"Error loading image: {str(e)}")

    def display_map_in_gui(self, real_coords: Tuple[float, float],
                          pred_coords: Tuple[float, float],
                          test_type: str, image_idx: int):
        """Display the map in the GUI using the EmbeddedMapDisplay component"""
        try:
            # Display the map using the EmbeddedMapDisplay component
            self.map_display.display_map(real_coords, pred_coords, test_type, image_idx)

        except Exception as e:
            self.map_display._show_error(f"Error displaying map: {str(e)}")

def run_gui():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = ProtoNetGUI(root)

    # Handle window close
    def on_close():
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

if __name__ == "__main__":
    run_gui()