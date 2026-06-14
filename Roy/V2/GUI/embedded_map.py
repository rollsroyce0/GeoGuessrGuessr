import tkinter as tk
from tkinter import ttk, messagebox
import tempfile
import os
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import math
import io
import base64
from PIL import Image, ImageTk

# Add the parent directory to the path so we can import from the main module
sys.path.append(str(Path(__file__).resolve().parents[2]))

class EmbeddedMapDisplay:
    """
    A true embedded map display component for Tkinter that doesn't require JavaScript.
    Uses tkintermapview for pure Tkinter map rendering.
    """

    def __init__(self, parent_frame: tk.Frame):
        """
        Initialize the embedded map display component.

        Args:
            parent_frame: The parent frame where the map will be displayed
        """
        self.parent_frame = parent_frame
        self.map_widget = None
        self.current_markers = []
        self.current_lines = []

        # Create the map display frame
        self.map_frame = tk.Frame(parent_frame, bg="#f0f0f0", bd=2, relief=tk.GROOVE)
        self.map_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Title
        self.title_label = tk.Label(
            self.map_frame,
            text="World Map with Location Markers",
            font=('Arial', 12, 'bold')
        )
        self.title_label.pack(pady=5, fill=tk.X)

        # Map container
        self.map_container = tk.Frame(self.map_frame, bg="white")
        self.map_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Status label
        self.status_label = tk.Label(
            self.map_container,
            text="Select a test image to view the map",
            font=('Arial', 10),
            wraplength=400,
            justify=tk.LEFT
        )
        self.status_label.pack(fill=tk.BOTH, expand=True)

        # Check if tkintermapview is available
        self.map_view_available = self._check_map_view_availability()

        if self.map_view_available:
            self._initialize_map_widget()

    def _check_map_view_availability(self):
        """Check if tkintermapview is available"""
        try:
            import tkintermapview
            return True
        except ImportError:
            return False

    def _initialize_map_widget(self):
        """Initialize the tkintermapview widget"""
        try:
            import tkintermapview

            # Clear any existing content
            for widget in self.map_container.winfo_children():
                widget.destroy()

            # Create the map widget
            self.map_widget = tkintermapview.TkinterMapView(
                self.map_container,
                width=800,
                height=600,
                corner_radius=0
            )
            self.map_widget.pack(fill=tk.BOTH, expand=True)

            # Add navigation controls
            self._add_navigation_controls()

            # Set default view
            self.map_widget.set_position(51.5074, -0.1278)  # London
            self.map_widget.set_zoom(2)

        except Exception as e:
            self._show_error(f"Failed to initialize map widget: {str(e)}")
            self.map_view_available = False

    def _add_navigation_controls(self):
        """Add navigation controls to the map"""
        if not self.map_widget:
            return

        # Navigation frame
        nav_frame = tk.Frame(self.map_container)
        nav_frame.pack(fill=tk.X, pady=5)

        # Zoom controls
        zoom_frame = tk.Frame(nav_frame)
        zoom_frame.pack(side=tk.LEFT, padx=5)

        zoom_in_btn = tk.Button(
            zoom_frame,
            text="+",
            width=3,
            command=lambda: self.map_widget.set_zoom(self.map_widget.zoom + 1)
        )
        zoom_in_btn.pack(side=tk.TOP)

        zoom_out_btn = tk.Button(
            zoom_frame,
            text="-",
            width=3,
            command=lambda: self.map_widget.set_zoom(self.map_widget.zoom - 1)
        )
        zoom_out_btn.pack(side=tk.BOTTOM)

        # Position display
        self.position_label = tk.Label(
            nav_frame,
            text="Lat: -, Lon: -",
            font=('Arial', 10),
            relief=tk.SUNKEN,
            width=30
        )
        self.position_label.pack(side=tk.LEFT, padx=5)

        # Bind position updates
        self.map_widget.add_right_click_menu_command(
            label="Center Here",
            command=self._center_map_at_click,
            pass_coords=True
        )

        # Update position display on click
        def update_position(coords):
            lat, lon = coords
            self.position_label.config(text=f"Lat: {lat:.6f}, Lon: {lon:.6f}")

        self.map_widget.add_left_click_map_command(update_position)

    def _center_map_at_click(self, coords):
        """Center the map at the clicked coordinates"""
        if self.map_widget:
            self.map_widget.set_position(coords[0], coords[1])

    def display_map(self, real_coords: Tuple[float, float],
                   pred_coords: Tuple[float, float],
                   test_type: str, image_idx: int):
        """
        Display the map with markers for real and predicted locations.

        Args:
            real_coords: (latitude, longitude) of real location
            pred_coords: (latitude, longitude) of predicted location
            test_type: Type of test image
            image_idx: Index of test image
        """
        try:
            if not self.map_view_available or not self.map_widget:
                self._show_error("Map widget not available. Install tkintermapview for embedded maps.")
                return

            # Clear previous markers and lines
            self._clear_map()

            # Calculate center point between the two locations
            center_lat = (real_coords[0] + pred_coords[0]) / 2
            center_lon = (real_coords[1] + pred_coords[1]) / 2

            # Set map view to show both locations
            self.map_widget.set_position(center_lat, center_lon)

            # Calculate appropriate zoom level
            distance = self._calculate_distance(real_coords, pred_coords)
            zoom_level = max(2, min(12, 12 - math.log(distance, 2)))
            self.map_widget.set_zoom(int(zoom_level))

            # Add real location marker (green)
            real_marker = self.map_widget.set_marker(
                real_coords[0], real_coords[1],
                text=f"Real: {real_coords[0]:.6f}, {real_coords[1]:.6f}",
                marker_color_outside="green",
                marker_color_circle="white",
                font=("Arial", 10, "bold")
            )

            # Add predicted location marker (red)
            pred_marker = self.map_widget.set_marker(
                pred_coords[0], pred_coords[1],
                text=f"Predicted: {pred_coords[0]:.6f}, {pred_coords[1]:.6f}",
                marker_color_outside="red",
                marker_color_circle="white",
                font=("Arial", 10, "bold")
            )

            # Add line connecting the two points (blue)
            line = self.map_widget.set_path([
                real_coords,
                pred_coords
            ], color="blue", width=2)

            # Store references to prevent garbage collection
            self.current_markers.extend([real_marker, pred_marker])
            self.current_lines.append(line)

            # Update status
            distance_km = distance / 1000
            self.status_label.config(
                text=f"Map showing {test_type} Image {image_idx + 1}\n"
                    f"Real: {real_coords[0]:.6f}, {real_coords[1]:.6f}\n"
                    f"Predicted: {pred_coords[0]:.6f}, {pred_coords[1]:.6f}\n"
                    f"Distance: {distance_km:.2f} km",
                justify=tk.LEFT
            )

        except Exception as e:
            self._show_error(f"Failed to display map: {str(e)}")

    def _calculate_distance(self, coord1: Tuple[float, float],
                          coord2: Tuple[float, float]) -> float:
        """
        Calculate distance between two coordinates in meters using Haversine formula.

        Args:
            coord1: (latitude, longitude) of first point
            coord2: (latitude, longitude) of second point

        Returns:
            Distance in meters
        """
        # Convert coordinates to radians
        lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
        lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        # Earth radius in meters
        r = 6371000
        return c * r

    def _clear_map(self):
        """Clear all markers and lines from the map"""
        try:
            # Remove all markers
            for marker in self.current_markers:
                if marker:
                    marker.delete()
            self.current_markers = []

            # Remove all lines
            for line in self.current_lines:
                if line:
                    line.delete()
            self.current_lines = []

        except Exception as e:
            # Silent failure - just clear the lists
            self.current_markers = []
            self.current_lines = []

    def clear(self):
        """Clear the current map display"""
        self._clear_map()

        # Reset status
        self.status_label.config(
            text="Select a test image to view the map",
            justify=tk.LEFT
        )

        # Reset map view if available
        if self.map_widget:
            self.map_widget.set_position(51.5074, -0.1278)  # London
            self.map_widget.set_zoom(2)

    def _show_error(self, error_message: str):
        """Show error message in the map display area"""
        # Clear the map container
        for widget in self.map_container.winfo_children():
            widget.destroy()

        # Show error message
        error_label = tk.Label(
            self.map_container,
            text=f"Error: {error_message}",
            font=('Arial', 10),
            fg="red",
            wraplength=400,
            justify=tk.LEFT
        )
        error_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Show installation instructions if map view not available
        if not self.map_view_available:
            install_label = tk.Label(
                self.map_container,
                text="Install tkintermapview for embedded maps:\n"
                    "pip install tkintermapview",
                font=('Arial', 10),
                fg="#2c3e50",
                wraplength=400,
                justify=tk.LEFT
            )
            install_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    def __del__(self):
        """Clean up resources when the component is destroyed"""
        self.clear()

def check_map_view_dependencies():
    """
    Check if map view dependencies are available.

    Returns:
        bool: True if map view is available, False otherwise
    """
    try:
        import tkintermapview
        return True
    except ImportError:
        return False

def get_map_view_installation_instructions():
    """
    Get installation instructions for map view dependencies.

    Returns:
        str: Installation instructions
    """
    return (
        "For embedded map display, install tkintermapview:\n\n"
        "pip install tkintermapview\n\n"
        "This provides a pure Tkinter map widget that doesn't require JavaScript.\n"
        "After installation, restart the application."
    )