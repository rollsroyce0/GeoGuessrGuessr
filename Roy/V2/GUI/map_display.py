import tkinter as tk
from tkinter import ttk, messagebox
import tempfile
import webbrowser
import os
import sys
from pathlib import Path
from typing import Tuple, Optional

# Add the parent directory to the path so we can import from the main module
sys.path.append(str(Path(__file__).resolve().parents[2]))

class MapDisplay:
    """
    A component for displaying interactive maps in Tkinter GUI.
    Uses a web view approach to render folium-generated HTML maps.
    """

    def __init__(self, parent_frame: tk.Frame):
        """
        Initialize the map display component.

        Args:
            parent_frame: The parent frame where the map will be displayed
        """
        self.parent_frame = parent_frame
        self.current_html_file = None
        self.web_view_available = False

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

        # Check if web view is available
        self._check_web_view_availability()

    def _check_web_view_availability(self):
        """Check if web view functionality is available"""
        try:
            # Try to import tkhtmlview (lightweight HTML viewer for Tkinter)
            import tkhtmlview
            self.web_view_available = True
        except ImportError:
            try:
                # Try to import webview (more comprehensive but heavier)
                import webview
                self.web_view_available = True
            except ImportError:
                self.web_view_available = False

    def display_map(self, html_content: str):
        """
        Display the map from HTML content.

        Args:
            html_content: The HTML content of the folium map
        """
        try:
            # Clear previous content
            for widget in self.map_container.winfo_children():
                widget.destroy()

            if not html_content:
                self.status_label = tk.Label(
                    self.map_container,
                    text="No map data available",
                    font=('Arial', 10)
                )
                self.status_label.pack(fill=tk.BOTH, expand=True)
                return

            # Save HTML to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            temp_file.write(html_content.encode('utf-8'))
            temp_file.close()
            self.current_html_file = temp_file.name

            if self.web_view_available:
                # Use embedded web view if available
                self._display_embedded_map(self.current_html_file)
            else:
                # Fallback: Show instructions and open in browser
                self._display_browser_fallback(self.current_html_file)

        except Exception as e:
            self._show_error(f"Failed to display map: {str(e)}")

    def _display_embedded_map(self, html_file: str):
        """Display map using embedded web view with proper JavaScript handling"""
        try:
            # Check if the HTML contains JavaScript (folium maps do)
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Detect if this is a JavaScript-heavy folium map
            has_javascript = ('<script' in html_content and
                            'Leaflet' in html_content and
                            'L.map' in html_content)

            if has_javascript:
                # Folium maps require JavaScript - tkhtmlview can't handle this
                self._show_javascript_limitation_message()
                self._display_enhanced_browser(html_file)
                return

            # If no JavaScript, try tkhtmlview (simple HTML only)
            try:
                import tkhtmlview
                html_view = tkhtmlview.HTMLLabel(
                    self.map_container,
                    html=html_content
                )
                html_view.pack(fill=tk.BOTH, expand=True)

                # Add scrollbars
                scrollbar_y = tk.Scrollbar(self.map_container, orient=tk.VERTICAL)
                scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
                scrollbar_x = tk.Scrollbar(self.map_container, orient=tk.HORIZONTAL)
                scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

                # Configure scrolling
                html_view.configure(yscrollcommand=scrollbar_y.set)
                html_view.configure(xscrollcommand=scrollbar_x.set)
                scrollbar_y.configure(command=html_view.yview)
                scrollbar_x.configure(command=html_view.xview)

            except ImportError:
                # tkhtmlview not available, use enhanced browser
                self._display_enhanced_browser(html_file)

            except Exception as e:
                # tkhtmlview failed to render, use enhanced browser
                self._show_error(f"tkhtmlview rendering failed: {str(e)}")
                self._display_enhanced_browser(html_file)

        except Exception as e:
            self._show_error(f"Map display preparation failed: {str(e)}")
            self._display_enhanced_browser(html_file)

    def _display_browser_fallback(self, html_file: str):
        """Display map by opening in browser with instructions"""
        try:
            # Open in browser
            webbrowser.open(html_file)

            # Show instructions in GUI
            instructions = (
                f"Map opened in your default browser.\n\n"
                f"Green Marker: Real Location\n"
                f"Red Marker: Predicted Location\n"
                f"Blue Line: Distance between locations\n\n"
                f"Click markers for coordinate details.\n\n"
                f"Note: For better integration, install tkhtmlview package."
            )

            self.status_label = tk.Label(
                self.map_container,
                text=instructions,
                font=('Arial', 10),
                wraplength=400,
                justify=tk.LEFT
            )
            self.status_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        except Exception as e:
            self._show_error(f"Failed to open map in browser: {str(e)}")

    def _show_javascript_limitation_message(self):
        """Show message explaining JavaScript limitations with tkhtmlview"""
        for widget in self.map_container.winfo_children():
            widget.destroy()

        message = (
            "🌍 Interactive Map Display\n\n"
            "This map requires JavaScript for full interactivity, "
            "which tkhtmlview cannot provide.\n\n"
            "✅ The map will open in your browser for full functionality.\n"
            "✅ You can still use all features: zoom, pan, click markers.\n"
            "✅ The browser window is positioned next to this GUI.\n\n"
            "For true embedded maps, consider:\n"
            "• cefpython (full browser embedding)\n"
            "• PyWebView (cross-platform web views)\n"
            "• Custom browser component integration"
        )

        info_label = tk.Label(
            self.map_container,
            text=message,
            font=('Arial', 10),
            fg="#2c3e50",
            wraplength=400,
            justify=tk.LEFT,
            padx=10, pady=10
        )
        info_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add a "Loading..." indicator
        self.loading_label = tk.Label(
            self.map_container,
            text="🔄 Loading map in browser...",
            font=('Arial', 10, 'bold'),
            fg="#3498db"
        )
        self.loading_label.pack(pady=10)

    def _display_enhanced_browser(self, html_file: str):
        """Display map in browser with enhanced integration"""
        try:
            import webbrowser
            import subprocess
            import sys
            import platform

            # Update loading message
            if hasattr(self, 'loading_label'):
                self.loading_label.config(
                    text="🌐 Opening browser window...",
                    fg="#2ecc71"
                )

            # Open the map in the default browser
            webbrowser.open(html_file)

            # Update status to show success
            if hasattr(self, 'loading_label'):
                self.loading_label.config(
                    text="✅ Map opened successfully!",
                    fg="#27ae60"
                )

            # Show comprehensive instructions
            instructions = (
                "📍 Map Display Guide\n\n"
                "🟢 GREEN MARKER = Real Location\n"
                "🔴 RED MARKER = Predicted Location\n"
                "🔵 BLUE LINE = Distance between locations\n\n"
                "💡 Tips:\n"
                "• Click markers to see exact coordinates\n"
                "• Use mouse wheel to zoom in/out\n"
                "• Drag to pan around the map\n"
                "• The map is fully interactive\n\n"
                "🌐 Browser window should appear shortly.\n"
                "If not, check your browser settings."
            )

            # Replace loading label with full instructions
            for widget in self.map_container.winfo_children():
                widget.destroy()

            info_frame = tk.Frame(self.map_container, bg="white")
            info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Title
            title_label = tk.Label(
                info_frame,
                text="Interactive Map Viewer",
                font=('Arial', 12, 'bold'),
                fg="#2c3e50"
            )
            title_label.pack(pady=(0, 10))

            # Instructions
            instructions_label = tk.Label(
                info_frame,
                text=instructions,
                font=('Arial', 10),
                wraplength=400,
                justify=tk.LEFT,
                bg="white"
            )
            instructions_label.pack(fill=tk.BOTH, expand=True)

            # Legend
            legend_frame = tk.Frame(info_frame)
            legend_frame.pack(pady=10)

            # Green marker legend
            green_label = tk.Label(
                legend_frame,
                text="🟢 Real Location",
                font=('Arial', 10),
                fg="#27ae60"
            )
            green_label.grid(row=0, column=0, padx=10)

            # Red marker legend
            red_label = tk.Label(
                legend_frame,
                text="🔴 Predicted Location",
                font=('Arial', 10),
                fg="#e74c3c"
            )
            red_label.grid(row=0, column=1, padx=10)

            # Blue line legend
            blue_label = tk.Label(
                legend_frame,
                text="🔵 Distance Line",
                font=('Arial', 10),
                fg="#3498db"
            )
            blue_label.grid(row=1, column=0, columnspan=2, pady=5)

        except Exception as e:
            self._show_error(f"Enhanced browser display failed: {str(e)}")

    def _show_error(self, error_message: str):
        """Show error message in the map display area"""
        for widget in self.map_container.winfo_children():
            widget.destroy()

        error_label = tk.Label(
            self.map_container,
            text=f"Error: {error_message}",
            font=('Arial', 10),
            fg="red",
            wraplength=400,
            justify=tk.LEFT
        )
        error_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    def clear(self):
        """Clear the current map display"""
        for widget in self.map_container.winfo_children():
            widget.destroy()

        self.status_label = tk.Label(
            self.map_container,
            text="Select a test image to view the map",
            font=('Arial', 10),
            wraplength=400,
            justify=tk.LEFT
        )
        self.status_label.pack(fill=tk.BOTH, expand=True)

        # Clean up temporary file
        if self.current_html_file and os.path.exists(self.current_html_file):
            try:
                os.unlink(self.current_html_file)
            except:
                pass
        self.current_html_file = None

    def __del__(self):
        """Clean up resources when the component is destroyed"""
        self.clear()

def check_web_view_dependencies():
    """
    Check if web view dependencies are available and provide installation instructions.

    Returns:
        bool: True if web view is available, False otherwise
    """
    try:
        import tkhtmlview
        return True
    except ImportError:
        try:
            import webview
            return True
        except ImportError:
            return False

def get_web_view_installation_instructions():
    """
    Get installation instructions for web view dependencies.

    Returns:
        str: Installation instructions
    """
    return (
        "For better map integration, install one of these packages:\n\n"
        "1. tkhtmlview (lightweight, recommended):\n"
        "   pip install tkhtmlview\n\n"
        "2. webview (more comprehensive):\n"
        "   pip install webview\n\n"
        "After installation, restart the application."
    )