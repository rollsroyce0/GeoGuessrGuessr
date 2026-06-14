import tkinter as tk
from tkinter import ttk
import threading
import time
from typing import Callable, Any

class ProgressTracker:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar()
        self.status_var.set("Initializing...")

        # Create progress window
        self.progress_window = tk.Toplevel(root)
        self.progress_window.title("ProtoNet Test - Progress")
        self.progress_window.geometry("400x150")
        self.progress_window.protocol("WM_DELETE_WINDOW", self._on_close)

        # Set window to stay on top and get focus
        self.progress_window.attributes("-topmost", True)
        self.progress_window.lift()
        self.progress_window.focus_force()

        # Position window in center of screen
        self.progress_window.update_idletasks()
        screen_width = self.progress_window.winfo_screenwidth()
        screen_height = self.progress_window.winfo_screenheight()
        window_width = 400
        window_height = 150
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.progress_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            self.progress_window,
            variable=self.progress_var,
            maximum=100,
            length=300
        )
        self.progress_bar.pack(pady=20)

        # Status label
        self.status_label = tk.Label(
            self.progress_window,
            textvariable=self.status_var,
            wraplength=350
        )
        self.status_label.pack(pady=10)

        # Percentage label
        self.percentage_var = tk.StringVar()
        self.percentage_var.set("0%")
        self.percentage_label = tk.Label(
            self.progress_window,
            textvariable=self.percentage_var,
            font=('Arial', 12, 'bold')
        )
        self.percentage_label.pack(pady=5)

        self.current_progress = 0
        self.total_items = 0
        self.completed_items = 0
        self.running = True

    def _on_close(self):
        """Handle window close event"""
        self.running = False
        self.progress_window.destroy()

    def start_progress(self, total_items: int, description: str = "Processing..."):
        """Start progress tracking"""
        self.total_items = total_items
        self.completed_items = 0
        self.current_progress = 0
        self.status_var.set(description)
        self.percentage_var.set("0%")
        self.progress_var.set(0)
        self.running = True

    def update_progress(self, increment: int = 1, status: str = None):
        """Update progress by increment"""
        if not self.running:
            return

        self.completed_items += increment
        if self.total_items > 0:
            self.current_progress = (self.completed_items / self.total_items) * 100
            self.current_progress = min(100, max(0, self.current_progress))

        self.progress_var.set(self.current_progress)
        self.percentage_var.set(f"{self.current_progress:.1f}%")

        if status:
            self.status_var.set(status)

        # Update the GUI
        self.root.update_idletasks()

    def set_status(self, status: str):
        """Update status text"""
        if self.running:
            self.status_var.set(status)

    def complete(self):
        """Mark progress as complete"""
        self.current_progress = 100
        self.progress_var.set(100)
        self.percentage_var.set("100%")
        self.status_var.set("Processing complete!")
        self.root.update_idletasks()

    def is_running(self) -> bool:
        """Check if progress tracker is still running"""
        return self.running

    def get_progress_window(self) -> tk.Toplevel:
        """Get the progress window reference"""
        return self.progress_window

    def close(self):
        """Close the progress window"""
        self.running = False
        if self.progress_window:
            self.progress_window.destroy()

    def create_progress_callback(self):
        """Create a progress callback function for external use"""
        def callback(progress_percent, status_message):
            if self.running:
                self.progress_var.set(progress_percent)
                self.percentage_var.set(f"{progress_percent:.1f}%")
                self.status_var.set(status_message)
                self.root.update_idletasks()
        return callback

class GUIProgressWrapper:
    def __init__(self, progress_tracker: ProgressTracker):
        self.progress_tracker = progress_tracker
        self.items = []
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index < len(self.items):
            item = self.items[self.current_index]
            self.current_index += 1
            self.progress_tracker.update_progress()
            return item
        else:
            raise StopIteration

    def __len__(self):
        return len(self.items)

    def set_items(self, items):
        self.items = list(items)
        self.current_index = 0
        self.progress_tracker.start_progress(len(items))