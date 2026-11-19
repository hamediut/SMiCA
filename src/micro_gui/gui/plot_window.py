"""
Plot window for displaying SMDS calculation results.
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtWidgets import QMainWindow, QFileDialog, QMessageBox

from .plot_style import figure_size, mark_phase_fraction, r_axis_label, save_figure


class PlotWindow(QMainWindow):
    """
    Separate window for displaying matplotlib plots with save options.

    This window displays the results of SMDS calculations as a line plot
    and provides options to save the plot as an image or export the data as CSV.

    Attributes:
        s2_values (np.ndarray): The S2 values to plot
        fig: Matplotlib figure object
        ax: Matplotlib axes object
    """

    def __init__(self, s2_values: np.ndarray, parent=None, ndim: int = 2):
        """
        Initialize the plot window.

        Args:
            s2_values: NumPy array of S2 values to display
            parent: Parent window (optional)
            ndim: 2 if S2 was computed on a 2D image, 3 for a 3D volume (sets the r-axis unit)

        Parent is ImageViewer. Why have a parent?
        - Plot window appears on top of main window
        - When main window closes, plot windows close too
        - Better memory management

        """
        super().__init__(parent)

        self.setWindowTitle("SMDS Calculation Results")
        self.setGeometry(200, 200, 800, 600)
        self.s2_values = s2_values

        # Create matplotlib figure
        # font sizes and line widths come from the shared style in plot_style.py
        self.fig, self.ax = plt.subplots(figsize=figure_size())
        self.ax.plot(s2_values)
        mark_phase_fraction(self.ax, s2_values[0])  # S2(0) is the phase fraction
        self.ax.set_xlabel(r_axis_label(ndim))
        self.ax.set_ylabel('$S_{2}$')
        self.ax.set_title('Two-point Correlation Function ($S_{2}$)')
        self.ax.grid(True, alpha=0.3)

        # Create canvas widget
        canvas = FigureCanvas(self.fig)

        # Set canvas as central widget
        self.setCentralWidget(canvas)

        # Create menu bar
        self._create_menu()

    def _create_menu(self):
        """Create the menu bar with file operations."""
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        # Save plot as image
        save_plot_action = file_menu.addAction("&Save Plot as Image...")
        save_plot_action.setShortcut("Ctrl+P")
        save_plot_action.triggered.connect(self.save_plot)

        # Export data as CSV
        export_csv_action = file_menu.addAction("&Export Data as CSV...")
        export_csv_action.setShortcut("Ctrl+E")
        export_csv_action.triggered.connect(self.export_csv)

    def save_plot(self):
        """Save the plot as PNG or JPEG image file."""
        # Open file dialog to select save location and format
        # It returns a tuple (file_path, selected_filter). selected_filter is the file type chosen by user.
        file_path, _ = QFileDialog.getSaveFileName(
            self, # parent (QWidget or QMainWindow)
            "Save Plot",
            "",# initial folder or suggested filename
            # selected filters (allowed file types)
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;PDF Document (*.pdf);;All Files (*)"
        )

        if file_path: # If user didn't cancel and selected a file path
            try:
                save_figure(self.fig, file_path)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")

    def export_csv(self):
        """Export S2 values to CSV file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export CSV",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            try:
                # Ensure .csv extension
                if not file_path.endswith('.csv'):
                    file_path += '.csv'

                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile, delimiter=';')
                    writer.writerow(['r', 's2'])
                    for i, value in enumerate(self.s2_values):
                        writer.writerow([i, np.round(value, decimals=4)])

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export CSV:\n{str(e)}")
