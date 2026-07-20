"""
Plot window for displaying REV analysis results.
"""

import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QTabWidget, QWidget, QVBoxLayout

from .save_dialog_helper import suggested_save_path, remember_save_dir


class RevPlotWindow(QMainWindow):
    """
    Window to display REV analysis results.
    Shows S2 and F2 correlation functions for different subvolume sizes
    to help determine the REV.
    """

    def __init__(self, s2_dict, f2_dict, calc_type="REV", parent = None):
        super().__init__(parent)

        self.setWindowTitle(f"{calc_type} Analysis Results")
        self.setGeometry(150, 150, 1000, 700)

        self.s2_dict = s2_dict
        self.f2_dict = f2_dict
        self.calc_type = calc_type

        # Create tab widget to show S2 and F2 separately
        self.tabs = QTabWidget() # we add self.tabs to access it later to check which tab is currently active

        # S2 Tab
        s2_widget = QWidget()
        s2_layout = QVBoxLayout()
        s2_canvas = self._create_s2_plot()
        s2_layout.addWidget(s2_canvas)
        s2_widget.setLayout(s2_layout)
        self.tabs.addTab(s2_widget, "S2")

        # F2 Tab
        f2_widget = QWidget()
        f2_layout = QVBoxLayout()
        f2_canvas = self._create_f2_plot()
        f2_layout.addWidget(f2_canvas)
        f2_widget.setLayout(f2_layout)
        self.tabs.addTab(f2_widget, "F2")

        self.setCentralWidget(self.tabs)

        # create menu
        self._create_menu()

    def _create_s2_plot(self):
        """ Create the S2 correlation plot."""

        self.fig_s2, self.ax_s2 = plt.subplots(figsize=(10, 6))

        # plot each subvolume size
        for key, value in self.s2_dict.items():

            if key == 'original':
                self.ax_s2.plot(value, linewidth = 2, color = 'black', label = 'Original')

            else:
                # subvolumes are Dataframes with mean and std
                self.ax_s2.plot(self.s2_dict[key]['s2']['mean'], linewidth = 1.5, label = f'Size {key.split("_")[1]}')
        
        self.ax_s2.set_xlabel('Distance (r)', fontsize=16)
        self.ax_s2.set_ylabel('$S_2$', fontsize=16)
        self.ax_s2.tick_params(axis='both', which='major', labelsize=12)
        self.ax_s2.set_title(f'{self.calc_type} Analysis - S2', fontsize=14, fontweight='bold')
        self.ax_s2.legend()
        self.ax_s2.grid(True, alpha=0.3)
        
        return FigureCanvas(self.fig_s2)

    def _create_f2_plot(self):
        """Create the F2 autocovariance plot."""
        self.fig_f2, self.ax_f2 = plt.subplots(figsize=(10, 6))
        
        # Plot each subvolume size
        for key, value in self.f2_dict.items():
            if key == 'original':
                self.ax_f2.plot(value, linewidth=2, color='black', label='Original')
            else:
                # Subvolumes are DataFrames with mean and std
                self.ax_f2.plot(self.f2_dict[key]['f2']['mean'], linewidth=1.5, label=f'Size {key.split("_")[1]}')
        
        self.ax_f2.set_xlabel('Distance (r)', fontsize=16)
        self.ax_f2.set_ylabel('$F_2$', fontsize=16)
        self.ax_f2.tick_params(axis='both', which='major', labelsize=12)
        self.ax_f2.set_title('REV Analysis - F2', fontsize=14, fontweight='bold')
        self.ax_f2.legend()
        self.ax_f2.grid(True, alpha=0.3)
        
        return FigureCanvas(self.fig_f2)
    
    def _create_menu(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        
        save_action = file_menu.addAction("&Save Plots...")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_plots)

    # def save_plots(self):
    #     """Save both plots."""
    #     # You can implement this similar to PlotWindow.save_plot()
    #     QMessageBox.information(self, "Info", "Save functionality - implement as needed")

    def save_plots(self):
        """Save the currently active plot as PNG, JPEG, or PDF image file."""
        
        # Get which tab is currently selected (0 for S2, 1 for F2)
        current_tab_index = self.tabs.currentIndex()

        # Select the correct figure based on active tab
        if current_tab_index == 0:
            current_fig = self.fig_s2
            plot_name = "S2"
        else:
            current_fig = self.fig_f2
            plot_name = "F2"

        # Open file dialog to select save location and format
        # It returns a tuple (file_path, selected_filter). selected_filter is the file type chosen by user.
        file_path, _ = QFileDialog.getSaveFileName(
            self, # parent (QWidget or QMainWindow)
            f"Save {plot_name} Plot",
            suggested_save_path(f"rev_{plot_name.lower()}_plot"),  # opens in the last-used save folder
            # selected filters (allowed file types)
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;PDF Document (*.pdf);;All Files (*)"
        )

        if file_path: # If user didn't cancel and selected a file path
            try:
                current_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                remember_save_dir(file_path)  # so the next Save dialog opens in this same folder
                QMessageBox.information(self, "Success", f"{plot_name} plot saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")
