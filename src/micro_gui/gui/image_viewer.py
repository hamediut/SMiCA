"""
Main image viewer window for the Micro_GUI application.
"""

import numpy as np
from typing import Optional, List

from PySide6.QtWidgets import (
    QMainWindow, QLabel, QFileDialog, QScrollArea, QStatusBar, QDialog,
    QSlider, QVBoxLayout, QWidget, QMessageBox, QProgressBar, QApplication
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QThread, Signal

from PIL import Image

from .widgets import ImageDisplayWidget
from .plot_window import PlotWindow
from .rev_plot_window import RevPlotWindow
from ..analysis.smds import RES, calculate_s2, calculate_s2_3d, REV, RES
from .rev_settings_dialog import REVSettingsDialog


class CalculationThread(QThread):
    """
    Thread for running SMDS calculation in background.

    Attributes:
        finished: Signal emitted when calculation completes with result
        error: Signal emitted when an error occurs with error message
    """
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, image_data: np.ndarray, is_3d: bool = False):
        """
        Initialize the calculation thread.

        Args:
            image_data: The image data to process
            is_3d: Whether this is a 3D image volume
        """
        super().__init__()
        self.image_data = image_data
        self.is_3d = is_3d

    def run(self):
        """Run the calculation in a separate thread."""
        try:
            if self.is_3d:
                result = calculate_s2_3d(self.image_data)
            else:
                result = calculate_s2(self.image_data)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class REVCalculationThread(QThread):
    """

    Thread for running REV calculation in background.
    """
    finished = Signal(object, object) # Emits (s2_dict, f2_dict)
    error = Signal(str)
    progress = Signal(str)  # Optional: to show progress

    def __init__(self, image_data: np.ndarray,
                  img_size_list: List[int] = [32, 64, 128],
                  n_rand_samples = 10):
        super().__init__()
        self.image_data = image_data
        self.img_size_list = img_size_list
        self.n_rand_samples = n_rand_samples
        self.calculation_type = 'REV'  # Default, will be updated in run()

    def run(self):
        """Run REV or RES calculation in background thread."""
        
        try:
            # check if image is 2D or 3D
            if self.image_data.ndim == 2:
                 # 2D image - calculate RES
                 s2_dict, f2_dict = RES(self.image_data,
                                        self.img_size_list,
                                       self.n_rand_samples
                                       )
                 self.calculation_type = "RES"
            elif self.image_data.ndim == 3:
                # 3D image - calculate REV
                s2_dict, f2_dict = REV(self.image_data,
                                       self.img_size_list,
                                       self.n_rand_samples
                                       )
                self.calculation_type = "REV"
            else:
                raise ValueError("Image data must be 2D or 3D for REV/RES calculation.")

            self.finished.emit(s2_dict, f2_dict)
        except Exception as e:
            self.error.emit(str(e))

class ImageViewer(QMainWindow):
    """
    Main window for the Micro_GUI image viewer application.

    Provides functionality to load binary TIF images (2D and 3D), display them,
    and perform SMDs (Statistical microstructure descriptors) calculations.

    Attributes:
        plot_windows: List of open plot windows
        current_image_data: Currently loaded image data (can be 2D or 3D)
        current_file_path: Path to the currently loaded file
        current_slice_index: Current slice being viewed (for 3D images)
    """

    def __init__(self):
        """Initialize the ImageViewer main window."""
        super().__init__()

        self.setWindowTitle("SMiCA - Statistical Microstructure Characterisation & Analysis")
        self.setGeometry(100, 100, 800, 600)

        # Store opened windows and image data
        self.plot_windows: List[PlotWindow] = []
        self.current_image_data: Optional[np.ndarray] = None
        self.current_file_path: Optional[str] = None
        self.current_slice_index: int = 0

        # Initialize UI components
        self._setup_ui()
        self._create_menu()
        self._create_status_bar()

    def _setup_ui(self):
        """Set up the main UI components."""
        # Create central widget with layout
        central_widget = QWidget()
        self.layout = QVBoxLayout()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

        # Create image display widget
        self.image_label = ImageDisplayWidget(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText(
            "No image loaded.\n\n"
            "Use File > Open Image to load a TIF image."
        )
        self.image_label.setWordWrap(True)

        # Create scroll area for image
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        self.layout.addWidget(scroll_area)

        # Create slider for 3D images (initially hidden)
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        self.slice_slider.setVisible(False)
        self.layout.addWidget(self.slice_slider)

        # Slider label
        self.slice_label = QLabel("Slice: 0 / 0")
        self.slice_label.setAlignment(Qt.AlignCenter)
        self.slice_label.setVisible(False)
        self.layout.addWidget(self.slice_label)

        # Store current pixmap
        self.current_pixmap: Optional[QPixmap] = None

    def _create_menu(self):
        """Create the application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = file_menu.addAction("&Open Image")
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("Open a TIF image file")
        open_action.triggered.connect(self.open_image)

        exit_action = file_menu.addAction("E&xit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)

        # Calculate menu
        calculate_menu = menubar.addMenu("&Calculate")

        smds_action = calculate_menu.addAction("Calculate &SMDS")
        smds_action.setShortcut("Ctrl+S")
        smds_action.setStatusTip("Calculate SMDS (S2) from the current image")
        smds_action.triggered.connect(self.calculate_smds)

        # REV menu
        rev_menu = menubar.addMenu("&REV/RES")
        rev_action = rev_menu.addAction("Calculate &REV/RES")
        rev_action.setShortcut("Ctrl+R")
        rev_action.setStatusTip("Calculate REV from the current image")
        rev_action.triggered.connect(self.calculate_rev)  # Placeholder, implement
        rev_action

    def _create_status_bar(self):
        """Create the status bar with progress indicator."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Create progress bar for status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(0)  # Indeterminate progress
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximumWidth(150)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        self.status_bar.showMessage("Ready")

    def validate_binary_image(self, image_data: np.ndarray) -> tuple[bool, np.ndarray]:
        """
        Validate that image contains only 0 and 1 values.

        Args:
            image_data: Image data to validate

        Returns:
            Tuple of (is_valid, unique_values)
        """
        unique_values = np.unique(image_data)
        if not np.all(np.isin(unique_values, [0, 1])):
            return False, unique_values
        return True, unique_values

    def numpy_to_qpixmap(self, image_data: np.ndarray) -> QPixmap:
        """
        Convert numpy array to QPixmap (binary: 1=white, 0=black).

        Args:
            image_data: 2D numpy array with binary values

        Returns:
            QPixmap for display
        """
        # For binary images, multiply by 255 to get full white/black contrast
        if image_data.dtype in [np.uint8, np.uint16]:
            normalized = (image_data * 255).astype(np.uint8)
        else:
            normalized = image_data.astype(np.uint8) * 255

        height, width = normalized.shape
        bytes_per_line = width

        # Create QImage from numpy array
        q_image = QImage(
            normalized.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_Grayscale8
        )

        return QPixmap.fromImage(q_image)

    def display_current_slice(self):
        """Display the current slice of the image."""
        if self.current_image_data is None:
            return

        # Get the current 2D slice
        if self.current_image_data.ndim == 3:
            current_2d = self.current_image_data[self.current_slice_index, :, :]
        else:
            current_2d = self.current_image_data

        # Update the image label data for hover
        self.image_label.set_image_data(current_2d)

        # Convert to pixmap and display
        self.current_pixmap = self.numpy_to_qpixmap(current_2d)
        scaled_pixmap = self.current_pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def on_slice_changed(self, value: int):
        """
        Handle slice slider change.

        Args:
            value: New slice index
        """
        self.current_slice_index = value
        if self.current_image_data is not None and self.current_image_data.ndim == 3:
            self.slice_label.setText(f"Slice: {value} / {self.current_image_data.shape[0] - 1}")
            self.display_current_slice()

    def open_image(self):
        """Open a file dialog to select a TIF image and display it."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open TIF Image",
            "",
            "TIF Files (*.tif *.tiff);;All Files (*)"
        )

        if file_path:
            try:
                # Load TIF image using PIL
                pil_image = Image.open(file_path)

                # Check if it's a multi-page TIFF (3D image)
                image_stack = []
                try:
                    while True:
                        image_stack.append(np.array(pil_image))
                        pil_image.seek(pil_image.tell() + 1)
                except EOFError:
                    pass  # End of frames

                # Convert to numpy array
                if len(image_stack) > 1:
                    image_data = np.stack(image_stack, axis=0)
                else:
                    image_data = image_stack[0]

                # Validate binary image
                is_valid, unique_values = self.validate_binary_image(image_data)
                if not is_valid:
                    QMessageBox.critical(
                        self,
                        "Invalid Image",
                        f"Error: Image must contain only binary values (0 and 1).\n\n"
                        f"Found values: {unique_values}"
                    )
                    return

                # Store current image data
                self.current_image_data = image_data
                self.current_file_path = file_path
                self.current_slice_index = 0

                # Configure slider for 3D images
                if image_data.ndim == 3:
                    num_slices = image_data.shape[0]
                    self.slice_slider.setMaximum(num_slices - 1)
                    self.slice_slider.setValue(0)
                    self.slice_slider.setVisible(True)
                    self.slice_label.setText(f"Slice: 0 / {num_slices - 1}")
                    self.slice_label.setVisible(True)
                else:
                    self.slice_slider.setVisible(False)
                    self.slice_label.setVisible(False)

                # Display the first slice
                self.display_current_slice()

                # Update status
                self.status_bar.showMessage(f"Loaded: {file_path}")
                import os
                file_name = os.path.basename(file_path)
                if image_data.ndim == 3:
                    dims = f"{image_data.shape[1]} x {image_data.shape[2]} x {image_data.shape[0]}"
                else:
                    dims = f"{image_data.shape[0]} x {image_data.shape[1]}"
                self.setWindowTitle(f"SMiCA - [{dims}] - {file_name}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")
                self.status_bar.showMessage(f"Error: {str(e)}")

    def calculate_smds(self):
        """Calculate SMDS using the appropriate function and display results."""
        if self.current_image_data is None:
            self.status_bar.showMessage("No image loaded. Please open an image first.")
            QMessageBox.warning(self, "No Image", "Please open an image first.")
            return

        # Check if 3D
        is_3d = self.current_image_data.ndim == 3

        # Show progress bar in status bar
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage("Calculating SMDS...")

        # Process events to ensure UI updates
        QApplication.processEvents()

        # Create and start calculation thread
        self.calc_thread = CalculationThread(self.current_image_data, is_3d)
        self.calc_thread.finished.connect(self.on_calculation_finished)
        self.calc_thread.error.connect(self.on_calculation_error)
        self.calc_thread.start()

    def on_calculation_finished(self, s2_values: np.ndarray):
        """
        Handle completion of SMDS calculation.

        Args:
            s2_values: Calculated S2 values
        """
        self.progress_bar.setVisible(False)

        # Create a new window to display the plot
        plot_window = PlotWindow(s2_values, self)
        self.plot_windows.append(plot_window)
        plot_window.show()

        self.status_bar.showMessage("SMDS calculation completed")

    def on_calculation_error(self, error_msg: str):
        """
        Handle error during SMDS calculation.

        Args:
            error_msg: Error message
        """
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Calculation Error", f"Error calculating SMDS:\n{error_msg}")
        self.status_bar.showMessage(f"Error: {error_msg}")

    ## calculate_rev placeholder
    def calculate_rev(self):
        """Calculate REV/RES from current image and display results."""
        if self.current_image_data is None:
            QMessageBox.warning(self, "No Image", "Please open an image first.")
            return
        
        #  Get image dimensions for dialog
        if self.current_image_data.ndim == 2:
            max_size = min(self.current_image_data.shape)
            calc_name = "RES (2D)"
        elif self.current_image_data.ndim == 3:
            max_size = min(self.current_image_data.shape)
            calc_name = "REV (3D)"
        else:
            QMessageBox.warning(
                self,
                "Invalid Image",
                f"Image must be either 2D or 3D, got {self.current_image_data.ndim}D."
            )
            return
        
        # Get maximum allowed size for subvolumes
        
        # Show settings dialog
        dialog = REVSettingsDialog(max_size, calc_name, self)

        if dialog.exec() != QDialog.Accepted:
            return  # User cancelled
        
        img_size_list, n_rand_samples = dialog.get_values()

        # # These should be smaller than the image dimensions
        # min_dim = min(self.current_image_data.shape)
        # img_size_list = [32, 64, 128, 256]  # Example sizes
        # # Filter out sizes larger than image
        # img_size_list = [s for s in img_size_list if s < min_dim]
        # n_rand_samples = 30  # Example number of random samples

        # Show progress bar in status bar
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage(f"Calculating {calc_name} with sizes {img_size_list}, {n_rand_samples} samples...")
        
        # Process events to ensure UI updates
        QApplication.processEvents()

        # Create and start calculation thread
        self.rev_thread = REVCalculationThread(
            self.current_image_data,
            img_size_list,
            n_rand_samples
        )
        self.rev_thread.finished.connect(self.on_rev_finished)
        self.rev_thread.error.connect(self.on_rev_error)
        self.rev_thread.start()

    def on_rev_finished(self, s2_dict: dict, f2_dict: dict):
        """
        Handle completion of REV/RES calculation.

        Args:
            s2_dict: Dictionary of S2 values for different subvolume sizes
            f2_dict: Dictionary of F2 values for different subvolume sizes
        """
        # Get calculation type from thread
        calc_type = self.rev_thread.calculation_type

        self.progress_bar.setVisible(False)

        # Create a new window to display the REV plot
        rev_window = RevPlotWindow(s2_dict, f2_dict, calc_type, self)
        self.plot_windows.append(rev_window)
        rev_window.show()

        self.status_bar.showMessage(f"{calc_type} calculation completed")
    def on_rev_error(self, error_msg: str):
        """
        Handle error during REV calculation.

        Args:
            error_msg: Error message
        """
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "REV Error", f"Error calculating REV:\n{error_msg}")
        self.status_bar.showMessage(f"Error: {error_msg}")

    def resizeEvent(self, event):
        """Handle window resize."""
        super().resizeEvent(event)
        if self.current_pixmap and not self.current_pixmap.isNull():
            self.display_current_slice()

    def update_status(self, message: str):
        """
        Update the status bar message.

        Args:
            message: Message to display
        """
        self.status_bar.showMessage(message)
