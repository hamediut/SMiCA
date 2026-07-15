"""
Main image viewer window for the Micro_GUI application.
"""

import os
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
from .rev_plot_window import RevPlotWindow
from ..analysis.smds import (
    RES, REV,
     calculate_s2, calculate_s2_periodic , calculate_s2_3d,
     calculate_polytopes_python, calculate_c2,
     cal_fn, scale_polytope_fn, scale_by_initial_value, scale_c2_by_connectedness,
     calculate_c2_3d, calculate_s2_periodic_3d,
     calculate_L_3d
)

from .rev_settings_dialog import REVSettingsDialog
from .polytope_settings_dialog import PolytopeSettingsDialog
from .polytope_plot_window import PolytopePlotWindow
from .binarize_dialog import BinarizeDialog

from .import_sequence_dialog import ImportSequenceDialog




## caclulation threads for background processing, so GUI remains responsive

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

class PolytopeCalculationThread(QThread):
    """
    Thread for running polytope function calculations in the background.

    S2 is computed with the existing calculate_s2() (non-periodic boundary)
     so it matches the "Calculate SMDS" feature elsewhere in the GUI. All other selected
    functions (P3H, P3V, P4, P6, L) are computed in one call to calculate_polytopes_python(),
      the pure-Python/numba port of the C++ code.
    """


    finished = Signal(object, object)  # Emits (raw_curves, scaled_curves)
    error = Signal(str)

    def __init__(self, image_data: np.ndarray, selected_polytopes: List[str]):
        """
        Initialize the polytope calculation thread.

        Args:
            image_data: The 2D image data to process
            selected_polytopes: List of selected polytope function names
        """
        super().__init__()
        self.image_data = image_data
        self.selected_polytopes = selected_polytopes

    def run(self):
        """Run the selected polytope calculations in a separate thread."""
        
        try:
            raw_curves = {}
            scaled_curves = {}

            # S2 uses the same (non-periodic) code as "Calculate SMDS", so results
            # are consistent between the two menu items for the same image.

            if 's2' in self.selected_polytopes:
                if self.image_data.ndim == 3:
                    s2_values = calculate_s2_3d(self.image_data)
                else:
                    s2_values = calculate_s2(self.image_data)
                f2_values = cal_fn(s2_values, n = 2)
                r_axis =  np.arange(len(s2_values), dtype=np.float64)

                # Wrapped as a 2-column [r, value] array, same shape convention as
                    # calculate_polytopes_python's output, so every curve downstream
                    # (plotting, CSV export) can be handled identically regardless of
                    # which function computed it.

                raw_curves['s2'] = np.column_stack((r_axis, s2_values))
                scaled_curves['s2'] = np.column_stack((r_axis, f2_values))

            if 'c2' in self.selected_polytopes:

                if self.image_data.ndim == 3:
                    c2_values = calculate_c2_3d(self.image_data)
                    s2_periodic_values = calculate_s2_periodic_3d(self.image_data)  # S2 with the SAME periodic convention as C2 (needed as the denominator)
                else:
                    c2_values = calculate_c2(self.image_data)
                    s2_periodic_values = calculate_s2_periodic(self.image_data)  # S2 with the SAME periodic convention as C2 (needed as the denominator)

                # --- earlier scaling approaches, kept for reference (see smds.py docstrings) ---
                # f2_c2_values = scale_polytope_fn(c2_values)     # subtracts an assumed long-range plateau - wrong for C2, whose plateau is percolation-driven, not phi^n
                # f2_c2_values = scale_by_initial_value(c2_values)  # simple C2(r)/C2(0) - starts at 1, but unbounded/noisy elsewhere

                # --- current approach: conditional connectedness C2(r)/S2(r) ---
                f2_c2_values = scale_c2_by_connectedness(c2_values, s2_periodic_values)  # P(same cluster | same phase) at r - bounded [0,1] at every r, not just r=0

                r_axis = np.arange(len(c2_values), dtype=np.float64)
                raw_curves['c2'] = np.column_stack((r_axis, c2_values))
                scaled_curves['c2'] = np.column_stack((r_axis, f2_c2_values))
            
            if 'L' in self.selected_polytopes and self.image_data.ndim == 3:
                L_values = calculate_L_3d(self.image_data)
                f_L_values = scale_by_initial_value(L_values)  # L decays to 0 (not a nonzero plateau like C2), so simple L(r)/L(0) is the right scaling here
                r_axis = np.arange(len(L_values), dtype=np.float64)
                raw_curves['L'] = np.column_stack((r_axis, L_values))
                scaled_curves['L'] = np.column_stack((r_axis, f_L_values))  # L is already bounded [0,1], so no further scaling needed
            
            # 2D 'L' still goes through calculate_polytopes_python below, same as before - only 3D 'L'
            # gets the dedicated branch above, since the 2D path is already validated against the C++.
            other_polytopes = [
                name for name in self.selected_polytopes
                if name not in ('s2', 'c2') and not (name == 'L' and self.image_data.ndim == 3)
            ]
            if other_polytopes:
                other_raw, other_scaled = calculate_polytopes_python(
                    self.image_data, polytopes=tuple(other_polytopes)
                    )
                raw_curves.update(other_raw)
                scaled_curves.update(other_scaled)

            self.finished.emit(raw_curves, scaled_curves)
        except Exception as e:
            self.error.emit(str(e))


##-----------------------------------------------------


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
        self.plot_windows: List[QMainWindow] = []
        self.current_image_data: Optional[np.ndarray] = None
        self.data_mode: Optional[str] = None
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

        import_volume_action = file_menu.addAction("Import &Volume from Slice Folder")
        import_volume_action.setStatusTip("Import a 3D volume from a folder of 2D Z-slice image files")
        import_volume_action.triggered.connect(self.open_import_volume_dialog)


        exit_action = file_menu.addAction("E&xit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)

        # Process menu
        process_menu = menubar.addMenu("&Process")

        binarize_action = process_menu.addAction("&Binarize")
        binarize_action.setStatusTip("Select a pixel value to become the foreground (1); everything else becomes background (0)")
        binarize_action.triggered.connect(self.open_binarize_dialog)


        # SMD menu
        smd_menu = menubar.addMenu("&SMDs")

        smds_action = smd_menu.addAction("Calculate &SMDs")
        # smds_action.setShortcut("Ctrl+S")
        smds_action.setStatusTip("Calculate SMDs from the current image")
        smds_action.triggered.connect(self.open_smd_dialog)

        # # Polytopes
        # polytope_action = calculate_menu.addAction("Calculate &Polytopes...")
        # polytope_action.setStatusTip("Calculate polytopes functions (S2, P3H, P3V, P4, P6, L) from the current image")
        # polytope_action.triggered.connect(self.calculate_polytopes_dialog)

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
        Convert numpy array to QPixmap for display.
        Uses min-max normalization instead of assuming binary 0/1 values, so this works both
        for binary images (0 -> 0, 1 -> 255, same as before) and multi-label segmentations
        (e.g. 0,1,2,3 -> 0,85,170,255) without overflowing.

        Args:
            image_data: 2D numpy array (binary or multi-label integer values)

        Returns:
            QPixmap for display
        """

        data_min = image_data.min()
        data_max = image_data.max()

        if data_max > data_min:
            # Normalize to [0, 255]
            normalized = ((image_data.astype(np.float64) - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        else:
            # every pixel has the same value (e.g. a blank slice) - avoid a divide-by-zero, show flat mid-grey
            normalized = np.full_like(image_data, 128, dtype=np.uint8)

        # # For binary images, multiply by 255 to get full white/black contrast
        # if image_data.dtype in [np.uint8, np.uint16]:
        #     normalized = (image_data * 255).astype(np.uint8)
        # else:
        #     normalized = image_data.astype(np.uint8) * 255

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

                # # Note whether this image needs binarizing before it can be used in calculations -
                # doesn't block loading, just informs the status message below.

                is_valid, unique_values = self.validate_binary_image(image_data)
                # if not is_valid:
                #     QMessageBox.critical(
                #         self,
                #         "Invalid Image",
                #         f"Error: Image must contain only binary values (0 and 1).\n\n"
                #         f"Found values: {unique_values}"
                #     )
                #     return

                # # Store current image data
                # self.current_image_data = image_data
                # self.current_file_path = file_path
                # self.current_slice_index = 0

                # # Configure slider for 3D images
                # if image_data.ndim == 3:
                #     num_slices = image_data.shape[0]
                #     self.slice_slider.setMaximum(num_slices - 1)
                #     self.slice_slider.setValue(0)
                #     self.slice_slider.setVisible(True)
                #     self.slice_label.setText(f"Slice: 0 / {num_slices - 1}")
                #     self.slice_label.setVisible(True)
                # else:
                #     self.slice_slider.setVisible(False)
                #     self.slice_label.setVisible(False)

                # # Display the first slice
                # self.display_current_slice()

                # # Update status
                # if is_valid:
                #     self.status_bar.showMessage(f"Loaded: {file_path}")
                # else:
                #     self.status_bar.showMessage("Multi-label image loaded. binarization required.")
                #     QMessageBox.warning(
                #         self,
                #         "Binarization Required",
                #         f"This image has {len(unique_values)} distinct pixel values: {list(unique_values)}.\n\n"
                #         "use Process > Binarize to convert it to select a foreground label before running calculations."
                #     )

                # file_name = os.path.basename(file_path)
                # if image_data.ndim == 3:
                #     dims = f"{image_data.shape[1]} x {image_data.shape[2]} x {image_data.shape[0]}"
                # else:
                #     dims = f"{image_data.shape[0]} x {image_data.shape[1]}"
                # self.setWindowTitle(f"SMiCA - [{dims}] - {file_name}")

                self.current_file_path = file_path
                self._finalize_loaded_image(image_data, os.path.basename(file_path), data_mode='3d_volume' if image_data.ndim == 3 else '2d')



            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")
                self.status_bar.showMessage(f"Error: {str(e)}")

    def _finalize_loaded_image(self, image_data: np.ndarray, source_label: str, data_mode: str):

        """
        Shared tail-end of loading any image (single file or imported sequence): validates
        binary-ness, configures the slice slider, displays the first slice, and updates the
        status bar/title.
        """

        is_valid, unique_values = self.validate_binary_image(image_data)

        self.current_image_data = image_data
        self.current_slice_index = 0
        self.data_mode = data_mode

        if image_data.ndim == 3:
            num_slices = image_data.shape[0]
            self.slice_slider.setMaximum(num_slices - 1)
            self.slice_slider.setValue(0)
            self.slice_slider.setVisible(True)
            self.slice_label.setText(f"Slice: 0 / {num_slices - 1}")
            self.slice_label.setVisible(True)
            dims = f"{image_data.shape[1]} x {image_data.shape[2]} x {image_data.shape[0]}"
        else:
            self.slice_slider.setVisible(False)
            self.slice_label.setVisible(False)
            dims = f"{image_data.shape[0]} x {image_data.shape[1]}" 

        self.display_current_slice()

        if is_valid:
            self.status_bar.showMessage(f"Loaded: {source_label}")
        else:
            self.status_bar.showMessage("Multi-label image loaded - binarization required")
            QMessageBox.warning(
                self, "Binarization Required",
                f"This image has {len(unique_values)} distinct pixel values: {list(unique_values)}.\n\n"
                "Use Process > Binarize to select a foreground label before running calculations."
            )

        self.setWindowTitle(f"SMiCA - [{dims}] - {source_label}")   


    def open_import_volume_dialog(self):

        """Import a 3D volume assembled from a folder of 2D Z-slice files."""

        dialog =  ImportSequenceDialog(ask_file_type = False, parent = self)
        if dialog.exec() != QDialog.Accepted:
            return
        
        file_paths =  dialog.get_sorted_file_paths()

        try:
            slices = [np.array(Image.open(p)) for p in file_paths]
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load one of the slice images:\n{str(e)}")
            return
        
        shapes = {s.shape for s in slices}
        if len(shapes) > 1:
            QMessageBox.critical(self, "Mismatched Slice Sizes", f"Not all slices have the same shape - found: {shapes}")
            return
        
        image_data = np.stack(slices, axis=0)
        self.current_file_path = dialog.folder_path
        self._finalize_loaded_image(image_data, os.path.basename(dialog.folder_path), data_mode='3d_volume')


    
    def open_binarize_dialog(self):
        """Open the binarize dialog and apply the chosen foreground value to the current image.
        (non-modal, so the user can still hover the image to check pixel values)
        """

        if self.current_image_data is None:
            QMessageBox.warning(self, "No Image", "Please open an image first.")
            return
        
        unique_values = np.unique(self.current_image_data)
        if len(unique_values) <= 1:
            QMessageBox.information(self, "Nothing to Binarize", "This image already has only one pixel value.")
            return
        
        # Keep a reference on self - otherwise Python would garbage-collect the dialog almost
        # immediately, since show() returns right away instead of blocking like exec() did.
        self.binarize_dialog = BinarizeDialog(unique_values, parent=self)
        self.binarize_dialog.accepted.connect(self._apply_binarization)
        self.binarize_dialog.show()

    def _apply_binarization(self):
        """Called when the (non-modal) binarize dialog is accepted - applies the chosen foreground value."""
        foreground_value = self.binarize_dialog.get_foreground_value()

        # Replace the current image with its binarized version: chosen value -> 1, everything else -> 0
        self.current_image_data = np.where(self.current_image_data == foreground_value, 1, 0).astype(np.uint8)

        # Re-display (same slice index as before, now showing the binarized result)
        self.display_current_slice()

        self.status_bar.showMessage(f"Binarized: pixel value {foreground_value} set as foreground")
        QMessageBox.information(
            self,
            "Binarization Complete",
            f"Pixel value {foreground_value} is now the foreground (1); everything else is background (0)."
        )
        
        # foreground_value = dialog.get_foreground_value()

        # # Replace the current image with its binarized version: chosen value -> 1, everything else -> 0
        # self.current_image_data =  np.where(self.current_image_data==foreground_value, 1, 0).astype(np.uint8)
        # # Re-display (same slice index as before, now showing the binarized result)
        # self.display_current_slice()

        # self.status_bar.showMessage(f"Binarized: pixel value {foreground_value} set as foreground")
        # QMessageBox.information(
        # self,
        # "Binarization Complete",
        # f"Pixel value {foreground_value} is now the foreground (1); everything else is background (0)."
        # )


    def open_smd_dialog(self):
        """Open the SMD selection dialog (real computation wired up in a later step)."""

        if self.current_image_data is None:
            QMessageBox.warning(self, "No Image", "Please open an image first.")
            return
        
        if self.current_image_data.ndim not in (2, 3):
            QMessageBox.warning(self, "Invalid Image", "Image must be 2D or 3D.")
            return

        is_3d = self.current_image_data.ndim == 3
        dialog = PolytopeSettingsDialog(is_3d=is_3d, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return  # User cancelled
        
        selected = dialog.get_selected_polytopes()
        # QMessageBox.information(self, "Selected Polytopes", f"You selected: {selected}")

        # heads-up about runtime when c2 is selected, since it can be slow for large images

        if 'c2' in selected and self.current_image_data.ndim == 3:
            reply = QMessageBox.question(
                self,
                "Slow calculation",
                "Calculating C2 on a full 3D volume of large size (>= 512^3) can take a few minutes, "
                "longer the first time, due to a one-time compilation step. Do you want to continue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return  # User chose not to continue



        # Show progress bar in status bar
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage(f"Calculating polytopes: {selected}...")

        # Process events to ensure UI updates
        QApplication.processEvents()

        # Create and start polytope calculation thread
        self.polytope_thread = PolytopeCalculationThread(self.current_image_data, selected)
        self.polytope_thread.finished.connect(self.on_polytope_calculation_finished)
        self.polytope_thread.error.connect(self.on_polytope_calculation_error)
        self.polytope_thread.start()

    def on_polytope_calculation_finished(self, raw_curves: dict, scaled_curves: dict):
        """Handle completion of polytope calculation.
        (Real plot window comes in the next step - for now just confirm it worked.)
        Args:
            raw_curves: Dictionary of raw curve data for each selected polytope
            scaled_curves: Dictionary of scaled curve data for each selected polytope
        """
        self.progress_bar.setVisible(False)
        

        # summary = "\n".join(f"{name}: {values.shape[0]} points" for name, values in raw_curves.items())
        # QMessageBox.information(self, "Polytope Calculation Done", f"Computed:\n{summary}")

        # Create a new window to display the polytope plots
        polytope_window = PolytopePlotWindow(raw_curves, scaled_curves, self)
        self.plot_windows.append(polytope_window)
        polytope_window.show()

        self.status_bar.showMessage("Polytope calculation completed")

    def on_polytope_calculation_error(self, error_msg: str):
        """Handle error during polytope calculation."""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Calculation Error", f"Error calculating polytopes:\n{error_msg}")
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
