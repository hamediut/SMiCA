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
from .save_dialog_helper import suggested_open_dir, remember_open_dir
from .rev_plot_window import RevPlotWindow
from ..analysis.smds import (
    RES, REV,
     calculate_s2, calculate_s2_periodic , calculate_s2_3d,
     calculate_polytopes_python, calculate_c2,
     cal_fn, scale_polytope_fn, scale_by_initial_value, scale_c2_by_connectedness,
     calculate_c2_3d, calculate_s2_periodic_3d,
     calculate_L_3d,
     omega_n, delta_omega,
     compute_2d_polytope_curves, compute_3d_polytope_curves
)

from .rev_settings_dialog import REVSettingsDialog
from .polytope_settings_dialog import PolytopeSettingsDialog
from .polytope_plot_window import PolytopePlotWindow
from .binarize_dialog import BinarizeDialog

from .import_sequence_dialog import ImportSequenceDialog

from .slice_evolution_settings_dialog import SliceEvolutionSettingsDialog
from .slice_evolution_plot_window import SliceEvolutionPlotWindow


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

            if self.image_data.ndim ==3:
                # 3D volume: s2/c2/L, computed by the same helper the 4D slice/time
                # evolution thread will use per time step (see compute_3d_polytope_curves
                # in smds.py) - one shared place for the 3D branching logic instead of two.
                raw_curves, scaled_curves = compute_3d_polytope_curves(self.image_data, self.selected_polytopes)

            else:
                raw_curves = {}
                scaled_curves = {}

                # S2 uses the same (non-periodic) code as "Calculate SMDS", so results
                # are consistent between the two menu items for the same image.

                if 's2' in self.selected_polytopes:
                    s2_values = calculate_s2(self.image_data)
                    f2_values = cal_fn(s2_values, n = 2)
                    r_axis =  np.arange(len(s2_values), dtype=np.float64)
                    raw_curves['s2'] = np.column_stack((r_axis, s2_values))
                    scaled_curves['s2'] = np.column_stack((r_axis, f2_values))

                if 'c2' in self.selected_polytopes:
                    c2_values = calculate_c2(self.image_data)
                    s2_periodic_values = calculate_s2_periodic(self.image_data)  # S2 with the SAME periodic convention as C2 (needed as the denominator)
                    # --- current approach: conditional connectedness C2(r)/S2(r) ---
                    f2_c2_values = scale_c2_by_connectedness(c2_values, s2_periodic_values)  # P(same cluster | same phase) at r - bounded [0,1] at every r, not just r=0
                    r_axis = np.arange(len(c2_values), dtype=np.float64)
                    raw_curves['c2'] = np.column_stack((r_axis, c2_values))
                    scaled_curves['c2'] = np.column_stack((r_axis, f2_c2_values))
            
                # 2D 'L' still goes through calculate_polytopes_python below, same as before - only 3D 'L'
                # gets the dedicated branch above, since the 2D path is already validated against the C++.
                other_polytopes = [name for name in self.selected_polytopes if name not in ('s2', 'c2')]
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


class SliceEvolutionThread(QThread):
    """
    Computes 2D polytope functions independently on every slice/time-step of a 3D stack,
    then derives the Omega/Delta-Omega evolution metrics from those per-slice curves.
    """
    # Signal carries everything the plot window needs: which slice/time indices were used,
    # the raw+scaled curves for each of them, and the two Omega-family metric dicts.
    finished = Signal(list, list, list, dict, dict)
    error = Signal(str)

    def __init__(self, image_data: np.ndarray, selected_polytopes: List[str], step: int,
                 reference_index: int = 0, compute_omega: bool = True,
                 compute_delta_omega: bool = True, signed_delta_omega: bool = False,
                 stack_labels=None):
        super().__init__()
        self.image_data = image_data
        self.selected_polytopes = selected_polytopes
        self.step = step
        self.reference_index = reference_index
        self.compute_omega = compute_omega
        self.compute_delta_omega = compute_delta_omega
        self.signed_delta_omega = signed_delta_omega
        # Real step/time numbers from the imported filenames (None if this data has none,
        # e.g. a native 3D volume) - see run() below for how these get used.
        self.stack_labels = stack_labels

    def run(self):
        try:
            n_slices = self.image_data.shape[0]

            # array_positions: real indices into self.image_data - ALWAYS 0, 1, 2, ...
            # regardless of what the filenames said, since that's what numpy indexing needs.
            # e.g. step=1 -> every slice; step=5 -> position 0, 5, 10, 15, ...
            array_positions = list(range(0, n_slices, self.step))

            # slice_indices: what actually gets plotted/exported later. Real filename-derived
            # numbers when we have them (self.stack_labels), otherwise just fall back to the
            # array position itself (e.g. a native 3D volume with no associated filenames).
            if self.stack_labels is not None:
                slice_indices = [self.stack_labels[pos] for pos in array_positions]
            else:
                slice_indices = array_positions

            # Compute the curves for every slice we're keeping, one call per slice - always
            # indexed by array POSITION, never by the display numbers above. Each "slice" is
            # either a 2D image (2D time series) or a 3D volume (4D data, self.image_data.ndim
            # == 4) - pick the matching curve function once, rather than re-checking per slice.
            
            compute_curves = compute_3d_polytope_curves if self.image_data.ndim ==4 else compute_2d_polytope_curves
            
            raw_curves_list = []
            scaled_curves_list = []
            for pos in array_positions:
                raw, scaled = compute_curves(self.image_data[pos], self.selected_polytopes)
                raw_curves_list.append(raw)
                scaled_curves_list.append(scaled)

            # reference_index (from the dialog) is an array POSITION (see
            # SliceEvolutionSettingsDialog) - find where that position sits within
            # array_positions to get the LIST position omega_n/delta_omega expect.
            reference_position = array_positions.index(self.reference_index)

            # For each selected function, pull out just that function's curve from every
            # slice (curve_series), then hand that list to omega_n()/delta_omega().
            omega_dict = {}
            delta_omega_dict = {}
            for name in self.selected_polytopes:
                curve_series = [curves[name] for curves in raw_curves_list]
                # Drop the odd-r placeholder zeros for P3H/P3V before computing Omega, so the
                # normalization (n_r inside omega_n/delta_omega) reflects the actual number of
                # meaningful points, not double-counting the always-zero entries.

                if name in ('p3h', 'p3v'):
                    curve_series = [curve[::2] for curve in curve_series]

                if self.compute_omega:
                    omega_dict[name] = omega_n(curve_series, reference_index=reference_position)
                if self.compute_delta_omega:
                    delta_omega_dict[name] = delta_omega(curve_series, signed=self.signed_delta_omega)
            # These will be calculated and sent to the plotwindow later
            self.finished.emit(slice_indices, raw_curves_list, scaled_curves_list, omega_dict, delta_omega_dict)
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
        self.plot_windows: List[QMainWindow] = []
        self.current_image_data: Optional[np.ndarray] = None
        self.data_mode: Optional[str] = None
        # Real step/time numbers per slice, from a folder import's filenames (e.g. [0, 5, 12, 20]).
        # None for anything without associated filenames (opened tif, native 3D volume).
        self.stack_labels: Optional[list] = None
        self.current_file_path: Optional[str] = None
        self.current_slice_index: int = 0
        self.current_time_index: int = 0 # which 3D volume (T axis) is showing, 4D data only


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

        # Create slider for the Time axis (4D data only, initially hidden)
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.setValue(0)
        self.time_slider.valueChanged.connect(self.on_time_changed)
        self.time_slider.setVisible(False)
        self.layout.addWidget(self.time_slider)

        self.time_label = QLabel("Time: 0 / 0")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setVisible(False)
        self.layout.addWidget(self.time_label)



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

        import_timeseries_action = file_menu.addAction("Import &Time Series from Folder...")
        import_timeseries_action.setStatusTip("Import a time series of 2D images from a folder")
        import_timeseries_action.triggered.connect(self.open_import_time_series_dialog)

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

    def numpy_to_qpixmap(self, image_data: np.ndarray, data_min=None, data_max=None) -> QPixmap:
        """
        Convert numpy array to QPixmap for display.
        Uses min-max normalization instead of assuming binary 0/1 values, so this works both
        for binary images (0 -> 0, 1 -> 255, same as before) and multi-label segmentations
        (e.g. 0,1,2,3 -> 0,85,170,255) without overflowing.

        Args:
            image_data: 2D numpy array (binary or multi-label integer values) to display
            data_min, data_max: the value range to normalize against. display_current_slice()
                passes the WHOLE volume's min/max here, not just this slice's - so a slice
                that happens to be uniform (e.g. an all-background slice near the end of a
                segmented stack) still renders correctly (0 -> black) instead of falling into
                the flat mid-grey fallback below, which should only trigger when the ENTIRE
                volume is uniform (nothing to normalize against at all). Defaults to this
                slice's own min/max if not given, for any other caller.

        Returns:
            QPixmap for display
        """

        if data_min is None:
            data_min = image_data.min()
        if data_max is None:
            data_max = image_data.max()

        if data_max > data_min:
            # Normalize to [0, 255]
            normalized = ((image_data.astype(np.float64) - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        else:
            # every pixel in the whole volume has the same value - avoid a divide-by-zero, show flat mid-grey
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
        if self.current_image_data.ndim == 4:
            current_2d = self.current_image_data[self.current_time_index, self.current_slice_index, :, :]
        elif self.current_image_data.ndim == 3:
            current_2d = self.current_image_data[self.current_slice_index, :, :]
        else:
            current_2d = self.current_image_data

        # Update the image label data for hover
        self.image_label.set_image_data(current_2d)

        # Convert to pixmap and display - pass the WHOLE volume's min/max (not just this
        # slice's) so a uniform slice (e.g. all-background) still renders correctly instead
        # of falling back to flat grey (see numpy_to_qpixmap's docstring).
        self.current_pixmap = self.numpy_to_qpixmap(current_2d, self._display_min, self._display_max)
        scaled_pixmap = self.current_pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def on_slice_changed(self, value: int):
        """
        Handle Z-slice slider change. Works for both a plain 3D volume (Z, Y, X) and a 4D
        stack (T, Z, Y, X) - shape[-3] is the Z-axis length either way.
        """
        self.current_slice_index = value
        if self.current_image_data is not None and self.current_image_data.ndim in (3, 4):
            self.slice_label.setText(f"Slice: {value} / {self.current_image_data.shape[-3] - 1}")
            self.display_current_slice()

    def on_time_changed(self, value: int):
        """Handle Time slider change (4D data only)."""

        self.current_time_index = value
        if self.current_image_data is not None and self.current_image_data.ndim == 4:
            self.time_label.setText(f"Time: {value} / {self.current_image_data.shape[0] - 1}")
            self.display_current_slice()

    def open_image(self):
        """Open a file dialog to select a TIF image and display it."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open TIF Image",
            suggested_open_dir(),  # opens in the last-used import/open folder
            "TIF Files (*.tif *.tiff);;All Files (*)"
        )

        if file_path:
            remember_open_dir(file_path)  # so the next Open/Import dialog starts in this same folder
            try:
                image_data = self._load_multipage_tif(file_path)

                # # Note whether this image needs binarizing before it can be used in calculations -
                # doesn't block loading, just informs the status message below.

                is_valid, unique_values = self.validate_binary_image(image_data)
                
                self.current_file_path = file_path
                self._finalize_loaded_image(image_data, os.path.basename(file_path), data_mode='3d_volume' if image_data.ndim == 3 else '2d')

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")
                self.status_bar.showMessage(f"Error: {str(e)}")

    def _finalize_loaded_image(self, image_data: np.ndarray, source_label: str, data_mode: str, stack_labels=None):

        """
        Shared tail-end of loading any image (single file or imported sequence): validates
        binary-ness, configures the slice slider, displays the first slice, and updates the
        status bar/title.

        stack_labels defaults to None, which RESETS it for every load path that doesn't pass
        one explicitly (open_image, open_import_volume_dialog) - only
        open_import_time_series_dialog currently passes real values, so a previously-loaded
        time series's numbers don't linger and get wrongly reused for whatever gets loaded next.
        """

        is_valid, unique_values = self.validate_binary_image(image_data)

        self.current_image_data = image_data
        self.current_slice_index = 0
        self.current_time_index = 0
        self.data_mode = data_mode
        self.stack_labels = stack_labels

        # Computed ONCE for the whole volume (not per-slice) so display_current_slice() can
        # normalize every slice against the same range - see numpy_to_qpixmap's docstring.
        self._display_min = float(image_data.min())
        self._display_max = float(image_data.max())

        if image_data.ndim == 4: 
            n_time, n_slices = image_data.shape[0], image_data.shape[1]

            self.time_slider.setMaximum(n_time - 1)
            self.time_slider.setValue(0)
            self.time_slider.setVisible(True)
            self.time_label.setText(f"Time: 0 / {n_time - 1}")
            self.time_label.setVisible(True)

            self.slice_slider.setMaximum(n_slices - 1)
            self.slice_slider.setValue(0)
            self.slice_slider.setVisible(True)
            self.slice_label.setText(f"Slice: 0 / {n_slices - 1}")
            self.slice_label.setVisible(True)

            dims = f"{image_data.shape[3]} x {image_data.shape[2]} x {n_slices} x {n_time} (X x Y x Z x T)"

        elif image_data.ndim == 3:
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


    def _load_and_stack_2d_files(self, file_paths):

        """Load a list of 2D image files and stack them into a single (N, H, W) array."""
        slices = [np.array(Image.open(p)) for p in file_paths]
        shapes = {s.shape for s in slices}
        if len(shapes) > 1:
            raise ValueError(f"Not all files have the same shape - found: {shapes}")
        return np.stack(slices, axis=0)

    def _load_and_stack_3d_files(self, file_paths):
        """Load a list of multi-page TIF files (each a 3D (Z, Y, X) volume) and stack them
        into a single 4D (T, Z, Y, X) array - one volume per time step."""
        volumes = [self._load_multipage_tif(p) for p in file_paths]

        not_3d = [v.shape for v in volumes if v.ndim != 3]
        if not_3d:
            raise ValueError(f"Expected every file to be a multi-page (3D) TIF, but found a single-page file: shape {not_3d[0]}")

        shapes = {v.shape for v in volumes}
        if len(shapes) > 1:
            raise ValueError(f"Not all volumes have the same shape - found: {shapes}")

        return np.stack(volumes, axis=0) 

    def open_import_volume_dialog(self):

        """Import a 3D volume assembled from a folder of 2D Z-slice files."""

        dialog =  ImportSequenceDialog(ask_file_type = False, parent = self)
        if dialog.exec() != QDialog.Accepted:
            return
        
        # file_paths =  dialog.get_sorted_file_paths()

        try:
            image_data = self._load_and_stack_2d_files(dialog.get_sorted_file_paths())
            # slices = [np.array(Image.open(p)) for p in file_paths]
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load slice images:\n{str(e)}")
            return
        
        # shapes = {s.shape for s in slices}
        # if len(shapes) > 1:
        #     QMessageBox.critical(self, "Mismatched Slice Sizes", f"Not all slices have the same shape - found: {shapes}")
        #     return
        
        # image_data = np.stack(slices, axis=0)
        self.current_file_path = dialog.folder_path
        self._finalize_loaded_image(image_data, os.path.basename(dialog.folder_path), data_mode='3d_volume')


    def _load_multipage_tif(self, file_path):
        """
        Load one TIF file. Returns a 2D (Y, X) array if it's a single-page file, or a 3D
        (Z, Y, X) array if it's a multi-page file (e.g. an XCT volume saved as one stacked
        TIF). Shared by open_image() (a single file) and _load_and_stack_3d_files() (many
        such files, one per time step).
        """

        pil_image =  Image.open(file_path)
        frames = []
        try:
            while True:
                frames.append(np.array(pil_image))
                pil_image.seek(pil_image.tell() + 1)
        except EOFError:
            pass # end of frames
        return np.stack(frames, axis = 0) if len(frames) > 1 else frames[0]



    def open_import_time_series_dialog(self):

        """Import a time series (2D slices, or full 3D volumes) of images from a folder."""

        dialog = ImportSequenceDialog(ask_file_type= True, parent = self)

        if dialog.exec() != QDialog.Accepted:
            return
        
        try:
            if dialog.get_is_3d_files():
                image_data =  self._load_and_stack_3d_files(dialog.get_sorted_file_paths())
                data_mode = '4d_time_series'
            else:
                image_data = self._load_and_stack_2d_files(dialog.get_sorted_file_paths())
                data_mode = 'time_series'


        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load time-step images:\n{str(e)}")
            return
        
        self.current_file_path = dialog.folder_path
        # dialog.get_sorted_values() = the real numbers extracted from each filename, in the
        # same order as the stack we just built - this is what makes the colorbar/x-axis in
        # the evolution plots show real step/time numbers instead of plain 0,1,2,3...
        self._finalize_loaded_image(
            image_data, os.path.basename(dialog.folder_path), data_mode = data_mode,
            stack_labels=dialog.get_sorted_values(),
        )

    
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

        # Value range changed (e.g. a multi-label 0-4 image is now just 0-1) - refresh the
        # stored display bounds so numpy_to_qpixmap normalizes against the new range.
        self._display_min = float(self.current_image_data.min())
        self._display_max = float(self.current_image_data.max())

        # Re-display (same slice index as before, now showing the binarized result)
        self.display_current_slice()

        self.status_bar.showMessage(f"Binarized: pixel value {foreground_value} set as foreground")
        QMessageBox.information(
            self,
            "Binarization Complete",
            f"Pixel value {foreground_value} is now the foreground (1); everything else is background (0)."
        )
        

    def open_smd_dialog(self):
        """Open the SMD selection dialog (real computation wired up in a later step)."""

        if self.current_image_data is None:
            QMessageBox.warning(self, "No Image", "Please open an image first.")
            return
        
        if self.current_image_data.ndim not in (2, 3, 4):
            QMessageBox.warning(self, "Invalid Image", "Image must be 2D, 3D, or 4D.")
            return
        
        if self.data_mode in ('time_series', '4d_time_series'):
            self.open_slice_evolution_dialog()
            return
        
        if self.current_image_data.ndim ==3 and self.data_mode == 'time_series':
            QMessageBox.warning(
                self, "Not Applicable",
                "This is a time series, not a spatial 3D volume - the 3D correlation functions "
                "don't apply across time. Time-series analysis tools are coming in a later step."
            )
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


    def open_slice_evolution_dialog(self):
        """Configure and run per-slice/per-time-step 2D correlation + Omega analysis on the current 3D stack."""

        if self.current_image_data is None:
            QMessageBox.warning(self, "No Image", "Please open an image first.")
            return
        if self.current_image_data.ndim not in (3, 4):
            QMessageBox.warning(self, "Invalid Image", "This requires a 3D volume or an imported volume time series.")
            return
        
        # Just controls wording in the dialog/plot titles - "Slice" for a real 3D volume,
        # "Time step" for an imported time series (2D or 3D-per-step). The underlying calculation is identical either way.

        axis_label = "time step" if self.data_mode in ('time_series', '4d_time_series') else "slice"
        n_slices = self.current_image_data.shape[0]

        is_3d = self.current_image_data.ndim ==4 # Each time step is itself a 3D volume

        dialog = SliceEvolutionSettingsDialog(n_slices, axis_label=axis_label, stack_labels=self.stack_labels, is_3d = is_3d, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return  # user cancelled
        
         # Show progress bar in status bar, same pattern as the other calculations
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage(f"Calculating {axis_label} evolution...")
        QApplication.processEvents()

         # Stash the axis label on self so on_slice_evolution_finished (below) can use it too -
        # it only gets the thread's signal arguments, not the dialog itself

        self._evolution_axis_label = axis_label

        self.evolution_thread = SliceEvolutionThread(
            self.current_image_data, dialog.get_selected_polytopes(), dialog.get_step(),
            reference_index=dialog.get_reference_index(),
            compute_omega=dialog.get_compute_omega(),
            compute_delta_omega=dialog.get_compute_delta_omega(),
            signed_delta_omega=dialog.get_signed_delta_omega(),
            stack_labels=self.stack_labels,
        )
        self.evolution_thread.finished.connect(self.on_slice_evolution_finished)
        self.evolution_thread.error.connect(self.on_slice_evolution_error)
        self.evolution_thread.start()

    def on_slice_evolution_finished(self, slice_indices, raw_curves_list, scaled_curves_list, omega_dict, delta_omega_dict):

        """Handle completion of the slice/time evolution calculation - open the results window."""
        self.progress_bar.setVisible(False)

        window = SliceEvolutionPlotWindow(
            slice_indices, raw_curves_list, scaled_curves_list, omega_dict, delta_omega_dict,
            axis_label=self._evolution_axis_label, parent=self
        )
        self.plot_windows.append(window)
        window.show()

        self.status_bar.showMessage(f"{self._evolution_axis_label.title()} evolution calculation completed")

    def on_slice_evolution_error(self, error_msg: str):
        """Handle error during the slice/time evolution calculation."""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Calculation Error", f"Error calculating evolution:\n{error_msg}")
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
        
        if self.current_image_data.ndim == 3 and self.data_mode == 'time_series':
            QMessageBox.warning(
                self, "Not Applicable",
                "This is a time series, not a spatial 3D volume - the 3D correlation functions "
                "don't apply across time. Time-series analysis tools are coming in a later step."
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
