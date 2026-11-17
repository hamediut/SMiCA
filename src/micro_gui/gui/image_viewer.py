"""
Main image viewer window for the Micro_GUI application.
"""

import os
import time
import weakref
import numpy as np
from typing import Optional, List

from PySide6.QtWidgets import (
    QMainWindow, QLabel, QFileDialog, QScrollArea, QStatusBar, QDialog,
    QSlider, QVBoxLayout, QWidget, QMessageBox, QProgressBar, QApplication, QCheckBox
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QThread, Signal

from PIL import Image

from .widgets import ImageDisplayWidget
from .save_dialog_helper import (
    suggested_open_dir, remember_open_dir, suggested_save_path, suggested_save_dir, remember_save_dir
)
from .rev_plot_window import RevPlotWindow
from ..analysis.smds import (
    RES, REV,
     omega_n, delta_omega,
     compute_2d_polytope_curves, compute_3d_polytope_curves,
     compute_chord_length_result
)

from .rev_settings_dialog import REVSettingsDialog
from .polytope_settings_dialog import PolytopeSettingsDialog
from .polytope_plot_window import PolytopePlotWindow
from .binarize_dialog import BinarizeDialog

from .import_sequence_dialog import ImportSequenceDialog

from .slice_evolution_settings_dialog import SliceEvolutionSettingsDialog
from .slice_evolution_plot_window import SliceEvolutionPlotWindow

from .chord_length_settings_dialog import ChordLengthSettingsDialog
from .chord_length_plot_window import ChordLengthPlotWindow

from .chord_length_evolution_settings_dialog import ChordLengthEvolutionSettingsDialog
from .chord_length_evolution_plot_window import ChordLengthEvolutionPlotWindow

from .minkowski_settings_dialog import MinkowskiSettingsDialog
from ..analysis.minkowski import minkowski_2d, minkowski_3d
from .minkowski_results_dialog import MinkowskiResultsDialog
from .minkowski_evolution_settings_dialog import MinkowskiEvolutionSettingsDialog
from .minkowski_evolution_plot_window import MinkowskiEvolutionPlotWindow

from ..utils.image_utils import (
    load_multipage_tif, long_path_safe, default_slice_names, save_tiff, save_tiff_sequence
)

#connected components
from ..analysis.connected_components import (
    connected_components_2d, connected_components_3d, connected_components_2d_stack,
    compute_shape_measurements, compute_advanced_shape_measurements, ADVANCED_MEASUREMENTS,
)

from .connected_components_results_dialog import ConnectedComponentsResultsDialog
from .measurements_settings_dialog import MeasurementsSettingsDialog
from .connected_components_settings_dialog import ConnectedComponentsSettingsDialog

# cropping
from .crop_center_dialog import CropCenterDialog
from ..processing.cropping import CropBox, crop_image, exterior_value_candidates, find_center_crop_box

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

    The calculation logic itself lives in smds.py: compute_2d_polytope_curves() for a 2D
    image, compute_3d_polytope_curves() for a 3D volume. The slice/time evolution threads
    call the same two helpers once per slice, so a single image and one frame of a time
    series are always computed identically.
    """


    finished = Signal(object, object)  # Emits (raw_curves, scaled_curves)
    error = Signal(str)

    def __init__(self, image_data: np.ndarray, selected_polytopes: List[str]):
        """
        Initialize the polytope calculation thread.

        Args:
            image_data: The 2D image or 3D volume to process
            selected_polytopes: List of selected polytope function names
        """
        super().__init__()
        self.image_data = image_data
        self.selected_polytopes = selected_polytopes

    def run(self):
        """Run the selected polytope calculations in a separate thread."""
        
        try:
            # Choose the helper once from the dimensionality - same pattern as
            # SliceEvolutionThread.run() uses per slice.

            compute_curves = (compute_3d_polytope_curves if self.image_data.ndim == 3
                              else compute_2d_polytope_curves)

            raw_curves, scaled_curves = compute_curves(self.image_data, self.selected_polytopes)
            self.finished.emit(raw_curves, scaled_curves)

               
        except Exception as e:
            self.error.emit(str(e))  # the thread can't show a dialog itself - hand the message to the GUI thread



##-----------------------------------------------------

class ChordLengthThread(QThread):
    """
    Thread for running Chord Length calculation in the background (keeps the GUI
    responsive - same reason every other calculation runs in a QThread here).
    """

    finished = Signal(dict)  # emits the single result dict from compute_chord_length_result
    error = Signal(str)      # emits a message if something goes wrong, instead of crashing the thread

    def __init__(self, image_data, num_points, nbins):
        super().__init__()
        self.image_data = image_data
        self.num_points = num_points # from the settings dialog - S2 slope fit window
        self.nbins = nbins           # from the settings dialog - histogram bin count

    def run(self):

        try:
            result = compute_chord_length_result(
                self.image_data,
                phase = 1,
                num_points = self.num_points,
                nbins = self.nbins
            )

            self.finished.emit(result) # success: hand the dict back to the GUI thread
        except Exception as e:
            self.error.emit(str(e))

class ChordLengthEvolutionThread(QThread):
    """
    Computes Chord Length independently on every slice/time-step of a stack (2D
    time-series of slices, or a time-series of 3D volumes).
    """
    # Just two things, unlike SliceEvolutionThread's 5 - no omega/delta-omega here,
    # and no separate raw/scaled split (compute_chord_length_result returns one dict).

    finished = Signal(list, list) # (slice_indices, results_list)
    error = Signal(str)
    progress = Signal(int, int)  # (completed, total)

    def __init__(self, image_data, step: int, num_points: int, nbins: int,
                 stack_labels = None, reverse: bool = False):

        super().__init__()
        self.image_data = image_data
        self.step = step
        self.num_points = num_points
        self.nbins = nbins
        self.stack_labels = stack_labels
        self.reverse = reverse

    def run(self):

        try:
            n_slices = self.image_data.shape[0]

            # Identical formula to SliceEvolutionThread.run() - which array positions
            # get included, and in what order, from step + direction.
            if self.reverse:
                array_positions = list(range(n_slices - 1, -1, -self.step))
            else:
                array_positions = list(range(0, n_slices, self.step))

            # Real filename-derived numbers when we have them, otherwise the array
            # position itself - same as SliceEvolutionThread.
            if self.stack_labels is not None:
                slice_indices = [self.stack_labels[pos] for pos in array_positions]
            else:
                slice_indices = array_positions

            # One compute_chord_length_result() call per kept slice. self.image_data[pos]
            # is a 2D image (2D time-series) or a 3D volume (4D time-series of volumes) -
            # compute_chord_length_result already branches on .ndim internally, so unlike
            # SliceEvolutionThread we don't need to pick a function based on self.image_data.ndim.

            total = len(array_positions)
            results_list = []
            for i, pos in enumerate(array_positions):
                result = compute_chord_length_result(
                    self.image_data[pos],
                    phase = 1,
                    num_points= self.num_points,
                    nbins= self.nbins,
                )
                results_list.append(result)
                self.progress.emit(i + 1, total)

            self.finished.emit(slice_indices, results_list)

        except Exception as e:
            self.error.emit(str(e))

class ChordLengthEvolutionStreamingThread(QThread):
    """
    Streaming counterpart to ChordLengthEvolutionThread - loads, computes, and discards
    ONE volume at a time from a list of file paths, instead of requiring the whole 4D
    stack to already be loaded in memory.
    """

    finished = Signal(list, list)
    error = Signal(str)
    progress = Signal(int, int)  # (completed, total)

    def __init__(self, file_paths, step: int, num_points: int, nbins: int,
                 stack_labels=None, reverse: bool = False):
        super().__init__()
        self.file_paths = file_paths
        self.step = step
        self.num_points = num_points
        self.nbins = nbins
        self.stack_labels = stack_labels
        self.reverse = reverse

    def run(self):
        try:
            n_files = len(self.file_paths)

            if self.reverse:
                array_positions = list(range(n_files - 1, -1, -self.step))
            else:
                array_positions = list(range(0, n_files, self.step))

            if self.stack_labels is not None:
                slice_indices = [self.stack_labels[pos] for pos in array_positions]
            else:
                slice_indices = array_positions

            total = len(array_positions)
            results_list = []
            for i, pos in enumerate(array_positions):
                volume = load_multipage_tif(self.file_paths[pos])
                result = compute_chord_length_result(
                    volume, phase=1, num_points=self.num_points, nbins=self.nbins,
                )
                results_list.append(result)
                self.progress.emit(i + 1, total)

            self.finished.emit(slice_indices, results_list)
        except Exception as e:
            self.error.emit(str(e))


class MinkowskiEvolutionThread(QThread):

    """
    Computes Minkowski functionals independently on every slice/time-step of a
    stack. Mirrors ChordLengthEvolutionThread's structure - the only real
    difference is picking minkowski_2d vs minkowski_3d once, based on whether
    each entry in the stack is a 2D slice or a 3D volume.
    """

    finished = Signal(list, list) # (slice_indices, results_list)
    error = Signal(str)
    progress = Signal(int, int)  # (completed, total)

    def __init__(self, image_data, step: int, resolution: float, is_3d: bool,
                 stack_labels=None, reverse: bool = False):
        super().__init__()
        self.image_data = image_data
        self.step = step
        self.resolution = resolution
        self.is_3d = is_3d
        self.stack_labels = stack_labels
        self.reverse = reverse

    def run(self):
            
        try:

            n_slices = self.image_data.shape[0]

            if self.reverse:
                array_positions = list(range(n_slices - 1, -1, -self.step))
            else:
                array_positions = list(range(0, n_slices, self.step))

            if self.stack_labels is not None:
                slice_indices = [self.stack_labels[pos] for pos in array_positions]
            else:
                slice_indices = array_positions

            compute = minkowski_3d if self.is_3d else minkowski_2d

            total = len(array_positions)
            results_list = []
            for i, pos in enumerate(array_positions):
                results_list.append(compute(self.image_data[pos], res=self.resolution))
                self.progress.emit(i + 1, total)

            

            self.finished.emit(slice_indices, results_list)


        except Exception as e:
            self.error.emit(str(e))


class MinkowskiEvolutionStreamingThread(QThread):
    """
    Streaming counterpart to MinkowskiEvolutionThread - for a 4D time series that
    was imported as a list of file paths (one 3D volume per time step) instead of
    eagerly loaded and stacked into one big in-memory array. Loads, computes, and
    discards ONE volume at a time, so peak memory is one volume's worth regardless
    of how many time steps there are - this is what makes datasets too large to
    fit in RAM as a single 4D array tractable at all.

    Always 3D (minkowski_3d) - streaming mode exists specifically for "many large
    3D volumes", where a 2D time-series wouldn't hit the same memory ceiling.
    """

    finished = Signal(list, list)  # (slice_indices, results_list)
    error = Signal(str)
    progress = Signal(int, int)

    def __init__(self, file_paths, step: int, resolution: float,
                 stack_labels=None, reverse: bool = False):
        super().__init__()
        self.file_paths = file_paths
        self.step = step
        self.resolution = resolution
        self.stack_labels = stack_labels
        self.reverse = reverse
    
    def run(self):
        try:
            n_files = len(self.file_paths)

            if self.reverse:
                array_positions = list(range(n_files - 1, -1, -self.step))
            else:
                array_positions = list(range(0, n_files, self.step))

            if self.stack_labels is not None:
                slice_indices = [self.stack_labels[pos] for pos in array_positions]
            else:
                slice_indices = array_positions

            total = len(array_positions)
            results_list = []
            for i, pos in enumerate(array_positions):
                # one volume in memory at a time - overwritten (and garbage
                # collected) the moment the next iteration loads a new one
                volume = load_multipage_tif(self.file_paths[pos])
                results_list.append(minkowski_3d(volume, res=self.resolution))
                self.progress.emit(i + 1, total)

            self.finished.emit(slice_indices, results_list)

        except Exception as e:
            self.error.emit(str(e))

class ConnectedComponentsThread(QThread):
    """
    Thread for running connected-components labeling in the background -
    label() on a full-size volume takes a few seconds, long enough to
    freeze the UI if run directly on the GUI thread.

    Runs with res=1.0 (raw voxel/pixel units) - real voxel size isn't known
    yet at this step. It's asked for later in MeasurementsSettingsDialog and
    applied by MeasurementsThread once labeling is done.
    """
    finished = Signal(object) # Emits the dict connected_components_2d/_3d returns
    error = Signal(str)

    def __init__(self, image_data:np.ndarray, connectivity:int, is_3d: bool, per_frame: bool = False):
        super().__init__()

        self.image_data = image_data
        self.connectivity = connectivity
        self.is_3d = is_3d
        self.per_frame = per_frame # 2D time series: label each frame on its own

    def run(self):
        try:
            if self.per_frame:
                compute  = connected_components_2d_stack
            else:
                compute = connected_components_3d if self.is_3d else connected_components_2d

            result = compute(self.image_data, connectivity=self.connectivity, res= 1.0)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class MeasurementsThread(QThread):
    """
    Thread for running the measurement step on an already-labeled connected-
    components result: rescales the raw count into physical volume/area now
    that the real voxel size is known, and computes the requested shape
    measurements. regionprops per component is fast, but still worth keeping
    off the GUI thread for large component counts.
    """

    finished = Signal(object) # Emits the merged, rescaled pandas DataFrame
    error = Signal(str)

    def __init__(self, cc_result: dict, is_3d: bool, resolution: float, selected_measurements: List, min_size: int = 0):
        super().__init__()
        self.cc_result = cc_result
        self.is_3d = is_3d
        self.resolution = resolution
        self.selected_measurements = selected_measurements
        self.min_size = min_size

    def run(self):
        try:
            table = self.cc_result['table'].copy()
            count_col = 'voxel_count' if self.is_3d else 'pixel_count'
            measure_col = 'volume' if self.is_3d else 'area'
            size_exponent = 3 if self.is_3d else 2

            # The Connected Components step ran with res=1.0 (raw units) -
            # rescale the measure column now with the real voxel/pixel size.
            table[measure_col] = table[count_col] * (self.resolution ** size_exponent)
            
            # Apply the min-size filter BEFORE measuring - otherwise every
            # sub-threshold component is measured and then thrown away.
            table = table[table[count_col] >= self.min_size].reset_index(drop=True)
            kept = set(table['label'].to_numpy().tolist())
            shape_table = compute_shape_measurements(
                self.cc_result['labels'], self.is_3d, res=self.resolution, only_labels=kept
                )

            merged_table = table.merge(shape_table, on='label', how='left')

            # Only run the expensive pass if something from it was actually ticked.
            # Unlike the cheap measurements above, these cost real time, so the
            # checkboxes gate the computation itself, not just what gets displayed.

            requested_advanced = [m for m in self.selected_measurements if m in ADVANCED_MEASUREMENTS]

            if self.is_3d and requested_advanced:
                advanced_table = compute_advanced_shape_measurements(
                    self.cc_result['labels'], res=self.resolution, requested = requested_advanced,
                    only_labels=kept
                    )
                merged_table = merged_table.merge(advanced_table, on='label', how='left')

            self.finished.emit(merged_table)

        except Exception as e:
            self.error.emit(str(e))



class SliceEvolutionThread(QThread):
    """
    Computes 2D polytope functions independently on every slice/time-step of a 3D stack,
    then derives the Omega/Delta-Omega evolution metrics from those per-slice curves.
    """
    # Signal carries everything the plot window needs: which slice/time indices were used,
    # the raw+scaled curves for each of them, and the two Omega-family metric dicts.
    finished = Signal(list, list, list, dict, dict)
    error = Signal(str)
    progress = Signal(int, int)  # (completed, total)

    def __init__(self, image_data: np.ndarray, selected_polytopes: List[str], step: int,
                 reference_index: int = 0, compute_omega: bool = True,
                 compute_delta_omega: bool = True, signed_delta_omega: bool = False,
                 stack_labels = None, reverse: bool = False):
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
        # False = first slice/time-step to last (default); True = last to first. Same
        # userData convention as SliceEvolutionSettingsDialog.direction_combo.
        self.reverse = reverse

    def run(self):
        try:
            n_slices = self.image_data.shape[0]

            # array_positions: real indices into self.image_data -  which ones, and in what
            # order, depends on direction. e.g. step=1, reverse=False -> 0, 1, 2, ...;
            # step=1, reverse=True -> n_slices-1, n_slices-2, ..., 0. Must match the exact
            # same formula SliceEvolutionSettingsDialog._update_reference_options() used to
            # build the reference dropdown, or reference_position below could fail to find
            # self.reference_index in this list.
            if self.reverse:
                array_positions = list(range(n_slices - 1, -1, -self.step))
            else:
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

            total = len(array_positions)
            raw_curves_list = []
            scaled_curves_list = []
            for i, pos in enumerate(array_positions):
                raw, scaled = compute_curves(self.image_data[pos], self.selected_polytopes)
                raw_curves_list.append(raw)
                scaled_curves_list.append(scaled)
                self.progress.emit(i + 1, total)

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

class SliceEvolutionStreamingThread(QThread):
    """
    Streaming counterpart to SliceEvolutionThread - for a 4D time series imported as a
    list of file paths (one 3D volume per time step) instead of eagerly loaded and
    stacked into one big in-memory array. Loads, computes, and discards ONE volume at a
    time - same reasoning as MinkowskiEvolutionStreamingThread.

    Always 3D (compute_3d_polytope_curves) - streaming mode only exists for "many large
    3D volumes".
    """

    finished = Signal(list, list, list, dict, dict)
    error = Signal(str)
    progress = Signal(int, int)  # (completed, total)

    def __init__(self, file_paths, selected_polytopes: List[str], step: int,
                 reference_index: int = 0, compute_omega: bool = True,
                 compute_delta_omega: bool = True, signed_delta_omega: bool = False,
                 stack_labels=None, reverse: bool = False):
        super().__init__()
        self.file_paths = file_paths
        self.selected_polytopes = selected_polytopes
        self.step = step
        self.reference_index = reference_index
        self.compute_omega = compute_omega
        self.compute_delta_omega = compute_delta_omega
        self.signed_delta_omega = signed_delta_omega
        self.stack_labels = stack_labels
        self.reverse = reverse

    def run(self):
        try:
            n_files = len(self.file_paths)

            if self.reverse:
                array_positions = list(range(n_files - 1, -1, -self.step))
            else:
                array_positions = list(range(0, n_files, self.step))

            if self.stack_labels is not None:
                slice_indices = [self.stack_labels[pos] for pos in array_positions]
            else:
                slice_indices = array_positions

            total = len(array_positions)
            raw_curves_list = []
            scaled_curves_list = []
            for i, pos in enumerate(array_positions):
                # one volume in memory at a time - overwritten (and garbage collected)
                # the moment the next iteration loads a new one
                volume = load_multipage_tif(self.file_paths[pos])
                raw, scaled = compute_3d_polytope_curves(volume, self.selected_polytopes)
                raw_curves_list.append(raw)
                scaled_curves_list.append(scaled)
                self.progress.emit(i + 1, total)

            reference_position = array_positions.index(self.reference_index)

            omega_dict = {}
            delta_omega_dict = {}
            for name in self.selected_polytopes:
                curve_series = [curves[name] for curves in raw_curves_list]
                if name in ('p3h', 'p3v'):
                    curve_series = [curve[::2] for curve in curve_series]

                if self.compute_omega:
                    omega_dict[name] = omega_n(curve_series, reference_index=reference_position)
                if self.compute_delta_omega:
                    delta_omega_dict[name] = delta_omega(curve_series, signed=self.signed_delta_omega)

            self.finished.emit(slice_indices, raw_curves_list, scaled_curves_list, omega_dict, delta_omega_dict)
        except Exception as e:
            self.error.emit(str(e))

class CropCenterThread(QThread):
    """
    Thread that finds the center crop box (see find_center_crop_box in
    processing/cropping.py) in the background. It labels the exterior in every slice,
    which takes several seconds on a large volume - long enough to freeze the UI if
    run directly on the GUI thread.

    Only FINDS the box. Cutting it out is instant and happens back on the GUI thread
    (ImageViewer._apply_center_crop), after the user has confirmed it.
    """

    finished = Signal(object)    # Emits the CropBox
    error = Signal(str)
    progress = Signal(int, int)  # (completed, total) - slices searched for the sample so far

    def __init__(self, image_data: np.ndarray, exterior_value, make_cube: bool, margin: int):
        super().__init__()
        self.image_data = image_data
        self.exterior_value = exterior_value
        self.make_cube = make_cube
        self.margin = margin

    def run(self):
        try:
            box = find_center_crop_box(
                self.image_data,
                self.exterior_value,
                make_cube=self.make_cube,
                margin=self.margin,
                # emit() can be called from this worker thread; Qt delivers the
                # signal to the GUI thread, which updates the progress bar.
                progress_callback=self.progress.emit,
            )
            self.finished.emit(box)
        except Exception as e:
            self.error.emit(str(e))

class SaveImageThread(QThread):
    """
    Thread that writes the current image to disk for File > Save As, so saving a large
    volume (often several GB) doesn't freeze the UI, and the progress bar can move.

    Two modes, picked by `file_names`:
        - file_names is None: save ONE TIFF at `destination` (a file path) - a 2D image,
          or a multi-page TIFF for a volume (see save_tiff in utils/image_utils.py).
        - file_names is a list: save one TIFF per slice into `destination` (a folder),
          slice z under file_names[z] (see save_tiff_sequence).
    """

    finished = Signal(str)       # Emits `destination` once everything is written
    error = Signal(str)
    progress = Signal(int, int)  # (completed, total) - pages or files written so far

    def __init__(self, image_data: np.ndarray, destination: str, file_names: Optional[List[str]] = None):
        super().__init__()
        self.image_data = image_data
        self.destination = destination
        self.file_names = file_names

    def run(self):
        try:
            if self.file_names is None:
                save_tiff(self.destination, self.image_data, progress_callback=self.progress.emit)
            else:
                save_tiff_sequence(
                    self.destination, self.image_data, self.file_names,
                    progress_callback=self.progress.emit,
                )
            self.finished.emit(self.destination)
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
        # One file name per slice along axis 0 (Z for a volume, T for a 2D time series),
        # used as the default names by File > Save As > Image Sequence. Either the original
        # slice files' names (folder imports) or "<name>_0000.tif", ... for a volume opened
        # from one multi-page TIFF. Cropping keeps the names of the slices it keeps, so each
        # saved slice still carries its original slice number. None for 2D and 4D data.
        self.slice_file_names: Optional[List[str]] = None
        #connected coponents results
        self.cc_labels_data: Optional[np.ndarray] = None # label volume from the last connected-components run
        self.cc_result: Optional[dict] = None # dict from connected_components_2d/_3d() with all the measurements
        # Labels currently shown in the colored view - the min-size filter sets
        # this, and deletions in the results dialog subtract from it.
        self._cc_visible_labels: set = set()

        # The results dialog currently open, if any. It's non-modal and stays
        # wired to this viewer, so it must be closed whenever the labelling it
        # describes is replaced - otherwise OK on it commits stale label ids.
        self._cc_results_window = None
        
        self.cc_color_lut: Optional[np.ndarray] = None # (n_labels+1, 3) uint8 - label id -> RGB color
        self.show_cc_colors: bool = False # is the checkbox currently on?

        self.current_file_path: Optional[str] = None
        self.current_slice_index: int = 0
        self.current_time_index: int = 0 # which 3D volume (T axis) is showing, 4D data only

        # Crop Center: the open (non-modal) settings dialog and the background search, if any.
        self.crop_center_dialog: Optional[CropCenterDialog] = None
        self.crop_thread: Optional[CropCenterThread] = None

        # File > Save As: the background thread writing the image to disk, if any.
        self.save_thread: Optional[SaveImageThread] = None

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

        # Toggle for showing connected-components colors instead of the raw grayscale
        # slice - hidden until a connected-components calculation has actually run.
        self.cc_color_checkbox = QCheckBox("Show Connected Components Colors")
        self.cc_color_checkbox.setVisible(False)
        self.cc_color_checkbox.stateChanged.connect(self.on_cc_color_toggle)
        self.layout.addWidget(self.cc_color_checkbox)

        # Store current pixmap
        self.current_pixmap: Optional[QPixmap] = None

    def _create_menu(self):
        """Create the application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = file_menu.addAction("&Import Single 2D/3D Image")
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("Open a single 2D or 3D TIF image file")
        open_action.triggered.connect(self.open_image)

        import_volume_action = file_menu.addAction("Import Single 3D &Volume (from Z-Slice Files)")
        import_volume_action.setStatusTip("Assemble a single 3D volume from a folder of 2D Z-slice image files")
        import_volume_action.triggered.connect(self.open_import_volume_dialog)

        import_timeseries_action = file_menu.addAction("Import &Time Series/4D Dataset (2D or 3D per step)")
        import_timeseries_action.setStatusTip(
            "Import multiple time steps from a folder - each step can be a single 2D slice "
            "or a full 3D volume; large 3D-volume datasets can use \n low-memory mode to avoid "
            "loading everything at once")
        import_timeseries_action.triggered.connect(self.open_import_time_series_dialog)

        file_menu.addSeparator()

        # Save As submenu - saves whatever is in the viewer now, including the result of
        # Binarize, Crop Center or a connected-components filter.
        save_as_menu = file_menu.addMenu("Save &As")

        save_sequence_action = save_as_menu.addAction("Image &Sequence (one TIFF per slice)...")
        save_sequence_action.setStatusTip(
            "Save every slice of the current 3D volume as its own TIFF file in a folder, "
            "named after the original slices")
        save_sequence_action.triggered.connect(self.save_as_image_sequence)

        save_single_action = save_as_menu.addAction("Single &Image (TIFF, 2D or 3D)...")
        save_single_action.setStatusTip(
            "Save the current image as one TIFF file - a multi-page TIFF for a 3D volume")
        save_single_action.triggered.connect(self.save_as_single_image)

        file_menu.addSeparator()

        exit_action = file_menu.addAction("E&xit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)

        # Process menu
        process_menu = menubar.addMenu("&Process")

        binarize_action = process_menu.addAction("&Binarize")
        binarize_action.setStatusTip("Select a pixel value to become the foreground (1); everything else becomes background (0)")
        binarize_action.triggered.connect(self.open_binarize_dialog)

        # Cropping submenu - room for more cropping tools (e.g. a manual box) later
        cropping_menu = process_menu.addMenu("&Cropping")

        crop_center_action = cropping_menu.addAction("Crop &Center...")
        crop_center_action.setStatusTip(
            "Crop the largest box around the center of a cylindrical sample that lies "
            "completely inside the sample, removing the exterior (air) around it"
        )
        crop_center_action.triggered.connect(self.open_crop_center_dialog)

        # SMD menu
        smd_menu = menubar.addMenu("&SMDs")

        smds_action = smd_menu.addAction("Calculate &SMDs")
        # smds_action.setShortcut("Ctrl+S")
        smds_action.setStatusTip("Calculate SMDs from the current image")
        smds_action.triggered.connect(self.open_smd_dialog)

        slice_evolution_action = smd_menu.addAction("Calculate Slice &Evolution (Z)...")
        slice_evolution_action.setStatusTip(
            "Compute 2D SMDs independently on every Z-slice of a 3D volume (top-to-bottom or "
            "bottom-to-top), with Omega/Delta-Omega evolution metrics"
        )
        slice_evolution_action.triggered.connect(self.open_slice_evolution_dialog)

        # Chord length
        chord_length_action = smd_menu.addAction("Calculate &Chord Length...")
        chord_length_action.setStatusTip(
            "Compute the mean chord length (via S2 slope and direct sampling) and the "
            "chord-length distribution of the foreground phase"
        )
        chord_length_action.triggered.connect(self.open_chord_length_dialog)

        chord_length_evolution_action = smd_menu.addAction("Calculate Chord Length &Evolution...")
        chord_length_evolution_action.setStatusTip(
            "Compute chord length independently on every slice/time-step of a stack"
        )
        chord_length_evolution_action.triggered.connect(self.open_chord_length_evolution_dialog)


        # REV menu
        rev_menu = menubar.addMenu("&REV/RES")
        rev_action = rev_menu.addAction("Calculate &REV/RES")
        rev_action.setShortcut("Ctrl+R")
        rev_action.setStatusTip("Calculate REV from the current image")
        rev_action.triggered.connect(self.calculate_rev)  # Placeholder, implement
        rev_action

        #Image analysis
        image_analysis_menu = menubar.addMenu("&Image Analysis")

        minkowski_action = image_analysis_menu.addAction("&Calculate Minkowski Functionals...")
        minkowski_action.setStatusTip(
            "Compute Minkowski functionals (volume/area, surface/perimeter, "
            "mean curvature, Euler characteristic) from the current image"
        )
        minkowski_action.triggered.connect(self.open_minkowski_dialog)

        # connected components
        cc_action = image_analysis_menu.addAction("Calculate &Connected Components...")
        cc_action.setStatusTip(
            "Label connected components of the foreground phase and measure each "
            "one's size (2D area or 3D volume)."
        )
        cc_action.triggered.connect(self.open_connected_components_dialog)

        # measurements
        measurements_action = image_analysis_menu.addAction("Calculate &Measurements...")
        measurements_action.setStatusTip(
            "Compute shape measurements for the components found by the last "
            "Connected Components run."
        )
        measurements_action.triggered.connect(self.open_measurements_dialog)


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

    def labels_to_qpixmap(self, rgb_image: np.ndarray)->QPixmap:

        """
        Convert an (H, W, 3) uint8 RGB array into a QPixmap - same idea as
        numpy_to_qpixmap but for an already-colored image (Format_RGB888
        instead of Format_Grayscale8), used for the connected-components color overlay.
        """

        height, width, _ = rgb_image.shape
        bytes_per_line = width * 3
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

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
        # self.current_pixmap = self.numpy_to_qpixmap(current_2d, self._display_min, self._display_max)

        # Connected-components colors replace the grayscale slice entirely when the
        # checkbox is on - cc_labels_data is either a full 3D label volume (index by
        # current_slice_index, same as current_2d above) or a single 2D label image
        # (used directly, same as current_2d's own ndim==2 fallback above).
        if self.show_cc_colors and self.cc_labels_data is not None:
            if self.current_image_data.ndim == 3:
                label_slice = self.cc_labels_data[self.current_slice_index, :, :]
            else:
                label_slice = self.cc_labels_data
            rgb_slice = self.cc_color_lut[label_slice]  # (H, W, 3) uint8, via LUT lookup
            self.current_pixmap = self.labels_to_qpixmap(rgb_slice)

        else:
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
    def on_cc_color_toggle(self, state):
        """Switch the main view between the raw grayscale slice and connected-components colors."""

        self.show_cc_colors = bool(state)
        self.display_current_slice()

    def _close_cc_results_window(self):
        """Close the open results dialog, if any, without letting it act.

        blockSignals() rather than disconnect(): closing the dialog normally
        emits changes_reverted, and PySide6's disconnect() warns when nothing
        is connected and raises on an already-deleted dialog.
        """
        window = self._cc_results_window
        self._cc_results_window = None
        if window is None:
            return
        try:
            window.blockSignals(True)
            window.close()
            window.deleteLater() # it's parented to the viewer, so free it now rather than at exit
        except RuntimeError:
            pass # already deleted, nothing to do

    def _reset_cc_state(self):

        """Forget the current connected-components result.

        Call this whenever the image or its labelling is replaced - a new
        image, Binarize, or a fresh Connected Components run. Every piece of
        CC state describes one labelling of one image, so none of it is valid
        afterwards.
        """
        self._close_cc_results_window()
        self.cc_result = None
        self.cc_labels_data = None
        self._cc_visible_labels = set()
        self.cc_color_lut = None
        self.show_cc_colors = False
        self.cc_color_checkbox.setChecked(False)
        self.cc_color_checkbox.setVisible(False)





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

    def _finalize_loaded_image(self, image_data: np.ndarray, source_label: str, data_mode: str, stack_labels=None,
                               warn_if_not_binary: bool = True, slice_file_names: Optional[List[str]] = None):

        """
        Shared tail-end of loading any image (single file or imported sequence): validates
        binary-ness, configures the slice slider, displays the first slice, and updates the
        status bar/title.

        stack_labels defaults to None, which RESETS it for every load path that doesn't pass
        one explicitly (open_image, open_import_volume_dialog) - only
        open_import_time_series_dialog currently passes real values, so a previously-loaded
        time series's numbers don't linger and get wrongly reused for whatever gets loaded next.

        warn_if_not_binary=False skips the "Binarization Required" pop-up - used when the
        image is replaced by a processed version of itself (e.g. a crop), where the user
        already saw that warning when the original was loaded.

        slice_file_names: one file name per slice along axis 0, used as the default names
        by File > Save As > Image Sequence. Folder imports pass the original slice files'
        names. Left as None, a 3D array gets "<source_label without extension>_0000.tif",
        ... instead (e.g. a volume opened from one multi-page TIFF); 2D and 4D data get
        None, since they can't be saved as a sequence.
        """

        # Normalise shapes that are really 2D but arrive with a third axis,
        # otherwise every `ndim == 3` check downstream treats them as volumes:
        # an RGB/RGBA image (H, W, 3|4), or a single-page stack (1, H, W).

        if image_data.ndim == 3:
            if image_data.shape[-1] in (3, 4):  # RGB or RGBA. in RGBA case, drop the alpha channel which controls transparency.
                # Colour channels, not slices. These images are expected to be
                # binary/greyscale here, so collapse to one channel rather than
                # doing a weighted RGB->grey conversion.
                image_data = image_data[..., 0]
            elif image_data.shape[0] == 1:  # single-page stack, not a volume
                image_data = image_data[0]

        # A 2D array can only be a plain 2D image, whatever mode the caller
        # guessed from the pre-normalisation shape. Several tools route on
        # data_mode (Minkowski sends 'time_series' to the evolution dialog),
        # so a stale mode would push a 2D image down a stack-processing path.

        if image_data.ndim == 2:
            data_mode = '2d'



        is_valid, unique_values = self.validate_binary_image(image_data)

        self.current_image_data = image_data
        # print(f"Loaded image shape: {image_data.shape}")
        self.current_slice_index = 0
        self.current_time_index = 0
        self.data_mode = data_mode
        self.stack_labels = stack_labels

        # Default names for saving as an image sequence. Checked against the shape AFTER the
        # normalisation above: e.g. a 1-file folder import arrives as (1, H, W) but is
        # displayed (and saved) as a 2D image, so its single name no longer applies.
        if image_data.ndim != 3:
            self.slice_file_names = None
        elif slice_file_names is not None and len(slice_file_names) == image_data.shape[0]:
            self.slice_file_names = list(slice_file_names)
        else:
            stem = os.path.splitext(source_label)[0]
            self.slice_file_names = default_slice_names(stem, image_data.shape[0])

        # A connected-components result belongs to the OLD image - reset it here so it
        # doesn't linger and get wrongly displayed over unrelated new data.
        self._reset_cc_state()


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
            if warn_if_not_binary:
                QMessageBox.warning(
                    self, "Binarization Required",
                    f"This image has {len(unique_values)} distinct pixel values: {list(unique_values)}.\n\n"
                    "Use Process > Binarize to select a foreground label before running calculations."
                )

        self.setWindowTitle(f"SMiCA - [{dims}] - {source_label}")   


    def _load_and_stack_2d_files(self, file_paths):

        """Load a list of 2D image files and stack them into a single (N, H, W) array."""
        slices = [np.array(Image.open(long_path_safe(p))) for p in file_paths]
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

        file_paths = dialog.get_sorted_file_paths()
        try:
            image_data = self._load_and_stack_2d_files(file_paths)
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
        self._finalize_loaded_image(
            image_data, os.path.basename(dialog.folder_path), data_mode='3d_volume',
            # the original slice files' names, so Save As > Image Sequence reuses them
            slice_file_names=[os.path.basename(p) for p in file_paths],
        )


    def _load_multipage_tif(self, file_path):
        return load_multipage_tif(file_path)


    def open_import_time_series_dialog(self):

        """Import a time series (2D slices, or full 3D volumes) of images from a folder."""

        dialog = ImportSequenceDialog(ask_file_type= True, parent = self)

        if dialog.exec() != QDialog.Accepted:
            return

        # Reset here (not just set below) so a PREVIOUS streaming import's file paths can
        # never leak into a differently-loaded dataset opened via this same method.
        self.current_volume_file_paths = None
        slice_file_names = None  # only set for 2D time steps, the one case that is a 3D (T, Y, X) stack
        try:
            if dialog.get_is_3d_files() and dialog.get_low_memory_mode():
                # Streaming mode: don't load every volume - remember the file paths (used by
                # MinkowskiEvolutionStreamingThread later) and preview only the FIRST volume,
                # so a 100-volume dataset that can't fit in RAM as one array still opens.
                file_paths = dialog.get_sorted_file_paths()
                self.current_volume_file_paths = file_paths
                image_data = load_multipage_tif(file_paths[0])
                data_mode = '4d_time_series_streaming'

            elif dialog.get_is_3d_files():
                          
                image_data =  self._load_and_stack_3d_files(dialog.get_sorted_file_paths())
                data_mode = '4d_time_series'
            else:
                image_data = self._load_and_stack_2d_files(dialog.get_sorted_file_paths())
                data_mode = 'time_series'
                slice_file_names = [os.path.basename(p) for p in dialog.get_sorted_file_paths()]


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
            slice_file_names=slice_file_names,
        )

    # ----------------------------------------------------------------------------------
    # File > Save As
    # ----------------------------------------------------------------------------------

    def _can_save_current_image(self) -> bool:
        """
        Checks shared by both Save As options. Shows a message and returns False if the
        current image can't be saved right now.
        """
        if self.current_image_data is None:
            QMessageBox.warning(self, "No Image", "Please open an image first.")
            return False

        # 4D data is a stack of volumes - one TIFF per slice doesn't map onto it, and in
        # low-memory mode only the first volume is even in memory.
        if self.data_mode in ('4d_time_series', '4d_time_series_streaming'):
            QMessageBox.information(
                self, "Save Not Available",
                "Saving is available for 2D images, 3D volumes and time series of 2D images.\n\n"
                "4D time series (a 3D volume per step) can't be saved yet."
            )
            return False

        # Replacing self.save_thread while it is still writing would destroy a running
        # QThread, which crashes the application.
        if self.save_thread is not None and self.save_thread.isRunning():
            QMessageBox.information(
                self, "Save in Progress",
                "The image is still being saved - please wait for it to finish."
            )
            return False

        return True

    def _default_save_stem(self) -> str:
        """
        Name (without extension) suggested when saving the image as a single file: the
        opened file's name, or the folder's name for a folder import.
        """
        if not self.current_file_path:
            return "image"
        name = os.path.basename(os.path.normpath(self.current_file_path))
        if os.path.isdir(self.current_file_path):
            return name                      # a folder: "scan.v2" is a name, not an extension
        return os.path.splitext(name)[0]     # a file: drop ".tif"

    def save_as_single_image(self):
        """
        File > Save As > Single Image: save the current image as ONE TIFF file - a plain
        TIFF for a 2D image, a multi-page TIFF (one page per slice) for a 3D volume.
        """
        if not self._can_save_current_image():
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image As Single TIFF",
            suggested_save_path(f"{self._default_save_stem()}.tif"),  # opens in the last save folder
            "TIFF Files (*.tif *.tiff)"
        )
        if not file_path:
            return  # cancelled

        # Make sure the file actually ends up as a TIFF, even if the user typed another
        # extension (or none) - the content is always TIFF.
        if not file_path.lower().endswith(('.tif', '.tiff')):
            file_path += ".tif"

        # (QFileDialog has already asked the user to confirm overwriting an existing file.)
        self._start_save_thread(file_path, file_names=None)

    def save_as_image_sequence(self):
        """
        File > Save As > Image Sequence: save every slice of the current 3D array as its
        own TIFF file in a folder the user picks.

        Each slice is saved under its name from self.slice_file_names: the original slice
        file's name for a folder import (so the slice number in it is kept, also after a
        crop), or "<name>_0000.tif", ... for a volume opened from one multi-page TIFF.
        """
        if not self._can_save_current_image():
            return

        if self.current_image_data.ndim != 3 or self.slice_file_names is None:
            QMessageBox.information(
                self, "Not a Volume",
                "A 2D image has only one slice - use File > Save As > Single Image instead."
            )
            return

        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder to Save the Image Sequence In", suggested_save_dir()
        )
        if not folder:
            return  # cancelled

        file_names = list(self.slice_file_names)

        # Unlike a file dialog, a folder dialog doesn't warn about overwriting - and saving
        # into the folder the slices were imported from would replace the original files.
        existing = [name for name in file_names if os.path.exists(os.path.join(folder, name))]
        if existing:
            reply = QMessageBox.question(
                self,
                "Overwrite Files?",
                f"{len(existing)} of the {len(file_names)} slice files already exist in:\n"
                f"{folder}\n\n"
                f"e.g. {existing[0]}\n\n"
                "Overwrite them?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,  # default button: don't overwrite
            )
            if reply != QMessageBox.Yes:
                self.status_bar.showMessage("Save cancelled - no files were written")
                return

        self._start_save_thread(folder, file_names=file_names)

    def _start_save_thread(self, destination: str, file_names: Optional[List[str]]):
        """
        Write the current image in the background (see SaveImageThread). `destination` is
        a file path when file_names is None, otherwise the folder for the slice files.
        """
        # Qt's file dialogs return "/" separators even on Windows; normpath gives the usual
        # "D:\data\out" form, so the status message doesn't mix "/" and "\".
        destination = os.path.normpath(destination)

        image = self.current_image_data
        total = image.shape[0] if image.ndim == 3 else 1   # pages / files to write

        # Determinate progress bar, driven by the shared on_evolution_progress handler,
        # which also shows the estimated remaining time in the status bar.
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(0)
        self._evolution_start_time = time.perf_counter()
        self._evolution_description = "Saving image sequence" if file_names is not None else "Saving TIFF"
        self.status_bar.showMessage(f"{self._evolution_description}: 0/{total}")

        # The thread keeps its own reference to the array, so it writes exactly what was
        # shown when Save was chosen, even if the user binarizes or crops in the meantime
        # (those make a new array rather than changing this one).
        self.save_thread = SaveImageThread(image, destination, file_names)
        self.save_thread.progress.connect(self.on_evolution_progress)
        self.save_thread.finished.connect(self.on_save_finished)
        self.save_thread.error.connect(self.on_save_error)
        self.save_thread.start()

    def on_save_finished(self, destination: str):
        """Saving is done - report it in the status bar (no pop-up, to keep saving quick)."""
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximum(0)  # restore indeterminate default for other calculations

        file_names = self.save_thread.file_names
        # The thread object stays around on self; drop its reference so the saved array
        # can be freed once the viewer moves on to another image.
        self.save_thread.image_data = None

        if file_names is None:
            remember_save_dir(destination)  # so the next Save dialog opens in this same folder
            self.status_bar.showMessage(f"Saved: {destination}")
        else:
            # remember_save_dir() expects a file path and keeps its folder
            remember_save_dir(os.path.join(destination, file_names[0]))
            self.status_bar.showMessage(
                f"Saved {len(file_names)} slices ({file_names[0]} ... {file_names[-1]}) to: {destination}"
            )

    def on_save_error(self, error_msg: str):
        """Saving failed (e.g. no write permission, or the disk is full)."""
        self.save_thread.image_data = None  # same reason as in on_save_finished
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximum(0)
        QMessageBox.critical(self, "Save Error", f"Failed to save the image:\n{error_msg}")
        self.status_bar.showMessage(f"Error: {error_msg}")


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

        # The previous labelling (and any open results dialog) describes the
        # image as it was before binarizing - e.g. the other phase.
        self._reset_cc_state()

        # Re-display (same slice index as before, now showing the binarized result)
        self.display_current_slice()

        self.status_bar.showMessage(f"Binarized: pixel value {foreground_value} set as foreground")
        QMessageBox.information(
            self,
            "Binarization Complete",
            f"Pixel value {foreground_value} is now the foreground (1); everything else is background (0)."
        )

    def open_crop_center_dialog(self):
        """
        Process > Cropping > Crop Center...: open the crop settings dialog.

        The dialog is non-modal (like Binarize), so the user can keep hovering over the
        image to check pixel values - e.g. to see which value the air has - while it is
        open. This method therefore returns straight away; clicking OK calls
        _start_center_crop_search, and the image is only cropped in
        on_crop_center_finished, after the user has seen and confirmed the box.
        """

        # Only one Crop Center dialog at a time - clicking the menu again just brings the
        # open one to the front.
        if self.crop_center_dialog is not None and self.crop_center_dialog.isVisible():
            self.crop_center_dialog.raise_()
            self.crop_center_dialog.activateWindow()
            return

        if self.current_image_data is None:
            QMessageBox.warning(self, "No Image", "Please open an image first.")
            return

        # In a time series the first axis is time, not Z, and streaming mode only has one
        # volume in memory - neither fits "one fixed box for the whole image" yet.
        if self.data_mode in ('time_series', '4d_time_series', '4d_time_series_streaming'):
            QMessageBox.information(
                self, "Crop Center Not Available",
                "Crop Center currently works on single 2D images and 3D volumes only.\n\n"
                "Time series (2D + time or 3D + time) aren't supported yet."
            )
            return

        is_3d = self.current_image_data.ndim == 3

        # Only values found along the image edges can be the air around the sample, so
        # the dialog offers just those (most common first) instead of every value.
        # Kept on self - with show() instead of exec() this method returns immediately, and
        # Python would otherwise garbage-collect the dialog while it is still on screen.
        self.crop_center_dialog = CropCenterDialog(
            exterior_value_candidates(self.current_image_data), is_3d=is_3d, parent=self
        )

        # Remember which image the dialog was opened for (checked when OK is clicked). A
        # weak reference doesn't keep that image alive by itself, so a large volume can
        # still be freed if another image replaces it while the dialog is open.
        self._crop_dialog_image_ref = weakref.ref(self.current_image_data)

        self.crop_center_dialog.accepted.connect(self._start_center_crop_search)
        self.crop_center_dialog.show()

    def _start_center_crop_search(self):
        """Called when the (non-modal) Crop Center dialog is accepted - starts the box search
        in a background thread."""

        # The main window stays usable while the dialog is open, so another image may have
        # been loaded or binarized in the meantime. The dialog's exterior values describe
        # the image it was opened for, so don't apply them to a different one.
        if self._crop_dialog_image_ref() is not self.current_image_data:
            QMessageBox.warning(
                self, "Image Changed",
                "The image changed while the Crop Center dialog was open.\n\n"
                "Please open Process > Cropping > Crop Center... again."
            )
            return

        # Replacing self.crop_thread while the previous search is still busy would destroy
        # a running QThread, which crashes the application.
        if self.crop_thread is not None and self.crop_thread.isRunning():
            QMessageBox.information(
                self, "Crop Center",
                "A center crop search is already running - please wait for it to finish."
            )
            return

        dialog = self.crop_center_dialog
        is_3d = self.current_image_data.ndim == 3

        # Progress = slices searched for the sample (the slow first step of the box search)
        total = self.current_image_data.shape[0] if is_3d else 1

        # Determinate progress bar, driven by the shared on_evolution_progress handler,
        # which also shows the estimated remaining time in the status bar.
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(0)
        self._evolution_start_time = time.perf_counter()
        self._evolution_description = "Finding center crop"
        self.status_bar.showMessage(f"{self._evolution_description}: 0/{total}")
        QApplication.processEvents()

        self.crop_thread = CropCenterThread(
            self.current_image_data,
            exterior_value=dialog.get_exterior_value(),
            make_cube=dialog.get_make_cube(),
            margin=dialog.get_margin(),
        )
        self.crop_thread.progress.connect(self.on_evolution_progress)
        self.crop_thread.finished.connect(self.on_crop_center_finished)
        self.crop_thread.error.connect(self.on_crop_center_error)
        self.crop_thread.start()

    def on_crop_center_finished(self, box: CropBox):
        """Show the box that was found and, if the user confirms, crop the image to it."""

        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximum(0)  # restore indeterminate default for other calculations

        # The box belongs to the image it was computed on. If the user loaded or binarized
        # another image while the search was running, don't apply it to that one.
        image = self.crop_thread.image_data
        # The thread object stays around on self; drop its reference so the full-size
        # original can be freed once the cropped copy replaces it.
        self.crop_thread.image_data = None
        if image is not self.current_image_data:
            self.status_bar.showMessage("Center crop discarded - the image changed while it was being computed")
            return

        is_3d = image.ndim == 3
        depth, height, width = box.size_zyx

        # Too small to be useful - and a (Z, Y, 3)-shaped volume would also be mistaken
        # for an RGB image by _finalize_loaded_image and collapsed to 2D.
        if min(height, width) < 5 or (is_3d and depth < 5):
            size = f"{width} x {height} x {depth}" if is_3d else f"{width} x {height}"
            QMessageBox.warning(
                self, "Crop Too Small",
                f"The largest box inside the sample is only {size} pixels.\n\n"
                "Check that the right exterior value was chosen, or reduce the safety margin."
            )
            self.status_bar.showMessage("Center crop cancelled - box too small")
            return

        # Ranges are shown inclusive ("120 - 379"), matching the pixel positions shown
        # when hovering over the image - box.x_stop itself is exclusive, like Python slicing.
        ranges = [
            f"X: {box.x_start} - {box.x_stop - 1}   ({width} px)",
            f"Y: {box.y_start} - {box.y_stop - 1}   ({height} px)",
        ]
        if is_3d:
            ranges.append(f"Z: {box.z_start} - {box.z_stop - 1}   ({depth} slices)")
            new_size = f"{width} x {height} x {depth} (X x Y x Z)"
            old_size = f"{image.shape[2]} x {image.shape[1]} x {image.shape[0]}"
        else:
            new_size = f"{width} x {height} (X x Y)"
            old_size = f"{image.shape[1]} x {image.shape[0]}"

        reply = QMessageBox.question(
            self,
            "Crop Center",
            "Largest box around the center that lies inside the sample:\n\n"
            + "\n".join(ranges) + "\n\n"
            f"New size: {new_size}, was {old_size}.\n\n"
            "This replaces the image in the viewer (the file on disk is not changed). Crop now?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            self.status_bar.showMessage("Center crop cancelled")
            return

        self._apply_center_crop(box)

    def on_crop_center_error(self, error_msg: str):
        """Handle an error while searching for the center crop box."""

        self.crop_thread.image_data = None  # same reason as in on_crop_center_finished
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximum(0)
        QMessageBox.critical(self, "Crop Center Error", f"Could not find a center crop:\n{error_msg}")
        self.status_bar.showMessage(f"Error: {error_msg}")

    def _apply_center_crop(self, box: CropBox):
        """Replace the current image with the contents of `box` and refresh the whole view."""

        cropped = crop_image(self.current_image_data, box)

        # Shown in the window title, e.g. "sample.tif (center crop)".
        base_name = os.path.basename(self.current_file_path) if self.current_file_path else "image"
        source_label = f"{base_name} (center crop)"

        # Keep the names of the slices that survive the crop, so a later Save As > Image
        # Sequence still writes e.g. "rock_0250.tif" for what was slice 250 before cropping.
        slice_file_names = None
        if cropped.ndim == 3 and self.slice_file_names is not None:
            slice_file_names = self.slice_file_names[box.z_start:box.z_stop]

        # Reuse the normal "new image loaded" path: it resets the sliders, the display range,
        # any connected-components result (its labels no longer line up with the cropped
        # image) and the window title. data_mode stays the same - a cropped 3D volume is
        # still a 3D volume.
        self._finalize_loaded_image(
            cropped, source_label, data_mode=self.data_mode, warn_if_not_binary=False,
            slice_file_names=slice_file_names,
        )

        depth, height, width = box.size_zyx
        size = f"{width} x {height} x {depth}" if cropped.ndim == 3 else f"{width} x {height}"
        self.status_bar.showMessage(f"Cropped to the center box: {size}")


    def open_smd_dialog(self):
        """Open the SMD selection dialog (real computation wired up in a later step)."""

        if self.current_image_data is None:
            QMessageBox.warning(self, "No Image", "Please open an image first.")
            return
        
        if self.current_image_data.ndim not in (2, 3, 4):
            QMessageBox.warning(self, "Invalid Image", "Image must be 2D, 3D, or 4D.")
            return
        
        if self.data_mode in ('time_series', '4d_time_series', '4d_time_series_streaming'):
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

    def open_minkowski_dialog(self):
        """Open the Minkowski functionals settings dialog and run the calculation on the current image."""

        if self.current_image_data is None:
            QMessageBox.warning(self, "No Image", "Please open an image first.")
            return
        if self.current_image_data.ndim not in (2, 3, 4):
            QMessageBox.warning(self, "Invalid Image", "Image must be 2D, 3D, or 4D.")
            return

        if self.data_mode in ('time_series', '4d_time_series', '4d_time_series_streaming'):

            self.open_minkowski_evolution_dialog()
            return
        if self.current_image_data.ndim == 3 and self.data_mode == 'time_series':
            QMessageBox.warning(
                self, "Not Applicable",
                "This is a time series, not a spatial 3D volume - Minkowski functionals "
                "need a real 3D volume. Time-series analysis tools are coming in a later step."
            )
            return
        is_3d = self.current_image_data.ndim == 3
        dialog = MinkowskiSettingsDialog(is_3d=is_3d, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return  # user cancelled

        resolution = dialog.get_resolution()
        unit = dialog.get_unit()
        result = minkowski_3d(self.current_image_data, res=resolution) if is_3d \
            else minkowski_2d(self.current_image_data, res=resolution)


        result_window = MinkowskiResultsDialog(result, unit, is_3d, self)
        result_window.show()

##------------------------------connected components--------------------------------------------
    def open_connected_components_dialog(self):
        """Open the connected-components settings dialog and run labeling on the current image."""

        if self.data_mode in ('4d_time_series', '4d_time_series_streaming'):
            # Streaming mode only holds the FIRST volume as a preview (ndim 3),
            # so without this it'd be labelled as if that one time step were
            # the whole dataset.
            QMessageBox.warning(
                self, "Not Available",
                "Connected components isn't available for 3D time series (4D data) "
                "yet - it works on single 2D images, 3D volumes, and 2D time series."
            )
            return


        if self.current_image_data is None:
            QMessageBox.warning(self, "No Image", "Please open an image first.")
            return
        if self.current_image_data.ndim not in (2, 3):
            QMessageBox.warning(
                self, "Not applicable",
                "Connected components needs a single 2D image, a 3D volume, or a 2D time series."
            )
            return

        # A 2D time series is stacked as (T, H, W) - ndim 3, same as a volume -
        # so shape alone can't tell them apart; data_mode can. Each frame gets
        # 2D connectivity (4/8), not 3D (6/18/26).
        per_frame = self.data_mode == 'time_series'
        is_3d = self.current_image_data.ndim == 3 and not per_frame

        dialog = ConnectedComponentsSettingsDialog(is_3d=is_3d, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return # user cancelled
        
        # Relabelling invalidates the previous result, and any results dialog
        # still open for it - OK on that dialog would commit label ids from
        # the old labelling against the new one.
        self._reset_cc_state()

        self._cc_is_3d = is_3d # stashed for open_measurements_dialog's is_3d branching later

        self.progress_bar.setVisible(True)
        self.status_bar.showMessage(f"Labeling connected components (connectivity={dialog.get_connectivity()})...")
        QApplication.processEvents()

        self.cc_thread = ConnectedComponentsThread(
            self.current_image_data, connectivity=dialog.get_connectivity(),
             is_3d=is_3d, per_frame=per_frame
        )

        self.cc_thread.finished.connect(self.on_connected_components_finished)
        self.cc_thread.error.connect(self.on_connected_components_error)
        self.cc_thread.start()

    def on_connected_components_finished(self, result:dict):
        """Handle completion of connected-components labeling - store the result and
        color every component (no filtering or measurements yet - those happen in the
        Measurements step, which needs this result to already exist)."""

        self.progress_bar.setVisible(False)
        self.cc_result = result

        labels = result['labels']
        n_labels = int(labels.max())
        rng = np.random.default_rng(0)
        colors = (rng.random((n_labels + 1, 3)) * 255).astype(np.uint8)
        colors[0] = 0  # background stays black
        self.cc_labels_data = labels
        self.cc_color_lut = colors

        self.cc_color_checkbox.setVisible(True)
        self.cc_color_checkbox.setChecked(True)
        self.show_cc_colors = True
        self.display_current_slice()

        if self.data_mode == 'time_series':
            n_frames = self.current_image_data.shape[0]
            self.status_bar.showMessage(
                f"Connected components: {result['num_components']} found across "
                f"{n_frames} frames, each frame labelled separately."
            )
        else:

            self.status_bar.showMessage(
                f"Connected components: {result['num_components']} found - "
                f"run Image Analysis > Measurements... to filter/measure them")

    def on_connected_components_error(self, error_msg: str):
        """Handle error during connected-components labeling."""

        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Connected Components Error", f"Error labeling components:\n{error_msg}")
        self.status_bar.showMessage(f"Error: {error_msg}")

    def open_measurements_dialog(self):
        """Open the measurements settings dialog and run measurements on the last
        connected-components result."""

        # Checked before the cc_result guard on purpose - "run Connected
        # Components first" would just send someone into a dead end.
        # Same mode tuple Minkowski uses to route to its evolution dialog.

        if self.data_mode in ('time_series', '4d_time_series', '4d_time_series_streaming'):
            QMessageBox.information(
                self, "Measurements Not Available",
                "Measurements are currently available for single 2D images and 3D"
                " volumes only.\n \n" \
                "Time series (2D + time or 3D + time) aren't supported yet - per-frame "
                "measurement tables are planned for a later version. Connected "
                "Components still works on 2D time series, labelling each frame "
                "separately."
            )
            return

        if self.cc_result is None:
            QMessageBox.warning(
                self, "No Connected Components Result",
                "Please run Image Analysis > Connected Components... first."
                )
            return

        dialog = MeasurementsSettingsDialog(
            is_3d=self._cc_is_3d, num_components=self.cc_result['num_components'], parent=self
        )

        if dialog.exec() != QDialog.Accepted:
            return  # user cancelled

        self._measurements_unit = dialog.get_unit()
        self._measurements_min_size = dialog.get_min_size()
        self._measurements_selected = dialog.get_selected_measurements()

        # A results dialog from the previous run would keep acting on state
        # this run is about to rebuild. Close it before the thread starts, so
        # OK on it can't race the new run, then refresh to drop any deletions
        # it was previewing but never committed.
        self._close_cc_results_window()
        self._refresh_cc_display()

        self.progress_bar.setVisible(True)
        self.status_bar.showMessage("Computing measurements...")
        QApplication.processEvents()

        self.measurements_thread = MeasurementsThread(
            self.cc_result, self._cc_is_3d, dialog.get_resolution(), self._measurements_selected,
            min_size=self._measurements_min_size)
        self.measurements_thread.finished.connect(self.on_measurements_finished)
        self.measurements_thread.error.connect(self.on_measurements_error)
        self.measurements_thread.start()

    def on_measurements_finished(self, merged_table):
        """Handle completion of the measurement step - filter by min size, keep only
        the selected measurement columns, refresh the colored view, and show results."""

        # print("DEBUG measurements finished: merged_table rows:", len(merged_table),
        #       "| cc_result table rows:", len(self.cc_result['table']))
        
        self.progress_bar.setVisible(False)

        count_col = 'voxel_count' if self._cc_is_3d else 'pixel_count'
        measure_col = 'volume' if self._cc_is_3d else 'area'
        total_components = self.cc_result['num_components']

        filtered_table = merged_table[merged_table[count_col] >= self._measurements_min_size].reset_index(drop=True)
        display_table = filtered_table[['label', count_col, measure_col] + self._measurements_selected]

        # The min-size filter sets the baseline set of visible components;
        # deletions in the results dialog subtract from it.
        self._cc_visible_labels = set(filtered_table['label'].to_numpy().tolist())

        # LUT is built over EVERY label, not just the visible ones - that way
        # deleting a component leaves every survivor's colour unchanged instead
        # of reshuffling the whole palette on each edit.

        labels = self.cc_result['labels']
        n_labels = int(labels.max())
        rng = np.random.default_rng(0)
        colors = (rng.random((n_labels + 1, 3)) * 255).astype(np.uint8)
        colors[0] = 0
        self.cc_color_lut = colors

        self.show_cc_colors = True
        self.cc_color_checkbox.setChecked(True)
        self._refresh_cc_display()   # replaces the inline masking + display_current_slice()

        # Show the results table in a new dialog
        result_window = ConnectedComponentsResultsDialog(
            display_table, self._measurements_unit, self._cc_is_3d, total_components, self
        )

        # Same signal/slot wiring as the calculation threads above - the dialog
        # reports what the user wants, the viewer owns the label volume and acts.
        result_window.preview_requested.connect(self._refresh_cc_display)
        result_window.changes_committed.connect(self.on_cc_changes_committed)
        result_window.changes_reverted.connect(self._refresh_cc_display)

        # Kept so the dialog can be closed whenever the labelling it shows is
        # replaced - see _close_cc_results_window.
        self._cc_results_window = result_window

        result_window.show()
        self.status_bar.showMessage(f"Measurements: {total_components} components found - {len(display_table)} shown")

    def _refresh_cc_display(self, hidden_labels=frozenset()):
        """Rebuild the colored slice from the visible components, minus whatever
        the results dialog is currently hiding.

        Only cc_labels_data changes - the colour LUT covers every label, so
        surviving components keep the same colour across edits instead of the
        whole palette reshuffling each time something is deleted.

        _refresh_cc_display takes a default of frozenset() because it gets 
        connected to two different signals — 
        preview_requested passes a set of labels to hide,
          while changes_reverted passes nothing at all. Revert is just "refresh with nothing hidden."

        """
        if self.cc_result is None:
            return

    
        labels = self.cc_result['labels']
        # visible = allowed labels - temporarily hidden labels.
        visible = self._cc_visible_labels - set(hidden_labels or ())

        # Same LUT-masking trick already used in on_measurements_finished:
        # build a boolean lookup indexed by label id, then index the whole
        # volume through it in one vectorised pass.
        max_label = int(labels.max())
        keep = np.zeros(max_label + 1, dtype=bool)
        # If there are any components to display, mark their
        # positions in the lookup table as True. Ignore any id beyond the
        # volume's current max - that only happens if the visible set has
        # drifted out of sync, and a missing colour beats an IndexError that
        # takes the window down.
        visible = {label for label in visible if label <= max_label}
        if visible:
            keep[np.fromiter(visible, dtype=int)] = True

        # Importantly, surviving components retain their original
        # label IDs, so their assigned colors remain unchanged.
        self.cc_labels_data = np.where(keep[labels], labels, 0)
        # Redraw the current slice using the updated labeled data.
        self.display_current_slice()

    def on_cc_changes_committed(self, deleted_labels):
        """
        Permanently delete selected connected components from the label volume.

        This function is called when the user confirms deletions by clicking
        OK in the results dialog.

        Deleted components are replaced with background (label 0), so they
        will not reappear if measurements are performed again.
        """

        # print("DEBUG commit: deleting", len(deleted_labels), "labels")
        # print("  BEFORE  table rows:", len(self.cc_result['table']),
        #       "| image fg:", int(self.current_image_data.sum()),
        #       "| num_components:", self.cc_result['num_components'])      # temp
        
        if self.cc_result is None:
            return

        # Keep exactly what the dialog was showing, minus whatever was deleted.
        # Components hidden by the min-size filter were never visible and the
        # preview already showed them gone, so OK means "make the volume match
        # what I'm looking at" - otherwise they survive in cc_result and come
        # straight back on the next Measurements run.

        keep_labels = self._cc_visible_labels - set(deleted_labels)

        labels = self.cc_result['labels']
        keep = np.zeros(int(labels.max()) + 1, dtype=bool)
        if keep_labels:
            keep[np.fromiter(keep_labels, dtype=int)] = True

        self.cc_result['labels'] = np.where(keep[labels], labels, 0)
        self.cc_result['table'] = self.cc_result['table'][
            self.cc_result['table']['label'].isin(keep_labels)
        ].reset_index(drop=True)
        self.cc_result['num_components'] = len(keep_labels)
        self._cc_visible_labels = keep_labels

        # The surviving components become the new working image, so Minkowski,
        # SMDs, REV and a fresh Connected Components run all see what's left.
        self.current_image_data = (self.cc_result['labels'] > 0).astype(np.uint8)
        self._display_min = float(self.current_image_data.min())
        self._display_max = float(self.current_image_data.max())

        self._refresh_cc_display()
        self.status_bar.showMessage(
            f"Kept {len(keep_labels)} component(s) - image updated"
        )
        
        
        # print("  AFTER   table rows:", len(self.cc_result['table']),
        #       "| image fg:", int(self.current_image_data.sum()),
        #       "| num_components:", self.cc_result['num_components'])

    def on_measurements_error(self, error_msg: str):
        """Handle error during the measurement step."""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Measurements Error", f"Error computing measurements:\n{error_msg}")
        self.status_bar.showMessage(f"Error: {error_msg}")














    
    ##------------------------------polytope calculation--------------------------------------------
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
        # ndim of the data the thread actually computed on (2D image or 3D volume) -> r axis in pixels or voxels
        polytope_window = PolytopePlotWindow(raw_curves, scaled_curves, self,
                                             ndim=self.polytope_thread.image_data.ndim)
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

        is_time_series = self.data_mode in ('time_series', '4d_time_series', '4d_time_series_streaming')
        axis_label = "time step" if is_time_series else "slice"

        streaming = self.data_mode == '4d_time_series_streaming'
        if streaming:
            n_slices = len(self.current_volume_file_paths)
            is_3d = True  # streaming mode only exists for "many 3D volumes"
        else:
            n_slices = self.current_image_data.shape[0]
            is_3d = self.current_image_data.ndim == 4  # Each time step is itself a 3D volume


        dialog = SliceEvolutionSettingsDialog(n_slices, axis_label=axis_label, stack_labels=self.stack_labels, is_3d = is_3d, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return  # user cancelled
        
        step = dialog.get_step()
        total = len(range(0, n_slices, step))  # same count regardless of direction

        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(0)
        self._evolution_start_time = time.perf_counter()
        self._evolution_description = "Calculating SMD evolution"
        self.status_bar.showMessage(f"{self._evolution_description}: 0/{total}")
        QApplication.processEvents()

         # Stash the axis label on self so on_slice_evolution_finished (below) can use it too -
        # it only gets the thread's signal arguments, not the dialog itself

        self._evolution_axis_label = axis_label
        # Same idea for the r-axis unit: each slice/time step is a 3D volume (voxels) or a 2D image (pixels)
        self._evolution_is_3d = is_3d
        # "Average" results tab only for the Z-slices of one 3D volume: the mean over slices
        # estimates the volume's 2D statistics, while a mean over time steps would mix states
        self._evolution_show_average = not is_time_series

        if streaming:
            self.evolution_thread = SliceEvolutionStreamingThread(
                self.current_volume_file_paths, dialog.get_selected_polytopes(), dialog.get_step(),
                reference_index=dialog.get_reference_index(),
                compute_omega=dialog.get_compute_omega(),
                compute_delta_omega=dialog.get_compute_delta_omega(),
                signed_delta_omega=dialog.get_signed_delta_omega(),
                stack_labels=self.stack_labels,
                reverse = dialog.get_reverse_direction(),
            )
        else:
            self.evolution_thread = SliceEvolutionThread(
                self.current_image_data, dialog.get_selected_polytopes(), dialog.get_step(),
                reference_index=dialog.get_reference_index(),
                compute_omega=dialog.get_compute_omega(),
                compute_delta_omega=dialog.get_compute_delta_omega(),
                signed_delta_omega=dialog.get_signed_delta_omega(),
                stack_labels=self.stack_labels,
                reverse = dialog.get_reverse_direction(),
            )
        self.evolution_thread.progress.connect(self.on_evolution_progress)
        self.evolution_thread.finished.connect(self.on_slice_evolution_finished)
        self.evolution_thread.error.connect(self.on_slice_evolution_error)
        self.evolution_thread.start()

    def on_slice_evolution_finished(self, slice_indices, raw_curves_list, scaled_curves_list, omega_dict, delta_omega_dict):

        """Handle completion of the slice/time evolution calculation - open the results window."""
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximum(0)  # restore indeterminate default for other calculations

        window = SliceEvolutionPlotWindow(
            slice_indices, raw_curves_list, scaled_curves_list, omega_dict, delta_omega_dict,
            axis_label=self._evolution_axis_label, parent=self,
            ndim=3 if self._evolution_is_3d else 2,
            show_average=self._evolution_show_average,
        )
        self.plot_windows.append(window)
        window.show()

        self.status_bar.showMessage(f"{self._evolution_axis_label.title()} evolution calculation completed")

    def on_slice_evolution_error(self, error_msg: str):
        """Handle error during the slice/time evolution calculation."""
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximum(0)
        QMessageBox.critical(self, "Calculation Error", f"Error calculating evolution:\n{error_msg}")
        self.status_bar.showMessage(f"Error: {error_msg}")

    def open_chord_length_dialog(self):
        """Open the Chord Length settings dialog and, on OK, run the calculation."""

        if self.current_image_data is None:
            QMessageBox.warning(self, "No Image", "Please open an image first.")
            return
        # Chord Length evolution across a stack isn't built yet - for now, just refuse
        # rather than silently doing the wrong thing on a stack. This gets swapped for a
        # redirect to open_chord_length_evolution_dialog() once that piece exists.
        if self.data_mode in ('time_series', '4d_time_series', '4d_time_series_streaming'):
            self.open_chord_length_evolution_dialog()
            return
        
        # compute_chord_length_result only supports 2D or 3D - same guard as open_smd_dialog.
        if self.current_image_data.ndim not in (2, 3):
            QMessageBox.warning(self, "Invalid Image", "Image must be 2D or 3D.")
            return
        
        dialog = ChordLengthSettingsDialog(parent = self)

        if dialog.exec() != QDialog.Accepted:
            return  # user cancelled

        # Show progress bar in status bar - same pattern as open_smd_dialog
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage("Calculating chord length...")
        QApplication.processEvents()  # force the UI to repaint before the (blocking) thread start

        # Keep the thread on self - if it were a local variable, Python could garbage-collect
        # it mid-run since nothing else holds a reference, silently killing the calculation.
        self.chord_length_thread = ChordLengthThread(
            self.current_image_data,
            dialog.get_num_points(),
            dialog.get_nbins(), 
        )
        self.chord_length_thread.finished.connect(self.on_chord_length_finished)
        self.chord_length_thread.error.connect(self.on_chord_length_error)
        self.chord_length_thread.start()

    def on_chord_length_finished(self, result: dict):
        """Handle completion of Chord Length calculation - runs on the GUI thread."""
        self.progress_bar.setVisible(False)

        chord_length_window = ChordLengthPlotWindow(result, self)
        self.plot_windows.append(chord_length_window) # keeps a reference so it isn't garbage-collected
        chord_length_window.show()

        self.status_bar.showMessage("Chord length calculation completed")

    def on_chord_length_error(self, error_msg: str):
        """Handle error during Chord Length calculation."""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Calculation Error", f"Error calculating chord length:\n{error_msg}")
        self.status_bar.showMessage(f"Error: {error_msg}")


    def open_chord_length_evolution_dialog(self):
        """Configure and run per-slice/per-time-step Chord Length calculation on the current stack."""

        if self.current_image_data is None:
            QMessageBox.warning(self, "No Image", "Please open an image first.")
            return

        if self.current_image_data.ndim not in (3, 4):
            QMessageBox.warning(self, "Invalid Image",
                                 "This requires a 3D volume or an imported volume time series.")
            return

        axis_label = ("time step" if self.data_mode in
                      ('time_series', '4d_time_series', '4d_time_series_streaming') else "slice")

        streaming = self.data_mode == '4d_time_series_streaming'
        n_slices = len(self.current_volume_file_paths) if streaming else self.current_image_data.shape[0]


        dialog = ChordLengthEvolutionSettingsDialog(n_slices, axis_label = axis_label, parent = self)
        if dialog.exec() != QDialog.Accepted:
            return  # user cancelled

        step = dialog.get_step()
        total = len(range(0, n_slices, step))  # same count regardless of direction

        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(0)
        self._evolution_start_time = time.perf_counter()
        self._evolution_description = "Calculating chord length evolution"
        self.status_bar.showMessage(f"{self._evolution_description}: 0/{total}")
        QApplication.processEvents()

        # Stash the axis label on self so on_chord_length_evolution_finished (below) can use
        # it too - same reason open_slice_evolution_dialog does this for _evolution_axis_label.
        self._chord_length_axis_label = axis_label

        if streaming:
            self.chord_length_evolution_thread = ChordLengthEvolutionStreamingThread(
                self.current_volume_file_paths,
                dialog.get_step(),
                dialog.get_num_points(),
                dialog.get_nbins(),
                stack_labels= self.stack_labels,
                reverse = dialog.get_reverse_direction()
            )

        else:
            self.chord_length_evolution_thread = ChordLengthEvolutionThread(
                self.current_image_data,
                dialog.get_step(),
                dialog.get_num_points(),
                dialog.get_nbins(),
                stack_labels= self.stack_labels,
                reverse = dialog.get_reverse_direction()
            )

        self.chord_length_evolution_thread.progress.connect(self.on_evolution_progress)
        self.chord_length_evolution_thread.finished.connect(self.on_chord_length_evolution_finished)
        self.chord_length_evolution_thread.error.connect(self.on_chord_length_evolution_error)
        self.chord_length_evolution_thread.start()

    def on_chord_length_evolution_finished(self, slice_indices, results_list):
        """Handle completion of the chord length evolution calculation - open the results window."""
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximum(0)  # restore indeterminate default for other calculations

        window = ChordLengthEvolutionPlotWindow(
            slice_indices, results_list, axis_label = self._chord_length_axis_label, parent=self
        )

        self.plot_windows.append(window)
        window.show()

        self.status_bar.showMessage(f"Chord length {self._chord_length_axis_label} evolution calculation completed")

    def on_chord_length_evolution_error(self, error_msg: str):
        """Handle error during the chord length evolution calculation."""
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximum(0)
        QMessageBox.critical(self, "Calculation Error", f"Error calculating chord length evolution:\n{error_msg}")
        self.status_bar.showMessage(f"Error: {error_msg}")


    def open_minkowski_evolution_dialog(self):
        """Configure and run per-slice/per-time-step Minkowski functionals calculation on the current stack."""

        if self.current_image_data is None:
            QMessageBox.warning(self, "No Image", "Please open an image first.")
            return

        streaming = self.data_mode == '4d_time_series_streaming' 

        if not streaming and self.current_image_data.ndim not in (3, 4):
            QMessageBox.warning(self, "Invalid Image",
                                "This requires a 3D volume or an imported volume time series.")
            return

        axis_label = ("time step" if self.data_mode in 
                      ('time_series', '4d_time_series', '4d_time_series_streaming') else "slice")
        if streaming:
            n_slices = len(self.current_volume_file_paths)
            is_3d = True # streaming mode only exists for "many 3D volumes"
        else:
            n_slices = self.current_image_data.shape[0]
            is_3d = self.current_image_data.ndim == 4  # each entry is a 3D volume, not a 2D slice

        dialog = MinkowskiEvolutionSettingsDialog(n_slices, axis_label=axis_label, is_3d=is_3d, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return  # user cancelled

        step = dialog.get_step()
        total = len(range(0, n_slices, step))  # same count regardless of direction

        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(0)
        self._evolution_start_time = time.perf_counter()
        self._evolution_description = "Calculating Minkowski functionals evolution"
        self.status_bar.showMessage(f"{self._evolution_description}: 0/{total}")
        QApplication.processEvents()

        self._minkowski_evolution_axis_label = axis_label
        self._minkowski_evolution_unit = dialog.get_unit()
        self._minkowski_evolution_is_3d = is_3d

        if streaming:
            self.minkowski_evolution_thread = MinkowskiEvolutionStreamingThread(
                self.current_volume_file_paths,
                dialog.get_step(),
                dialog.get_resolution(),
                stack_labels=self.stack_labels,
                reverse=dialog.get_reverse_direction(),
            )

        else:
            self.minkowski_evolution_thread = MinkowskiEvolutionThread(
            self.current_image_data,
            dialog.get_step(),
            dialog.get_resolution(),
            is_3d,
            stack_labels=self.stack_labels,
            reverse=dialog.get_reverse_direction(),

            )

        self.minkowski_evolution_thread.progress.connect(self.on_evolution_progress)
        self.minkowski_evolution_thread.finished.connect(self.on_minkowski_evolution_finished)
        self.minkowski_evolution_thread.error.connect(self.on_minkowski_evolution_error)
        self.minkowski_evolution_thread.start()

    def on_minkowski_evolution_finished(self, slice_indices, results_list):
        """Handle completion of the Minkowski functionals evolution calculation - open the results window."""
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximum(0)  # restore indeterminate default for other calculations

        window = MinkowskiEvolutionPlotWindow(
            slice_indices, results_list, self._minkowski_evolution_unit, self._minkowski_evolution_is_3d,
            axis_label=self._minkowski_evolution_axis_label, parent=self
        )
        self.plot_windows.append(window)
        window.show()

        self.status_bar.showMessage(
            f"Minkowski functionals {self._minkowski_evolution_axis_label} evolution calculation completed"
        )

    def on_minkowski_evolution_error(self, error_msg: str):
            
            """Handle error during the Minkowski functionals evolution calculation."""
            self.progress_bar.setVisible(False)
            self.progress_bar.setMaximum(0)
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

    def on_evolution_progress(self, current: int, total: int):
        """Shared progress handler for every evolution-style calculation (Minkowski, SMD,
        chord length - streaming or not), for the slice-by-slice center-crop search, and
        for File > Save As (pages / files written).
        Each thread's `progress` signal connects here;
        self._evolution_description (set right before that thread starts) says which
        calculation is actually running."""

        self.progress_bar.setValue(current)

        elapsed = time.perf_counter() - self._evolution_start_time
        eta_text = ""
        if current > 0:
            remaining = elapsed / current * (total - current)
            eta_text = f" - about {remaining:.0f}s remaining"

        self.status_bar.showMessage(f"{self._evolution_description}: {current}/{total}{eta_text}")
    
