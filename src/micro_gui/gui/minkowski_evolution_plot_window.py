"""
Plot window for displaying how Minkowski functionals change across a stack
(Z-slices of a volume, or time steps of an imported time series).
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QTabWidget, QWidget, QVBoxLayout

from .save_dialog_helper import suggested_save_path, remember_save_dir

class MinkowskiEvolutionPlotWindow(QMainWindow):
    """
    One tab per functional - raw value (left) and specific/density value
    (right), both vs. slice/time index.
    """

    _TABS_2D = [
        ('area', 'area_fraction', 'Area', 'Area Fraction', 2, None),
        ('perimeter', 'specific_perimeter', 'Perimeter', 'Specific Perimeter', 1, -1),
        ('euler', 'specific_euler', 'Euler Characteristic', 'Specific Euler Characteristic', None, -2),
    ]

    _TABS_3D = [
        ('volume', 'porosity', 'Volume', 'Porosity', 3, None),
        ('surface_area', 'specific_surface_area', 'Surface Area', 'Specific Surface Area', 2, -1),
        ('mean_curv', 'specific_mean_curv', 'Mean Curvature', 'Specific Mean Curvature', 1, -2),
        ('euler', 'specific_euler', 'Euler Characteristic', 'Specific Euler Characteristic', None, -3),
    ]

    _SUPERSCRIPTS = {1: '', 2: '\u00b2', 3: '\u00b3', -1: '\u207b\u00b9', -2: '\u207b\u00b2', -3: '\u207b\u00b3'}


    def __init__(self, slice_indices, results_list, unit: str, is_3d: bool, axis_label = "slice", parent = None):

        super().__init__(parent)

        self.setWindowTitle(f"Minkowski Functionals {axis_label.title()} Evolution Results")
        self.setGeometry(150, 150, 1000, 650)

        self.slice_indices = slice_indices
        self.results_list = results_list
        self.unit = unit
        self.axis_label = axis_label
        self.tab_defs = self._TABS_3D if is_3d else self._TABS_2D

        self._figures = {}
        self._tab_keys = []

        self.tabs = QTabWidget()
        for raw_key, specific_key, raw_label, specific_label, raw_exp, specific_exp  in self.tab_defs:
            widget = QWidget()
            layout = QVBoxLayout()
            layout.addWidget(self._build_tab(raw_key, specific_key, raw_label, specific_label, raw_exp, specific_exp))
            widget.setLayout(layout)
            self.tabs.addTab(widget, raw_label)
            self._tab_keys.append(raw_key)

        self.setCentralWidget(self.tabs)
        self._create_menu()

    def _axis_label_with_unit(self, label, exponent):
        """Build 'Surface Area (um^2)', or just 'Porosity' if exponent is None (dimensionless)."""
        if exponent is None:
            return label
        return f"{label} ({self.unit}{self._SUPERSCRIPTS[exponent]})"

    def _build_tab(self, raw_key, specific_key, raw_label, specific_label, raw_exponent, specific_exponent):
        """raw_key vs. specific_key, side by side, both vs. slice/time index."""

        fig, (ax_raw, ax_specific) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        raw_values = [r[raw_key] for r in self.results_list]
        specific_values = [r[specific_key] for r in self.results_list]

        ax_raw.plot(self.slice_indices, raw_values, marker='o')
        ax_raw.set_title(raw_label)
        ax_raw.set_xlabel(self.axis_label.title())
        ax_raw.set_ylabel(self._axis_label_with_unit(raw_label, raw_exponent))
        ax_raw.grid(alpha=0.3)

        ax_specific.plot(self.slice_indices, specific_values, marker='o', color='tab:orange')
        ax_specific.set_title(specific_label)
        ax_specific.set_xlabel(self.axis_label.title())
        ax_specific.set_ylabel(self._axis_label_with_unit(specific_label, specific_exponent))
        ax_specific.grid(alpha=0.3)

        self._figures[raw_key] = fig
        return FigureCanvas(fig)

    def _create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        save_action = file_menu.addAction("&Save Current Tab Plot...")
        save_action.triggered.connect(self.save_plots)
        export_action = file_menu.addAction("&Export All Data as CSV...")
        export_action.triggered.connect(self.export_csv)

    def _current_tab_key(self):
        return self._tab_keys[self.tabs.currentIndex()]

    def closeEvent(self, event):
        """Release every figure this window created - see SliceEvolutionPlotWindow.closeEvent for why."""
        for fig in self._figures.values():
            plt.close(fig)
        super().closeEvent(event)

    def save_plots(self):
        key = self._current_tab_key()
        fig = self._figures[key]
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", suggested_save_path(f"minkowski_{self.axis_label}_evolution_{key}"),
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;PDF Document (*.pdf);;All Files (*)"
        )
        if file_path:
            try:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                remember_save_dir(file_path)
                QMessageBox.information(self, "Success", f"Plot saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")
    def export_csv(self):
        """Every functional gets exported together, one row per slice/time index -
        unlike ChordLengthEvolutionPlotWindow, every tab here shares the exact same
        row shape (one dict per frame), so there's no need to branch on the active tab."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", suggested_save_path(f"minkowski_{self.axis_label}_evolution"),
            "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return
        if not file_path.endswith('.csv'):
            file_path += '.csv'

        keys = list(self.results_list[0].keys())
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow([self.axis_label] + keys)
                for idx, result in zip(self.slice_indices, self.results_list):
                    writer.writerow([idx] + [np.round(result[k], 6) for k in keys])
            remember_save_dir(file_path)
            QMessageBox.information(self, "Success", f"Data exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export CSV:\n{str(e)}")