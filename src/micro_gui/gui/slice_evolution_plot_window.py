"""
Plot window for displaying how 2D correlation functions (and their Omega/Delta-Omega
evolution metrics) change across a 3D stack (Z-slices of a volume, or time steps of an
imported time series).
"""


import csv
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from PySide6.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QTabWidget, QWidget, QVBoxLayout

class SliceEvolutionPlotWindow(QMainWindow):

    """
    One tab per selected polytope function (raw + scaled curves over time/slice, side by
    side, colored by slice/time index), plus one "Omega" tab (Omega and Delta-Omega, each
    with one line per function, side by side).
    """

    def __init__(self, slice_indices, raw_curves_list, scaled_curves_list,
                 omega_dict, delta_omega_dict, axis_label="slice", parent=None):
        super().__init__(parent)

        self.setWindowTitle(f"{axis_label.title()} Evolution Results")
        self.setGeometry(150, 150, 1100, 700)

        # Just store everything the thread sent us - the tab-building methods below read
        # straight from these.
        self.slice_indices = slice_indices
        self.raw_curves_list = raw_curves_list
        self.scaled_curves_list = scaled_curves_list
        self.omega_dict = omega_dict
        self.delta_omega_dict = delta_omega_dict
        self.axis_label = axis_label  # "slice" or "time step", used in titles/labels below

        # every slice's curve dict has the same keys (same functions were requested for all
        # of them), so we can just look at the first one to know which functions to build tabs for
        self.polytope_names = list(raw_curves_list[0].keys())

        # dict of {tab_key: matplotlib Figure} - lets Save/Export know which figure belongs
        # to whichever tab is currently active, without re-building anything
        self._figures = {}

        # tab index -> key into self._figures/self.raw_curves_list dicts, in the same order
        # tabs get added below. Looking this up by POSITION (see _current_tab_key) avoids
        # having to reconstruct the key from the tab's display text, which broke for 'L'
        # (lineal path): its real key is uppercase 'L', but the tab is labelled "L" too, and
        # "L".lower() gives 'l' - which doesn't match the actual dict key.
        self._tab_keys = []

        self.tabs = QTabWidget()

        # one tab per function, e.g. "S2", "C2", "P4", ...
        for name in self.polytope_names:
            widget = QWidget()
            layout = QVBoxLayout()
            layout.addWidget(self._build_function_tab(name))
            widget.setLayout(layout)
            self.tabs.addTab(widget, name.upper())
            self._tab_keys.append(name)

        # one extra tab for the Omega/Delta-Omega plots, only if at least one was computed
        if self.omega_dict or self.delta_omega_dict:
            omega_widget = QWidget()
            omega_layout = QVBoxLayout()
            omega_layout.addWidget(self._build_omega_tab())
            omega_widget.setLayout(omega_layout)
            self.tabs.addTab(omega_widget, "Omega")
            self._tab_keys.append('omega')

        self.setCentralWidget(self.tabs)
        self._create_menu()

    def _build_function_tab(self, name):
        """One function's tab: raw curves (left) and scaled curves (right), both colored by slice/time index."""
        fig, (ax_raw, ax_scaled) = plt.subplots(1, 2, figsize=(16, 5), constrained_layout = True)

        # cmap + norm together turn a slice/time index into a color - this is what makes the
        # colorbar (added below) meaningful: same mapping used for both the lines and the bar
        cmap = plt.cm.viridis
        norm = Normalize(vmin=min(self.slice_indices), vmax=max(self.slice_indices))

        # left subplot: raw Pn(r) curve for this function, one line per slice/time step
        for idx, curves in zip(self.slice_indices, self.raw_curves_list):
            curve = curves[name]  # curve is a 2-column [r, value] array
            # P3H/P3V only compute even r (odd r is always exactly 0 by construction - see
            # compute_p3h_polytope/compute_p3v_polytope) - skip those zero placeholders so the
            # line doesn't zigzag down to 0 every other point.
            if name in ('p3h', 'p3v'):
                ax_raw.plot(curve[::2, 0], curve[::2, 1], color=cmap(norm(idx)), linewidth=1.2, alpha=0.85)

            else:
                ax_raw.plot(curve[:, 0], curve[:, 1], color=cmap(norm(idx)), linewidth=1.2, alpha=0.85)
        ax_raw.set_title(f"{name.upper()} (raw)")
        ax_raw.set_xlabel("r")
        ax_raw.set_ylabel("Pn(r)")
        ax_raw.grid(alpha=0.3)

        # right subplot: same thing, but the scaled fn(r) version
        for idx, curves in zip(self.slice_indices, self.scaled_curves_list):
            curve = curves[name]
            if name in ('p3h', 'p3v'):
                ax_scaled.plot(curve[::2, 0], curve[::2, 1], color=cmap(norm(idx)), linewidth=1.2, alpha=0.85)
            else:
                ax_scaled.plot(curve[:, 0], curve[:, 1], color=cmap(norm(idx)), linewidth=1.2, alpha=0.85)
        ax_scaled.set_title(f"{name.upper()} (scaled)")
        ax_scaled.set_xlabel("r")
        ax_scaled.set_ylabel("fn(r)")
        ax_scaled.grid(alpha=0.3)

        # one shared colorbar for both subplots, showing what each line's color means
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # required by matplotlib even though we're not using an image
        fig.colorbar(sm, ax=[ax_raw, ax_scaled], label=self.axis_label.title(), shrink=0.8)

        self._figures[name] = fig  # remember this figure, keyed by function name, for Save/Export later
        return FigureCanvas(fig)
    
    def _build_omega_tab(self):
        """Omega (left) and Delta-Omega (right), each with one line per selected function."""
        fig, (ax_omega, ax_delta) = plt.subplots(1, 2, figsize=(16, 5), constrained_layout = True)

        # left subplot: Omega vs slice/time index, one line per function (only the functions
        # that actually have Omega computed - some might have been skipped if the "Compute
        # Omega" checkbox was off)
        for name in self.polytope_names:
            if name in self.omega_dict:
                ax_omega.plot(self.slice_indices, self.omega_dict[name], marker='o', markersize=3, label=name.upper())
        ax_omega.set_title("Omega (vs. reference slice)")
        ax_omega.set_xlabel(self.axis_label.title())
        ax_omega.set_ylabel("Omega")
        ax_omega.grid(alpha=0.3)
        if self.omega_dict:
            ax_omega.legend()  # only a handful of functions, so a normal legend (not a colorbar) works fine here

        # right subplot: same idea, for Delta-Omega
        for name in self.polytope_names:
            if name in self.delta_omega_dict:
                ax_delta.plot(self.slice_indices, self.delta_omega_dict[name], marker='o', markersize=3, label=name.upper())
        ax_delta.set_title("Delta-Omega (vs. previous slice)")
        ax_delta.set_xlabel(self.axis_label.title())
        ax_delta.set_ylabel("Delta-Omega")
        ax_delta.grid(alpha=0.3)
        if self.delta_omega_dict:
            ax_delta.legend()

        self._figures['omega'] = fig
        return FigureCanvas(fig)
    
    def _create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        save_action = file_menu.addAction("&Save Current Tab Plot...")
        save_action.triggered.connect(self.save_plots)
        export_action = file_menu.addAction("&Export Data as CSV...")
        export_action.triggered.connect(self.export_csv)

    def _current_tab_key(self):
        """Map whichever tab is currently showing back to a key in self._figures, by position (see self._tab_keys)."""
        return self._tab_keys[self.tabs.currentIndex()]

    def save_plots(self):
        """Save whichever tab's figure is currently active."""
        key = self._current_tab_key()
        fig = self._figures[key]
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", f"{self.axis_label}_evolution_{key}",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;PDF Document (*.pdf);;All Files (*)"
        )
        if file_path:
            try:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Plot saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")

    def export_csv(self):
        """Export whichever tab is currently active - the Omega tab and the per-function tabs have different columns."""
        key = self._current_tab_key()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", f"{self.axis_label}_evolution_{key}", "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return
        if not file_path.endswith('.csv'):
            file_path += '.csv'

        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')

                if key == 'omega':
                    # one row per slice/time index, one column per function's Omega and Delta-Omega
                    writer.writerow([self.axis_label] + [f"omega_{n}" for n in self.omega_dict] + [f"delta_omega_{n}" for n in self.delta_omega_dict])
                    for i, idx in enumerate(self.slice_indices):
                        row = [idx]
                        row += [np.round(self.omega_dict[n][i], 6) for n in self.omega_dict]
                        row += [np.round(self.delta_omega_dict[n][i], 6) for n in self.delta_omega_dict]
                        writer.writerow(row)
                else:
                    # one row per (slice/time, r) pair, for the currently active function's tab
                    writer.writerow([self.axis_label, 'r', 'raw', 'scaled'])
                    for idx, raw_curves, scaled_curves in zip(self.slice_indices, self.raw_curves_list, self.scaled_curves_list):
                        raw_curve = raw_curves[key]
                        scaled_curve = scaled_curves[key]
                        for i, r in enumerate(raw_curve[:, 0]):
                            writer.writerow([idx, int(r), np.round(raw_curve[i, 1], 4), np.round(scaled_curve[i, 1], 4)])
            QMessageBox.information(self, "Success", f"Data exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export CSV:\n{str(e)}")




