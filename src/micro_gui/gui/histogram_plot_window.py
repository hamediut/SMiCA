"""
Histogram plot window for a single measurement column's values.
"""

import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtWidgets import QMainWindow, QFileDialog, QMessageBox

from .save_dialog_helper import suggested_save_path, remember_save_dir


class HistogramPlotWindow(QMainWindow):
    """
    Histogram of a single measurement column's values, opened from a results
    table's "Histogram" button, after HistogramSettingsDialog collects the
    binning parameters below. Own small File menu to save the plot, same
    pattern as MinkowskiEvolutionPlotWindow.
    """

    def __init__(self, values, column_label: str, percentile: float, num_bins: int, log_y: bool, parent=None):
        super().__init__(parent)

        self.setWindowTitle(f"Histogram: {column_label}")
        self.setGeometry(150, 150, 600, 500)

        values = np.asarray(values, dtype=float)

        # Fixed, bounded bin count spanning [min, percentile cutoff] - not
        # bins='auto' (Freedman-Diaconis-style auto binning can explode into
        # millions of bins when a few extreme outliers make the range huge
        # while most values cluster tightly - e.g. thousands of tiny
        # components alongside one giant percolated one - and matplotlib
        # hangs trying to build/draw that many bars) and not a plain fixed
        # range either (with that same data, 50 bins over the full range
        # still crams almost everything into the first bin or two).
        cutoff = np.percentile(values, percentile)
        if cutoff <= values.min():
            # Degenerate choice (e.g. percentile low enough the cutoff lands
            # at or below the minimum) - fall back to the real max so bin
            # edges aren't zero-width.
            cutoff = values.max()

        n_above = int((values > cutoff).sum())
        # Values above the cutoff fold into the last bin instead of being
        # dropped - clip() maps them exactly to the cutoff, which is the
        # last bin edge.
        clipped_values = np.clip(values, None, cutoff)
        bin_edges = np.linspace(values.min(), cutoff, num_bins + 1)

        self.fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(clipped_values, bins=bin_edges)
        if log_y:
            ax.set_yscale('log')

        ax.set_xlabel(column_label)
        ax.set_ylabel("Number of components")
        ax.set_title(f"Distribution of {column_label} (n={len(values)})")
        ax.grid(alpha=0.3)

        self.canvas = FigureCanvas(self.fig)
        self.setCentralWidget(self.canvas)
        self._create_menu()

        # Status bar instead of an on-plot annotation - keeps Save Plot... exports
        # clean, and the message is still visible without cluttering the figure.
        if n_above > 0:
            self.statusBar().showMessage(
                f"{n_above} value(s) above the {percentile:g}th percentile grouped into the last bin"
            )

    def _create_menu(self):

        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        save_action = file_menu.addAction("&Save Plot...")
        save_action.triggered.connect(self._save_plot)

    def _save_plot(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", suggested_save_path("histogram"),
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;PDF Document (*.pdf);;All Files (*)"
        )

        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                remember_save_dir(file_path)
                QMessageBox.information(self, "Success", f"Plot saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")

    def closeEvent(self, event):
        """Release the figure this window created - see MinkowskiEvolutionPlotWindow.closeEvent for why."""
        plt.close(self.fig)
        super().closeEvent(event)


        
