"""
Plot window for displaying single-image Chord Length results.
"""

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtWidgets import QMainWindow, QFileDialog, QMessageBox

from .save_dialog_helper import suggested_save_path, remember_save_dir

class ChordLengthPlotWindow(QMainWindow):
    """
    Window to display Chord Length results for a single 2D/3D image.

    Unlike RevPlotWindow/PolytopePlotWindow, there's only ONE thing to show here
    (the chord-length distribution of the foreground phase), so no QTabWidget is
    needed - just one FigureCanvas as the central widget.
    """

    def __init__(self, result: dict, parent = None):
        super().__init__(parent)

        self.setWindowTitle("Chord Length Results")
        self.setGeometry(150, 150, 800, 600)

        self.result = result # keep the whole dict - export_csv (add later) will need result['chords']

        canvas = self._create_plot()
        self.setCentralWidget(canvas) # QMainWindow needs exactly ONE central widget - unlike
                                         # QDialog, you can't just addWidget() to a layout here

        self._create_menu()

    def _create_plot(self):


        """Build the histogram + the two methods' mean-chord-length lines."""

        # self.fig/self.ax stored as attributes (not local vars) so save_plot() can reach
        # back into this exact figure later - same reason RevPlotWindow does it.

        self.fig, self.ax = plt.subplots(figsize = (8, 6))

        z = self.result['z']
        p_z = self.result['p_z']
        bin_width = z[1] - z[0]  # np.histogram bins are evenly spaced, so any adjacent gap works

        self.ax.bar(z, p_z, width=bin_width, alpha=0.7, color='C0', label='Chord-length PDF')

        # Two vertical reference lines, one per method, so the two estimates' agreement
        # is visible at a glance against the actual distribution shape.
        self.ax.axvline(self.result['ell_via_s2'], color='red', ls='--', lw=2,
                         label=f"via S2 = {self.result['ell_via_s2']:.2f}")
        self.ax.axvline(self.result['ell_direct'], color='black', ls='--', lw=2,
                         label=f"direct = {self.result['ell_direct']:.2f}")

        unit = 'pixels' if self.result['ndim'] == 2 else 'voxels'
        self.ax.set_xlabel(f'Chord length ({unit})', fontsize=14)
        self.ax.set_ylabel('p(z)', fontsize=14)
        self.ax.set_title(f"Chord Length Distribution (phase={self.result['phase']})",
                           fontsize=14, fontweight='bold')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

        return FigureCanvas(self.fig)
    
    def _create_menu(self):
        """File menu with Save Plot - Export CSV can be added the same way later."""

        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        save_action = file_menu.addAction("&Save plot")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_plot)

    def closeEvent(self, event):
        """Release the matplotlib figure when the window closes - without this, repeatedly
        running this feature leaks figures and matplotlib eventually warns about it."""
        plt.close(self.fig)
        return super().closeEvent(event)
    
    def save_plot(self):
        """Save the plot as PNG, JPEG, or PDF."""

        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Chord Length Plot",
            suggested_save_path("chord_length_plot"), # opens in the last-used save folder
             "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;PDF Document (*.pdf);;All Files (*)"
        )

        if file_path:

            try:
                self.fig.savefig(file_path, dpi = 500, bbox_inches='tight')
                remember_save_dir(file_path) # so the next Save dialog opens in this same folder
                QMessageBox.information(self, "Success", f"Plot saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")

    









