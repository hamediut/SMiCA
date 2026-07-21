
"""
Plot window for displaying polytope calculation results.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QTabWidget, QWidget, QVBoxLayout

from .save_dialog_helper import suggested_save_path, remember_save_dir


class PolytopePlotWindow(QMainWindow):
    """
    Window to display polytope calculation results.

    Shows two tabs - raw Pn(r) curves and their scaled fn(r) companions - each
    overlaying every selected function on one plot, mirroring RevPlotWindow's
    S2/F2 tab structure.
    """

    def __init__(self, raw_curves: dict, scaled_curves: dict, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Polytope Calculation Results")
        self.setGeometry(150, 150, 1000, 700)

        self.raw_curves = raw_curves
        self.scaled_curves = scaled_curves

        # Create tab widget to show raw and scaled curves separately
        self.tabs = QTabWidget()

        # Raw Curves Tab
        raw_widget = QWidget()
        raw_layout = QVBoxLayout()
        raw_canvas = self._create_raw_plot()
        raw_layout.addWidget(raw_canvas)
        raw_widget.setLayout(raw_layout)
        self.tabs.addTab(raw_widget, "Pn(r) Curves")

        # Scaled Curves Tab
        scaled_widget = QWidget()
        scaled_layout = QVBoxLayout()
        scaled_canvas = self._create_scaled_plot()
        scaled_layout.addWidget(scaled_canvas)
        scaled_widget.setLayout(scaled_layout)
        self.tabs.addTab(scaled_widget, "Fn(r) Curves")

        self.setCentralWidget(self.tabs)

        self._create_menu()

    def _create_raw_plot(self):
         """Create the raw Pn(r) plot, one line per selected function."""

         self.fig_raw, self.ax_raw = plt.subplots(figsize=(10, 6))

         # Why store self.fig_raw/self.ax_raw as instance attributes instead of local variables?
         # save_plots() needs to reach back into whichever figure belongs to the currently active tab
         # same reason RevPlotWindow keeps self.fig_s2/self.fig_f2 around.

         # Each curve is a 2-column [r, value] array (see PolytopeCalculationThread).
         # We don't pick colors explicitly - matplotlib assigns each plot() call
         # the next color in its default cycle automatically, so every curve gets
         # a distinct color for free.

         for name, curve in self.raw_curves.items():

            if name in ['p3h', 'p3v']:
                # For the two triangle functions, plot them every other points as it is zero for every other point, so the line is not continuous and looks weird. This is a temporary fix until we can figure out how to fix the calculation of these two functions.
                self.ax_raw.plot(curve[::2, 0], curve[::2, 1], linewidth=1.5, label=name.upper())
            else:
                self.ax_raw.plot(curve[:, 0], curve[:, 1], linewidth=1.5, label=name.upper())


         self.ax_raw.set_title('Polytope Functions (raw)', fontsize=14, fontweight='bold')
        
         self.ax_raw.set_xlabel('Distance (r)', fontsize=16)
         self.ax_raw.set_ylabel('$P_n(r)$', fontsize=16)
         self.ax_raw.tick_params(axis='both', which='major', labelsize=12)
         self.ax_raw.legend()
         self.ax_raw.grid(True, alpha=0.3)

         return FigureCanvas(self.fig_raw)

    def _create_scaled_plot(self):
        """Create the scaled fn(r) plot, one line per selected function."""
        self.fig_scaled, self.ax_scaled = plt.subplots(figsize=(10, 6))

        for name, curve in self.scaled_curves.items():
            if name in ['p3h', 'p3v']:
                # For the two triangle functions, plot them every other points as it is zero for every other point, so the line is not continuous and looks weird. This is a temporary fix until we can figure out how to fix the calculation of these two functions.
                self.ax_scaled.plot(curve[::2, 0], curve[::2, 1], linewidth=1.5, label=name.upper())
            else:
                self.ax_scaled.plot(curve[:, 0], curve[:, 1], linewidth=1.5, label=name.upper())

        self.ax_scaled.set_title('Polytope Functions (scaled)', fontsize=14, fontweight='bold')
        
        self.ax_scaled.set_xlabel('Distance (r)', fontsize=16)
        self.ax_scaled.set_ylabel('$f_n(r)$', fontsize=16)
        self.ax_scaled.tick_params(axis='both', which='major', labelsize=12)
        self.ax_scaled.legend()
        self.ax_scaled.grid(True, alpha=0.3)

        return FigureCanvas(self.fig_scaled)
    
    def _create_menu(self):
        """Create the menu bar with a File menu for saving plots."""

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')

        save_action = file_menu.addAction('&Save Plot as...')
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self._save_plots)

        export_csv_action = file_menu.addAction('&Export Data as CSV...')
        export_csv_action.setShortcut('Ctrl+E')
        export_csv_action.triggered.connect(self.export_csv)

    def closeEvent(self, event):
        """Release both matplotlib figures (raw + scaled tabs) when this window closes - see
        SliceEvolutionPlotWindow.closeEvent for why this is needed."""
        plt.close(self.fig_raw)
        plt.close(self.fig_scaled)
        super().closeEvent(event)

    def _save_plots(self):
        """Save the currently active tab's plot as PNG, JPEG, or PDF."""

        current_tab_index = self.tabs.currentIndex()

        if current_tab_index == 0:
            current_fig = self.fig_raw
            plot_name = "Raw"
        else:
            current_fig = self.fig_scaled
            plot_name = "Scaled"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Save {plot_name} Plot",
            suggested_save_path(f"polytope_{plot_name.lower()}"),  # opens in the last-used save folder
            "PNG Image (*.png);;JPEG Image (*.jpg);;PDF Document (*.pdf);;All Files (*)"
            )

        if file_path:
            try:
                current_fig.savefig(file_path, dpi=500, bbox_inches='tight')
                remember_save_dir(file_path)  # so the next Save dialog opens in this same folder
                QMessageBox.information(self, "Success", f"{plot_name} plot saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")

    def export_csv(self):
            
            """Export the currently active tab's curves to a single CSV file (one column per function)."""

            current_tab_index = self.tabs.currentIndex()

            if current_tab_index == 0:
                curves = self.raw_curves
                default_name = "polytope_raw_data"
            else:
                curves = self.scaled_curves
                default_name = "polytope_scaled_data"
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export CSV",
                suggested_save_path(default_name),  # opens in the last-used save folder
                "CSV Files (*.csv);;All Files (*)"
            )

            if file_path:
                try:
                    # Ensure .csv extension
                    if not file_path.endswith('.csv'):
                        file_path += '.csv'

                    names = list(curves.keys())
                    r_values = list(curves.values())[0][:, 0] # every curve shares the same r-axis (same image, same Nt)

                    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile, delimiter=';')
                        writer.writerow(['r'] + names)

                        for i, r in enumerate(r_values):
                            row = [int(r)] + [np.round(curves[name][i, 1], decimals=4) for name in names]
                            writer.writerow(row)

                    remember_save_dir(file_path)  # so the next Save dialog opens in this same folder
                    QMessageBox.information(self, "Success", f"Data exported to:\n{file_path}")
                
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to export csv:\n{str(e)}")            


