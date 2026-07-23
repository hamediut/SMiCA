import csv
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtWidgets import (QMainWindow, QFileDialog, QMessageBox, QTabWidget, QWidget, QVBoxLayout,
                               QHBoxLayout, QListWidget, QListWidgetItem, QAbstractItemView,
                               QLabel, QSpinBox)

from PySide6.QtCore import Qt


from .save_dialog_helper import suggested_save_path, remember_save_dir

class ChordLengthEvolutionPlotWindow(QMainWindow):
    """
    "Mean Chord Length" tab: ell_via_s2 and ell_direct vs slice/time index - always shows
    every computed slice, mirrors SliceEvolutionPlotWindow's Omega tab.

    A "Chord Length Distribution" tab (interactive slice picker) gets added in the next step.
    """

    MAX_CHORD_LENGTH_DISPLAY = 100  # pixels/voxels - adjust if your images typically have longer/shorter chords

    def __init__(self, slice_indices, results_list, axis_label = "slice", parent = None):
        super().__init__(parent)

        self.setWindowTitle(f"Chord Length {axis_label.title()} Evolution Results")
        self.setGeometry(150, 150, 1000, 650)

        # Just store what the thread sent us - same as SliceEvolutionPlotWindow.
        self.slice_indices = slice_indices
        self.results_list = results_list
        self.axis_label = axis_label

        # {tab_key: matplotlib Figure} - same reason as SliceEvolutionPlotWindow: lets
        # Save/Export find the right figure for whichever tab is active, by key not position.
        self._figures = {}
        self._tab_keys = []   # tab index -> key, in the order tabs get added below

        self.tabs = QTabWidget()

        mean_widget = QWidget()
        mean_layout = QVBoxLayout()
        mean_layout.addWidget(self._build_mean_tab())
        mean_widget.setLayout(mean_layout)
        self.tabs.addTab(mean_widget, "Mean Chord Length")
        self._tab_keys.append('mean')

        distribution_widget = self._build_distribution_tab()
        self.tabs.addTab(distribution_widget, "Chord Length Distribution")
        self._tab_keys.append('distribution')

        self.setCentralWidget(self.tabs)
        self._create_menu()


    def _build_mean_tab(self):
        """ell_via_s2 and ell_direct vs. slice/time index - one line each, both methods together."""

        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

        # Pull one number out of every slice's result dict - this is exactly the same
        # "list of dicts -> one list per field" extraction _build_omega_tab does with
        # self.omega_dict[name], just without a per-function loop since there's only ever
        # one thing (chord length) here, not several functions to choose from.

        ell_via_s2 = [r['ell_via_s2'] for r in self.results_list]
        ell_direct = [r['ell_direct'] for r in self.results_list]

        ax.plot(self.slice_indices, ell_via_s2, marker='o', label='via S2')
        ax.plot(self.slice_indices, ell_direct, marker='s', label='direct')
        ax.set_title(f"Mean Chord Length vs. {self.axis_label.title()}")
        ax.set_xlabel(self.axis_label.title())
        ax.set_ylabel("Mean chord length")
        ax.grid(alpha=0.3)
        ax.legend()

        self._figures['mean'] = fig
        return FigureCanvas(fig)

    def _build_distribution_tab(self):
        """QListWidget picker (left) + a max-range control above the matplotlib axes (right),
        both redrawing on change."""

        widget = QWidget()
        layout = QHBoxLayout()

        #----picker----
        self.slice_list = QListWidget()
        self.slice_list.setSelectionMode(QAbstractItemView.ExtendedSelection) # ctrl/shift-click multi-select

        for i, idx in enumerate(self.slice_indices):
            item = QListWidgetItem(f"{self.axis_label.title()} {idx}")
            item.setData(Qt.UserRole, i)   # store the RESULTS_LIST position, not the displayed slice number -
                                             # these differ whenever stack_labels renumbers the display
            self.slice_list.addItem(item)

        # --- right side: a max-range control stacked above the plot ---
        right_layout = QVBoxLayout()

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Max chord length to display:"))
        self.max_display_spinbox = QSpinBox()
        self.max_display_spinbox.setRange(1, 100000)
        # MAX_CHORD_LENGTH_DISPLAY is now just the STARTING value shown here - the user can
        # change it live from this spinbox, that's the whole point of this control.
        self.max_display_spinbox.setValue(self.MAX_CHORD_LENGTH_DISPLAY)
        self.max_display_spinbox.setToolTip(
            "Chord lengths beyond this are excluded from the histogram (not just cropped from "
            "the view) - lowering it gives more bin resolution to the meaningful part of the "
            "distribution instead of wasting it on a long, sparse tail."
        )
        range_row.addWidget(self.max_display_spinbox)
        range_row.addStretch()  # keeps the label+spinbox pinned to the left instead of spreading out
        right_layout.addLayout(range_row)

        # --- the plot: created BEFORE connecting either signal below, so self.ax_dist already
        # exists by the time the default-selection lines trigger the first redraw ---
        self.fig_dist, self.ax_dist = plt.subplots(figsize=(8, 6), constrained_layout=True)
        self._figures['distribution'] = self.fig_dist
        canvas = FigureCanvas(self.fig_dist)
        right_layout.addWidget(canvas)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        layout.addWidget(self.slice_list, stretch=1)
        layout.addWidget(right_widget, stretch=3)

        self.slice_list.itemSelectionChanged.connect(self._redraw_distribution)
        self.max_display_spinbox.valueChanged.connect(self._redraw_distribution)

         # Pre-select first/middle/last so the tab isn't blank when it first opens. Each
        # setSelected(True) call fires itemSelectionChanged immediately, so _redraw_distribution
        # runs a few times here during setup - harmless, just a touch redundant.
        n = len(self.slice_indices)
        for row in sorted({0, n // 2, n - 1}):
            self.slice_list.item(row).setSelected(True)

        widget.setLayout(layout)
        return widget
    
    def _redraw_distribution(self):
        """Clear and redraw the distribution axes from whatever's currently selected."""
        self.ax_dist.clear()

        selected_items = self.slice_list.selectedItems()
        n_selected = len(selected_items)

        # Nothing selected (e.g. ctrl-clicking the last selected item to deselect it) - draw an
        # empty axes rather than an empty rwidths list feeding into the loop below for nothing.
        if n_selected == 0:
            self.fig_dist.canvas.draw_idle()
            return

        nbins = len(self.results_list[0]['z'])   # reuse whatever bin count you configured in the dialog
        max_display = self.max_display_spinbox.value()   # live value from the spinbox, not the class constant
        bin_width = max_display / nbins

        # "Telescoping" widths, like plt.hist's rwidth: every selected slice shares the SAME
        # bin centers (no offset), but each one gets a narrower bar than the last, from 0.9x
        # down to 0.3x of the bin width. Drawn in that same order (widest first), each
        # narrower bar ends up layered ON TOP of the previous, wider one - that's what
        # creates the nested/concentric look, combined with alpha making the ones underneath
        # still show through.
        if n_selected == 1:
            rwidths = [0.9]
        else:
            rwidths = np.linspace(0.9, 0.3, n_selected)

        for color_i, item in enumerate(selected_items):
            i = item.data(Qt.UserRole)          # position into self.results_list
            result = self.results_list[i]

            # Re-histogram from the raw chords over a SHARED range/bin-count - unlike
            # result['z']/result['p_z'], this guarantees every selected slice lands on the
            # exact same bin positions, which is what makes the nested bars below line up.
            p_z, edges = np.histogram(result['chords'], bins=nbins, range=(0, max_display), density=True)
            z = 0.5 * (edges[:-1] + edges[1:])

            self.ax_dist.bar(
                z, p_z, width=rwidths[color_i] * bin_width,
                alpha=0.6, edgecolor='k',
                color=plt.cm.tab10(color_i % 10),  # a small discrete palette - this is a hand-picked
                                                     # SUBSET, unlike the viridis-gradient-over-everything
                                                     # convention used for the "all slices" plots elsewhere
                label=f"{self.axis_label} {self.slice_indices[i]}",
            )

        unit = 'pixels' if self.results_list[0]['ndim'] == 2 else 'voxels'
        self.ax_dist.set_title("Chord-Length Distribution (selected slices)")
        self.ax_dist.set_xlabel(f"Chord Length ({unit})")
        self.ax_dist.set_ylabel("p(z)")
        self.ax_dist.set_xlim(0, max_display)
        self.ax_dist.grid(alpha=0.3)
        self.ax_dist.legend()

        self.fig_dist.canvas.draw_idle()   # tell Qt to actually repaint - plot()/clear() alone don't



    def _create_menu(self):

        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")

        save_action = file_menu.addAction("&Save Current Tab Plot")
        save_action.triggered.connect(self.save_plots)

        export_action = file_menu.addAction("&Export Data as CSV")
        export_action.triggered.connect(self.export_csv)

    def _current_tab_key(self):
        return self._tab_keys[self.tabs.currentIndex()]

    def closeEvent(self, event):
        """Release every figure this window created - see SliceEvolutionPlotWindow.closeEvent
        for why this matters (matplotlib's global figure registry doesn't know a Qt window closed)."""
        for fig in self._figures.values():
            plt.close(fig)
        super().closeEvent(event)

    def save_plots(self):
        key = self._current_tab_key()
        fig = self._figures[key]

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot",
            suggested_save_path(f"chord_length_{self.axis_label}_evolution_{key}"),
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;PDF Document (*.pdf);;All Files (*)"
        )

        if file_path:

            try:
                fig.savefig(file_path, dpi = 500, bbox_inches='tight')
                remember_save_dir(file_path)
                QMessageBox.information(self, "Success", f"Plot saved to:\n{file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")

    def export_csv(self):
        key = self._current_tab_key()
        file_path, _ = QFileDialog.getSaveFileName(

            self, "Export CSV",
            suggested_save_path(f"chord_length_{self.axis_label}_evolution_{key}"),
            "CSV Files (*.csv);;All Files (*)"
        )

        if not file_path:
            return
        if not file_path.endswith('.csv'):
            file_path += '.csv'

        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                # only 'mean' exists so far - the distribution tab's branch gets added
                # to this same if/else next step, same shape as SliceEvolutionPlotWindow's
                # export_csv (which branches on 'omega' vs. everything else).
                if key == 'mean':
                    writer.writerow([self.axis_label, 'ell_via_s2', 'ell_direct'])
                    for idx, result in zip(self.slice_indices, self.results_list):
                        writer.writerow([idx, np.round(result['ell_via_s2'], 4), np.round(result['ell_direct'], 4)])
                else:
                    # only the currently SELECTED slices - matches what's actually drawn
                    writer.writerow([self.axis_label, 'z', 'p_z'])
                    for item in self.slice_list.selectedItems():
                        i = item.data(Qt.UserRole)
                        result = self.results_list[i]
                        for z_val, p_val in zip(result['z'], result['p_z']):
                            writer.writerow([self.slice_indices[i], np.round(z_val, 4), np.round(p_val, 6)])

            remember_save_dir(file_path)
            QMessageBox.information(self, "Success", f"Data exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export CSV:\n{str(e)}")

###===================
#Two things worth noticing:

# item.setData(Qt.UserRole, i) / item.data(Qt.UserRole) is how you attach arbitrary Python data to a Qt list item
#  — Qt.UserRole is just a reserved slot Qt promises never to use for its own built-in item data,
#  so it's safe for you to store whatever you want there.


# _redraw_distribution is connected once, in _build_distribution_tab,
#  but gets called every time the selection changes for the rest of the window's life —
#  that's the whole "interactive, no recomputation" behavior you asked for at the start: 
# self.results_list already has every slice's PDF sitting in memory, so toggling the picker only re-draws, 
# it never re-runs compute_chord_length_result.

        











        