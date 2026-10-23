"""
Results dialog for connected-components output: one row per component.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QPushButton, QFileDialog, QMessageBox, QMenu, QCheckBox
)
from PySide6.QtCore import Qt, Signal, QItemSelection, QItemSelectionModel

from .save_dialog_helper import suggested_save_path, remember_save_dir

import numpy as np
from .histogram_plot_window import HistogramPlotWindow
from .histogram_settings_dialog import HistogramSettingsDialog


class _NumericTableItem(QTableWidgetItem):
    """Table cell that displays formatted text but sorts on its real number.

    QTableWidgetItem's default comparison is on the displayed string, so a
    numeric column would sort lexicographically - "1000" before "925", because
    "1" < "9". Keeping the raw value alongside the formatted text lets the
    display stay readable (.4g) while sorting stays correct.
    """

    def __init__(self, text, value):
        super().__init__(text)
        self.value = value

    def __lt__(self, other):
        #__lt__ is Python's special method for the < operator.
        if not isinstance(other, _NumericTableItem):
            return super().__lt__(other)

        # NaN compares False against everything, which would leave the order
        # unstable - sink NaNs instead. Plenty of components produce NaN
        # elongation/flatness, so this matters here.

        if np.isnan(self.value):
            return False  # self is NaN, sink it
        if np.isnan(other.value):
            return True  # other is NaN, sink it

        return self.value < other.value  # both are real numbers, compare normally



class ConnectedComponentsResultsDialog(QDialog):
    """
    Read-only results table for a connected-components calculation - one row
    per component, already sorted largest-first by the analysis function.
    Not modal (shown with .show(), not .exec()) so the user can keep it open
    while comparing against the image or running another calculation.
    """

    # The dialog doesn't own the label volume - ImageViewer does. It reports
    # intent and lets the viewer act on it.

    preview_requested = Signal(object) # set of labels to hide, live preview
    changes_committed = Signal(object) # set of deleted labels, on OK
    changes_reverted = Signal()        # on Cancel / window close

    # Column -> unit exponent, for measurements that carry a physical unit
    # (1 = length, -1 = 1/length). A column absent from this dict is treated as
    # dimensionless - no unit suffix shown. List-driven so a future unit-bearing
    # measurement (e.g. surface_area) is one new entry, not new UI code.
    _MEASUREMENT_UNIT_EXPONENTS = {
        'equivalent_diameter': 1,
        'perimeter': 1,
        'specific_perimeter': -1,
        'surface_area': 2,
        'specific_surface_area': -1,
    }

    # Columns whose auto-generated header reads badly ("Inclination Deg"), and whose
    # unit is fixed (degrees) rather than derived from the voxel size.
    _HEADER_OVERRIDES = {
        'inclination_deg': 'Inclination (deg)',
        'azimuth_deg': 'Azimuth (deg)',
    }

    # Maps an exponent to its Unicode superscript suffix, so a unit string can be
    # built like f"{unit}{_SUPERSCRIPTS[exponent]}" -> "um", "um2", "um-1", etc.
    # Negative exponents combine two characters: the superscript minus sign with
    # the superscript digit (1/2/3). No suffix needed for exponent 1 - a plain
    # length reads as "um", not "um1".
    _SUPERSCRIPTS = {1: '', -1: '\u207b\u00b9', 2: '\u00b2', -2: '\u207b\u00b2', 3: '\u00b3', -3: '\u207b\u00b3'}




    def __init__(self, table, unit: str, is_3d: bool, total_components:int, parent = None):
        super().__init__(parent)

        # Keep the DataFrame itself (not just what ends up in the table
        # widget) - _export_csv writes this straight to disk later.

        self.table_data = table
        # connected_components_2d/_3d name their columns differently
        # (pixel_count/area vs voxel_count/volume) - is_3d picks which pair
        # of column names and display labels this dialog should use.

        self.unit = unit
        self.count_col = 'voxel_count' if is_3d else 'pixel_count'
        self.measure_col = 'volume' if is_3d else 'area'
        count_label = 'Voxel count' if is_3d else 'Pixel count'
        measure_label = f"Volume ({unit}\u00b3)" if is_3d else f"Area ({unit}\u00b2)"

        #for plotting histogram of the measure column
        self.selected_column_index = None

        # Deletions are tracked as a set of label ids, not a copy of the label
        # volume - that array is hundreds of MB on a 512^3 dataset, and this
        # makes revert free.

        self.deleted_labels = set()
        self._committed = False # closeEvent uses this to decide revert vs keep

        self.setWindowTitle("Connected Components Results")
        self.setMinimumWidth(700)
        self._setup_ui(total_components, count_label, measure_label)

    def _setup_ui(self, total_components, count_label, measure_label):
        layout = QVBoxLayout(self)

        # total_components is the raw count BEFORE any min-size filtering;
        # self.table_data is already the filtered table. Showing both makes
        # it obvious how much a filter removed, e.g.
        # "5902 components found - 2746 shown below".

        shown = len(self.table_data)
        summary = QLabel(f"{total_components} components found - {shown} shown after filtering below")
        layout.addWidget(summary)

        # Show every column actually present in the table, not just label/count/measure -
        # this way, checking/unchecking measurements in MeasurementsSettingsDialog just
        # changes what shows up here, with no changes needed to this dialog.

        known_labels = {'label': 'Label', self.count_col: count_label, self.measure_col: measure_label}
        columns = list(self.table_data.columns)
        # headers = [known_labels.get(col, col.replace('_', ' ').title()) for col in columns]
        # adding units for all the measuremetns headers in the resulting table
        headers = []
        for col in columns:
            if col in known_labels:
                headers.append(known_labels[col])
                continue
            if col in self._HEADER_OVERRIDES:
                headers.append(self._HEADER_OVERRIDES[col])
                continue
            label = col.replace('_', ' ').title()
            exponent = self._MEASUREMENT_UNIT_EXPONENTS.get(col)
            if exponent is not None:
                label+= f" ({self.unit}{self._SUPERSCRIPTS[exponent]})"
            headers.append(label)

        # Kept for _export_csv, so the CSV gets the same unit-labeled headers
        # shown on screen instead of the raw DataFrame column names.
        self.columns = columns
        self.headers = headers

        table = QTableWidget(shown, len(columns))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setStretchLastSection(True) # last column fills leftover width
        table.verticalHeader().setVisible(False)               # no numbered row headers needed
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)  # read-only results, not an editable grid
        # options for enabling selection...
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.ExtendedSelection) # ctrl/shift multi-select
        table.setContextMenuPolicy(Qt.CustomContextMenu)
        table.customContextMenuRequested.connect(self._show_context_menu)

        # itertuples() is faster than iterrows() for looping a DataFrame;
        # ._asdict() turns each row into a plain dict so columns can be
        # looked up by name (self.count_col/self.measure_col) instead of a
        # fixed attribute - needed since which column is "the measure"
        # depends on is_3d.
        for row, record in enumerate(self.table_data.itertuples(index=False)):
            record = record._asdict()  # convert namedtuple to dict for easier access by column name
            for col_idx, col in enumerate(columns):
                value = record[col]
                text = str(int(value)) if col == 'label' else f"{value:.4g}"
                table.setItem(row, col_idx, _NumericTableItem(text, float(value)))


        # Clicking a column header selects that column for histogram plotting.
        table.horizontalHeader().sectionClicked.connect(self._on_header_clicked)

        # Enabled only AFTER the rows are in. With sorting active during
        # population, Qt re-sorts on every insertion and the rows come out
        # scrambled - a classic QTableWidget gotcha.

        table.setSortingEnabled(True)
        self.table = table
        layout.addWidget(table)

        # Preview / Histogram / Save / OK / Cancel, right-aligned via the stretch.
        button_layout = QHBoxLayout()

        self.histogram_button = QPushButton("Histogram")
        self.histogram_button.setEnabled(False)  # enabled once a column is selected by clicking its header
        self.histogram_button.setToolTip("Click a column header below to select it, then click here to plot its histogram.")
        self.histogram_button.clicked.connect(self._show_histogram)


        save_button = QPushButton("Save to CSV...")
        save_button.clicked.connect(self._export_csv)

        # Same QCheckBox pattern as cc_color_checkbox in image_viewer - create,
        # set default state, connect its toggled signal to a handler.

        self.preview_checkbox = QCheckBox("Preview")
        self.preview_checkbox.setChecked(True)
        self.preview_checkbox.setToolTip(
            "Update the colored view as you delete components. Changes are only "
            "kept when you click OK."
        )
        self.preview_checkbox.toggled.connect(self._emit_preview)

        # Same QPushButton + clicked.connect pattern as save_button above.
        ok_button = QPushButton("OK")
        ok_button.setToolTip("Keep the deletions and close.")
        ok_button.clicked.connect(self._accept_changes)

        cancel_button = QPushButton("Cancel")
        cancel_button.setToolTip("Discard the deletions and close.")
        cancel_button.clicked.connect(self.close)   # closeEvent does the actual revert

        # Order here is left-to-right on screen. addStretch() pushes everything
        # after it to the right, so Preview sits alone on the left.
        button_layout.addWidget(self.preview_checkbox)
        button_layout.addStretch()
        button_layout.addWidget(self.histogram_button)
        button_layout.addWidget(save_button)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

    def _on_header_clicked(self, column_index):
            
            """Select a column by clicking its header - highlights it and enables the Histogram button."""

            self.selected_column_index = column_index
            self.histogram_button.setEnabled(True)
            self.histogram_button.setText(f"Histogram: {self.headers[column_index]}")


    def _selected_labels(self):
        """Component labels for the currently selected rows.

        Reads the label out of column 0's stored value rather than indexing
        table_data by row - once the table has been sorted, widget row order
        no longer matches the DataFrame's.
        """
        rows = {index.row() for index in self.table.selectedIndexes()}
        return sorted(int(self.table.item(row, 0).value) for row in rows)

    def _show_context_menu(self, position):
        """Right-click menu for the selected rows."""

        menu = QMenu(self)

        # Same addAction + triggered.connect pattern as the File menu actions
        # in image_viewer.

        invert_action = menu.addAction("Invert Selection")
        invert_action.triggered.connect(self._invert_selection)

        delete_action = menu.addAction("Delete Selected Components")
        delete_action.setEnabled(bool(self._selected_labels())) # greyed out if nothing selected
        delete_action.triggered.connect(self._delete_selected)

        # position is in viewport coordinates; exec wants global ones
        menu.exec(self.table.viewport().mapToGlobal(position))

    def _invert_selection(self):
        """Flip which rows are selected - a Toggle over every row does exactly that."""

        #Get the table's model
        model = self.table.model()
        #Create a selection covering the entire table
        everything = QItemSelection(
            model.index(0, 0),
            model.index(self.table.rowCount() - 1, self.table.columnCount() - 1),
            #(top_left_index, bottom_right_index)
        )

        # selectionModel() tracks which cells or rows are selected
        #QItemSelectionModel.Toggle -> reverse the selection state
        #QItemSelectionModel.Rows -> select entire rows, not individual cells
        self.table.selectionModel().select(
            everything, QItemSelectionModel.Toggle | QItemSelectionModel.Rows
        )

    def _delete_selected(self):
        """Drop the selected components from the table and the preview.

        Rows come out of the widget AND out of table_data, so the CSV export
        and histogram stay consistent with what's on screen. The label volume
        is untouched until OK.
        """

        labels = self._selected_labels()
        if not labels:
            return  # shouldn't happen - menu item is disabled if nothing selected

        self.deleted_labels.update(labels)
        # drop the rows from the DataFrame, reindex 0..N-1
        self.table_data = self.table_data[~self.table_data['label'].isin(labels)].reset_index(drop=True)

        # Remove bottom-up so the earlier row indices stay valid as rows go away.
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

        self._emit_preview()

    def _emit_preview(self):
        """Show the deletions in the colored view, or clear the preview if the
        checkbox is off."""
        self.preview_requested.emit(self.deleted_labels if self.preview_checkbox.isChecked() else set())

    def _accept_changes(self):
        # print("DEBUG: _accept_changes, deleting:", self.deleted_labels)   # temp
        self._committed = True
        self.changes_committed.emit(self.deleted_labels)
        self.close()
        
    def closeEvent(self, event):
        """Cancel and the window's X both land here - revert unless OK already
        committed."""
        if not self._committed:
            self.changes_reverted.emit()
        super().closeEvent(event)

    def _show_histogram(self):

        column_name = self.columns[self.selected_column_index]
        if column_name == 'label':
            QMessageBox.warning(self, "Not a Measurement", "Label is just a component ID, not a measurement - select a different column.")
            return

        values = self.table_data[column_name].to_numpy(dtype=float)
        values = values[~np.isnan(values)]  # degenerate components can produce NaN (see _safe_prop)
        if len(values) == 0:
            QMessageBox.warning(self, "No Data", "Every value in this column is NaN - nothing to plot.")
            return

        column_label = self.headers[self.selected_column_index]
        settings_dialog = HistogramSettingsDialog(column_label, self)
        if settings_dialog.exec() != QDialog.Accepted:
            return  # user cancelled

        window = HistogramPlotWindow(
            values, column_label,
            settings_dialog.get_percentile(), settings_dialog.get_num_bins(), settings_dialog.get_log_y(),
            self
        )
        window.show()

    def _export_csv(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV",
            suggested_save_path("connected_components"),  # remembers/suggests the last-used save folder
            "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return  # user cancelled the save dialog
        if not file_path.endswith('.csv'):
            file_path += '.csv'

        try:
            # Export with the same unit-labeled headers shown on screen, not
            # the raw DataFrame column names. table_data itself is untouched -
            # rename() returns a new DataFrame, index=False skips pandas' own
            # 0..N row index.
            export_table = self.table_data.rename(columns=dict(zip(self.columns, self.headers)))
            # utf-8-sig adds a UTF-8 BOM - without it, Excel doesn't auto-detect
            # UTF-8 for a plain .csv and misreads the unit symbols (mu, superscripts)
            # as Windows-1252, garbling them into things like "Âµm".
            export_table.to_csv(file_path, index=False, encoding='utf-8-sig')
            remember_save_dir(file_path)  # so the next Save dialog opens in this same folder
            QMessageBox.information(self, "Success", f"Data exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export CSV:\n{str(e)}")
