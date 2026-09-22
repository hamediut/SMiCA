"""
Results dialog for connected-components output: one row per component.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QPushButton, QFileDialog, QMessageBox
)

from .save_dialog_helper import suggested_save_path, remember_save_dir

import numpy as np
from .histogram_plot_window import HistogramPlotWindow
from .histogram_settings_dialog import HistogramSettingsDialog


class ConnectedComponentsResultsDialog(QDialog):
    """
    Read-only results table for a connected-components calculation - one row
    per component, already sorted largest-first by the analysis function.
    Not modal (shown with .show(), not .exec()) so the user can keep it open
    while comparing against the image or running another calculation.
    """

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

        # itertuples() is faster than iterrows() for looping a DataFrame;
        # ._asdict() turns each row into a plain dict so columns can be
        # looked up by name (self.count_col/self.measure_col) instead of a
        # fixed attribute - needed since which column is "the measure"
        # depends on is_3d.
        for row, record in enumerate(self.table_data.itertuples(index=False)):
            record = record._asdict()  # convert namedtuple to dict for easier access by column name
            for col_idx, col in enumerate(columns):
                text = str(int(record[col])) if col == 'label' else f"{record[col]:.4g}"
                table.setItem(row, col_idx, QTableWidgetItem(text))


        # Clicking a column header selects that column for histogram plotting.
        table.horizontalHeader().sectionClicked.connect(self._on_header_clicked)

        self.table = table
        layout.addWidget(table)

        # Save/Close buttons, right-aligned via the stretch placed before them.
        button_layout = QHBoxLayout()

        self.histogram_button = QPushButton("Histogram")
        self.histogram_button.setEnabled(False)  # enabled once a column is selected by clicking its header
        self.histogram_button.setToolTip("Click a column header below to select it, then click here to plot its histogram.")
        self.histogram_button.clicked.connect(self._show_histogram)


        save_button = QPushButton("Save to CSV...")
        save_button.clicked.connect(self._export_csv)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)

        button_layout.addStretch()
        button_layout.addWidget(self.histogram_button)
        button_layout.addWidget(save_button)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

    def _on_header_clicked(self, column_index):
            
            """Select a column by clicking its header - highlights it and enables the Histogram button."""

            self.selected_column_index = column_index
            self.table.selectColumn(column_index)
            self.histogram_button.setEnabled(True)
            self.histogram_button.setText(f"Histogram: {self.headers[column_index]}")


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
