"""
Results dialog for single-image Minkowski Functionals output.
"""

import csv
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QPushButton, QFileDialog, QMessageBox
)

from .save_dialog_helper import suggested_save_path, remember_save_dir

class MinkowskiResultsDialog(QDialog):
    """
    Read-only results table for a single 2D/3D Minkowski functionals
    calculation, with a "Save to CSV" export. Not modal (shown with .show(),
    not .exec()) so the user can keep it open while comparing against the
    image or running another calculation.
    """

    # (result dict key, display label, unit exponent - None means dimensionless)
    _ROWS_2D = [
        ('area', 'Area', 2),
        ('perimeter', 'Perimeter', 1),
        ('euler', 'Euler characteristic', None),
        ('area_fraction', 'Area fraction', None),
        ('specific_perimeter', 'Specific perimeter', -1),
        ('specific_euler', 'Specific Euler characteristic', -2),
    ]

    _ROWS_3D = [
        ('volume', 'Volume', 3),
        ('surface_area', 'Surface area', 2),
        ('mean_curv', 'Mean curvature', 1),
        ('euler', 'Euler characteristic', None),
        ('porosity', 'Porosity', None),
        ('specific_surface_area', 'Specific surface area', -1),
        ('specific_mean_curv', 'Specific mean curvature', -2),
        ('specific_euler', 'Specific Euler characteristic', -3),
    ]

    _SUPERSCRIPTS = {1: '', 2: '\u00b2', 3: '\u00b3', -1: '\u207b\u00b9', -2: '\u207b\u00b2', -3: '\u207b\u00b3'}

    #'\u00b2' --> ^2--> e.g., f"{unit}{'\u00b2'}" → "µm²"
    #'\u00b3' — U+00B3 → ³, giving "µm³"
    # -1: '\u207b\u00b9' --> "µm⁻¹"
    # -2: '\u207b\u00b2' → ⁻² → "µm⁻²".
    # -3: '\u207b\u00b3' → ⁻³ → "µm⁻³"
    
    def __init__(self, result: dict, unit: str, is_3d: bool, parent = None):
        super().__init__(parent)

        self.result = result
        self.unit = unit
        self.rows = self._ROWS_3D if is_3d else self._ROWS_2D

        self.setWindowTitle("Minkowski Functionals Results")
        self.setMinimumWidth(420)
        self._setup_ui()

    def _unit_label(self, exponent):

        """Build a unit string like 'um^2' for a functional, or '' if dimensionless."""
        if exponent is None:
            return ''
        return f"{self.unit}{self._SUPERSCRIPTS[exponent]}"

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        table = QTableWidget(len(self.rows), 3)
        table.setHorizontalHeaderLabels(['Quantity', 'Value', 'Unit'])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        for row, (key, label, exponent) in enumerate(self.rows):
            table.setItem(row, 0, QTableWidgetItem(label))
            table.setItem(row, 1, QTableWidgetItem(f"{self.result[key]:.6g}"))
            table.setItem(row, 2, QTableWidgetItem(self._unit_label(exponent)))

        self.table = table
        layout.addWidget(table)

        button_layout = QHBoxLayout()
        save_button = QPushButton("Save to CSV...")
        save_button.clicked.connect(self._export_csv)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)

        button_layout.addStretch()
        button_layout.addWidget(save_button)
        button_layout.addWidget(close_button)

    def _export_csv(self):


        file_path, _ = QFileDialog.getSaveFileName(
        self, "Export CSV",
        suggested_save_path("minkowski_functionals"),
        "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return

        if not file_path.endswith('.csv'):
            file_path += '.csv'

        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Quantity', 'Value', 'Unit'])
                for key, label, exponent in self.rows:
                    writer.writerow([label, self.result[key], self._unit_label(exponent)])
            remember_save_dir(file_path)
            QMessageBox.information(self, "Success", f"Data exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export CSV:\n{str(e)}")

            

