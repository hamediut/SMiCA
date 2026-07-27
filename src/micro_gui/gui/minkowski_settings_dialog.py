"""
Dialog for single-image Minkowski Functionals calculation settings.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QComboBox, QPushButton, QGroupBox, QFormLayout
)

class MinkowskiSettingsDialog(QDialog):
    """
    Dialog to configure the Minkowski Functionals calculation for the current
    2D/3D image. Only one parameter - pixel/voxel size - since minkowski_2d/
    minkowski_3d handle the rest. Same QFormLayout numeric-dialog pattern as
    ChordLengthSettingsDialog.
    """

    def __init__(self, is_3d: bool = False, parent = None):
        super().__init__(parent)

        self.is_3d = is_3d
        self.resolution = None # will hold the chosen value after OK
        self.unit = None

        self.setWindowTitle("Minkowski Functionals Settings")
        self.setModal(True)
        self.setMinimumWidth(360)

        self._setup_ui()

    def _setup_ui(self):

        layout = QVBoxLayout(self)

        dims = ("3D (volume, surface area, mean curvature, Euler characteristic)" if self.is_3d
                else "2D (area, perimeter, Euler characteristic)")

        info_label = QLabel(f"Computing Minkowski functionals for a {dims} image.")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        settings_group = QGroupBox("Minkowski Functionals Settings")
        form_layout = QFormLayout()

        self.resolution_spinbox = QDoubleSpinBox()
        self.resolution_spinbox.setRange(0.0001, 100000.0)
        self.resolution_spinbox.setDecimals(4)
        self.resolution_spinbox.setValue(1.0)
        self.resolution_spinbox.setToolTip(
            "Physical size of one pixel/voxel (isotropic). Scales area/volume, "
            "surface/perimeter, and curvature into physical units - leave at 1.0 "
            "for results in pixel/voxel units."
        )
        form_layout.addRow("Pixel/voxel size:", self.resolution_spinbox)

        self.unit_combo = QComboBox()
        self.unit_combo.addItems(['\u00b5m', 'mm', 'nm', 'pixels/voxels'])
        self.unit_combo.setToolTip(
            "Physical unit the pixel/voxel size above is expressed in - purely "
            "a label for results tables and plot axes, no conversion is applied."
        )
        form_layout.addRow("Unit:", self.unit_combo)

        settings_group.setLayout(form_layout)
        layout.addWidget(settings_group)

        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self._accept)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

    def _accept(self):
        self.resolution = self.resolution_spinbox.value()
        self.unit = self.unit_combo.currentText()
        self.accept()

    def get_resolution(self):
        return self.resolution

    def get_unit(self):
        return self.unit












