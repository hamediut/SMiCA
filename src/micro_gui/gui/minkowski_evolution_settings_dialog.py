"""
Dialog for configuring the Minkowski Functionals calculation across a stack (2D
time-series of slices, or a time-series of 3D volumes) - the "evolution"
counterpart to MinkowskiSettingsDialog.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QComboBox, QDoubleSpinBox, QPushButton, QGroupBox, QFormLayout
)


class MinkowskiEvolutionSettingsDialog(QDialog):
    """
    Step size (skip slices for speed) and direction - same fields as
    ChordLengthEvolutionSettingsDialog - plus the resolution/unit pair from
    MinkowskiSettingsDialog.
    """

    def __init__(self, n_slices: int, axis_label: str = 'slice', is_3d: bool = False, parent = None):

        super().__init__(parent)

        self.n_slices = n_slices
        self.axis_label = axis_label
        self.is_3d = is_3d

        self.step = None
        self.reverse_direction = None
        self.resolution = None
        self.unit = None

        self.setWindowTitle("Minkowski Functionals Evolution Settings")
        self.setModal(True)
        self.setMinimumWidth(380)

        self._setup_ui()


    def _setup_ui(self):

        layout = QVBoxLayout(self)

        dims = "3D volume" if self.is_3d else "2D image"
        info_label = QLabel(
            f"Computes Minkowski functionals independently on every {self.axis_label} "
            f"({self.n_slices} total, each a {dims})."
        )

        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel(f"Use every Nth {self.axis_label}:"))
        self.step_spinbox = QSpinBox()
        self.step_spinbox.setRange(1, max(1, self.n_slices - 1))
        self.step_spinbox.setValue(1)
        self.step_spinbox.setToolTip(f"1 = every {self.axis_label}; 5 = every 5th, etc.")
        step_row.addWidget(self.step_spinbox)
        layout.addLayout(step_row)

        direction_row = QHBoxLayout()
        direction_row.addWidget(QLabel("Direction"))
        self.direction_combo = QComboBox()
        self.direction_combo.addItem(f"First -> Last {self.axis_label}", userData=False)
        self.direction_combo.addItem(f"Last -> First {self.axis_label}", userData=True)
        direction_row.addWidget(self.direction_combo, stretch=1)
        layout.addLayout(direction_row)

        settings_group = QGroupBox("Minkowski Functionals Settings")
        form_layout = QFormLayout()

        self.resolution_spinbox = QDoubleSpinBox()
        self.resolution_spinbox.setRange(0.0001, 100000.0)
        self.resolution_spinbox.setDecimals(4)
        self.resolution_spinbox.setValue(1.0)
        self.resolution_spinbox.setToolTip(
            "Physical size of one pixel/voxel (isotropic), same as the single-image dialog."
        )

        form_layout.addRow("Pixel/voxel size:", self.resolution_spinbox)

        self.unit_combo = QComboBox()
        self.unit_combo.addItems(['\u00b5m', 'mm', 'nm', 'pixels/voxels'])
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
        self.step = self.step_spinbox.value()
        self.reverse_direction = self.direction_combo.currentData()
        self.resolution = self.resolution_spinbox.value()
        self.unit = self.unit_combo.currentText()
        self.accept()

    def get_step(self):
        return self.step

    def get_reverse_direction(self):
        return self.reverse_direction

    def get_resolution(self):
        return self.resolution

    def get_unit(self):
        return self.unit