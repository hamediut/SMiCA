"""
Dialog for configuring the measurement step that follows connected-components
labeling.
"""


from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox,
    QPushButton, QGroupBox, QFormLayout
)


from ..analysis.connected_components import ADVANCED_MEASUREMENTS


class MeasurementsSettingsDialog(QDialog):
    """
    Dialog to configure the measurement step run on an already-labeled
    connected-components result: physical units, a minimum-size filter, and
    which shape measurements to include in the results table. Connectivity
    isn't here - that's already fixed by the time this dialog opens, chosen
    back in ConnectedComponentsSettingsDialog.
    """

    # (column name compute_shape_measurements produces, display label) -
    # list-driven so adding a new measurement later (once the marching-cubes/
    # orientation ones exist) is a new tuple here, not new UI code.

    _MEASUREMENTS_3D = [
        ('equivalent_diameter', 'Equivalent diameter'),
        ('aspect_ratio', 'Aspect ratio'),
        ('surface_area', 'Surface area'),
        ('specific_surface_area', 'Specific surface area'),
        ('sphericity', 'Sphericity'),
        ('elongation', 'Elongation'),
        ('flatness', 'Flatness'),
        ('theta_deg', 'Orientation theta'),
        ('phi_deg', 'Orientation phi'),
    ]


    _MEASUREMENTS_2D = [
        ('equivalent_diameter', 'Equivalent diameter'),
        ('aspect_ratio', 'Aspect ratio'),
        ('perimeter', 'Perimeter'),
        ('specific_perimeter', 'Specific perimeter'),
        ('circularity', 'Circularity'),
    ]


    def __init__(self, is_3d: bool, num_components: int, parent = None):
        super().__init__(parent)

        self.is_3d = is_3d
        self.measurement_defs = self._MEASUREMENTS_3D if is_3d else self._MEASUREMENTS_2D

        self.resolution = None
        self.unit = None
        self.min_size = None
        self.selected_measurements = None

        self.setWindowTitle("Measurements Settings")
        self.setModal(True)
        self.setMinimumWidth(360)

        self._setup_ui(num_components)

    def _setup_ui(self, num_components):
        layout = QVBoxLayout(self)

        count_unit = "voxel" if self.is_3d else "pixel"
        measure_word = "volume" if self.is_3d else "area"

        info_label = QLabel(
            f"Configure measurements for the {num_components} component(s) found by "
            f"the last Connected Components run."
        )

        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        #---physical units---
        units_group = QGroupBox("Physical Units")
        units_form = QFormLayout()

        self.resolution_spinbox = QDoubleSpinBox()
        self.resolution_spinbox.setRange(0.0001, 1000.0)
        self.resolution_spinbox.setDecimals(4)
        self.resolution_spinbox.setValue(1.0)
        self.resolution_spinbox.setToolTip(
            f"Physical size of one {count_unit} (isotropic). Scales {count_unit} counts "
            f"into physical {measure_word}s - leave at 1.0 for results in {count_unit} units."
        )

        units_form.addRow(f"{count_unit.capitalize()} size:", self.resolution_spinbox)

        self.unit_combo = QComboBox()
        self.unit_combo.addItems(['\u00b5m', 'mm', 'nm', count_unit + 's'])
        self.unit_combo.setToolTip(
            f"Unit of measurement for the physical {measure_word}."
        )
        units_form.addRow(f"Unit:", self.unit_combo)

        units_group.setLayout(units_form)
        layout.addWidget(units_group)

        # --- Filtering ---

        filter_group = QGroupBox("Filtering")
        filter_form = QFormLayout()

        self.min_size_spinbox = QSpinBox()
        self.min_size_spinbox.setRange(0, 10_000_000)
        self.min_size_spinbox.setValue(1)
        self.min_size_spinbox.setToolTip(
            f"Components smaller than this (in {count_unit}s) are dropped from the "
            f"results table and the colored view."
        )

        filter_form.addRow(f"Minimum size ({count_unit}s):", self.min_size_spinbox)
        filter_group.setLayout(filter_form)
        layout.addWidget(filter_group)

        # --- Shape measurements ---
        measurements_group = QGroupBox("Shape Measurements")
        measurements_layout = QVBoxLayout()

        self.measurement_checkboxes = {}
        for column_name, display_name in self.measurement_defs:
            checkbox = QCheckBox(display_name)
            # Cheap regionprops-based measurements default on; the mesh/eigenvector ones
            # default off, since they cost ~45s on a volume with thousands of components.

            checkbox.setChecked(column_name not in ADVANCED_MEASUREMENTS)
            measurements_layout.addWidget(checkbox)
            self.measurement_checkboxes[column_name] = checkbox

        measurements_group.setLayout(measurements_layout)
        layout.addWidget(measurements_group)

        if self.is_3d:
            slow_note = QLabel(
                "Surface area, specific surface area, sphericity, elongation, flatness and "
                "orientation need a per-component mesh reconstruction - noticeably slower on "
                "volumes with many components."
            )

            slow_note.setWordWrap(True)
            layout.addWidget(slow_note)


         # --- OK / Cancel ---
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
        self.min_size = self.min_size_spinbox.value()
        self.selected_measurements = [
            column_name for column_name, checkbox in self.measurement_checkboxes.items()
            if checkbox.isChecked()
        ]
        self.accept()

    def get_resolution(self):
        return self.resolution

    def get_unit(self):
        return self.unit

    def get_min_size(self):
        return self.min_size

    def get_selected_measurements(self):
        return self.selected_measurements





        

        
