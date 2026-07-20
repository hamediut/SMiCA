"""
Dialog for configuring a slice-by-slice (or time-step-by-time-step) 2D correlation
function analysis across a 3D stack, including the Omega/Delta-Omega evolution metrics.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton,
    QMessageBox, QGroupBox, QLabel, QSpinBox, QComboBox
)


class SliceEvolutionSettingsDialog(QDialog):

    """
    Lets the user pick which 2D functions to compute independently on every slice of a 3D
    stack, a step size to skip slices for speed, and the Omega/Delta-Omega evolution
    metrics (reference slice for Omega, magnitude-vs-signed choice for Delta-Omega).
    """

    POLYTOPE_OPTIONS = [
        ('s2', 'S2 (two-point correlation)'),
        ('c2', 'C2 (two-point cluster function)'),
        ('p3h', 'P3H (triangle, horizontal)'),
        ('p3v', 'P3V (triangle, vertical)'),
        ('p4', 'P4 (square)'),
        ('p6', 'P6 (hexagon)'),
        ('L', 'L (lineal path)'),
    ]

    SUPPORT_3D = {'s2', 'c2', 'L'}  # only S2, C2, and L are supported for 3D volumes-over-time (same set as PolytopeSettingsDialog)

    def __init__(self, n_slices: int, axis_label: str = "slice", stack_labels=None, is_3d: bool = False, parent=None):
        super().__init__(parent)

        self.n_slices = n_slices
        self.axis_label = axis_label
        # True when each "slice"/time step is itself a 3D volume (imported via "a full 3D
        # volume" in ImportSequenceDialog) - restricts the checkboxes below to S2/C2/L, same
        # as PolytopeSettingsDialog does for a single 3D volume.
        self.is_3d = is_3d
        # Real step/time numbers pulled from the imported filenames (None for a native 3D
        # volume/regular tif, which has no such numbers) - used below only to LABEL the
        # reference dropdown; the actual stored reference is still the array position.
        self.stack_labels = stack_labels
        self.selected_polytopes = None
        self.step = None
        self.reference_index = None
        self.compute_omega = None
        self.compute_delta_omega = None
        self.signed_delta_omega = None

        self.setWindowTitle("Slice/Time Evolution Settings")
        self.setModal(True)
        self.setMinimumWidth(360)

        self._checkboxes = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        info_label = QLabel(
            f"Computes each selected 2D function independently on every {self.axis_label} "
            f"({self.n_slices} total)."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        if self.is_3d:
            volume_info_label = QLabel(
                "Each " + self.axis_label + " here is a full 3D volume. Only S2, C2, and L are "
                "currently supported for 3D - the other functions (P3H, P3V, P4, P6) only work "
                "on 2D images, so they are disabled below."
            )
            volume_info_label.setWordWrap(True)
            layout.addWidget(volume_info_label)

        group = QGroupBox("Select functions to calculate")
        group_layout = QVBoxLayout()
        for internal_name, label in self.POLYTOPE_OPTIONS:
            checkbox = QCheckBox(label)

            is_2d_only = internal_name not in self.SUPPORT_3D
            if self.is_3d and is_2d_only:
                checkbox.setChecked(False)  # uncheck 2D-only options for volume time series
                checkbox.setEnabled(False)  # grey out - stops clicks/keyboard focus
            else:
                checkbox.setChecked(True)

            self._checkboxes[internal_name] = checkbox
            group_layout.addWidget(checkbox)
        group.setLayout(group_layout)
        layout.addWidget(group)

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel(f"Use every Nth {self.axis_label}:"))
        self.step_spinbox = QSpinBox()
        self.step_spinbox.setRange(1, max(1, self.n_slices - 1))
        self.step_spinbox.setValue(1)
        self.step_spinbox.setToolTip(f"1 = every {self.axis_label}; 5 = every 5th, etc.")
        self.step_spinbox.valueChanged.connect(self._update_reference_options)
        step_row.addWidget(self.step_spinbox)
        layout.addLayout(step_row)

        omega_group = QGroupBox("Evolution metrics")
        omega_layout = QVBoxLayout()

        self.omega_checkbox = QCheckBox("Compute Omega (distance from a reference slice)")
        self.omega_checkbox.setChecked(True)
        omega_layout.addWidget(self.omega_checkbox)

        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel(f"    Reference {self.axis_label}:"))
        self.reference_combo = QComboBox()
        ref_row.addWidget(self.reference_combo, stretch=1)
        omega_layout.addLayout(ref_row)

        self.delta_omega_checkbox = QCheckBox("Compute Delta-Omega (distance from the previous slice)")
        self.delta_omega_checkbox.setChecked(True)
        omega_layout.addWidget(self.delta_omega_checkbox)

        self.signed_checkbox = QCheckBox("    Signed Delta-Omega (net/directional change, instead of magnitude)")
        self.signed_checkbox.setToolTip(
            "Unchecked (default): L1 norm - always >= 0, measures how much the curve changed.\n"
            "Checked: plain sum of differences - can be positive or negative, but opposite-\n"
            "direction changes can cancel out."
        )
        omega_layout.addWidget(self.signed_checkbox)

        omega_group.setLayout(omega_layout)
        layout.addWidget(omega_group)

        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self._validate_and_accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self._update_reference_options()

    def _update_reference_options(self):
        """Repopulate the reference dropdown to match whichever slices the current step will actually compute."""
        available_positions = list(range(0, self.n_slices, self.step_spinbox.value()))
        self.reference_combo.clear()
        for pos in available_positions:
            # Show the real filename-derived number when we have one, so this matches what
            # the resulting plot will show - but userData always stays the ARRAY POSITION,
            # since that's what's actually needed to index into the image data later.
            display_value = self.stack_labels[pos] if self.stack_labels is not None else pos
            self.reference_combo.addItem(f"{self.axis_label.title()} {display_value}", userData=pos)

    def _validate_and_accept(self):
        selected = [name for name, _ in self.POLYTOPE_OPTIONS if self._checkboxes[name].isChecked()]
        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select at least one function.")
            return

        self.selected_polytopes = selected
        self.step = self.step_spinbox.value()
        self.compute_omega = self.omega_checkbox.isChecked()
        self.compute_delta_omega = self.delta_omega_checkbox.isChecked()
        self.signed_delta_omega = self.signed_checkbox.isChecked()
        self.reference_index = self.reference_combo.currentData()
        self.accept()

    def get_selected_polytopes(self):
        return self.selected_polytopes

    def get_step(self):
        return self.step

    def get_reference_index(self):
        return self.reference_index

    def get_compute_omega(self):
        return self.compute_omega

    def get_compute_delta_omega(self):
        return self.compute_delta_omega

    def get_signed_delta_omega(self):
        return self.signed_delta_omega