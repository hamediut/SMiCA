"""
Dialog for configuring the Chord Length calculation across a stack (2D time-series of
slices, or a time-series of 3D volumes) - the "evolution" counterpart to
ChordLengthSettingsDialog.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QComboBox, QPushButton, QGroupBox, QFormLayout
)


class ChordLengthEvolutionSettingsDialog(QDialog):
    """
    Lets the user pick a step size (skip slices for speed) and direction, plus the same
    num_points/nbins parameters as the single-image dialog. Which SPECIFIC slices get
    overlaid in the distribution plot is decided later, interactively, in the plot
    window - not here (this dialog only controls what gets COMPUTED).
    """

    def __init__(self, n_slices : int, axis_label: str = "slice", parent = None):
        super().__init__(parent)

        self.n_slices = n_slices
        self.axis_label = axis_label

        # Will hold the chosen values after OK - None means "dialog was cancelled"
        self.step = None
        self.reverse_direction = None
        self.num_points = None
        self.nbins = None

        self.setWindowTitle("Chord Length Evolution Settings")
        self.setModal(True)
        self.setMinimumWidth(380)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        info_label = QLabel(
            f"Computes the chord length of the foreground phase independently on every "
            f"{self.axis_label} ({self.n_slices} total)."
        
        )

        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # --- step + direction: copied from SliceEvolutionSettingsDialog verbatim ---

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel(f"Use every Nth {self.axis_label}:"))
        self.step_spinbox = QSpinBox()
        self.step_spinbox.setRange(1, max(1, self.n_slices -1))
        self.step_spinbox.setValue(1)
        self.step_spinbox.setToolTip(f"1 = every {self.axis_label}; 5 = every 5th, etc.")
        step_row.addWidget(self.step_spinbox)
        layout.addLayout(step_row)

        direction_row = QHBoxLayout()
        direction_row.addWidget(QLabel(f"Direction"))
        self.direction_combo = QComboBox()
        self.direction_combo.addItem(f"First -> Last {self.axis_label}", userData=False)
        self.direction_combo.addItem(f"Last -> First {self.axis_label}", userData=True)
        direction_row.addWidget(self.direction_combo, stretch = 1)
        layout.addLayout(direction_row)

        # --- num_points/nbins: same fields as ChordLengthSettingsDialog --

        setting_group = QGroupBox("Chord Length Settings")
        form_layout = QFormLayout()

        self.num_points_spinbox = QSpinBox()
        self.num_points_spinbox.setRange(2, 20)
        self.num_points_spinbox.setValue(2)
        self.num_points_spinbox.setToolTip(
            "Number of small-r S2 samples used to fit the slope at r=0 - keep at 2 (see "
            "chord_length_from_S2's docstring)."
        )
        form_layout.addRow("num_points (S2 slope fit):", self.num_points_spinbox)

        self.nbins_spinbox = QSpinBox()
        self.nbins_spinbox.setRange(5, 500)
        self.nbins_spinbox.setValue(40)
        self.nbins_spinbox.setToolTip("Number of histogram bins for each slice's chord-length distribution.")
        form_layout.addRow("Distribution bins:", self.nbins_spinbox)

        setting_group.setLayout(form_layout)

        layout.addWidget(setting_group)

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
        """No validation needed here - every field is already range/choice-constrained."""
        self.step = self.step_spinbox.value()
        self.reverse_direction =  self.direction_combo.currentData()
        self.num_points = self.num_points_spinbox.value()
        self.nbins = self.nbins_spinbox.value()
        self.accept()

    def get_step(self):
        return self.step
    
    def get_reverse_direction(self):
        return self.reverse_direction

    def get_num_points(self):
        return self.num_points

    def get_nbins(self):
        return self.nbins





        




