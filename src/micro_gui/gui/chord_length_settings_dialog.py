"""
Dialog for single-image Chord Length calculation settings.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QPushButton, QGroupBox, QFormLayout
)


class ChordLengthSettingsDialog(QDialog):
    """
    Dialog to configure the Chord Length calculation for the current 2D/3D image.

    Only two numeric parameters (num_points, nbins) - no phase selector, since the
    calculation always runs on the foreground phase (pixel value 1). Same
    QFormLayout numeric-dialog pattern as REVSettingsDialog.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.num_points = None  # will hold the chosen values after OK
        self.nbins = None

        self.setWindowTitle("Chord Length Calculation Settings")
        self.setModal(True)
        self.setMinimumWidth(380)

        self._setup_ui()

    def _setup_ui(self):
        """Set up the dialog UI."""

        layout = QVBoxLayout(self)

        info_label = QLabel(
            "Computes the mean chord length of the foreground phase (pixel value 1) two "
            "ways - from the slope of S2(r) at r=0, and by directly sampling run-lengths - "
            "plus the chord-length distribution from the direct method."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        settings_group = QGroupBox("Chord Length Settings")
        form_layout = QFormLayout() #is a layout manager designed specifically for forms: pairs of labels and input widgets

        self.num_points_spinbox = QSpinBox()
        self.num_points_spinbox.setRange(2, 20)
        self.num_points_spinbox.setValue(2)  # validated default - see chord_length_from_S2's docstring
        self.num_points_spinbox.setToolTip(
            "Number of small-r S2 samples used to fit the slope at r=0. S2(r) is concave "
            "near the origin, so values above 2 systematically overestimate the chord "
            "length - keep this at 2 unless you have a specific reason to change it."
        )
        form_layout.addRow("num_points (S2 slope fit):", self.num_points_spinbox)

        self.nbins_spinbox = QSpinBox()
        self.nbins_spinbox.setRange(5, 500)
        self.nbins_spinbox.setValue(40)
        self.nbins_spinbox.setToolTip("Number of histogram bins for the chord-length distribution.")
        form_layout.addRow("Distribution bins:", self.nbins_spinbox)

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
        """No validation needed - the spin boxes already constrain the ranges."""
        self.num_points = self.num_points_spinbox.value()
        self.nbins = self.nbins_spinbox.value()
        self.accept()

    def get_num_points(self):
        return self.num_points

    def get_nbins(self):
        return self.nbins
