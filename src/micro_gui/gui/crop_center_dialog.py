"""
Dialog for the Process > Cropping > Crop Center... settings.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QGroupBox, QFormLayout, QSpinBox
)


class CropCenterDialog(QDialog):
    """
    Settings for cropping the largest box from the center of a cylindrical sample
    that lies completely inside the sample (see processing/cropping.py for how the
    box is found).

    Asks for:
        - which pixel value is the exterior (the air around the sample),
        - for 3D volumes, whether the box should be a cube or a square column,
        - an optional safety margin from the sample boundary.

    Non-modal, so the user can keep hovering over the image in the main window to check
    pixel values while choosing. The caller therefore opens it with show() (not exec())
    and reacts to its `accepted` signal.
    """

    def __init__(self, exterior_candidates, is_3d: bool = True, parent=None):
        """
        Args:
            exterior_candidates: [(value, fraction_of_edge_pixels), ...], most common
                first, as returned by exterior_value_candidates(). Only values that
                touch the image edges can be the air around the sample, so the combo
                box lists just these, with the most common one preselected.
            is_3d: True for a (Z, Y, X) volume - shows the cube/column choice.
            parent: parent window.
        """
        super().__init__(parent)

        self.is_3d = is_3d
        self._exterior_candidates = list(exterior_candidates)

        # Filled in by _accept() when the user clicks OK.
        self.exterior_value = None
        self.make_cube = True
        self.margin = 0

        self.setWindowTitle("Crop Center")
        self.setModal(False)  # non-modal: the main window (hover, sliders) stays usable while this is open
        self.setMinimumWidth(420)

        self._setup_ui()

    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)

        dims = "3D volume" if self.is_3d else "2D image"
        info_label = QLabel(
            f"Crop the largest {'box' if self.is_3d else 'square'} around the center of "
            f"the sample that lies completely inside it, removing the exterior (air) "
            f"around a cylindrical sample in this {dims}.\n\n"
            "Pores inside the sample may have the same value as the exterior - they are "
            "kept, because only exterior pixels connected to the image edge count as outside.\n\n"
            "Tip: hover over the image to check pixel values while this dialog is open."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        settings_group = QGroupBox("Crop Settings")
        form_layout = QFormLayout()

        # Exterior value. The text shows how much of the image edge has each value as a
        # hint; userData holds the actual value, so _accept() never has to parse the text.
        self.exterior_combo = QComboBox()
        for value, fraction in self._exterior_candidates:
            self.exterior_combo.addItem(f"{value}   ({fraction:.0%} of edge pixels)", value)
        self.exterior_combo.setToolTip(
            "Pixel value of the air around the sample. Only values found along the image "
            "edges are listed, the most common first."
        )
        form_layout.addRow("Exterior (air) value:", self.exterior_combo)

        # Cube or square column - only meaningful when there is a Z axis.
        if self.is_3d:
            self.shape_combo = QComboBox()
            self.shape_combo.addItem("Cube (same size in X, Y and Z)", True)
            self.shape_combo.addItem("Square column (largest volume, any height)", False)
            self.shape_combo.setToolTip(
                "Cube: equal size in all three directions, placed in the middle of the sample.\n"
                "Square column: the same square in every slice, usually over the full sample height."
            )
            form_layout.addRow("Shape:", self.shape_combo)

        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, 1000)
        self.margin_spin.setValue(0)
        self.margin_spin.setSuffix(" px")
        self.margin_spin.setToolTip(
            "Extra distance to keep from the sample boundary in X and Y, e.g. to stay clear "
            "of blurry or poorly segmented edge pixels. 0 lets the box touch the boundary."
        )
        form_layout.addRow("Safety margin:", self.margin_spin)

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
        """Store the chosen settings and close the dialog."""
        self.exterior_value = self.exterior_combo.currentData()
        self.make_cube = self.shape_combo.currentData() if self.is_3d else False
        self.margin = self.margin_spin.value()
        self.accept()

    def get_exterior_value(self):
        """Pixel value of the exterior (air), or None if the dialog was cancelled."""
        return self.exterior_value

    def get_make_cube(self) -> bool:
        """True for a cube, False for a square column (always False for 2D images)."""
        return self.make_cube

    def get_margin(self) -> int:
        """Safety margin from the sample boundary, in pixels."""
        return self.margin
