"""
Dialog for selecting which polytope functions to calculate.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton,
    QMessageBox, QGroupBox, QLabel
)

class PolytopeSettingsDialog(QDialog):
    """
    Dialog to let the user pick which polytope (and S2) functions to compute
    for the current 2D image.

    Unlike REVSettingsDialog (which asks for numeric parameters via a
    QFormLayout of label+field rows), this dialog just needs a set of
    checkboxes - one per available function - so the "form" here is a
    QGroupBox full of QCheckBox widgets instead.
    
    """

    # (internal name, display label) - internal name is exactly what
    # calculate_s2()/calculate_polytopes_python() expect as keys, so there's
    # only one place to edit if a new polytope function is added later.

    POLYTOPE_OPTIONS = [
        ('s2', 'S2 (two-point correlation)'),
        ('c2', 'C2 (two-point cluster function)'),
        ('p3h', 'P3H (triangle, horizontal)'),
        ('p3v', 'P3V (triangle, vertical)'),
        ('p4', 'P4 (square)'),
        ('p6', 'P6 (hexagon)'),
        ('L', 'L (lineal path)'),
    ]

    SUPPORT_3D = {'s2', 'c2', 'L'}  # only S2, C2, and L are supported for 3D images

    def __init__(self, is_3d: bool = False, parent=None):
        super().__init__(parent)

        self.is_3d = is_3d
        self.selected_polytopes = None  # will hold the chosen names after OK

        self.setWindowTitle("Polytope Calculation Settings")
        self.setModal(True) ## Block interaction with main window, blocks main window, User MUST respond before using main window.
        self.setMinimumWidth(300)

        self._checkboxes = {}  # internal name -> QCheckBox, so we can read them back in _validate_and_accept
        self._setup_ui()

    
    def _setup_ui(self):
        """ Set up the dialog UI. """

        layout = QVBoxLayout(self)

        if self.is_3d:
            info_label = QLabel(
                "This is a 3D image. Only S2 AND C2, and L are currently supported for 3D - the "
            "other polytope functions (P3H, P3V, P4, P6) only work on 2D "
            "images, so they are disabled below."
            )

            info_label.setWordWrap(True)
            layout.addWidget(info_label)

        group = QGroupBox("Select functions to calculate")
        group_layout = QVBoxLayout(group)

        for internal_name, label in self.POLYTOPE_OPTIONS:
            checkbox = QCheckBox(label)

            is_2d_only = internal_name not in self.SUPPORT_3D  # all except S2 and C2 are 2D-only
            if self.is_3d and is_2d_only:
                checkbox.setChecked(False)  # uncheck 2D-only options for 3D images
                checkbox.setEnabled(False)  # disable 2D-only options for 3D images. This is the actual "grey out" mechanism in Qt — a disabled widget renders dimmed and stops receiving clicks/keyboard focus
            else:
                checkbox.setChecked(True)  # default: everything selected
            self._checkboxes[internal_name] = checkbox
            group_layout.addWidget(checkbox)

        group.setLayout(group_layout)
        layout.addWidget(group)

        # Buttons (same pattern as REVSettingsDialog)
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self._validate_and_accept)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addStretch() # Pushes buttons to the right
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

    def _validate_and_accept(self):
        """Collect the checked boxes; refuse to close if nothing is selected."""

        selected = [name for name, _ in self.POLYTOPE_OPTIONS if self._checkboxes[name].isChecked()]

        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select at least one function.")
            return  # don't close the dialog - let the user fix it
        
        self.selected_polytopes = selected
        self.accept()  # close the dialog with QDialog.Accepted

    def get_selected_polytopes(self):
        """Return the list of selected internal names, or None if the dialog was cancelled."""
        return self.selected_polytopes
    


            
