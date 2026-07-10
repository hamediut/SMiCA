"""
Dialog for selecting which polytope functions to calculate.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton,
    QMessageBox, QGroupBox
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
        ('p3h', 'P3H (triangle, horizontal)'),
        ('p3v', 'P3V (triangle, vertical)'),
        ('p4', 'P4 (square)'),
        ('p6', 'P6 (hexagon)'),
        ('L', 'L (lineal path)'),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)

        self.selected_polytopes = None  # will hold the chosen names after OK

        self.setWindowTitle("Polytope Calculation Settings")
        self.setModal(True) ## Block interaction with main window, blocks main window, User MUST respond before using main window.
        self.setMinimumWidth(300)

        self._checkboxes = {}  # internal name -> QCheckBox, so we can read them back in _validate_and_accept
        self._setup_ui()

    
    def _setup_ui(self):
        """ Set up the dialog UI. """

        layout = QVBoxLayout(self)

        group = QGroupBox("Select functions to calculate")
        group_layout = QVBoxLayout(group)

        for internal_name, label in self.POLYTOPE_OPTIONS:
            checkbox = QCheckBox(label)
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
    


            
