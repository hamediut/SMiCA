"""
Dialog for selecting which pixel value should become the foreground (1) when binarizing
a multi-label segmented image.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel
)

class BinarizeDialog(QDialog):
    """
    Dialog to let the user pick a foreground label value from the labels actually present
    in the current image. Everything that isn't the chosen value becomes background (0) -
    this works even if 0 isn't already one of the image's labels, since the background is
    constructed by the binarize step, not assumed to already exist.

    """

    def __init__(self, unique_values, parent=None):
        """
        Args: 
            unique_values: array/list of the distinct pixel values in the current image.
            parent: parent window
        """

        super().__init__(parent)

        self.foreground_value = None  # will hold the chosen value after OK

        self.setWindowTitle("Binarize Image")
        self.setModal(False)  # non-modal, so the user can still interact with (and hover over) the main window while this is open
        self.setMinimumWidth(300)

        self._unique_values = list(unique_values)
        self._setup_ui()

    def _setup_ui(self):
        """ Set up the dialog UI. """

        layout = QVBoxLayout(self)

        info_label = QLabel(
            "Select the pixel value that represents the phase of interest (foreground).\n"
            "Every other value in the image will be converted to background (0) during binarization."
        )

        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        self.value_combo = QComboBox()
        for value in self._unique_values:
            self.value_combo.addItem(str(value))
        layout.addWidget(self.value_combo)

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
        """ Handle OK button click. """

        # currentIndex() lines up with self._unique_values since we added items in that
        # same order - this reads back the actual numpy value, not the display strin

        self.foreground_value = self._unique_values[self.value_combo.currentIndex()]
        self.accept()

    def get_foreground_value(self):
        """Return the chosen foreground pixel value, or None if the dialog was cancelled."""

        return self.foreground_value
