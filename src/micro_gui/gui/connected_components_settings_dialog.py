"""
Dialog for connected-components labeling settings.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QGroupBox, QFormLayout
)

class ConnectedComponentsSettingsDialog(QDialog):

    """
    Dialog to configure connected-components labeling for the current 2D or
    3D image. Just the connectivity choice here. Voxel size, unit, and minimum
    size now live in MeasurementsSettingsDialog, since labeling itself
    doesn't need physical units, only the measurement step that follows it.
    """

    def __init__(self, is_3d: bool = True, parent = None):
        super().__init__(parent)


        self.is_3d = is_3d
        self.connectivity = None

        self.setWindowTitle("Connected Components Settings")
        self.setModal(True)
        self.setMinimumWidth(360)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        dims = "3D volume" if self.is_3d else "2D image"

        info_label = QLabel(
            f"Label connected components of the foreground phase (value 1) "
            f"in this {dims}."
        )

        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        settings_group = QGroupBox("Connected Components Settings")
        form_layout = QFormLayout()

        # Text carries the actual connectivity number so _accept() can parse
        # it straight back out - avoids a second combo -> int lookup table.
        self.connectivity_combo = QComboBox()
        if self.is_3d:
            self.connectivity_combo.addItems(["6 (face)", "18 (face + edge)", "26 (face + edge + corner)"])
            self.connectivity_combo.setCurrentIndex(2) # default 26, most permissive
            self.connectivity_combo.setToolTip(
                        "Neighbor rule for 'connected' - 6 shares a full face only, 18 also "
                        "merges components touching along just an edge, 26 also merges "
                        "components touching at only a single corner voxel."
                    )
        else:
            self.connectivity_combo.addItems(["4 (face)", "8 (face + corner)"])
            self.connectivity_combo.setCurrentIndex(1) # default 8, most permissive
            self.connectivity_combo.setToolTip(
                        "Neighbor rule for 'connected' - 4 shares a full face only, 8 also "
                        "merges components touching along just an edge."
                    )
        form_layout.addRow("Connectivity:", self.connectivity_combo)

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
        self.connectivity = int(self.connectivity_combo.currentText().split()[0])
        self.accept()

    def get_connectivity(self):
        return self.connectivity



      