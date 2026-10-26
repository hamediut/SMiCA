"""
Dialog for configuring how a histogram is binned and scaled before plotting.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton, QGroupBox, QFormLayout
)


class HistogramSettingsDialog(QDialog):
    """
    Dialog to configure a histogram plot: a percentile-based x-axis cutoff
    (values above it are folded into the last bin, not dropped), the number
    of bins, and whether the y-axis (count) is log-scaled. Needed because a
    fixed linear x-axis with real microstructure data (many small
    components, a few huge ones) leaves almost every bin empty otherwise.
    """

    def __init__(self, column_label: str, parent=None):
        super().__init__(parent)

        self.percentile = None
        self.num_bins = None
        self.log_y = None

        self.setWindowTitle("Histogram Settings")
        self.setModal(True)
        self.setMinimumWidth(340)

        self._setup_ui(column_label)

    def _setup_ui(self, column_label):
        layout = QVBoxLayout(self)

        info_label = QLabel(f"Configure the histogram for: {column_label}")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        settings_group = QGroupBox("Binning")
        form_layout = QFormLayout()

        self.percentile_spinbox = QDoubleSpinBox()
        self.percentile_spinbox.setRange(1.0, 100.0)
        self.percentile_spinbox.setDecimals(1)
        self.percentile_spinbox.setValue(99.0)
        self.percentile_spinbox.setToolTip(
            "Values above this percentile are grouped into the last bin instead "
            "of being dropped - lower it to zoom in past extreme outliers "
            "(e.g. one huge percolated component dominating a size distribution)."
        )
        form_layout.addRow("Max value percentile:", self.percentile_spinbox)

        # Bounded range (not unlimited) so a bin count can never be large
        # enough to reproduce the bins='auto' hang, no matter what's typed in.
        self.bins_spinbox = QSpinBox()
        self.bins_spinbox.setRange(5, 200)
        self.bins_spinbox.setValue(30)
        self.bins_spinbox.setToolTip(
            "Number of equal-width bins between the minimum value and the "
            "percentile cutoff above."
        )
        form_layout.addRow("Number of bins:", self.bins_spinbox)

        settings_group.setLayout(form_layout)
        layout.addWidget(settings_group)

        self.log_y_checkbox = QCheckBox("Log-scale y-axis (component count)")
        self.log_y_checkbox.setChecked(False)
        self.log_y_checkbox.setToolTip(
            "Keeps small bins visible instead of being flattened by one very "
            "tall bin - useful for the same kind of skewed data the "
            "percentile cutoff above is meant for."
        )
        layout.addWidget(self.log_y_checkbox)

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
        self.percentile = self.percentile_spinbox.value()
        self.num_bins = self.bins_spinbox.value()
        self.log_y = self.log_y_checkbox.isChecked()
        self.accept()

    def get_percentile(self):
        return self.percentile

    def get_num_bins(self):
        return self.num_bins

    def get_log_y(self):
        return self.log_y
