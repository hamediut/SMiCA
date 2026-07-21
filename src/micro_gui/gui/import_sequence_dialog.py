"""
Dialog for importing an ordered sequence of image files from a folder (e.g. Z-slices of a
3D volume, or 2D/3D images at different time steps), sorted by a number extracted from each
filename.
"""

import os
import re

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QComboBox, QTableWidget, QTableWidgetItem, QMessageBox, QRadioButton, QGroupBox,
    QSpinBox
)

from .save_dialog_helper import suggested_open_dir, remember_open_dir


class ImportSequenceDialog(QDialog):

    """
    Lets the user pick a folder of image files and sort them by a number found in each
    filename. The number's POSITION within the filename (1st, 2nd, ..., last) is
    user-selectable, since naming conventions vary between users/datasets - e.g. a filename
    might contain a date AND a slice index, and only one of those should drive the sort.

    Optionally (when `ask_file_type=True`) also lets the user say whether each file is a
    single 2D slice or a full 3D volume - used by the "Import Time Series" entry point,
    where both are valid, but not by "Import Volume from Slice Folder" (always 2D slices,
    so that choice is hidden there).
    """

    def __init__(self, ask_file_type: bool = False, parent = None):


        super().__init__(parent)

        self.ask_file_type =  ask_file_type

        self.folder_path = None
        self.sorted_file_paths = None  # list of full paths, in final sort order
        self.sorted_values = None      # the actual numbers extracted from each filename, same order as sorted_file_paths
        self.is_3d_files = False       # only meaningful if ask_file_type=True

        self._file_names = []          # filenames only (no path)
        self._candidate_numbers = []   # numbers found in each filename, left to right
        self._sorted_names_preview = []

        self.setWindowTitle("Import Image Sequence")
        self.setModal(True)
        self.setMinimumSize(520, 420)

        self._setup_ui()


    def _setup_ui(self):

        layout = QVBoxLayout(self)

        folder_row = QHBoxLayout()
        self.folder_label = QLabel("No folder selected.")
        self.folder_label.setWordWrap(True)

        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_folder)

        folder_row.addWidget(self.folder_label, stretch=1)
        folder_row.addWidget(browse_button)
        layout.addLayout(folder_row)

        if self.ask_file_type:

            type_group = QGroupBox("Each file in the folder is:")
            type_layout = QHBoxLayout()

            self.radio_2d = QRadioButton("a single 2D slice")
            self.radio_3d = QRadioButton("a full 3D volume")

            self.radio_2d.setChecked(True)

            type_layout.addWidget(self.radio_2d)
            type_layout.addWidget(self.radio_3d)

            type_group.setLayout(type_layout)

            layout.addWidget(type_group)

        number_row = QHBoxLayout()
        number_row.addWidget(QLabel("Sort by:"))
        self.number_combo = QComboBox()
        self.number_combo.currentIndexChanged.connect(self._update_preview)
        number_row.addWidget(self.number_combo, stretch = 1)
        layout.addLayout(number_row)

        select_row = QHBoxLayout()
        select_all_button = QPushButton("Select All")
        select_all_button.clicked.connect(lambda: self._set_all_checked(True))
        select_none_button = QPushButton("Select None")
        select_none_button.clicked.connect(lambda: self._set_all_checked(False))
        select_row.addWidget(select_all_button)
        select_row.addWidget(select_none_button)

        select_row.addSpacing(20) # # visual gap before the range controls

        select_row.addWidget(QLabel("Range - sort value from:"))
        self.range_from_spin =  QSpinBox()
        self.range_from_spin.setRange(0, 0)
        select_row.addWidget(self.range_from_spin)
        select_row.addWidget(QLabel("to:"))
        self.range_to_spin = QSpinBox()
        self.range_to_spin.setRange(0, 0)
        select_row.addWidget(self.range_to_spin)
        select_range_button = QPushButton("Select Range")
        select_range_button.clicked.connect(self._select_range)
        select_row.addWidget(select_range_button)

        select_row.addStretch()
        layout.addLayout(select_row)


        self.preview_table = QTableWidget(0, 3)
        self.preview_table.setHorizontalHeaderLabels(["Include" , "File name", "Sort value"])
        self.preview_table.horizontalHeader().setStretchLastSection(True)
        self.preview_table.setColumnWidth(0, 55)
        layout.addWidget(self.preview_table, stretch=1)

        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self._accept)
        self.ok_button.setEnabled(False)  # nothing to confirm until a folder is picked

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)


    def _browse_folder(self):


        folder = QFileDialog.getExistingDirectory(self, "Select Folder", suggested_open_dir())
        if not folder:
            return
        remember_open_dir(folder)  # so the next Open/Import dialog starts in this same folder

        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(('tif', 'tiff')))

        if not files:
            QMessageBox.warning(self, "No Images Found", "No tif/tiff files were found in the folder")
            return
        
        self.folder_path = folder
        self._file_names = files
        self._candidate_numbers = [re.findall(r'\d+', name) for name in files]
        self.folder_label.setText(f"{folder} ({len(files)} files)")

        # Populate "sort by" based on the file with the MOST numbers found, so there are
        # always enough dropdown options - e.g. a folder mixing "img_5.tif" and
        # "img_2024_05.tif" gets up to 2 positional options plus "Last number".
        
        max_numbers = max((len(nums) for nums in self._candidate_numbers), default=0)
        self.number_combo.blockSignals(True)
        self.number_combo.clear()
        ordinals = ["1st", "2nd", "3rd"] + [f"{i}th" for i in range(4, max_numbers + 1)]
        for i in range(max_numbers):
            self.number_combo.addItem(f"{ordinals[i]} number in filename")
        if max_numbers > 1:
            self.number_combo.addItem("Last number in filename")
            self.number_combo.setCurrentIndex(self.number_combo.count() - 1)  # default: most robust choice
        self.number_combo.blockSignals(False)

        if max_numbers == 0:
            QMessageBox.warning(self, "No Numbers Found", "None of the filenames contain a number to sort by.")
        self._update_preview()

    def _selected_occurrence_index(self, n_numbers_in_file: int) -> int:
        """Map the current dropdown choice to an index into one file's own list of found numbers."""

        if self.number_combo.currentText().startswith("Last"):
            return n_numbers_in_file - 1  # relative to THIS file, so this stays robust even if counts vary
        return self.number_combo.currentIndex()  # a fixed position, e.g. "2nd number" -> index 1 for every file
    
    def _update_preview(self):
        self.preview_table.setRowCount(0)
        if not self._file_names:
            return
            
        rows = []
        missing = []
        for name, numbers in zip(self._file_names, self._candidate_numbers):
            idx = self._selected_occurrence_index(len(numbers))
            if 0 <= idx < len(numbers):
                rows.append((name, int(numbers[idx])))
            else:
                missing.append(name)

        if missing:
            QMessageBox.warning(
                self, "Inconsistent File Names",
                f"{len(missing)} file(s) don't have that many numbers in their name, "
                f"e.g.: {missing[0]}\nPick a different sort option, or rename these files."
            )
            self.ok_button.setEnabled(False)
            return
            
        rows.sort(key=lambda pair: pair[1])

        self.preview_table.setRowCount(len(rows))
        for row_idx, (name, value) in enumerate(rows):
            include_item = QTableWidgetItem()
            include_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            include_item.setCheckState(Qt.Checked)  # default: everything selected, same as before this feature existed
            self.preview_table.setItem(row_idx, 0, include_item)

            self.preview_table.setItem(row_idx, 1, QTableWidgetItem(name))
            self.preview_table.setItem(row_idx, 2, QTableWidgetItem(str(value)))

        self._sorted_names_preview = [name for name, _ in rows]
        self._sorted_values_preview = [value for _, value in rows]  # the real extracted numbers, same order
        
        # Keep the range spinboxes in sync with whatever sort values actually exist, and
        # default to the full range - re-runs every time the folder or sort choice changes.
        self.range_from_spin.setRange(min(self._sorted_values_preview), max(self._sorted_values_preview))
        self.range_to_spin.setRange(min(self._sorted_values_preview), max(self._sorted_values_preview))
        self.range_from_spin.setValue(min(self._sorted_values_preview))
        self.range_to_spin.setValue(max(self._sorted_values_preview))
        
        self.ok_button.setEnabled(True)

    def _set_all_checked(self, checked: bool):
        """Toggle every row's checkbox at once - used by the Select All/None buttons."""

        state = Qt.Checked if checked else Qt.Unchecked
        for row in range(self.preview_table.rowCount()):
            self.preview_table.item(row, 0).setCheckState(state)

    def _select_range(self):
        """Check only the rows whose sort value falls within [from, to] - unchecks everything
        else, so this always sets the selection to exactly that range rather than adding to
        whatever was checked before."""

        lo, hi = self.range_from_spin.value(), self.range_to_spin.value()

        if lo > hi:
            QMessageBox.warning(self, "Invalid Range", " 'From' must be less than or equalto 'To'.")
            return
        for row, value in enumerate(self._sorted_values_preview):
            state = Qt.Checked if lo <= value <= hi else Qt.Unchecked
            self.preview_table.item(row, 0).setCheckState(state)


    def _accept(self):
        checked_rows = [
            row for row in range(self.preview_table.rowCount())
            if self.preview_table.item(row, 0).checkState() == Qt.Checked
        ]
        if not checked_rows:
            QMessageBox.warning(self, "No files Selected", "Please check at least one file to import.")
            return
        
        self.sorted_file_paths = [os.path.join(self.folder_path, self._sorted_names_preview[row]) for row in checked_rows]
        self.sorted_values = [self._sorted_values_preview[row] for row in checked_rows]  # same rows, same order
        if self.ask_file_type:
            self.is_3d_files = self.radio_3d.isChecked()
        self.accept()


    def get_sorted_file_paths(self):
        """Return the list of full file paths, in sorted order, or None if cancelled."""
        return self.sorted_file_paths

    def get_sorted_values(self):
        """Return the actual numbers extracted from each filename, in the same order as get_sorted_file_paths()."""
        return self.sorted_values
    
    def get_is_3d_files(self):
        """Only meaningful when ask_file_type=True: True if each file is a 3D volume, False if 2D."""
        return self.is_3d_files
    

            
        










