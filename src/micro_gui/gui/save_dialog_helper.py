"""
Small shared helper so every Save/Export dialog in the app opens in the folder the user
last saved something to, instead of always defaulting back to the working directory.

Uses QSettings (Qt's built-in persistent-settings mechanism) rather than a plain module-level
variable, so this is remembered across app restarts too, not just within one session. It
automatically stores itself under this app's name/organization, which main.py already sets
via QApplication.setApplicationName()/setOrganizationName() - no extra setup needed here.
"""

import os
from PySide6.QtCore import QSettings

_SETTINGS_KEY = "last_save_directory"
_OPEN_SETTINGS_KEY = "last_open_directory"


def suggested_save_path(default_filename: str) -> str:
    """
    Build the path to hand to QFileDialog.getSaveFileName() as its "dir" argument: the last
    remembered save folder (if any) plus the given default filename, so the dialog opens
    there with that name pre-filled. Falls back to just the bare filename (Qt's normal
    default behaviour - opens in the working directory) the first time, before anything has
    been saved yet.
    """
    last_dir = QSettings().value(_SETTINGS_KEY, "")
    if last_dir:
        return os.path.join(last_dir, default_filename)
    return default_filename


def remember_save_dir(file_path: str) -> None:
    """Call this after a successful save, so the NEXT Save dialog starts in the same folder."""
    folder = os.path.dirname(file_path)
    if folder:
        QSettings().setValue(_SETTINGS_KEY, folder)


def suggested_open_dir() -> str:
    """
    The folder to hand to QFileDialog.getOpenFileName()/getExistingDirectory() as its "dir"
    argument: the last folder something was imported/opened from, if any. Falls back to ""
    (Qt's normal default - the working directory) the first time, before anything has been
    opened yet.

    Kept separate from the save-folder memory above (different QSettings key) since where you
    last opened data from and where you last saved results to are usually different folders.
    """
    return QSettings().value(_OPEN_SETTINGS_KEY, "")


def remember_open_dir(path: str) -> None:
    """
    Call this after a successful open/import, so the NEXT Open/Import dialog starts in the
    same place. Accepts either a file path (its containing folder is remembered) or a folder
    path itself (e.g. from getExistingDirectory).
    """
    if not path:
        return
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    if folder:
        QSettings().setValue(_OPEN_SETTINGS_KEY, folder)