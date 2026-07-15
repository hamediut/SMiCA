"""
Main entry point for the Micro_GUI application.
"""

import sys
import os

# Fix Qt plugin path issue for frozen applications
if hasattr(sys, 'frozen'):
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(
        sys._MEIPASS, 'PySide6', 'plugins', 'platforms'
    )
else:
    import PySide6
    plugin_path = os.path.join(os.path.dirname(PySide6.__file__), 'plugins', 'platforms')
    if os.path.exists(plugin_path):
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

from PySide6.QtWidgets import QApplication
from .gui.image_viewer import ImageViewer


def main():
    """
    Main entry point for the Micro_GUI application.

    This function initializes the Qt application and displays
    the main image viewer window.
    """
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setApplicationName("Micro_GUI")
    app.setOrganizationName("Microstructure Analysis")

    window = ImageViewer()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
