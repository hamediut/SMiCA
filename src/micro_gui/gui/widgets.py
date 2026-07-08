"""
Custom Qt widgets for the Micro_GUI application.
"""

import numpy as np
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QMouseEvent


class ImageDisplayWidget(QLabel):
    """
    Custom QLabel that tracks mouse movement and displays pixel values.

    This widget extends QLabel to provide real-time pixel coordinate and value
    information as the user moves the mouse over the displayed image.

    Attributes:
        image_data (np.ndarray): The raw image data for pixel value lookup
        parent_window: Reference to the parent window for status updates
    """

    def __init__(self, parent=None):
        """
        Initialize the ImageDisplayWidget.

        Args:
            parent: The parent widget (typically the main window)
        """
        super().__init__(parent)
        self.setMouseTracking(True)
        self.image_data = None
        self.parent_window = parent

    def set_image_data(self, image_data: np.ndarray):
        """
        Store the raw image data for pixel value lookup.

        Args:
            image_data: NumPy array containing the image data
        """
        self.image_data = image_data

    def mouseMoveEvent(self, event: QMouseEvent):
        """
        Handle mouse movement to display pixel values in the status bar.

        Args:
            event: Mouse event containing position information
        """
        if self.image_data is not None and self.pixmap() and not self.pixmap().isNull():
            # Get mouse position relative to the label
            pos = event.pos()

            # Get the pixmap dimensions
            pixmap_rect = self.pixmap().rect() # actual image dimensions (e.g., 512x512)
            label_rect = self.rect() # the label dimensions (e.g., 800x600)

            # Calculate the offset (centering)
            x_offset = (label_rect.width() - pixmap_rect.width()) // 2
            y_offset = (label_rect.height() - pixmap_rect.height()) // 2

            # Adjust mouse position relative to the pixmap
            adjusted_x = pos.x() - x_offset
            adjusted_y = pos.y() - y_offset

            # Check if mouse is within the pixmap bounds
            if 0 <= adjusted_x < pixmap_rect.width() and 0 <= adjusted_y < pixmap_rect.height(): # Mouse is over the image?
                # Calculate scaling factors
                scale_x = self.image_data.shape[1] / pixmap_rect.width()
                scale_y = self.image_data.shape[0] / pixmap_rect.height()

                # Get the actual pixel coordinates in the original image
                pixel_x = int(adjusted_x * scale_x)
                pixel_y = int(adjusted_y * scale_y)

                # Ensure we're within bounds
                if 0 <= pixel_y < self.image_data.shape[0] and 0 <= pixel_x < self.image_data.shape[1]:
                    pixel_value = self.image_data[pixel_y, pixel_x]
                    if self.parent_window:
                        self.parent_window.update_status(
                            f"Position: ({pixel_x}, {pixel_y}) | Pixel Value: {pixel_value}"
                        )
            else:
                if self.parent_window:
                    self.parent_window.update_status("")

        super().mouseMoveEvent(event)
