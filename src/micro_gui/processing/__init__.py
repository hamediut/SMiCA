"""
Image processing operations for the Micro_GUI application (the tools behind the
Process menu), kept free of any Qt code so they can run in background threads and
be unit-tested on plain numpy arrays.
"""

from .cropping import (
    CropBox,
    crop_image,
    exterior_value_candidates,
    find_center_crop_box,
)

__all__ = [
    'CropBox',
    'crop_image',
    'exterior_value_candidates',
    'find_center_crop_box',
]
