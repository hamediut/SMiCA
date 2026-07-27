"""Image utility functions."""

import numpy as np
from PIL import Image
import tifffile


def get_image_info(image_data):
    """
    Get basic information about an image.
    
    Args:
        image_data: numpy array
        
    Returns:
        dict with image info
    """
    return {
        'shape': image_data.shape,
        'dtype': str(image_data.dtype),
        'min': float(image_data.min()),
        'max': float(image_data.max()),
        'mean': float(image_data.mean())
    }

def load_multipage_tif(file_path):
    """
    Load one TIF file. Returns a 2D (Y, X) array if it's a single-page file, or a 3D
    (Z, Y, X) array if it's a multi-page file (e.g. an XCT volume saved as one stacked
    TIF).

    Standalone (no GUI dependency) so it can be called from a background QThread.
    Uses tifffile instead of PIL's Image.open()+.seek() loop - PIL re-scans the TIFF's
    page directory on every .seek() call, which is much slower for many-page volumes.
    """
    return tifffile.imread(file_path)