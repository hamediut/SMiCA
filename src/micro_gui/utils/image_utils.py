"""Image utility functions."""

import os
import sys

import numpy as np
from PIL import Image
import tifffile


def long_path_safe(file_path):
    """
    Return a version of `file_path` that Windows can open even when it is longer than
    260 characters (the legacy MAX_PATH limit), which is easy to hit with deep
    instrument-export folders plus long slice filenames.

    Windows lifts that limit for paths written in the extended-length form `\\\\?\\C:\\...`
    (or `\\\\?\\UNC\\server\\share\\...` for network paths). That form requires an absolute
    path with backslashes only - `os.path.abspath` takes care of both. On other operating
    systems the path is returned unchanged.
    """
    if sys.platform != "win32":
        return file_path

    path = os.path.abspath(file_path)
    if path.startswith("\\\\?\\"):
        return path                             # already extended-length
    if path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path[2:]        # \\server\share\... -> \\?\UNC\server\share\...
    return "\\\\?\\" + path                     # X:\...            -> \\?\X:\...


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
    return tifffile.imread(long_path_safe(file_path))