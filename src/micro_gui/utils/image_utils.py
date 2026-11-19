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


# ---------------------------------------------------------------------------------------
# Saving (File > Save As). Like load_multipage_tif, these have no GUI dependency, so they
# can run in a background QThread and report progress through `progress_callback`.
# ---------------------------------------------------------------------------------------

# Plain TIFF files are limited to 4 GB. Above this size (with some room left for the
# page headers) the file has to be written as a BigTIFF instead.
_BIGTIFF_THRESHOLD_BYTES = 2**32 - 2**25


def default_slice_names(stem, n_slices):
    """
    File names for saving the slices of a volume that has no original slice files, e.g.
    one opened from a single multi-page TIFF: "stem_0000.tif", "stem_0001.tif", ...

    The numbers are zero-padded (at least 4 digits) so the files sort in the right order
    in any file browser, and the folder can be read back with File > Import Single 3D
    Volume (from Z-Slice Files), which sorts by the number in each name.

    Args:
        stem: start of every file name, usually the original file name without extension.
        n_slices: number of slices (Z).

    Returns:
        list of n_slices file names (names only, no folder).
    """
    digits = max(4, len(str(n_slices - 1)))
    return [f"{stem}_{z:0{digits}d}.tif" for z in range(n_slices)]


def _as_tiff_page(page):
    """
    Return a 2D slice in a form every TIFF reader handles.

    Boolean images would be written as 1-bit TIFFs, which can't be appended page by page
    (see save_tiff) and which some programs open differently - so they are written as
    0/1 uint8 instead, the same values a binarized image has in the viewer.
    """
    return page.view(np.uint8) if page.dtype == bool else page


def save_tiff(file_path, image, progress_callback=None):
    """
    Save a 2D (Y, X) image, or a 3D (Z, Y, X) volume, as ONE TIFF file.

    A volume becomes a multi-page TIFF with one page per Z slice - the format
    File > Import Single 2D/3D Image reads (and Fiji/ImageJ opens as a stack).

    The pages are written one at a time, rather than with a single tifffile.imwrite()
    call, only so the caller can show progress while a large volume is being written.

    Args:
        file_path: where to write the file (.tif / .tiff).
        image: 2D or 3D numpy array.
        progress_callback: optional f(pages_written, total_pages), called after each page.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D image or a 3D volume, got an array with shape {image.shape}")

    pages = image[np.newaxis] if image.ndim == 2 else image  # a 2D image is a 1-page "volume"
    n_pages = pages.shape[0]
    bigtiff = image.nbytes > _BIGTIFF_THRESHOLD_BYTES

    with tifffile.TiffWriter(long_path_safe(file_path), bigtiff=bigtiff) as tif:
        for z in range(n_pages):
            # contiguous=True appends each page to the SAME image series, so the file reads
            # back as one (Z, Y, X) array instead of Z separate 2D images.
            # photometric='minisblack' = grayscale; without it tifffile may store a slice
            # that happens to be 3 or 4 pixels wide as an RGB(A) colour image.
            tif.write(_as_tiff_page(pages[z]), photometric='minisblack', contiguous=True)
            if progress_callback is not None:
                progress_callback(z + 1, n_pages)


def save_tiff_sequence(folder, image, file_names, progress_callback=None):
    """
    Save every slice of a 3D (Z, Y, X) array as its own single-page TIFF file in `folder`:
    slice z is written to file_names[z].

    Files that already exist with those names are overwritten - the caller is
    responsible for asking the user first.

    Args:
        folder: existing folder to write into.
        image: 3D numpy array, one slice per file along axis 0.
        file_names: one file name per slice (names only, no folder), e.g. from
            default_slice_names() or the original slice files' names.
        progress_callback: optional f(files_written, total_files), called after each file.
    """
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D (Z, Y, X) array, got an array with shape {image.shape}")
    if len(file_names) != image.shape[0]:
        raise ValueError(f"Got {len(file_names)} file names for {image.shape[0]} slices")

    n_slices = image.shape[0]
    for z, name in enumerate(file_names):
        path = long_path_safe(os.path.join(folder, name))
        tifffile.imwrite(path, _as_tiff_page(image[z]), photometric='minisblack')
        if progress_callback is not None:
            progress_callback(z + 1, n_slices)