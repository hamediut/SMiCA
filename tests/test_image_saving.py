"""
Unit tests for the File > Save As helpers in utils/image_utils.py.

Every saved file is read back the same way the app imports it again: tifffile for a
single (multi-page) TIFF (File > Import Single 2D/3D Image), PIL for the slice files of
a sequence (File > Import Single 3D Volume (from Z-Slice Files)).
"""

import numpy as np
import pytest
from PIL import Image

from src.micro_gui.utils.image_utils import (
    default_slice_names,
    load_multipage_tif,
    save_tiff,
    save_tiff_sequence,
)


def make_volume(shape=(6, 20, 30), seed=0):
    """A small multi-label uint8 volume (values 0-3)."""
    return np.random.default_rng(seed).integers(0, 4, size=shape, dtype=np.uint8)


def test_default_slice_names_are_zero_padded_and_sortable():
    assert default_slice_names("rock", 3) == ["rock_0000.tif", "rock_0001.tif", "rock_0002.tif"]

    # More than 10 000 slices needs a 5th digit, otherwise "rock_10000" would sort before "rock_2000".
    names = default_slice_names("rock", 10001)
    assert names[-1] == "rock_10000.tif"
    assert names == sorted(names)


def test_save_tiff_volume_reads_back_as_the_same_volume(tmp_path):
    volume = make_volume()
    path = tmp_path / "volume.tif"

    save_tiff(str(path), volume)

    back = load_multipage_tif(str(path))
    assert back.shape == volume.shape
    assert back.dtype == volume.dtype
    np.testing.assert_array_equal(back, volume)


def test_save_tiff_2d_image(tmp_path):
    image = make_volume(shape=(1, 25, 40))[0]
    path = tmp_path / "image.tif"

    save_tiff(str(path), image)

    np.testing.assert_array_equal(load_multipage_tif(str(path)), image)


def test_save_tiff_keeps_narrow_volume_grayscale(tmp_path):
    # 3 pixels wide: without photometric='minisblack' this could be stored as RGB.
    volume = make_volume(shape=(5, 10, 3))
    path = tmp_path / "narrow.tif"

    save_tiff(str(path), volume)

    np.testing.assert_array_equal(load_multipage_tif(str(path)), volume)


def test_save_tiff_writes_bool_as_zero_one_uint8(tmp_path):
    mask = make_volume() > 1
    path = tmp_path / "mask.tif"

    save_tiff(str(path), mask)

    back = load_multipage_tif(str(path))
    assert back.dtype == np.uint8
    np.testing.assert_array_equal(back, mask.astype(np.uint8))


def test_save_tiff_reports_progress_per_page(tmp_path):
    calls = []
    save_tiff(str(tmp_path / "v.tif"), make_volume(), progress_callback=lambda done, total: calls.append((done, total)))
    assert calls == [(z, 6) for z in range(1, 7)]


def test_save_tiff_sequence_writes_one_file_per_slice_with_given_names(tmp_path):
    volume = make_volume()
    names = [f"scan_{z:04d}.tif" for z in range(250, 256)]  # e.g. slices kept after a crop
    calls = []

    save_tiff_sequence(str(tmp_path), volume, names, progress_callback=lambda d, t: calls.append((d, t)))

    assert sorted(p.name for p in tmp_path.iterdir()) == names
    for z, name in enumerate(names):
        np.testing.assert_array_equal(np.array(Image.open(tmp_path / name)), volume[z])
    assert calls[-1] == (6, 6)


def test_save_tiff_sequence_rejects_wrong_number_of_names(tmp_path):
    with pytest.raises(ValueError, match="file names"):
        save_tiff_sequence(str(tmp_path), make_volume(), ["only_one.tif"])


def test_save_tiff_sequence_rejects_2d_image(tmp_path):
    with pytest.raises(ValueError, match="3D"):
        save_tiff_sequence(str(tmp_path), make_volume()[0], ["a.tif"])
