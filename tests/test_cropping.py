"""
Unit tests for the center-crop tools in processing/cropping.py.

Every test builds a small synthetic sample: a multi-label cylinder (phases 1 and 2)
standing along Z, surrounded by air with value 0.
"""

import numpy as np
import pytest

from src.micro_gui.processing.cropping import (
    CropBox,
    crop_image,
    exterior_value_candidates,
    find_center_crop_box,
)

AIR = 0


def make_cylinder(shape=(60, 80, 90), center=(40, 45), radius=30, z_range=(5, 55), drift_x=0):
    """
    Build a segmented cylinder volume.

    Args:
        shape: (Z, Y, X) of the volume.
        center: (y, x) of the cylinder axis in the first sample slice.
        radius: cylinder radius in pixels.
        z_range: (first, stop) slices that contain the sample; the rest is air.
        drift_x: how many pixels the axis moves in X from the first to the last
            sample slice, to mimic a sample that shifted during the scan.

    Returns:
        (volume, inside): the uint8 volume, and a bool mask that is True wherever the
        sample is (the ground truth the crop box must stay inside).
    """
    n_z, n_y, n_x = shape
    yy, xx = np.mgrid[:n_y, :n_x]
    inside = np.zeros(shape, dtype=bool)
    first, stop = z_range
    for z in range(first, stop):
        shift = drift_x * (z - first) / max(stop - first - 1, 1)
        inside[z] = (yy - center[0]) ** 2 + (xx - center[1] - shift) ** 2 <= radius ** 2

    # Two phases in a striped pattern, so the sample is multi-label like real data.
    phases = np.where((xx + yy) % 7 < 3, 1, 2).astype(np.uint8)
    volume = np.where(inside, phases[np.newaxis], AIR).astype(np.uint8)
    return volume, inside


def test_exterior_candidates_list_air_first():
    volume, _ = make_cylinder()
    candidates = exterior_value_candidates(volume)
    assert candidates[0][0] == AIR
    assert sum(fraction for _, fraction in candidates) == pytest.approx(1.0)


def test_cube_is_cubic_and_inside_sample():
    volume, inside = make_cylinder()
    box = find_center_crop_box(volume, AIR, make_cube=True)

    depth, height, width = box.size_zyx
    assert depth == height == width
    assert inside[box.slices(3)].all()
    # The biggest square in a circle of radius 30 has half-width 30 / sqrt(2) = 21.2,
    # i.e. a side of 43 pixels; allow a pixel of rounding.
    assert width >= 41


def test_column_keeps_full_sample_height():
    volume, inside = make_cylinder(z_range=(5, 55))
    box = find_center_crop_box(volume, AIR, make_cube=False)

    assert (box.z_start, box.z_stop) == (5, 55)
    assert inside[box.slices(3)].all()


def test_pores_with_air_value_inside_sample_are_ignored():
    volume, _ = make_cylinder()
    clean_box = find_center_crop_box(volume, AIR, make_cube=False)

    # Enclosed pores right at the center, labelled with the same value as the air.
    porous = volume.copy()
    porous[:, 38:43, 43:48] = AIR
    porous[:, 50:52, 30:33] = AIR
    porous_box = find_center_crop_box(porous, AIR, make_cube=False)

    assert porous_box == clean_box


def test_margin_shrinks_the_square():
    volume, _ = make_cylinder()
    box = find_center_crop_box(volume, AIR, make_cube=False)
    box_with_margin = find_center_crop_box(volume, AIR, make_cube=False, margin=3)

    assert box_with_margin.size_zyx[2] == box.size_zyx[2] - 2 * 3


def test_center_moves_to_best_spot_when_centroid_is_off():
    # A thin "tail" sticking out of the cylinder pulls the centroid to the right, but
    # the biggest square still sits in the middle of the round part. Without moving the
    # center away from the centroid, the square comes out 2 px narrower.
    volume, inside = make_cylinder()
    plain_box = find_center_crop_box(volume, AIR, make_cube=False)

    volume[5:55, 35:45, 75:89] = 1
    inside[5:55, 35:45, 75:89] = True
    box = find_center_crop_box(volume, AIR, make_cube=False)

    assert inside[box.slices(3)].all()
    assert box.size_zyx == plain_box.size_zyx


def test_drifting_sample_box_fits_every_kept_slice():
    volume, inside = make_cylinder(drift_x=8)
    for make_cube in (True, False):
        box = find_center_crop_box(volume, AIR, make_cube=make_cube)
        assert inside[box.slices(3)].all()


def test_2d_image_gets_biggest_square():
    volume, inside = make_cylinder()
    image, image_inside = volume[30], inside[30]

    box = find_center_crop_box(image, AIR)
    cropped = crop_image(image, box)

    assert (box.z_start, box.z_stop) == (0, 1)
    assert image_inside[box.slices(2)].all()
    assert cropped.ndim == 2
    assert cropped.shape[0] == cropped.shape[1] >= 41


def test_all_air_raises():
    with pytest.raises(ValueError, match="no sample"):
        find_center_crop_box(np.zeros((5, 20, 20), dtype=np.uint8), AIR)


def test_too_large_margin_raises():
    volume, _ = make_cylinder()
    with pytest.raises(ValueError, match="margin"):
        find_center_crop_box(volume, AIR, margin=100)


def test_crop_image_returns_independent_copy():
    volume = np.arange(4 * 5 * 6).reshape(4, 5, 6)
    box = CropBox(z_start=1, z_stop=3, y_start=0, y_stop=2, x_start=2, x_stop=5)

    cropped = crop_image(volume, box)

    assert cropped.shape == (2, 2, 3)
    np.testing.assert_array_equal(cropped, volume[1:3, 0:2, 2:5])
    assert cropped.base is None  # owns its data, so the original can be freed
