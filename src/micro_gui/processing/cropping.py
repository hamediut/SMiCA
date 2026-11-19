"""
Cropping tools for segmented microstructure images.

The main tool here is the *center crop*. XCT scans of rock cores and similar samples
usually show a cylinder of material surrounded by air. Most analyses (SMDs, REV,
Minkowski functionals, ...) only make sense on the material itself, so this module
finds the largest box around the center of the sample that lies completely inside
it in every slice it covers, and cuts it out. Real samples are rarely perfect
cylinders and may shift a little from slice to slice, so the box is fitted to the
actual sample outline in each slice rather than to an ideal circle.

Axis convention (same as the rest of the app): a 3D volume is (Z, Y, X), with Z the
vertical axis / slice index, and a 2D image is (Y, X).
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy import ndimage


# Upper limit on how many times find_center_crop_box() moves the center to a better
# position. It normally settles after one or two moves; this just guarantees it stops.
MAX_CENTER_REFINEMENTS = 5


@dataclass(frozen=True)
class CropBox:
    """
    A box to cut out of an image, in pixel/voxel indices.

    Starts are inclusive and stops are exclusive, exactly like Python/numpy slicing,
    so `x_stop - x_start` is the width in pixels. For a 2D image the Z range is just
    (0, 1) and is ignored when cropping.
    """

    z_start: int
    z_stop: int
    y_start: int
    y_stop: int
    x_start: int
    x_stop: int

    @property
    def size_zyx(self) -> Tuple[int, int, int]:
        """(depth, height, width) of the box, in voxels."""
        return (self.z_stop - self.z_start, self.y_stop - self.y_start, self.x_stop - self.x_start)

    def slices(self, ndim: int) -> tuple:
        """
        The box as a tuple of slice objects, ready for indexing:
        `image[box.slices(image.ndim)]`.

        Args:
            ndim: 2 for a (Y, X) image, 3 for a (Z, Y, X) volume.
        """
        yx = (slice(self.y_start, self.y_stop), slice(self.x_start, self.x_stop))
        if ndim == 2:
            return yx
        return (slice(self.z_start, self.z_stop),) + yx


def exterior_value_candidates(image: np.ndarray) -> List[Tuple[object, float]]:
    """
    List the pixel values found along the outer edges of the image, most common first.

    The air around a cylindrical sample always reaches the image edges (the corners of
    every slice lie outside a cylinder), so the exterior value has to be one of these,
    and is normally the most common one. Looking only at the four edges of each slice
    is also much cheaper than np.unique() over a whole volume.

    Args:
        image: 2D (Y, X) image or 3D (Z, Y, X) volume.

    Returns:
        [(value, fraction_of_edge_pixels), ...] from most to least common, e.g.
        [(0, 0.81), (2, 0.12), (1, 0.07)]. Values are plain Python numbers.
    """
    # The "..." makes this work for both 2D and 3D: for a volume it takes the
    # first/last row and the first/last column of EVERY slice.
    edges = np.concatenate([
        image[..., 0, :].ravel(), image[..., -1, :].ravel(),
        image[..., :, 0].ravel(), image[..., :, -1].ravel(),
    ])
    values, counts = np.unique(edges, return_counts=True)
    order = np.argsort(counts)[::-1]  # most common first
    return [(values[i].item(), counts[i] / edges.size) for i in order]


def exterior_mask_2d(slice_2d: np.ndarray, exterior_value) -> np.ndarray:
    """
    Find the air around the sample in one 2D slice.

    A pixel counts as exterior only if it has the exterior value AND is connected to
    the edge of the slice through other exterior-valued pixels. This matters because
    pores inside the sample are often labelled with the same value as the air. Those
    pores are enclosed by sample and never reach the edge, so they are (correctly)
    not treated as outside the sample.

    Args:
        slice_2d: one (Y, X) slice.
        exterior_value: the pixel value of the air/exterior.

    Returns:
        Boolean (Y, X) array, True where the pixel is outside the sample.
    """
    candidates = slice_2d == exterior_value

    # Give every connected patch of exterior-valued pixels its own id (1, 2, 3, ...;
    # 0 means "not a candidate"). The default connectivity is 4 (shared edges only),
    # so pores that touch only at a corner don't count as connected.
    patch_ids, _ = ndimage.label(candidates)

    # Ids of the patches that touch any of the four edges of the slice = the air.
    edge_ids = np.unique(np.concatenate((
        patch_ids[0, :], patch_ids[-1, :], patch_ids[:, 0], patch_ids[:, -1],
    )))
    edge_ids = edge_ids[edge_ids != 0]

    return np.isin(patch_ids, edge_ids)


def _unpack_slice(packed_slice: np.ndarray, width: int) -> np.ndarray:
    """Turn one bit-packed slice (8 pixels per byte, see np.packbits) back into a bool (Y, X) mask."""
    return np.unpackbits(packed_slice, axis=-1, count=width).astype(bool)


def _largest_square_half_width(inside: np.ndarray, center_y: int, center_x: int) -> int:
    """
    Half-width h of the biggest square centered on (center_y, center_x) that lies
    completely inside the sample and inside the slice. The square covers rows
    center_y-h .. center_y+h and columns center_x-h .. center_x+h, so its side is 2h+1.

    Args:
        inside: boolean (Y, X) mask, True where the pixel belongs to the sample.

    Returns:
        h >= 0, or -1 if the center pixel itself is outside the sample (e.g. a slice
        above or below it).
    """
    height, width = inside.shape
    # Largest half-width the slice edges allow, before looking at any pixel values.
    edge_limit = min(center_y, center_x, height - 1 - center_y, width - 1 - center_x)

    # Binary search. "A square of half-width h fits" is true for every h up to the
    # answer and false for every h after it, so instead of growing the square one
    # pixel at a time we halve the range of possible answers on every step.
    fits, too_big = -1, edge_limit + 1  # h = -1 (no square) always fits; edge_limit + 1 never does
    while too_big - fits > 1:
        h = (fits + too_big) // 2
        window = inside[center_y - h: center_y + h + 1, center_x - h: center_x + h + 1]
        if window.all():
            fits = h
        else:
            too_big = h
    return fits


def _deepest_point(inside: np.ndarray, near: Tuple[int, int]) -> Tuple[int, int]:
    """
    The pixel lying deepest inside a 2D mask, i.e. the center of the biggest square
    that fits in its True region.

    Uses the chessboard distance transform: for every True pixel, the distance to the
    nearest False pixel, counted the way a chess king moves (diagonal steps count as 1).
    A square of half-width h fits around a pixel exactly when that distance is h + 1
    or more, so the deepest pixel is the center of the biggest square.

    Args:
        inside: boolean (Y, X) mask.
        near: (y, x). If several pixels are equally deep, the one closest to this point
            is returned, so the center doesn't jump around between equally good spots.

    Returns:
        (y, x) of the deepest pixel.
    """
    # Pad with False so the image edge also counts as "outside" - otherwise the
    # transform would happily let a square stick out of the image.
    padded = np.pad(inside, 1, constant_values=False)
    depth = ndimage.distance_transform_cdt(padded, metric='chessboard')[1:-1, 1:-1]

    deepest = np.argwhere(depth == depth.max())  # (k, 2) array of (y, x) positions
    closest = np.argmin(((deepest - np.array(near)) ** 2).sum(axis=1))
    return int(deepest[closest][0]), int(deepest[closest][1])


def _longest_run(flags: np.ndarray) -> Tuple[int, int]:
    """
    Find the longest stretch of consecutive True values in a 1D boolean array.

    Returns:
        (start, stop) with stop exclusive, e.g. [F, T, T, T, F, T] -> (1, 4).
        (0, 0) if there is no True value at all.
    """
    # Pad with 0 on both sides, then diff: +1 marks where a run starts and -1 marks
    # the position just after a run ends.
    steps = np.diff(np.concatenate(([0], flags.astype(np.int8), [0])))
    starts = np.flatnonzero(steps == 1)
    stops = np.flatnonzero(steps == -1)
    if starts.size == 0:
        return 0, 0
    longest = np.argmax(stops - starts)
    return int(starts[longest]), int(stops[longest])


def _fit_cube(half_widths: np.ndarray) -> Tuple[int, int, int]:
    """
    Biggest cube that fits the per-slice square sizes.

    A cube of half-width h needs 2h+1 consecutive slices that each fit a square of
    half-width h or more. Try h from the largest value downwards and stop at the first
    one that has a long enough run of slices.

    Returns:
        (h, z_start, z_stop), with z_stop exclusive.
    """
    for h in range(int(half_widths.max()), -1, -1):
        run_start, run_stop = _longest_run(half_widths >= h)
        side = 2 * h + 1
        if run_stop - run_start >= side:
            # The run can be taller than the cube - put the cube in its middle.
            z_start = run_start + (run_stop - run_start - side) // 2
            return h, z_start, z_start + side
    # Only reachable if no slice fits even a 1-pixel square, which the caller rules out.
    raise ValueError("No cube fits inside the sample.")


def _fit_column(half_widths: np.ndarray) -> Tuple[int, int, int]:
    """
    Square column (any height) with the largest volume.

    Every half-width h gives one candidate: the longest run of slices that all fit a
    square of half-width h, times that square's area. Wider squares fit in fewer
    slices, so this weighs width against height, e.g. it drops a rounded end of the
    core rather than shrinking the square for the whole height.

    Returns:
        (h, z_start, z_stop), with z_stop exclusive.
    """
    best = None  # (volume, h, z_start, z_stop)
    for h in range(int(half_widths.max()) + 1):
        run_start, run_stop = _longest_run(half_widths >= h)
        volume = (run_stop - run_start) * (2 * h + 1) ** 2
        # ">=" so that on a tie the wider square wins.
        if best is None or volume >= best[0]:
            best = (volume, h, run_start, run_stop)
    return best[1:]


def find_center_crop_box(
    image: np.ndarray,
    exterior_value,
    make_cube: bool = True,
    margin: int = 0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> CropBox:
    """
    Find the largest box around the center of a sample that lies completely inside the
    sample in every slice it covers.

    How it works:

    1. Find the sample in every slice: everything except the exterior (air) connected
       to the slice edge - see exterior_mask_2d(). This is the slow part, so the masks
       are kept in memory (bit-packed, 1/8 of the size of a uint8 volume) for the
       steps below.
    2. Start with the centroid (average position) of the whole sample as the center.
    3. Measure, in every slice, the biggest square around that center that is all
       sample. Slices above/below the sample, where the center is air, get -1.
    4. From that per-slice profile, choose the slices to keep and the square size:
       - make_cube=True: the biggest cube, placed in the middle of the slices it fits.
       - make_cube=False: the square column with the largest volume. It is usually
         as tall as the sample, but can leave out a few end slices if keeping them
         would force a much narrower square.
    5. Improve the center: take the region that is sample in ALL the kept slices and
       move the center to its deepest point (see _deepest_point). If the sample shifts
       between slices or isn't a perfect cylinder, this is where the biggest common
       square fits - usually a little off the centroid. Repeat from step 3 with the
       new center until it stops moving. Every repeat can only keep or grow the box.

    Args:
        image: segmented 2D (Y, X) image or 3D (Z, Y, X) volume.
        exterior_value: pixel value of the air around the sample.
        make_cube: 3D only - True for a cube, False for a square column (see above).
            Ignored for 2D images, which always get the biggest square.
        margin: extra distance in pixels to keep from the sample boundary in X and Y,
            e.g. to stay clear of blurry edge pixels. 0 lets the box touch the boundary.
        progress_callback: optional function called as progress_callback(done, total)
            after every slice of step 1 (the slow part); the GUI uses it to drive its
            progress bar.

    Returns:
        The CropBox to cut out (for a 2D image its Z range is (0, 1)).

    Raises:
        ValueError: if there is no sample at all, or no square around the center fits
            inside the sample (wrong exterior value, or a margin that is too large).
    """
    is_2d = image.ndim == 2
    # Treat a 2D image as a volume with one slice, so the same code handles both.
    volume = image[np.newaxis] if is_2d else image
    n_slices, height, width = volume.shape

    # --- Step 1 (+ step 2's sums): find the sample in every slice ----------------
    # packbits stores 8 True/False pixels per byte, so keeping every slice's mask costs
    # only 1/8 of the memory of a uint8 volume - much cheaper than relabelling each
    # slice again every time the center moves.
    inside_packed = np.empty((n_slices, height, (width + 7) // 8), dtype=np.uint8)

    # Centroid without collecting every sample pixel's coordinates: count sample pixels
    # per row and per column, then weight the row/column positions by those counts.
    row_positions = np.arange(height)
    col_positions = np.arange(width)
    sum_y = sum_x = n_inside = 0  # plain Python ints, so the sums can't overflow

    for z in range(n_slices):
        inside = ~exterior_mask_2d(volume[z], exterior_value)
        inside_packed[z] = np.packbits(inside, axis=-1)

        pixels_per_row = inside.sum(axis=1)  # shape (height,)
        pixels_per_col = inside.sum(axis=0)  # shape (width,)
        sum_y += int(pixels_per_row @ row_positions)
        sum_x += int(pixels_per_col @ col_positions)
        n_inside += int(pixels_per_row.sum())

        if progress_callback is not None:
            progress_callback(z + 1, n_slices)

    if n_inside == 0:
        raise ValueError(
            f"Every pixel is exterior (value {exterior_value}) - there is no sample to crop."
        )

    # --- Step 2: start at the centroid of the whole sample -----------------------
    center = (int(round(sum_y / n_inside)), int(round(sum_x / n_inside)))

    for _ in range(MAX_CENTER_REFINEMENTS):
        # --- Step 3: biggest square around the center, slice by slice ------------
        half_widths = np.array([
            _largest_square_half_width(_unpack_slice(packed, width), *center)
            for packed in inside_packed
        ], dtype=np.int64)

        # Shrinking every square by `margin` keeps it at least that many pixels away
        # from the sample boundary. Slices that drop below 0 no longer fit any square.
        half_widths -= margin

        if half_widths.max() < 0:
            raise ValueError(
                f"No square around the sample center (x={center[1]}, y={center[0]}) fits "
                f"inside the sample. Check that {exterior_value} is really the exterior "
                f"value, and that the safety margin ({margin} px) isn't too large."
            )

        # --- Step 4: choose the slices and the square size ------------------------
        if make_cube and not is_2d:
            half_width, z_start, z_stop = _fit_cube(half_widths)
        else:
            # For a single 2D slice this simply returns that slice's biggest square.
            half_width, z_start, z_stop = _fit_column(half_widths)
        box_center = center  # the center these numbers were measured around

        # --- Step 5: move the center to where the biggest common square fits -------
        common_inside = np.ones((height, width), dtype=bool)
        for z in range(z_start, z_stop):
            common_inside &= _unpack_slice(inside_packed[z], width)
        center = _deepest_point(common_inside, near=box_center)

        if center == box_center:
            break  # already at the best spot for these slices

    center_y, center_x = box_center
    return CropBox(
        z_start=z_start,
        z_stop=z_stop,
        y_start=center_y - half_width,
        y_stop=center_y + half_width + 1,
        x_start=center_x - half_width,
        x_stop=center_x + half_width + 1,
    )


def crop_image(image: np.ndarray, box: CropBox) -> np.ndarray:
    """
    Cut a CropBox out of a 2D or 3D image.

    Returns a copy, not a view: a numpy slice keeps the WHOLE original array alive in
    memory for as long as the slice exists, which would defeat the point of cropping
    a large volume. The copy is also stored contiguously, which keeps later
    calculations on it fast.
    """
    return image[box.slices(image.ndim)].copy()
