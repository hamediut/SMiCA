"""
Connected-component labeling and per-component size measurement for binary
microstructure images.
"""
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops_table, regionprops, marching_cubes, mesh_surface_area

# Avizo/Dragonfly describe neighbor rules by how many touching neighbors count
# as "connected" - 6/18/26 in 3D (face, face+edge, face+edge+corner), 4/8 in
# 2D (face, face+corner). skimage.measure.label instead takes a "connectivity"
# argument that is the RANK of the neighborhood (1, 2, or 3 in 3D; 1 or 2 in
# 2D), not a neighbor count - these maps translate one vocabulary into the
# other so the rest of the app (and its UI) can stay in the terms users expect.
_CONNECTIVITY_MAP_3D = {6: 1, 18: 2, 26: 3}
_CONNECTIVITY_MAP_2D = {4: 1, 8: 2}


def _label_and_measure(
    image: np.ndarray, connectivity_map: dict, connectivity: int, res: float,
    size_exponent: int, count_col: str, measure_col: str,
) -> dict:
    """Label connected components of the foreground phase (value 1) in a
    binary image, and measure the size of each one.

    Shared core used by connected_components_2d/connected_components_3d -
    not part of the public API.

    Args:
        image: binary array, 2D or 3D. Any nonzero element counts as
            foreground.
        connectivity_map: maps the connectivity numbers a caller can pass
            (e.g. {6: 1, 18: 2, 26: 3}) to the "connectivity" rank
            skimage.measure.label expects.
        connectivity: neighbor rule to use - must be a key in
            connectivity_map.
        res: pixel/voxel size (physical units per pixel/voxel, assumed
            isotropic).
        size_exponent: power to raise `res` to when converting a raw
            element count into a physical size (2 for area, 3 for volume).
        count_col: name to give the raw element-count column in the
            returned table.
        measure_col: name to give the physical-size column in the returned
            table.

    Returns:
        dict with:
            'labels': int array, same shape as `image` - 0 is background,
                1..N are component ids.
            'num_components': N.
            'table': pandas DataFrame, one row per component, sorted by
                `measure_col` descending - columns 'label', `count_col`,
                `measure_col`.

    Raises:
        ValueError: if `connectivity` is not a key in `connectivity_map`.
    """
    if connectivity not in connectivity_map:
        raise ValueError(f"connectivity must be one of {sorted(connectivity_map)}, got {connectivity}")

    labels, num_components = label(
        image.astype(bool),
        connectivity=connectivity_map[connectivity],
        return_num=True,
    )

    # regionprops_table skips label 0 (background) automatically.
    props = regionprops_table(labels, properties=('label', 'area'))
    table = pd.DataFrame(props).rename(columns={'area': count_col})
    table[measure_col] = table[count_col] * res ** size_exponent
    table = table.sort_values(measure_col, ascending=False).reset_index(drop=True)

    return {
        'labels': labels,
        'num_components': num_components,
        'table': table,
    }


def connected_components_2d(image: np.ndarray, connectivity: int = 8, res: float = 1.0) -> dict:
    """Label connected components of the foreground phase (value 1) in a 2D
    binary image, and measure the area of each one.
    """
    return _label_and_measure(
        image, _CONNECTIVITY_MAP_2D, connectivity, res,
        size_exponent=2, count_col='pixel_count', measure_col='area',
    )

def connected_components_3d(image: np.ndarray, connectivity: int = 26, res: float = 1.0) -> dict:
    """Label connected components of the foreground phase (value 1) in a 3D
    binary image, and measure the volume of each one.

    Args:
        image: 3D array (Z, Y, X). Any nonzero voxel counts as foreground.
        connectivity: neighbor rule - 6 (face neighbors only), 18 (+ edge
            neighbors), or 26 (+ corner neighbors).
        res: voxel size (physical units per voxel, assumed isotropic).

    Returns:
        dict with:
            'labels': int array, same shape as `image` - 0 is background,
                1..N are component ids.
            'num_components': N.
            'table': pandas DataFrame, one row per component, sorted by
                volume descending - columns 'label', 'voxel_count', 'volume'.
    """
    return _label_and_measure(
        image, _CONNECTIVITY_MAP_3D, connectivity, res,
        size_exponent=3, count_col='voxel_count', measure_col='volume',
    )

def _safe_prop(region, name):
    """Some regionprops properties (axis_major_length/axis_minor_length in
    particular) can raise ValueError('math domain error') for degenerate
    components - e.g. one only a single voxel/pixel thick along an axis,
    common among the tiny segmentation speckle a min-size filter is meant to
    catch. Returns NaN instead of letting one bad component crash the whole
    measurement pass.
    """
    try:
        return getattr(region, name)
    except ValueError:
        return np.nan


def compute_shape_measurements(labels: np.ndarray, is_3d: bool, res: float = 1.0)-> pd.DataFrame:
    """Compute per-component shape measurements available directly from
    skimage's regionprops - no custom surface reconstruction or eigenvector
    math involved. Surface area, sphericity, elongation/flatness, and
    orientation need that extra math and are a separate follow-up function.

    Args:
        labels: label array from connected_components_2d/_3d's 'labels' key.
        is_3d: whether `labels` is a 3D volume or a 2D image.
        res: pixel/voxel size (physical units), same convention as
            connected_components_2d/_3d's `res`.

    Returns:
        pandas DataFrame, one row per component (background excluded), meant
        to be merged with connected_components_2d/_3d's own table on 'label'.
    """

    rows = []
    for region in regionprops(labels):
        row = {'label': region.label}
        row['equivalent_diameter'] = _safe_prop(region, 'equivalent_diameter') * res

        # axis_major_length/axis_minor_length are native in both 2D and 3D.
        major_length = _safe_prop(region, 'axis_major_length') * res
        minor_length = _safe_prop(region, 'axis_minor_length') * res
        row['aspect_ratio'] = major_length / minor_length if minor_length > 0 else np.nan

        if not is_3d:
            perimeter = _safe_prop(region, 'perimeter') * res
            area = region.area * res ** 2
            row['perimeter'] = perimeter

            # "Specific" here means per THIS COMPONENT's own area - a shape
            # property, unlike minkowski_2d's specific_perimeter which
            # normalizes by the whole image domain.
            row['specific_perimeter'] = perimeter / area if area > 0 else np.nan
            row['circularity'] = 4 * np.pi * area / perimeter ** 2 if perimeter > 0 else np.nan

        rows.append(row)
    return pd.DataFrame(rows)


def _safe_surface_area(region):
    """Marching-cubes surface area of a component's own cropped mask, padded
    by 1 voxel so the surface closes at the component's boundary. Tested
    against all 5902 components of the real XCT test volume with zero
    failures, but kept defensive (NaN instead of a crash) since marching
    cubes can in principle fail on a degenerate isosurface.

    the standard algorithm for turning a 3D voxel grid into a triangulated surface (a mesh)
    — the same technique for 3D visualization to render a segmented structure
    as a smooth-looking surface instead of blocky cubes.


    """
    try:
        padded = np.pad(region.image, 1)
        verts, faces, _, _ = marching_cubes(padded, level=0.5)
        # verts — the (x, y, z) coordinates of every triangle corner in the mesh
        # for each triangle, which 3 vertices (by index into verts) form it
        return mesh_surface_area(verts, faces)
    except (RuntimeError, ValueError):
        return np.nan

def _safe_axes_and_orientation(region):
    """Major/intermediate/minor axis lengths and the major axis's direction
    (theta/phi), from the inertia tensor's eigenvalues/eigenvectors.

    Length formula: for a uniform-density ellipsoid, eigenvalue_i = (sum of
    the OTHER two semi-axes squared) / 5, which inverts to
    axis_length_i = sqrt(10*sum(eigvals) - 20*eigval_i) - verified against
    skimage's own axis_major_length/axis_minor_length on ellipsoids of known
    semi-axes.

    Orientation: the eigenvector of the SMALLEST eigenvalue points along the
    major (longest) axis - verified against ellipsoids individually
    elongated along each array axis. theta = angle from array axis 0, phi =
    angle from array axis 2 in the axis2/axis1 plane, matching the
    Avizo/Dragonfly convention for reporting an axis's direction. An axis has
    no inherent direction, so the sign is fixed by convention (axis-0
    component >= 0).
    """

    #For a component, the inertia tensor is a 3×3 matrix that describes how its voxels are spread out around its own centroid
    # literally the same math as the physics "moment of inertia" (resistance to rotation about an axis),
    #  just applied to the shape's voxel positions instead of physical mass.
    #  It captures not just how much spread there is, but in which directions.
    try:
        eigvals, eigvecs = np.linalg.eigh(region.inertia_tensor)  # ascending
        #  eigenvalues in ascending order, so eigvals[0] (smallest) pairs with the major axis,
        #  eigvals[2] (largest) with the minor axis.
        #Eigendecomposing this matrix finds the component's 3 principal axes — the natural "long / medium / short" directions the shape is oriented along,
        #  regardless of how it happens to sit in the array's X/Y/Z coordinate frame.
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    total = eigvals.sum()
    lengths = np.sqrt(np.clip(10 * total - 20 * eigvals, 0, None))  # ascending eigval -> descending length
    major_length, intermediate_length, minor_length = lengths[0], lengths[1], lengths[2]

    major_vec = eigvecs[:, 0]  # smallest eigenvalue -> major axis direction
    #  small moment → long axis, large moment → short axis
    # eigvecs[:, 0] isn't just a number — it's the actual 3D direction the major axis points in,
    # paired with that smallest eigenvalue. We convert that direction into two angles (theta, phi)
    # the same way you'd describe any line's orientation in 3D — like latitude/longitude, or dip/azimuth in geology:
    #  theta = angle away from array axis 0, phi = angle around that axis, measured in the axis-2/axis-1 plane.
    v0, v1, v2 = major_vec
    if v0 < 0:
        v0, v1, v2 = -v0, -v1, -v2
    theta_deg = np.degrees(np.arccos(np.clip(v0, -1, 1)))
    phi_deg = np.degrees(np.arctan2(v1, v2))

    return major_length, intermediate_length, minor_length, theta_deg, phi_deg


def compute_advanced_shape_measurements(labels: np.ndarray, res: float, requested: list) -> pd.DataFrame:
    """Compute the remaining per-component 3D shape/orientation measurements:
    surface_area, specific_surface_area, sphericity (need a per-component
    marching-cubes mesh), and elongation, flatness, theta_deg, phi_deg (need
    the inertia tensor's eigenvectors, not just its eigenvalues).

    3D only - none of these have a 2D equivalent. Separate from
    compute_shape_measurements because marching_cubes/eigh are meaningfully
    slower (measured ~45s for 5902 components) than a regionprops property
    lookup - `requested` lets the caller skip whichever group isn't needed.

    Args:
        labels: 3D label array from connected_components_3d's 'labels' key.
        res: voxel size (physical units per voxel, assumed isotropic).
        requested: which columns to compute - any of 'surface_area',
            'specific_surface_area', 'sphericity', 'elongation', 'flatness',
            'theta_deg', 'phi_deg'.

    Returns:
        pandas DataFrame, one row per component (background excluded), with
        'label' plus whichever of the requested columns were asked for -
        meant to be merged onto the table connected_components_3d/
        compute_shape_measurements already built, on 'label'.
    """
    requested = set(requested)
    need_surface = bool(requested & {'surface_area', 'specific_surface_area', 'sphericity'})
    need_axes = bool(requested & {'elongation', 'flatness', 'theta_deg', 'phi_deg'})

    rows = []
    for region in regionprops(labels):
        row = {'label': region.label}

        if need_surface:
            surface_area = _safe_surface_area(region) * res ** 2
            volume = region.area * res ** 3
            if 'surface_area' in requested:
                row['surface_area'] = surface_area
            if 'specific_surface_area' in requested:
                row['specific_surface_area'] = (
                    surface_area / volume if volume > 0 and not np.isnan(surface_area) else np.nan
                )
            if 'sphericity' in requested:
                row['sphericity'] = (
                    (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / surface_area
                    if surface_area > 0 and not np.isnan(surface_area) else np.nan
                )

        if need_axes:
            major_length, intermediate_length, minor_length, theta_deg, phi_deg = _safe_axes_and_orientation(region)
            if 'elongation' in requested:
                row['elongation'] = intermediate_length / major_length if major_length > 0 else np.nan
            if 'flatness' in requested:
                row['flatness'] = minor_length / intermediate_length if intermediate_length > 0 else np.nan
            if 'theta_deg' in requested:
                row['theta_deg'] = theta_deg
            if 'phi_deg' in requested:
                row['phi_deg'] = phi_deg

        rows.append(row)

    return pd.DataFrame(rows)