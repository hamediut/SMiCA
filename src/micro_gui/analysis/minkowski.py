"""
Minkowski functional (MF) calculations for 2D and 3D binary microstructure images.
"""
import numpy as np
from quantimpy import minkowski as mk

# QuantImPy's raw functionals() output follows Mecke's normalized Minkowski-
# functional convention, not the plain integral-geometry definitions (V, S, M,
# chi) used in the report equations. These constants convert QuantImPy's raw
# output back to the true physical/topological quantities. Verified empirically
# against analytic sphere/disk/cube values - do not "simplify" these away.
_EULER_NORM_2D = np.pi
_LENGTH_NORM_2D = 2 * np.pi

_SURFACE_NORM_3D = 8.0
_CURVATURE_NORM_3D = 2 * np.pi ** 2
_EULER_NORM_3D = 4 * np.pi / 3

def minkowski_2d(image: np.ndarray, res: float = 1.0) -> dict:
    """Compute true Minkowski functionals for a single 2D binary image, plus
    their "specific" (density) versions, normalized by the physical area of
    the whole image domain (not just the foreground phase).

    res: pixel size (physical units per pixel).
    """
    area, length, euler = mk.functionals(image.astype(bool), (res, res))

    perimeter = length * _LENGTH_NORM_2D
    euler_true = euler * _EULER_NORM_2D
    domain_area = image.size * res ** 2  # physical area of the whole image

    return {
        'area': area,
        'perimeter': perimeter,
        'euler': euler_true,
        'area_fraction': area / domain_area,              # dimensionless
        'specific_perimeter': perimeter / domain_area,     # 1/length
        'specific_euler': euler_true / domain_area,        # 1/length^2
    }


def minkowski_3d(image: np.ndarray, res: float = 1.0) -> dict:
    """Compute true Minkowski functionals for a single 3D binary image, plus
    their "specific" (density) versions, normalized by the physical volume of
    the whole image domain (not just the foreground phase).

    res: voxel size (physical units per voxel, assumed isotropic).
    """
    volume, surface, curvature, euler = mk.functionals(image.astype(bool), (res, res, res))

    surface_area = surface * _SURFACE_NORM_3D
    mean_curv = curvature * _CURVATURE_NORM_3D
    euler_true = euler * _EULER_NORM_3D
    domain_volume = image.size * res ** 3  # physical volume of the whole image

    return {
        'volume': volume,
        'surface_area': surface_area,
        'mean_curv': mean_curv,
        'euler': euler_true,
        'porosity': volume / domain_volume,                       # dimensionless
        'specific_surface_area': surface_area / domain_volume,    # 1/length
        'specific_mean_curv': mean_curv / domain_volume,          # 1/length^2
        'specific_euler': euler_true / domain_volume,             # 1/length^3
    }

