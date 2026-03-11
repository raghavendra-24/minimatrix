"""
matrixa.applications — Real-world application modules.

Available modules
-----------------
- graphics  : 2-D/3-D transformation matrices (rotation, scale, shear, …)
"""

from matrixa.applications.graphics import (
    rotation_2d,
    rotation_3d_x,
    rotation_3d_y,
    rotation_3d_z,
    scale,
    shear_2d,
    reflect_2d,
    homogeneous_translate_2d,
)

__all__ = [
    "rotation_2d",
    "rotation_3d_x",
    "rotation_3d_y",
    "rotation_3d_z",
    "scale",
    "shear_2d",
    "reflect_2d",
    "homogeneous_translate_2d",
]
