"""
matrixa.applications.graphics – 2-D and 3-D transformation matrices.

All functions return a plain Matrix object so they compose naturally
via the @ (matmul) operator.

Reference:
  https://en.wikipedia.org/wiki/Transformation_matrix

Examples
--------
>>> from matrixa.applications import rotation_2d, scale
>>> R = rotation_2d(45)          # 45-degree rotation
>>> S = scale(2, 0.5)            # stretch x, squish y
>>> T = R @ S                    # compose: scale then rotate
"""

from __future__ import annotations
import math
from matrixa.matrix.core import Matrix


def rotation_2d(degrees: float) -> Matrix:
    """
    Return the 2×2 rotation matrix for *degrees* counter-clockwise.

    Example
    -------
    >>> R = rotation_2d(90)
    >>> R @ Matrix([[1], [0]])   # rotates unit-x to unit-y
    """
    θ = math.radians(degrees)
    c, s = math.cos(θ), math.sin(θ)
    return Matrix([[c, -s], [s, c]])


def rotation_3d_x(degrees: float) -> Matrix:
    """Return the 3×3 rotation matrix around the X axis."""
    θ = math.radians(degrees)
    c, s = math.cos(θ), math.sin(θ)
    return Matrix([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c],
    ])


def rotation_3d_y(degrees: float) -> Matrix:
    """Return the 3×3 rotation matrix around the Y axis."""
    θ = math.radians(degrees)
    c, s = math.cos(θ), math.sin(θ)
    return Matrix([
        [ c,  0,  s],
        [ 0,  1,  0],
        [-s,  0,  c],
    ])


def rotation_3d_z(degrees: float) -> Matrix:
    """Return the 3×3 rotation matrix around the Z axis."""
    θ = math.radians(degrees)
    c, s = math.cos(θ), math.sin(θ)
    return Matrix([
        [c, -s,  0],
        [s,  c,  0],
        [0,  0,  1],
    ])


def scale(sx: float, sy: float, sz: float = None) -> Matrix:
    """
    Return a scaling matrix.

    Parameters
    ----------
    sx, sy : float
        Scale factors for the x and y axes (2-D transform).
    sz : float, optional
        If provided, returns a 3×3 matrix for 3-D scaling.

    Example
    -------
    >>> scale(2, 3)       # 2×2 matrix: double x, triple y
    >>> scale(2, 3, 4)    # 3×3 matrix: 3-D scaling
    """
    if sz is None:
        return Matrix([[sx, 0], [0, sy]])
    return Matrix([[sx, 0, 0], [0, sy, 0], [0, 0, sz]])


def shear_2d(shx: float = 0.0, shy: float = 0.0) -> Matrix:
    """
    Return a 2-D shear matrix.

    Parameters
    ----------
    shx : float
        Horizontal shear (shear parallel to x axis).
    shy : float
        Vertical shear (shear parallel to y axis).

    Example
    -------
    >>> shear_2d(shx=1.5)   # slant rightward
    """
    return Matrix([[1, shx], [shy, 1]])


def reflect_2d(axis: str = "x") -> Matrix:
    """
    Return a 2-D reflection matrix.

    Parameters
    ----------
    axis : {'x', 'y', 'origin', 'y=x'}
        Axis of reflection.

    Example
    -------
    >>> reflect_2d('y')   # flip left-right
    """
    axis = axis.lower().strip()
    if axis == "x":
        return Matrix([[1, 0], [0, -1]])
    if axis == "y":
        return Matrix([[-1, 0], [0, 1]])
    if axis == "origin":
        return Matrix([[-1, 0], [0, -1]])
    if axis in ("y=x", "yx"):
        return Matrix([[0, 1], [1, 0]])
    raise ValueError(f"Unknown axis {axis!r}. Choose from: 'x', 'y', 'origin', 'y=x'")


def homogeneous_translate_2d(tx: float, ty: float) -> Matrix:
    """
    Return the 3×3 homogeneous translation matrix for 2-D graphics.

    Apply to homogeneous column vectors [x, y, 1]ᵀ.

    Example
    -------
    >>> T = homogeneous_translate_2d(3, -1)
    >>> p = Matrix([[2], [4], [1]])
    >>> T @ p    # translates point (2,4) to (5,3)
    """
    return Matrix([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0,  1],
    ])
