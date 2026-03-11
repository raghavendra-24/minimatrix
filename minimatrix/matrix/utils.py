"""
minimatrix.matrix.utils – Standalone utility functions operating on Matrix objects.
"""

from __future__ import annotations
from typing import Callable, List


def apply(mat, func: Callable[[float], float]):
    """
    Apply *func* element-wise and return a new Matrix.

    Example
    -------
    >>> import math
    >>> B = apply(A, math.sqrt)
    """
    from minimatrix.matrix.core import Matrix
    return Matrix(
        [[func(mat.data[r][c]) for c in range(mat.cols)] for r in range(mat.rows)],
        dtype=mat.dtype,
    )


def element_wise(mat_a, mat_b, func: Callable[[float, float], float]):
    """
    Combine two same-shaped matrices element-wise using *func*.

    Example
    -------
    >>> C = element_wise(A, B, lambda x, y: x ** y)
    """
    from minimatrix.matrix.core import Matrix
    mat_a._require_same_shape(mat_b, "element_wise")
    return Matrix(
        [[func(mat_a.data[r][c], mat_b.data[r][c]) for c in range(mat_a.cols)]
         for r in range(mat_a.rows)],
        dtype=mat_a.dtype,
    )


def from_flat(flat: List[float], rows: int, cols: int, dtype: str = "float"):
    """
    Reshape a flat list into a *rows* × *cols* Matrix.

    Example
    -------
    >>> M = from_flat([1, 2, 3, 4, 5, 6], 2, 3)
    """
    from minimatrix.matrix.core import Matrix
    if len(flat) != rows * cols:
        raise ValueError(
            f"Cannot reshape {len(flat)} elements into ({rows}×{cols})."
        )
    return Matrix(
        [flat[r * cols:(r + 1) * cols] for r in range(rows)],
        dtype=dtype,
    )


def diag(values: List[float], dtype: str = "float"):
    """
    Create a diagonal matrix from *values*.

    Example
    -------
    >>> D = diag([1, 2, 3])
    """
    from minimatrix.matrix.core import Matrix
    n = len(values)
    return Matrix(
        [[values[i] if i == j else 0.0 for j in range(n)] for i in range(n)],
        dtype=dtype,
    )


def is_symmetric(mat, tol: float = 1e-9) -> bool:
    """Return True if *mat* is symmetric (A == Aᵀ) within *tol*."""
    if mat.rows != mat.cols:
        return False
    for r in range(mat.rows):
        for c in range(r + 1, mat.cols):
            a = complex(mat.data[r][c]) if not isinstance(mat.data[r][c], (int, float)) else float(mat.data[r][c])
            b = complex(mat.data[c][r]) if not isinstance(mat.data[c][r], (int, float)) else float(mat.data[c][r])
            if abs(a - b) > tol:
                return False
    return True


def is_identity(mat, tol: float = 1e-9) -> bool:
    """Return True if *mat* is an identity matrix within *tol*."""
    if mat.rows != mat.cols:
        return False
    for r in range(mat.rows):
        for c in range(mat.cols):
            expected = 1.0 if r == c else 0.0
            if abs(float(mat.data[r][c]) - expected) > tol:
                return False
    return True


def frobenius_norm(mat) -> float:
    """Return the Frobenius norm: sqrt( sum of squares of all elements )."""
    import math
    return math.sqrt(sum(float(mat.data[r][c]) ** 2
                         for r in range(mat.rows)
                         for c in range(mat.cols)))
