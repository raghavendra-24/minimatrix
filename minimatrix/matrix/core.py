"""
minimatrix.matrix.core – Matrix data type definition, construction, validation,
indexing, representation, dtype support, and NumPy-style slicing.

Supported dtypes
----------------
'float'    – native Python float        (default)
'fraction' – fractions.Fraction         (exact rational arithmetic)
'complex'  – native Python complex      (complex number support)
"""

from __future__ import annotations

import math
import warnings
from copy import deepcopy
from fractions import Fraction
from typing import Any, Callable, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "float":    float,
    "fraction": Fraction,
    "complex":  complex,
}


def _cast(value: Any, dtype_type) -> Any:
    """Cast a single value to the target dtype."""
    if isinstance(value, dtype_type):
        return value
    if dtype_type is Fraction:
        # Fraction can be constructed from int, float (via string to avoid
        # floating-point noise), or Fraction.
        if isinstance(value, float):
            return Fraction(value).limit_denominator(10 ** 12)
        return Fraction(value)
    if dtype_type is complex:
        return complex(value)
    return dtype_type(value)


# ---------------------------------------------------------------------------
# Slice helpers
# ---------------------------------------------------------------------------

def _resolve_index(idx, length: int) -> Union[int, range]:
    """Return an int or a range from an int/slice index."""
    if isinstance(idx, int):
        if idx < 0:
            idx += length
        return idx
    if isinstance(idx, slice):
        return range(*idx.indices(length))
    raise TypeError(f"Indices must be int or slice, not {type(idx).__name__}")


# ---------------------------------------------------------------------------
# Matrix class
# ---------------------------------------------------------------------------

class Matrix:
    """
    A 2-D numeric matrix that behaves like a built-in numeric type.

    Parameters
    ----------
    data : list[list[number]]
        Nested list of numbers.  Must be rectangular (non-empty).
    dtype : {'float', 'fraction', 'complex'}
        Internal numeric type.  Defaults to 'float'.

    Examples
    --------
    >>> A = Matrix([[1, 2], [3, 4]])
    >>> A.determinant()
    -2.0

    >>> from fractions import Fraction
    >>> B = Matrix([[1, 2], [3, 4]], dtype='fraction')
    >>> B.determinant()
    Fraction(-2, 1)
    """

    # ------------------------------------------------------------------
    # Construction & validation
    # ------------------------------------------------------------------

    def __init__(self, data: List[List[Any]], dtype: str = "float") -> None:
        if dtype not in _DTYPE_MAP:
            raise ValueError(
                f"Unknown dtype {dtype!r}. Choose from: {list(_DTYPE_MAP)}"
            )
        self.dtype: str = dtype
        self._dtype_type = _DTYPE_MAP[dtype]

        if not data or not data[0]:
            raise ValueError("Matrix cannot be empty.")

        row_len = len(data[0])
        for i, row in enumerate(data):
            if len(row) != row_len:
                raise ValueError(
                    f"Matrix must be rectangular: row 0 has {row_len} elements "
                    f"but row {i} has {len(row)} elements."
                )

        self.data: List[List[Any]] = [
            [_cast(x, self._dtype_type) for x in row] for row in data
        ]
        self.rows: int = len(data)
        self.cols: int = row_len

    # ------------------------------------------------------------------
    # Hash — Matrix is mutable so it must NOT be hashable
    # ------------------------------------------------------------------

    __hash__ = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Indexing & slicing
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        """
        Support a[i], a[i][j], a[i, j] and NumPy-style slicing.

        Examples
        --------
        >>> A[0]          # whole row as Python list
        >>> A[0, 1]       # element
        >>> A[0:2, 1:3]   # sub-matrix
        >>> A[:, 0]       # entire column as column-Matrix
        >>> A[1, :]       # entire row as row-Matrix
        """
        if isinstance(key, tuple):
            row_key, col_key = key
            r_idx = _resolve_index(row_key, self.rows)
            c_idx = _resolve_index(col_key, self.cols)

            r_is_range = isinstance(r_idx, range)
            c_is_range = isinstance(c_idx, range)

            if r_is_range and c_is_range:
                # sub-matrix
                return Matrix(
                    [[self.data[r][c] for c in c_idx] for r in r_idx],
                    dtype=self.dtype,
                )
            if r_is_range and not c_is_range:
                # column slice — return column Matrix
                return Matrix([[self.data[r][c_idx]] for r in r_idx], dtype=self.dtype)
            if not r_is_range and c_is_range:
                # row slice — return row Matrix
                return Matrix([[self.data[r_idx][c] for c in c_idx]], dtype=self.dtype)
            # plain element
            return self.data[r_idx][c_idx]

        # Single key → return raw list row (backward compat: A[i][j] still works)
        return self.data[key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            r, c = key
            self.data[r][c] = _cast(value, self._dtype_type)
        else:
            if not isinstance(value, list):
                raise TypeError("Row assignment requires a list.")
            self.data[key] = [_cast(v, self._dtype_type) for v in value]

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Round-trippable representation: eval(repr(A)) == A."""
        rows_str = ", ".join(
            "[" + ", ".join(repr(v) for v in row) + "]"
            for row in self.data
        )
        if self.dtype == "float":
            return f"Matrix([{rows_str}])"
        return f"Matrix([{rows_str}], dtype={self.dtype!r})"

    def __str__(self) -> str:
        """Pretty-printed, column-aligned representation."""
        # Format each cell
        def fmt(v):
            if isinstance(v, float):
                return f"{v:.6g}"
            return str(v)

        cells = [[fmt(self.data[r][c]) for c in range(self.cols)] for r in range(self.rows)]
        col_widths = [max(len(cells[r][c]) for r in range(self.rows)) for c in range(self.cols)]
        lines = []
        for row_cells in cells:
            padded = "  ".join(f"{cell:>{col_widths[c]}}" for c, cell in enumerate(row_cells))
            lines.append(f"  [ {padded} ]")
        body = "\n".join(lines)
        return f"Matrix(\n{body}\n)"

    # ------------------------------------------------------------------
    # Equality
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.rows != other.rows or self.cols != other.cols:
            return False
        for r in range(self.rows):
            for c in range(self.cols):
                a, b = self.data[r][c], other.data[r][c]
                # For exact types use exact equality; for floats, use tolerance
                if isinstance(a, float) or isinstance(b, float):
                    if abs(float(a) - float(b)) > 1e-9:
                        return False
                elif isinstance(a, complex) or isinstance(b, complex):
                    if abs(a - b) > 1e-9:
                        return False
                else:
                    if a != b:
                        return False
        return True

    # ------------------------------------------------------------------
    # Shape, copy, flatten
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, int]:
        """Return (rows, cols)."""
        return (self.rows, self.cols)

    def copy(self) -> "Matrix":
        """Return a deep copy."""
        return Matrix(deepcopy(self.data), dtype=self.dtype)

    def flatten(self) -> List[Any]:
        """Return all elements in row-major order."""
        return [self.data[r][c] for r in range(self.rows) for c in range(self.cols)]

    # ------------------------------------------------------------------
    # LaTeX export
    # ------------------------------------------------------------------

    def to_latex(self, env: str = "pmatrix") -> str:
        """
        Return a LaTeX string for this matrix.

        Parameters
        ----------
        env : str
            LaTeX matrix environment.  Common choices:
            ``'pmatrix'`` (parentheses), ``'bmatrix'`` (brackets),
            ``'vmatrix'`` (pipes for determinant).

        Examples
        --------
        >>> print(Matrix([[1, 2], [3, 4]]).to_latex())
        \\begin{pmatrix}1 & 2\\\\3 & 4\\end{pmatrix}
        """
        def _fmt(v):
            if isinstance(v, float):
                # Drop trailing .0 for whole numbers
                return str(int(v)) if v == int(v) else f"{v:.6g}"
            if isinstance(v, Fraction):
                if v.denominator == 1:
                    return str(v.numerator)
                return f"\\frac{{{v.numerator}}}{{{v.denominator}}}"
            return str(v)

        rows_str = " \\\\ ".join(
            " & ".join(_fmt(self.data[r][c]) for c in range(self.cols))
            for r in range(self.rows)
        )
        return f"\\begin{{{env}}}{rows_str}\\end{{{env}}}"

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(self, symbol: str = "█", width: int = 8) -> None:
        """
        Print an ASCII heatmap of the matrix values.

        Larger (more positive) values use denser block characters;
        more negative values use lighter characters.

        Parameters
        ----------
        symbol : str
            Not used directly — shading uses Unicode block chars.
        width : int
            Number of characters per cell.
        """
        shades = [" ", "░", "▒", "▓", "█"]
        flat = self.flatten()
        floats = [abs(float(v)) for v in flat]
        max_val = max(floats) if floats else 1.0
        if max_val == 0:
            max_val = 1.0

        print(f"  Matrix heatmap  {self.rows}×{self.cols}")
        print("  +" + ("─" * (width + 2) + "+") * self.cols)
        for r in range(self.rows):
            row_chars = []
            for c in range(self.cols):
                v = float(self.data[r][c])
                ratio = abs(v) / max_val
                shade = shades[min(int(ratio * len(shades)), len(shades) - 1)]
                sign = "-" if v < 0 else " "
                cell = f"{sign}{shade * (width - 1)}"
                row_chars.append(f" {cell[:width]} ")
            print("  |" + "|".join(row_chars) + "|")
        print("  +" + ("─" * (width + 2) + "+") * self.cols)

    # ------------------------------------------------------------------
    # dtype conversion
    # ------------------------------------------------------------------

    def astype(self, dtype: str) -> "Matrix":
        """Return a copy of this matrix in a different dtype."""
        return Matrix(
            [[v for v in row] for row in self.data],
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Class-level constructors
    # ------------------------------------------------------------------

    @classmethod
    def identity(cls, n: int, dtype: str = "float") -> "Matrix":
        """Return the n×n identity matrix."""
        if n < 1:
            raise ValueError("Size must be a positive integer.")
        one = _cast(1, _DTYPE_MAP[dtype])
        zero = _cast(0, _DTYPE_MAP[dtype])
        return cls([[one if r == c else zero for c in range(n)] for r in range(n)], dtype=dtype)

    @classmethod
    def zeros(cls, rows: int, cols: int, dtype: str = "float") -> "Matrix":
        """Return a rows×cols zero matrix."""
        zero = _cast(0, _DTYPE_MAP[dtype])
        return cls([[zero] * cols for _ in range(rows)], dtype=dtype)

    @classmethod
    def ones(cls, rows: int, cols: int, dtype: str = "float") -> "Matrix":
        """Return a rows×cols matrix filled with ones."""
        one = _cast(1, _DTYPE_MAP[dtype])
        return cls([[one] * cols for _ in range(rows)], dtype=dtype)

    # ------------------------------------------------------------------
    # Validation helpers (used by arithmetic / linalg modules)
    # ------------------------------------------------------------------

    def _require_same_shape(self, other: "Matrix", op: str = "operation") -> None:
        if self.shape != other.shape:
            raise ValueError(
                f"Matrix dimensions mismatch for {op}: "
                f"{self.shape} vs {other.shape}."
            )

    def _require_square(self, op: str = "operation") -> None:
        if self.rows != self.cols:
            raise ValueError(
                f"Matrix must be square for {op}: got shape {self.shape}."
            )

    def _require_multipliable(self, other: "Matrix") -> None:
        if self.cols != other.rows:
            raise ValueError(
                f"Matrix dimensions mismatch for multiplication: "
                f"({self.rows}×{self.cols}) @ ({other.rows}×{other.cols})."
            )
