"""
minimatrix.matrix.linalg – Core linear-algebra methods attached to Matrix.

Covers:
  transpose, trace, determinant (with verbose mode), minor, cofactor,
  adjugate, inverse (Gauss-Jordan, with condition-number warning),
  rank, LU decomposition (returns P, L, U correctly),
  solve (Ax = b, with verbose mode), matrix norms.
"""

from __future__ import annotations

import math
import warnings
from copy import deepcopy
from fractions import Fraction
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _float(v) -> float:
    """Convert any supported dtype value to float."""
    if isinstance(v, complex):
        return abs(v)
    return float(v)


def _augment_float(mat, rhs) -> List[List[float]]:
    """Build [mat | rhs] as a list-of-floats for Gaussian elimination."""
    aug = [
        [_float(mat.data[r][c]) for c in range(mat.cols)]
        + [_float(rhs.data[r][c]) for c in range(rhs.cols)]
        for r in range(mat.rows)
    ]
    return aug


def _lu_decompose(mat) -> Tuple[List, List, List, int]:
    """
    Doolittle LU with partial pivoting.

    Returns (L, U, P, sign) where:
      - P is the row-permutation vector (P[i] = original row now at position i)
      - L is lower-triangular with unit diagonal
      - U is upper-triangular
      - sign is the parity of the permutation (±1), used for determinant
    All returned as plain list-of-lists (not Matrix objects).
    """
    n = mat.rows
    L = [[0.0] * n for _ in range(n)]
    U = [[ _float(v) for v in row] for row in mat.data]
    P = list(range(n))
    sign = 1

    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(U[r][col]))
        if col != max_row:
            U[col], U[max_row] = U[max_row], U[col]
            L[col], L[max_row] = L[max_row], L[col]
            P[col], P[max_row] = P[max_row], P[col]
            sign *= -1

        L[col][col] = 1.0
        if abs(U[col][col]) < 1e-12:
            continue  # singular — determinant will be ~0

        for row in range(col + 1, n):
            factor = U[row][col] / U[col][col]
            L[row][col] = factor
            for j in range(col, n):
                U[row][j] -= factor * U[col][j]

    return L, U, P, sign


def _build_permutation_matrix(P: List[int]) -> List[List[float]]:
    """Convert a permutation vector P into an n×n permutation matrix."""
    n = len(P)
    PM = [[0.0] * n for _ in range(n)]
    for i, j in enumerate(P):
        PM[i][j] = 1.0
    return PM


# ---------------------------------------------------------------------------
# Methods attached to Matrix
# ---------------------------------------------------------------------------

def transpose(self):
    """Return the transpose of this matrix."""
    from minimatrix.matrix.core import Matrix
    return Matrix(
        [[self.data[r][c] for r in range(self.rows)] for c in range(self.cols)],
        dtype=self.dtype,
    )


def trace(self) -> float:
    """Return the sum of the main diagonal (square matrices only)."""
    self._require_square("trace")
    return sum(self.data[i][i] for i in range(self.rows))


def minor(self, row: int, col: int):
    """Return the (row, col) minor — the matrix with that row & col removed."""
    from minimatrix.matrix.core import Matrix
    sub = [
        [self.data[r][c] for c in range(self.cols) if c != col]
        for r in range(self.rows) if r != row
    ]
    return Matrix(sub, dtype=self.dtype)


def cofactor(self, row: int, col: int) -> float:
    """Return the (row, col) cofactor."""
    sign = (-1) ** (row + col)
    return sign * self.minor(row, col).determinant()


def adjugate(self):
    """Return the adjugate (classical adjoint) matrix = transpose of cofactor matrix."""
    from minimatrix.matrix.core import Matrix
    self._require_square("adjugate")
    n = self.rows
    cof = [[self.cofactor(r, c) for c in range(n)] for r in range(n)]
    # adjugate = transpose of cofactor matrix
    return Matrix([[cof[c][r] for c in range(n)] for r in range(n)])


def determinant(self, verbose: bool = False) -> float:
    """
    Return the determinant.

    Uses the closed-form formula for 1×1 and 2×2, then LU decomposition
    with partial pivoting for n≥3 (O(n³), numerically stable).

    Parameters
    ----------
    verbose : bool
        If True, print step-by-step explanation of the computation.

    Examples
    --------
    >>> Matrix([[6, 1, 1], [4, -2, 5], [2, 8, 7]]).determinant(verbose=True)
    """
    self._require_square("determinant")
    n = self.rows

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  determinant()  —  {n}×{n} matrix")
        print(f"{'─'*55}")
        for r in range(n):
            print(f"  Row {r}: {self.data[r]}")

    if n == 1:
        result = _float(self.data[0][0])
        if verbose:
            print(f"\n  1×1 matrix → det = {result}")
        return result

    if n == 2:
        a, b = _float(self.data[0][0]), _float(self.data[0][1])
        c, d = _float(self.data[1][0]), _float(self.data[1][1])
        result = a * d - b * c
        if verbose:
            print(f"\n  2×2 shortcut: det = ({a})×({d}) − ({b})×({c})")
            print(f"              = {a*d:.6g} − {b*c:.6g} = {result:.6g}")
        return result

    # LU with partial pivoting for n≥3
    if verbose:
        print(f"\n  Using LU decomposition with partial pivoting (Doolittle):")

    L_data, U_data, P_vec, sign = _lu_decompose(self)

    if verbose:
        print(f"\n  Permutation vector P = {P_vec}")
        print(f"  Row-swap parity (sign) = {sign:+d}")
        print(f"\n  Upper-triangular U diagonal:")
        diag_product = 1.0
        for i in range(n):
            print(f"    U[{i},{i}] = {U_data[i][i]:.6g}")
            diag_product *= U_data[i][i]
        print(f"\n  det = sign × ∏ U[i,i]")
        print(f"      = {sign} × {diag_product:.6g}")

    det = sign * 1.0
    for i in range(n):
        det *= U_data[i][i]

    if verbose:
        print(f"      = {det:.6g}")
        print(f"{'─'*55}\n")

    return det


def inverse(self, verbose: bool = False):
    """
    Return the inverse using Gauss-Jordan elimination.

    Issues a warning (instead of crashing) when the matrix is
    near-singular (condition number > 1e10).  Raises ValueError if
    exactly singular (pivot < 1e-12).

    Parameters
    ----------
    verbose : bool
        If True, print augmented-matrix elimination steps.
    """
    from minimatrix.matrix.core import Matrix
    self._require_square("inverse")
    n = self.rows
    # Build float augmented matrix [self | I]
    aug = [
        [_float(self.data[r][c]) for c in range(n)] + ([1.0 if r == c else 0.0 for c in range(n)])
        for r in range(n)
    ]

    if verbose:
        print(f"\n{'─'*55}")
        print(f"  inverse()  —  Gauss-Jordan elimination on [A | I]")
        print(f"{'─'*55}")

    # Track row norms for condition number estimate
    original_norms = [math.sqrt(sum(aug[r][c]**2 for c in range(n))) for r in range(n)]

    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[max_row][col]) < 1e-12:
            raise ValueError(
                f"Matrix is singular (pivot ≈ 0 at column {col}). Inverse does not exist."
            )
        if col != max_row:
            aug[col], aug[max_row] = aug[max_row], aug[col]
            if verbose:
                print(f"  Swap R{col} ↔ R{max_row}")

        pivot = aug[col][col]
        aug[col] = [v / pivot for v in aug[col]]
        if verbose:
            print(f"  Scale R{col} by 1/{pivot:.4g}  →  pivot = 1")

        for r in range(n):
            if r != col:
                factor = aug[r][col]
                if factor != 0:
                    aug[r] = [aug[r][j] - factor * aug[col][j] for j in range(2 * n)]
                    if verbose:
                        print(f"  R{r} ← R{r} − ({factor:.4g})·R{col}")

    result_data = [aug[r][n:] for r in range(n)]
    inv_mat = Matrix(result_data)

    # Condition number estimate: max(|row norms of inv|) * max(|row norms of A|)
    inv_norms = [math.sqrt(sum(inv_mat.data[r][c]**2 for c in range(n))) for r in range(n)]
    cond_est = max(original_norms) * max(inv_norms)
    if cond_est > 1e10:
        warnings.warn(
            f"Matrix may be ill-conditioned (condition estimate ≈ {cond_est:.2e}). "
            "Inverse may have significant floating-point errors.",
            RuntimeWarning,
            stacklevel=2,
        )

    if verbose:
        print(f"\n  Condition number estimate: {cond_est:.2e}")
        print(f"{'─'*55}\n")

    return inv_mat


def rank(self) -> int:
    """Return the rank using Gaussian elimination with partial pivoting."""
    mat = [[_float(v) for v in row] for row in self.data]
    rows, cols = self.rows, self.cols
    pivot_row = 0

    for col in range(cols):
        found = None
        for r in range(pivot_row, rows):
            if abs(mat[r][col]) > 1e-12:
                found = r
                break
        if found is None:
            continue
        mat[pivot_row], mat[found] = mat[found], mat[pivot_row]
        pivot = mat[pivot_row][col]
        mat[pivot_row] = [v / pivot for v in mat[pivot_row]]
        for r in range(rows):
            if r != pivot_row:
                factor = mat[r][col]
                mat[r] = [mat[r][j] - factor * mat[pivot_row][j] for j in range(cols)]
        pivot_row += 1

    return pivot_row


# ---------------------------------------------------------------------------
# LU Decomposition (Phase 1 fix: returns P, L, U)
# ---------------------------------------------------------------------------

def lu_decomposition(self) -> Tuple["Matrix", "Matrix", "Matrix"]:
    """
    Return ``(P, L, U)`` such that ``P @ A == L @ U``.

    - **P** is a permutation matrix (row reordering from partial pivoting).
    - **L** is lower-triangular with unit diagonal.
    - **U** is upper-triangular.

    This is the mathematically correct LU signature.

    Examples
    --------
    >>> P, L, U = A.lu_decomposition()
    >>> P @ A == L @ U   # always True (up to float tolerance)
    True
    """
    from minimatrix.matrix.core import Matrix
    self._require_square("LU decomposition")
    L_data, U_data, P_vec, _ = _lu_decompose(self)
    P_data = _build_permutation_matrix(P_vec)
    return Matrix(P_data), Matrix(L_data), Matrix(U_data)


# ---------------------------------------------------------------------------
# Linear System Solver  (Phase 2b)
# ---------------------------------------------------------------------------

def solve(self, b, verbose: bool = False):
    """
    Solve the linear system  Ax = b  using LU decomposition.

    Parameters
    ----------
    b : Matrix
        Right-hand side.  Must be a column vector (n×1) or a Matrix with
        the same number of rows as ``self``.
    verbose : bool
        If True, print step-by-step solution walkthrough.

    Returns
    -------
    Matrix
        Solution column vector x.

    Raises
    ------
    ValueError
        If the system has no unique solution (singular matrix).

    Examples
    --------
    >>> A = Matrix([[2, 1], [5, 3]])
    >>> b = Matrix([[4], [7]])
    >>> x = A.solve(b)
    >>> A @ x   # should equal b
    """
    from minimatrix.matrix.core import Matrix
    self._require_square("solve")
    if self.rows != b.rows:
        raise ValueError(
            f"A has {self.rows} rows but b has {b.rows} rows."
        )

    n = self.rows
    # Build float augmented matrix [A | b]
    aug = _augment_float(self, b)
    col_count = self.cols + b.cols

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  solve(Ax = b)  —  Forward elimination with partial pivoting")
        print(f"{'─'*60}")
        print(f"  Augmented matrix [A | b]:")
        for r in range(n):
            row_str = "  ".join(f"{v:>9.4g}" for v in aug[r])
            print(f"    [ {row_str} ]")

    # Forward elimination (partial pivoting)
    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[max_row][col]) < 1e-12:
            raise ValueError(
                f"Matrix is singular at column {col}. System has no unique solution."
            )
        if col != max_row:
            aug[col], aug[max_row] = aug[max_row], aug[col]
            if verbose:
                print(f"\n  Step: Swap R{col} ↔ R{max_row}  (partial pivot)")

        pivot = aug[col][col]
        aug[col] = [v / pivot for v in aug[col]]
        if verbose:
            print(f"  Step: Scale R{col} ÷ {pivot:.4g}  (make pivot = 1)")

        for r in range(n):
            if r != col:
                factor = aug[r][col]
                if abs(factor) > 1e-15:
                    aug[r] = [aug[r][j] - factor * aug[col][j] for j in range(col_count)]
                    if verbose:
                        print(f"  Step: R{r} ← R{r} − ({factor:.4g})·R{col}")

    # Extract solution columns
    x_data = [[aug[r][self.cols + c] for c in range(b.cols)] for r in range(n)]
    x = Matrix(x_data)

    if verbose:
        print(f"\n  Solution x:")
        for r in range(x.rows):
            vals = "  ".join(f"{v:>9.4g}" for v in x.data[r])
            print(f"    x[{r}] = {vals}")
        print(f"\n  Verification  A @ x:")
        Ax = self * x
        for r in range(Ax.rows):
            vals = "  ".join(f"{_float(v):>9.4g}" for v in Ax.data[r])
            bv   = "  ".join(f"{_float(v):>9.4g}" for v in b.data[r])
            print(f"    Ax[{r}] = {vals}   b[{r}] = {bv}")
        print(f"{'─'*60}\n")

    return x


# ---------------------------------------------------------------------------
# Matrix Norms  (Phase 2e)
# ---------------------------------------------------------------------------

def norm(self, kind="frobenius") -> float:
    """
    Compute a matrix norm.

    Parameters
    ----------
    kind : {'frobenius', 'fro', 1, 'inf', 2}
        - ``'frobenius'`` or ``'fro'`` : √(Σ aᵢⱼ²)  (default)
        - ``1``                         : max column-sum  (|A|₁)
        - ``'inf'``                     : max row-sum     (|A|∞)
        - ``2``                         : spectral norm via power iteration

    Examples
    --------
    >>> A.norm('frobenius')
    >>> A.norm(1)
    >>> A.norm('inf')
    >>> A.norm(2)
    """
    if kind in ("frobenius", "fro"):
        return math.sqrt(sum(_float(self.data[r][c]) ** 2
                             for r in range(self.rows)
                             for c in range(self.cols)))

    if kind == 1:
        # Max absolute column sum
        col_sums = [
            sum(abs(_float(self.data[r][c])) for r in range(self.rows))
            for c in range(self.cols)
        ]
        return max(col_sums)

    if kind == "inf":
        # Max absolute row sum
        row_sums = [
            sum(abs(_float(self.data[r][c])) for c in range(self.cols))
            for r in range(self.rows)
        ]
        return max(row_sums)

    if kind == 2:
        # Spectral norm (largest singular value) via power iteration on AᵀA
        # Works for any shape
        import random
        n = self.cols
        # Random starting vector
        v = [random.gauss(0, 1) for _ in range(n)]
        for _ in range(200):
            # Compute A @ v
            Av = [sum(_float(self.data[r][c]) * v[c] for c in range(self.cols))
                  for r in range(self.rows)]
            # Compute Aᵀ @ Av
            AtAv = [sum(_float(self.data[r][c]) * Av[r] for r in range(self.rows))
                    for c in range(self.cols)]
            mag = math.sqrt(sum(x * x for x in AtAv))
            if mag < 1e-15:
                return 0.0
            v = [x / mag for x in AtAv]
        # sigma_max = sqrt( vᵀAᵀAv / vᵀv ) = ||Av||
        Av_final = [sum(_float(self.data[r][c]) * v[c] for c in range(self.cols))
                    for r in range(self.rows)]
        return math.sqrt(sum(x * x for x in Av_final))

    raise ValueError(
        f"Unknown norm kind {kind!r}. "
        "Choose from: 'frobenius', 'fro', 1, 'inf', 2"
    )


# ---------------------------------------------------------------------------
# Attach everything to Matrix
# ---------------------------------------------------------------------------

def _attach_linalg(matrix_cls):
    matrix_cls.transpose       = transpose
    matrix_cls.trace           = trace
    matrix_cls.minor           = minor
    matrix_cls.cofactor        = cofactor
    matrix_cls.adjugate        = adjugate
    matrix_cls.determinant     = determinant
    matrix_cls.inverse         = inverse
    matrix_cls.rank            = rank
    matrix_cls.lu_decomposition = lu_decomposition
    matrix_cls.solve           = solve
    matrix_cls.norm            = norm
