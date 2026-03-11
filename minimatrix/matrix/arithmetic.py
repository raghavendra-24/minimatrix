"""
minimatrix.matrix.arithmetic – Operator overloading for Matrix arithmetic.

Attached to Matrix via _attach_arithmetic().
Operators covered: +, -, *, rmul, /, //, -, **, @
"""

from __future__ import annotations
from typing import Union


def _add(self, other):
    """Element-wise addition: A + B."""
    from minimatrix.matrix.core import Matrix
    if isinstance(other, Matrix):
        self._require_same_shape(other, "addition")
        return Matrix(
            [[self.data[r][c] + other.data[r][c] for c in range(self.cols)]
             for r in range(self.rows)],
            dtype=self.dtype,
        )
    return NotImplemented


def _sub(self, other):
    """Element-wise subtraction: A - B."""
    from minimatrix.matrix.core import Matrix
    if isinstance(other, Matrix):
        self._require_same_shape(other, "subtraction")
        return Matrix(
            [[self.data[r][c] - other.data[r][c] for c in range(self.cols)]
             for r in range(self.rows)],
            dtype=self.dtype,
        )
    return NotImplemented


def _mul(self, other):
    """Matrix multiplication (A * B) or scalar multiplication (A * k)."""
    from minimatrix.matrix.core import Matrix, _cast
    if isinstance(other, (int, float, complex)):
        s = _cast(other, self._dtype_type)
        return Matrix(
            [[self.data[r][c] * s for c in range(self.cols)]
             for r in range(self.rows)],
            dtype=self.dtype,
        )
    if isinstance(other, Matrix):
        self._require_multipliable(other)
        zero = self._dtype_type(0)
        result = [[zero] * other.cols for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(other.cols):
                acc = zero
                for k in range(self.cols):
                    acc += self.data[r][k] * other.data[k][c]
                result[r][c] = acc
        return Matrix(result, dtype=self.dtype)
    return NotImplemented


def _matmul(self, other):
    """Matrix multiplication via @ operator."""
    return _mul(self, other)


def _rmul(self, scalar):
    """Scalar multiplication: k * A."""
    if isinstance(scalar, (int, float, complex)):
        return _mul(self, scalar)
    return NotImplemented


def _neg(self):
    """Unary negation: -A."""
    from minimatrix.matrix.core import Matrix
    return Matrix(
        [[-self.data[r][c] for c in range(self.cols)]
         for r in range(self.rows)],
        dtype=self.dtype,
    )


def _truediv(self, scalar):
    """Scalar division: A / k."""
    from minimatrix.matrix.core import Matrix, _cast
    if isinstance(scalar, (int, float, complex)):
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide matrix by zero.")
        s = _cast(scalar, self._dtype_type)
        return Matrix(
            [[self.data[r][c] / s for c in range(self.cols)]
             for r in range(self.rows)],
            dtype=self.dtype,
        )
    return NotImplemented


def _pow(self, n: int):
    """Integer matrix power: A ** n (square matrices only)."""
    from minimatrix.matrix.core import Matrix
    if not isinstance(n, int) or n < 0:
        raise ValueError("Exponent must be a non-negative integer.")
    self._require_square("matrix power")
    if n == 0:
        return Matrix.identity(self.rows, dtype=self.dtype)
    result = Matrix.identity(self.rows, dtype=self.dtype)
    base = self.copy()
    while n:
        if n % 2 == 1:
            result = result * base
        base = base * base
        n //= 2
    return result


def _attach_arithmetic(matrix_cls):
    """Attach all arithmetic dunder methods to *matrix_cls*."""
    matrix_cls.__add__      = _add
    matrix_cls.__radd__     = lambda self, other: _add(self, other)
    matrix_cls.__sub__      = _sub
    matrix_cls.__mul__      = _mul
    matrix_cls.__matmul__   = _matmul
    matrix_cls.__rmul__     = _rmul
    matrix_cls.__neg__      = _neg
    matrix_cls.__truediv__  = _truediv
    matrix_cls.__pow__      = _pow
