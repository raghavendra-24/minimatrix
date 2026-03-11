"""
minimatrix.matrix._bootstrap – Wires arithmetic & linalg methods onto Matrix.

This file is imported by minimatrix/matrix/__init__.py immediately after
Matrix is defined, so Matrix is always fully equipped.
"""

from minimatrix.matrix.arithmetic import _attach_arithmetic
from minimatrix.matrix.linalg import _attach_linalg
from minimatrix.matrix.core import Matrix

_attach_arithmetic(Matrix)
_attach_linalg(Matrix)
