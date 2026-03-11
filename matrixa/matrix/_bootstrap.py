"""
matrixa.matrix._bootstrap – Wires arithmetic & linalg methods onto Matrix.

This file is imported by matrixa/matrix/__init__.py immediately after
Matrix is defined, so Matrix is always fully equipped.
"""

from matrixa.matrix.arithmetic import _attach_arithmetic
from matrixa.matrix.linalg import _attach_linalg
from matrixa.matrix.core import Matrix

_attach_arithmetic(Matrix)
_attach_linalg(Matrix)
