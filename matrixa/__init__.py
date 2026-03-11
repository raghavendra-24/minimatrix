"""
matrixa — A pure-Python linear algebra library with a custom Matrix type.

Quick start
-----------
>>> from matrixa import Matrix
>>> A = Matrix([[1, 2], [3, 4]])
>>> A.determinant()
-2.0
>>> A.inverse()
Matrix([
  [ -2.0   1.0 ]
  [  1.5  -0.5 ]
])
"""

from matrixa.matrix.core import Matrix

__all__ = ["Matrix"]
__version__ = "0.2.0"
__author__ = "Raghavendra Raju Palagani"
