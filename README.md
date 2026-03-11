<div align="center">

<h1>matrixa</h1>

<p><strong>A pure-Python matrix library — no dependencies, built for learning and correctness.</strong></p>

[![PyPI version](https://img.shields.io/pypi/v/matrixa?color=blue)](https://pypi.org/project/matrixa/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-110%20passed-brightgreen)](#)

</div>

---

**matrixa** gives Python a first-class `Matrix` type that works the way you'd expect —
supporting operator overloading, multiple numeric types, NumPy-style slicing, and, uniquely,
a **verbose "show your work" mode** that prints the algorithm as it runs.

```python
from matrixa import Matrix

A = Matrix([[6, 1, 1],
            [4, -2, 5],
            [2,  8, 7]])

A.determinant(verbose=True)
```

```
─────────────────────────────────────────────────────
  determinant()  —  3×3 matrix
─────────────────────────────────────────────────────
  Row 0: [6.0, 1.0, 1.0]
  Row 1: [4.0, -2.0, 5.0]
  Row 2: [2.0, 8.0, 7.0]

  Using LU decomposition with partial pivoting (Doolittle):

  Permutation vector P = [0, 2, 1]
  Row-swap parity (sign) = -1

  Upper-triangular U diagonal:
    U[0,0] = 6
    U[1,1] = 7.66667
    U[2,2] = 6.65217

  det = sign × ∏ U[i,i]
      = -1 × 306
      = -306
─────────────────────────────────────────────────────
```

---

## Why matrixa?

Most matrix libraries are black boxes. **matrixa is a glass box.**

It is aimed at students, educators, and engineers who want to understand what
the algorithm is doing — not just get an answer. Every operation can explain itself.

| | matrixa | NumPy |
|---|---|---|
| Dependencies | **Zero** | C compiler + BLAS |
| `verbose=True` step-by-step mode | ✅ | ❌ |
| Exact rational arithmetic (`Fraction`) | ✅ | ❌ |
| LaTeX export | ✅ | ❌ |
| GPU / large-data performance | ❌ | ✅ |

> Use matrixa when you want to **understand** the math.  
> Use NumPy when you need to **crunch** the math.

---

## Installation

```bash
pip install matrixa
```

Python 3.8+ — no other dependencies required.

---

## Quick Start

```python
from matrixa import Matrix

# ── Construction ──────────────────────────────
A = Matrix([[1, 2],
            [3, 4]])

# ── Arithmetic operators ──────────────────────
B = Matrix([[5, 6], [7, 8]])

print(A + B)          # element-wise addition
print(A @ B)          # matrix multiplication
print(3 * A)          # scalar multiplication
print(A ** 3)         # matrix power (binary exp)

# ── Linear algebra ────────────────────────────
print(A.determinant())        # -2.0
print(A.inverse())
print(A.rank())               # 2
print(A.trace())              # 5.0

P, L, U = A.lu_decomposition()
assert P @ A == L @ U         # always holds

# ── Solve Ax = b ─────────────────────────────
b = Matrix([[1], [2]])
x = A.solve(b)
print(A @ x)                  # should equal b

# ── Slicing like NumPy ───────────────────────
M = Matrix([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])

print(M[0:2, 1:3])    # 2×2 sub-matrix
print(M[:, 0])        # first column
print(M[1, :])        # second row
```

---

## Exact Arithmetic with Fractions

Float rounding errors are eliminated when you use `dtype='fraction'`:

```python
from fractions import Fraction

A = Matrix([[1, 2],
            [3, 4]], dtype='fraction')

print(A.determinant())    # Fraction(-2, 1)   ← exact, not -2.0000000003
print(A.to_latex())       # \begin{pmatrix}1 & 2 \\ 3 & 4\end{pmatrix}
```

Three dtypes are supported:

```python
Matrix([[1, 2], [3, 4]])                     # dtype='float'    (default)
Matrix([[1, 2], [3, 4]], dtype='fraction')   # exact rationals
Matrix([[1+2j, 3]], dtype='complex')         # complex numbers
```

---

## Verbose Mode — The Unique Feature

Every major operation accepts `verbose=True` to print a step-by-step walkthrough.
**There is nothing else like this in pure Python.**

```python
A = Matrix([[2, 1], [5, 3]])
b = Matrix([[4], [7]])
x = A.solve(b, verbose=True)
```

```
────────────────────────────────────────────────────────────
  solve(Ax = b)  —  Forward elimination with partial pivoting
────────────────────────────────────────────────────────────
  Augmented matrix [A | b]:
    [    2.0       1.0    |     4.0 ]
    [    5.0       3.0    |     7.0 ]

  Step: Swap R0 ↔ R1  (partial pivot)
  Step: Scale R0 ÷ 5  (make pivot = 1)
  Step: R1 ← R1 − (0.4)·R0
  Step: Scale R1 ÷ 0.8  (make pivot = 1)
  Step: R0 ← R0 − (0.2)·R1

  Solution x:
    x[0] =       5.0
    x[1] =      -6.0
────────────────────────────────────────────────────────────
```

---

## Matrix Norms

```python
A.norm()           # Frobenius norm  (default)
A.norm('fro')      # same
A.norm(1)          # max column-sum  norm
A.norm('inf')      # max row-sum     norm
A.norm(2)          # spectral norm   (power iteration)
```

---

## 2D / 3D Graphics Transforms

```python
from matrixa.applications import (
    rotation_2d, rotation_3d_x,
    scale, shear_2d, reflect_2d,
    homogeneous_translate_2d,
)

# Compose transforms with @
T = rotation_2d(45) @ scale(2, 0.5)

# Homogeneous coordinates
T = homogeneous_translate_2d(3, -1)
p = Matrix([[2.0], [4.0], [1.0]])
print(T @ p)   # → [[5], [3], [1]]
```

---

## LaTeX & Visualization

```python
print(A.to_latex())
# \begin{pmatrix}1 & 2\\3 & 4\end{pmatrix}

print(A.to_latex(env='bmatrix'))   # square brackets
print(A.to_latex(env='vmatrix'))   # determinant notation

A.visualize()   # Unicode block heatmap in the terminal
```

---

## Full API Reference

### Core `Matrix` class

```python
Matrix(data, dtype='float')       # constructor
Matrix.identity(n, dtype)         # n×n identity
Matrix.zeros(rows, cols, dtype)   # zero matrix
Matrix.ones(rows, cols, dtype)    # ones matrix
```

### Operators

| Syntax | Operation |
|--------|-----------|
| `A + B` | element-wise add |
| `A - B` | element-wise subtract |
| `A * B` or `A @ B` | matrix multiply |
| `k * A` / `A * k` | scalar multiply |
| `A / k` | scalar divide |
| `A ** n` | integer matrix power |
| `-A` | negation |

### Linear Algebra Methods

| Method | Returns |
|--------|---------|
| `A.determinant(verbose=False)` | `float` or `Fraction` |
| `A.inverse(verbose=False)` | `Matrix` |
| `A.transpose()` | `Matrix` |
| `A.trace()` | scalar |
| `A.rank()` | `int` |
| `A.lu_decomposition()` | `(P, L, U)` — `P @ A == L @ U` |
| `A.solve(b, verbose=False)` | solution `Matrix` x |
| `A.norm(kind)` | `float` — kind: `'frobenius'`, `1`, `'inf'`, `2` |
| `A.minor(r, c)` | submatrix |
| `A.cofactor(r, c)` | `float` |
| `A.adjugate()` | `Matrix` |

### Utilities (`from matrixa.matrix.utils import ...`)

```python
apply(mat, func)               # element-wise unary function
element_wise(A, B, func)       # element-wise binary function
from_flat(flat, rows, cols)    # reshape a flat list
diag(values)                   # diagonal matrix from list
is_symmetric(mat)              # bool
is_identity(mat)               # bool
frobenius_norm(mat)            # float
```

---

## Project Structure

```
matrixa/
├── __init__.py
├── matrix/
│   ├── core.py           Matrix class, dtype, slicing, LaTeX, visualize
│   ├── arithmetic.py     Operator overloading
│   ├── linalg.py         det, inv, LU, solve, norms, …
│   └── utils.py          Standalone utility functions
└── applications/
    └── graphics.py       2D / 3D transformation matrices
tests/
└── test_matrix.py        110 tests across 20 test classes
```

---

## Contributing

Contributions, bug reports, and feature requests are welcome!  
Open an issue at [github.com/raghavendra-24/matrixa/issues](https://github.com/raghavendra-24/matrixa/issues).

```bash
git clone https://github.com/raghavendra-24/matrixa
pip install -e ".[dev]"     # installs pytest
pytest tests/ -v
```

---

## License

[MIT](LICENSE) © 2026 [Raghavendra Raju Palagani](https://github.com/raghavendra-24)
