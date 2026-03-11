"""
examples/demo.py – End-to-end demonstration of minimatrix.

Run from the project root:
    python examples/demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
from minimatrix import Matrix
from minimatrix.matrix.utils import apply, diag, from_flat, frobenius_norm, is_symmetric


def section(title: str) -> None:
    width = 50
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


# ── Construction ────────────────────────────────────────────────────────
section("1. Matrix Construction")

A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A =")
print(A)
print("\nB =")
print(B)
print(f"\nA.shape = {A.shape}")


# ── Arithmetic ──────────────────────────────────────────────────────────
section("2. Arithmetic Operators")

print("A + B =")
print(A + B)

print("\nA - B =")
print(A - B)

print("\nA * B  (matrix multiplication) =")
print(A * B)

print("\n3 * A  (scalar multiplication) =")
print(3 * A)

print("\nA / 2  (scalar division) =")
print(A / 2)

print("\n-A  (negation) =")
print(-A)

print("\nA ** 3  (matrix power) =")
print(A ** 3)


# ── Indexing ─────────────────────────────────────────────────────────────
section("3. Indexing")

print(f"A[0]      → {A[0]}")
print(f"A[1][0]   → {A[1][0]}")
print(f"A[0, 1]   → {A[0, 1]}")


# ── Transpose ────────────────────────────────────────────────────────────
section("4. Transpose")

R = Matrix([[1, 2, 3], [4, 5, 6]])
print("R ="); print(R)
print("\nR.transpose() ="); print(R.transpose())


# ── Trace & Determinant ──────────────────────────────────────────────────
section("5. Trace & Determinant")

print(f"A.trace()        = {A.trace()}")
print(f"A.determinant()  = {A.determinant()}")

C3 = Matrix([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
print(f"\ndet of 3×3 matrix = {C3.determinant():.4f}  (expected -306)")


# ── Inverse ──────────────────────────────────────────────────────────────
section("6. Inverse  (Gauss-Jordan)")

Ainv = A.inverse()
print("A.inverse() ="); print(Ainv)
print("\nA * A.inverse() (should be Identity) ="); print(A * Ainv)


# ── Rank ──────────────────────────────────────────────────────────────────
section("7. Rank")

print(f"A.rank()                     = {A.rank()}")
print(f"Matrix([[1,2],[2,4]]).rank() = {Matrix([[1,2],[2,4]]).rank()}  (rank-deficient)")


# ── LU Decomposition ─────────────────────────────────────────────────────
section("8. LU Decomposition")

M = Matrix([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
L, U = M.lu_decomposition()
print("L ="); print(L)
print("\nU ="); print(U)
print("\nL * U  (should equal M) ="); print(L * U)


# ── Class Constructors ────────────────────────────────────────────────────
section("9. Special Matrices")

print("Identity(3) ="); print(Matrix.identity(3))
print("\nZeros(2, 4) ="); print(Matrix.zeros(2, 4))
print("\nOnes(3, 2) =");  print(Matrix.ones(3, 2))
print("\nDiagonal([1,2,3]) ="); print(diag([1, 2, 3]))


# ── Utilities ─────────────────────────────────────────────────────────────
section("10. Utility Functions")

D = Matrix([[4, 9], [16, 25]])
print("apply(sqrt) to [[4,9],[16,25]] ="); print(apply(D, math.sqrt))

flat = from_flat(list(range(1, 7)), 2, 3)
print("\nfrom_flat([1..6], 2, 3) ="); print(flat)

print(f"\nA.flatten()           = {A.flatten()}")
print(f"frobenius_norm(A)     = {frobenius_norm(A):.6f}")
print(f"is_symmetric(A)       = {is_symmetric(A)}")
print(f"is_symmetric(A+A.T()) = ", end="")
print(is_symmetric(A + A.transpose()))


# ── Error Handling ────────────────────────────────────────────────────────
section("11. Error Handling")

def try_op(desc, fn):
    try:
        fn()
        print(f"  {desc}: no error (unexpected)")
    except (ValueError, ZeroDivisionError) as e:
        print(f"  {desc}: ✓  {e}")

try_op("Add mismatched shapes", lambda: A + Matrix([[1, 2, 3]]))
try_op("Multiply mismatched",   lambda: A * Matrix([[1, 2, 3]]))
try_op("Det non-square",        lambda: R.determinant())
try_op("Inverse singular",      lambda: Matrix([[1, 2], [2, 4]]).inverse())
try_op("Divide by zero",        lambda: A / 0)
try_op("from_flat wrong size",  lambda: from_flat([1, 2, 3], 2, 2))

print("\n✅  Demo complete.")
