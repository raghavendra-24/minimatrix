"""
tests/test_matrix.py – Full pytest test suite for minimatrix v0.2.0

Run with:  pytest tests/ -v
"""

import math
import pytest
from fractions import Fraction

from minimatrix import Matrix
from minimatrix.matrix.utils import (
    apply, element_wise, from_flat, diag,
    is_symmetric, is_identity, frobenius_norm,
)


# =====================================================================
# 1. Construction & Validation
# =====================================================================

class TestConstruction:
    def test_basic_creation(self):
        A = Matrix([[1, 2], [3, 4]])
        assert A.rows == 2 and A.cols == 2

    def test_floats_stored_internally(self):
        A = Matrix([[1, 2]])
        assert isinstance(A.data[0][0], float)

    def test_fraction_dtype(self):
        A = Matrix([[1, 2], [3, 4]], dtype='fraction')
        assert isinstance(A.data[0][0], Fraction)
        assert A.dtype == 'fraction'

    def test_complex_dtype(self):
        A = Matrix([[1+2j, 3+4j]], dtype='complex')
        assert isinstance(A.data[0][0], complex)

    def test_unknown_dtype_raises(self):
        with pytest.raises(ValueError):
            Matrix([[1, 2]], dtype='int128')

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            Matrix([])

    def test_empty_row_raises(self):
        with pytest.raises(ValueError):
            Matrix([[]])

    def test_jagged_raises(self):
        with pytest.raises(ValueError):
            Matrix([[1, 2], [3]])

    def test_single_element(self):
        A = Matrix([[42]])
        assert A.rows == 1 and A.cols == 1 and A[0][0] == 42.0

    def test_shape_property(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        assert A.shape == (2, 3)


# =====================================================================
# 2. Indexing & Slicing
# =====================================================================

class TestIndexing:
    def test_row_access(self):
        A = Matrix([[1, 2], [3, 4]])
        assert A[0] == [1.0, 2.0]

    def test_element_access(self):
        A = Matrix([[1, 2], [3, 4]])
        assert A[1][0] == 3.0

    def test_tuple_index(self):
        A = Matrix([[1, 2], [3, 4]])
        assert A[0, 1] == 2.0

    def test_setitem_element(self):
        A = Matrix([[1, 2], [3, 4]])
        A[0, 0] = 99
        assert A[0, 0] == 99.0

    def test_setitem_row(self):
        A = Matrix([[1, 2], [3, 4]])
        A[1] = [7, 8]
        assert A[1] == [7.0, 8.0]

    def test_slice_submatrix(self):
        A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        sub = A[0:2, 1:3]
        assert sub.shape == (2, 2)
        assert sub[0, 0] == pytest.approx(2.0)
        assert sub[1, 1] == pytest.approx(6.0)

    def test_slice_column(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        col = A[:, 1]
        assert col.shape == (2, 1)
        assert col[0, 0] == pytest.approx(2.0)
        assert col[1, 0] == pytest.approx(5.0)

    def test_slice_row(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        row = A[1, :]
        assert row.shape == (1, 3)
        assert row[0, 2] == pytest.approx(6.0)


# =====================================================================
# 3. Representation
# =====================================================================

class TestRepresentation:
    def test_repr_round_trip(self):
        A = Matrix([[1.0, 2.0], [3.0, 4.0]])
        B = eval(repr(A))
        assert A == B

    def test_repr_with_dtype(self):
        A = Matrix([[1, 2], [3, 4]], dtype='fraction')
        r = repr(A)
        assert "fraction" in r

    def test_str_returns_string(self):
        A = Matrix([[1, 2], [3, 4]])
        assert isinstance(str(A), str)


# =====================================================================
# 4. Hash — Matrix must be unhashable
# =====================================================================

class TestHash:
    def test_not_hashable(self):
        A = Matrix([[1, 2], [3, 4]])
        with pytest.raises(TypeError):
            hash(A)

    def test_not_in_set(self):
        A = Matrix([[1, 2]])
        with pytest.raises(TypeError):
            {A}


# =====================================================================
# 5. Equality
# =====================================================================

class TestEquality:
    def test_equal_matrices(self):
        assert Matrix([[1, 2], [3, 4]]) == Matrix([[1, 2], [3, 4]])

    def test_unequal_matrices(self):
        assert Matrix([[1, 2], [3, 4]]) != Matrix([[1, 2], [3, 5]])

    def test_different_shapes_not_equal(self):
        assert Matrix([[1, 2]]) != Matrix([[1], [2]])

    def test_float_tolerance(self):
        assert Matrix([[1.0000000001]]) == Matrix([[1.0]])

    def test_fraction_exact(self):
        A = Matrix([[1, 2]], dtype='fraction')
        B = Matrix([[1, 2]], dtype='fraction')
        assert A == B


# =====================================================================
# 6. Arithmetic
# =====================================================================

class TestAddition:
    def test_basic_addition(self):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[5, 6], [7, 8]])
        assert A + B == Matrix([[6, 8], [10, 12]])

    def test_addition_commutativity(self):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[5, 6], [7, 8]])
        assert A + B == B + A

    def test_addition_shape_mismatch(self):
        with pytest.raises(ValueError):
            Matrix([[1, 2]]) + Matrix([[1, 2, 3]])


class TestSubtraction:
    def test_basic(self):
        A = Matrix([[5, 6], [7, 8]])
        B = Matrix([[1, 2], [3, 4]])
        assert A - B == Matrix([[4, 4], [4, 4]])

    def test_self_zero(self):
        A = Matrix([[3, 7], [1, 5]])
        assert A - A == Matrix([[0, 0], [0, 0]])


class TestMultiplication:
    def test_matrix_mul(self):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[5, 6], [7, 8]])
        assert A * B == Matrix([[19, 22], [43, 50]])

    def test_matmul_operator(self):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[5, 6], [7, 8]])
        assert A @ B == Matrix([[19, 22], [43, 50]])

    def test_non_square_mul(self):
        A = Matrix([[1, 2, 3]])
        B = Matrix([[4], [5], [6]])
        assert A * B == Matrix([[32]])

    def test_scalar_right(self):
        A = Matrix([[1, 2], [3, 4]])
        assert A * 2 == Matrix([[2, 4], [6, 8]])

    def test_scalar_left(self):
        A = Matrix([[1, 2], [3, 4]])
        assert 3 * A == Matrix([[3, 6], [9, 12]])

    def test_identity_invariant(self):
        A = Matrix([[1, 2], [3, 4]])
        I = Matrix.identity(2)
        assert A * I == A and I * A == A

    def test_division(self):
        A = Matrix([[2, 4], [6, 8]])
        assert A / 2 == Matrix([[1, 2], [3, 4]])

    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            Matrix([[1, 2]]) / 0

    def test_negation(self):
        A = Matrix([[1, -2], [3, -4]])
        assert -A == Matrix([[-1, 2], [-3, 4]])


class TestMatrixPower:
    def test_power_zero(self):
        A = Matrix([[2, 1], [0, 3]])
        assert A ** 0 == Matrix.identity(2)

    def test_power_two(self):
        A = Matrix([[1, 1], [0, 1]])
        assert A ** 2 == Matrix([[1, 2], [0, 1]])

    def test_power_non_square_raises(self):
        with pytest.raises(ValueError):
            Matrix([[1, 2, 3]]) ** 2


# =====================================================================
# 7. Transpose & Trace
# =====================================================================

class TestTranspose:
    def test_square(self):
        A = Matrix([[1, 2], [3, 4]])
        assert A.transpose() == Matrix([[1, 3], [2, 4]])

    def test_rectangular(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        T = A.transpose()
        assert T.shape == (3, 2)

    def test_double_identity(self):
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        assert A.transpose().transpose() == A


class TestTrace:
    def test_basic(self):
        assert Matrix([[1, 2], [3, 4]]).trace() == 5.0

    def test_identity(self):
        assert Matrix.identity(4).trace() == 4.0

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            Matrix([[1, 2, 3]]).trace()


# =====================================================================
# 8. Determinant
# =====================================================================

class TestDeterminant:
    def test_1x1(self):
        assert Matrix([[7]]).determinant() == pytest.approx(7.0)

    def test_2x2(self):
        assert Matrix([[1, 2], [3, 4]]).determinant() == pytest.approx(-2.0)

    def test_3x3(self):
        A = Matrix([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
        assert A.determinant() == pytest.approx(-306.0, rel=1e-6)

    def test_singular(self):
        assert Matrix([[1, 2], [2, 4]]).determinant() == pytest.approx(0.0, abs=1e-9)

    def test_identity(self):
        assert Matrix.identity(4).determinant() == pytest.approx(1.0)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            Matrix([[1, 2, 3]]).determinant()

    def test_fraction_exact(self):
        A = Matrix([[1, 2], [3, 4]], dtype='fraction')
        assert A.determinant() == Fraction(-2, 1)

    def test_verbose_runs(self, capsys):
        A = Matrix([[1, 2], [3, 4]])
        A.determinant(verbose=True)
        out = capsys.readouterr().out
        assert "det" in out.lower()


# =====================================================================
# 9. Inverse
# =====================================================================

class TestInverse:
    def test_2x2(self):
        A = Matrix([[4, 7], [2, 6]])
        assert A * A.inverse() == Matrix.identity(2)

    def test_3x3(self):
        A = Matrix([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
        assert A * A.inverse() == Matrix.identity(3)

    def test_singular_raises(self):
        with pytest.raises(ValueError):
            Matrix([[1, 2], [2, 4]]).inverse()

    def test_identity_inverse_is_identity(self):
        I = Matrix.identity(3)
        assert I.inverse() == I


# =====================================================================
# 10. Rank
# =====================================================================

class TestRank:
    def test_full_rank(self):
        assert Matrix([[1, 0], [0, 1]]).rank() == 2

    def test_rank_deficient(self):
        assert Matrix([[1, 2], [2, 4]]).rank() == 1

    def test_rectangular(self):
        assert Matrix([[1, 0, 0], [0, 1, 0]]).rank() == 2

    def test_zero_matrix(self):
        assert Matrix([[0, 0], [0, 0]]).rank() == 0


# =====================================================================
# 11. LU Decomposition — now returns (P, L, U)
# =====================================================================

class TestLU:
    def test_returns_three_matrices(self):
        A = Matrix([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
        result = A.lu_decomposition()
        assert len(result) == 3

    def test_pa_equals_lu(self):
        """P @ A == L @ U must hold exactly."""
        A = Matrix([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
        P, L, U = A.lu_decomposition()
        assert P @ A == L @ U

    def test_l_lower_triangular(self):
        A = Matrix([[4, 3], [6, 3]])
        P, L, U = A.lu_decomposition()
        for r in range(L.rows):
            for c in range(r + 1, L.cols):
                assert abs(L[r, c]) < 1e-9

    def test_u_upper_triangular(self):
        A = Matrix([[4, 3], [6, 3]])
        P, L, U = A.lu_decomposition()
        for r in range(1, U.rows):
            for c in range(r):
                assert abs(U[r, c]) < 1e-9

    def test_p_is_permutation_matrix(self):
        A = Matrix([[0, 1], [1, 0]])
        P, L, U = A.lu_decomposition()
        # Each row of P has exactly one 1
        for r in range(P.rows):
            row_sum = sum(P[r][c] for c in range(P.cols))
            assert abs(row_sum - 1.0) < 1e-9


# =====================================================================
# 12. Linear System Solver (Phase 2b)
# =====================================================================

class TestSolve:
    def test_2x2_system(self):
        A = Matrix([[2, 1], [5, 3]])
        b = Matrix([[4], [7]])
        x = A.solve(b)
        assert A @ x == b

    def test_3x3_system(self):
        A = Matrix([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
        b = Matrix([[8], [-11], [-3]])
        x = A.solve(b)
        Ax = A @ x
        for r in range(3):
            assert Ax[r, 0] == pytest.approx(b[r, 0], rel=1e-6)

    def test_singular_raises(self):
        A = Matrix([[1, 2], [2, 4]])
        b = Matrix([[1], [2]])
        with pytest.raises(ValueError):
            A.solve(b)

    def test_verbose_runs(self, capsys):
        A = Matrix([[2, 1], [5, 3]])
        b = Matrix([[4], [7]])
        A.solve(b, verbose=True)
        out = capsys.readouterr().out
        assert "solve" in out.lower()


# =====================================================================
# 13. Matrix Norms (Phase 2e)
# =====================================================================

class TestNorms:
    def setup_method(self):
        self.A = Matrix([[1, -2], [3, 4]])

    def test_frobenius(self):
        # sqrt(1 + 4 + 9 + 16) = sqrt(30)
        assert self.A.norm() == pytest.approx(math.sqrt(30), rel=1e-6)
        assert self.A.norm('fro') == pytest.approx(math.sqrt(30), rel=1e-6)

    def test_norm_1(self):
        # col sums: |1|+|3|=4, |-2|+|4|=6  → max=6
        assert self.A.norm(1) == pytest.approx(6.0)

    def test_norm_inf(self):
        # row sums: |1|+|-2|=3, |3|+|4|=7  → max=7
        assert self.A.norm('inf') == pytest.approx(7.0)

    def test_norm_2_identity(self):
        # Spectral norm of identity is 1
        I = Matrix.identity(3)
        assert I.norm(2) == pytest.approx(1.0, abs=1e-3)

    def test_norm_unknown_raises(self):
        with pytest.raises(ValueError):
            self.A.norm('banana')


# =====================================================================
# 14. LaTeX Export (Phase 3)
# =====================================================================

class TestLatex:
    def test_basic(self):
        A = Matrix([[1, 2], [3, 4]])
        latex = A.to_latex()
        assert "pmatrix" in latex
        assert "1" in latex and "4" in latex

    def test_custom_env(self):
        A = Matrix([[1, 2]])
        latex = A.to_latex(env="bmatrix")
        assert "bmatrix" in latex

    def test_fraction_latex(self):
        A = Matrix([[1, 2], [3, 4]], dtype='fraction')
        latex = A.to_latex()
        assert "pmatrix" in latex


# =====================================================================
# 15. Visualize (Phase 3) — just confirm no crash
# =====================================================================

class TestVisualize:
    def test_runs_without_error(self, capsys):
        A = Matrix([[1, 2], [3, 4]])
        A.visualize()
        out = capsys.readouterr().out
        assert "Matrix" in out


# =====================================================================
# 16. dtype — Fraction exact arithmetic
# =====================================================================

class TestFractionDtype:
    def test_exact_multiply(self):
        A = Matrix([[1, 2], [3, 4]], dtype='fraction')
        B = Matrix([[5, 6], [7, 8]], dtype='fraction')
        C = A * B
        assert C[0, 0] == Fraction(19)
        assert C[1, 1] == Fraction(50)

    def test_exact_inverse(self):
        A = Matrix([[1, 2], [3, 4]])  # float
        Ainv = A.inverse()
        product = A * Ainv
        assert product == Matrix.identity(2)


# =====================================================================
# 17. Graphics Applications (Phase 4)
# =====================================================================

class TestGraphics:
    def test_rotation_2d_90(self):
        from minimatrix.applications import rotation_2d
        R = rotation_2d(90)
        p = Matrix([[1.0], [0.0]])
        Rp = R @ p
        assert Rp[0, 0] == pytest.approx(0.0, abs=1e-9)
        assert Rp[1, 0] == pytest.approx(1.0, abs=1e-9)

    def test_rotation_2d_identity(self):
        from minimatrix.applications import rotation_2d
        R = rotation_2d(0)
        assert R == Matrix.identity(2)

    def test_scale_2d(self):
        from minimatrix.applications import scale
        S = scale(2, 3)
        assert S.shape == (2, 2)
        p = Matrix([[1.0], [1.0]])
        Sp = S @ p
        assert Sp[0, 0] == pytest.approx(2.0)
        assert Sp[1, 0] == pytest.approx(3.0)

    def test_shear_2d(self):
        from minimatrix.applications import shear_2d
        Sh = shear_2d(shx=1.0)
        p = Matrix([[1.0], [1.0]])
        Shp = Sh @ p
        assert Shp[0, 0] == pytest.approx(2.0)
        assert Shp[1, 0] == pytest.approx(1.0)

    def test_translate_homogeneous(self):
        from minimatrix.applications import homogeneous_translate_2d
        T = homogeneous_translate_2d(3, -1)
        p = Matrix([[2.0], [4.0], [1.0]])
        Tp = T @ p
        assert Tp[0, 0] == pytest.approx(5.0)
        assert Tp[1, 0] == pytest.approx(3.0)

    def test_reflect_x(self):
        from minimatrix.applications import reflect_2d
        R = reflect_2d('x')
        p = Matrix([[1.0], [1.0]])
        Rp = R @ p
        assert Rp[1, 0] == pytest.approx(-1.0)

    def test_3d_rotation_x_90(self):
        from minimatrix.applications import rotation_3d_x
        R = rotation_3d_x(90)
        p = Matrix([[0.0], [1.0], [0.0]])
        Rp = R @ p
        assert Rp[1, 0] == pytest.approx(0.0, abs=1e-9)
        assert Rp[2, 0] == pytest.approx(1.0, abs=1e-9)


# =====================================================================
# 18. Class Constructors
# =====================================================================

class TestConstructors:
    def test_identity(self):
        assert is_identity(Matrix.identity(3))

    def test_zeros(self):
        Z = Matrix.zeros(2, 3)
        assert Z.shape == (2, 3)
        assert all(Z[r][c] == 0.0 for r in range(2) for c in range(3))

    def test_ones(self):
        O = Matrix.ones(2, 2)
        assert all(O[r][c] == 1.0 for r in range(2) for c in range(2))

    def test_identity_size_zero_raises(self):
        with pytest.raises(ValueError):
            Matrix.identity(0)


# =====================================================================
# 19. Utility Functions
# =====================================================================

class TestUtils:
    def test_apply(self):
        A = Matrix([[4, 9], [16, 25]])
        B = apply(A, math.sqrt)
        assert B == Matrix([[2, 3], [4, 5]])

    def test_element_wise(self):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[5, 6], [7, 8]])
        C = element_wise(A, B, lambda x, y: x * y)
        assert C == Matrix([[5, 12], [21, 32]])

    def test_from_flat(self):
        M = from_flat([1, 2, 3, 4, 5, 6], 2, 3)
        assert M.shape == (2, 3) and M[1][2] == 6.0

    def test_from_flat_wrong_size(self):
        with pytest.raises(ValueError):
            from_flat([1, 2, 3], 2, 2)

    def test_diag(self):
        D = diag([1, 2, 3])
        assert D[0, 0] == 1.0 and D[1, 1] == 2.0 and D[2, 2] == 3.0
        assert D[0, 1] == 0.0

    def test_is_symmetric(self):
        assert is_symmetric(Matrix([[1, 2], [2, 1]]))

    def test_frobenius_norm(self):
        A = Matrix([[1, 0], [0, 1]])
        assert frobenius_norm(A) == pytest.approx(math.sqrt(2))

    def test_flatten(self):
        A = Matrix([[1, 2], [3, 4]])
        assert A.flatten() == [1.0, 2.0, 3.0, 4.0]

    def test_copy_independent(self):
        A = Matrix([[1, 2], [3, 4]])
        B = A.copy()
        B[0, 0] = 99
        assert A[0, 0] == 1.0


# =====================================================================
# 20. Minor & Cofactor & Adjugate
# =====================================================================

class TestMinorCofactorAdjugate:
    def test_minor_2x2(self):
        A = Matrix([[1, 2], [3, 4]])
        assert A.minor(0, 0) == Matrix([[4]])
        assert A.minor(0, 1) == Matrix([[3]])

    def test_cofactor_sign(self):
        A = Matrix([[1, 2], [3, 4]])
        assert A.cofactor(0, 0) == pytest.approx(4.0)
        assert A.cofactor(0, 1) == pytest.approx(-3.0)

    def test_adjugate_times_matrix_is_det_times_identity(self):
        A = Matrix([[1, 2], [3, 4]])
        adj = A.adjugate()
        det = A.determinant()
        # A * adj(A) = det(A) * I
        product = A * adj
        expected = Matrix([[det, 0], [0, det]])
        assert product == expected
