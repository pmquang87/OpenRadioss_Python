"""
M481 — Unit tests for common/fastmath.py numeric primitives.

The fastmath module provides four vectorized NumPy primitives optimized
for the cycle path (small-array regime):

* ``cross3``       — cross product of (..., 3) arrays
* ``norm3``        — Euclidean norm over last axis of (..., 3) arrays
* ``det_inv33``    — determinant + inverse of stacked (n, 3, 3) matrices
* ``scatter_add3`` — force scatter-assembly (Fortran ``asspar``)

These are the core numeric building blocks used by every element kernel.
The module docstring documents that cross3 and norm3 are bitwise identical
to np.cross and np.linalg.norm; det_inv33 uses cofactor expansion (differs
from LAPACK at machine precision); scatter_add3 uses np.bincount (bitwise
identical to np.add.at for zero-initialized targets).

Tests verify correctness against NumPy reference, edge cases, and the
bitwise-identical contract where claimed.
"""

import numpy as np
import pytest

from pyradioss.common.fastmath import cross3, norm3, det_inv33, scatter_add3


# ======================================================================
# cross3 — cross product
# ======================================================================
class TestCross3:
    """cross3(a, b) must match np.cross for (..., 3) arrays."""

    def test_unit_vectors(self):
        """i × j = k, j × k = i, k × i = j."""
        i = np.array([1.0, 0.0, 0.0])
        j = np.array([0.0, 1.0, 0.0])
        k = np.array([0.0, 0.0, 1.0])

        np.testing.assert_array_equal(cross3(i, j), k)
        np.testing.assert_array_equal(cross3(j, k), i)
        np.testing.assert_array_equal(cross3(k, i), j)

    def test_anti_commutative(self):
        """a × b = -(b × a)."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((10, 3))
        b = rng.standard_normal((10, 3))

        np.testing.assert_allclose(cross3(a, b), -cross3(b, a), atol=1e-15)

    def test_parallel_vectors(self):
        """Parallel vectors → zero cross product."""
        a = np.array([[3.0, 0.0, 0.0]])
        b = np.array([[7.0, 0.0, 0.0]])

        result = cross3(a, b)

        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_bitwise_identical_to_np_cross(self):
        """cross3 must produce bitwise identical results to np.cross."""
        rng = np.random.default_rng(123)
        a = rng.standard_normal((50, 3))
        b = rng.standard_normal((50, 3))

        result = cross3(a, b)
        reference = np.cross(a, b)

        np.testing.assert_array_equal(result, reference)

    def test_broadcasting(self):
        """Single vector crossed with batch."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([[0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])

        result = cross3(a, b)

        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result[0], [0.0, 0.0, 1.0])   # i × j = k
        np.testing.assert_array_equal(result[1], [0.0, -1.0, 0.0])  # i × k = -j

    def test_single_pair(self):
        """Single 1D pair (no batch dimension)."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])

        result = cross3(a, b)
        reference = np.cross(a, b)

        np.testing.assert_array_equal(result, reference)


# ======================================================================
# norm3 — Euclidean norm
# ======================================================================
class TestNorm3:
    """norm3(a) must match np.linalg.norm(a, axis=-1) for (..., 3)."""

    def test_unit_vectors(self):
        """Unit vectors have norm 1."""
        vecs = np.eye(3)
        norms = norm3(vecs)
        np.testing.assert_allclose(norms, [1.0, 1.0, 1.0])

    def test_known_values(self):
        """3-4-5 triangle → norm = 5."""
        a = np.array([[3.0, 4.0, 0.0]])
        np.testing.assert_allclose(norm3(a), [5.0])

    def test_zero_vector(self):
        """Zero vector has norm 0."""
        a = np.array([[0.0, 0.0, 0.0]])
        np.testing.assert_allclose(norm3(a), [0.0])

    def test_batch(self):
        """Batch of vectors."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((100, 3))

        result = norm3(a)
        reference = np.linalg.norm(a, axis=-1)

        np.testing.assert_allclose(result, reference, rtol=1e-15)

    def test_single_vector(self):
        """1D input (shape (3,))."""
        a = np.array([1.0, 2.0, 2.0])
        np.testing.assert_allclose(norm3(a), 3.0)


# ======================================================================
# det_inv33 — determinant + inverse of (n, 3, 3)
# ======================================================================
class TestDetInv33:
    """det_inv33(J) returns (det, inv) for stacked 3×3 matrices."""

    def test_identity(self):
        """Identity matrix: det=1, inv=I."""
        J = np.eye(3).reshape(1, 3, 3)
        det, inv = det_inv33(J)

        np.testing.assert_allclose(det, [1.0])
        np.testing.assert_allclose(inv[0], np.eye(3), atol=1e-15)

    def test_diagonal(self):
        """Diagonal matrix: det = product of diags, inv = 1/diags."""
        J = np.zeros((1, 3, 3))
        J[0] = np.diag([2.0, 3.0, 5.0])

        det, inv = det_inv33(J)

        np.testing.assert_allclose(det, [30.0])
        np.testing.assert_allclose(inv[0], np.diag([0.5, 1.0/3.0, 0.2]),
                                   rtol=1e-14)

    def test_inverse_product(self):
        """J @ inv(J) = I for random well-conditioned matrices."""
        rng = np.random.default_rng(42)
        n = 20
        J = rng.standard_normal((n, 3, 3)) + 3.0 * np.eye(3)  # well-conditioned

        det, inv = det_inv33(J)

        for k in range(n):
            product = J[k] @ inv[k]
            np.testing.assert_allclose(product, np.eye(3), atol=1e-12)

    def test_determinant_vs_numpy(self):
        """Determinants match np.linalg.det to machine precision."""
        rng = np.random.default_rng(123)
        n = 30
        J = rng.standard_normal((n, 3, 3)) + 5.0 * np.eye(3)

        det, _ = det_inv33(J)
        det_ref = np.linalg.det(J)

        # Cofactor expansion differs from LAPACK LU, but difference is
        # at machine precision for well-conditioned matrices
        np.testing.assert_allclose(det, det_ref, rtol=1e-13)

    def test_batch_of_one(self):
        """Single matrix (n=1)."""
        J = np.array([[[1.0, 2.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]]])

        det, inv = det_inv33(J)

        np.testing.assert_allclose(det, [1.0])
        product = J[0] @ inv[0]
        np.testing.assert_allclose(product, np.eye(3), atol=1e-15)

    def test_negative_determinant(self):
        """Matrix with negative determinant (reflection)."""
        J = np.array([[[-1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]]])

        det, inv = det_inv33(J)

        np.testing.assert_allclose(det, [-1.0])
        product = J[0] @ inv[0]
        np.testing.assert_allclose(product, np.eye(3), atol=1e-15)


# ======================================================================
# scatter_add3 — force scatter-assembly
# ======================================================================
class TestScatterAdd3:
    """scatter_add3(target, idx, values) accumulates values into target."""

    def test_simple_scatter(self):
        """Values scattered to distinct indices."""
        target = np.zeros((4, 3))
        idx = np.array([0, 1, 2, 3])
        values = np.array([[1.0, 0.0, 0.0],
                           [0.0, 2.0, 0.0],
                           [0.0, 0.0, 3.0],
                           [4.0, 5.0, 6.0]])

        scatter_add3(target, idx, values)

        expected = np.array([[1.0, 0.0, 0.0],
                             [0.0, 2.0, 0.0],
                             [0.0, 0.0, 3.0],
                             [4.0, 5.0, 6.0]])
        np.testing.assert_allclose(target, expected)

    def test_duplicate_indices(self):
        """Multiple values to the same index accumulate."""
        target = np.zeros((2, 3))
        idx = np.array([0, 0, 1, 1])
        values = np.array([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [10.0, 0.0, 0.0],
                           [0.0, 20.0, 0.0]])

        scatter_add3(target, idx, values)

        np.testing.assert_allclose(target[0], [5.0, 7.0, 9.0])
        np.testing.assert_allclose(target[1], [10.0, 20.0, 0.0])

    def test_additive_to_existing(self):
        """Values add to existing target contents."""
        target = np.array([[100.0, 200.0, 300.0],
                           [0.0, 0.0, 0.0]])
        idx = np.array([0])
        values = np.array([[1.0, 2.0, 3.0]])

        scatter_add3(target, idx, values)

        np.testing.assert_allclose(target[0], [101.0, 202.0, 303.0])
        np.testing.assert_allclose(target[1], [0.0, 0.0, 0.0])

    def test_matches_np_add_at(self):
        """Result matches np.add.at for zero-initialized target."""
        rng = np.random.default_rng(42)
        n_nodes = 50
        n_entries = 200
        idx = rng.integers(0, n_nodes, size=n_entries)
        values = rng.standard_normal((n_entries, 3))

        target1 = np.zeros((n_nodes, 3))
        scatter_add3(target1, idx, values)

        target2 = np.zeros((n_nodes, 3))
        np.add.at(target2, idx, values)

        np.testing.assert_allclose(target1, target2, atol=1e-14)

    def test_empty_scatter(self):
        """Zero-length idx/values → target unchanged."""
        target = np.array([[1.0, 2.0, 3.0]])
        idx = np.array([], dtype=int)
        values = np.zeros((0, 3))

        scatter_add3(target, idx, values)

        np.testing.assert_array_equal(target, [[1.0, 2.0, 3.0]])
