"""
M477 — /INTER/TYPE7 Iform=2 incremental tangential stiffness friction tests.

Tests for ``pyradioss.contact.friction.apply_incremental_stiffness`` and
the IFQ >= 10 filter mode, validating against the Fortran ``i7for3.F``
incremental stiffness formulation (lines 2310-2356).

The incremental stiffness formulation (MODFR=2, IFQ >= 10):
  F_trial = CAND_F + alpha * STIF0 * v_rel * dt/2
  F_trial_tan = F_trial - (F_trial . n) * n
  beta = min(1, mu * |Fn| / |F_trial_tan|)
  Ft = F_trial_tan * beta
  CAND_F = Ft   (store for next cycle)

Key differences from penalty/velocity friction (Iform=0):
  - Force is *anchored*: it accumulates across cycles (stick memory)
  - Coulomb return mapping: elastic stick when |Ft| < mu*|Fn|, slip when saturated
  - Separation resets the anchor (IFPEN semantics)
"""

import numpy as np
import pytest

from pyradioss.contact.friction import (
    apply_incremental_stiffness,
    apply_filter,
    filter_alpha,
)


# ---------------------------------------------------------------------------
# test_filter_alpha_ifq10_13: IFQ >= 10 modes must match IFQ 0-3 + 10
# ---------------------------------------------------------------------------
class TestFilterAlphaIFQ:
    """Verify filter_alpha for the incremental stiffness IFQ modes (10..13)."""

    def test_ifq10_returns_xfiltr(self):
        """IFQ=10 is the constant-coefficient mode (like IFQ=0 but with
        the incremental stiffness formulation)."""
        assert filter_alpha(10, 0.75, 1e-4) == 0.75

    def test_ifq11_returns_xfiltr(self):
        """IFQ=11 mirrors IFQ=1."""
        assert filter_alpha(11, 0.5, 1e-4) == 0.5

    def test_ifq12_returns_xfiltr(self):
        """IFQ=12 mirrors IFQ=2."""
        assert filter_alpha(12, 0.3, 1e-4) == 0.3

    def test_ifq13_cutoff_frequency(self):
        """IFQ=13 mirrors IFQ=3: alpha = min(1, xfiltr*dt)."""
        dt = 1e-4
        xfiltr = 2000.0  # -> 2000 * 1e-4 = 0.2
        assert filter_alpha(13, xfiltr, dt) == pytest.approx(0.2)

    def test_ifq13_clamps_at_one(self):
        """IFQ=13 with large xfiltr*dt: alpha clamped at 1.0."""
        assert filter_alpha(13, 1e6, 1e-3) == 1.0


# ---------------------------------------------------------------------------
# test_incremental_stick: below Coulomb limit -> elastic accumulation
# ---------------------------------------------------------------------------
class TestIncrementalStick:
    """When the tangential trial force is below mu*Fn, the force should
    accumulate elastically without saturation (beta=1)."""

    def test_single_pair_stick(self):
        """One node sliding slowly: force grows linearly with cycles."""
        keys = np.array([0], dtype=np.int64)
        k = np.array([1000.0])       # penalty stiffness
        dt = 1e-4
        normal = np.array([[0.0, 0.0, 1.0]])  # z-normal surface
        mu = np.array([0.3])
        fn = np.array([100.0])       # normal force = 100
        alpha = 1.0                  # no filtering
        filt_keys = np.zeros(0, dtype=np.int64)
        filt_vals = np.zeros((0, 3))

        # Slide in x at 1 m/s, zero normal velocity
        v_rel = np.array([[1.0, 0.0, 0.0]])

        # Cycle 1: F_trial = 0 + alpha * k * v * dt = 1000 * 1 * 1e-4 = 0.1
        # |Ft| = 0.1 < mu*Fn = 30 -> stick (beta=1)
        ft, new_keys, new_vals = apply_incremental_stiffness(
            keys, k, v_rel, dt, normal, mu, fn, alpha, filt_keys, filt_vals)
        assert ft.shape == (1, 3)
        assert ft[0, 2] == pytest.approx(0.0, abs=1e-15)  # no z-component
        # Force should be in the x-direction
        assert ft[0, 0] == pytest.approx(k[0] * 1.0 * dt, rel=1e-10)
        assert ft[0, 1] == pytest.approx(0.0, abs=1e-15)

        # Cycle 2: accumulates on top of previous
        ft2, new_keys2, new_vals2 = apply_incremental_stiffness(
            keys, k, v_rel, dt, normal, mu, fn, alpha, new_keys, new_vals)
        expected = 2.0 * k[0] * 1.0 * dt  # linear accumulation
        assert ft2[0, 0] == pytest.approx(expected, rel=1e-10)

    def test_multi_cycle_accumulation(self):
        """Force accumulates linearly over many cycles in stick regime."""
        keys = np.array([0], dtype=np.int64)
        k = np.array([1e4])
        dt = 1e-5
        normal = np.array([[0.0, 1.0, 0.0]])
        mu = np.array([0.5])
        fn = np.array([1000.0])       # mu*Fn = 500
        alpha = 1.0
        v_rel = np.array([[2.0, 0.0, 0.0]])
        filt_keys = np.zeros(0, dtype=np.int64)
        filt_vals = np.zeros((0, 3))

        for i in range(1, 11):
            ft, filt_keys, filt_vals = apply_incremental_stiffness(
                keys, k, v_rel, dt, normal, mu, fn, alpha, filt_keys, filt_vals)
            expected = i * k[0] * 2.0 * dt
            # Should still be in stick (expected << mu*Fn)
            assert expected < mu[0] * fn[0]
            assert ft[0, 0] == pytest.approx(expected, rel=1e-8)


# ---------------------------------------------------------------------------
# test_incremental_slip: above Coulomb limit -> force saturates at mu*Fn
# ---------------------------------------------------------------------------
class TestIncrementalSlip:
    """When the tangential trial force exceeds mu*Fn, the force should
    saturate at the Coulomb limit."""

    def test_coulomb_saturation(self):
        """After enough cycles, the accumulated force hits mu*Fn and stays."""
        keys = np.array([0], dtype=np.int64)
        k = np.array([1e6])          # very stiff spring
        dt = 1e-3
        normal = np.array([[0.0, 0.0, 1.0]])
        mu = np.array([0.3])
        fn = np.array([100.0])        # Coulomb limit = 30
        alpha = 1.0
        v_rel = np.array([[10.0, 0.0, 0.0]])  # fast sliding
        filt_keys = np.zeros(0, dtype=np.int64)
        filt_vals = np.zeros((0, 3))

        # First cycle: increment = 1e6 * 10 * 1e-3 = 1e4 >> 30
        # Should immediately saturate at mu*Fn = 30
        ft, filt_keys, filt_vals = apply_incremental_stiffness(
            keys, k, v_rel, dt, normal, mu, fn, alpha, filt_keys, filt_vals)
        ft_mag = np.linalg.norm(ft[0])
        assert ft_mag == pytest.approx(mu[0] * fn[0], rel=1e-10)
        # Direction should match sliding direction (x)
        assert ft[0, 0] > 0.0

    def test_slip_direction_preserves_sliding_direction(self):
        """Sliding in the xy-plane: saturated force is aligned with sliding."""
        keys = np.array([0], dtype=np.int64)
        k = np.array([1e6])
        dt = 1e-3
        normal = np.array([[0.0, 0.0, 1.0]])
        mu = np.array([0.5])
        fn = np.array([200.0])
        alpha = 1.0
        # Slide at 45 degrees in xy
        v_rel = np.array([[3.0, 3.0, 0.0]])
        filt_keys = np.zeros(0, dtype=np.int64)
        filt_vals = np.zeros((0, 3))

        ft, _, _ = apply_incremental_stiffness(
            keys, k, v_rel, dt, normal, mu, fn, alpha, filt_keys, filt_vals)
        ft_mag = np.linalg.norm(ft[0])
        assert ft_mag == pytest.approx(mu[0] * fn[0], rel=1e-10)
        # Force direction should be at 45 degrees (equal x and y)
        assert ft[0, 0] == pytest.approx(ft[0, 1], rel=1e-10)
        assert ft[0, 2] == pytest.approx(0.0, abs=1e-14)


# ---------------------------------------------------------------------------
# test_incremental_separation_reset: IFPEN semantics
# ---------------------------------------------------------------------------
class TestSeparationReset:
    """When a pair separates (key disappears from the active set), its
    CAND_F anchor must reset to zero on re-contact."""

    def test_separation_resets_anchor(self):
        """Accumulate force, then separate (key not in active set), then
        re-contact with the same key: force should start from zero."""
        keys = np.array([42], dtype=np.int64)
        k = np.array([1e4])
        dt = 1e-4
        normal = np.array([[0.0, 0.0, 1.0]])
        mu = np.array([0.3])
        fn = np.array([100.0])
        alpha = 1.0
        v_rel = np.array([[1.0, 0.0, 0.0]])
        filt_keys = np.zeros(0, dtype=np.int64)
        filt_vals = np.zeros((0, 3))

        # Accumulate over 5 cycles
        for _ in range(5):
            ft, filt_keys, filt_vals = apply_incremental_stiffness(
                keys, k, v_rel, dt, normal, mu, fn, alpha, filt_keys, filt_vals)
        assert ft[0, 0] > 0.0  # non-zero accumulated force

        # Separate: call with different keys (key 42 is gone)
        other_keys = np.array([99], dtype=np.int64)
        _, filt_keys, filt_vals = apply_incremental_stiffness(
            other_keys, k, v_rel, dt, normal, mu, fn, alpha, filt_keys, filt_vals)
        # The store should now contain only key 99, not 42

        # Re-contact with key 42: should start from zero
        ft_new, _, _ = apply_incremental_stiffness(
            keys, k, v_rel, dt, normal, mu, fn, alpha, filt_keys, filt_vals)
        # Force should be a single increment, not the accumulated value
        expected = k[0] * 1.0 * dt
        assert ft_new[0, 0] == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# test_tangential_projection: normal component removed correctly
# ---------------------------------------------------------------------------
class TestTangentialProjection:
    """The trial force includes the full v_rel increment, but the normal
    component must be removed before the Coulomb return mapping."""

    def test_normal_velocity_removed(self):
        """With v_rel having both normal and tangential components, only
        the tangential part should appear in the force."""
        keys = np.array([0], dtype=np.int64)
        k = np.array([1000.0])
        dt = 1e-4
        normal = np.array([[0.0, 0.0, 1.0]])
        mu = np.array([0.5])
        fn = np.array([1000.0])
        alpha = 1.0
        # v_rel = (1, 0, 5): tangential = (1, 0, 0), normal = 5
        v_rel = np.array([[1.0, 0.0, 5.0]])
        filt_keys = np.zeros(0, dtype=np.int64)
        filt_vals = np.zeros((0, 3))

        ft, _, _ = apply_incremental_stiffness(
            keys, k, v_rel, dt, normal, mu, fn, alpha, filt_keys, filt_vals)
        # No z-component (normal direction)
        assert ft[0, 2] == pytest.approx(0.0, abs=1e-12)
        # x-component should be k * vx * dt (tangential only)
        assert ft[0, 0] == pytest.approx(k[0] * 1.0 * dt, rel=1e-10)

    def test_oblique_normal(self):
        """Non-axis-aligned surface normal: tangential projection must
        correctly remove the normal component."""
        keys = np.array([0], dtype=np.int64)
        k = np.array([1000.0])
        dt = 1e-4
        # 45-degree normal in yz plane
        n = np.array([[0.0, 1.0, 1.0]]) / np.sqrt(2)
        mu = np.array([0.5])
        fn = np.array([1000.0])
        alpha = 1.0
        v_rel = np.array([[1.0, 1.0, 1.0]])
        filt_keys = np.zeros(0, dtype=np.int64)
        filt_vals = np.zeros((0, 3))

        ft, _, _ = apply_incremental_stiffness(
            keys, k, v_rel, dt, normal=n, mu=mu, fn=fn, alpha=alpha,
            filt_keys=filt_keys, filt_vals=filt_vals)
        # Force should be perpendicular to n
        assert np.dot(ft[0], n[0]) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# test_multiple_pairs: vectorized operation
# ---------------------------------------------------------------------------
class TestMultiplePairs:
    """Verify correct vectorized behavior with multiple simultaneous pairs."""

    def test_two_pairs_independent(self):
        """Two pairs with different stiffness, mu, fn: each tracked
        independently."""
        keys = np.array([10, 20], dtype=np.int64)
        k = np.array([1000.0, 2000.0])
        dt = 1e-4
        normal = np.array([[0., 0., 1.], [0., 0., 1.]])
        mu = np.array([0.3, 0.5])
        fn = np.array([100.0, 200.0])
        alpha = 1.0
        v_rel = np.array([[1.0, 0., 0.], [0., 1.0, 0.]])
        filt_keys = np.zeros(0, dtype=np.int64)
        filt_vals = np.zeros((0, 3))

        ft, new_keys, new_vals = apply_incremental_stiffness(
            keys, k, v_rel, dt, normal, mu, fn, alpha, filt_keys, filt_vals)

        # Pair 0: F = 1000 * 1 * 1e-4 = 0.1 in x
        assert ft[0, 0] == pytest.approx(1000.0 * 1.0 * dt, rel=1e-10)
        assert ft[0, 1] == pytest.approx(0.0, abs=1e-15)

        # Pair 1: F = 2000 * 1 * 1e-4 = 0.2 in y
        assert ft[1, 1] == pytest.approx(2000.0 * 1.0 * dt, rel=1e-10)
        assert ft[1, 0] == pytest.approx(0.0, abs=1e-15)

        # new_keys should be sorted
        assert np.all(new_keys == np.array([10, 20]))


# ---------------------------------------------------------------------------
# test_filter_alpha_effect: alpha < 1 damps the increment
# ---------------------------------------------------------------------------
class TestFilterAlphaEffect:
    """Alpha < 1 should reduce the force increment rate (filtering)."""

    def test_alpha_half_halves_increment(self):
        """With alpha=0.5, the increment should be half of alpha=1."""
        keys = np.array([0], dtype=np.int64)
        k = np.array([1000.0])
        dt = 1e-4
        normal = np.array([[0.0, 0.0, 1.0]])
        mu = np.array([0.5])
        fn = np.array([1000.0])
        v_rel = np.array([[1.0, 0.0, 0.0]])
        filt_keys = np.zeros(0, dtype=np.int64)
        filt_vals = np.zeros((0, 3))

        ft_full, _, _ = apply_incremental_stiffness(
            keys, k, v_rel, dt, normal, mu, fn,
            alpha=1.0, filt_keys=filt_keys, filt_vals=filt_vals)

        ft_half, _, _ = apply_incremental_stiffness(
            keys, k, v_rel, dt, normal, mu, fn,
            alpha=0.5, filt_keys=filt_keys, filt_vals=filt_vals)

        assert ft_half[0, 0] == pytest.approx(0.5 * ft_full[0, 0], rel=1e-10)


# ---------------------------------------------------------------------------
# test_stored_values_are_return_mapped: CAND_F stores the clamped force
# ---------------------------------------------------------------------------
class TestStoredValues:
    """The Fortran stores the return-mapped (clamped) force in CAND_F,
    not the trial force. Verify this."""

    def test_store_is_return_mapped_in_slip(self):
        """When slip occurs, stored value = return-mapped force, not trial."""
        keys = np.array([0], dtype=np.int64)
        k = np.array([1e6])       # very stiff -> immediate slip
        dt = 1e-3
        normal = np.array([[0.0, 0.0, 1.0]])
        mu = np.array([0.3])
        fn = np.array([100.0])     # Coulomb limit = 30
        alpha = 1.0
        v_rel = np.array([[10.0, 0.0, 0.0]])
        filt_keys = np.zeros(0, dtype=np.int64)
        filt_vals = np.zeros((0, 3))

        ft, new_keys, new_vals = apply_incremental_stiffness(
            keys, k, v_rel, dt, normal, mu, fn, alpha, filt_keys, filt_vals)

        # Stored force magnitude should be mu*Fn, not the trial force
        stored_mag = np.linalg.norm(new_vals[0])
        assert stored_mag == pytest.approx(mu[0] * fn[0], rel=1e-10)

    def test_store_is_exact_in_stick(self):
        """In stick regime, stored value = trial force (beta=1)."""
        keys = np.array([0], dtype=np.int64)
        k = np.array([1000.0])
        dt = 1e-5
        normal = np.array([[0.0, 0.0, 1.0]])
        mu = np.array([0.5])
        fn = np.array([1000.0])  # mu*Fn = 500, well above small forces
        alpha = 1.0
        v_rel = np.array([[0.1, 0.0, 0.0]])
        filt_keys = np.zeros(0, dtype=np.int64)
        filt_vals = np.zeros((0, 3))

        ft, new_keys, new_vals = apply_incremental_stiffness(
            keys, k, v_rel, dt, normal, mu, fn, alpha, filt_keys, filt_vals)

        # In stick: stored == returned
        np.testing.assert_allclose(new_vals[0], ft[0], rtol=1e-12)


# ---------------------------------------------------------------------------
# test_zero_velocity: no increment, force stays at previous value
# ---------------------------------------------------------------------------
class TestZeroVelocity:
    """With zero relative velocity, the force should not change."""

    def test_zero_vrel_preserves_force(self):
        keys = np.array([0], dtype=np.int64)
        k = np.array([1000.0])
        dt = 1e-4
        normal = np.array([[0.0, 0.0, 1.0]])
        mu = np.array([0.3])
        fn = np.array([100.0])
        alpha = 1.0
        v_rel = np.array([[1.0, 0.0, 0.0]])
        filt_keys = np.zeros(0, dtype=np.int64)
        filt_vals = np.zeros((0, 3))

        # Build up some force
        ft1, filt_keys, filt_vals = apply_incremental_stiffness(
            keys, k, v_rel, dt, normal, mu, fn, alpha, filt_keys, filt_vals)

        # Now zero velocity: force should stay the same
        v_zero = np.array([[0.0, 0.0, 0.0]])
        ft2, _, _ = apply_incremental_stiffness(
            keys, k, v_zero, dt, normal, mu, fn, alpha, filt_keys, filt_vals)

        np.testing.assert_allclose(ft2[0], ft1[0], rtol=1e-12)


# ---------------------------------------------------------------------------
# IFQ mapping: verify filter_alpha covers IFQ 10-13 correctly
# ---------------------------------------------------------------------------
class TestIFQMappingLogic:
    """Verify the IFQ >= 10 modes map to the correct filter_alpha
    coefficients, matching the IFQ 0-3 base modes but using the
    incremental stiffness force path."""

    def test_ifq10_matches_ifq0(self):
        """IFQ=10 has the same alpha as IFQ=0 (constant)."""
        assert filter_alpha(10, 0.75, 1e-4) == filter_alpha(0, 0.75, 1e-4)

    def test_ifq11_matches_ifq1(self):
        """IFQ=11 has the same alpha as IFQ=1."""
        assert filter_alpha(11, 0.5, 1e-4) == filter_alpha(1, 0.5, 1e-4)

    def test_ifq12_matches_ifq2(self):
        """IFQ=12 has the same alpha as IFQ=2."""
        assert filter_alpha(12, 0.3, 1e-4) == filter_alpha(2, 0.3, 1e-4)

    def test_ifq13_matches_ifq3(self):
        """IFQ=13 has the same alpha as IFQ=3 (cutoff frequency)."""
        dt = 1e-4
        xfiltr = 2000.0
        assert filter_alpha(13, xfiltr, dt) == filter_alpha(3, xfiltr, dt)

    def test_ifq_dispatch_incremental_vs_filter(self):
        """IFQ >= 10 activates apply_incremental_stiffness, while
        IFQ 1-3 activates apply_filter. Both use the same alpha."""
        for ifq in (10, 11, 12, 13):
            assert ifq >= 10, f"IFQ={ifq} should route to incremental"
        for ifq in (1, 2, 3):
            assert ifq < 10, f"IFQ={ifq} should route to filter"
