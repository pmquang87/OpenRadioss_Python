"""
M478 — Unit tests for /DAMP Rayleigh mass damping (engine/damping.py).

The ``Dampers.apply()`` method implements the exact integrating factor:

    v <- v * exp(-alpha * dt)

with dissipated energy booked from the kinetic-energy identity:

    E_damp = sum 0.5 * m * (|v_before|^2 - |v_after|^2)

Fortran origin: ``engine/source/assembly/damping.F`` (damping51 subroutine,
lines 100–170 for the mass-proportional branch in global coordinates).

The Python port deliberately uses the exact ODE solution (integrating factor)
instead of the Fortran's implicit-trapezoidal acceleration correction, yielding
unconditional stability and exact energy booking by construction.

Tests bypass ``Dampers.__init__`` (which needs a full Model with node groups)
and directly set ``.items`` to test the physics of ``apply()`` in isolation.
"""

import numpy as np
import pytest

from pyradioss.engine.damping import Dampers


def _make_dampers(items):
    """Create a Dampers object with pre-set items, bypassing __init__."""
    d = object.__new__(Dampers)
    d.items = items
    all_idx = np.concatenate([it[0] for it in items]) if items else np.array([], dtype=np.int64)
    d.all_idx = np.unique(all_idx)
    return d


# ======================================================================
# Exact integrating factor:  v <- v * exp(-alpha * dt)
# ======================================================================
class TestExactFactor:
    """Verify the velocity decay formula."""

    def test_1d_single_node(self):
        """A single node with 1D velocity decays by exp(-alpha*dt)."""
        alpha, dt = 100.0, 1e-3
        idx = np.array([0], dtype=np.int64)
        d = _make_dampers([(idx, alpha, 0.0, 1e30)])

        v = np.array([[5.0, 0.0, 0.0]])
        vr = np.zeros((1, 3))
        mass = np.array([2.0])
        inertia = np.array([0.0])

        d.apply(t=0.0, dt=dt, v=v, vr=vr, mass=mass, inertia=inertia)

        expected = 5.0 * np.exp(-alpha * dt)
        np.testing.assert_allclose(v[0, 0], expected, rtol=1e-14)
        assert v[0, 1] == 0.0
        assert v[0, 2] == 0.0

    def test_3d_velocity(self):
        """All three components decay by the same factor."""
        alpha, dt = 200.0, 5e-4
        idx = np.array([0], dtype=np.int64)
        d = _make_dampers([(idx, alpha, 0.0, 1e30)])

        v0 = np.array([[1.0, 2.0, 3.0]])
        v = v0.copy()
        vr = np.zeros((1, 3))
        mass = np.array([1.0])
        inertia = np.array([0.0])

        d.apply(t=0.001, dt=dt, v=v, vr=vr, mass=mass, inertia=inertia)

        fac = np.exp(-alpha * dt)
        np.testing.assert_allclose(v[0], v0[0] * fac, rtol=1e-14)

    def test_multi_node(self):
        """Multiple nodes in the group all decay correctly."""
        alpha, dt = 50.0, 2e-3
        idx = np.array([0, 1, 2], dtype=np.int64)
        d = _make_dampers([(idx, alpha, 0.0, 1e30)])

        v0 = np.array([[1.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 3.0]])
        v = v0.copy()
        vr = np.zeros((3, 3))
        mass = np.array([1.0, 2.0, 3.0])
        inertia = np.zeros(3)

        d.apply(t=0.0, dt=dt, v=v, vr=vr, mass=mass, inertia=inertia)

        fac = np.exp(-alpha * dt)
        np.testing.assert_allclose(v, v0 * fac, rtol=1e-14)


# ======================================================================
# Energy identity:  de = sum 0.5 * m * (|v_before|^2 - |v_after|^2)
# ======================================================================
class TestEnergyIdentity:
    """The returned dissipation must match the KE drop exactly."""

    def test_translation_only(self):
        """KE identity for translational DOFs with no rotation."""
        alpha, dt = 100.0, 1e-3
        idx = np.array([0, 1], dtype=np.int64)
        d = _make_dampers([(idx, alpha, 0.0, 1e30)])

        v = np.array([[3.0, 4.0, 0.0],
                       [0.0, 0.0, 5.0]])
        vr = np.zeros((2, 3))
        mass = np.array([2.0, 3.0])
        inertia = np.zeros(2)

        ke_before = 0.5 * (mass[:, None] * v ** 2).sum()
        de = d.apply(t=0.0, dt=dt, v=v, vr=vr, mass=mass, inertia=inertia)
        ke_after = 0.5 * (mass[:, None] * v ** 2).sum()

        np.testing.assert_allclose(de, ke_before - ke_after, rtol=1e-13)
        assert de > 0.0  # damping always removes energy

    def test_rotation_only(self):
        """KE identity for rotational DOFs."""
        alpha, dt = 150.0, 1e-3
        idx = np.array([0], dtype=np.int64)
        d = _make_dampers([(idx, alpha, 0.0, 1e30)])

        v = np.array([[0.0, 0.0, 0.0]])       # no translation
        vr = np.array([[10.0, 20.0, 30.0]])    # spinning
        mass = np.array([1.0])
        inertia = np.array([0.5])

        ke_before_rot = 0.5 * (inertia[:, None] * vr ** 2).sum()
        de = d.apply(t=0.0, dt=dt, v=v, vr=vr, mass=mass, inertia=inertia)
        ke_after_rot = 0.5 * (inertia[:, None] * vr ** 2).sum()

        np.testing.assert_allclose(de, ke_before_rot - ke_after_rot, rtol=1e-13)

    def test_combined_translation_rotation(self):
        """KE identity when both translation and rotation are present."""
        alpha, dt = 80.0, 2e-3
        idx = np.array([0], dtype=np.int64)
        d = _make_dampers([(idx, alpha, 0.0, 1e30)])

        v = np.array([[5.0, 0.0, 0.0]])
        vr = np.array([[0.0, 10.0, 0.0]])
        mass = np.array([3.0])
        inertia = np.array([1.5])

        ke_t_before = 0.5 * (mass[:, None] * v ** 2).sum()
        ke_r_before = 0.5 * (inertia[:, None] * vr ** 2).sum()
        ke_before = ke_t_before + ke_r_before

        de = d.apply(t=0.0, dt=dt, v=v, vr=vr, mass=mass, inertia=inertia)

        ke_t_after = 0.5 * (mass[:, None] * v ** 2).sum()
        ke_r_after = 0.5 * (inertia[:, None] * vr ** 2).sum()
        ke_after = ke_t_after + ke_r_after

        np.testing.assert_allclose(de, ke_before - ke_after, rtol=1e-13)


# ======================================================================
# Time windowing:  Tstart / Tstop
# ======================================================================
class TestTimeWindow:
    """Damping is only active when tstart <= t <= tstop."""

    def test_before_tstart(self):
        """No damping when t < tstart."""
        idx = np.array([0], dtype=np.int64)
        d = _make_dampers([(idx, 100.0, 0.01, 1e30)])  # tstart=0.01

        v = np.array([[5.0, 0.0, 0.0]])
        vr = np.zeros((1, 3))
        mass = np.array([1.0])
        inertia = np.zeros(1)

        de = d.apply(t=0.005, dt=1e-3, v=v, vr=vr, mass=mass, inertia=inertia)

        assert de == 0.0
        assert v[0, 0] == 5.0  # unchanged

    def test_after_tstop(self):
        """No damping when t > tstop."""
        idx = np.array([0], dtype=np.int64)
        d = _make_dampers([(idx, 100.0, 0.0, 0.005)])  # tstop=0.005

        v = np.array([[5.0, 0.0, 0.0]])
        vr = np.zeros((1, 3))
        mass = np.array([1.0])
        inertia = np.zeros(1)

        de = d.apply(t=0.006, dt=1e-3, v=v, vr=vr, mass=mass, inertia=inertia)

        assert de == 0.0
        assert v[0, 0] == 5.0  # unchanged

    def test_within_window(self):
        """Damping active when tstart <= t <= tstop."""
        idx = np.array([0], dtype=np.int64)
        d = _make_dampers([(idx, 100.0, 0.001, 0.010)])

        v = np.array([[5.0, 0.0, 0.0]])
        vr = np.zeros((1, 3))
        mass = np.array([1.0])
        inertia = np.zeros(1)

        de = d.apply(t=0.005, dt=1e-3, v=v, vr=vr, mass=mass, inertia=inertia)

        assert de > 0.0
        assert v[0, 0] < 5.0  # velocity reduced


# ======================================================================
# Rotational damping
# ======================================================================
class TestRotationalDamping:
    """Rotational DOFs are damped only when inertia > 0."""

    def test_inertia_zero_skips_rotation(self):
        """Nodes with inertia=0 have their vr left untouched."""
        idx = np.array([0], dtype=np.int64)
        d = _make_dampers([(idx, 100.0, 0.0, 1e30)])

        v = np.array([[1.0, 0.0, 0.0]])
        vr0 = np.array([[10.0, 20.0, 30.0]])
        vr = vr0.copy()
        mass = np.array([1.0])
        inertia = np.array([0.0])  # no inertia

        d.apply(t=0.0, dt=1e-3, v=v, vr=vr, mass=mass, inertia=inertia)

        # translation damped, rotation untouched
        assert v[0, 0] < 1.0
        np.testing.assert_array_equal(vr, vr0)

    def test_inertia_positive_damps_rotation(self):
        """Nodes with inertia > 0 have their vr damped."""
        alpha, dt = 100.0, 1e-3
        idx = np.array([0], dtype=np.int64)
        d = _make_dampers([(idx, alpha, 0.0, 1e30)])

        v = np.zeros((1, 3))
        vr0 = np.array([[10.0, 0.0, 0.0]])
        vr = vr0.copy()
        mass = np.array([1.0])
        inertia = np.array([2.0])

        d.apply(t=0.0, dt=dt, v=v, vr=vr, mass=mass, inertia=inertia)

        fac = np.exp(-alpha * dt)
        np.testing.assert_allclose(vr[0, 0], vr0[0, 0] * fac, rtol=1e-14)


# ======================================================================
# Multiple dampers
# ======================================================================
class TestMultipleDampers:
    """Two dampers on overlapping node sets apply sequentially."""

    def test_two_dampers_different_alpha(self):
        """Two dampers with different alpha compound their effect."""
        alpha1, alpha2, dt = 50.0, 100.0, 1e-3
        idx1 = np.array([0, 1], dtype=np.int64)
        idx2 = np.array([1, 2], dtype=np.int64)
        d = _make_dampers([
            (idx1, alpha1, 0.0, 1e30),
            (idx2, alpha2, 0.0, 1e30),
        ])

        v = np.array([[1.0, 0.0, 0.0],
                       [2.0, 0.0, 0.0],
                       [3.0, 0.0, 0.0]])
        vr = np.zeros((3, 3))
        mass = np.ones(3)
        inertia = np.zeros(3)

        d.apply(t=0.0, dt=dt, v=v, vr=vr, mass=mass, inertia=inertia)

        fac1 = np.exp(-alpha1 * dt)
        fac2 = np.exp(-alpha2 * dt)

        # node 0: only damper 1
        np.testing.assert_allclose(v[0, 0], 1.0 * fac1, rtol=1e-14)
        # node 1: both dampers (sequential: first fac1, then fac2)
        np.testing.assert_allclose(v[1, 0], 2.0 * fac1 * fac2, rtol=1e-14)
        # node 2: only damper 2
        np.testing.assert_allclose(v[2, 0], 3.0 * fac2, rtol=1e-14)


# ======================================================================
# High alpha — unconditional stability
# ======================================================================
class TestHighAlpha:
    """The exact factor is stable for any alpha*dt."""

    def test_large_alpha_dt(self):
        """Even with alpha*dt = 100 (wildly large), velocity goes to ~0
        without oscillation or blowup."""
        alpha, dt = 1e5, 1e-3   # alpha*dt = 100
        idx = np.array([0], dtype=np.int64)
        d = _make_dampers([(idx, alpha, 0.0, 1e30)])

        v = np.array([[1e6, -2e6, 3e6]])
        vr = np.zeros((1, 3))
        mass = np.array([1.0])
        inertia = np.zeros(1)

        de = d.apply(t=0.0, dt=dt, v=v, vr=vr, mass=mass, inertia=inertia)

        # velocity should be essentially zero
        assert np.all(np.abs(v[0]) < 1e-30)
        # energy removed should be positive
        assert de > 0.0


# ======================================================================
# Edge cases
# ======================================================================
class TestEdgeCases:
    """Boundary conditions and degenerate inputs."""

    def test_zero_dt(self):
        """dt=0 means no damping (skip)."""
        idx = np.array([0], dtype=np.int64)
        d = _make_dampers([(idx, 100.0, 0.0, 1e30)])

        v = np.array([[5.0, 0.0, 0.0]])
        vr = np.zeros((1, 3))
        mass = np.array([1.0])
        inertia = np.zeros(1)

        de = d.apply(t=0.0, dt=0.0, v=v, vr=vr, mass=mass, inertia=inertia)

        assert de == 0.0
        assert v[0, 0] == 5.0

    def test_zero_alpha(self):
        """alpha=0 means no damping (exp(0)=1)."""
        idx = np.array([0], dtype=np.int64)
        d = _make_dampers([(idx, 0.0, 0.0, 1e30)])

        v = np.array([[5.0, 3.0, 1.0]])
        v0 = v.copy()
        vr = np.zeros((1, 3))
        mass = np.array([1.0])
        inertia = np.zeros(1)

        de = d.apply(t=0.0, dt=1e-3, v=v, vr=vr, mass=mass, inertia=inertia)

        np.testing.assert_allclose(de, 0.0, atol=1e-30)
        np.testing.assert_array_equal(v, v0)

    def test_empty_items(self):
        """No dampers at all — returns 0, touches nothing."""
        d = _make_dampers([])

        v = np.array([[5.0, 0.0, 0.0]])
        vr = np.zeros((1, 3))
        mass = np.array([1.0])
        inertia = np.zeros(1)

        de = d.apply(t=0.0, dt=1e-3, v=v, vr=vr, mass=mass, inertia=inertia)

        assert de == 0.0
        assert v[0, 0] == 5.0

    def test_multi_cycle_accumulation(self):
        """Apply damping N times — velocity should equal v0 * exp(-alpha * N * dt)."""
        alpha, dt, N = 100.0, 1e-3, 10
        idx = np.array([0], dtype=np.int64)
        d = _make_dampers([(idx, alpha, 0.0, 1e30)])

        v = np.array([[10.0, 0.0, 0.0]])
        vr = np.zeros((1, 3))
        mass = np.array([1.0])
        inertia = np.zeros(1)

        total_de = 0.0
        for step in range(N):
            total_de += d.apply(t=step * dt, dt=dt, v=v, vr=vr,
                                mass=mass, inertia=inertia)

        expected_v = 10.0 * np.exp(-alpha * N * dt)
        np.testing.assert_allclose(v[0, 0], expected_v, rtol=1e-12)

        # total KE removed = initial KE - final KE
        ke_init = 0.5 * 1.0 * 10.0 ** 2
        ke_final = 0.5 * 1.0 * v[0, 0] ** 2
        np.testing.assert_allclose(total_de, ke_init - ke_final, rtol=1e-12)
