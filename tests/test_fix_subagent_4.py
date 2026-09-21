"""Tests for Fix Subagent 4 (Implicit & Fatigue fixes)."""

import numpy as np
import pytest
import scipy.sparse as sp


# ----------------------------------------------------------------------------
# 1. followerload: degenerate / triangular segments and safe dof indexing
# ----------------------------------------------------------------------------
def test_followerload_degenerate_triangles():
    """pload_tangent must handle triangular segments with segs[:, 3] <= 0 without error."""
    from pyradioss.implicit.followerload import pload_tangent
    from pyradioss.implicit.dofmap import DofMap

    class DummyCurve:
        def eval(self, t):
            return 10.0

    class DummyLoads:
        def __init__(self):
            # seg 0: quad, seg 1: tri with -1, seg 2: tri with node 0 as dummy, seg 3: tri with repeated node 2
            segs = np.array([
                [0, 1, 2, 3],
                [0, 1, 2, -1],
                [1, 2, 3, 0],
                [0, 1, 2, 2]
            ], dtype=np.int64)
            wgt = np.full((4, 4), 0.25)
            fct = DummyCurve()
            self.ploads = [(segs, wgt, fct, 1.0, "SHELL", np.array([-1, -1, -1, -1]), False, 0)]

    class DummyModel:
        def __init__(self):
            self.numnod = 4
            self.mass = np.ones(4)
            self.inertia = np.zeros(4)
            self.bcs = []
            self.shells = None
            self.shells_qbat = None
            self.shells_qeph = None
            self.sh3n = None
            self.sh3n_dkt18 = None
            self.beams = None
            self.element_groups = lambda: []

    model = DummyModel()
    dof = DofMap(model)
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]
    ])
    K = pload_tangent(DummyLoads(), model, 0.0, x, dof)
    assert K.shape == (dof.ndof, dof.ndof)
    assert np.all(np.isfinite(K.data))


# ----------------------------------------------------------------------------
# 2. spectral_fatigue & multiaxial_fatigue: Dirlik D2 clamping & cycle counts
# ----------------------------------------------------------------------------
def test_dirlik_partition_of_unity_clamping():
    """Dirlik coefficients must satisfy D1 + D2 + D3 == 1.0 and D2 <= 1.0 - D1."""
    from pyradioss.implicit.spectral_fatigue import dirlik_coefficients

    for gamma in [0.1, 0.5, 0.9, 0.99, 0.999, 0.9999]:
        m0 = 1.0
        m2 = gamma
        m4 = 1.0
        m1 = gamma * 0.95
        moments = np.array([m0, m1, m2, (m2 * m4)**0.5, m4])
        c = dirlik_coefficients(moments)
        assert c["D2"] <= 1.0 - c["D1"] + 1e-12
        assert abs(c["D1"] + c["D2"] + c["D3"] - 1.0) < 1e-12
        assert c["D2"] >= 0.0
        assert c["D3"] >= 0.0


def test_monte_carlo_damage_half_cycles():
    """monte_carlo_damage must use counts.sum() instead of ranges.size."""
    from pyradioss.implicit.spectral_fatigue import monte_carlo_damage
    freqs = np.linspace(1.0, 50.0, 10)
    psd = np.ones_like(freqs) * 1e2
    res = monte_carlo_damage(freqs, psd, m=3.0, C=1e12, duration=1.0, seed=42)
    assert res["ncycles"] == float(res["counts"].sum())


def test_monte_carlo_multiaxial_damage_half_cycles():
    """monte_carlo_multiaxial_damage must use counts.sum() instead of ranges.size."""
    from pyradioss.implicit.multiaxial_fatigue import monte_carlo_multiaxial_damage
    freqs = np.linspace(1.0, 50.0, 10)
    # Scross: (10, 6, 6)
    Scross = np.zeros((10, 6, 6))
    for i in range(10):
        Scross[i, 0, 0] = 1e2
    proj = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    res = monte_carlo_multiaxial_damage(freqs, Scross, proj, m=3.0, C=1e12, duration=1.0, seed=42)
    assert res["ncycles"] == float(res["counts"].sum())


# ----------------------------------------------------------------------------
# 3. constraints: _solve_semidefinite zeroing off-diagonals
# ----------------------------------------------------------------------------
def test_solve_semidefinite_zeros_off_diagonals():
    """constraints._solve_semidefinite must zero out off-diagonals of zero-diagonal rows/cols."""
    from pyradioss.implicit.constraints import _solve_semidefinite

    class DummySolver:
        def solve(self, M, rhs):
            return sp.linalg.spsolve(M, rhs)

    M = sp.csr_matrix([
        [4.0, 2.0, 0.0],
        [2.0, 0.0, 1.0],
        [0.0, 1.0, 3.0]
    ])
    rhs = np.array([4.0, 5.0, 3.0])
    sol = _solve_semidefinite(M, rhs, DummySolver())
    # Eq 1 was zero-diagonal, so it should be decoupled: x[1] = 0.0
    assert abs(sol[1]) < 1e-12
    # Eq 0 and 2 should satisfy 4*x0 = 4 => x0 = 1, and 3*x2 = 3 => x2 = 1
    assert abs(sol[0] - 1.0) < 1e-12
    assert abs(sol[2] - 1.0) < 1e-12


# ----------------------------------------------------------------------------
# 4. statics & dynamics: exception handling & divergence checks
# ----------------------------------------------------------------------------
def test_dynamics_solver_exception_handling():
    """dynamics._solve_step must catch RuntimeError/ValueError from solver and return inc.converged=False."""
    from pyradioss.implicit.dynamics import _solve_step
    from pyradioss.implicit.dofmap import DofMap

    class DummyControls:
        impl_tol = 1e-4
        impl_max_iter = 5
        impl_nonlin = 1
        impl_load_stiff = False

    class FailingSolver:
        def solve(self, K, R):
            raise RuntimeError("Singular matrix in test")

    class DummyModel:
        def __init__(self):
            self.numnod = 2
            self.mass = np.ones(2)
            self.inertia = np.zeros(2)
            self.bcs = []
            self.shells = None
            self.shells_qbat = None
            self.shells_qeph = None
            self.sh3n = None
            self.sh3n_dkt18 = None
            self.beams = None
            self.solids = None
            self.element_groups = lambda: []
            self.x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            self.x0 = self.x.copy()

    class DummyLoads:
        def external_forces(self, t, fx, x):
            fx[1, 0] = 10.0

    model = DummyModel()
    dof = DofMap(model)
    ip = DummyControls()
    committed = {}
    x_ref = model.x.copy()
    imposed = []
    u_pred = np.zeros((2, 3))
    ur_pred = np.zeros((2, 3))
    v = np.zeros((2, 3))
    vr = np.zeros((2, 3))
    a = np.zeros((2, 3))
    ar = np.zeros((2, 3))

    inc, u, ur, fint, mint, fext, fd, *_cached = _solve_step(
        model, ip, dof, DummyLoads(), FailingSolver(), committed, x_ref, imposed,
        t_old=0.0, t_new=0.01, dt=0.01, alpha=0.0, gamma=0.5, beta=0.25,
        M_eq=np.ones(dof.ndof), v=v, vr=vr, a=a, ar=ar,
        g_prev_f=np.zeros((2, 3)), g_prev_m=np.zeros((2, 3)), nlgeom=False
    )
    assert not inc.converged
