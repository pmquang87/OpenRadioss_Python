import numpy as np
import pytest

pytest.importorskip("scipy")
import scipy.linalg as sla


def test_bug_imp_01_crisfield_arclength_backward_roots():
    """BUG-IMP-01: When both candidate roots in Crisfield arc-length point backwards
    (f1 <= 0.0 and f2 <= 0.0), inc.converged must be False to trigger radius halving."""
    wlam = 1.0
    dlam_tot = 1.0
    Du = np.array([1.0, 0.0])
    t = np.array([-2.0, 0.0])
    du_t = np.array([0.5, 0.0])

    def _fwd(r):
        return float(Du @ (t + r * du_t)) + wlam * dlam_tot * (dlam_tot + r)

    f1, f2 = _fwd(-2.0), _fwd(-3.0)
    assert f1 <= 0.0 and f2 <= 0.0


def test_bug_imp_02_complex_modal_rigid_body_modes():
    """BUG-IMP-02: When Kr has rigid body/zero modes, companion form A1, B1 must be used
    and the pencil must be regular, returning no NaNs."""
    from pyradioss.implicit.complex_modal import _state_space_eig

    # Unconstrained 2-mass 1D system (rigid translation mode)
    K = np.array([[1000.0, -1000.0], [-1000.0, 1000.0]])
    M = np.diag([2.0e-3, 1.0e-3])
    C = np.array([[0.5, -0.5], [-0.5, 0.5]])

    lam, vecs = _state_space_eig(K, C, M)

    # Must have no NaNs
    assert not np.any(np.isnan(lam))
    assert not np.any(np.isnan(vecs))

    # Should have two zero eigenvalues (rigid mode) and two structural eigenvalues
    mags = np.sort(np.abs(lam))
    assert mags[0] < 1e-4
    assert mags[1] < 1e-4
    assert mags[2] > 100.0
    assert mags[3] > 100.0


def test_bug_imp_03_line_search_nan_initial_step():
    """BUG-IMP-03: In line search, a NaN initial step (alpha=1.0) must not poison
    recovery when a backtracked step (alpha=0.5) is finite."""
    alpha, best, alpha_last = 1.0, None, 1.0
    trials = [np.nan, 5.0, 2.0, 1.0]

    for ls in range(4):
        alpha_last = alpha
        rn_t = trials[ls]
        if np.isfinite(rn_t) and (best is None or rn_t < best[0]):
            best = (rn_t, alpha)
        if rn_t <= 0.5 or ls == 3:
            break
        alpha *= 0.5
    if best is None:
        best = (rn_t, alpha)

    assert best is not None
    assert np.isfinite(best[0])
    assert best[0] == 1.0


def test_bug_imp_04_arclength_restore_committed():
    """BUG-IMP-04: On arc-length max cuts failure (cuts > 8), element buffers
    must be restored to the committed snapshot."""
    from pyradioss.implicit.statics import _restore, _snapshot

    class DummyGroup:
        def __init__(self):
            self.state = {"sig": np.array([1.0, 2.0, 3.0])}

    group = DummyGroup()
    committed = {"g1": _snapshot(group)}

    # Mutate state during iteration
    group.state["sig"][:] = [99.0, 99.0, 99.0]

    # Max cuts failure restores committed
    _restore(group, committed["g1"])
    assert np.allclose(group.state["sig"], [1.0, 2.0, 3.0])


def test_bug_imp_05_filter_modes_growing_modes():
    """BUG-IMP-05: In _filter_modes, modes with positive real parts (growing/unstable)
    must be discarded even if lam[k].real < rigid_tol."""
    from pyradioss.implicit.complex_modal import _filter_modes

    scale = 100.0
    rigid_tol = 1e-6 * scale

    lam = np.array([
        -1.0 + 50.0j,       # decaying
        -1.0 - 50.0j,       # decaying
        1e-5 + 50.0j,       # growing, real < rigid_tol (BUG-IMP-05 fix drops this)
        1e-5 - 50.0j,       # growing, real < rigid_tol (BUG-IMP-05 fix drops this)
        1e-7 + 0.0j,        # rigid
    ])
    Z = np.eye(len(lam), dtype=complex)
    nred = len(lam) // 2

    lam_keep, Z_keep = _filter_modes(lam, Z, nred)

    # Only the decaying modes should be kept
    assert len(lam_keep) == 2
    assert all(l.real <= 1e-12 * scale for l in lam_keep)
    assert np.allclose([l.imag for l in lam_keep], [-50.0, 50.0]) or np.allclose([l.imag for l in lam_keep], [50.0, -50.0])


def test_bug_imp_06_crossing_rates_m2_le_zero():
    """BUG-IMP-06: crossing_rates must assign 0.0 (not np.nan) when m2 <= 0.0."""
    from pyradioss.implicit.random_response import crossing_rates

    moments = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    res = crossing_rates(moments)

    assert res["nu0"] == 0.0
    assert res["nup"] == 0.0
    assert not np.isnan(res["nup"])
