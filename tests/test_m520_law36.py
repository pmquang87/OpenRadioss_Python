"""
Milestone M520: Tabulated Elasto-Plastic Material Law (/MAT/LAW36 /MAT/PLAS_TAB)
Hardening, Multi-Rate Tables, Kinematic Hardening, Rate Filtering, Algorithmic Tangents & Unit Tests.

Covers:
1. Empty array guards for solids, shells, and tangents
2. Single-curve elastic trial response
3. Single-curve plastic yielding exactly on the curve
4. Radial collinearity and total pressure conservation
5. End-slope extrapolation beyond table bounds
6. Multi-rate linear strain-rate interpolation
7. Multi-rate logarithmic strain-rate interpolation (f_smooth=2)
8. Rate clamping at bounds
9. Zero and negative dt guards against division-by-zero
10. Softening curve plastic flow and tangent stability
11. Kinematic hardening backstress tensor evolution
12. Kinematic hardening Bauschinger effect under reverse loading
13. Strain-rate exponential moving average filtering (f_cut > 0)
14. Viscoplastic formulation dynamic rate coupling (vp > 0)
15. Solid total pressure with density (rho / rho0 - 1)
16. Shell plane-stress radial projection onto curve
17. Consistent solid tangent algorithmic derivative verification
18. Consistent shell tangent algorithmic derivative verification
19. Elastic unloading Kuhn-Tucker condition verification
20. Multi-element batched vectorization equivalence
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.common.tables import FunctTable
from pyradioss.materials import law01_elastic, law36_tabulated
from pyradioss.model.entities import Material


def _make_law36(curves, rates=None, E=200000.0, nu=0.3, rho0=7.8e-9, **kwargs):
    """Helper to create a LAW36 Material instance with resolved curve tables."""
    if rates is None:
        rates = [0.0] * len(curves)
    params = {
        "E": E,
        "nu": nu,
        "eps_p_max": 1e30,
        "funct_ids": list(range(len(curves))),
    }
    params.update(kwargs)
    cxs, cys, css = [], [], []
    for (x, y) in curves:
        f = FunctTable(1, x, y)
        cxs.append(f.x)
        cys.append(f.y)
        css.append(f.slope)
    params.update(curve_x=cxs, curve_y=cys, curve_s=css,
                  rates=np.asarray(rates, dtype=float))
    return Material(id=1, law=36, rho0=rho0, params=params)


def test_law36_empty_arrays():
    mat = _make_law36([([0.0, 0.1], [300.0, 400.0])])
    sig = np.empty((0, 6))
    deps = np.empty((0, 6))
    epsp = np.empty((0,))
    s, ep = law36_tabulated.solid_update(mat, sig, deps, epsp, dt=1e-4)
    assert s.shape == (0, 6)
    assert ep.shape == (0,)

    sig3 = np.empty((0, 3))
    deps3 = np.empty((0, 3))
    s3, ep3 = law36_tabulated.shell_update(mat, sig3, deps3, epsp, dt=1e-4)
    assert s3.shape == (0, 3)
    assert ep3.shape == (0,)

    D_solid = law36_tabulated.consistent_solid_tangent(mat, sig, epsp, epsp)
    assert D_solid.shape == (0, 6, 6)

    D_shell = law36_tabulated.consistent_shell_tangent(mat, sig3, epsp, epsp)
    assert D_shell.shape == (0, 3, 3)


def test_law36_single_curve_elastic_trial():
    mat = _make_law36([([0.0, 0.1], [300.0, 400.0])])
    sig = np.zeros((1, 6))
    deps = np.array([[1e-4, -0.3 * 1e-4, -0.3 * 1e-4, 0.0, 0.0, 0.0]])
    epsp = np.zeros(1)
    s, ep = law36_tabulated.solid_update(mat, sig, deps, epsp, dt=1e-4)
    assert ep[0] == 0.0
    assert s[0, 0] == pytest.approx(mat.E * 1e-4, rel=1e-5)
    assert abs(s[0, 1]) < 1e-3
    assert abs(s[0, 2]) < 1e-3


def test_law36_single_curve_plastic_yielding():
    mat = _make_law36([([0.0, 0.02, 0.05], [300.0, 400.0, 500.0])])
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    d = 2e-4
    deps = np.array([[d, -d / 2.0, -d / 2.0, 0.0, 0.0, 0.0]])
    for _ in range(100):
        law36_tabulated.solid_update(mat, sig, deps, epsp, dt=1e-4)
    assert epsp[0] > 0.01
    vm = np.sqrt(0.5 * ((sig[0, 0] - sig[0, 1]) ** 2
                        + (sig[0, 1] - sig[0, 2]) ** 2
                        + (sig[0, 2] - sig[0, 0]) ** 2))
    expected_sy = 300.0 + (100.0 / 0.02) * epsp[0]
    assert vm == pytest.approx(expected_sy, rel=1e-12)


def test_law36_radial_collinearity_and_pressure():
    mat = _make_law36([([0.0, 0.05], [200.0, 300.0])])
    sig = np.zeros((1, 6))
    deps = np.array([[1e-3, -2e-4, -1e-4, 1.5e-3, 0.8e-3, 0.5e-3]])
    epsp = np.zeros(1)

    tr3 = np.sum(deps[0, :3]) / 3.0
    s_tr = np.zeros(6)
    s_tr[:3] = 2.0 * mat.G * (deps[0, :3] - tr3)
    s_tr[3:] = mat.G * deps[0, 3:]

    s, ep = law36_tabulated.solid_update(mat, sig, deps, epsp, dt=1e-4)
    assert ep[0] > 0.0

    p = np.sum(s[0, :3]) / 3.0
    assert p == pytest.approx(mat.K * 3.0 * tr3, rel=1e-9)

    s_dev = s[0].copy()
    s_dev[:3] -= p
    ratio = s_dev / s_tr
    assert np.allclose(ratio, ratio[0], rtol=1e-10)


def test_law36_end_slope_extrapolation():
    mat = _make_law36([([0.0, 0.01, 0.02], [300.0, 350.0, 400.0])])
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    d = 5e-4
    deps = np.array([[d, -d / 2.0, -d / 2.0, 0.0, 0.0, 0.0]])
    for _ in range(120):
        law36_tabulated.solid_update(mat, sig, deps, epsp, dt=1e-4)
    assert epsp[0] > 0.03
    vm = np.sqrt(0.5 * ((sig[0, 0] - sig[0, 1]) ** 2
                        + (sig[0, 1] - sig[0, 2]) ** 2
                        + (sig[0, 2] - sig[0, 0]) ** 2))
    expected = 400.0 + 5000.0 * (epsp[0] - 0.02)
    assert vm == pytest.approx(expected, rel=1e-10)


def test_law36_multi_rate_linear_interpolation():
    mat = _make_law36([([0.0, 1.0], [300.0, 300.0]),
                       ([0.0, 1.0], [500.0, 500.0])],
                      rates=[0.0, 100.0])
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    d, dt = 4e-3, 1e-4  # rate = 4e-3 / 1e-4 = 40.0
    deps = np.array([[d, -d / 2.0, -d / 2.0, 0.0, 0.0, 0.0]])
    for _ in range(20):
        law36_tabulated.solid_update(mat, sig, deps, epsp, dt)
    vm = np.sqrt(0.5 * ((sig[0, 0] - sig[0, 1]) ** 2
                        + (sig[0, 1] - sig[0, 2]) ** 2
                        + (sig[0, 2] - sig[0, 0]) ** 2))
    assert vm == pytest.approx(380.0, rel=1e-9)


def test_law36_multi_rate_log_interpolation():
    mat = _make_law36([([0.0, 1.0], [300.0, 300.0]),
                       ([0.0, 1.0], [500.0, 500.0])],
                      rates=[1.0, 100.0],
                      f_smooth=2)
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    d, dt = 1e-3, 1e-4  # rate = 1e-3 / 1e-4 = 10.0 -> log(10/1)/log(100/1) = 0.5
    deps = np.array([[d, -d / 2.0, -d / 2.0, 0.0, 0.0, 0.0]])
    for _ in range(20):
        law36_tabulated.solid_update(mat, sig, deps, epsp, dt)
    vm = np.sqrt(0.5 * ((sig[0, 0] - sig[0, 1]) ** 2
                        + (sig[0, 1] - sig[0, 2]) ** 2
                        + (sig[0, 2] - sig[0, 0]) ** 2))
    assert vm == pytest.approx(400.0, rel=1e-9)


def test_law36_rate_clamping_or_extrapolation():
    mat = _make_law36([([0.0, 1.0], [300.0, 300.0]),
                       ([0.0, 1.0], [500.0, 500.0])],
                      rates=[0.0, 100.0])
    sy, H = law36_tabulated._yield_stress(mat, np.array([0.0]), np.array([0.0]))
    assert sy[0] == pytest.approx(300.0, rel=1e-9)


def test_law36_zero_negative_dt_guard():
    mat = _make_law36([([0.0, 0.1], [300.0, 400.0])])
    sig = np.zeros((1, 6))
    deps = np.array([[5e-3, -2.5e-3, -2.5e-3, 0.0, 0.0, 0.0]])
    epsp = np.zeros(1)
    s, ep = law36_tabulated.solid_update(mat, sig, deps, epsp, dt=0.0)
    assert not np.isnan(s).any()
    assert ep[0] > 0.0

    sig2 = np.zeros((1, 6))
    epsp2 = np.zeros(1)
    s2, ep2 = law36_tabulated.solid_update(mat, sig2, deps, epsp2, dt=-1e-4)
    assert not np.isnan(s2).any()
    assert ep2[0] > 0.0

    sig_sh = np.zeros((1, 3))
    deps_sh = np.array([[5e-3, 0.0, 0.0]])
    epsp_sh = np.zeros(1)
    s_sh, ep_sh = law36_tabulated.shell_update(mat, sig_sh, deps_sh, epsp_sh, dt=0.0)
    assert not np.isnan(s_sh).any()
    assert ep_sh[0] > 0.0


def test_law36_softening_curve_plastic_flow():
    mat = _make_law36([([0.0, 0.05], [400.0, 300.0])])
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    d = 2e-4
    deps = np.array([[d, -d / 2.0, -d / 2.0, 0.0, 0.0, 0.0]])
    for _ in range(80):
        law36_tabulated.solid_update(mat, sig, deps, epsp, dt=1e-4)
    assert epsp[0] > 0.005
    vm = np.sqrt(0.5 * ((sig[0, 0] - sig[0, 1]) ** 2
                        + (sig[0, 1] - sig[0, 2]) ** 2
                        + (sig[0, 2] - sig[0, 0]) ** 2))
    expected = 400.0 - 2000.0 * epsp[0]
    assert vm == pytest.approx(expected, rel=1e-10)
    D = law36_tabulated.consistent_solid_tangent(mat, sig, epsp, np.array([1e-4]))
    assert not np.isnan(D).any()


def test_law36_kinematic_hardening_backstress_growth():
    mat = _make_law36([([0.0, 0.05], [300.0, 400.0])], c_hard=0.5)
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    extra = {"sigb36": np.zeros((1, 6))}
    d = 5e-4
    deps = np.array([[d, -d / 2.0, -d / 2.0, 0.0, 0.0, 0.0]])
    for _ in range(30):
        law36_tabulated.solid_update(mat, sig, deps, epsp, dt=1e-4, extra=extra)
    assert epsp[0] > 0.005
    sigb = extra["sigb36"]
    assert sigb[0, 0] > 0.0
    assert sigb[0, 1] < 0.0
    assert abs(np.sum(sigb[0, :3])) < 1e-10


def test_law36_kinematic_hardening_bauschinger_effect():
    # Pure isotropic vs mixed kinematic hardening
    mat_iso = _make_law36([([0.0, 0.05], [300.0, 400.0])], c_hard=0.0)
    mat_kin = _make_law36([([0.0, 0.05], [300.0, 400.0])], c_hard=1.0)

    sig_iso = np.zeros((1, 6))
    epsp_iso = np.zeros(1)
    sig_kin = np.zeros((1, 6))
    epsp_kin = np.zeros(1)
    extra_kin = {"sigb36": np.zeros((1, 6))}

    d = 5e-4
    deps_fwd = np.array([[d, -d / 2.0, -d / 2.0, 0.0, 0.0, 0.0]])
    for _ in range(30):
        law36_tabulated.solid_update(mat_iso, sig_iso, deps_fwd, epsp_iso, dt=1e-4)
        law36_tabulated.solid_update(mat_kin, sig_kin, deps_fwd, epsp_kin, dt=1e-4, extra=extra_kin)

    # Now reverse load into compression with a small increment
    d_rev = -2.5e-3
    deps_rev = np.array([[d_rev, -d_rev / 2.0, -d_rev / 2.0, 0.0, 0.0, 0.0]])
    law36_tabulated.solid_update(mat_iso, sig_iso, deps_rev, epsp_iso, dt=1e-4)
    law36_tabulated.solid_update(mat_kin, sig_kin, deps_rev, epsp_kin, dt=1e-4, extra=extra_kin)

    # Kinematic hardening yields earlier in reverse direction -> more plastic strain increment
    assert epsp_kin[0] > epsp_iso[0]


def test_law36_strain_rate_filtering():
    f_cut = 50.0
    mat = _make_law36([([0.0, 1.0], [300.0, 300.0]),
                       ([0.0, 1.0], [500.0, 500.0])],
                      rates=[0.0, 1000.0],
                      f_cut=f_cut)
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    extra = {"epsd36": np.zeros(1)}

    d, dt = 1e-3, 1e-4  # raw rate = 10.0
    deps = np.array([[d, -d / 2.0, -d / 2.0, 0.0, 0.0, 0.0]])
    law36_tabulated.solid_update(mat, sig, deps, epsp, dt, extra=extra)

    expected_alpha = min(1.0, 2.0 * np.pi * f_cut * dt)
    expected_rate = expected_alpha * 10.0
    assert extra["epsd36"][0] == pytest.approx(expected_rate, rel=1e-5)


def test_law36_viscoplastic_formulation():
    mat_vp = _make_law36([([0.0, 1.0], [300.0, 300.0]),
                          ([0.0, 1.0], [600.0, 600.0])],
                         rates=[0.0, 1000.0],
                         vp=1.0)
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    d, dt = 5e-3, 1e-4
    deps = np.array([[d, -d / 2.0, -d / 2.0, 0.0, 0.0, 0.0]])
    law36_tabulated.solid_update(mat_vp, sig, deps, epsp, dt)
    assert epsp[0] > 0.0
    vm = np.sqrt(0.5 * ((sig[0, 0] - sig[0, 1]) ** 2
                        + (sig[0, 1] - sig[0, 2]) ** 2
                        + (sig[0, 2] - sig[0, 0]) ** 2))
    assert vm > 300.0


def test_law36_solid_total_pressure_with_density():
    mat = _make_law36([([0.0, 0.1], [300.0, 400.0])])
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    epsp = np.zeros(1)
    extra = {"rho": 1.05 * mat.rho0}
    s, ep = law36_tabulated.solid_update(mat, sig, deps, epsp, dt=1e-4, extra=extra)
    expected_p = -mat.K * (1.05 - 1.0)
    assert s[0, 0] == pytest.approx(expected_p, rel=1e-9)
    assert s[0, 1] == pytest.approx(expected_p, rel=1e-9)
    assert s[0, 2] == pytest.approx(expected_p, rel=1e-9)


def test_law36_shell_plane_stress_return():
    mat = _make_law36([([0.0, 0.05], [300.0, 450.0])])
    sig = np.zeros((1, 3))
    epsp = np.zeros(1)
    deps = np.array([[2e-4, 2e-4, 0.0]])
    for _ in range(50):
        law36_tabulated.shell_update(mat, sig, deps, epsp, dt=1e-4)
    assert epsp[0] > 0.005
    vm = np.sqrt(sig[0, 0] ** 2 - sig[0, 0] * sig[0, 1] + sig[0, 1] ** 2 + 3.0 * sig[0, 2] ** 2)
    expected_sy = 300.0 + (150.0 / 0.05) * epsp[0]
    assert vm == pytest.approx(expected_sy, rel=1e-11)


def test_law36_consistent_solid_tangent_derivative():
    mat = _make_law36([([0.0, 0.05], [300.0, 350.0])])
    sig_init = np.array([[200.0, -100.0, -100.0, 50.0, 0.0, 0.0]])
    epsp_init = np.array([0.002])

    deps_base = np.array([[0.002, -0.0005, -0.0005, 0.001, 0.0, 0.0]])
    sig = sig_init.copy()
    epsp = epsp_init.copy()
    law36_tabulated.solid_update(mat, sig, deps_base, epsp, dt=1e-4)
    dep_incr = epsp - epsp_init
    D_alg = law36_tabulated.consistent_solid_tangent(mat, sig, epsp, dep_incr)[0]

    delta_eps = np.array([[0.0005, -0.0002, 0.0001, 0.0004, -0.0002, 0.0003]])
    eps = 1e-7

    sig_p = sig_init.copy()
    epsp_p = epsp_init.copy()
    law36_tabulated.solid_update(mat, sig_p, deps_base + eps * delta_eps, epsp_p, dt=1e-4)

    sig_m = sig_init.copy()
    epsp_m = epsp_init.copy()
    law36_tabulated.solid_update(mat, sig_m, deps_base - eps * delta_eps, epsp_m, dt=1e-4)

    d_sig_num = (sig_p[0] - sig_m[0]) / (2.0 * eps)
    d_sig_tan = D_alg @ delta_eps[0]

    rel_err = np.linalg.norm(d_sig_num - d_sig_tan) / np.linalg.norm(d_sig_tan)
    assert rel_err < 1e-4


def test_law36_consistent_shell_tangent_derivative():
    mat = _make_law36([([0.0, 0.05], [300.0, 350.0])])
    sig_init = np.array([[180.0, -80.0, 40.0]])
    epsp_init = np.array([0.0015])

    deps_base = np.array([[0.002, -0.0008, 0.0012]])
    sig = sig_init.copy()
    epsp = epsp_init.copy()
    law36_tabulated.shell_update(mat, sig, deps_base, epsp, dt=1e-4)
    dep_incr = epsp - epsp_init
    D_shell = law36_tabulated.consistent_shell_tangent(mat, sig, epsp, dep_incr)[0]

    delta_eps = np.array([[0.0004, -0.0003, 0.0005]])
    eps = 1e-7

    sig_p = sig_init.copy()
    epsp_p = epsp_init.copy()
    law36_tabulated.shell_update(mat, sig_p, deps_base + eps * delta_eps, epsp_p, dt=1e-4)

    sig_m = sig_init.copy()
    epsp_m = epsp_init.copy()
    law36_tabulated.shell_update(mat, sig_m, deps_base - eps * delta_eps, epsp_m, dt=1e-4)

    d_sig_num = (sig_p[0] - sig_m[0]) / (2.0 * eps)
    d_sig_tan = D_shell @ delta_eps[0]

    rel_err = np.linalg.norm(d_sig_num - d_sig_tan) / np.linalg.norm(d_sig_tan)
    assert rel_err < 1e-4


def test_law36_elastic_unloading_kuhn_tucker():
    mat = _make_law36([([0.0, 0.05], [300.0, 400.0])])
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    deps = np.array([[2e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0]])
    law36_tabulated.solid_update(mat, sig, deps, epsp, dt=1e-4)
    ep_yielded = float(epsp[0])
    assert ep_yielded > 0.0

    sig_pre = sig.copy()
    deps_unload = np.array([[-1e-4, 3e-5, 3e-5, 0.0, 0.0, 0.0]])
    law36_tabulated.solid_update(mat, sig, deps_unload, epsp, dt=1e-4)
    assert epsp[0] == ep_yielded
    assert sig[0, 0] < sig_pre[0, 0]


def test_law36_multi_element_batch_vectorization():
    mat = _make_law36([([0.0, 0.05], [300.0, 450.0])])
    N = 8
    np.random.seed(42)
    sig_batch = np.zeros((N, 6))
    deps_batch = (np.random.rand(N, 6) - 0.5) * 4e-3
    epsp_batch = np.zeros(N)

    s_bat, ep_bat = law36_tabulated.solid_update(mat, sig_batch.copy(), deps_batch, epsp_batch.copy(), dt=1e-4)

    for i in range(N):
        s_ind, ep_ind = law36_tabulated.solid_update(mat, sig_batch[i:i+1].copy(), deps_batch[i:i+1], epsp_batch[i:i+1].copy(), dt=1e-4)
        assert np.allclose(s_bat[i], s_ind[0], atol=1e-12)
        assert np.allclose(ep_bat[i], ep_ind[0], atol=1e-12)
