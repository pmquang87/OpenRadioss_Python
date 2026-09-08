import numpy as np
import pytest

from pyradioss.materials import law70_tabfoam
from pyradioss.model.entities import Material


def _make_law70(e0=1000.0, nu=0.2, rho0=1.0e-9, emax=5000.0, epsmax=0.5,
                iflag=0, shape=1.0, hys=0.5, itens=0, fcut=1e30,
                xg=None, rl=None, yl=None, xu=None, ru=None, yu=None, **kwargs):
    if xg is None:
        xg = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    if rl is None:
        rl = np.array([0.0])
    if yl is None:
        # Default static loading curve: linear elastic up to 0.1, then plateau
        yl = np.array([[0.0], [50.0], [60.0], [80.0], [120.0], [200.0], [400.0]])

    aa = (emax - e0) / epsmax

    # Compute YLD_EMAX
    stat = yl[:, 0]
    k = int(np.searchsorted(xg, epsmax, side="left"))
    k = min(max(k, 1), len(xg) - 1)
    deri = (stat[k] - stat[k - 1]) / (xg[k] - xg[k - 1])
    yld_emax = float(stat[k - 1] + deri * (epsmax - xg[k - 1]))

    params = {
        "E": e0, "nu": nu, "E0": e0, "EMAX": emax,
        "EPSMAX": epsmax, "AA": aa,
        "iflag": iflag, "shape": shape, "hys": hys,
        "itens": itens, "fcut": fcut, "ismooth": 0,
        "xg_load": xg, "r_load": rl, "y_load": yl,
        "YLD_EMAX": yld_emax,
    }
    if xu is not None and ru is not None and yu is not None:
        params["xg_un"] = xu
        params["r_un"] = ru
        params["y_un"] = yu
    elif iflag in (0, 1, 2):
        # Default unloading curve: lower stress
        params["xg_un"] = xg
        params["r_un"] = rl
        params["y_un"] = yl * 0.4
    else:
        params["xg_un"] = None

    if itens > 0:
        params["tens_fid"] = 1
        params["tens_scale"] = 1.0
        params["tens_x"] = np.array([0.0, 0.1, 0.5])
        params["tens_y"] = np.array([1.0, 0.5, 0.1])

    params.update(kwargs)
    mat = Material(1, law=70, rho0=rho0, params=params)
    return mat


def _init_extra(n=1, e0=1000.0, rho0=1.0e-9):
    uv = np.zeros((n, 10))
    uv[:, 2] = e0
    return {
        "eps70": np.zeros((n, 6)),
        "uv70": uv,
        "epsd70": np.zeros(n),
        "rho": np.full(n, rho0),
    }


def test_law70_empty_arrays():
    mat = _make_law70()
    sig = np.empty((0, 6))
    deps = np.empty((0, 6))
    extra = _init_extra(0)

    s_out, c_out = law70_tabfoam.solid_update(mat, sig, deps, 1e-4, extra)
    assert s_out.shape == (0, 6)
    assert c_out.shape == (0,)

    D = law70_tabfoam.consistent_solid_tangent(mat, extra)
    assert D.shape == (0, 6, 6)

    D_none = law70_tabfoam.consistent_solid_tangent(mat, None)
    assert D_none.shape == (0, 6, 6)


def test_law70_shell_update_raises_not_implemented():
    mat = _make_law70()
    with pytest.raises(NotImplementedError, match="3D solid elements only"):
        law70_tabfoam.shell_update(mat, np.zeros((1, 3)), np.zeros((1, 3)), None, 1e-4)


def test_law70_static_loading_curve_exact_radial_projection():
    # Pure uniaxial strain exx = 0.2
    # At exx = 0.2, epst = 0.2, yld = yl[2] = 60.0
    mat = _make_law70(e0=1000.0, nu=0.0)
    extra = _init_extra(1, e0=1000.0)
    sig = np.zeros((1, 6))
    deps = np.array([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]])

    s, c = law70_tabfoam.solid_update(mat, sig, deps, 1e-4, extra)

    # Tensor Frobenius norm of stress must match yld = 60.0
    snorm_val = np.sqrt(s[0, 0]**2 + s[0, 1]**2 + s[0, 2]**2 + 2.0*(s[0, 3]**2 + s[0, 4]**2 + s[0, 5]**2))
    assert snorm_val == pytest.approx(60.0, rel=1e-10)
    # Uniaxial strain with nu=0 produces stress purely along xx
    assert s[0, 0] == pytest.approx(60.0, rel=1e-10)
    assert abs(s[0, 1]) < 1e-12
    assert abs(s[0, 2]) < 1e-12


def test_law70_table_resolution_slopes_and_emax():
    from pyradioss.common.tables import FunctTable
    class DummyModel:
        def __init__(self):
            self.functions = {
                1: FunctTable(1, np.array([0.0, 0.2, 0.5, 1.0]),
                              np.array([0.0, 100.0, 200.0, 600.0]), "load"),
                2: FunctTable(2, np.array([0.0, 0.5, 1.0]),
                              np.array([0.0, 50.0, 150.0]), "unload")
            }
    class DummyLog:
        def warning(self, msg, cat): pass
        def error(self, msg, cat): pass

    model = DummyModel()
    log = DummyLog()

    mat = Material(1, law=70, rho0=1.0e-9, params={
        "E": 200.0, "nu": 0.25, "E0": 200.0, "EMAX": 0.0, "EPSMAX": 1.0,
        "iflag": 0, "shape": 1.0, "hys": 1.0, "itens": 0, "fcut": 1e30,
        "ismooth": 0,
        "load_fids": [1], "load_rates": [0.0], "load_scales": [1.0],
        "unload_fids": [2], "unload_rates": [0.0], "unload_scales": [1.0],
    })

    law70_tabfoam.resolve(mat, model, log)
    # Initial slope is (100 - 0) / (0.2 - 0) = 500.0. Since E0 = 200 < 500, E0 is raised to 500!
    assert mat.params["E0"] == pytest.approx(500.0, rel=1e-10)
    # Max table slope is (600 - 200) / (1.0 - 0.5) = 800.0. EMAX is auto-computed as 800!
    assert mat.params["EMAX"] == pytest.approx(800.0, rel=1e-10)


def test_law70_initial_sound_speed_pwave():
    e0 = 1200.0
    nu = 0.25
    rho0 = 1.2e-9
    mat = _make_law70(e0=e0, nu=nu, rho0=rho0)
    extra = _init_extra(1, e0=e0, rho0=rho0)
    sig = np.zeros((1, 6))

    deps = np.zeros((1, 6))
    _, c = law70_tabfoam.solid_update(mat, sig, deps, 1e-4, extra)

    # Theoretical P-wave speed: sqrt(AA1 / rho0)
    aa1 = e0 * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    expected_c = np.sqrt(aa1 / rho0)
    assert c[0] == pytest.approx(expected_c, rel=1e-10)


def test_law70_evolving_unloading_modulus_growth():
    e0 = 1000.0
    emax = 5000.0
    epsmax = 0.5
    mat = _make_law70(e0=e0, emax=emax, epsmax=epsmax, nu=0.2)
    extra = _init_extra(1, e0=e0)
    sig = np.zeros((1, 6))

    # Initial modulus in uv70 is e0
    assert extra["uv70"][0, 2] == pytest.approx(e0, rel=1e-10)

    # Load in several steps to exx = 0.2
    for _ in range(4):
        law70_tabfoam.solid_update(mat, sig, np.array([[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)

    # Modulus must have relaxed upwards: E_new > E0
    e_evolved = extra["uv70"][0, 2]
    assert e_evolved > e0
    assert e_evolved <= emax

    # Further loading past epsmax (0.5) increases modulus further towards emax
    law70_tabfoam.solid_update(mat, sig, np.array([[0.4, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)
    e_further = extra["uv70"][0, 2]
    assert e_further > e_evolved
    assert e_further <= emax


def test_law70_evolving_modulus_direction_reversal():
    mat = _make_law70(e0=1000.0, emax=5000.0, epsmax=0.5, nu=0.2, iflag=1)
    extra = _init_extra(1, e0=1000.0)
    sig = np.zeros((1, 6))

    # Load to exx = 0.2
    law70_tabfoam.solid_update(mat, sig, np.array([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)
    e_at_peak = extra["uv70"][0, 2]
    assert extra["uv70"][0, 4] == 1.0  # iload = 1 (loading)

    # Reversal: first unloading step
    # Direction reversal rule: e_new restarts from stored modulus (e_old)
    law70_tabfoam.solid_update(mat, sig, np.array([[-0.01, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)
    assert extra["uv70"][0, 4] == -1.0  # iload = -1 (unloading)
    assert extra["uv70"][0, 2] == pytest.approx(e_at_peak, rel=1e-10)


def test_law70_modulus_growth_increases_sound_speed():
    e0 = 1000.0
    emax = 4000.0
    mat = _make_law70(e0=e0, emax=emax, epsmax=0.5, nu=0.2)
    extra = _init_extra(1, e0=e0)
    sig = np.zeros((1, 6))

    # Cycle 1: small strain
    _, c1 = law70_tabfoam.solid_update(mat, sig.copy(), np.array([[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)
    # Cycle 2: large strain (driving modulus growth)
    _, c2 = law70_tabfoam.solid_update(mat, sig.copy(), np.array([[0.35, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)

    assert c2[0] > c1[0]


def test_law70_iflag0_unloading_between_yield_bounds():
    mat = _make_law70(e0=1000.0, iflag=0, nu=0.2)
    extra = _init_extra(1, e0=1000.0)
    sig = np.zeros((1, 6))

    # Step 1: Load to exx = 0.2 -> yld = 60.0
    sig, _ = law70_tabfoam.solid_update(mat, sig, np.array([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)
    yld_peak = extra["uv70"][0, 5]
    assert yld_peak == pytest.approx(60.0, rel=1e-10)

    # Step 2: Unload slightly to exx = 0.18
    sig, _ = law70_tabfoam.solid_update(mat, sig, np.array([[-0.02, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)
    yld_un = extra["uv70"][0, 5]
    # In Iflag=0 unloading, yld is between yldmin (0.4 * 52 = 20.8) and yld_peak (60.0)
    assert yld_un <= yld_peak
    assert yld_un >= 20.0


def test_law70_iflag0_escape_regime():
    mat = _make_law70(e0=1000.0, iflag=0, nu=0.0)
    extra = _init_extra(1, e0=1000.0)
    sig = np.zeros((1, 6))

    # Step 1: Load to exx = 0.4 -> yld = 80.0
    law70_tabfoam.solid_update(mat, sig, np.array([[0.4, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)

    # Step 2: Massive sudden unloading jump back to exx = 0.05
    # Triggers escape condition: dsig > yldmin and dsig > svm
    law70_tabfoam.solid_update(mat, sig, np.array([[-0.35, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)
    yld_escaped = extra["uv70"][0, 5]
    # Should escape to yldmin at exx = 0.05
    # At exx = 0.05, yldmin = 0.4 * 25.0 = 10.0
    assert yld_escaped == pytest.approx(10.0, rel=1e-5)


def test_law70_iflag1_damage_deviator_scaling():
    # Iflag = 1: scales deviator only by r2 = yldmin / yldelas during unloading
    mat = _make_law70(e0=1000.0, iflag=1, nu=0.25)
    extra = _init_extra(1, e0=1000.0)
    sig = np.zeros((1, 6))

    # Step 1: Load
    law70_tabfoam.solid_update(mat, sig, np.array([[0.2, 0.1, 0.1, 0.0, 0.0, 0.0]]), 1e-4, extra)

    # Step 2: Unload
    s_un, _ = law70_tabfoam.solid_update(mat, sig, np.array([[-0.05, -0.02, -0.02, 0.0, 0.0, 0.0]]), 1e-4, extra)

    # Mean hydrostatic pressure P = (sxx + syy + szz)/3 is preserved while deviators are scaled
    pm = (s_un[0, 0] + s_un[0, 1] + s_un[0, 2]) / 3.0
    assert abs(pm) > 0.0


def test_law70_iflag2_damage_whole_tensor_scaling():
    # Iflag = 2: scales whole tensor by r2 = yldmin / yldelas
    mat = _make_law70(e0=1000.0, iflag=2, nu=0.25)
    extra = _init_extra(1, e0=1000.0)
    sig = np.zeros((1, 6))

    # Step 1: Load
    law70_tabfoam.solid_update(mat, sig, np.array([[0.2, 0.1, 0.1, 0.0, 0.0, 0.0]]), 1e-4, extra)
    # Step 2: Unload
    s_un, _ = law70_tabfoam.solid_update(mat, sig, np.array([[-0.05, -0.02, -0.02, 0.0, 0.0, 0.0]]), 1e-4, extra)
    assert not np.isnan(s_un).any()


def test_law70_iflag3_energy_hysteresis_deviator_scaling():
    mat = _make_law70(e0=1000.0, iflag=3, shape=2.0, hys=0.4, nu=0.2)
    extra = _init_extra(1, e0=1000.0)
    sig = np.zeros((1, 6))

    # Load
    law70_tabfoam.solid_update(mat, sig, np.array([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)
    # Unload
    s_un, _ = law70_tabfoam.solid_update(mat, sig, np.array([[-0.05, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)
    assert extra["uv70"][0, 7] > 0.0  # Energy dissipated accumulated


def test_law70_iflag4_energy_hysteresis_whole_tensor():
    mat = _make_law70(e0=1000.0, iflag=4, shape=2.0, hys=0.4, nu=0.2)
    extra = _init_extra(1, e0=1000.0)
    sig = np.zeros((1, 6))

    # Load
    law70_tabfoam.solid_update(mat, sig, np.array([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)
    # Unload
    s_un, _ = law70_tabfoam.solid_update(mat, sig, np.array([[-0.05, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)
    assert extra["uv70"][0, 7] > 0.0


def test_law70_dynamic_rate_interpolation():
    # 2 rate curves: 0.0 and 100.0
    xg = np.array([0.0, 0.2, 0.5])
    rl = np.array([0.0, 100.0])
    yl = np.array([[0.0, 0.0],
                   [50.0, 100.0],
                   [100.0, 200.0]])
    mat = _make_law70(e0=1000.0, xg=xg, rl=rl, yl=yl, nu=0.0)

    # Static rate (dt = 1.0, deps = 0.2 -> rate = 0.2 approx static)
    ex1 = _init_extra(1, e0=1000.0)
    s_stat, _ = law70_tabfoam.solid_update(mat, np.zeros((1, 6)), np.array([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1.0, ex1)

    # High rate (dt = 0.002, deps = 0.2 -> rate = 100.0)
    ex2 = _init_extra(1, e0=1000.0)
    s_dyn, _ = law70_tabfoam.solid_update(mat, np.zeros((1, 6)), np.array([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]), 0.002, ex2)

    # Dynamic stress must be significantly higher than static
    assert s_dyn[0, 0] > 1.8 * s_stat[0, 0]


def test_law70_rate_filtering_fcut():
    mat = _make_law70(fcut=20.0)  # moderate filter frequency
    extra = _init_extra(1)
    sig = np.zeros((1, 6))

    # Instantaneous spike in strain rate: deps = 0.1, dt = 1e-4 -> raw rate = 1000.0
    dt = 1e-4
    law70_tabfoam.solid_update(mat, sig, np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]), dt, extra)

    filtered_rate = extra["epsd70"][0]
    raw_rate = 0.1 / dt  # 1000.0
    # Alpha = min(1, 2 * pi * 20 * 1e-4) = 0.012566
    # Filtered rate should be ~12.57 << 1000.0
    assert filtered_rate < 0.1 * raw_rate
    assert filtered_rate > 0.0


def test_law70_tension_scaling_itens():
    mat = _make_law70(itens=1, nu=0.0)
    extra = _init_extra(1)
    # Net volumetric tension: rho < rho0 -> mu = 1 - rho/rho0 = 0.1
    extra["rho"] = np.array([0.9 * mat.rho0])
    sig = np.zeros((1, 6))

    # At mu = 0.1, tens_y = 0.5 (from _make_law70 setup)
    s, _ = law70_tabfoam.solid_update(mat, sig, np.array([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)

    # Without tension scaling, sxx would be 60.0. With 0.5 scaling, sxx = 30.0!
    assert s[0, 0] == pytest.approx(30.0, rel=1e-5)


def test_law70_zero_negative_dt_guard():
    mat = _make_law70()
    extra1 = _init_extra(1)
    s1, c1 = law70_tabfoam.solid_update(mat, np.zeros((1, 6)), np.array([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]]), 0.0, extra1)
    assert not np.isnan(s1).any()
    assert not np.isnan(c1).any()

    extra2 = _init_extra(1)
    s2, c2 = law70_tabfoam.solid_update(mat, np.zeros((1, 6)), np.array([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]]), -1e-4, extra2)
    assert not np.isnan(s2).any()
    assert not np.isnan(c2).any()


def test_law70_consistent_solid_tangent_structure():
    e0 = 1500.0
    nu = 0.25
    mat = _make_law70(e0=e0, nu=nu)
    extra = _init_extra(1, e0=e0)

    D = law70_tabfoam.consistent_solid_tangent(mat, extra)
    assert D.shape == (1, 6, 6)
    # Matrix must be exactly symmetric
    assert np.allclose(D[0], D[0].T, atol=1e-12)
    # Matrix must be positive definite
    eigvals = np.linalg.eigvalsh(D[0])
    assert np.all(eigvals > 0.0)

    # Diagonal terms
    aa1 = e0 * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    g = 0.5 * e0 / (1.0 + nu)
    assert D[0, 0, 0] == pytest.approx(aa1, rel=1e-10)
    assert D[0, 3, 3] == pytest.approx(g, rel=1e-10)


def test_law70_unloading_reversal_restart():
    mat = _make_law70(e0=1000.0, emax=4000.0, epsmax=0.5)
    extra = _init_extra(1, e0=1000.0)
    sig = np.zeros((1, 6))

    # Step 1: Load to 0.3
    law70_tabfoam.solid_update(mat, sig, np.array([[0.3, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)
    assert extra["uv70"][0, 4] == 1.0  # loading

    # Step 2: Unload to 0.2
    law70_tabfoam.solid_update(mat, sig, np.array([[-0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)
    assert extra["uv70"][0, 4] == -1.0  # unloading

    # Step 3: Reload back to 0.25
    law70_tabfoam.solid_update(mat, sig, np.array([[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1e-4, extra)
    assert extra["uv70"][0, 4] == 1.0  # reloaded


def test_law70_multi_element_batch_vectorization():
    mat = _make_law70(e0=1000.0, emax=5000.0, epsmax=0.5)
    N = 4
    extra_batch = _init_extra(N, e0=1000.0)

    deps_batch = np.array([
        [0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.25, 0.1, 0.0, 0.0, 0.0, 0.0],
        [-0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.40, 0.0, 0.0, 0.02, 0.0, 0.0],
    ])

    sig_batch = np.zeros((N, 6))
    s_bat, c_bat = law70_tabfoam.solid_update(mat, sig_batch, deps_batch, 1e-4, extra_batch)
    D_bat = law70_tabfoam.consistent_solid_tangent(mat, extra_batch)

    # Single element comparison
    for i in range(N):
        extra_ind = _init_extra(1, e0=1000.0)
        sig_ind = np.zeros((1, 6))
        s_ind, c_ind = law70_tabfoam.solid_update(mat, sig_ind, deps_batch[i:i+1], 1e-4, extra_ind)
        D_ind = law70_tabfoam.consistent_solid_tangent(mat, extra_ind)

        assert np.allclose(s_bat[i], s_ind[0], atol=1e-12)
        assert np.allclose(c_bat[i], c_ind[0], atol=1e-12)
        assert np.allclose(D_bat[i], D_ind[0], atol=1e-12)
