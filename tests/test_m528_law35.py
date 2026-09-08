"""
Tests for Milestone M528: LAW35 / FOAM_VISC Kelvin-Maxwell visco-elastic foam
constitutive model hardening, closed-form relaxation, air pressure, consistent
solid tangents, and starter checks integration.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import Material
import pyradioss.materials as materials
from pyradioss.materials import law35_kelvinmax
from pyradioss.starter.checks import _ALLOWED_LAWS


class _DummyFunction:
    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        self.slope = np.where(dx != 0.0, dy / dx, 0.0)


class _DummyModel:
    def __init__(self, funcs=None):
        self.functions = funcs or {}


def _make_law35_mat(**kwargs) -> Material:
    defaults = {
        "MAT_E": 10.0,
        "MAT_NU": 0.25,
        "MAT_ETAN": 2.0,
        "MAT_NUt": 0.2,
        "MAT_ETA2": 5.0,
        "rho": 1.2e-3,
    }
    defaults.update(kwargs)
    return law35_kelvinmax.build_law35(defaults)


# ----------------------------------------------------------------------------
# 1-4. Validation tests
# ----------------------------------------------------------------------------

def test_law35_validation_e_positive():
    with pytest.raises(ValueError, match="Young modulus E must be > 0"):
        law35_kelvinmax.build_law35({"MAT_E": 0.0, "MAT_NU": 0.2, "MAT_ETA2": 1.0})
    with pytest.raises(ValueError, match="Young modulus E must be > 0"):
        law35_kelvinmax.build_law35({"MAT_E": -10.0, "MAT_NU": 0.2, "MAT_ETA2": 1.0})


def test_law35_validation_nu_range():
    with pytest.raises(ValueError, match="Poisson ratio nu="):
        law35_kelvinmax.build_law35({"MAT_E": 10.0, "MAT_NU": -1.0, "MAT_ETA2": 1.0})
    with pytest.raises(ValueError, match="Poisson ratio nu="):
        law35_kelvinmax.build_law35({"MAT_E": 10.0, "MAT_NU": 0.5, "MAT_ETA2": 1.0})
    with pytest.raises(ValueError, match="Poisson ratio nu="):
        law35_kelvinmax.build_law35({"MAT_E": 10.0, "MAT_NU": 0.6, "MAT_ETA2": 1.0})


def test_law35_validation_nut_range():
    with pytest.raises(ValueError, match="tangent Poisson ratio nu_t="):
        law35_kelvinmax.build_law35({"MAT_E": 10.0, "MAT_NU": 0.2, "MAT_NUt": -1.0, "MAT_ETA2": 1.0})
    with pytest.raises(ValueError, match="tangent Poisson ratio nu_t="):
        law35_kelvinmax.build_law35({"MAT_E": 10.0, "MAT_NU": 0.2, "MAT_NUt": 0.5, "MAT_ETA2": 1.0})


def test_law35_validation_mu_positive():
    with pytest.raises(ValueError, match="Navier viscosity .* must be > 0"):
        law35_kelvinmax.build_law35({"MAT_E": 10.0, "MAT_NU": 0.2, "MAT_ETA2": 0.0})
    with pytest.raises(ValueError, match="Navier viscosity .* must be > 0"):
        law35_kelvinmax.build_law35({"MAT_E": 10.0, "MAT_NU": 0.2, "MAT_ETA2": -1.5})


# ----------------------------------------------------------------------------
# 5. Defaults and keys
# ----------------------------------------------------------------------------

def test_law35_defaults():
    mat = _make_law35_mat(MAT_PC=0.0, IFscale=0.0, Fsmooth=0, Fcut=0.0)
    assert mat.params["pmin"] == -1e20
    assert mat.params["fscale"] == 1.0
    assert mat.params["ismooth"] == 0
    assert mat.params["fcut"] == 0.0

    mat_pc_pos = _make_law35_mat(MAT_PC=50.0)
    assert mat_pc_pos.params["pmin"] == -50.0

    mat_cut = _make_law35_mat(Fcut=2500.0)
    assert mat_cut.params["ismooth"] == 1
    assert mat_cut.params["fcut"] == 2500.0

    mat_flag = _make_law35_mat(Fsmooth=1)
    assert mat_flag.params["ismooth"] == 1
    assert mat_flag.params["fcut"] == 10000.0


def test_law35_cfg_vs_direct_keys():
    cfg_input = {
        "MAT_E": 12.0, "MAT_NU": 0.3, "MAT_E1": 1.5, "MAT_E2": 2.5, "MAT_N": 0.8,
        "MAT_ETAN": 3.0, "MAT_NUt": 0.25, "MAT_ETA2": 4.0, "MAT_ETA1": 0.5,
        "MAT_CO1": 1.1, "MAT_CO2": 1.2, "MAT_CO3": 1.3, "Itype": 1,
        "MAT_PC": 10.0, "MAT_P0": 0.1, "MAT_PHI": 0.05, "MAT_GAMA0": 0.02,
        "FUN_A1": 101, "IFscale": 2.0, "Fcut": 5000.0,
    }
    dir_input = {
        "e": 12.0, "nu": 0.3, "e1": 1.5, "e2": 2.5, "n": 0.8,
        "et": 3.0, "nut": 0.25, "mu": 4.0, "lambda_visc": 0.5,
        "c1": 1.1, "c2": 1.2, "c3": 1.3, "itype": 1,
        "pmin": 10.0, "p0": 0.1, "phi": 0.05, "gama0": 0.02,
        "fct_id": 101, "fscale": 2.0, "fcut": 5000.0,
    }
    mat_cfg = law35_kelvinmax.build_law35(cfg_input)
    mat_dir = law35_kelvinmax.build_law35(dir_input)
    for k in ["E", "nu", "K", "G", "E1", "E2", "N", "Et", "nut", "mu_visc",
              "lambda_visc", "C1", "C2", "C3", "itype", "pmin", "P0", "phi",
              "gama0", "fct_id", "fscale", "ismooth", "fcut"]:
        assert pytest.approx(mat_cfg.params[k]) == mat_dir.params[k]


# ----------------------------------------------------------------------------
# 6-8. Empty array guards and shell rejection
# ----------------------------------------------------------------------------

def test_law35_solid_update_empty_array():
    mat = _make_law35_mat()
    sig = np.empty((0, 6), dtype=float)
    deps = np.empty((0, 6), dtype=float)
    sig_out, c_out = law35_kelvinmax.solid_update(mat, sig, deps, dt=1e-5)
    assert sig_out.shape == (0, 6)
    assert c_out.shape == (0,)


def test_law35_consistent_tangent_empty_array():
    mat = _make_law35_mat()
    sig = np.empty((0, 6), dtype=float)
    C = law35_kelvinmax.consistent_solid_tangent(mat, sig)
    assert C.shape == (0, 6, 6)


def test_law35_shell_update_rejection():
    mat = _make_law35_mat()
    sig = np.zeros((1, 3))
    deps = np.zeros((1, 3))
    with pytest.raises(NotImplementedError, match="LAW35 .* implemented for 3D solid"):
        law35_kelvinmax.shell_update(mat, sig, deps)
    with pytest.raises(NotImplementedError, match="LAW35 .* implemented for 3D solid"):
        materials.shell_update(mat, sig, deps, epsp=None, dt=1e-5)


# ----------------------------------------------------------------------------
# 9-10. Extra state auto-allocation
# ----------------------------------------------------------------------------

def test_law35_extra_auto_init_none():
    mat = _make_law35_mat()
    sig = np.zeros((2, 6))
    deps = np.zeros((2, 6))
    sig_out, c_out = law35_kelvinmax.solid_update(mat, sig, deps, dt=1e-5, extra=None)
    assert sig_out.shape == (2, 6)
    assert c_out.shape == (2,)
    assert np.all(c_out > 0)


def test_law35_extra_auto_init_missing_keys():
    mat = _make_law35_mat()
    sig = np.zeros((3, 6))
    deps = np.zeros((3, 6))
    extra = {"eps35": np.zeros((3, 6))}  # missing sigair35, edot35, rho
    sig_out, c_out = law35_kelvinmax.solid_update(mat, sig, deps, dt=1e-5, extra=extra)
    assert "sigair35" in extra
    assert "edot35" in extra
    assert "rho" in extra
    assert len(extra["sigair35"]) == 3
    assert len(extra["edot35"]) == 3
    assert len(extra["rho"]) == 3


# ----------------------------------------------------------------------------
# 11-13. Closed-form exponential relaxation, asymptotic stress, midstep
# ----------------------------------------------------------------------------

def test_law35_closed_form_relaxation_time_constant():
    E = 12.0
    nu = 0.2
    Et = 3.0
    nut = 0.2
    mu = 6.0
    mat = _make_law35_mat(MAT_E=E, MAT_NU=nu, MAT_ETAN=Et, MAT_NUt=nut, MAT_ETA2=mu)

    # G2 = E / (1+nu), Gt2 = Et / (1+nut)
    # tau = 2*mu / (G2 + Gt2)
    G2 = E / (1.0 + nu)
    Gt2 = Et / (1.0 + nut)
    tau = 2.0 * mu / (G2 + Gt2)

    n = 1
    sig = np.zeros((n, 6))
    deps0 = np.array([[0.01, -0.005, -0.005, 0.0, 0.0, 0.0]])
    extra = {
        "eps35": np.zeros((n, 6)),
        "sigair35": np.zeros(n),
        "edot35": np.zeros(n),
        "rho": np.full(n, mat.rho0),
    }
    # Initial step to apply strain
    dt0 = 1e-6
    sig, _ = law35_kelvinmax.solid_update(mat, sig, deps0, dt=dt0, extra=extra)
    s_initial = (sig[0, 0] - sig[0, 1])

    # Hold strain constant (deps = 0) and relax for time t_relax
    t_relax = tau
    n_steps = 1000
    dt_step = t_relax / n_steps
    for _ in range(n_steps):
        sig, _ = law35_kelvinmax.solid_update(mat, sig, np.zeros((n, 6)), dt=dt_step, extra=extra)

    s_relaxed = (sig[0, 0] - sig[0, 1])
    s_inf = (G2 * Gt2 / (G2 + Gt2)) * (deps0[0, 0] - deps0[0, 1])

    # Standard linear solid: s(t) - s_inf = (s(0) - s_inf) * exp(-t/tau)
    # At t = tau, ratio should be 1/e
    ratio_measured = (s_relaxed - s_inf) / (s_initial - s_inf)
    ratio_analytical = math.exp(-1.0)
    assert pytest.approx(ratio_analytical, rel=1e-2) == ratio_measured


def test_law35_asymptotic_stress_under_constant_strain():
    E = 10.0
    nu = 0.25
    Et = 2.0
    nut = 0.25
    mu = 1.0
    mat = _make_law35_mat(MAT_E=E, MAT_NU=nu, MAT_ETAN=Et, MAT_NUt=nut, MAT_ETA2=mu)
    G2 = E / (1.0 + nu)
    Gt2 = Et / (1.0 + nut)

    n = 1
    sig = np.zeros((n, 6))
    deps0 = np.array([[0.02, -0.01, -0.01, 0.0, 0.0, 0.0]])
    extra = {
        "eps35": np.zeros((n, 6)),
        "sigair35": np.zeros(n),
        "edot35": np.zeros(n),
        "rho": np.full(n, mat.rho0),
    }
    sig, _ = law35_kelvinmax.solid_update(mat, sig, deps0, dt=1e-6, extra=extra)

    # Relax for a very long time (100 * tau)
    tau = 2.0 * mu / (G2 + Gt2)
    dt_step = tau * 0.1
    for _ in range(1000):
        sig, _ = law35_kelvinmax.solid_update(mat, sig, np.zeros((n, 6)), dt=dt_step, extra=extra)

    dev_sig = sig[0, 0] - (sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
    dev_eps = deps0[0, 0] - (deps0[0, 0] + deps0[0, 1] + deps0[0, 2]) / 3.0
    s_inf_expected = (G2 * Gt2 / (G2 + Gt2)) * dev_eps
    assert pytest.approx(s_inf_expected, rel=1e-3) == dev_sig


def test_law35_crank_nicolson_midstep_accuracy():
    # Comparing single large step vs multiple small steps under strain hold
    mat = _make_law35_mat(MAT_E=15.0, MAT_NU=0.2, MAT_ETAN=3.0, MAT_NUt=0.2, MAT_ETA2=5.0)
    sig1 = np.zeros((1, 6))
    deps0 = np.array([[0.01, -0.005, -0.005, 0.0, 0.0, 0.0]])
    extra1 = {"eps35": np.zeros((1, 6)), "sigair35": np.zeros(1), "edot35": np.zeros(1), "rho": np.full(1, mat.rho0)}
    sig1, _ = law35_kelvinmax.solid_update(mat, sig1, deps0, dt=1e-6, extra=extra1)

    sig2 = sig1.copy()
    extra2 = {"eps35": extra1["eps35"].copy(), "sigair35": extra1["sigair35"].copy(),
              "edot35": extra1["edot35"].copy(), "rho": extra1["rho"].copy()}

    T = 0.5
    # 1 step of T
    sig1, _ = law35_kelvinmax.solid_update(mat, sig1, np.zeros((1, 6)), dt=T, extra=extra1)
    # 500 steps of T/500
    for _ in range(500):
        sig2, _ = law35_kelvinmax.solid_update(mat, sig2, np.zeros((1, 6)), dt=T/500, extra=extra2)

    # Crank-Nicolson is 2nd-order accurate in time; single large step is within a few percent
    assert pytest.approx(sig2[0, 0], rel=0.05) == sig1[0, 0]


# ----------------------------------------------------------------------------
# 14-16. Volumetric visco-elastic response and tabulated curves
# ----------------------------------------------------------------------------

def test_law35_volumetric_visco_elastic_c1_c2_c3():
    # Pure volumetric compression: deps_11 = deps_22 = deps_33 = -0.001
    mat = _make_law35_mat(MAT_CO1=1.0, MAT_CO2=1.0, MAT_CO3=0.5, MAT_ETA1=2.0)
    sig = np.zeros((1, 6))
    deps = np.full((1, 6), 0.0)
    deps[0, 0] = deps[0, 1] = deps[0, 2] = -0.001
    extra = {"eps35": np.zeros((1, 6)), "sigair35": np.zeros(1), "edot35": np.zeros(1), "rho": np.full(1, mat.rho0)}

    sig, c = law35_kelvinmax.solid_update(mat, sig, deps, dt=1e-4, extra=extra)
    # Mean stress pr is negative for compression (Cauchy stress convention: tension positive, compression negative)
    p_mean = (sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
    assert p_mean < 0.0
    assert abs(p_mean) > 0.0
    assert c[0] > 0.0


def test_law35_tabulated_pressure_curve_itype0():
    # Itype=0: pure curve lookup pr = -fscale * f(mu_v)
    f_x = np.array([-0.2, 0.0, 0.2, 0.5])
    f_y = np.array([-10.0, 0.0, 20.0, 60.0])
    func = _DummyFunction(f_x, f_y)
    model = _DummyModel({10: func})

    mat = _make_law35_mat(FUN_A1=10, IFscale=1.5, Itype=0)
    law35_kelvinmax.resolve(mat, model, None)

    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    # Relative compression: rho = 1.1 * rho0 -> amu = 0.1
    # f(0.1) should be 10.0 (halfway between 0 and 0.2 with y=20)
    # pr = -fscale * f(0.1) = -1.5 * 10 = -15.0
    extra = {
        "eps35": np.zeros((1, 6)),
        "sigair35": np.zeros(1),
        "edot35": np.zeros(1),
        "rho": np.array([1.1 * mat.rho0]),
    }
    sig, c = law35_kelvinmax.solid_update(mat, sig, deps, dt=1e-4, extra=extra)
    # Since pr = -15.0 and pmin = -1e20, pr is -15.0
    p_mean = (sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
    assert pytest.approx(-15.0, rel=1e-3) == p_mean


def test_law35_tabulated_pressure_curve_itype1():
    # Itype=1: volumetric standard solid PLUS curve
    f_x = np.array([-0.2, 0.0, 0.2])
    f_y = np.array([-5.0, 0.0, 15.0])
    func = _DummyFunction(f_x, f_y)
    model = _DummyModel({12: func})

    mat = _make_law35_mat(FUN_A1=12, IFscale=1.0, Itype=1, MAT_CO1=1.0)
    law35_kelvinmax.resolve(mat, model, None)

    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    deps[0, 0] = deps[0, 1] = deps[0, 2] = -0.001
    extra = {
        "eps35": np.zeros((1, 6)),
        "sigair35": np.zeros(1),
        "edot35": np.zeros(1),
        "rho": np.array([1.1 * mat.rho0]),
    }
    sig, c = law35_kelvinmax.solid_update(mat, sig, deps, dt=1e-4, extra=extra)
    p_mean = (sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
    # Both volumetric rate term and curve contribute
    assert p_mean != 0.0


# ----------------------------------------------------------------------------
# 17-18. Closed-cell air pressure and tension floor
# ----------------------------------------------------------------------------

def test_law35_closed_cell_air_pressure():
    p0 = 0.1013  # 1 atm
    phi = 0.1
    mat = _make_law35_mat(MAT_P0=p0, MAT_PHI=phi, MAT_GAMA0=0.0)
    sig = np.zeros((1, 6))
    deps = np.zeros((1, 6))
    # Compressed: V/V0 = rho0/rho = 0.8 -> gamma = 0.8 - 1 = -0.2
    # sigair = max(0, -P0 * gamma / (1 + gamma - phi)) = -(0.1013 * (-0.2)) / (1 - 0.2 - 0.1)
    # = 0.02026 / 0.7 = 0.02894
    extra = {
        "eps35": np.zeros((1, 6)),
        "sigair35": np.zeros(1),
        "edot35": np.zeros(1),
        "rho": np.array([mat.rho0 / 0.8]),
    }
    sig, c = law35_kelvinmax.solid_update(mat, sig, deps, dt=1e-4, extra=extra)
    expected_air = -(p0 * (-0.2)) / (1.0 - 0.2 - phi)
    assert pytest.approx(expected_air, rel=1e-3) == extra["sigair35"][0]


def test_law35_pressure_tension_floor_pmin():
    pmin = -5.0
    mat = _make_law35_mat(MAT_PC=pmin)
    sig = np.zeros((1, 6))
    # Huge tensile strain
    deps = np.full((1, 6), 0.0)
    deps[0, 0] = deps[0, 1] = deps[0, 2] = 0.1
    extra = {"eps35": np.zeros((1, 6)), "sigair35": np.zeros(1), "edot35": np.zeros(1), "rho": np.full(1, mat.rho0)}
    sig, c = law35_kelvinmax.solid_update(mat, sig, deps, dt=1e-4, extra=extra)
    p_mean = (sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
    assert p_mean >= pmin


# ----------------------------------------------------------------------------
# 19-21. Modulus stiffening: rate, volume ratio, and filtering
# ----------------------------------------------------------------------------

def test_law35_strain_rate_stiffening():
    E = 10.0
    E1 = 2.0
    E2 = 1.0
    mat = _make_law35_mat(MAT_E=E, MAT_E1=E1, MAT_E2=E2, MAT_ETA2=100.0)
    sig = np.zeros((1, 6))
    dt = 1e-4
    # edot = deps / dt = 0.01 / 1e-4 = 100.0
    # E_stiff = E1 * 100 + E2 = 201.0 > E (10.0)
    deps = np.array([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0]])
    extra = {"eps35": np.zeros((1, 6)), "sigair35": np.zeros(1), "edot35": np.zeros(1), "rho": np.full(1, mat.rho0)}
    sig, c = law35_kelvinmax.solid_update(mat, sig, deps, dt=dt, extra=extra)
    # Shear modulus should reflect 201.0 instead of 10.0
    assert c[0] > math.sqrt(E / mat.rho0)


def test_law35_relative_volume_stiffening():
    E = 10.0
    N = 1.5
    mat = _make_law35_mat(MAT_E=E, MAT_N=N, MAT_ETA2=100.0)
    # Compressed: relvol = V/V0 = 0.5 -> E_new = E / (0.5)^1.5 = 10 / 0.3535 = 28.28
    sig = np.zeros((1, 6))
    deps = np.array([[0.001, -0.0005, -0.0005, 0.0, 0.0, 0.0]])
    extra = {"eps35": np.zeros((1, 6)), "sigair35": np.zeros(1), "edot35": np.zeros(1),
             "rho": np.array([mat.rho0 / 0.5])}
    sig, c = law35_kelvinmax.solid_update(mat, sig, deps, dt=1e-4, extra=extra)
    # Sound speed is substantially higher than at relvol = 1
    c_uncompressed = math.sqrt((E / (1.0 + mat.params["nu"]) * (2.0/3.0) + (E / (1.0 - 2.0*mat.params["nu"]))/3.0) / mat.rho0)
    assert c[0] > c_uncompressed


def test_law35_rate_filtering_fcut():
    mat = _make_law35_mat(Fcut=500.0)
    assert mat.params["ismooth"] == 1
    sig = np.zeros((1, 6))
    deps = np.array([[0.001, 0.0, 0.0, 0.0, 0.0, 0.0]])
    dt = 1e-4
    rate_raw = 0.001 / dt  # 10.0
    extra = {"eps35": np.zeros((1, 6)), "sigair35": np.zeros(1), "edot35": np.zeros(1), "rho": np.full(1, mat.rho0)}
    law35_kelvinmax.solid_update(mat, sig, deps, dt=dt, extra=extra)
    # alpha = min(1, 2*pi*500 * 1e-4) = 0.314159
    # edot = alpha * 10.0 + (1-alpha)*0 = 3.14159
    alpha = min(1.0, 2.0 * math.pi * 500.0 * dt)
    assert pytest.approx(alpha * rate_raw, rel=1e-3) == extra["edot35"][0]


# ----------------------------------------------------------------------------
# 22-24. Sound speed branches
# ----------------------------------------------------------------------------

def test_law35_sound_speed_branches():
    # Branch 1: kf = 0
    mat1 = _make_law35_mat()
    sig1 = np.zeros((1, 6))
    deps1 = np.zeros((1, 6))
    _, c1 = law35_kelvinmax.solid_update(mat1, sig1, deps1, dt=1e-5)
    assert c1[0] > 0.0

    # Branch 2: kf > 0, Itype = 0
    func = _DummyFunction([-0.1, 0.1], [-10.0, 10.0])
    model = _DummyModel({20: func})
    mat2 = _make_law35_mat(FUN_A1=20, Itype=0)
    law35_kelvinmax.resolve(mat2, model, None)
    sig2 = np.zeros((1, 6))
    deps2 = np.zeros((1, 6))
    _, c2 = law35_kelvinmax.solid_update(mat2, sig2, deps2, dt=1e-5)
    assert c2[0] > 0.0

    # Branch 3: kf > 0, Itype = 1
    mat3 = _make_law35_mat(FUN_A1=20, Itype=1)
    law35_kelvinmax.resolve(mat3, model, None)
    sig3 = np.zeros((1, 6))
    deps3 = np.zeros((1, 6))
    _, c3 = law35_kelvinmax.solid_update(mat3, sig3, deps3, dt=1e-5)
    assert c3[0] > 0.0


# ----------------------------------------------------------------------------
# 25. Spatial objectivity under 3D rotation
# ----------------------------------------------------------------------------

def test_law35_spatial_objectivity_3d_rotation():
    mat = _make_law35_mat(MAT_E=15.0, MAT_NU=0.25, MAT_ETAN=3.0, MAT_NUt=0.25, MAT_ETA2=4.0)

    # 45-degree rotation about z-axis
    theta = math.pi / 4.0
    c, s = math.cos(theta), math.sin(theta)
    Q = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])

    # Unrotated strain increment
    deps_mat = np.array([
        [0.01, 0.002, 0.0],
        [0.002, -0.005, 0.0],
        [0.0, 0.0, -0.003],
    ])
    deps1 = np.array([[deps_mat[0, 0], deps_mat[1, 1], deps_mat[2, 2],
                       2.0 * deps_mat[0, 1], 2.0 * deps_mat[1, 2], 2.0 * deps_mat[2, 0]]])

    # Rotated strain increment
    deps_mat_rot = Q @ deps_mat @ Q.T
    deps2 = np.array([[deps_mat_rot[0, 0], deps_mat_rot[1, 1], deps_mat_rot[2, 2],
                       2.0 * deps_mat_rot[0, 1], 2.0 * deps_mat_rot[1, 2], 2.0 * deps_mat_rot[2, 0]]])

    sig1 = np.zeros((1, 6))
    sig2 = np.zeros((1, 6))
    dt = 1e-4

    extra1 = {"eps35": np.zeros((1, 6)), "sigair35": np.zeros(1), "edot35": np.zeros(1), "rho": np.full(1, mat.rho0)}
    extra2 = {"eps35": np.zeros((1, 6)), "sigair35": np.zeros(1), "edot35": np.zeros(1), "rho": np.full(1, mat.rho0)}

    law35_kelvinmax.solid_update(mat, sig1, deps1, dt=dt, extra=extra1)
    law35_kelvinmax.solid_update(mat, sig2, deps2, dt=dt, extra=extra2)

    sig1_mat = np.array([
        [sig1[0, 0], sig1[0, 3], sig1[0, 5]],
        [sig1[0, 3], sig1[0, 1], sig1[0, 4]],
        [sig1[0, 5], sig1[0, 4], sig1[0, 2]],
    ])
    sig1_mat_rot = Q @ sig1_mat @ Q.T

    sig2_mat = np.array([
        [sig2[0, 0], sig2[0, 3], sig2[0, 5]],
        [sig2[0, 3], sig2[0, 1], sig2[0, 4]],
        [sig2[0, 5], sig2[0, 4], sig2[0, 2]],
    ])
    np.testing.assert_allclose(sig1_mat_rot, sig2_mat, atol=1e-6)


# ----------------------------------------------------------------------------
# 26-28. Consistent solid tangent tests
# ----------------------------------------------------------------------------

def test_law35_consistent_solid_tangent_symmetry_and_positive_definite():
    mat = _make_law35_mat(MAT_E=20.0, MAT_NU=0.3, MAT_ETAN=5.0, MAT_NUt=0.25, MAT_ETA2=4.0)
    sig = np.zeros((2, 6))
    C = law35_kelvinmax.consistent_solid_tangent(mat, sig, dt=1e-4)
    assert C.shape == (2, 6, 6)
    for k in range(2):
        Ck = C[k]
        np.testing.assert_allclose(Ck, Ck.T, atol=1e-12)
        evals = np.linalg.eigvalsh(Ck)
        assert np.all(evals > 0)


def test_law35_consistent_solid_tangent_air_and_curve():
    func = _DummyFunction([-0.1, 0.1], [-10.0, 10.0])
    model = _DummyModel({30: func})
    mat = _make_law35_mat(FUN_A1=30, Itype=1, MAT_P0=0.5, MAT_PHI=0.1)
    law35_kelvinmax.resolve(mat, model, None)

    sig = np.zeros((1, 6))
    extra = {"rho": np.array([1.1 * mat.rho0])}
    C = law35_kelvinmax.consistent_solid_tangent(mat, sig, extra=extra, dt=1e-4)
    assert C.shape == (1, 6, 6)
    # Bulk modulus includes air and curve terms
    assert C[0, 0, 0] > mat.params["K"] + (4.0 / 3.0) * mat.params["G"]


def test_law35_solid_tangent_dispatch():
    mat = _make_law35_mat()
    sig = np.zeros((3, 6))
    C_direct = law35_kelvinmax.consistent_solid_tangent(mat, sig)
    C_dispatched = materials.solid_tangent(mat, sig, None, None)
    np.testing.assert_allclose(C_direct, C_dispatched, atol=1e-12)


# ----------------------------------------------------------------------------
# 29. Batch equivalence
# ----------------------------------------------------------------------------

def test_law35_batch_equivalence():
    mat = _make_law35_mat(MAT_E=15.0, MAT_NU=0.2, MAT_ETAN=3.0, MAT_NUt=0.2, MAT_ETA2=5.0)
    n = 5
    sig_batch = np.zeros((n, 6))
    np.random.seed(42)
    deps_batch = np.random.uniform(-0.01, 0.01, size=(n, 6))
    dt = 1e-4

    extra_batch = {
        "eps35": np.zeros((n, 6)),
        "sigair35": np.zeros(n),
        "edot35": np.zeros(n),
        "rho": np.full(n, mat.rho0),
    }

    sig_res_batch, c_batch = law35_kelvinmax.solid_update(
        mat, sig_batch.copy(), deps_batch.copy(), dt=dt, extra=extra_batch
    )

    for i in range(n):
        sig_single = np.zeros((1, 6))
        deps_single = deps_batch[i:i+1].copy()
        extra_single = {
            "eps35": np.zeros((1, 6)),
            "sigair35": np.zeros(1),
            "edot35": np.zeros(1),
            "rho": np.full(1, mat.rho0),
        }
        sig_res_single, c_single = law35_kelvinmax.solid_update(
            mat, sig_single, deps_single, dt=dt, extra=extra_single
        )
        np.testing.assert_allclose(sig_res_single[0], sig_res_batch[i], atol=1e-12)
        np.testing.assert_allclose(c_single[0], c_batch[i], atol=1e-12)


# ----------------------------------------------------------------------------
# 30. Cycle 0 / static stability (dt <= 0)
# ----------------------------------------------------------------------------

def test_law35_cycle0_static_stability():
    mat = _make_law35_mat()
    sig = np.array([[10.0, 5.0, 5.0, 1.0, 0.0, 0.0]])
    deps = np.zeros((1, 6))

    # dt = 0.0
    sig_out0, c0 = law35_kelvinmax.solid_update(mat, sig.copy(), deps, dt=0.0)
    np.testing.assert_allclose(sig, sig_out0, atol=1e-12)
    assert c0[0] > 0.0

    # dt < 0.0
    sig_out_neg, c_neg = law35_kelvinmax.solid_update(mat, sig.copy(), deps, dt=-1.0)
    np.testing.assert_allclose(sig, sig_out_neg, atol=1e-12)
    assert c_neg[0] > 0.0


# ----------------------------------------------------------------------------
# 31. Starter checks and extra shapes integration
# ----------------------------------------------------------------------------

def test_law35_starter_checks_and_extra_shapes():
    mat = _make_law35_mat()
    assert 35 in _ALLOWED_LAWS["bricks"]
    assert 35 in _ALLOWED_LAWS["tetras"]
    assert 35 not in _ALLOWED_LAWS["shells"]
    assert materials.needs_env(mat) is True
    assert materials.needs_defgrad(mat) is False

    shapes = materials.extra_shapes(mat)
    assert "eps35" in shapes and shapes["eps35"] == (6,)
    assert "sigair35" in shapes and shapes["sigair35"] == ()
    assert "edot35" in shapes and shapes["edot35"] == ()
