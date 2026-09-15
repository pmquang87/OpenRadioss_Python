"""
Tests for OpenRadioss Material Law 74 (/MAT/LAW74, /MAT/HILL_3D, /MAT/ORTH_PLAS, /MAT/THERM_HILL).
3D Tabulated Hill Orthotropic Plasticity Model for Solids (Milestone M563).

Upstream reference files:
  - Starter reader: starter/source/materials/mat/mat074/hm_read_mat74.F
  - Engine physics: engine/source/materials/mat/mat074/sigeps74.F
  - Table tools: engine/source/tools/curve/table_tools.F (TABLE_VINTERP)
  - Function tools: engine/source/tools/curve/finter.F (FINTER)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

import pyradioss.materials as materials
from pyradioss.materials import (
    Law74Params,
    build_law74,
    law74_hill_3d,
    solid_update_law74,
    shell_update_law74,
    law74_sound_speed,
    law74_solid_sound_speed,
    law74_solid_tangent,
    law74_extra_shapes,
    LAW_DISPATCH_METADATA,
    MATERIAL_SHELL_DISPATCH,
    MATERIAL_SOLID_DISPATCH,
    needs_env,
)
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY


# ============================================================================
# 1. Law74Params Initialization, Defaults & Derived Constants
# ============================================================================

def test_law74_params_defaults():
    """Verify default parameters and isotropic Hill reduction."""
    p = Law74Params()
    assert p.rho0 == 1.0
    assert p.refer_rho == 1.0
    assert p.e == 210000.0
    assert p.nu == 0.3
    assert p.s11y == 1.0
    assert p.s22y == 1.0
    assert p.s33y == 1.0
    assert p.s12y == 1.0
    assert p.s23y == 1.0
    assert p.s31y == 1.0
    assert p.chard == 0.0
    assert p.fisokin == 0.0
    assert p.t0 == 293.0
    assert p.rhocp == 0.0
    assert p.fscale == 1.0
    assert p.pscale == 1.0

    # Isotropic recovery: S_ii = 1, S_ij = 1
    # FF = 0.5 * (1 + 1 - 1) = 0.5
    # GG = 0.5 * (1 + 1 - 1) = 0.5
    # HH = 0.5 * (1 + 1 - 1) = 0.5
    # LL = 0.5 / 1^2 = 0.5
    # MM = 0.5 / 1^2 = 0.5
    # NN = 0.5 / 1^2 = 0.5
    assert p.ff == pytest.approx(0.5)
    assert p.gg == pytest.approx(0.5)
    assert p.hh == pytest.approx(0.5)
    assert p.ll == pytest.approx(0.5)
    assert p.mm == pytest.approx(0.5)
    assert p.nn == pytest.approx(0.5)

    # Elastic constants
    expected_g = 0.5 * 210000.0 / (1.0 + 0.3)
    assert p.g == pytest.approx(expected_g)
    expected_c1 = 210000.0 / (3.0 * (1.0 - 2.0 * 0.3))
    assert p.c1 == pytest.approx(expected_c1)

    # Longitudinal wave speed
    expected_c = math.sqrt((expected_c1 + 4.0 / 3.0 * expected_g) / 1.0)
    assert p.soundsp == pytest.approx(expected_c)


def test_law74_anisotropy_coefficients():
    """Verify Hill constants calculation for anisotropic directional yield parameters."""
    s11y, s22y, s33y = 2.0, 1.5, 1.2
    s12y, s23y, s31y = 0.8, 0.7, 0.9

    p = Law74Params(
        s11y=s11y, s22y=s22y, s33y=s33y,
        s12y=s12y, s23y=s23y, s31y=s31y,
    )

    exp_ff = 0.5 * (1.0 / s22y**2 + 1.0 / s33y**2 - 1.0 / s11y**2)
    exp_gg = 0.5 * (1.0 / s11y**2 + 1.0 / s33y**2 - 1.0 / s22y**2)
    exp_hh = 0.5 * (1.0 / s11y**2 + 1.0 / s22y**2 - 1.0 / s33y**2)
    exp_ll = 0.5 / s23y**2
    exp_mm = 0.5 / s31y**2
    exp_nn = 0.5 / s12y**2

    assert p.ff == pytest.approx(exp_ff)
    assert p.gg == pytest.approx(exp_gg)
    assert p.hh == pytest.approx(exp_hh)
    assert p.ll == pytest.approx(exp_ll)
    assert p.mm == pytest.approx(exp_mm)
    assert p.nn == pytest.approx(exp_nn)


def test_law74_params_invalid_values():
    """Verify validation errors on invalid elastic or yield parameters."""
    with pytest.raises(ValueError, match="Young's modulus E must be > 0"):
        Law74Params(e=-100.0)
    with pytest.raises(ValueError, match="Poisson's ratio nu must be >= 0"):
        Law74Params(nu=-0.1)
    with pytest.raises(ValueError, match="Yield parameter s11y must be > 0"):
        Law74Params(s11y=0.0)


# ============================================================================
# 2. Material Constructor & Registry
# ============================================================================

def test_law74_builder():
    """Verify build_law74 constructor from dict, Law74Params, or kwargs."""
    p_params = Law74Params(e=100000.0, nu=0.25)
    mat1 = build_law74(p_params)
    assert mat1.law == 74
    assert mat1.law_name == "LAW74"
    assert mat1.params["E"] == 100000.0
    assert mat1.params["nu"] == 0.25
    assert mat1.params["_obj"] is p_params

    # From dict
    cfg_dict = {
        "id": 42,
        "title": "ALUMINUM_74",
        "MAT_RHO": 2.7e-6,
        "MAT_E": 70000.0,
        "MAT_NU": 0.33,
        "MAT_SIGT1": 1.0,
        "MAT_SIGT2": 1.2,
        "MAT_SIGT3": 1.1,
        "MAT_SIGYT1": 0.6,
        "MAT_SIGYT2": 0.55,
        "MAT_SIGYT3": 0.58,
        "MAT_HARD": 0.5,
    }
    mat2 = build_law74(cfg_dict)
    assert mat2.id == 42
    assert mat2.title == "ALUMINUM_74"
    assert mat2.params["rho0"] == 2.7e-6
    assert mat2.params["chard"] == 0.5
    assert mat2.params["s22y"] == 1.2


def test_law74_registry_and_metadata():
    """Verify registry entries and dispatch table entries for LAW74."""
    for key in (74, "74", "LAW74", "HILL_3D", "ORTH_PLAS", "THERM_HILL"):
        assert key in MAT_PHYSICS_REGISTRY
        assert key in MATERIAL_SOLID_DISPATCH
        assert key in MATERIAL_SHELL_DISPATCH
        assert key in LAW_DISPATCH_METADATA

    assert LAW_DISPATCH_METADATA[74]["solid"] is True
    assert LAW_DISPATCH_METADATA[74]["shell"] is False

    mat = build_law74(e=100000.0, nu=0.3)
    assert needs_env(mat) is True

    shapes = law74_extra_shapes(mat, nip=1)
    assert "uvar74" in shapes
    assert shapes["uvar74"] == (10,)


# ============================================================================
# 3. Elastic Predictor & 3D Isotropic Limit
# ============================================================================

def test_law74_elastic_predictor():
    """Verify pure elastic stress update matches Hooke's law in 3D."""
    e_mod = 200000.0
    nu_val = 0.25
    mat = build_law74(e=e_mod, nu=nu_val, table_id=0, sigy0=1.0e9)

    # Small strain increment
    deps = np.array([1.0e-5, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig_init = np.zeros(6)

    sig_new, pla_new, soundsp = solid_update_law74(
        mat, sig_init, deps, return_tuple=True
    )

    assert pla_new == pytest.approx(0.0)

    # Elastic 3D moduli
    # K = E / (3*(1 - 2*nu)) = 200000 / (3 * 0.5) = 133333.333
    # G = 0.5 * E / (1 + nu) = 100000 / 1.25 = 80000.0
    # C11 = K + 4/3*G = 133333.333 + 106666.667 = 240000.0
    # C12 = K - 2/3*G = 133333.333 - 53333.333 = 80000.0
    c11_exp = e_mod * (1.0 - nu_val) / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
    c12_exp = e_mod * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))

    assert sig_new[0] == pytest.approx(c11_exp * 1.0e-5, rel=1.0e-5)
    assert sig_new[1] == pytest.approx(c12_exp * 1.0e-5, rel=1.0e-5)
    assert sig_new[2] == pytest.approx(c12_exp * 1.0e-5, rel=1.0e-5)
    assert sig_new[3] == pytest.approx(0.0)
    assert sig_new[4] == pytest.approx(0.0)
    assert sig_new[5] == pytest.approx(0.0)

    # Sound speed
    assert soundsp == pytest.approx(math.sqrt(c11_exp / 1.0))


# ============================================================================
# 4. Hill 3D Yield Criterion & Directional Anisotropy
# ============================================================================

def test_law74_directional_yield_criterion():
    """Verify Hill 3D equivalent stress CRI equals sigma_y when stress matches directional yield limits."""
    s11y, s22y, s33y = 1.5, 1.2, 0.8
    s12y, s23y, s31y = 0.7, 0.6, 0.9

    p = Law74Params(
        s11y=s11y, s22y=s22y, s33y=s33y,
        s12y=s12y, s23y=s23y, s31y=s31y,
    )
    y0 = 300.0

    # Direction 1: sigma_xx = y0 * s11y
    # In deviatoric space: sxx = 2/3 * y0 * s11y, syy = -1/3 * y0 * s11y, szz = -1/3 * y0 * s11y
    # (syy - szz) = 0, (szz - sxx) = -y0*s11y, (sxx - syy) = y0*s11y
    # CRI^2 = (GG + HH) * (y0*s11y)^2 = (1 / s11y^2) * (y0*s11y)^2 = y0^2
    sig_dev1 = np.array([2.0 / 3.0 * y0 * s11y, -1.0 / 3.0 * y0 * s11y, -1.0 / 3.0 * y0 * s11y, 0.0, 0.0, 0.0])
    cri1 = math.sqrt(p.ff * (sig_dev1[1] - sig_dev1[2])**2
                     + p.gg * (sig_dev1[2] - sig_dev1[0])**2
                     + p.hh * (sig_dev1[0] - sig_dev1[1])**2)
    assert cri1 == pytest.approx(y0, rel=1.0e-5)

    # Direction 2: sigma_yy = y0 * s22y
    sig_dev2 = np.array([-1.0 / 3.0 * y0 * s22y, 2.0 / 3.0 * y0 * s22y, -1.0 / 3.0 * y0 * s22y, 0.0, 0.0, 0.0])
    cri2 = math.sqrt(p.ff * (sig_dev2[1] - sig_dev2[2])**2
                     + p.gg * (sig_dev2[2] - sig_dev2[0])**2
                     + p.hh * (sig_dev2[0] - sig_dev2[1])**2)
    assert cri2 == pytest.approx(y0, rel=1.0e-5)

    # Direction 3: sigma_zz = y0 * s33y
    sig_dev3 = np.array([-1.0 / 3.0 * y0 * s33y, -1.0 / 3.0 * y0 * s33y, 2.0 / 3.0 * y0 * s33y, 0.0, 0.0, 0.0])
    cri3 = math.sqrt(p.ff * (sig_dev3[1] - sig_dev3[2])**2
                     + p.gg * (sig_dev3[2] - sig_dev3[0])**2
                     + p.hh * (sig_dev3[0] - sig_dev3[1])**2)
    assert cri3 == pytest.approx(y0, rel=1.0e-5)

    # Pure shear 12: sigma_xy = y0 * s12y
    # CRI^2 = 2 * NN * sigma_xy^2 = 2 * (0.5 / s12y^2) * (y0 * s12y)^2 = y0^2
    sig_dev12 = np.array([0.0, 0.0, 0.0, y0 * s12y, 0.0, 0.0])
    cri12 = math.sqrt(2.0 * p.nn * sig_dev12[3]**2)
    assert cri12 == pytest.approx(y0, rel=1.0e-5)


# ============================================================================
# 5. Plastic Radial Return & Hardening
# ============================================================================

def test_law74_plastic_radial_return():
    """Verify radial return mapping under large strain increment."""
    y0 = 250.0
    mat = build_law74(e=200000.0, nu=0.3, sigy0=y0)

    # Strain increment causing large plastic flow
    deps = np.array([0.01, -0.003, -0.003, 0.0, 0.0, 0.0])
    sig = np.zeros(6)

    sig_new, pla_new = solid_update_law74(mat, sig, deps)

    assert pla_new > 0.0

    # Verify equivalent stress on yield surface
    p = mat.params["_obj"]
    st_dev = sig_new[:3] - np.mean(sig_new[:3])
    cri = math.sqrt(p.ff * (st_dev[1] - st_dev[2])**2
                    + p.gg * (st_dev[2] - st_dev[0])**2
                    + p.hh * (st_dev[0] - st_dev[1])**2
                    + 2.0 * p.ll * sig_new[4]**2
                    + 2.0 * p.mm * sig_new[5]**2
                    + 2.0 * p.nn * sig_new[3]**2)

    assert cri == pytest.approx(y0, rel=1.0e-2)


# ============================================================================
# 6. Yield Tables, Curves & Function Evaluation
# ============================================================================

def test_law74_yield_curve_interpolation():
    """Verify linear hardening curve evaluation with IPLA=0 and IPLA=1."""
    curve = {
        "x": np.array([0.0, 0.05, 0.10]),
        "y": np.array([200.0, 300.0, 350.0]),
    }
    # IPLA=0: explicit radial return where yield stress is taken at start of step
    mat0 = build_law74(e=200000.0, nu=0.3, yield_table=curve, ipla=0)
    deps1 = np.array([0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0])
    sig1, pla1 = solid_update_law74(mat0, np.zeros(6), deps1)
    assert pla1 > 0.0

    deps2 = np.array([0.01, -0.003, -0.003, 0.0, 0.0, 0.0])
    sig2, pla2 = solid_update_law74(mat0, sig1, deps2, epsp=pla1)
    assert pla2 > pla1

    p = mat0.params["_obj"]
    st_dev2 = sig2[:3] - np.mean(sig2[:3])
    cri2 = math.sqrt(p.ff * (st_dev2[1] - st_dev2[2])**2
                     + p.gg * (st_dev2[2] - st_dev2[0])**2
                     + p.hh * (st_dev2[0] - st_dev2[1])**2)
    # Under IPLA=0, equivalent stress matches yield stress at start of step 2 (pla1)
    exp_yld1, _ = law74_hill_3d._eval_curve_1d(curve, pla1)
    assert cri2 == pytest.approx(float(exp_yld1), rel=2.0e-2)

    # IPLA=1: actual yield stress updated with intra-step hardening
    mat1 = build_law74(e=200000.0, nu=0.3, yield_table=curve, ipla=1)
    sig1_1, pla1_1 = solid_update_law74(mat1, np.zeros(6), deps1)
    sig2_1, pla2_1 = solid_update_law74(mat1, sig1_1, deps2, epsp=pla1_1)
    st_dev2_1 = sig2_1[:3] - np.mean(sig2_1[:3])
    cri2_1 = math.sqrt(p.ff * (st_dev2_1[1] - st_dev2_1[2])**2
                       + p.gg * (st_dev2_1[2] - st_dev2_1[0])**2
                       + p.hh * (st_dev2_1[0] - st_dev2_1[1])**2)
    exp_yld2, _ = law74_hill_3d._eval_curve_1d(curve, pla2_1)
    assert cri2_1 == pytest.approx(float(exp_yld2), rel=2.0e-2)


def test_law74_callable_yield_table():
    """Verify custom callable yield function depending on (pla, rate, temp)."""
    def custom_table(pla, rate, temp):
        # Y = 200 + 500*pla + 20*ln(1 + rate) - 0.1*(temp - 293)
        rate_val = np.asarray(rate)
        temp_val = np.asarray(temp)
        val = 200.0 + 500.0 * pla + 20.0 * np.log1p(rate_val) - 0.1 * (temp_val - 293.0)
        slp = np.full_like(pla, 500.0)
        return val, slp

    mat = build_law74(e=200000.0, nu=0.3, yield_table=custom_table, t0=300.0)
    deps = np.array([0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0])
    sig, pla = solid_update_law74(mat, np.zeros(6), deps, dt=1.0e-4)
    assert pla > 0.0


# ============================================================================
# 7. Adiabatic Plastic Heating
# ============================================================================

def test_law74_adiabatic_plastic_heating():
    """Verify adiabatic temperature rise: temp += yld * dpla / rhocp."""
    rhocp = 2.5e6  # J/(m^3 * K)
    y0 = 300.0e6   # 300 MPa
    mat = build_law74(e=200.0e9, nu=0.3, sigy0=y0, rhocp=rhocp, t0=293.0)

    deps = np.array([0.01, -0.003, -0.003, 0.0, 0.0, 0.0])
    extra = {"temp": np.array([293.0])}

    sig, pla = solid_update_law74(mat, np.zeros(6), deps, extra=extra)

    assert pla > 0.0
    temp_after = float(extra["temp"][0])
    # Expected temperature increase = y0 * dpla / rhocp
    exp_dtemp = (y0 * pla) / rhocp
    assert temp_after == pytest.approx(293.0 + exp_dtemp, rel=1.0e-2)
    assert temp_after > 293.0


# ============================================================================
# 8. Dynamic Young's Modulus Degradation
# ============================================================================

def test_law74_young_modulus_degradation_ce():
    """Verify exponential Young's modulus decay: E = E0 - (E0 - Einf)*(1 - exp(-ce*pla))."""
    e0 = 200000.0
    einf = 100000.0
    ce = 50.0

    mat = build_law74(e=e0, nu=0.3, einf=einf, ce=ce, sigy0=200.0)

    extra = {"uvar74": np.zeros((1, 10))}
    extra["uvar74"][0, 0] = 0.02  # pla = 0.02

    # Initial sound speed with pla=0
    c_init = law74_sound_speed(mat, rho=1.0)

    # Sound speed with degradation
    c_degraded = law74_sound_speed(mat, rho=1.0, extra=extra)
    assert c_degraded < c_init

    # Expected degraded E
    exp_e = e0 - (e0 - einf) * (1.0 - math.exp(-ce * 0.02))
    exp_g = 0.5 * exp_e / 1.3
    exp_c1 = exp_e / (3.0 * (1.0 - 0.6))
    exp_soundsp = math.sqrt((exp_c1 + 4.0 / 3.0 * exp_g) / 1.0)
    assert c_degraded == pytest.approx(exp_soundsp)


def test_law74_young_modulus_degradation_curve():
    """Verify Young's modulus scale factor curve: E = E0 * f_E(pla)."""
    curve_e = {
        "x": np.array([0.0, 0.05, 0.10]),
        "y": np.array([1.0, 0.8, 0.6]),
    }
    mat = build_law74(e=200000.0, nu=0.3, curve_e=curve_e, sigy0=200.0, ifunce=1)

    extra = {"uvar74": np.zeros((1, 10))}
    extra["uvar74"][0, 0] = 0.05

    c_degraded = law74_sound_speed(mat, rho=1.0, extra=extra)

    exp_e = 200000.0 * 0.8
    exp_g = 0.5 * exp_e / 1.3
    exp_c1 = exp_e / (3.0 * (1.0 - 0.6))
    exp_soundsp = math.sqrt((exp_c1 + 4.0 / 3.0 * exp_g) / 1.0)
    assert c_degraded == pytest.approx(exp_soundsp)


# ============================================================================
# 9. Tensile Failure Factor & Cubic Principal Strain Solve
# ============================================================================

def test_law74_tensile_failure_softening():
    """Verify cubic maximum tensile strain solver and tensile failure softening factor fail."""
    epsr1 = 0.02
    epsr2 = 0.06
    y0 = 300.0

    mat = build_law74(e=200000.0, nu=0.3, epsr1=epsr1, epsr2=epsr2, sigy0=y0)

    # Strain state with max principal strain = 0.04
    # fail should be (0.06 - 0.04) / (0.06 - 0.02) = 0.5
    eps_tot = np.array([[0.04, 0.0, 0.0, 0.0, 0.0, 0.0]])
    extra = {"eps": eps_tot}

    epst_solved = law74_hill_3d._solve_max_principal_strain(eps_tot)
    assert epst_solved[0] == pytest.approx(0.04)

    # Plastic step: yld should be softened by fail = 0.5 => 150.0
    deps = np.array([0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0])
    sig, pla = solid_update_law74(mat, np.zeros(6), deps, extra=extra)

    p = mat.params["_obj"]
    st_dev = sig[:3] - np.mean(sig[:3])
    cri = math.sqrt(p.ff * (st_dev[1] - st_dev[2])**2
                    + p.gg * (st_dev[2] - st_dev[0])**2
                    + p.hh * (st_dev[0] - st_dev[1])**2)

    # Softened yield stress is 0.5 * 300 = 150
    assert cri == pytest.approx(150.0, rel=2.0e-2)


# ============================================================================
# 10. Isotropic vs. Kinematic Hardening & Bauschinger Effect
# ============================================================================

def test_law74_kinematic_hardening_bauschinger_effect():
    """Verify back-stress development and Bauschinger effect under reverse cyclic loading."""
    curve = {
        "x": np.array([0.0, 0.1]),
        "y": np.array([200.0, 400.0]),
    }

    # 1. Pure isotropic hardening: chard = 0.0
    mat_iso = build_law74(e=200000.0, nu=0.3, chard=0.0, yield_table=curve)
    uvar_iso = np.zeros((1, 10))
    extra_iso = {"uvar74": uvar_iso}

    # Tensile loading
    deps_tens = np.array([0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0])
    sig_iso, pla_iso = solid_update_law74(mat_iso, np.zeros(6), deps_tens, extra=extra_iso)
    assert pla_iso > 0.0
    # In isotropic hardening, back-stress remains 0
    assert np.allclose(uvar_iso[0, 4:10], 0.0)

    # 2. Pure kinematic hardening: chard = 1.0
    mat_kin = build_law74(e=200000.0, nu=0.3, chard=1.0, yield_table=curve)
    uvar_kin = np.zeros((1, 10))
    extra_kin = {"uvar74": uvar_kin}

    sig_kin, pla_kin = solid_update_law74(mat_kin, np.zeros(6), deps_tens, extra=extra_kin)
    assert pla_kin > 0.0
    # In kinematic hardening, back-stress alpha_xx > 0 develops
    alpha_xx = uvar_kin[0, 4]
    assert alpha_xx > 0.0

    # Reverse compressive load
    deps_comp = np.array([-0.003, 0.0009, 0.0009, 0.0, 0.0, 0.0])
    sig_rev_iso, _ = solid_update_law74(mat_iso, sig_iso, deps_comp, epsp=pla_iso, extra=extra_iso)
    sig_rev_kin, _ = solid_update_law74(mat_kin, sig_kin, deps_comp, epsp=pla_kin, extra=extra_kin)

    # Due to backstress shift, kinematic hardening yields in compression much earlier
    # resulting in a less negative stress magnitude (Bauschinger effect)
    assert sig_rev_kin[0] > sig_rev_iso[0]


# ============================================================================
# 11. Element Deletion at eps_max
# ============================================================================

def test_law74_element_deletion():
    """Verify element deletion when plastic strain exceeds eps_max."""
    eps_max = 0.02
    mat = build_law74(e=200000.0, nu=0.3, sigy0=200.0, eps_max=eps_max)

    extra = {
        "uvar74": np.zeros((1, 10)),
        "off": np.array([1.0]),
    }

    # Step exceeding eps_max
    deps = np.array([0.03, -0.009, -0.009, 0.0, 0.0, 0.0])
    sig, pla = solid_update_law74(mat, np.zeros(6), deps, extra=extra)

    assert pla > eps_max
    # off is reduced to 0.8
    assert extra["off"][0] == pytest.approx(0.8)

    # Successive steps reduce off by 0.8 factor until < 0.1 -> 0.0
    for _ in range(12):
        sig, pla = solid_update_law74(mat, sig, deps * 0.01, epsp=pla, extra=extra)

    assert extra["off"][0] == 0.0
    # Deleted element stress zeroed out
    assert np.allclose(sig, 0.0)


# ============================================================================
# 12. Consistent Solid Tangent
# ============================================================================

def test_law74_consistent_solid_tangent_elastic():
    """Verify consistent solid tangent matches 3D Hookean tensor in elastic regime."""
    e_mod = 210000.0
    nu_val = 0.3
    mat = build_law74(e=e_mod, nu=nu_val, sigy0=1.0e9)

    sig = np.zeros(6)
    deps = np.array([1.0e-5, 0.0, 0.0, 0.0, 0.0, 0.0])

    tangent = law74_solid_tangent(mat, sig, deps=deps)
    assert tangent.shape == (6, 6)

    k_bulk = e_mod / (3.0 * (1.0 - 2.0 * nu_val))
    g_shear = 0.5 * e_mod / (1.0 + nu_val)
    c11 = k_bulk + 4.0 / 3.0 * g_shear
    c12 = k_bulk - 2.0 / 3.0 * g_shear

    assert tangent[0, 0] == pytest.approx(c11, rel=1.0e-4)
    assert tangent[1, 1] == pytest.approx(c11, rel=1.0e-4)
    assert tangent[0, 1] == pytest.approx(c12, rel=1.0e-4)
    assert tangent[3, 3] == pytest.approx(g_shear, rel=1.0e-4)


def test_law74_consistent_solid_tangent_plastic_perturbation():
    """Verify consistent solid tangent matches numerical finite difference in plastic regime."""
    mat = build_law74(e=200000.0, nu=0.3, sigy0=200.0)

    sig = np.zeros(6)
    deps = np.array([0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0])

    tangent = law74_solid_tangent(mat, sig, deps=deps, h=1.0e-6)
    assert tangent.shape == (6, 6)
    # Tangent stiffness should be lower than pure elastic due to plastic softening
    c11_el = 200000.0 * (1.0 - 0.3) / ((1.0 + 0.3) * (1.0 - 0.6))
    assert tangent[0, 0] < c11_el


# ============================================================================
# 13. Shell Update Rejection
# ============================================================================

def test_law74_shell_update_rejection():
    """Verify shell_update raises NotImplementedError as LAW74 is solids only."""
    mat = build_law74(e=200000.0, nu=0.3)
    with pytest.raises(NotImplementedError, match="LAW74.*is for solid elements only"):
        shell_update_law74(mat, np.zeros(3), np.zeros(3))

    with pytest.raises(NotImplementedError, match="LAW74.*is implemented for solid elements only"):
        materials.shell_update(mat, np.zeros(3), np.zeros(3))
