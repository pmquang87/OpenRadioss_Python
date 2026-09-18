"""Unit tests for BATCH 5 of material laws in pyradioss:
Laws: LAW51, LAW127, LAW128, LAW129, LAW130, LAW131, LAW132, LAW133, LAW134, LAW135, LAW151, LAW158, LAW187.

Tests parameter building, extra_shapes, solid_update, shell_update, sound_speed, and tangents.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import Material
from pyradioss.materials import (
    law51_granular_soil,
    law127_mohr_coulomb,
    law128_hill_rate_shell,
    law129_barlat_yld2000,
    law130_nonlocal_damage,
    law131_modular_elastoplas,
    law132_rate_modifier,
    law133_microplane_concrete,
    law134_fabric_wrinkle,
    law135_modular_user,
    law151_multimat_ale,
    law158_progressive_composite,
    law187_samp1_polymer,
)

LAWS = [
    (51, law51_granular_soil, "build_law51", "Law51Params"),
    (127, law127_mohr_coulomb, "build_law127", "Law127Params"),
    (128, law128_hill_rate_shell, "build_law128", "Law128Params"),
    (129, law129_barlat_yld2000, "build_law129", "Law129Params"),
    (130, law130_nonlocal_damage, "build_law130", "Law130Params"),
    (131, law131_modular_elastoplas, "build_law131", "Law131Params"),
    (132, law132_rate_modifier, "build_law132", "Law132Params"),
    (133, law133_microplane_concrete, "build_law133", "Law133Params"),
    (134, law134_fabric_wrinkle, "build_law134", "Law134Params"),
    (135, law135_modular_user, "build_law135", "Law135Params"),
    (151, law151_multimat_ale, "build_law151", "Law151Params"),
    (158, law158_progressive_composite, "build_law158", "Law158Params"),
    (187, law187_samp1_polymer, "build_law187", "Law187Params"),
]


@pytest.mark.parametrize("law_num, mod, builder_name, params_cls_name", LAWS)
def test_params_building_and_resolve(law_num, mod, builder_name, params_cls_name):
    """Verify building LawXXParams from Material and dictionary."""
    builder = getattr(mod, builder_name)
    resolver = getattr(mod, "resolve")
    params_cls = getattr(mod, params_cls_name)

    # 1. From empty / default Material
    mat = Material(id=1, law=law_num, rho0=2000.0, params={"E": 2.0e11, "nu": 0.3})
    p1 = builder(mat)
    assert isinstance(p1, params_cls)
    assert resolver(mat) == p1

    # 2. From dict
    d = {"E": 1.5e11, "nu": 0.25, "rho0": 2500.0}
    p2 = resolver(d)
    assert isinstance(p2, params_cls)
    assert p2.E == 1.5e11
    assert p2.nu == 0.25

    # 3. From existing instance
    p3 = resolver(p2)
    assert p3 is p2


@pytest.mark.parametrize("law_num, mod, _a, _b", LAWS)
def test_extra_shapes_and_defgrad(law_num, mod, _a, _b):
    """Verify extra_shapes and needs_defgrad interface."""
    mat = Material(id=1, law=law_num, rho0=2000.0, params={})
    shapes1 = mod.extra_shapes(mat, nip=1)
    assert isinstance(shapes1, dict)
    shapes4 = mod.extra_shapes(mat, nip=4)
    assert isinstance(shapes4, dict)

    needs_f = mod.needs_defgrad(mat)
    assert isinstance(needs_f, bool)


@pytest.mark.parametrize("law_num, mod, _a, _b", LAWS)
def test_sound_speed(law_num, mod, _a, _b):
    """Verify sound speed calculation."""
    mat = Material(id=1, law=law_num, rho0=2000.0, params={"E": 2.0e11, "nu": 0.3})
    c_solid = mod.sound_speed(mat)
    assert isinstance(c_solid, float)
    assert c_solid > 0.0

    c_shell = mod.sound_speed(mat, is_shell=True)
    assert isinstance(c_shell, float)
    assert c_shell > 0.0


@pytest.mark.parametrize("law_num, mod, _a, _b", LAWS)
def test_solid_update_1d_and_2d(law_num, mod, _a, _b):
    """Verify 3D solid constitutive update for 1D and 2D arrays with and without sound speed."""
    mat = Material(id=1, law=law_num, rho0=2000.0, params={"E": 2.0e11, "nu": 0.3})

    # 1. 1D single element update
    sig1d = np.zeros(6, dtype=np.float64)
    deps1d = np.array([1.0e-4, -3.0e-5, -3.0e-5, 0.0, 0.0, 0.0], dtype=np.float64)

    # Return sound speed = True (default)
    res = mod.solid_update(mat, sig1d, deps1d, epsp=0.0, dt=1.0e-6, return_sound_speed=True)
    assert len(res) == 3
    sig_out, ep_out, c_out = res
    assert sig_out.shape == (6,)
    assert isinstance(ep_out, (float, np.floating))
    assert isinstance(c_out, (float, np.floating))
    assert c_out > 0.0

    # Return sound speed = False
    res2 = mod.solid_update(mat, sig1d, deps1d, epsp=0.0, dt=1.0e-6, return_sound_speed=False)
    assert len(res2) == 2
    sig_out2, ep_out2 = res2
    assert sig_out2.shape == (6,)
    assert isinstance(ep_out2, (float, np.floating))

    # 2. 2D vectorized update (n=3)
    n = 3
    sig2d = np.zeros((n, 6), dtype=np.float64)
    deps2d = np.tile(deps1d, (n, 1))

    res2d = mod.solid_update(mat, sig2d, deps2d, epsp=np.zeros(n), dt=1.0e-6, return_sound_speed=True)
    assert len(res2d) == 3
    s2, ep2, c2 = res2d
    assert s2.shape == (n, 6)
    assert ep2.shape == (n,)
    assert c2.shape == (n,)
    assert np.all(c2 > 0.0)


@pytest.mark.parametrize("law_num, mod, _a, _b", LAWS)
def test_shell_update_1d_and_2d(law_num, mod, _a, _b):
    """Verify plane-stress shell constitutive update for 1D and 2D arrays."""
    mat = Material(id=1, law=law_num, rho0=2000.0, params={"E": 2.0e11, "nu": 0.3})

    # 1. 1D single element update
    sig1d = np.zeros(3, dtype=np.float64)
    deps1d = np.array([1.0e-4, 5.0e-5, 2.0e-5], dtype=np.float64)

    res = mod.shell_update(mat, sig1d, deps1d, epsp=0.0, dt=1.0e-6, return_sound_speed=True)
    assert len(res) == 3
    sig_out, ep_out, c_out = res
    assert sig_out.shape == (3,)
    assert isinstance(ep_out, (float, np.floating))
    assert isinstance(c_out, (float, np.floating))
    assert c_out > 0.0

    # 2. 2D vectorized update
    n = 4
    sig2d = np.zeros((n, 3), dtype=np.float64)
    deps2d = np.tile(deps1d, (n, 1))

    res2d = mod.shell_update(mat, sig2d, deps2d, epsp=np.zeros(n), dt=1.0e-6, return_sound_speed=True)
    assert len(res2d) == 3
    s2, ep2, c2 = res2d
    assert s2.shape == (n, 3)
    assert ep2.shape == (n,)
    assert c2.shape == (n,)
    assert np.all(c2 > 0.0)


@pytest.mark.parametrize("law_num, mod, _a, _b", LAWS)
def test_solid_and_shell_tangents(law_num, mod, _a, _b):
    """Verify solid and shell tangent matrix dimensions and properties."""
    mat = Material(id=1, law=law_num, rho0=2000.0, params={"E": 2.0e11, "nu": 0.3})

    # Solid tangent 1D / unvectorized
    c_sol = mod.solid_tangent(mat)
    assert c_sol.shape == (6, 6)
    assert np.all(np.diag(c_sol) > 0.0)

    # Consistent solid tangent
    c_sol_cons = mod.consistent_solid_tangent(mat)
    assert c_sol_cons.shape == (6, 6)

    # Solid tangent 2D / broadcasted
    sig2d = np.zeros((5, 6), dtype=np.float64)
    c_sol_2d = mod.solid_tangent(mat, sig=sig2d)
    assert c_sol_2d.shape == (5, 6, 6)

    # Shell tangent 1D / unvectorized
    c_sh = mod.shell_tangent(mat)
    assert c_sh.shape == (3, 3)
    assert np.all(np.diag(c_sh) > 0.0)

    # Consistent shell tangent
    c_sh_cons = mod.consistent_shell_tangent(mat)
    assert c_sh_cons.shape == (3, 3)

    # Shell tangent 2D / broadcasted
    sig_sh_2d = np.zeros((5, 3), dtype=np.float64)
    c_sh_2d = mod.shell_tangent(mat, sig=sig_sh_2d)
    assert c_sh_2d.shape == (5, 3, 3)


# =============================================================================
# Law-specific physics tests
# =============================================================================

def test_law51_pore_pressure_and_fracture_cutoff():
    """Verify LAW51 pore pressure evolution and fracture cutoff."""
    mat = Material(id=51, law=51, rho0=2000.0, params={
        "E": 1.0e8, "nu": 0.3, "pfrac": -5.0e4, "a0": 5.0e5, "a1": 0.6,
        "k_pore": 2.0e8, "skempton_b": 0.8,
    })
    extra = {"p_pore": [1.0e4]}
    # Large tensile volumetric expansion
    deps = np.array([1.0e-3, 1.0e-3, 1.0e-3, 0.0, 0.0, 0.0], dtype=np.float64)
    sig = np.zeros(6, dtype=np.float64)
    sig_out, ep_out, c = law51_granular_soil.solid_update(mat, sig, deps, extra=extra)
    # Under large tension, pressure hits cutoff pfrac
    p_eff = -np.mean(sig_out[:3])
    assert p_eff <= -5.0e4 or math.isclose(p_eff, -5.0e4, rel_tol=1e-2)


def test_law127_mohr_coulomb_shear_failure():
    """Verify LAW127 Drucker-Prager / Mohr-Coulomb shear yield and plastic flow."""
    mat = Material(id=127, law=127, rho0=2000.0, params={
        "E": 3.0e10, "nu": 0.25, "cohesion": 5.0e6, "phi": 30.0, "p_cap": 1.0e8
    })
    sig = np.zeros(6, dtype=np.float64)
    # High shear strain
    deps = np.array([0.0, 0.0, 0.0, 1.0e-2, 0.0, 0.0], dtype=np.float64)
    sig_out, ep_out, c = law127_mohr_coulomb.solid_update(mat, sig, deps)
    assert ep_out > 0.0  # Plastic dissipation occurred
    assert sig_out[3] < 3.0e10 / (2.0 * 1.25) * 1.0e-2  # Yield cut the shear stress


def test_law128_hill_orthotropy_and_rate_effect():
    """Verify LAW128 Hill anisotropy and strain rate enhancement."""
    # Transverse anisotropy with higher yield in y than x
    mat_static = Material(id=128, law=128, rho0=7850.0, params={
        "E": 2.1e11, "nu": 0.3, "sig0": 1.0e8, "F": 0.3, "G_hill": 0.7, "H": 0.5, "N": 1.5,
        "c_rate": 100.0, "p_rate": 2.0,
    })
    sig = np.zeros(6, dtype=np.float64)
    deps_slow = np.array([1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_slow, ep_slow, _ = law128_hill_rate_shell.solid_update(mat_static, sig, deps_slow, dt=1.0)

    # Fast strain rate dt = 1e-4 => strain rate = 10 s^-1
    sig_fast, ep_fast, _ = law128_hill_rate_shell.solid_update(mat_static, sig, deps_slow, dt=1.0e-4)
    assert sig_fast[0] > sig_slow[0]  # Rate hardening increases flow stress


def test_law129_barlat_yld2000_anisotropy():
    """Verify LAW129 Barlat Yld2000-2d equivalent stress evaluation."""
    p = law129_barlat_yld2000.Law129Params(
        E=7.0e10, nu=0.33, m_exp=8.0,
        alpha1=1.2, alpha2=0.9, alpha3=1.1, alpha4=1.0,
        alpha5=1.0, alpha6=1.0, alpha7=1.0, alpha8=1.0
    )
    s_eq = law129_barlat_yld2000._barlat_yld2000_eq_stress(p, 1.0e8, 0.0, 0.0)
    assert s_eq > 0.0
    assert not math.isnan(s_eq)


def test_law130_nonlocal_damage_softening():
    """Verify LAW130 continuous damage degradation."""
    mat = Material(id=130, law=130, rho0=1500.0, params={
        "E": 2.0e10, "nu": 0.2, "eps_d0": 1.0e-4, "eps_df": 1.0e-2, "alpha_d": 0.9, "beta_d": 100.0
    })
    extra = {}
    sig = np.zeros(6, dtype=np.float64)
    deps_small = np.array([5.0e-5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_small, _, _ = law130_nonlocal_damage.solid_update(mat, sig, deps_small, extra=extra)
    assert extra["dmg"] == 0.0

    # Large strain exceeding damage threshold
    deps_large = np.array([5.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_large, _, _ = law130_nonlocal_damage.solid_update(mat, sig_small, deps_large, extra=extra)
    assert extra["dmg"] > 0.0  # Damage accumulated


def test_law131_modular_yield_and_hardening():
    """Verify LAW131 modularity (Swift hardening with AF kinematic hardening)."""
    mat = Material(id=131, law=131, rho0=7850.0, params={
        "E": 2.1e11, "nu": 0.3, "sig0": 2.0e8, "hard_type": 2, "a_swift": 5.0e8, "eps0_swift": 0.002, "n_swift": 0.25,
        "c_af": 1.0e10, "gamma_af": 10.0
    })
    extra = {}
    sig = np.zeros(6, dtype=np.float64)
    deps = np.array([3.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_out, ep_out, _ = law131_modular_elastoplas.solid_update(mat, sig, deps, extra=extra)
    assert ep_out > 0.0
    assert "backstress" in extra
    assert np.any(extra["backstress"] != 0.0)


def test_law132_rate_modifier_strengths():
    """Verify LAW132 strain-rate dependent strength scaling."""
    mat = Material(id=132, law=132, rho0=1600.0, params={
        "E": 1.4e11, "eb": 9.0e9, "nu": 0.3, "xt": 1.5e9, "c_xt": 0.1, "eps0_rate": 1.0
    })
    # Fast strain rate: 100 s^-1 => dt = 1e-4 for deps = 1e-2
    deps = np.array([1.0e-2, 0.0, 0.0], dtype=np.float64)
    sig = np.zeros(3, dtype=np.float64)
    sig_fast, _, _ = law132_rate_modifier.shell_update(mat, sig, deps, dt=1.0e-4)
    assert sig_fast[0] > 0.0


def test_law133_microplane_concrete_pressure_dependence():
    """Verify LAW133 yield stress increases under confining pressure."""
    mat = Material(id=133, law=133, rho0=2400.0, params={
        "E": 3.0e10, "nu": 0.2, "a0": 2.0e7, "a1": 1.5, "pmin": -2.0e6
    })
    # Pure shear under zero pressure
    sig_zero = np.zeros(6, dtype=np.float64)
    deps_shear = np.array([0.0, 0.0, 0.0, 2.0e-3, 0.0, 0.0], dtype=np.float64)
    sig_out_unconfined, _, _ = law133_microplane_concrete.solid_update(mat, sig_zero, deps_shear)

    # Pure shear under high triaxial confining pressure P = 50 MPa
    sig_conf = np.array([-5.0e7, -5.0e7, -5.0e7, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_out_confined, _, _ = law133_microplane_concrete.solid_update(mat, sig_conf, deps_shear)
    # Higher confinement allows higher shear stress
    assert sig_out_confined[3] > sig_out_unconfined[3]


def test_law134_fabric_wrinkling():
    """Verify LAW134 fabric wrinkling: compression vanishes in membrane state."""
    mat = Material(id=134, law=134, rho0=1000.0, params={
        "E": 1.0e9, "nu": 0.3, "wrinkle_flag": 1
    })
    sig = np.zeros(3, dtype=np.float64)
    # Tension in x, compression in y
    deps = np.array([1.0e-3, -1.0e-3, 0.0], dtype=np.float64)
    sig_out, _, _ = law134_fabric_wrinkle.shell_update(mat, sig, deps)
    assert sig_out[0] > 0.0
    assert math.isclose(sig_out[1], 0.0, abs_tol=1e-5)  # Compression wrinkling zeros out syy


def test_law135_modular_user_39_uvars():
    """Verify LAW135 internal variable tracking."""
    mat = Material(id=135, law=135, rho0=7850.0, params={"E": 2.0e11, "nu": 0.3, "sig0": 2.5e8})
    shapes = law135_modular_user.extra_shapes(mat, nip=1)
    assert shapes["uvar"] == (39,)
    extra = {}
    sig = np.zeros(6, dtype=np.float64)
    deps = np.array([2.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_out, ep_out, _ = law135_modular_user.solid_update(mat, sig, deps, extra=extra)
    assert ep_out > 0.0
    assert extra["uvar"].shape == (39,)


def test_law151_multimat_ale_mixture():
    """Verify LAW151 multi-material ALE volume fraction homogenization."""
    mat = Material(id=151, law=151, rho0=1000.0, params={
        "nbmat": 2, "vfrac": [0.6, 0.4], "rho_phases": [1000.0, 2000.0], "bulk_phases": [2.2e9, 1.0e10]
    })
    p = law151_multimat_ale.resolve(mat)
    expected_rho = 0.6 * 1000.0 + 0.4 * 2000.0
    assert math.isclose(p.rho0, expected_rho)
    c = law151_multimat_ale.sound_speed(p)
    assert c > 0.0


def test_law158_progressive_composite_damage():
    """Verify LAW158 progressive composite damage accumulation."""
    mat = Material(id=158, law=158, rho0=1600.0, params={
        "E": 5.0e10, "nu": 0.25, "dt0": 1.0e-3
    })
    extra = {}
    sig = np.zeros(3, dtype=np.float64)
    deps = np.array([3.0e-3, 0.0, 0.0], dtype=np.float64)
    sig_out, ep_out, _ = law158_progressive_composite.shell_update(mat, sig, deps, extra=extra)
    assert "uvar" in extra
    assert extra["uvar"][15] > 0.0  # Tensile fiber damage accumulated


def test_law187_samp1_polymer_crazing():
    """Verify LAW187 polymer crazing and cavitation damage."""
    mat = Material(id=187, law=187, rho0=1000.0, params={
        "E": 2.0e9, "nu": 0.35, "sig0": 3.0e7, "sig_craze": 2.0e7
    })
    extra = {}
    sig = np.zeros(6, dtype=np.float64)
    # Massive triaxial tensile strain triggering cavitation / crazing
    deps = np.array([2.0e-2, 2.0e-2, 2.0e-2, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_out, ep_out, _ = law187_samp1_polymer.solid_update(mat, sig, deps, extra=extra)
    assert "dmg" in extra
    assert extra["dmg"][0] > 0.0  # Crazing cavitation damage activated
