"""Unit tests for Batch 4 OpenRadioss Material Laws.

Laws tested:
  - LAW98: Loss Function & Optimization Container (pyradioss.materials.law98_loss_optim)
  - LAW108: Tabulated Yield Surface Fitting & Optimization Updater (pyradioss.materials.law108_yield_fit)
  - LAW111: Barlat Yld2004-18p 3D Anisotropic Yield & Marlow Model (pyradioss.materials.law111_barlat_yld2004)
  - LAW112: Xia Ductile Damage and Void Nucleation Model (pyradioss.materials.law112_xia_damage)
  - LAW113: Yield Curve Fitting & Hardening Optimizer (pyradioss.materials.law113_yield_curve_fit)
  - LAW115: Perzyna Viscoplastic & Deshpande-Fleck Foam Formulation (pyradioss.materials.law115_perzyna)
  - LAW116: Spotweld Connector with Progressive Damage (pyradioss.materials.law116_spotweld)
  - LAW118: Reserved Law Stub (pyradioss.materials.law118_reserved)
  - LAW122: Chaboche Nonlinear Kinematic Hardening Model (pyradioss.materials.law122_chaboche)
  - LAW125: Tabulated Anisotropic Plasticity with Strain-Rate Dependency (pyradioss.materials.law125_tab_aniso_rate)
"""

import math
import numpy as np
import pytest

from pyradioss.model.entities import Material
from pyradioss.materials import (
    law98_loss_optim,
    law108_yield_fit,
    law111_barlat_yld2004,
    law112_xia_damage,
    law113_yield_curve_fit,
    law115_perzyna,
    law116_spotweld,
    law118_reserved,
    law122_chaboche,
    law125_tab_aniso_rate,
)


# ============================================================================
# LAW98 Tests
# ============================================================================

def test_law98_params_and_resolve():
    mat = Material(
        id=98,
        law=98,
        rho0=7800.0,
        title="Mat_98",
        params={
            "E": 210000.0,
            "nu": 0.3,
            "A": 250.0,
            "B": 500.0,
            "N": 0.5,
            "C": 0.02,
            "EPSO": 1.0,
            "SIGMAX": 800.0,
            "SIGSAT": 750.0,
        },
    )
    p = law98_loss_optim.build_law98(mat)
    assert isinstance(p, law98_loss_optim.Law98Params)
    assert p.rho0 == 7800.0
    assert p.young == 210000.0
    assert p.nu == 0.3
    assert p.a == 250.0
    assert p.b == 500.0
    assert p.n == 0.5
    assert p.sigsat == 750.0

    p_resolved = law98_loss_optim.resolve(mat)
    assert isinstance(p_resolved, law98_loss_optim.Law98Params)
    assert law98_loss_optim.needs_defgrad(mat) is False

    shapes_single = law98_loss_optim.extra_shapes(mat)
    assert "uvar98" in shapes_single
    assert shapes_single["uvar98"] == (6,)

    shapes_nip = law98_loss_optim.extra_shapes(mat, nip=5)
    assert shapes_nip["uvar98"] == (5, 6)


def test_law98_solid_and_shell_update():
    p = law98_loss_optim.build_law98(young=200000.0, nu=0.3, rho0=7800.0, a=200.0, b=300.0, n=0.5)

    # 1D solid update
    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([0.005, -0.001, -0.001, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_new, epsp_new, c = law98_loss_optim.solid_update(p, sig0, deps, epsp=0.0, dt=1.0e-6)
    assert sig_new.shape == (6,)
    assert sig_new[0] > 0.0
    assert epsp_new >= 0.0
    assert c > 0.0

    # 2D batch solid update
    sig_batch = np.zeros((3, 6), dtype=np.float64)
    deps_batch = np.tile(deps, (3, 1))
    sig_out, epsp_out, c_out = law98_loss_optim.solid_update(p, sig_batch, deps_batch, epsp=np.zeros(3), dt=1.0e-6)
    assert sig_out.shape == (3, 6)
    assert len(epsp_out) == 3
    assert len(c_out) == 3

    # 1D shell update
    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([0.005, -0.001, 0.0], dtype=np.float64)
    sig_sh_new, epsp_sh_new, c_sh = law98_loss_optim.shell_update(p, sig_sh, deps_sh, epsp=0.0, dt=1.0e-6)
    assert sig_sh_new.shape == (3,)
    assert sig_sh_new[0] > 0.0
    assert c_sh > 0.0

    # Tangents
    t_solid = law98_loss_optim.solid_tangent(p, sig=sig_new, epsp=epsp_new)
    assert t_solid.shape == (6, 6)
    t_shell = law98_loss_optim.shell_tangent(p, sig=sig_sh_new, epsp=epsp_sh_new)
    assert t_shell.shape == (3, 3)


def test_law98_loss_function():
    uparam = np.zeros(20, dtype=np.float64)
    uparam[8:16] = 1.0
    x = np.array([0.05, 100.0, 0.05, 100.0], dtype=np.float64)
    fun, dfun = law98_loss_optim.lossfun_98(x, uparam, [1, 2], 2)
    assert isinstance(fun, float)
    assert dfun.shape == (4,)


# ============================================================================
# LAW108 Tests
# ============================================================================

def test_law108_params_and_resolve():
    mat = Material(
        id=108,
        law=108,
        rho0=1000.0,
        title="Spr_Gene_108",
        params={
            "STIFF1": 50000.0,
            "DAMP1": 10.0,
            "HFLAG1": 2,
            "STIFF2": 40000.0,
            "young": 50000.0,
            "nu": 0.25,
        },
    )
    p = law108_yield_fit.build_law108(mat)
    assert isinstance(p, law108_yield_fit.Law108Params)
    assert p.rho0 == 1000.0
    assert p.dofs[0].stiff == 50000.0
    assert p.dofs[0].damp == 10.0
    assert p.dofs[1].stiff == 40000.0
    assert law108_yield_fit.needs_defgrad(mat) is False

    shapes = law108_yield_fit.extra_shapes(mat, nip=2)
    assert shapes["uvar108"] == (2, 12)

    # Test law108_upd tangent fitting
    curves = {1: (np.array([0.0, 0.1, 0.2]), np.array([0.0, 10000.0, 30000.0]))}
    p.dofs[0].fun_a = 1
    law108_yield_fit.law108_upd(p, curves)
    assert p.dofs[0].stiff >= 200000.0


def test_law108_solid_and_shell_update():
    p = law108_yield_fit.build_law108(young=100000.0, nu=0.3, rho0=2000.0)
    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    sig_new, epsp_new, c = law108_yield_fit.solid_update(p, sig0, deps)
    assert sig_new[0] > 0.0
    assert c > 0.0

    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([0.001, 0.0, 0.0], dtype=np.float64)
    sig_sh_new, epsp_sh_new, c_sh = law108_yield_fit.shell_update(p, sig_sh, deps_sh)
    assert sig_sh_new[0] > 0.0
    assert c_sh > 0.0

    t_solid = law108_yield_fit.solid_tangent(p)
    assert t_solid.shape == (6, 6)
    t_shell = law108_yield_fit.shell_tangent(p)
    assert t_shell.shape == (3, 3)


# ============================================================================
# LAW111 Tests
# ============================================================================

def test_law111_params_and_cardan():
    mat = Material(
        id=111,
        law=111,
        rho0=2700.0,
        title="Barlat_2004",
        params={
            "MAT_E": 70000.0,
            "MAT_NU": 0.33,
            "M": 8.0,
            "SIGY0": 120.0,
            "H": 150.0,
            "N": 0.3,
            "CP12": 1.1,
            "CPP12": 0.9,
        },
    )
    p = law111_barlat_yld2004.build_law111(mat)
    assert p.m_exp == 8.0
    assert p.sigy0 == 120.0
    assert p.cp_12 == 1.1
    assert p.cpp_12 == 0.9
    assert law111_barlat_yld2004.needs_defgrad(mat) is True

    # Test Cardan cubic equation solver
    # (x - 1)(x - 2)(x - 3) = x^3 - 6x^2 + 11x - 6 = 0 -> roots 1, 2, 3
    r1, r2, r3 = law111_barlat_yld2004.cardan_method(1.0, -6.0, 11.0, -6.0)
    roots = sorted([r1, r2, r3])
    np.testing.assert_allclose(roots, [1.0, 2.0, 3.0], atol=1.0e-5)


def test_law111_solid_and_shell_update():
    p = law111_barlat_yld2004.build_law111(
        young=70000.0, nu=0.33, rho0=2700.0, sigy0=100.0, h_mod=200.0, n_hard=0.5
    )
    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0], dtype=np.float64)

    sig_new, epsp_new, c = law111_barlat_yld2004.solid_update(p, sig0, deps, epsp=0.0)
    assert sig_new[0] > 0.0
    assert epsp_new > 0.0
    assert c > 0.0

    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([0.005, -0.0015, 0.0], dtype=np.float64)
    sig_sh_new, epsp_sh_new, c_sh = law111_barlat_yld2004.shell_update(p, sig_sh, deps_sh, epsp=0.0)
    assert sig_sh_new[0] > 0.0
    assert epsp_sh_new > 0.0

    t_solid = law111_barlat_yld2004.solid_tangent(p)
    assert t_solid.shape == (6, 6)
    t_shell = law111_barlat_yld2004.shell_tangent(p)
    assert t_shell.shape == (3, 3)


# ============================================================================
# LAW112 Tests
# ============================================================================

def test_law112_params_and_damage():
    mat = Material(
        id=112,
        law=112,
        rho0=800.0,
        title="Xia_Paper",
        params={
            "MAT_E1": 6000.0,
            "MAT_E2": 3000.0,
            "MAT_E3": 1000.0,
            "MAT_NU12": 0.35,
            "SIGY1": 30.0,
            "SIGY2": 15.0,
            "EPS_D0": 0.02,
            "EPS_DF": 0.10,
            "DMAX": 0.95,
        },
    )
    p = law112_xia_damage.build_law112(mat)
    assert p.young1 == 6000.0
    assert p.young2 == 3000.0
    assert p.sigy1 == 30.0
    assert p.eps_d0 == 0.02
    assert p.dmax == 0.95
    assert law112_xia_damage.needs_defgrad(mat) is False

    # Check damage function
    assert law112_xia_damage._compute_damage(p, 0.01) == 0.0
    assert law112_xia_damage._compute_damage(p, 0.06) == pytest.approx(0.5, abs=0.01)
    assert law112_xia_damage._compute_damage(p, 0.15) == 0.95


def test_law112_solid_and_shell_update():
    p = law112_xia_damage.build_law112(
        young1=5000.0, young2=2500.0, young3=800.0, nu12=0.3, rho0=800.0,
        sigy1=20.0, sigy2=10.0, eps_d0=0.01, eps_df=0.05
    )
    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    sig_new, epsp_new, c = law112_xia_damage.solid_update(p, sig0, deps, epsp=0.0)
    assert sig_new[0] > 0.0
    assert c > 0.0

    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([0.01, 0.0, 0.0], dtype=np.float64)
    sig_sh_new, epsp_sh_new, c_sh = law112_xia_damage.shell_update(p, sig_sh, deps_sh, epsp=0.0)
    assert sig_sh_new[0] > 0.0

    t_solid = law112_xia_damage.solid_tangent(p, epsp=0.02)
    assert t_solid.shape == (6, 6)
    t_shell = law112_xia_damage.shell_tangent(p, epsp=0.02)
    assert t_shell.shape == (3, 3)


# ============================================================================
# LAW113 Tests
# ============================================================================

def test_law113_params_and_upd():
    mat = Material(
        id=113,
        law=113,
        rho0=1200.0,
        title="Spr_Beam_113",
        params={
            "STIFF1": 80000.0,
            "DAMP1": 5.0,
            "STIFF2": 60000.0,
            "young": 80000.0,
            "nu": 0.28,
        },
    )
    p = law113_yield_curve_fit.build_law113(mat)
    assert p.dofs[0].stiff == 80000.0
    assert p.dofs[1].stiff == 60000.0
    assert law113_yield_curve_fit.needs_defgrad(mat) is False

    shapes = law113_yield_curve_fit.extra_shapes(mat, nip=3)
    assert shapes["uvar113"] == (3, 12)

    curves = {2: (np.array([0.0, 0.05, 0.1]), np.array([10.0, 5000.0, 15000.0]))}
    p.dofs[1].fun_a = 2
    law113_yield_curve_fit.law113_upd(p, curves)
    assert p.dofs[1].stiff >= 200000.0
    assert p.dofs[1].f_x0 == 10.0


def test_law113_solid_and_shell_update():
    p = law113_yield_curve_fit.build_law113(young=60000.0, nu=0.3, rho0=1500.0)
    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([0.002, -0.0005, -0.0005, 0.0, 0.0, 0.0], dtype=np.float64)

    sig_new, epsp_new, c = law113_yield_curve_fit.solid_update(p, sig0, deps)
    assert sig_new[0] > 0.0
    assert c > 0.0

    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([0.002, -0.0005, 0.0], dtype=np.float64)
    sig_sh_new, epsp_sh_new, c_sh = law113_yield_curve_fit.shell_update(p, sig_sh, deps_sh)
    assert sig_sh_new[0] > 0.0

    t_solid = law113_yield_curve_fit.solid_tangent(p)
    assert t_solid.shape == (6, 6)
    t_shell = law113_yield_curve_fit.shell_tangent(p)
    assert t_shell.shape == (3, 3)


# ============================================================================
# LAW115 Tests
# ============================================================================

def test_law115_params_and_deshfleck():
    mat = Material(
        id=115,
        law=115,
        rho0=300.0,
        title="DeshFleck_Foam",
        params={
            "MAT_E": 1500.0,
            "MAT_NU": 0.1,
            "MAT_ALPHA": 0.8,
            "MAT_SIGP": 5.0,
            "MAT_GAMMA": 20.0,
            "MAT_EPSD": 0.5,
            "MAT_ALPHA2": 50.0,
            "MAT_BETA": 2.5,
        },
    )
    p = law115_perzyna.build_law115(mat)
    assert p.young == 1500.0
    assert p.alpha == 0.8
    assert p.sigp == 5.0
    assert p.gamma == 20.0
    assert p.alpha2 == 50.0
    assert law115_perzyna.needs_defgrad(mat) is False

    # Test Deshpande-Fleck equivalent stress calculation
    sig_hydro = np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_hat, p_m, s = law115_perzyna._compute_deshfleck_equivalent_stress(sig_hydro, alpha=0.8)
    assert p_m == pytest.approx(10.0)
    assert np.allclose(s, 0.0)
    assert sig_hat > 0.0


def test_law115_solid_and_shell_update():
    p = law115_perzyna.build_law115(
        young=2000.0, nu=0.15, rho0=400.0, alpha=0.5, sigp=4.0, gamma=15.0
    )
    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([0.01, 0.005, 0.005, 0.0, 0.0, 0.0], dtype=np.float64)

    sig_new, epsp_new, c = law115_perzyna.solid_update(p, sig0, deps, epsp=0.0)
    assert sig_new[0] > 0.0
    assert epsp_new > 0.0
    assert c > 0.0

    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([0.01, 0.005, 0.0], dtype=np.float64)
    sig_sh_new, epsp_sh_new, c_sh = law115_perzyna.shell_update(p, sig_sh, deps_sh, epsp=0.0)
    assert sig_sh_new[0] > 0.0

    t_solid = law115_perzyna.solid_tangent(p, sig=sig_new, epsp=epsp_new)
    assert t_solid.shape == (6, 6)
    t_shell = law115_perzyna.shell_tangent(p, sig=sig_sh_new, epsp=epsp_sh_new)
    assert t_shell.shape == (3, 3)


# ============================================================================
# LAW116 Tests
# ============================================================================

def test_law116_params_and_damage():
    mat = Material(
        id=116,
        law=116,
        rho0=7850.0,
        title="Spotweld_116",
        params={
            "MAT_E": 210000.0,
            "MAT_G": 80000.0,
            "MAT_THICK": 1.5,
            "MAT_GC1_ini": 5.0,
            "MAT_GC2_ini": 10.0,
            "MAT_SIGA1": 300.0,
            "MAT_SIGA2": 200.0,
        },
    )
    p = law116_spotweld.build_law116(mat)
    assert p.young == 210000.0
    assert p.g_mod == 80000.0
    assert p.thick == 1.5
    assert p.gc1_ini == 5.0
    assert p.siga1 == 300.0
    assert law116_spotweld.needs_defgrad(mat) is False

    dmg, delta = law116_spotweld._compute_spotweld_damage(p, eps_n=0.0001, gamma_t=0.0)
    assert dmg == 0.0

    dmg_fail, _ = law116_spotweld._compute_spotweld_damage(p, eps_n=0.1, gamma_t=0.1)
    assert dmg_fail == 1.0


def test_law116_solid_and_shell_update():
    p = law116_spotweld.build_law116(
        young=200000.0, g_mod=75000.0, thick=1.0, rho0=7800.0,
        siga1=250.0, gc1_ini=4.0, siga2=150.0, gc2_ini=8.0
    )
    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([0.0, 0.0, 0.002, 0.0, 0.001, 0.0], dtype=np.float64)

    sig_new, epsp_new, c = law116_spotweld.solid_update(p, sig0, deps)
    assert sig_new[2] > 0.0
    assert sig_new[4] > 0.0
    assert c > 0.0

    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([0.001, 0.0, 0.001], dtype=np.float64)
    sig_sh_new, epsp_sh_new, c_sh = law116_spotweld.shell_update(p, sig_sh, deps_sh)
    assert sig_sh_new[0] > 0.0
    assert c_sh > 0.0

    t_solid = law116_spotweld.solid_tangent(p)
    assert t_solid.shape == (6, 6)
    t_shell = law116_spotweld.shell_tangent(p)
    assert t_shell.shape == (3, 3)


# ============================================================================
# LAW118 Tests
# ============================================================================

def test_law118_reserved_stub():
    mat = Material(
        id=118,
        law=118,
        rho0=5000.0,
        title="Reserved_118",
        params={"young": 150000.0, "nu": 0.25},
    )
    p = law118_reserved.build_law118(mat)
    assert isinstance(p, law118_reserved.Law118Params)
    assert p.young == 150000.0
    assert p.nu == 0.25
    assert law118_reserved.needs_defgrad(mat) is False

    shapes = law118_reserved.extra_shapes(mat)
    assert "uvar118" in shapes

    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_new, epsp_new, c = law118_reserved.solid_update(p, sig0, deps)
    assert sig_new[0] > 0.0
    assert c > 0.0

    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([0.001, 0.0, 0.0], dtype=np.float64)
    sig_sh_new, epsp_sh_new, c_sh = law118_reserved.shell_update(p, sig_sh, deps_sh)
    assert sig_sh_new[0] > 0.0

    t_solid = law118_reserved.solid_tangent(p)
    assert t_solid.shape == (6, 6)
    t_shell = law118_reserved.shell_tangent(p)
    assert t_shell.shape == (3, 3)


# ============================================================================
# LAW122 Tests
# ============================================================================

def test_law122_params_and_chaboche():
    mat = Material(
        id=122,
        law=122,
        rho0=7800.0,
        title="Chaboche_122",
        params={
            "young": 200000.0,
            "nu": 0.3,
            "SIGY0": 250.0,
            "R_INF": 100.0,
            "B_ISO": 10.0,
            "C_KIN": 10000.0,
            "GAMMA_KIN": 50.0,
        },
    )
    p = law122_chaboche.build_law122(mat)
    assert p.young == 200000.0
    assert p.sigy0 == 250.0
    assert p.r_inf == 100.0
    assert p.c_kin == 10000.0
    assert p.gamma_kin == 50.0
    assert law122_chaboche.needs_defgrad(mat) is False

    sig_y, h = law122_chaboche._eval_chaboche_yield_stress(p, eps_p=0.05)
    assert sig_y > 250.0
    assert h > 0.0


def test_law122_solid_and_shell_update():
    p = law122_chaboche.build_law122(
        young1=190000.0, nu12=0.29, rho0=7800.0,
        sigy0=200.0, r_inf=50.0, b_iso=5.0, c_kin=8000.0, gamma_kin=40.0
    )
    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0], dtype=np.float64)
    extra = {"backstress": np.zeros(6, dtype=np.float64)}

    sig_new, epsp_new, c = law122_chaboche.solid_update(p, sig0, deps, epsp=0.0, extra=extra)
    assert sig_new[0] > 0.0
    assert epsp_new > 0.0
    assert c > 0.0
    assert np.any(extra["backstress"] != 0.0)

    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([0.005, -0.0015, 0.0], dtype=np.float64)
    sig_sh_new, epsp_sh_new, c_sh = law122_chaboche.shell_update(p, sig_sh, deps_sh, epsp=0.0)
    assert sig_sh_new[0] > 0.0

    t_solid = law122_chaboche.solid_tangent(p, sig=sig_new, epsp=epsp_new)
    assert t_solid.shape == (6, 6)
    t_shell = law122_chaboche.shell_tangent(p, sig=sig_sh_new, epsp=epsp_sh_new)
    assert t_shell.shape == (3, 3)


# ============================================================================
# LAW125 Tests
# ============================================================================

def test_law125_params_and_composite_damage():
    mat = Material(
        id=125,
        law=125,
        rho0=1600.0,
        title="Laminated_Comp_125",
        params={
            "MAT_E1": 120000.0,
            "MAT_E2": 10000.0,
            "MAT_E3": 10000.0,
            "MAT_NU12": 0.3,
            "MAT_G12": 4500.0,
            "MAT_XT": 1500.0,
            "MAT_XC": 1000.0,
            "MAT_YT": 50.0,
            "MAT_YC": 150.0,
            "MAT_SC": 70.0,
            "EM11T": 0.015,
            "EF11T": 0.03,
        },
    )
    p = law125_tab_aniso_rate.build_law125(mat)
    assert p.young1 == 120000.0
    assert p.young2 == 10000.0
    assert p.xt == 1500.0
    assert p.em11t == 0.015
    assert p.ef11t == 0.03
    assert law125_tab_aniso_rate.needs_defgrad(mat) is False

    # Damage progression
    d0 = law125_tab_aniso_rate._compute_composite_damage(0.01, p.em11t, p.ef11t, p.m1t, p.al1t, p.dmax)
    assert d0 == 0.0
    d_mid = law125_tab_aniso_rate._compute_composite_damage(0.022, p.em11t, p.ef11t, p.m1t, p.al1t, p.dmax)
    assert 0.0 < d_mid < 1.0
    d_fail = law125_tab_aniso_rate._compute_composite_damage(0.04, p.em11t, p.ef11t, p.m1t, p.al1t, p.dmax)
    assert d_fail == p.dmax


def test_law125_solid_and_shell_update():
    p = law125_tab_aniso_rate.build_law125(
        young1=100000.0, young2=8000.0, young3=8000.0, nu12=0.3, rho0=1500.0,
        xt=1200.0, xc=800.0, yt=40.0, yc=100.0, sc=50.0
    )
    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([0.005, 0.001, 0.0, 0.001, 0.0, 0.0], dtype=np.float64)

    sig_new, epsp_new, c = law125_tab_aniso_rate.solid_update(p, sig0, deps)
    assert sig_new[0] > 0.0
    assert sig_new[1] > 0.0
    assert c > 0.0

    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([0.005, 0.001, 0.001], dtype=np.float64)
    sig_sh_new, epsp_sh_new, c_sh = law125_tab_aniso_rate.shell_update(p, sig_sh, deps_sh)
    assert sig_sh_new[0] > 0.0
    assert sig_sh_new[1] > 0.0

    t_solid = law125_tab_aniso_rate.solid_tangent(p)
    assert t_solid.shape == (6, 6)
    t_shell = law125_tab_aniso_rate.shell_tangent(p)
    assert t_shell.shape == (3, 3)


# ============================================================================
# Interface Contract Test for All 10 Laws
# ============================================================================

@pytest.mark.parametrize(
    "law_mod, law_num, params_cls_name, build_fn_name",
    [
        (law98_loss_optim, 98, "Law98Params", "build_law98"),
        (law108_yield_fit, 108, "Law108Params", "build_law108"),
        (law111_barlat_yld2004, 111, "Law111Params", "build_law111"),
        (law112_xia_damage, 112, "Law112Params", "build_law112"),
        (law113_yield_curve_fit, 113, "Law113Params", "build_law113"),
        (law115_perzyna, 115, "Law115Params", "build_law115"),
        (law116_spotweld, 116, "Law116Params", "build_law116"),
        (law118_reserved, 118, "Law118Params", "build_law118"),
        (law122_chaboche, 122, "Law122Params", "build_law122"),
        (law125_tab_aniso_rate, 125, "Law125Params", "build_law125"),
    ],
)
def test_batch4_interface_contract(law_mod, law_num, params_cls_name, build_fn_name):
    """Verify that each law module implements all required methods with standard signatures."""
    # 1. Check LawXXParams dataclass exists
    assert hasattr(law_mod, params_cls_name), f"Missing {params_cls_name} in {law_mod.__name__}"
    params_cls = getattr(law_mod, params_cls_name)

    # 2. Check build_lawXX(mat) exists and works with Material entity
    assert hasattr(law_mod, build_fn_name), f"Missing {build_fn_name} in {law_mod.__name__}"
    build_fn = getattr(law_mod, build_fn_name)
    mat = Material(id=law_num, law=law_num, rho0=7800.0, title=f"LAW{law_num}", params={"E": 200000.0, "nu": 0.3})
    p = build_fn(mat)
    assert isinstance(p, params_cls)

    # 3. Check resolve(mat) -> LawXXParams
    assert hasattr(law_mod, "resolve"), f"Missing resolve in {law_mod.__name__}"
    p_res = law_mod.resolve(mat)
    assert isinstance(p_res, params_cls)

    # 4. Check extra_shapes(mat, nip=1) -> dict
    assert hasattr(law_mod, "extra_shapes"), f"Missing extra_shapes in {law_mod.__name__}"
    shapes1 = law_mod.extra_shapes(mat, nip=1)
    assert isinstance(shapes1, dict)
    shapes_multi = law_mod.extra_shapes(mat, nip=4)
    assert isinstance(shapes_multi, dict)

    # 5. Check needs_defgrad(mat) -> bool
    assert hasattr(law_mod, "needs_defgrad"), f"Missing needs_defgrad in {law_mod.__name__}"
    nd = law_mod.needs_defgrad(mat)
    assert isinstance(nd, bool)

    # 6. Check solid_update(mat, sig, deps, ...) -> (sig, epsp, c)
    assert hasattr(law_mod, "solid_update"), f"Missing solid_update in {law_mod.__name__}"
    sig0 = np.zeros(6, dtype=np.float64)
    deps0 = np.array([0.001, -0.0003, -0.0003, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_new, epsp_new, c_solid = law_mod.solid_update(mat, sig0, deps0, epsp=0.0, dt=1.0e-6, return_sound_speed=True)
    assert sig_new.shape == (6,)
    assert np.all(np.isfinite(sig_new))
    assert np.isfinite(epsp_new)
    assert c_solid > 0.0

    # 7. Check shell_update(mat, sig, deps, ...) -> (sig, epsp, c)
    assert hasattr(law_mod, "shell_update"), f"Missing shell_update in {law_mod.__name__}"
    sig0_sh = np.zeros(3, dtype=np.float64)
    deps0_sh = np.array([0.001, -0.0003, 0.0], dtype=np.float64)
    sig_sh_new, epsp_sh_new, c_shell = law_mod.shell_update(mat, sig0_sh, deps0_sh, epsp=0.0, dt=1.0e-6, return_sound_speed=True)
    assert sig_sh_new.shape == (3,)
    assert np.all(np.isfinite(sig_sh_new))
    assert np.isfinite(epsp_sh_new)
    assert c_shell > 0.0

    # 8. Check sound_speed(mat)
    assert hasattr(law_mod, "sound_speed"), f"Missing sound_speed in {law_mod.__name__}"
    ss = law_mod.sound_speed(mat)
    assert float(ss) > 0.0

    # 9. Check solid_tangent & consistent_solid_tangent
    assert hasattr(law_mod, "solid_tangent"), f"Missing solid_tangent in {law_mod.__name__}"
    assert hasattr(law_mod, "consistent_solid_tangent"), f"Missing consistent_solid_tangent in {law_mod.__name__}"
    ctan_sol = law_mod.solid_tangent(mat, sig=sig_new, epsp=epsp_new)
    assert ctan_sol.shape == (6, 6)
    assert np.all(np.isfinite(ctan_sol))

    # 10. Check shell_tangent & consistent_shell_tangent
    assert hasattr(law_mod, "shell_tangent"), f"Missing shell_tangent in {law_mod.__name__}"
    assert hasattr(law_mod, "consistent_shell_tangent"), f"Missing consistent_shell_tangent in {law_mod.__name__}"
    ctan_sh = law_mod.shell_tangent(mat, sig=sig_sh_new, epsp=epsp_sh_new)
    assert ctan_sh.shape == (3, 3)
    assert np.all(np.isfinite(ctan_sh))

