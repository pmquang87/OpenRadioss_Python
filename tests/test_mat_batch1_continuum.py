"""
Unit tests for BATCH 1 missing material laws in OpenRadioss:
LAW00, LAW11, LAW13, LAW16, LAW17, LAW18, LAW20, LAW23, LAW26, LAW41, LAW46.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import Material
from pyradioss.materials import (
    law00_void,
    law11_fluid_visc,
    law13_fluid_ale,
    law16_gray_ewing,
    law17_orth_elastic,
    law18_plas_iso,
    law20_rigid,
    law23_user_mat,
    law26_honeycomb_sesame,
    law41_jwl_burn,
    law46_kin_hard,
)


# ============================================================================
# LAW00: Void Material
# ============================================================================

def test_law00_void():
    mat = Material(
        id=1, law=0, rho0=1.2e-3, title="Void",
        params={"MAT_E": 1000.0, "MAT_NU": 0.25}
    )
    p = law00_void.build_law00(mat)
    assert isinstance(p, law00_void.Law00Params)
    assert p.young == 1000.0
    assert p.nu == 0.25
    assert p.rho0 == 1.2e-3

    resolved = law00_void.resolve(mat)
    assert isinstance(resolved, law00_void.Law00Params)
    assert law00_void.extra_shapes(mat) == {}
    assert not law00_void.needs_defgrad(mat)

    # Sound speed
    c_sol = law00_void.sound_speed(p, is_shell=False)
    c_sh = law00_void.sound_speed(p, is_shell=True)
    assert c_sol > 0.0
    assert c_sh > 0.0

    # 1D solid update
    sig1d = np.array([10.0, 5.0, 0.0, 1.0, 0.0, 0.0])
    deps1d = np.array([1e-3, 1e-3, 1e-3, 0.0, 0.0, 0.0])
    sig_out, epsp_out, c = law00_void.solid_update(p, sig1d, deps1d)
    assert np.allclose(sig_out, 0.0)
    assert epsp_out == 0.0
    assert c == c_sol

    # 2D solid update
    sig2d = np.zeros((4, 6))
    deps2d = np.ones((4, 6)) * 1e-3
    sig_out2, epsp_out2, c2 = law00_void.solid_update(p, sig2d, deps2d)
    assert sig_out2.shape == (4, 6)
    assert np.allclose(sig_out2, 0.0)
    assert np.allclose(epsp_out2, 0.0)

    # 1D shell update
    sig_sh1d = np.array([5.0, 5.0, 1.0])
    deps_sh1d = np.array([1e-3, 1e-3, 0.0])
    sig_sh_out, epsp_sh_out, c_s = law00_void.shell_update(p, sig_sh1d, deps_sh1d)
    assert np.allclose(sig_sh_out, 0.0)
    assert epsp_sh_out == 0.0
    assert c_s == c_sh

    # Tangents
    c_tan = law00_void.solid_tangent(p)
    assert c_tan.shape == (6, 6)
    assert np.allclose(c_tan, 0.0)
    q_tan = law00_void.shell_tangent(p)
    assert q_tan.shape == (3, 3)
    assert np.allclose(q_tan, 0.0)


# ============================================================================
# LAW11: Viscoelastic Navier-Stokes Fluid
# ============================================================================

def test_law11_fluid_visc():
    mat = Material(
        id=2, law=11, rho0=1000.0, title="Water",
        params={"MAT_C1": 2.2e9, "MAT_C0": 1500.0, "mu_vis": 1.0e-3}
    )
    p = law11_fluid_visc.build_law11(mat)
    assert isinstance(p, law11_fluid_visc.Law11Params)
    assert p.c1 == 2.2e9
    assert p.mu_vis == 1.0e-3

    resolved = law11_fluid_visc.resolve(mat)
    assert isinstance(resolved, law11_fluid_visc.Law11Params)
    assert "uvar11" in law11_fluid_visc.extra_shapes(mat)
    assert not law11_fluid_visc.needs_defgrad(mat)

    # Sound speed
    c_sol = law11_fluid_visc.sound_speed(p)
    assert math.isclose(c_sol, math.sqrt(2.2e9 / 1000.0), rel_tol=1e-5)

    # 1D solid update: volumetric compression produces positive pressure (-sig_ii)
    sig1d = np.zeros(6)
    deps1d = np.array([-1e-4, -1e-4, -1e-4, 1e-4, 0.0, 0.0])
    dt = 1e-5
    sig_out, epsp_out, c = law11_fluid_visc.solid_update(p, sig1d, deps1d, dt=dt)
    # Volumetric strain = -3e-4, Delta P = -K * (-3e-4) = +K * 3e-4, so sig_ii = -P < 0
    assert sig_out[0] < 0.0
    assert sig_out[1] < 0.0
    assert sig_out[2] < 0.0
    # Viscous shear: tau_xy = mu * deps_xy / dt = 1e-3 * 1e-4 / 1e-5 = 1e-2
    assert sig_out[3] > 0.0

    # 2D multi-element solid update
    sig2d = np.zeros((3, 6))
    deps2d = np.tile(deps1d, (3, 1))
    sig_out2, _, _ = law11_fluid_visc.solid_update(p, sig2d, deps2d, dt=dt)
    assert sig_out2.shape == (3, 6)

    # Shell update
    sig_sh = np.zeros(3)
    deps_sh = np.array([-1e-4, -1e-4, 1e-4])
    sig_sh_out, _, _ = law11_fluid_visc.shell_update(p, sig_sh, deps_sh, dt=dt)
    assert sig_sh_out[0] < 0.0
    assert sig_sh_out[2] > 0.0

    # Tangents
    c_tan = law11_fluid_visc.solid_tangent(p, dt=dt)
    assert c_tan.shape == (6, 6)
    assert c_tan[0, 0] > 0.0
    q_tan = law11_fluid_visc.shell_tangent(p, dt=dt)
    assert q_tan.shape == (3, 3)


# ============================================================================
# LAW13: Fluid ALE Formulation / Rigid Body Constraint
# ============================================================================

def test_law13_fluid_ale():
    mat = Material(
        id=3, law=13, rho0=7800.0, title="SteelRigid",
        params={"MAT_E": 210000.0, "MAT_NU": 0.3}
    )
    p = law13_fluid_ale.build_law13(mat)
    assert isinstance(p, law13_fluid_ale.Law13Params)
    assert p.young == 210000.0
    assert p.nu == 0.3

    assert law13_fluid_ale.resolve(mat) == p
    assert not law13_fluid_ale.needs_defgrad(mat)
    assert law13_fluid_ale.extra_shapes(mat) == {}

    # Sound speed
    c_sol = law13_fluid_ale.sound_speed(p, is_shell=False)
    c_sh = law13_fluid_ale.sound_speed(p, is_shell=True)
    assert c_sol > 0.0
    assert c_sh > 0.0

    # Solid update
    sig1d = np.zeros(6)
    deps1d = np.array([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig_out, epsp_out, c = law13_fluid_ale.solid_update(p, sig1d, deps1d)
    assert sig_out[0] > 0.0
    assert epsp_out == 0.0

    # Shell update
    sig_sh = np.zeros(3)
    deps_sh = np.array([1e-3, 0.0, 0.0])
    sig_sh_out, epsp_sh_out, _ = law13_fluid_ale.shell_update(p, sig_sh, deps_sh)
    assert sig_sh_out[0] > 0.0
    assert epsp_sh_out == 0.0

    # Tangents
    c_tan = law13_fluid_ale.solid_tangent(p)
    assert c_tan.shape == (6, 6)
    q_tan = law13_fluid_ale.shell_tangent(p)
    assert q_tan.shape == (3, 3)


# ============================================================================
# LAW16: Gray-Ewing Explosive EOS & Constitutive Model
# ============================================================================

def test_law16_gray_ewing():
    mat = Material(
        id=4, law=16, rho0=2000.0, title="Explosive",
        params={
            "MAT_E": 10000.0, "MAT_NU": 0.25,
            "MAT_SIGY": 50.0, "MAT_BETA": 100.0, "MAT_HARD": 0.5,
            "MAT_SRC": 0.1, "MAT_SRP": 1.0,
            "MAT_TMELT": 800.0, "MAT_T0": 300.0, "MAT_M": 1.0
        }
    )
    p = law16_gray_ewing.build_law16(mat)
    assert isinstance(p, law16_gray_ewing.Law16Params)
    assert p.a == 50.0
    assert p.b == 100.0
    assert p.t_melt == 800.0

    assert "temp16" in law16_gray_ewing.extra_shapes(mat)
    assert not law16_gray_ewing.needs_defgrad(mat)

    # Sound speed
    c = law16_gray_ewing.sound_speed(p)
    assert c > 0.0

    # Solid update with plastic yielding
    sig1d = np.zeros(6)
    deps1d = np.array([0.02, -0.01, -0.01, 0.0, 0.0, 0.0])
    sig_out, epsp_out, _ = law16_gray_ewing.solid_update(p, sig1d, deps1d, epsp=0.0, dt=1e-4)
    assert epsp_out > 0.0
    assert sig_out[0] > 0.0

    # Thermal melting test (T >= Tmelt -> deviator vanishes)
    extra_melt = {"temp": 850.0}
    sig_melt, _, _ = law16_gray_ewing.solid_update(p, sig1d, deps1d, extra=extra_melt)
    # Hydrostatic pressure only: deviatoric part should be 0
    p_hydro = (sig_melt[0] + sig_melt[1] + sig_melt[2]) / 3.0
    assert np.allclose(sig_melt[0], p_hydro)
    assert np.allclose(sig_melt[1], p_hydro)
    assert np.allclose(sig_melt[2], p_hydro)
    assert np.allclose(sig_melt[3:], 0.0)

    # Shell update
    sig_sh = np.zeros(3)
    deps_sh = np.array([0.02, -0.01, 0.0])
    sig_sh_out, epsp_sh_out, _ = law16_gray_ewing.shell_update(p, sig_sh, deps_sh)
    assert epsp_sh_out > 0.0

    # Tangents
    assert law16_gray_ewing.solid_tangent(p).shape == (6, 6)
    assert law16_gray_ewing.shell_tangent(p).shape == (3, 3)


# ============================================================================
# LAW17: 3D Orthotropic Elastic Material
# ============================================================================

def test_law17_orth_elastic():
    mat = Material(
        id=5, law=17, rho0=1500.0, title="Composite",
        params={
            "MAT_E1": 150000.0, "MAT_E2": 10000.0, "MAT_E3": 10000.0,
            "MAT_NU12": 0.28, "MAT_NU23": 0.4, "MAT_NU31": 0.02,
            "MAT_G12": 5000.0, "MAT_G23": 3500.0, "MAT_G31": 5000.0
        }
    )
    p = law17_orth_elastic.build_law17(mat)
    assert isinstance(p, law17_orth_elastic.Law17Params)
    assert p.e1 == 150000.0
    assert p.e2 == 10000.0
    assert p.g12 == 5000.0

    assert law17_orth_elastic.extra_shapes(mat) == {}
    assert not law17_orth_elastic.needs_defgrad(mat)

    # Sound speed
    c_sol = law17_orth_elastic.sound_speed(p, is_shell=False)
    c_sh = law17_orth_elastic.sound_speed(p, is_shell=True)
    assert c_sol > 0.0
    assert c_sh > 0.0

    # Solid update
    sig1d = np.zeros(6)
    deps1d = np.array([1e-3, 1e-3, 0.0, 1e-3, 0.0, 0.0])
    sig_out, epsp_out, c = law17_orth_elastic.solid_update(p, sig1d, deps1d)
    assert sig_out[0] > sig_out[1]  # E1 > E2
    assert sig_out[3] == p.g12 * 1e-3
    assert epsp_out == 0.0

    # Shell update
    sig_sh = np.zeros(3)
    deps_sh = np.array([1e-3, 1e-3, 1e-3])
    sig_sh_out, epsp_sh_out, _ = law17_orth_elastic.shell_update(p, sig_sh, deps_sh)
    assert sig_sh_out[0] > sig_sh_out[1]
    assert sig_sh_out[2] == p.g12 * 1e-3

    # Tangents
    c_mat = law17_orth_elastic.solid_tangent(p)
    assert c_mat.shape == (6, 6)
    assert c_mat[0, 0] > c_mat[1, 1]
    q_mat = law17_orth_elastic.shell_tangent(p)
    assert q_mat.shape == (3, 3)


# ============================================================================
# LAW18: Isotropic Hardening Plasticity with Thermal Softening
# ============================================================================

def test_law18_plas_iso():
    mat = Material(
        id=6, law=18, rho0=7800.0, title="ThermalSteel",
        params={
            "MAT_E": 210000.0, "MAT_NU": 0.3,
            "MAT_A": 300.0, "MAT_B": 400.0, "MAT_HARD": 0.5,
            "MAT_T0": 293.15, "MAT_SPHEAT": 450.0, "MAT_TMELT": 1500.0
        }
    )
    p = law18_plas_iso.build_law18(mat)
    assert isinstance(p, law18_plas_iso.Law18Params)
    assert p.a == 300.0
    assert p.cp == 450.0

    assert "temp18" in law18_plas_iso.extra_shapes(mat)
    assert not law18_plas_iso.needs_defgrad(mat)

    # Sound speed
    assert law18_plas_iso.sound_speed(p) > 0.0

    # Solid update with plastic deformation
    sig1d = np.zeros(6)
    deps1d = np.array([0.01, -0.005, -0.005, 0.0, 0.0, 0.0])
    sig_out, epsp_out, _ = law18_plas_iso.solid_update(p, sig1d, deps1d, epsp=0.0)
    assert epsp_out > 0.0
    assert sig_out[0] > 0.0

    # Shell update
    sig_sh = np.zeros(3)
    deps_sh = np.array([0.01, -0.005, 0.0])
    sig_sh_out, epsp_sh_out, _ = law18_plas_iso.shell_update(p, sig_sh, deps_sh)
    assert epsp_sh_out > 0.0

    # Tangents
    assert law18_plas_iso.solid_tangent(p).shape == (6, 6)
    assert law18_plas_iso.shell_tangent(p).shape == (3, 3)


# ============================================================================
# LAW20: Rigid Material Handler
# ============================================================================

def test_law20_rigid():
    mat = Material(
        id=7, law=20, rho0=2700.0, title="RigidAlu",
        params={"MAT_E": 70000.0, "MAT_NU": 0.33, "MAT1": 1, "MAT2": 2}
    )
    p = law20_rigid.build_law20(mat)
    assert isinstance(p, law20_rigid.Law20Params)
    assert p.young == 70000.0
    assert p.mat1 == 1

    assert law20_rigid.extra_shapes(mat) == {}
    assert not law20_rigid.needs_defgrad(mat)

    # Sound speed
    assert law20_rigid.sound_speed(p) > 0.0

    # Solid update: rigid element without plastic deformation
    sig1d = np.zeros(6)
    deps1d = np.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig_out, epsp_out, _ = law20_rigid.solid_update(p, sig1d, deps1d)
    assert epsp_out == 0.0
    assert sig_out[0] > 0.0

    # Shell update
    sig_sh = np.zeros(3)
    deps_sh = np.array([1e-4, 0.0, 0.0])
    sig_sh_out, epsp_sh_out, _ = law20_rigid.shell_update(p, sig_sh, deps_sh)
    assert epsp_sh_out == 0.0

    # Tangents
    assert law20_rigid.solid_tangent(p).shape == (6, 6)
    assert law20_rigid.shell_tangent(p).shape == (3, 3)


# ============================================================================
# LAW23: General User-Defined Elastoplastic Material
# ============================================================================

def test_law23_user_mat():
    mat = Material(
        id=8, law=23, rho0=7800.0, title="UserMat",
        params={
            "MAT_E": 200000.0, "MAT_NU": 0.3,
            "MAT_SIGY": 250.0, "MAT_BETA": 300.0, "MAT_HARD": 0.3,
            "MAT_SRC": 0.05, "MAT_SRP": 1.0, "MAT_ETAN": 2000.0
        }
    )
    p = law23_user_mat.build_law23(mat)
    assert isinstance(p, law23_user_mat.Law23Params)
    assert p.a == 250.0
    assert p.etan == 2000.0
    assert p.hl > 0.0

    assert "epsp23" in law23_user_mat.extra_shapes(mat)
    assert not law23_user_mat.needs_defgrad(mat)

    # Sound speed
    assert law23_user_mat.sound_speed(p) > 0.0

    # Solid update with plastic yielding
    sig1d = np.zeros(6)
    deps1d = np.array([0.01, -0.005, -0.005, 0.0, 0.0, 0.0])
    sig_out, epsp_out, _ = law23_user_mat.solid_update(p, sig1d, deps1d, dt=1e-4)
    assert epsp_out > 0.0
    assert sig_out[0] > 0.0

    # Shell update
    sig_sh = np.zeros(3)
    deps_sh = np.array([0.01, -0.005, 0.0])
    sig_sh_out, epsp_sh_out, _ = law23_user_mat.shell_update(p, sig_sh, deps_sh, dt=1e-4)
    assert epsp_sh_out > 0.0

    # Tangents
    assert law23_user_mat.solid_tangent(p).shape == (6, 6)
    assert law23_user_mat.shell_tangent(p).shape == (3, 3)


# ============================================================================
# LAW26: Honeycomb with SESAME Tabular EOS
# ============================================================================

def test_law26_honeycomb_sesame():
    mat = Material(
        id=9, law=26, rho0=500.0, title="Honeycomb",
        params={
            "MAT_E": 10000.0, "MAT_NU": 0.2,
            "MAT_SIGY": 20.0, "MAT_BETA": 50.0, "MAT_HARD": 0.8,
            "MAT_SRC": 0.02, "MAT_SRP": 1.0, "MAT_TMELT": 600.0
        }
    )
    p = law26_honeycomb_sesame.build_law26(mat)
    assert isinstance(p, law26_honeycomb_sesame.Law26Params)
    assert p.a == 20.0
    assert p.young == 10000.0

    assert "temp26" in law26_honeycomb_sesame.extra_shapes(mat)
    assert not law26_honeycomb_sesame.needs_defgrad(mat)

    # Sound speed
    assert law26_honeycomb_sesame.sound_speed(p) > 0.0

    # Solid update
    sig1d = np.zeros(6)
    deps1d = np.array([0.01, -0.005, -0.005, 0.0, 0.0, 0.0])
    sig_out, epsp_out, _ = law26_honeycomb_sesame.solid_update(p, sig1d, deps1d, dt=1e-4)
    assert epsp_out > 0.0
    assert sig_out[0] > 0.0

    # Shell update
    sig_sh = np.zeros(3)
    deps_sh = np.array([0.01, -0.005, 0.0])
    sig_sh_out, epsp_sh_out, _ = law26_honeycomb_sesame.shell_update(p, sig_sh, deps_sh, dt=1e-4)
    assert epsp_sh_out > 0.0

    # Tangents
    assert law26_honeycomb_sesame.solid_tangent(p).shape == (6, 6)
    assert law26_honeycomb_sesame.shell_tangent(p).shape == (3, 3)


# ============================================================================
# LAW41: Lee-Tarver JWL Reactive Burn
# ============================================================================

def test_law41_jwl_burn():
    mat = Material(
        id=10, law=41, rho0=1.84, title="PBX9502",
        params={
            "A": 600.0e9, "B": 12.0e9, "R1": 4.5, "R2": 1.5, "omega": 0.35, "E0": 4.0e9,
            "IREAC": 1, "I": 1.0e6, "G1": 1.0e8, "x": 4.0, "y": 1.0
        }
    )
    p = law41_jwl_burn.build_law41(mat)
    assert isinstance(p, law41_jwl_burn.Law41Params)
    assert p.a_r == 600.0e9
    assert p.e0 == 4.0e9
    assert p.i_ign == 1.0e6

    assert "burn41" in law41_jwl_burn.extra_shapes(mat)
    assert not law41_jwl_burn.needs_defgrad(mat)

    # Sound speed
    assert law41_jwl_burn.sound_speed(p) > 0.0

    # Solid update under shock compression (tr(deps) < 0)
    sig1d = np.zeros(6)
    deps1d = np.array([-0.05, -0.05, -0.05, 0.0, 0.0, 0.0])
    sig_out, epsp_out, c = law41_jwl_burn.solid_update(p, sig1d, deps1d, dt=1e-6)
    # Detonation / reactive burn produces large compressive hydrostatic stress
    assert sig_out[0] < 0.0
    assert sig_out[1] < 0.0
    assert sig_out[2] < 0.0
    assert np.allclose(sig_out[3:], 0.0)
    assert epsp_out >= 0.0

    # Shell update
    sig_sh = np.zeros(3)
    deps_sh = np.array([-0.05, -0.05, 0.0])
    sig_sh_out, _, _ = law41_jwl_burn.shell_update(p, sig_sh, deps_sh, dt=1e-6)
    assert sig_sh_out[0] < 0.0

    # Tangents
    assert law41_jwl_burn.solid_tangent(p).shape == (6, 6)
    assert law41_jwl_burn.shell_tangent(p).shape == (3, 3)


# ============================================================================
# LAW46: Combined Isotropic & Kinematic Hardening Plasticity
# ============================================================================

def test_law46_kin_hard():
    mat = Material(
        id=11, law=46, rho0=7800.0, title="KinSteel",
        params={
            "MAT_E": 200000.0, "MAT_NU": 0.3,
            "MAT_SIGY": 200.0, "MAT_C": 50000.0, "GAMMA": 250.0,
            "MAT_H": 500.0, "MAT_B": 10.0, "MAT_Q": 100.0
        }
    )
    p = law46_kin_hard.build_law46(mat)
    assert isinstance(p, law46_kin_hard.Law46Params)
    assert p.sig_y == 200.0
    assert p.c_kin == 50000.0
    assert p.gamma_kin == 250.0

    assert "alpha46" in law46_kin_hard.extra_shapes(mat)
    assert not law46_kin_hard.needs_defgrad(mat)

    # Sound speed
    assert law46_kin_hard.sound_speed(p) > 0.0

    # Solid update with kinematic backstress evolution
    sig1d = np.zeros(6)
    deps1d = np.array([0.01, -0.005, -0.005, 0.0, 0.0, 0.0])
    extra = {}
    sig_out, epsp_out, _ = law46_kin_hard.solid_update(p, sig1d, deps1d, epsp=0.0, extra=extra)
    assert epsp_out > 0.0
    assert sig_out[0] > 0.0
    assert "alpha46" in extra
    alpha = extra["alpha46"]
    # Backstress should develop tensile component in direction 1
    assert alpha[0, 0] > 0.0

    # Shell update
    sig_sh = np.zeros(3)
    deps_sh = np.array([0.01, -0.005, 0.0])
    sig_sh_out, epsp_sh_out, _ = law46_kin_hard.shell_update(p, sig_sh, deps_sh)
    assert epsp_sh_out > 0.0

    # Tangents
    assert law46_kin_hard.solid_tangent(p).shape == (6, 6)
    assert law46_kin_hard.shell_tangent(p).shape == (3, 3)
