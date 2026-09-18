"""Unit tests for Batch 3 OpenRadioss Material Laws (Anisotropic, Viscoelastic & Cyclic Plasticity).

Laws tested:
  - LAW72: Hill Anisotropic Yield with MMC Ductile Fracture (pyradioss.materials.law72_hill_mmc)
  - LAW75: Porous Carroll-Holt P-alpha & Phase Transformation (pyradioss.materials.law75_thermo_trans)
  - LAW77: Nonlinear Viscoelastic Foam & Air Interaction (pyradioss.materials.law77_visc_poly)
  - LAW78: Yoshida-Uemori Two-Surface Cyclic Plasticity (pyradioss.materials.law78_yoshida)
  - LAW80: Ramberg-Osgood Plasticity & Steel Kinetics (pyradioss.materials.law80_ramberg)
  - LAW84: Mooney-Rivlin & Swift-Voce Orthotropic Plasticity (pyradioss.materials.law84_mooney_rivlin)
  - LAW85: Void Material with Shell Transverse Pinching (pyradioss.materials.law85_void_pinch)
  - LAW86: Orthotropic Honeycomb Shell with Mixed Hardening (pyradioss.materials.law86_honeycomb_shell)
  - LAW91: Transverse Pinching Shell Formulation with Elastoplasticity (pyradioss.materials.law91_pinch_shell)
  - LAW96: Thermo-Elasto-Viscoplastic Polymer Formulation (pyradioss.materials.law96_thermo_visc)
  - LAW97: JWL-Baker Explosive EOS & Orthotropic Non-linear Formulation (pyradioss.materials.law97_orth_nonlinear)
"""

import math
import numpy as np
import pytest

from pyradioss.model.entities import Material
from pyradioss.materials import (
    law72_hill_mmc,
    law75_thermo_trans,
    law77_visc_poly,
    law78_yoshida,
    law80_ramberg,
    law84_mooney_rivlin,
    law85_void_pinch,
    law86_honeycomb_shell,
    law91_pinch_shell,
    law96_thermo_visc,
    law97_orth_nonlinear,
)


# ============================================================================
# LAW72 Tests
# ============================================================================

def test_law72_params_and_resolve():
    mat = Material(
        id=72,
        law=72,
        rho0=7.8e-3,
        title="Mat_LAW72",
        params={
            "E": 210000.0,
            "nu": 0.3,
            "sigy": 250.0,
            "b": 400.0,
            "n": 0.3,
            "h_f": 1.1,
            "h_g": 0.9,
            "c1": 0.1,
            "c2": 300.0,
            "c3": 1.0,
        },
    )
    p = law72_hill_mmc.build_law72(mat)
    assert isinstance(p, law72_hill_mmc.Law72Params)
    assert p.young == 210000.0
    assert p.nu == 0.3
    assert p.sigy == 250.0
    assert p.c1 == 0.1

    p_resolved = law72_hill_mmc.resolve(mat)
    assert isinstance(p_resolved, law72_hill_mmc.Law72Params)
    assert law72_hill_mmc.needs_defgrad(mat) is False

    shapes_single = law72_hill_mmc.extra_shapes(mat)
    assert "uvar72" in shapes_single
    assert shapes_single["uvar72"] == (1,)

    shapes_nip = law72_hill_mmc.extra_shapes(mat, nip=3)
    assert shapes_nip["uvar72"] == (3, 1)


def test_law72_updates_and_tangents():
    p = law72_hill_mmc.build_law72(young=200000.0, nu=0.3, rho0=7.8e-3, sigy=200.0, b=300.0, n=0.5)

    # Sound speed
    c_s = law72_hill_mmc.sound_speed(p, is_shell=False)
    c_sh = law72_hill_mmc.sound_speed(p, is_shell=True)
    assert c_s > 0.0
    assert c_sh > 0.0

    # 1D Solid update
    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([0.005, -0.001, -0.001, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_new, epsp_new, c = law72_hill_mmc.solid_update(p, sig0, deps, epsp=0.0, dt=1.0e-6)
    assert sig_new.shape == (6,)
    assert sig_new[0] > 0.0
    assert epsp_new >= 0.0

    # 2D Batch Solid update
    sig_batch = np.zeros((3, 6), dtype=np.float64)
    deps_batch = np.tile(deps, (3, 1))
    sig_out, epsp_out, c_out = law72_hill_mmc.solid_update(p, sig_batch, deps_batch, epsp=np.zeros(3), dt=1.0e-6)
    assert sig_out.shape == (3, 6)
    assert len(epsp_out) == 3

    # Shell update
    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([0.004, -0.001, 0.0], dtype=np.float64)
    sig_sh_new, ep_sh, c_sh_res = law72_hill_mmc.shell_update(p, sig_sh, deps_sh, epsp=0.0, dt=1.0e-6)
    assert sig_sh_new.shape == (3,)
    assert sig_sh_new[0] > 0.0

    # Tangents
    tan_sol = law72_hill_mmc.solid_tangent(p, sig=sig_batch)
    assert tan_sol.shape == (3, 6, 6)
    tan_sh = law72_hill_mmc.shell_tangent(p, sig=np.zeros((2, 3)))
    assert tan_sh.shape == (2, 3, 3)


# ============================================================================
# LAW75 Tests
# ============================================================================

def test_law75_params_and_resolve():
    mat = Material(
        id=75,
        law=75,
        rho0=2.5,
        title="Mat_LAW75",
        params={
            "E": 50000.0,
            "nu": 0.22,
            "pe": 15.0,
            "ps": 120.0,
            "nn": 2.5,
        },
    )
    p = law75_thermo_trans.build_law75(mat)
    assert isinstance(p, law75_thermo_trans.Law75Params)
    assert p.young == 50000.0
    assert p.pe == 15.0
    assert p.ps == 120.0

    assert law75_thermo_trans.needs_defgrad(mat) is False
    assert law75_thermo_trans.extra_shapes(mat)["uvar75"] == (4,)
    assert law75_thermo_trans.extra_shapes(mat, nip=4)["uvar75"] == (4, 4)


def test_law75_updates_and_tangents():
    p = law75_thermo_trans.build_law75(young=60000.0, nu=0.25, rho0=2.7, pe=20.0, ps=150.0)

    # 1D Solid update
    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([-0.002, -0.002, -0.002, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_new, epsp_new, c = law75_thermo_trans.solid_update(p, sig0, deps, dt=1.0e-6)
    assert sig_new.shape == (6,)
    assert c > 0.0

    # Shell update
    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([-0.001, -0.001, 0.0], dtype=np.float64)
    sig_sh_new, ep_sh, c_sh = law75_thermo_trans.shell_update(p, sig_sh, deps_sh, dt=1.0e-6)
    assert sig_sh_new.shape == (3,)

    # Tangents
    assert law75_thermo_trans.solid_tangent(p).shape == (1, 6, 6)
    assert law75_thermo_trans.shell_tangent(p).shape == (1, 3, 3)


# ============================================================================
# LAW77 Tests
# ============================================================================

def test_law77_params_and_resolve():
    mat = Material(
        id=77,
        law=77,
        rho0=0.08,
        title="Mat_LAW77",
        params={
            "E0": 10.0,
            "nu": 0.15,
            "EMAX": 80.0,
            "EPSMAX": 0.85,
            "P0": 0.1013,
            "GAMMA": 1.4,
        },
    )
    p = law77_visc_poly.build_law77(mat)
    assert isinstance(p, law77_visc_poly.Law77Params)
    assert p.e0 == 10.0
    assert p.emax == 80.0
    assert p.epsmax == 0.85

    assert law77_visc_poly.needs_defgrad(mat) is False
    assert law77_visc_poly.extra_shapes(mat)["uvar77"] == (23,)


def test_law77_updates_and_tangents():
    p = law77_visc_poly.build_law77(e0=8.0, nu=0.1, emax=60.0, epsmax=0.75, rho0=0.06)

    # 1D Solid update (foam compression)
    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([-0.05, -0.01, -0.01, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_new, epsp_new, c = law77_visc_poly.solid_update(p, sig0, deps, dt=1.0e-6)
    assert sig_new.shape == (6,)
    assert sig_new[0] < 0.0  # compressive stress

    # Shell update
    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([-0.03, -0.005, 0.0], dtype=np.float64)
    sig_sh_new, ep_sh, c_sh = law77_visc_poly.shell_update(p, sig_sh, deps_sh, dt=1.0e-6)
    assert sig_sh_new.shape == (3,)

    # Tangents
    assert law77_visc_poly.solid_tangent(p).shape == (1, 6, 6)
    assert law77_visc_poly.shell_tangent(p).shape == (1, 3, 3)


# ============================================================================
# LAW78 Tests
# ============================================================================

def test_law78_params_and_resolve():
    mat = Material(
        id=78,
        law=78,
        rho0=7.8e-3,
        title="Mat_LAW78",
        params={
            "E": 210000.0,
            "nu": 0.3,
            "Y": 240.0,
            "C": 1500.0,
            "B": 350.0,
            "R_SAT": 120.0,
            "M": 15.0,
            "B_SAT": 250.0,
        },
    )
    p = law78_yoshida.build_law78(mat)
    assert isinstance(p, law78_yoshida.Law78Params)
    assert p.young == 210000.0
    assert p.yield_stress == 240.0
    assert p.cyu == 1500.0
    assert p.rsat == 120.0

    assert law78_yoshida.needs_defgrad(mat) is False
    assert law78_yoshida.extra_shapes(mat)["uvar78"] == (6,)


def test_law78_updates_and_cyclic():
    p = law78_yoshida.build_law78(young=200000.0, nu=0.3, rho0=7.8e-3, y=200.0, c=1000.0, b=300.0)

    # 1D Solid update
    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_new, epsp_new, c = law78_yoshida.solid_update(p, sig0, deps, epsp=0.0, dt=1.0e-6)
    assert sig_new.shape == (6,)
    assert sig_new[0] > 0.0
    assert epsp_new > 0.0

    # Shell update
    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([0.004, -0.001, 0.0], dtype=np.float64)
    sig_sh_new, ep_sh, c_sh = law78_yoshida.shell_update(p, sig_sh, deps_sh, epsp=0.0, dt=1.0e-6)
    assert sig_sh_new.shape == (3,)
    assert sig_sh_new[0] > 0.0

    # Tangents
    assert law78_yoshida.solid_tangent(p).shape == (1, 6, 6)
    assert law78_yoshida.shell_tangent(p).shape == (1, 3, 3)


# ============================================================================
# LAW80 Tests
# ============================================================================

def test_law80_params_and_resolve():
    mat = Material(
        id=80,
        law=80,
        rho0=7.8e-3,
        title="Mat_LAW80",
        params={
            "E": 210000.0,
            "nu": 0.3,
            "SIGY0": 350.0,
            "N": 6.0,
            "ALPHA0": 0.03,
            "EPS0": 0.002,
        },
    )
    p = law80_ramberg.build_law80(mat)
    assert isinstance(p, law80_ramberg.Law80Params)
    assert p.young == 210000.0
    assert p.sigy0 == 350.0
    assert p.n_ramberg == 6.0

    assert law80_ramberg.needs_defgrad(mat) is False
    assert law80_ramberg.extra_shapes(mat)["uvar80"] == (15,)


def test_law80_updates_and_tangents():
    p = law80_ramberg.build_law80(young=205000.0, nu=0.3, rho0=7.8e-3, sigy0=300.0, n_ramberg=5.0)

    # 1D Solid update
    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([0.004, -0.001, -0.001, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_new, epsp_new, c = law80_ramberg.solid_update(p, sig0, deps, epsp=0.0, dt=1.0e-6)
    assert sig_new.shape == (6,)
    assert sig_new[0] > 0.0

    # Shell update
    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([0.003, -0.001, 0.0], dtype=np.float64)
    sig_sh_new, ep_sh, c_sh = law80_ramberg.shell_update(p, sig_sh, deps_sh, epsp=0.0, dt=1.0e-6)
    assert sig_sh_new.shape == (3,)

    # Tangents
    assert law80_ramberg.solid_tangent(p).shape == (1, 6, 6)
    assert law80_ramberg.shell_tangent(p).shape == (1, 3, 3)


# ============================================================================
# LAW84 Tests
# ============================================================================

def test_law84_params_and_resolve():
    mat = Material(
        id=84,
        law=84,
        rho0=7.8e-3,
        title="Mat_LAW84",
        params={
            "E": 210000.0,
            "nu": 0.3,
            "SIGY": 280.0,
            "QVOCE": 180.0,
            "CVOCE": 12.0,
            "CSWIFT": 500.0,
            "NSWIFT": 0.25,
            "C10": 20.0,
            "C01": 5.0,
        },
    )
    p = law84_mooney_rivlin.build_law84(mat)
    assert isinstance(p, law84_mooney_rivlin.Law84Params)
    assert p.young == 210000.0
    assert p.k0 == 280.0
    assert p.qvoce == 180.0
    assert p.c10 == 20.0

    assert law84_mooney_rivlin.needs_defgrad(mat) is False
    assert law84_mooney_rivlin.extra_shapes(mat)["uvar84"] == (4,)


def test_law84_updates_and_tangents():
    p = law84_mooney_rivlin.build_law84(young=200000.0, nu=0.3, rho0=7.8e-3, sigy=250.0, qvoce=150.0)

    # 1D Solid update
    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([0.005, -0.001, -0.001, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_new, epsp_new, c = law84_mooney_rivlin.solid_update(p, sig0, deps, epsp=0.0, dt=1.0e-6)
    assert sig_new.shape == (6,)
    assert sig_new[0] > 0.0

    # Shell update
    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([0.004, -0.001, 0.0], dtype=np.float64)
    sig_sh_new, ep_sh, c_sh = law84_mooney_rivlin.shell_update(p, sig_sh, deps_sh, epsp=0.0, dt=1.0e-6)
    assert sig_sh_new.shape == (3,)

    # Tangents
    assert law84_mooney_rivlin.solid_tangent(p).shape == (1, 6, 6)
    assert law84_mooney_rivlin.shell_tangent(p).shape == (1, 3, 3)


# ============================================================================
# LAW85 Tests
# ============================================================================

def test_law85_params_and_resolve():
    mat = Material(
        id=85,
        law=85,
        rho0=1.0,
        title="Mat_LAW85",
        params={
            "E": 1500.0,
            "nu": 0.28,
        },
    )
    p = law85_void_pinch.build_law85(mat)
    assert isinstance(p, law85_void_pinch.Law85Params)
    assert p.young == 1500.0
    assert p.nu == 0.28

    assert law85_void_pinch.needs_defgrad(mat) is False
    assert law85_void_pinch.extra_shapes(mat)["uvar85"] == (2,)


def test_law85_updates_and_tangents():
    p = law85_void_pinch.build_law85(young=1200.0, nu=0.3, rho0=1.2)

    # 1D Solid update
    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([0.001, 0.001, -0.002, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_new, epsp_new, c = law85_void_pinch.solid_update(p, sig0, deps, dt=1.0e-6)
    assert sig_new.shape == (6,)
    assert c > 0.0

    # Shell update
    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([0.002, -0.001, 0.0], dtype=np.float64)
    sig_sh_new, ep_sh, c_sh = law85_void_pinch.shell_update(p, sig_sh, deps_sh, dt=1.0e-6)
    assert sig_sh_new.shape == (3,)

    # Tangents
    assert law85_void_pinch.solid_tangent(p).shape == (1, 6, 6)
    assert law85_void_pinch.shell_tangent(p).shape == (1, 3, 3)


# ============================================================================
# LAW86 Tests
# ============================================================================

def test_law86_params_and_resolve():
    mat = Material(
        id=86,
        law=86,
        rho0=0.15,
        title="Mat_LAW86",
        params={
            "E": 1200.0,
            "nu": 0.25,
            "yield_stress": 25.0,
            "h_iso": 60.0,
            "h_kin": 40.0,
        },
    )
    p = law86_honeycomb_shell.build_law86(mat)
    assert isinstance(p, law86_honeycomb_shell.Law86Params)
    assert p.young == 1200.0
    assert p.yield_stress == 25.0
    assert p.h_iso == 60.0

    assert law86_honeycomb_shell.needs_defgrad(mat) is False
    assert law86_honeycomb_shell.extra_shapes(mat)["uvar86"] == (5,)


def test_law86_updates_and_tangents():
    p = law86_honeycomb_shell.build_law86(young=1000.0, nu=0.25, rho0=0.1, yield_stress=20.0, h_iso=50.0)

    # 1D Solid update
    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([0.03, -0.01, -0.01, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_new, epsp_new, c = law86_honeycomb_shell.solid_update(p, sig0, deps, epsp=0.0, dt=1.0e-6)
    assert sig_new.shape == (6,)
    assert sig_new[0] > 0.0
    assert epsp_new > 0.0

    # Shell update
    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([0.04, -0.01, 0.0], dtype=np.float64)
    sig_sh_new, ep_sh, c_sh = law86_honeycomb_shell.shell_update(p, sig_sh, deps_sh, epsp=0.0, dt=1.0e-6)
    assert sig_sh_new.shape == (3,)
    assert sig_sh_new[0] > 0.0
    assert ep_sh > 0.0

    # Tangents
    assert law86_honeycomb_shell.solid_tangent(p).shape == (1, 6, 6)
    assert law86_honeycomb_shell.shell_tangent(p).shape == (1, 3, 3)


# ============================================================================
# LAW91 Tests
# ============================================================================

def test_law91_params_and_resolve():
    mat = Material(
        id=91,
        law=91,
        rho0=1.0,
        title="Mat_LAW91",
        params={
            "E": 1200.0,
            "nu": 0.3,
            "SIGMA_r": 25.0,
            "HM": 45.0,
        },
    )
    p = law91_pinch_shell.build_law91(mat)
    assert isinstance(p, law91_pinch_shell.Law91Params)
    assert p.young == 1200.0
    assert p.yield_stress == 25.0
    assert p.hardening == 45.0

    assert law91_pinch_shell.needs_defgrad(mat) is False
    assert law91_pinch_shell.extra_shapes(mat)["uvar91"] == (10,)
    assert law91_pinch_shell.extra_shapes(mat, nip=3)["uvar91"] == (3, 10)


def test_law91_updates_and_tangents():
    p = law91_pinch_shell.build_law91(young=1000.0, nu=0.3, rho0=1.0, yield_stress=20.0, hardening=50.0)

    # 1D Solid update
    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([0.03, -0.01, -0.01, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_new, epsp_new, c = law91_pinch_shell.solid_update(p, sig0, deps, epsp=0.0, dt=1.0e-6)
    assert sig_new.shape == (6,)
    assert sig_new[0] > 0.0
    assert epsp_new > 0.0

    # Batch Solid update
    sig_b = np.zeros((3, 6), dtype=np.float64)
    deps_b = np.tile(deps, (3, 1))
    s_out, ep_out, c_out = law91_pinch_shell.solid_update(p, sig_b, deps_b, epsp=np.zeros(3), dt=1.0e-6)
    assert s_out.shape == (3, 6)
    assert len(ep_out) == 3

    # Shell update with 3 components and transverse pinching
    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([0.04, -0.01, 0.0], dtype=np.float64)
    sig_sh_new, ep_sh, c_sh = law91_pinch_shell.shell_update(p, sig_sh, deps_sh, epsp=0.0, dt=1.0e-6)
    assert sig_sh_new.shape == (3,)
    assert sig_sh_new[0] > 0.0
    assert ep_sh > 0.0

    # Shell update with 6 components (transverse pinching normal stress)
    sig_sh6 = np.zeros(6, dtype=np.float64)
    deps_sh6 = np.array([0.02, 0.02, -0.04, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_sh6_new, ep_sh6, c_sh6 = law91_pinch_shell.shell_update(p, sig_sh6, deps_sh6, epsp=0.0, dt=1.0e-6)
    assert sig_sh6_new.shape == (6,)

    # Tangents
    assert law91_pinch_shell.solid_tangent(p).shape == (1, 6, 6)
    assert law91_pinch_shell.shell_tangent(p).shape == (1, 3, 3)


# ============================================================================
# LAW96 Tests
# ============================================================================

def test_law96_params_and_resolve():
    mat = Material(
        id=96,
        law=96,
        rho0=1.1,
        title="Mat_LAW96",
        params={
            "E": 1500.0,
            "nu": 0.38,
            "TANB": 0.25,
            "TANP": 0.12,
            "SIGY": 35.0,
            "SIGA": 20.0,
            "SIGB": 12.0,
            "SIGR": 6.0,
            "RA1": 40.0,
            "RB": 15.0,
            "RC": 0.4,
        },
    )
    p = law96_thermo_visc.build_law96(mat)
    assert isinstance(p, law96_thermo_visc.Law96Params)
    assert p.young == 1500.0
    assert p.tanb == 0.25
    assert p.tanp == 0.12
    assert p.sigy == 35.0
    assert p.siga == 20.0

    assert law96_thermo_visc.needs_defgrad(mat) is False
    assert law96_thermo_visc.extra_shapes(mat)["uvar96"] == (4,)
    assert law96_thermo_visc.extra_shapes(mat, nip=2)["uvar96"] == (2, 4)


def test_law96_updates_and_tangents():
    p = law96_thermo_visc.build_law96(
        young=1200.0,
        nu=0.35,
        rho0=1.0,
        tanb=0.2,
        tanp=0.1,
        sigy=30.0,
        siga=15.0,
        sigb=10.0,
        sigr=5.0,
    )

    # 1D Solid update
    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([0.05, -0.015, -0.015, 0.0, 0.0, 0.0], dtype=np.float64)
    sig_new, epsp_new, c = law96_thermo_visc.solid_update(p, sig0, deps, epsp=0.0, dt=1.0e-6)
    assert sig_new.shape == (6,)
    assert sig_new[0] > 0.0
    assert epsp_new > 0.0

    # Batch Solid update
    sig_b = np.zeros((2, 6), dtype=np.float64)
    deps_b = np.tile(deps, (2, 1))
    s_out, ep_out, c_out = law96_thermo_visc.solid_update(p, sig_b, deps_b, epsp=np.zeros(2), dt=1.0e-6)
    assert s_out.shape == (2, 6)
    assert len(ep_out) == 2

    # Shell update
    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([0.04, -0.01, 0.0], dtype=np.float64)
    sig_sh_new, ep_sh, c_sh = law96_thermo_visc.shell_update(p, sig_sh, deps_sh, epsp=0.0, dt=1.0e-6)
    assert sig_sh_new.shape == (3,)
    assert sig_sh_new[0] > 0.0

    # Tangents
    assert law96_thermo_visc.solid_tangent(p).shape == (1, 6, 6)
    assert law96_thermo_visc.shell_tangent(p).shape == (1, 3, 3)


# ============================================================================
# LAW97 Tests
# ============================================================================

def test_law97_params_and_resolve():
    mat = Material(
        id=97,
        law=97,
        rho0=1.63,
        title="Mat_LAW97",
        params={
            "D": 8500.0,
            "PCJ": 3.0e10,
            "E0": 9.0e9,
            "Omega": 0.38,
            "C": 1.2e9,
            "A1": 6.0e11,
            "R1": 4.8,
            "A2": 1.5e11,
            "R2": 1.6,
        },
    )
    p = law97_orth_nonlinear.build_law97(mat)
    assert isinstance(p, law97_orth_nonlinear.Law97Params)
    assert p.d == 8500.0
    assert p.pcj == 3.0e10
    assert p.e0 == 9.0e9
    assert p.w == 0.38
    assert p.a[0] == 6.0e11

    assert law97_orth_nonlinear.needs_defgrad(mat) is False
    assert law97_orth_nonlinear.extra_shapes(mat)["uvar97"] == (5,)
    assert law97_orth_nonlinear.extra_shapes(mat, nip=3)["uvar97"] == (3, 5)


def test_law97_updates_and_eos():
    p = law97_orth_nonlinear.build_law97(
        rho0=1.63,
        d=8000.0,
        pcj=2.8e10,
        e0=8.5e9,
        w=0.35,
        c=1.0e9,
    )

    # Sound speed
    c_s = law97_orth_nonlinear.sound_speed(p, is_shell=False)
    assert c_s >= 8000.0

    # 1D Solid update with explosive expansion
    sig0 = np.zeros(6, dtype=np.float64)
    deps = np.array([0.05, 0.05, 0.05, 0.0, 0.0, 0.0], dtype=np.float64)
    extra = {"time": 0.001}
    sig_new, epsp_new, c = law97_orth_nonlinear.solid_update(p, sig0, deps, extra=extra, dt=1.0e-6)
    assert sig_new.shape == (6,)
    assert c > 0.0

    # Shell update
    sig_sh = np.zeros(3, dtype=np.float64)
    deps_sh = np.array([0.02, 0.02, 0.0], dtype=np.float64)
    sig_sh_new, ep_sh, c_sh = law97_orth_nonlinear.shell_update(p, sig_sh, deps_sh, extra=extra, dt=1.0e-6)
    assert sig_sh_new.shape == (3,)

    # Tangents
    assert law97_orth_nonlinear.solid_tangent(p).shape == (1, 6, 6)
    assert law97_orth_nonlinear.shell_tangent(p).shape == (1, 3, 3)
