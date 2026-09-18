"""
Unit tests for BATCH 2 material laws in OpenRadioss:
LAW45, LAW53, LAW54, LAW55, LAW56, LAW59, LAW63, LAW64, LAW65, LAW68.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.model.entities import Material
from pyradioss.materials import (
    law45_orth_fabric,
    law53_tab_foam,
    law54_hyd_visc,
    law55_shell_orth,
    law56_orth_fail,
    law59_spring_rate,
    law63_composite_shell,
    law64_rate_comp,
    law65_thermo_visc,
    law68_tab_plas,
)


# ============================================================================
# LAW45: Viscoelastic Orthotropic Fabric
# ============================================================================

def test_law45_orth_fabric():
    mat = Material(
        id=45, law=45, rho0=1.5e-3, title="Fabric",
        params={
            "MAT_E": 2000.0, "MAT_NU": 0.25,
            "MAT_A": 100.0, "MAT_B": 200.0, "MAT_N": 0.5,
        }
    )
    p = law45_orth_fabric.build_law45(mat)
    assert isinstance(p, law45_orth_fabric.Law45Params)
    assert p.e == 2000.0
    assert p.nu == 0.25
    assert p.ca == 100.0
    assert p.cb == 200.0
    assert p.cn == 0.5

    assert isinstance(law45_orth_fabric.resolve(mat), law45_orth_fabric.Law45Params)
    assert "uvar45" in law45_orth_fabric.extra_shapes(mat)
    assert not law45_orth_fabric.needs_defgrad(mat)
    assert law45_orth_fabric.sound_speed(p) > 0.0

    # 1D solid update
    sig1d = np.zeros(6)
    deps1d = np.array([1e-3, 5e-4, 0.0, 1e-4, 0.0, 0.0])
    s_out, epsp_out, c = law45_orth_fabric.solid_update(p, sig1d, deps1d)
    assert s_out.shape == (6,)
    assert s_out[0] > 0.0
    assert c > 0.0

    # 2D solid update
    sig2d = np.zeros((3, 6))
    deps2d = np.full((3, 6), 5e-3)
    s2_out, ep2_out, c2 = law45_orth_fabric.solid_update(p, sig2d, deps2d)
    assert s2_out.shape == (3, 6)
    assert ep2_out.shape == (3,)
    assert c2.shape == (3,)

    # 1D & 2D shell update
    sig_sh = np.zeros(3)
    deps_sh = np.array([2e-3, 1e-3, 5e-4])
    s_sh, ep_sh, c_sh = law45_orth_fabric.shell_update(p, sig_sh, deps_sh)
    assert s_sh.shape == (3,)
    assert s_sh[0] > 0.0

    # Tangents
    ctan = law45_orth_fabric.solid_tangent(p)
    assert ctan.shape == (6, 6)
    qtan = law45_orth_fabric.shell_tangent(p)
    assert qtan.shape == (3, 3)
    qtan2d = law45_orth_fabric.shell_tangent(p, sig=np.zeros((2, 3)))
    assert qtan2d.shape == (2, 3, 3)


# ============================================================================
# LAW53: Tabulated Crushable Foam
# ============================================================================

def test_law53_tab_foam():
    mat = Material(
        id=53, law=53, rho0=0.08, title="Foam53",
        params={
            "MAT_E1": 30.0, "MAT_E2": 25.0, "MAT_G12": 10.0, "MAT_G23": 8.0,
            "sfac11": 1.5, "sfac22": 1.2,
        }
    )
    p = law53_tab_foam.build_law53(mat)
    assert isinstance(p, law53_tab_foam.Law53Params)
    assert p.e11 == 30.0
    assert p.e22 == 25.0
    assert p.g12 == 10.0

    assert "uvar53" in law53_tab_foam.extra_shapes(mat)
    assert not law53_tab_foam.needs_defgrad(mat)
    assert law53_tab_foam.sound_speed(p) > 0.0

    # Solid update
    sig1d = np.zeros(6)
    deps1d = np.array([-0.02, -0.02, -0.02, 0.0, 0.0, 0.0])
    s_out, ep_out, c = law53_tab_foam.solid_update(p, sig1d, deps1d)
    assert s_out[0] < 0.0  # compressive stress

    # Shell update
    sig_sh = np.zeros(3)
    deps_sh = np.array([-0.01, -0.01, 0.0])
    s_sh, ep_sh, c_sh = law53_tab_foam.shell_update(p, sig_sh, deps_sh)
    assert s_sh.shape == (3,)

    # Tangents
    assert law53_tab_foam.solid_tangent(p).shape == (6, 6)
    assert law53_tab_foam.shell_tangent(p).shape == (3, 3)


# ============================================================================
# LAW54: Hydrodynamic Viscous Fluid
# ============================================================================

def test_law54_hyd_visc():
    mat = Material(
        id=54, law=54, rho0=1000.0, title="Fluid54",
        params={
            "MAT_E": 1000.0, "MAT_NU": 0.3, "sig0": 20.0,
            "dc": 0.95, "pr": 0.1, "ps": 0.01,
        }
    )
    p = law54_hyd_visc.build_law54(mat)
    assert isinstance(p, law54_hyd_visc.Law54Params)
    assert p.e == 1000.0
    assert p.sig0 == 20.0
    assert p.pr == 0.1
    assert p.c_sound > 0.0

    assert "dmg54" in law54_hyd_visc.extra_shapes(mat)
    assert not law54_hyd_visc.needs_defgrad(mat)
    assert law54_hyd_visc.sound_speed(p) > 0.0

    # Solid update (viscous + plastic step)
    sig1d = np.zeros(6)
    deps1d = np.array([0.01, -0.005, -0.005, 0.002, 0.0, 0.0])
    s_out, ep_out, c = law54_hyd_visc.solid_update(p, sig1d, deps1d, dt=1e-5)
    assert s_out.shape == (6,)
    assert c > 0.0

    # Rupture test
    deps_large = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    s_rupt, ep_rupt, _ = law54_hyd_visc.solid_update(p, sig1d, deps_large, dt=1e-5)
    assert np.allclose(s_rupt, 0.0)

    # Shell update
    sig_sh = np.zeros(3)
    deps_sh = np.array([0.005, 0.005, 0.001])
    s_sh, ep_sh, _ = law54_hyd_visc.shell_update(p, sig_sh, deps_sh, dt=1e-5)
    assert s_sh.shape == (3,)

    # Tangents
    assert law54_hyd_visc.solid_tangent(p).shape == (6, 6)
    assert law54_hyd_visc.shell_tangent(p).shape == (3, 3)


# ============================================================================
# LAW55: Shell Orthotropic Elastic-Plastic Formulation
# ============================================================================

def test_law55_shell_orth():
    mat = Material(
        id=55, law=55, rho0=2000.0, title="OrthShell55",
        params={
            "MAT_E": 70000.0, "MAT_NU": 0.33, "sigy0": 200.0, "visc": 0.05,
        }
    )
    p = law55_shell_orth.build_law55(mat)
    assert isinstance(p, law55_shell_orth.Law55Params)
    assert p.e == 70000.0
    assert p.c_shell > 0.0

    assert "uvar55" in law55_shell_orth.extra_shapes(mat)
    assert not law55_shell_orth.needs_defgrad(mat)
    assert law55_shell_orth.sound_speed(p) > 0.0

    # Shell update
    sig_sh = np.zeros(3)
    deps_sh = np.array([0.005, 0.0, 0.0])
    s_sh, ep_sh, c = law55_shell_orth.shell_update(p, sig_sh, deps_sh)
    assert s_sh[0] > 0.0
    assert ep_sh > 0.0

    # Solid update
    sig_sol = np.zeros(6)
    deps_sol = np.array([0.005, -0.001, -0.001, 0.0, 0.0, 0.0])
    s_sol, ep_sol, _ = law55_shell_orth.solid_update(p, sig_sol, deps_sol)
    assert s_sol.shape == (6,)

    # Tangents
    assert law55_shell_orth.shell_tangent(p).shape == (3, 3)
    assert law55_shell_orth.solid_tangent(p).shape == (6, 6)


# ============================================================================
# LAW56: Orthotropic Elasto-Plastic with Failure
# ============================================================================

def test_law56_orth_fail():
    mat = Material(
        id=56, law=56, rho0=2700.0, title="OrthFail56",
        params={
            "MAT_E": 69000.0, "MAT_NU": 0.3, "sigy0": 150.0,
            "epsmax": 0.2, "epsr1": 0.15, "epsr2": 0.25,
        }
    )
    p = law56_orth_fail.build_law56(mat)
    assert isinstance(p, law56_orth_fail.Law56Params)
    assert p.e == 69000.0
    assert p.c_solid > 0.0
    assert p.c_shell > 0.0

    assert "uvar56" in law56_orth_fail.extra_shapes(mat)
    assert not law56_orth_fail.needs_defgrad(mat)
    assert law56_orth_fail.sound_speed(p) > 0.0

    # Solid update
    sig_sol = np.zeros(6)
    deps_sol = np.array([0.003, -0.001, -0.001, 0.0, 0.0, 0.0])
    s_sol, ep_sol, c = law56_orth_fail.solid_update(p, sig_sol, deps_sol)
    assert s_sol[0] > 0.0
    assert ep_sol > 0.0

    # Shell update
    sig_sh = np.zeros(3)
    deps_sh = np.array([0.003, 0.0, 0.0])
    s_sh, ep_sh, _ = law56_orth_fail.shell_update(p, sig_sh, deps_sh)
    assert s_sh.shape == (3,)

    # Tangents
    assert law56_orth_fail.solid_tangent(p).shape == (6, 6)
    assert law56_orth_fail.shell_tangent(p).shape == (3, 3)


# ============================================================================
# LAW59: Rate-Dependent Connector/Spring Plasticity
# ============================================================================

def test_law59_spring_rate():
    mat = Material(
        id=59, law=59, rho0=7800.0, title="Spring59",
        params={
            "MAT_E": 5000.0, "MAT_G": 2500.0,
            "sigy_n": 100.0, "sigy_t": 50.0,
        }
    )
    p = law59_spring_rate.build_law59(mat)
    assert isinstance(p, law59_spring_rate.Law59Params)
    assert p.e == 5000.0
    assert p.g == 2500.0
    assert p.c_sound > 0.0

    assert "eplas_n" in law59_spring_rate.extra_shapes(mat)
    assert not law59_spring_rate.needs_defgrad(mat)
    assert law59_spring_rate.sound_speed(p) > 0.0

    # Solid update (normal + shear components)
    sig_sol = np.zeros(6)
    deps_sol = np.array([0.0, 0.0, 0.03, 0.0, 0.03, 0.0])
    s_sol, ep_sol, c = law59_spring_rate.solid_update(p, sig_sol, deps_sol)
    assert s_sol[2] > 0.0
    assert s_sol[4] > 0.0
    assert ep_sol > 0.0

    # Shell update
    sig_sh = np.zeros(3)
    deps_sh = np.array([0.03, 0.0, 0.02])
    s_sh, ep_sh, _ = law59_spring_rate.shell_update(p, sig_sh, deps_sh)
    assert s_sh.shape == (3,)

    # Tangents
    assert law59_spring_rate.solid_tangent(p).shape == (6, 6)
    assert law59_spring_rate.shell_tangent(p).shape == (3, 3)


# ============================================================================
# LAW63: TRIP Steel Transformation-Induced Plasticity
# ============================================================================

def test_law63_composite_shell():
    mat = Material(
        id=63, law=63, rho0=7850.0, title="TRIP63",
        params={
            "MAT_E": 200000.0, "MAT_NU": 0.3, "MAT_CP": 450.0,
            "ahs": 300.0, "bhs": 800.0, "cm": 10.0, "cn": 0.2,
        }
    )
    p = law63_composite_shell.build_law63(mat)
    assert isinstance(p, law63_composite_shell.Law63Params)
    assert p.e == 200000.0
    assert p.ahs == 300.0
    assert p.c_shell > 0.0

    assert "uvar63" in law63_composite_shell.extra_shapes(mat)
    assert not law63_composite_shell.needs_defgrad(mat)
    assert law63_composite_shell.sound_speed(p) > 0.0

    # Shell update with plastic deformation
    sig_sh = np.zeros(3)
    deps_sh = np.array([0.005, 0.0, 0.0])
    s_sh, ep_sh, c = law63_composite_shell.shell_update(p, sig_sh, deps_sh)
    assert s_sh[0] > 0.0
    assert ep_sh > 0.0

    # Solid update
    sig_sol = np.zeros(6)
    deps_sol = np.array([0.005, -0.001, -0.001, 0.0, 0.0, 0.0])
    s_sol, ep_sol, _ = law63_composite_shell.solid_update(p, sig_sol, deps_sol)
    assert s_sol.shape == (6,)

    # Tangents
    assert law63_composite_shell.shell_tangent(p).shape == (3, 3)
    assert law63_composite_shell.solid_tangent(p).shape == (6, 6)


# ============================================================================
# LAW64: Rate-Dependent TRIP Steel Plasticity Model
# ============================================================================

def test_law64_rate_comp():
    mat = Material(
        id=64, law=64, rho0=7800.0, title="TRIP64",
        params={
            "MAT_E": 210000.0, "MAT_NU": 0.29, "MAT_CP": 480.0,
            "MAT_D": 1.2, "MAT_N": 0.8, "M64_Md": 360.0,
            "sigy1_default": 350.0, "sigy2_default": 850.0,
        }
    )
    p = law64_rate_comp.build_law64(mat)
    assert isinstance(p, law64_rate_comp.Law64Params)
    assert p.e == 210000.0
    assert p.md == 360.0
    assert p.c_shell > 0.0
    assert p.c_solid > 0.0

    assert "uvar64" in law64_rate_comp.extra_shapes(mat)
    assert not law64_rate_comp.needs_defgrad(mat)
    assert law64_rate_comp.sound_speed(p) > 0.0

    # Shell update
    sig_sh = np.zeros(3)
    deps_sh = np.array([0.004, 0.0, 0.0])
    s_sh, ep_sh, c = law64_rate_comp.shell_update(p, sig_sh, deps_sh)
    assert s_sh[0] > 0.0
    assert ep_sh > 0.0

    # Solid update
    sig_sol = np.zeros(6)
    deps_sol = np.array([0.004, -0.001, -0.001, 0.0, 0.0, 0.0])
    s_sol, ep_sol, _ = law64_rate_comp.solid_update(p, sig_sol, deps_sol)
    assert s_sol.shape == (6,)
    assert ep_sol > 0.0

    # Tangents
    assert law64_rate_comp.shell_tangent(p).shape == (3, 3)
    assert law64_rate_comp.solid_tangent(p).shape == (6, 6)


# ============================================================================
# LAW65: Tabulated Thermo-Viscoelastic Elastomer / Plastic Law
# ============================================================================

def test_law65_thermo_visc():
    mat = Material(
        id=65, law=65, rho0=1100.0, title="Elastomer65",
        params={
            "MAT_E": 500.0, "MAT_NU": 0.46,
            "sigy_load_default": 15.0, "sigy_unload_default": 8.0,
            "epsmax": 0.5,
        }
    )
    p = law65_thermo_visc.build_law65(mat)
    assert isinstance(p, law65_thermo_visc.Law65Params)
    assert p.e == 500.0
    assert p.epsmax == 0.5
    assert p.c_solid > 0.0
    assert p.c_shell > 0.0

    assert "uvar65" in law65_thermo_visc.extra_shapes(mat)
    assert not law65_thermo_visc.needs_defgrad(mat)
    assert law65_thermo_visc.sound_speed(p) > 0.0

    # Solid update
    sig_sol = np.zeros(6)
    deps_sol = np.array([0.05, -0.02, -0.02, 0.0, 0.0, 0.0])
    s_sol, ep_sol, c = law65_thermo_visc.solid_update(p, sig_sol, deps_sol)
    assert s_sol[0] > 0.0
    assert ep_sol > 0.0

    # Shell update
    sig_sh = np.zeros(3)
    deps_sh = np.array([0.05, 0.0, 0.0])
    s_sh, ep_sh, _ = law65_thermo_visc.shell_update(p, sig_sh, deps_sh)
    assert s_sh.shape == (3,)

    # Tangents
    assert law65_thermo_visc.solid_tangent(p).shape == (6, 6)
    assert law65_thermo_visc.shell_tangent(p).shape == (3, 3)


# ============================================================================
# LAW68: Honeycomb Orthotropic Tabulated Plasticity Law
# ============================================================================

def test_law68_tab_plas():
    mat = Material(
        id=68, law=68, rho0=500.0, title="Honeycomb68",
        params={
            "MAT_EA": 800.0, "MAT_EB": 600.0, "MAT_EC": 1200.0,
            "MAT_GAB": 300.0, "MAT_GBC": 250.0, "MAT_GCA": 350.0,
            "sigy_default": 40.0, "emx11": 0.1,
        }
    )
    p = law68_tab_plas.build_law68(mat)
    assert isinstance(p, law68_tab_plas.Law68Params)
    assert p.e11 == 800.0
    assert p.e22 == 600.0
    assert p.e33 == 1200.0
    assert p.c_sound > 0.0

    assert "uvar68" in law68_tab_plas.extra_shapes(mat)
    assert not law68_tab_plas.needs_defgrad(mat)
    assert law68_tab_plas.sound_speed(p) > 0.0

    # Solid update (elastic clamping)
    sig_sol = np.zeros(6)
    deps_sol = np.array([0.08, 0.05, 0.02, 0.01, 0.0, 0.0])
    s_sol, ep_sol, c = law68_tab_plas.solid_update(p, sig_sol, deps_sol)
    assert s_sol[0] <= 40.0  # clamped to sigy_default
    assert ep_sol > 0.0

    # Failure check
    deps_fail = np.array([0.15, 0.0, 0.0, 0.0, 0.0, 0.0])
    s_fail, _, _ = law68_tab_plas.solid_update(p, sig_sol, deps_fail)
    assert np.allclose(s_fail, 0.0)

    # Shell update
    sig_sh = np.zeros(3)
    deps_sh = np.array([0.04, 0.02, 0.01])
    s_sh, ep_sh, _ = law68_tab_plas.shell_update(p, sig_sh, deps_sh)
    assert s_sh.shape == (3,)

    # Tangents
    assert law68_tab_plas.solid_tangent(p).shape == (6, 6)
    assert law68_tab_plas.shell_tangent(p).shape == (3, 3)


# ============================================================================
# Generic Multi-law Consistency Tests
# ============================================================================

@pytest.mark.parametrize("mod, law_id, p_dict", [
    (law45_orth_fabric, 45, {"MAT_E": 1500.0, "MAT_NU": 0.3}),
    (law53_tab_foam, 53, {"MAT_E1": 40.0, "MAT_E2": 35.0}),
    (law54_hyd_visc, 54, {"MAT_E": 800.0, "MAT_NU": 0.28}),
    (law55_shell_orth, 55, {"MAT_E": 65000.0, "MAT_NU": 0.3}),
    (law56_orth_fail, 56, {"MAT_E": 72000.0, "MAT_NU": 0.32}),
    (law59_spring_rate, 59, {"MAT_E": 4000.0, "MAT_G": 2000.0}),
    (law63_composite_shell, 63, {"MAT_E": 190000.0, "MAT_NU": 0.28}),
    (law64_rate_comp, 64, {"MAT_E": 205000.0, "MAT_NU": 0.3}),
    (law65_thermo_visc, 65, {"MAT_E": 600.0, "MAT_NU": 0.44}),
    (law68_tab_plas, 68, {"MAT_EA": 900.0, "MAT_EB": 700.0, "MAT_EC": 1100.0}),
])
def test_all_batch2_consistency(mod, law_id, p_dict):
    mat = Material(id=law_id, law=law_id, rho0=1.2, title=f"Test_{law_id}", params=p_dict)
    p = mod.resolve(mat)

    # 1. nip > 1 extra_shapes
    shapes = mod.extra_shapes(mat, nip=3)
    assert isinstance(shapes, dict)
    for k, v in shapes.items():
        assert isinstance(v, tuple)
        if len(v) > 0:
            assert v[0] == 3

    # 2. return_sound_speed=False
    sig_1d = np.zeros(6)
    deps_1d = np.array([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    s_sol, ep_sol, c_sol = mod.solid_update(p, sig_1d, deps_1d, return_sound_speed=False)
    assert c_sol is None

    sig_sh_1d = np.zeros(3)
    deps_sh_1d = np.array([1e-3, 0.0, 0.0])
    s_sh, ep_sh, c_sh = mod.shell_update(p, sig_sh_1d, deps_sh_1d, return_sound_speed=False)
    assert c_sh is None

    # 3. Consistent tangents
    c_sol_tan = mod.consistent_solid_tangent(p)
    assert c_sol_tan.shape == (6, 6)
    c_sh_tan = mod.consistent_shell_tangent(p)
    assert c_sh_tan.shape == (3, 3)

    # Broadcasted tangents for 2D inputs
    sig_2d = np.zeros((4, 6))
    c_sol_tan_2d = mod.consistent_solid_tangent(p, sig=sig_2d)
    assert c_sol_tan_2d.shape == (4, 6, 6)
    sig_sh_2d = np.zeros((4, 3))
    c_sh_tan_2d = mod.consistent_shell_tangent(p, sig=sig_sh_2d)
    assert c_sh_tan_2d.shape == (4, 3, 3)

