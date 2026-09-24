"""Unit tests for /INICRACK initial crack state and failure model integration.

Upstream Fortran references:
- ``starter/source/initial_conditions/inicrack/hm_read_inicrack.F``
- ``starter/source/materials/fail/windshield_alter/brokmann_crack_init.F90``
- ``engine/source/materials/fail/alter/fail_brokmann.F``
- ``common_source/fail/newman_raju.F90``

Verifies:
1. Initial crack geometry setup (depth a, half-length c, aspect ratio a/c, center, title).
2. Crack orientation vector and projection onto shell element surface.
3. Crack angle mapping theta_mohr = 2 * CR_ANG.
4. Newman-Raju stress intensity factors K_1A (deepest point) and K_1C (surface point).
5. State variable initialization for /FAIL/ALTER Brokmann model.
6. Coupling with Brokmann subcritical crack growth step and critical rupture.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.failure import inicrack
from pyradioss.failure.inicrack import (
    IniCrack,
    IniCrackData,
    IniCrackSegment,
    advance_brokmann_crack,
    build_inicrack,
    compute_crack_sif,
    compute_initial_sif,
    init_brokmann_crack_state,
    map_crack_angle,
    project_crack_orientation,
)
from pyradioss.failure.brokmann import brokmann_step


def test_inicrack_data_creation_and_properties():
    """Verify initial crack geometric attributes and property calculations."""
    crack = IniCrackData(
        id=12,
        part_id=3,
        element_id=105,
        title="SURFACE_MICROCRACK_1",
        center=(1.5, 2.5, 0.0),
        a0=120.0,   # depth in micrometers
        c0=240.0,   # half-length in micrometers
        normal=(0.0, 1.0, 0.0),
        sigma_ini=15.0,
        angle=math.pi / 4.0,
    )

    assert crack.id == 12
    assert crack.part_id == 3
    assert crack.element_id == 105
    assert crack.title == "SURFACE_MICROCRACK_1"
    assert crack.x0 == 1.5
    assert crack.y0 == 2.5
    assert crack.z0 == 0.0
    assert crack.depth == 120.0
    assert crack.half_length == 240.0
    assert crack.length == 480.0
    assert crack.aspect_ratio == pytest.approx(0.5, rel=1e-6)
    assert crack.sigma_ini == 15.0
    assert crack.angle == pytest.approx(math.pi / 4.0, rel=1e-6)

    # Test setter methods
    crack.depth = 150.0
    assert crack.a0 == 150.0
    crack.half_length = 300.0
    assert crack.c0 == 300.0
    assert crack.aspect_ratio == pytest.approx(0.5, rel=1e-6)


def test_build_inicrack_from_dict_and_kwargs():
    """Verify builder helper handles dictionaries, aliases, and keyword arguments."""
    data = {
        "ID": 25,
        "pid": 5,
        "eid": 204,
        "title": "FLAW_CORNER",
        "x0": 10.0,
        "y0": 20.0,
        "z0": -5.0,
        "CR_DEPTH": 80.0,
        "CR_LEN": 160.0,
        "norm": [0.0, 0.0, 1.0],
        "SIG_INI": 12.5,
        "theta": 0.35,
    }
    c = build_inicrack(data)
    assert isinstance(c, IniCrackData)
    assert c.id == 25
    assert c.part_id == 5
    assert c.element_id == 204
    assert c.title == "FLAW_CORNER"
    assert c.center == (10.0, 20.0, -5.0)
    assert c.a0 == 80.0
    assert c.c0 == 160.0
    assert c.normal == (0.0, 0.0, 1.0)
    assert c.sigma_ini == 12.5
    assert c.angle == pytest.approx(0.35, rel=1e-6)

    # Keyword argument override
    c2 = build_inicrack(id=99, a0=50.0, c0=100.0)
    assert c2.id == 99
    assert c2.a0 == 50.0
    assert c2.c0 == 100.0


def test_crack_orientation_projection_onto_shell():
    """Verify crack plane normal projection onto shell surface and in-plane angle extraction."""
    # Case 1: Shell in xy-plane (normal e3 = (0, 0, 1), e1 = (1, 0, 0))
    # Crack normal along y-axis (0, 1, 0) -> in-plane angle should be 90 deg (pi/2)
    proj, theta = project_crack_orientation(
        crack_normal=(0.0, 1.0, 0.0),
        shell_normal=(0.0, 0.0, 1.0),
        shell_x_axis=(1.0, 0.0, 0.0),
    )
    assert proj[0] == pytest.approx(0.0, abs=1e-6)
    assert proj[1] == pytest.approx(1.0, abs=1e-6)
    assert proj[2] == pytest.approx(0.0, abs=1e-6)
    assert theta == pytest.approx(math.pi / 2.0, rel=1e-5)

    # Case 2: Crack normal at 45 deg between x and y
    proj45, theta45 = project_crack_orientation(
        crack_normal=(1.0, 1.0, 0.0),
        shell_normal=(0.0, 0.0, 1.0),
        shell_x_axis=(1.0, 0.0, 0.0),
    )
    assert theta45 == pytest.approx(math.pi / 4.0, rel=1e-5)
    np.testing.assert_allclose(proj45, [math.sqrt(0.5), math.sqrt(0.5), 0.0], rtol=1e-5)

    # Case 3: Crack normal parallel to shell normal (degenerate normal projection)
    proj_deg, theta_deg = project_crack_orientation(
        crack_normal=(0.0, 0.0, 1.0),
        shell_normal=(0.0, 0.0, 1.0),
        shell_x_axis=(1.0, 0.0, 0.0),
    )
    assert theta_deg == 0.0

    # Case 4: Tilted shell in 3D (normal along (0, 1, 1)/sqrt(2))
    shell_norm_tilted = (0.0, math.sqrt(0.5), math.sqrt(0.5))
    crack_norm_3d = (1.0, 0.0, 0.0)  # purely along x
    proj_tilted, theta_tilted = project_crack_orientation(
        crack_normal=crack_norm_3d,
        shell_normal=shell_norm_tilted,
        shell_x_axis=(1.0, 0.0, 0.0),
    )
    # Projected vector must be perpendicular to shell normal: dot(proj, shell_normal) == 0
    assert abs(np.dot(proj_tilted, shell_norm_tilted)) < 1e-12
    assert theta_tilted == pytest.approx(0.0, abs=1e-6)


def test_angle_mapping_to_brokmann_state():
    """Verify in-plane angle mapping: theta_mohr = 2 * CR_ANG (fail_brokmann.F line 114)."""
    angles = [0.0, math.pi / 6.0, math.pi / 4.0, math.pi / 3.0, math.pi / 2.0]
    for th in angles:
        cr_ang, theta_mohr = map_crack_angle(th)
        assert cr_ang == pytest.approx(th, rel=1e-6)
        assert theta_mohr == pytest.approx(2.0 * th, rel=1e-6)


def test_compute_initial_sif():
    """Verify initial stress intensity factors K_1A and K_1C with Newman-Raju correction."""
    crack = IniCrackData(
        a0=200.0,   # depth 200 micrometers (0.2 mm)
        c0=400.0,   # half-length 400 micrometers (0.4 mm)
        sigma_ini=10.0e6,  # 10 MPa residual compression
    )
    thickness = 2000.0  # 2 mm thickness in micrometers
    width = 10000.0     # 10 mm width in micrometers

    # Applied tensile stress of 60 MPa -> effective stress 50 MPa
    sigma_applied = 60.0e6
    sif_res = compute_initial_sif(crack, thickness, width, sigma_applied, unit_fac_l=1.0)

    assert sif_res["sigma_eff"] == pytest.approx(50.0e6, rel=1e-6)
    assert sif_res["aspect_ratio"] == pytest.approx(0.5, rel=1e-6)
    assert sif_res["depth_ratio"] == pytest.approx(0.1, rel=1e-6)

    # Geometry factors: Y_A (deepest point) and Y_C (surface point)
    assert sif_res["Y_A"] > 0.9
    assert sif_res["Y_C"] > 0.5
    # Mode I SIF: positive and on the order of MPa*sqrt(m)
    assert sif_res["K1_A"] > 1.0e5
    assert sif_res["K1_C"] > 1.0e5

    # When applied stress is below residual stress, SIF must be 0
    sif_comp = compute_initial_sif(crack, thickness, width, 5.0e6)
    assert sif_comp["sigma_eff"] == 0.0
    assert sif_comp["K1_A"] == 0.0
    assert sif_comp["K1_C"] == 0.0


def test_init_brokmann_crack_state():
    """Verify internal state variables initialization matching brokmann_crack_init.F90."""
    crack = IniCrackData(
        id=7,
        a0=150.0,
        c0=300.0,
        angle=0.42,
        sigma_ini=8.0,
    )
    thk = 0.003   # 3 mm
    aldt = 0.015  # 15 mm

    uvar = init_brokmann_crack_state(crack, thickness=thk, width=aldt, fac_l=1.0)

    assert len(uvar) >= 21
    # 0-based index [14] = FAIL_B (0 = alive)
    assert uvar[14] == 0.0
    # [15] = CR_LEN (c in micrometers)
    assert uvar[15] == 300.0
    # [16] = CR_DEPTH (a in micrometers)
    assert uvar[16] == 150.0
    # [17] = CR_ANG (angle in radians)
    assert uvar[17] == pytest.approx(0.42, rel=1e-6)
    # [18] = THK0 (thickness in micrometers = 0.003 * 1e6 = 3000)
    assert uvar[18] == pytest.approx(3000.0, rel=1e-6)
    # [19] = ALDT0 (width in micrometers = 0.015 * 1e6 = 15000)
    assert uvar[19] == pytest.approx(15000.0, rel=1e-6)
    # [20] = SIG_COS (sigma_ini)
    assert uvar[20] == pytest.approx(8.0, rel=1e-6)


def test_coupling_with_brokmann_crack_growth_step():
    """Verify subcritical crack growth (Paris law) and critical failure under stress."""
    crack = IniCrackData(
        a0=100.0,
        c0=200.0,
        angle=0.0,   # aligned with x-axis
        sigma_ini=0.0,
    )
    thickness = 2.0e-3  # 2 mm
    width = 10.0e-3     # 10 mm
    dt = 1.0e-3         # 1 ms

    k_ic = 1.0e6        # 1.0 MPa*sqrt(m)
    k_th = 0.1e6        # 0.1 MPa*sqrt(m)
    v0 = 1.0e-3         # 1 mm/s reference velocity
    exp_n = 4.0         # Paris exponent

    # Case 1: Moderate tensile stress sig_xx = 30 MPa (produces K_th <= K1 < K_IC)
    stress_sub = np.array([30.0e6, 0.0, 0.0])
    is_failed, new_a, new_c, diag = advance_brokmann_crack(
        crack=crack,
        stress=stress_sub,
        thickness=thickness,
        width=width,
        dt=dt,
        k_ic=k_ic,
        k_th=k_th,
        v0=v0,
        exp_n=exp_n,
    )
    assert is_failed is False
    # Subcritical growth occurred: depth and length increased
    assert new_a > 100.0
    assert new_c > 200.0
    assert diag["K1_A"] >= k_th
    assert diag["K1_A"] < k_ic

    # Case 2: Extreme stress exceeding fracture toughness (sig_xx = 200 MPa)
    crack_crit = IniCrackData(a0=100.0, c0=200.0, angle=0.0)
    stress_crit = np.array([200.0e6, 0.0, 0.0])
    is_failed_crit, _, _, diag_crit = advance_brokmann_crack(
        crack=crack_crit,
        stress=stress_crit,
        thickness=thickness,
        width=width,
        dt=dt,
        k_ic=k_ic,
        k_th=k_th,
        v0=v0,
        exp_n=exp_n,
    )
    assert is_failed_crit is True
    assert diag_crit["K1_A"] >= k_ic or diag_crit["K1_C"] >= k_ic


def test_brokmann_step_vectorized_integration():
    """Verify integration with pyradioss.failure.brokmann.brokmann_step using inicrack state."""
    nel = 3
    crack = IniCrackData(a0=80.0, c0=160.0, angle=0.0)
    thickness = 2.0e-3
    width = 10.0e-3

    uvar = np.zeros((nel, 21), dtype=float)
    for i in range(nel):
        uvar[i] = init_brokmann_crack_state(crack, thickness, width)

    off = np.ones(nel, dtype=float)
    tdel = np.zeros(nel, dtype=float)

    # Parameters matching uparam layout in brokmann.py:
    # uparam[0] = exp_n, [5] = k_ic, [6] = k_th, [7] = v0, [9] = alpha, [29] = sig_ini
    uparam = np.zeros(35, dtype=float)
    uparam[0] = 4.0        # EXP_N
    uparam[5] = 1.0e6      # K_IC
    uparam[6] = 0.1e6      # K_TH
    uparam[7] = 1.0e-3     # V0
    uparam[9] = 0.9        # ALPHA
    uparam[29] = 0.0       # SIG_INI
    uparam[32] = 1.0       # FAC_M
    uparam[33] = 1.0       # FAC_L
    uparam[34] = 1.0       # FAC_T

    # Element 0: moderate stress (growth, no failure)
    # Element 1: zero stress (no growth, no failure)
    # Element 2: massive stress (critical failure)
    signxx = np.array([25.0e6, 0.0, 300.0e6])
    signyy = np.zeros(nel)
    signxy = np.zeros(nel)

    failed_mask = brokmann_step(
        uvar=uvar,
        off=off,
        signxx=signxx,
        signyy=signyy,
        signxy=signxy,
        uparam=uparam,
        timestep=1.0e-3,
        time=1.0e-3,
        tdel=tdel,
    )

    # Element 0 should still be alive
    assert failed_mask[0] is False or failed_mask[0] == 0
    assert uvar[0, 14] == 0.0

    # Element 1 should have no growth
    assert uvar[1, 14] == 0.0

    # Element 2 should have failed
    assert bool(failed_mask[2]) is True
    assert uvar[2, 14] == 1.0
    off[failed_mask] = 0.0
    assert off[2] == 0.0
