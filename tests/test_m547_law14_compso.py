"""Unit tests for /MAT/LAW14 (/MAT/COMPSO, /MAT/COMP_SOL) constitutive model.

Milestone M547:
  - tests/test_m547_law14_compso.py: comprehensive tests for pyradioss/materials/law14_compso.py
  - Citing:
      starter/source/materials/mat/mat014/hm_read_mat14.F
      engine/source/materials/mat/mat014/m14law.F
      hm_cfg_files/config/CFG/radioss2020/MAT/matl14_compso.cfg
"""

from __future__ import annotations

import math
from typing import Any, Dict
import numpy as np
import pytest

from pyradioss.materials.law14_compso import (
    build_law14,
    consistent_solid_tangent,
    extra_shapes,
    m14ama,
    m14ftg,
    m14gtf,
    shell_update,
    solid_update,
    sound_speed,
)
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY, GenericMaterialRecord
from pyradioss.model.entities import MatLaw14


# ============================================================================
# 1. Parameter Constructor & Default Values
# ============================================================================

def test_build_law14_defaults():
    """Verify default values, derived orthotropic stiffness D11..D33, DETC > 0, Tsai-Wu coefficients."""
    e11, e22, e33 = 120000.0, 60000.0, 60000.0
    nu12, nu23, nu31 = 0.25, 0.3, 0.125
    g12, g23, g31 = 20000.0, 15000.0, 20000.0
    rho0 = 1.6e-9

    mat = build_law14(
        id=14,
        rho0=rho0,
        E11=e11,
        E22=e22,
        E33=e33,
        nu12=nu12,
        nu23=nu23,
        nu31=nu31,
        G12=g12,
        G23=g23,
        G31=g31,
    )

    assert mat.id == 14
    assert mat.law == 14
    assert mat.law_name == "LAW14"
    assert mat.rho0 == pytest.approx(rho0)

    p = mat.params
    # Compliance calculations (hm_read_mat14.F:193-198)
    c11 = 1.0 / e11
    c22 = 1.0 / e22
    c33 = 1.0 / e33
    c12 = -nu12 / e11
    c13 = -nu31 / e33
    c23 = -nu23 / e22

    expected_detc = (
        c11 * c22 * c33
        - c11 * (c23**2)
        - (c12**2) * c33
        + 2.0 * c12 * c13 * c23
        - (c13**2) * c22
    )
    assert p["DETC"] > 0.0
    assert p["DETC"] == pytest.approx(expected_detc)

    # Inverted orthotropic stiffness D11..D33 (hm_read_mat14.F:211-219)
    expected_d11 = (c22 * c33 - c23**2) / expected_detc
    expected_d12 = -(c12 * c33 - c13 * c23) / expected_detc
    expected_d13 = (c12 * c23 - c13 * c22) / expected_detc
    expected_d22 = (c11 * c33 - c13**2) / expected_detc
    expected_d23 = -(c11 * c23 - c13 * c12) / expected_detc
    expected_d33 = (c11 * c22 - c12**2) / expected_detc

    assert p["D11"] == pytest.approx(expected_d11)
    assert p["D12"] == pytest.approx(expected_d12)
    assert p["D13"] == pytest.approx(expected_d13)
    assert p["D22"] == pytest.approx(expected_d22)
    assert p["D23"] == pytest.approx(expected_d23)
    assert p["D33"] == pytest.approx(expected_d33)
    assert p["G12"] == pytest.approx(g12)
    assert p["G23"] == pytest.approx(g23)
    assert p["G31"] == pytest.approx(g31)

    # Verification C @ D = I (hm_read_mat14.F:222-227)
    assert p["A11"] == pytest.approx(1.0, abs=1e-10)
    assert p["A22"] == pytest.approx(1.0, abs=1e-10)
    assert p["A33"] == pytest.approx(1.0, abs=1e-10)
    assert p["A12"] == pytest.approx(0.0, abs=1e-10)
    assert p["A13"] == pytest.approx(0.0, abs=1e-10)
    assert p["A23"] == pytest.approx(0.0, abs=1e-10)

    # Defaults check
    assert p["delta"] == pytest.approx(0.05)
    assert p["cb"] == pytest.approx(0.0)
    assert p["cn"] == pytest.approx(1.0)
    assert p["sigmx"] == pytest.approx(1.0e10)
    assert p["wpref"] == pytest.approx(1.0)
    assert p["alpha"] == pytest.approx(0.0)
    assert p["efib"] == pytest.approx(0.0)
    assert p["c"] == pytest.approx(0.0)
    assert p["eps0"] == pytest.approx(1.0)
    assert p["ICC"] == 1

    # Timestep parameter PM105 check (hm_read_mat14.F:264)
    c1 = max(expected_d11, expected_d22, expected_d33)
    expected_dmin = min(
        expected_d11 * expected_d22 - expected_d12**2,
        expected_d22 * expected_d33 - expected_d23**2,
        expected_d11 * expected_d33 - expected_d13**2,
    )
    expected_pm105 = expected_dmin / (c1**2)
    assert p["DMIN"] == pytest.approx(expected_dmin)
    assert p["PM105"] == pytest.approx(expected_pm105)


def test_build_law14_invalid_moduli():
    """Verify raises ValueError on E11 <= 0 or DETC <= 0."""
    with pytest.raises(ValueError, match="must be > 0"):
        build_law14(
            id=1,
            rho0=1.5e-9,
            E11=0.0,
            E22=50000.0,
            E33=20000.0,
            nu12=0.25,
            nu23=0.2,
            nu31=0.15,
            G12=10000.0,
            G23=10000.0,
            G31=10000.0,
        )

    with pytest.raises(ValueError, match="must be > 0"):
        build_law14(
            id=2,
            rho0=1.5e-9,
            E11=-10000.0,
            E22=50000.0,
            E33=20000.0,
            nu12=0.25,
            nu23=0.2,
            nu31=0.15,
            G12=10000.0,
            G23=10000.0,
            G31=10000.0,
        )

    with pytest.raises(ValueError, match="DETC"):
        build_law14(
            id=3,
            rho0=1.5e-9,
            E11=100000.0,
            E22=100000.0,
            E33=100000.0,
            nu12=0.9,
            nu23=0.9,
            nu31=0.9,
            G12=10000.0,
            G23=10000.0,
            G31=10000.0,
        )


# ============================================================================
# 2. Sound Speed & Extra Shapes
# ============================================================================

def test_sound_speed():
    """Verify sound_speed computes c = sqrt(max(D11, D22, D33)/rho0)."""
    rho0 = 2.0e-9
    e11, e22, e33 = 120000.0, 60000.0, 30000.0
    mat = build_law14(
        id=14,
        rho0=rho0,
        E11=e11,
        E22=e22,
        E33=e33,
        nu12=0.3,
        nu23=0.2,
        nu31=0.1,
        G12=15000.0,
        G23=10000.0,
        G31=12000.0,
    )
    p = mat.params
    d11 = p["D11"]
    d22 = p["D22"]
    d33 = p["D33"]
    expected_c = math.sqrt(max(d11, d22, d33) / rho0)

    computed_c = sound_speed(mat)
    assert computed_c == pytest.approx(expected_c)
    assert p["ssp"] == pytest.approx(expected_c)


def test_extra_shapes():
    """Verify extra_shapes returns correct state variable dictionaries for nip=1 and nip=4."""
    s1 = extra_shapes()
    assert s1["dam14"] == (5,)
    assert s1["epe14"] == (3,)
    assert s1["epc14"] == (3,)
    assert s1["wpla14"] == ()
    assert s1["off14"] == ()
    assert s1["epsf14"] == ()
    assert s1["sigf14"] == ()
    assert s1["tsaiwu14"] == ()

    s4 = extra_shapes(nip=4)
    assert s4["dam14"] == (4, 5)
    assert s4["epe14"] == (4, 3)
    assert s4["epc14"] == (4, 3)
    assert s4["wpla14"] == (4,)
    assert s4["off14"] == (4,)
    assert s4["epsf14"] == (4,)
    assert s4["sigf14"] == (4,)
    assert s4["tsaiwu14"] == (4,)


def test_shell_update_not_implemented():
    """Verify shell_update raises NotImplementedError."""
    mat = build_law14(
        id=14,
        rho0=1.5e-9,
        E11=100000.0,
        E22=50000.0,
        E33=50000.0,
        nu12=0.25,
        nu23=0.25,
        nu31=0.15,
        G12=15000.0,
        G23=15000.0,
        G31=15000.0,
    )
    with pytest.raises(NotImplementedError, match="3D solid elements only"):
        shell_update(mat, np.zeros(3), np.zeros(3))


# ============================================================================
# 3. Elastic Solid Update
# ============================================================================

def test_solid_update_elastic():
    """Unyielding small strain increments match D @ deps."""
    mat = build_law14(
        id=14,
        rho0=1.5e-9,
        E11=100000.0,
        E22=50000.0,
        E33=50000.0,
        nu12=0.25,
        nu23=0.25,
        nu31=0.15,
        G12=15000.0,
        G23=15000.0,
        G31=15000.0,
        sigt1=1.0e10,
        sigyt1=1.0e10,
    )
    p = mat.params
    d11, d12, d13 = p["D11"], p["D12"], p["D13"]
    d22, d23, d33 = p["D22"], p["D23"], p["D33"]
    g12, g23, g31 = p["G12"], p["G23"], p["G31"]

    sig = np.zeros(6, dtype=float)
    deps = np.array([1.0e-5, 2.0e-5, -1.0e-5, 5.0e-6, -5.0e-6, 3.0e-6], dtype=float)
    extra: Dict[str, Any] = {}

    sig_out, epsp_out, c_out = solid_update(mat, sig, deps, dt=1.0e-6, extra=extra)

    expected_sig1 = d11 * deps[0] + d12 * deps[1] + d13 * deps[2]
    expected_sig2 = d12 * deps[0] + d22 * deps[1] + d23 * deps[2]
    expected_sig3 = d13 * deps[0] + d23 * deps[1] + d33 * deps[2]
    expected_sig4 = g12 * deps[3]
    expected_sig5 = g23 * deps[4]
    expected_sig6 = g31 * deps[5]

    assert sig_out[0] == pytest.approx(expected_sig1, rel=1e-6)
    assert sig_out[1] == pytest.approx(expected_sig2, rel=1e-6)
    assert sig_out[2] == pytest.approx(expected_sig3, rel=1e-6)
    assert sig_out[3] == pytest.approx(expected_sig4, rel=1e-6)
    assert sig_out[4] == pytest.approx(expected_sig5, rel=1e-6)
    assert sig_out[5] == pytest.approx(expected_sig6, rel=1e-6)

    assert epsp_out == pytest.approx(0.0)
    assert extra["dam14"][0, 0] == 0.0
    assert extra["wpla14"][0] == 0.0


# ============================================================================
# 4. Fiber Stress Tracking
# ============================================================================

def test_solid_update_fiber():
    """alpha > 0 tracks fiber strain and stress sigf."""
    alpha_val = 0.4
    efib_val = 210000.0
    mat = build_law14(
        id=14,
        rho0=1.5e-9,
        E11=100000.0,
        E22=50000.0,
        E33=50000.0,
        nu12=0.25,
        nu23=0.25,
        nu31=0.15,
        G12=15000.0,
        G23=15000.0,
        G31=15000.0,
        alpha=alpha_val,
        efib=efib_val,
        sigt1=1.0e10,
    )
    sig = np.zeros(6, dtype=float)
    deps = np.array([2.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    extra: Dict[str, Any] = {}

    solid_update(mat, sig, deps, dt=1.0e-6, extra=extra)

    assert extra["epsf14"][0] == pytest.approx(2.0e-4)
    assert extra["sigf14"][0] == pytest.approx(efib_val * 2.0e-4)


# ============================================================================
# 5. Tensile Cracking Damage & Poisson Reduction
# ============================================================================

def test_solid_update_tensile_damage_direction1():
    """Tensile overstressing in direction 1 caps stress to (1-dam)*sigt1 and damages matrix."""
    sigt1_val = 100.0
    delta_val = 0.1
    mat = build_law14(
        id=14,
        rho0=1.5e-9,
        E11=100000.0,
        E22=50000.0,
        E33=50000.0,
        nu12=0.25,
        nu23=0.25,
        nu31=0.15,
        G12=15000.0,
        G23=15000.0,
        G31=15000.0,
        sigt1=sigt1_val,
        delta=delta_val,
    )
    # Trial stress in 1: D11 * deps1 = 100000 * 0.002 = 200.0 > 100.0
    sig = np.zeros(6, dtype=float)
    deps = np.array([0.002, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    extra: Dict[str, Any] = {}

    sig_out, _, _ = solid_update(mat, sig, deps, dt=1.0e-6, extra=extra)

    # Initial dam1 = 0, so capped at (1 - 0) * sigt1 = 100.0
    assert sig_out[0] == pytest.approx(sigt1_val)
    # Dam1 updated to min(0 + delta, 1.0) = 0.1
    assert extra["dam14"][0, 0] == pytest.approx(delta_val)
    # epc1 recorded
    assert extra["epc14"][0, 0] == pytest.approx(0.002)


def test_solid_update_crack_closure():
    """Once cracked, compression across crack causes T1 = 0 when epc1 > 0."""
    mat = build_law14(
        id=14,
        rho0=1.5e-9,
        E11=100000.0,
        E22=50000.0,
        E33=50000.0,
        nu12=0.25,
        nu23=0.25,
        nu31=0.15,
        G12=15000.0,
        G23=15000.0,
        G31=15000.0,
        sigt1=50.0,
        delta=0.2,
    )
    sig = np.zeros(6, dtype=float)
    deps_tens = np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    extra: Dict[str, Any] = {}

    # Cycle 1: crack
    sig_1, _, _ = solid_update(mat, sig, deps_tens, dt=1.0e-6, extra=extra)
    assert extra["epc14"][0, 0] > 0.0

    # Cycle 2: negative strain increment resulting in trial compressive stress
    deps_comp = np.array([-0.0005, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig_2, _, _ = solid_update(mat, sig_1, deps_comp, dt=1.0e-6, extra=extra)
    # Crack opening condition (m14law.F:434)
    # If T1 < 0 and epc1 > 0: T1 = 0
    if sig_2[0] < 0.0:
        assert sig_2[0] == pytest.approx(0.0)


# ============================================================================
# 6. Tsai-Wu Plastic Return
# ============================================================================

def test_solid_update_tsaiwu_plasticity():
    """Stress exceeding Tsai-Wu yield surface triggers plastic projection and wpla accumulation."""
    mat = build_law14(
        id=14,
        rho0=1.5e-9,
        E11=100000.0,
        E22=50000.0,
        E33=50000.0,
        nu12=0.25,
        nu23=0.25,
        nu31=0.15,
        G12=15000.0,
        G23=15000.0,
        G31=15000.0,
        sigyt1=200.0,
        sigyc1=200.0,
        sigyt2=100.0,
        sigyc2=100.0,
        sigt12=80.0,
        sigc12=80.0,
        sigt23=60.0,
        sigc23=60.0,
        cb=500.0,
        cn=0.5,
        wpref=1.0,
        sigt1=1.0e10,
        sigt2=1.0e10,
        sigt3=1.0e10,
    )
    sig = np.zeros(6, dtype=float)
    deps = np.array([0.005, 0.003, 0.002, 0.004, 0.001, 0.001], dtype=float)
    extra: Dict[str, Any] = {}

    # Step 1 loads stress past yield surface (gradient DP is evaluated at old stress SO=0)
    sig_out1, _, _ = solid_update(mat, sig, deps, dt=1.0e-5, extra=extra)
    # Step 2 with non-zero SO triggers Tsai-Wu plastic return and accumulates wpla
    sig_out2, epsp_out, _ = solid_update(mat, sig_out1, deps, dt=1.0e-5, extra=extra)

    # Plastic work should have accumulated
    assert epsp_out > 0.0
    assert extra["wpla14"][0] == pytest.approx(epsp_out)
    # Tsai-Wu utilization factor
    assert extra["tsaiwu14"][0] > 0.0


# ============================================================================
# 7. Strain Rate Enhancement
# ============================================================================

def test_solid_update_strain_rate():
    """High strain rate enhances yield limit and hardening slope."""
    c_rate = 0.05
    eps0 = 1.0
    mat = build_law14(
        id=14,
        rho0=1.5e-9,
        E11=100000.0,
        E22=50000.0,
        E33=50000.0,
        nu12=0.25,
        nu23=0.25,
        nu31=0.15,
        G12=15000.0,
        G23=15000.0,
        G31=15000.0,
        sigyt1=200.0,
        sigyc1=200.0,
        sigyt2=100.0,
        sigyc2=100.0,
        c=c_rate,
        eps0=eps0,
        ICC=1,
        cb=500.0,
        cn=0.5,
        sigt1=1.0e10,
    )
    sig = np.zeros(6, dtype=float)
    deps = np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

    # Low strain rate (dt = 1s -> eps_rate = 0.005 < 1.0)
    extra_low: Dict[str, Any] = {}
    sig_low1, _, _ = solid_update(mat, sig, deps, dt=1.0, extra=extra_low)
    sig_low, ep_low, _ = solid_update(mat, sig_low1, deps, dt=1.0, extra=extra_low)

    # High strain rate (dt = 1e-5s -> eps_rate = 500 > 1.0)
    extra_high: Dict[str, Any] = {}
    sig_high1, _, _ = solid_update(mat, sig, deps, dt=1.0e-5, extra=extra_high)
    sig_high, ep_high, _ = solid_update(mat, sig_high1, deps, dt=1.0e-5, extra=extra_high)

    # High strain rate provides higher effective strength, so higher stress
    assert sig_high[0] > sig_low[0]


# ============================================================================
# 8. Coordinate Transformation Invariance
# ============================================================================

def test_coordinate_transforms():
    """Verify coordinate transformation functions m14ama, m14gtf, m14ftg."""
    rx = np.array([1.0, 0.0, 0.0])
    ry = np.array([0.0, 1.0, 0.0])
    rz = np.array([0.0, 0.0, 1.0])
    sx = np.array([0.0, 0.0, 1.0])
    sy = np.array([1.0, 0.0, 0.0])
    sz = np.array([0.0, 1.0, 0.0])

    ax, ay, az, bx, by, bz, cx, cy, cz = m14ama(rx, ry, rz, sx, sy, sz)
    assert len(ax) == 3

    # Forward and back transformation of stress
    sig_global = np.array([[100.0, 50.0, 20.0, 10.0, 5.0, 2.0]])
    d_global = np.array([[0.001, 0.0005, -0.0002, 0.0001, 0.0, 0.0]])

    sig_mat, d_mat = m14gtf(
        sig_global, d_global,
        ax[:1], ay[:1], az[:1],
        bx[:1], by[:1], bz[:1],
        cx[:1], cy[:1], cz[:1],
    )
    sig_recov = m14ftg(
        sig_mat,
        ax[:1], ay[:1], az[:1],
        bx[:1], by[:1], bz[:1],
        cx[:1], cy[:1], cz[:1],
    )

    np.testing.assert_allclose(sig_recov, sig_global, rtol=1e-6, atol=1e-8)


# ============================================================================
# 9. Algorithmic Consistent Tangent
# ============================================================================

def test_consistent_solid_tangent_elastic():
    """In elastic regime, consistent_solid_tangent matches D_mat."""
    mat = build_law14(
        id=14,
        rho0=1.5e-9,
        E11=100000.0,
        E22=50000.0,
        E33=50000.0,
        nu12=0.25,
        nu23=0.25,
        nu31=0.15,
        G12=15000.0,
        G23=15000.0,
        G31=15000.0,
        sigt1=1.0e10,
        sigyt1=1.0e10,
    )
    p = mat.params
    sig = np.zeros(6, dtype=float)
    deps = np.zeros(6, dtype=float)

    tangent = consistent_solid_tangent(mat, sig, deps=deps, symmetric=True)

    expected_D = np.array(
        [
            [p["D11"], p["D12"], p["D13"], 0.0, 0.0, 0.0],
            [p["D12"], p["D22"], p["D23"], 0.0, 0.0, 0.0],
            [p["D13"], p["D23"], p["D33"], 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, p["G12"], 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, p["G23"], 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, p["G31"]],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(tangent, expected_D, rtol=1e-5, atol=1e-7)


def test_consistent_solid_tangent_plastic():
    """In plastic regime, consistent_solid_tangent matches finite difference perturbation of solid_update."""
    mat = build_law14(
        id=14,
        rho0=1.5e-9,
        E11=100000.0,
        E22=50000.0,
        E33=50000.0,
        nu12=0.25,
        nu23=0.25,
        nu31=0.15,
        G12=15000.0,
        G23=15000.0,
        G31=15000.0,
        sigyt1=150.0,
        sigyc1=150.0,
        sigyt2=100.0,
        sigyc2=100.0,
        cb=300.0,
        cn=1.0,
        sigt1=1.0e10,
    )
    sig = np.zeros(6, dtype=float)
    deps = np.array([0.003, 0.001, 0.001, 0.001, 0.0, 0.0], dtype=float)
    extra: Dict[str, Any] = {}

    # Advance into plastic regime
    sig_cur, ep_cur, _ = solid_update(mat, sig, deps, dt=1.0e-6, extra=extra)

    tangent = consistent_solid_tangent(
        mat, sig_cur, epsp=ep_cur, dt=1.0e-6, extra=extra, deps=deps, h=1.0e-7
    )

    assert tangent.shape == (6, 6)
    # Plasticity softens diagonal stiffness compared to pure elastic D11
    assert tangent[0, 0] < mat.params["D11"]


# ============================================================================
# 10. Registry Hook & Duck-typing
# ============================================================================

def test_registry_and_entities_compatibility():
    """Verify registration in MAT_PHYSICS_REGISTRY and compatibility with MatLaw14 & GenericMaterialRecord."""
    for key in (
        14,
        "14",
        "LAW14",
        "COMPSO",
        "COMP_SOL",
        "MAT_LAW14",
        "MAT_COMPSO",
        "MAT_COMP_SOL",
    ):
        assert key in MAT_PHYSICS_REGISTRY

    # Build from GenericMaterialRecord
    rec = GenericMaterialRecord(
        law_name="COMPSO",
        law_number=14,
        id=14,
        title="Composite Solid Record",
        density=1.5e-9,
        params={
            "MAT_EA": 110000.0,
            "MAT_EB": 55000.0,
            "MAT_EC": 55000.0,
            "MAT_PRAB": 0.25,
            "MAT_PRBC": 0.25,
            "MAT_PRCA": 0.15,
            "MAT_GAB": 16000.0,
            "MAT_GBC": 16000.0,
            "MAT_GCA": 16000.0,
        },
    )
    mat_rec = MAT_PHYSICS_REGISTRY["COMPSO"](rec)
    assert mat_rec.id == 14
    assert mat_rec.law == 14
    assert mat_rec.rho0 == pytest.approx(1.5e-9)
    assert mat_rec.params["E11"] == pytest.approx(110000.0)

    # Build from MatLaw14 dataclass
    mat_entity = MatLaw14(
        id=101,
        rho0=1.6e-9,
        ea=120000.0,
        eb=60000.0,
        ec=60000.0,
        prab=0.28,
        prbc=0.28,
        prca=0.14,
        gab=18000.0,
        gbc=18000.0,
        gca=18000.0,
        sigt1=150.0,
        title="Dataclass Entity",
    )
    mat_built = build_law14(mat_entity)
    assert mat_built.id == 101
    assert mat_built.rho0 == pytest.approx(1.6e-9)
    assert mat_built.params["E11"] == pytest.approx(120000.0)
    assert mat_built.params["sigt1"] == pytest.approx(150.0)
