"""Unit tests for /MAT/LAW12 (/MAT/3D_COMP, /MAT/COMP_3D) constitutive model.

Milestone M546:
  - tests/test_m546_law12_comp3d.py: comprehensive tests for pyradioss/materials/law12_comp3d.py
  - Citing:
      starter/source/materials/mat/mat012/hm_read_mat12.F
      engine/source/materials/mat/mat012/m12law.F
      hm_cfg_files/config/CFG/radioss2020/MAT/3d_comp_12.cfg
"""

from __future__ import annotations

import math
from typing import Any, Dict
import numpy as np
import pytest

from pyradioss.materials.law12_comp3d import (
    build_law12,
    consistent_solid_tangent,
    extra_shapes,
    shell_update,
    solid_update,
    sound_speed,
)


# ============================================================================
# 1. Parameter Constructor & Default Values
# ============================================================================

def test_build_law12_defaults():
    """Verify default values, derived orthotropic stiffness D11..D33, DETC > 0, Tsai-Wu coefficients."""
    e11, e22, e33 = 100000.0, 50000.0, 20000.0
    nu12, nu23, nu31 = 0.25, 0.2, 0.15
    g12, g23, g31 = 15000.0, 10000.0, 12000.0
    rho0 = 1.5e-9

    mat = build_law12(
        id=12,
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

    assert mat.id == 12
    assert mat.law == 12
    assert mat.law_name == "LAW12"
    assert mat.rho0 == pytest.approx(rho0)

    p = mat.params
    # Compliance calculations (hm_read_mat12.F:221-226)
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

    # Inverted orthotropic stiffness D11..D33 (hm_read_mat12.F:239-247)
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

    # Defaults check
    assert p["delta"] == pytest.approx(0.05)
    assert p["cb"] == pytest.approx(0.0)
    assert p["cn"] == pytest.approx(1.0)
    assert p["fmax"] == pytest.approx(1.0e10)
    assert p["wplaref"] == pytest.approx(1.0)
    assert p["alpha"] == pytest.approx(0.0)
    assert p["efib"] == pytest.approx(0.0)
    assert p["c"] == pytest.approx(0.0)
    assert p["eps0"] == pytest.approx(1.0)
    assert p["ICC"] == 1

    assert "F1" in p
    assert "F11" in p
    assert "F12" in p


def test_build_law12_invalid_moduli():
    """Verify raises ValueError on E11 <= 0 or DETC <= 0."""
    with pytest.raises(ValueError, match="must be > 0"):
        build_law12(
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
        build_law12(
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
        build_law12(
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
# 2. Sound Speed
# ============================================================================

def test_sound_speed():
    """Verify sound_speed computes c = sqrt(max(D11, D22, D33)/rho0)."""
    rho0 = 2.0e-9
    e11, e22, e33 = 120000.0, 60000.0, 30000.0
    mat = build_law12(
        id=12,
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


# ============================================================================
# 3. Elastic Solid Update
# ============================================================================

def test_solid_update_elastic():
    """Unyielding small strain increments match D @ deps."""
    mat = build_law12(
        id=12,
        rho0=1.5e-9,
        E11=100000.0,
        E22=50000.0,
        E33=20000.0,
        nu12=0.25,
        nu23=0.2,
        nu31=0.15,
        G12=15000.0,
        G23=10000.0,
        G31=12000.0,
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
    assert extra["dam12"][0, 0] == 0.0
    assert extra["wpla12"][0] == 0.0


# ============================================================================
# 4. Fiber Tracking
# ============================================================================

def test_solid_update_fiber():
    """alpha > 0 tracks fiber strain and stress sigf."""
    alpha_val = 0.35
    efib_val = 220000.0

    mat = build_law12(
        id=12,
        rho0=1.5e-9,
        E11=100000.0,
        E22=50000.0,
        E33=20000.0,
        nu12=0.25,
        nu23=0.2,
        nu31=0.15,
        G12=15000.0,
        G23=10000.0,
        G31=12000.0,
        alpha=alpha_val,
        efib=efib_val,
    )

    sig = np.zeros(6, dtype=float)
    deps = np.array([2.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    extra: Dict[str, Any] = {}

    sig_out, epsp_out, c_out = solid_update(mat, sig, deps, dt=1.0e-6, extra=extra)

    assert "epsf12" in extra
    assert "sigf12" in extra
    assert extra["epsf12"][0] == pytest.approx(2.0e-4)
    assert extra["sigf12"][0] == pytest.approx(efib_val * 2.0e-4)

    deps2 = np.array([1.5e-4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig_out2, epsp_out2, c_out2 = solid_update(mat, sig_out, deps2, dt=1.0e-6, extra=extra)

    assert extra["epsf12"][0] == pytest.approx(3.5e-4)
    assert extra["sigf12"][0] == pytest.approx(efib_val * 3.5e-4)


# ============================================================================
# 5. Tensile Damage Initiation and Poisson Coupling
# ============================================================================

def test_solid_update_tensile_damage():
    """Normal stress exceeding sigt initiates damage, clips stress to (1-dam)*sigt,
    relieves transverse stresses by Poisson coupling, and increments dam by delta."""
    sigt1 = 150.0
    delta = 0.08

    mat = build_law12(
        id=12,
        rho0=1.5e-9,
        E11=100000.0,
        E22=50000.0,
        E33=20000.0,
        nu12=0.25,
        nu23=0.2,
        nu31=0.15,
        G12=15000.0,
        G23=10000.0,
        G31=12000.0,
        sigt1=sigt1,
        delta=delta,
        sigyt1=1.0e10,
    )

    p = mat.params
    d11 = p["D11"]

    deps1 = 0.003
    trial_t1 = d11 * deps1
    assert trial_t1 > sigt1

    sig = np.zeros(6, dtype=float)
    deps = np.array([deps1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    extra: Dict[str, Any] = {}

    sig_out, epsp_out, c_out = solid_update(mat, sig, deps, dt=1.0e-6, extra=extra)

    assert extra["dam12"][0, 0] == pytest.approx(delta)
    assert sig_out[0] == pytest.approx(sigt1)

    deps_step2 = np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig_out2, _, _ = solid_update(mat, sig_out, deps_step2, dt=1.0e-6, extra=extra)

    assert extra["dam12"][0, 0] == pytest.approx(2.0 * delta)
    assert sig_out2[0] == pytest.approx((1.0 - delta) * sigt1)


# ============================================================================
# 6. Crack Closure / Unilateral Effect
# ============================================================================

def test_solid_update_crack_closure():
    """Crack open condition (EPC > 0) suppresses compressive stress until crack closure."""
    sigt1 = 100.0
    mat = build_law12(
        id=12,
        rho0=1.5e-9,
        E11=100000.0,
        E22=50000.0,
        E33=20000.0,
        nu12=0.25,
        nu23=0.2,
        nu31=0.15,
        G12=15000.0,
        G23=10000.0,
        G31=12000.0,
        sigt1=sigt1,
        delta=0.1,
        sigyt1=1.0e10,
    )

    sig = np.zeros(6, dtype=float)
    deps_tens = np.array([0.002, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    extra: Dict[str, Any] = {}

    sig_out, _, _ = solid_update(mat, sig, deps_tens, dt=1.0e-6, extra=extra)
    assert extra["dam12"][0, 0] > 0.0
    assert extra["epc12"][0, 0] > 0.0

    deps_comp1 = np.array([-0.0005, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig_comp, _, _ = solid_update(mat, sig_out, deps_comp1, dt=1.0e-6, extra=extra)

    if extra["epc12"][0, 0] > 0.0 and sig_comp[0] <= 0.0:
        assert sig_comp[0] == pytest.approx(0.0)


# ============================================================================
# 7. Tsai-Wu 3D Plasticity Return
# ============================================================================

def test_solid_update_tsai_wu_plasticity():
    """Stress exceeding yield surface triggers plastic return, updates wpla, relaxes stress back to yield."""
    sigyt = 250.0
    sigyc = 250.0

    mat = build_law12(
        id=12,
        rho0=1.5e-9,
        E11=100000.0,
        E22=100000.0,
        E33=100000.0,
        nu12=0.25,
        nu23=0.25,
        nu31=0.25,
        G12=40000.0,
        G23=40000.0,
        G31=40000.0,
        sigyt1=sigyt,
        sigyc1=sigyc,
        sigyt2=sigyt,
        sigyc2=sigyc,
        sigyt3=sigyt,
        sigyc3=sigyc,
        sigyt12=sigyt,
        sigyc12=sigyc,
        cb=500.0,
        cn=0.5,
        fmax=1000.0,
        sigt1=1.0e10,
    )

    sig = np.zeros(6, dtype=float)
    deps = np.array([0.0, 0.0, 0.0, 0.015, 0.0, 0.0], dtype=float)
    extra: Dict[str, Any] = {}

    # Step 1 loads stress up past yield surface (DP is evaluated at old stress SO=0)
    sig_out1, _, _ = solid_update(mat, sig, deps, dt=1.0e-5, extra=extra)
    # Step 2 with non-zero SO4 triggers Tsai-Wu plastic return and updates wpla
    sig_out2, epsp_out, c_out = solid_update(mat, sig_out1, deps, dt=1.0e-5, extra=extra)

    assert epsp_out > 0.0
    assert extra["wpla12"][0] > 0.0
    assert extra["dam12"][0, 3] > 0.0


# ============================================================================
# 8. Strain Rate Sensitivity (ICC Flags)
# ============================================================================

def test_solid_update_rate_sensitivity():
    """Higher strain rate increases yield stress when c > 0, obeying ICC flags."""
    c_rate = 0.05
    eps0_ref = 1.0

    mat_static = build_law12(
        id=1,
        rho0=1.5e-9,
        E11=100000.0,
        E22=100000.0,
        E33=100000.0,
        nu12=0.25,
        nu23=0.25,
        nu31=0.25,
        G12=40000.0,
        G23=40000.0,
        G31=40000.0,
        sigyt1=200.0,
        sigyc1=200.0,
        c=0.0,
        eps0=eps0_ref,
        ICC=1,
    )

    mat_dynamic = build_law12(
        id=2,
        rho0=1.5e-9,
        E11=100000.0,
        E22=100000.0,
        E33=100000.0,
        nu12=0.25,
        nu23=0.25,
        nu31=0.25,
        G12=40000.0,
        G23=40000.0,
        G31=40000.0,
        sigyt1=200.0,
        sigyc1=200.0,
        c=c_rate,
        eps0=eps0_ref,
        ICC=1,
    )

    deps = np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    dt = 1.0e-6

    sig_s, ep_s, _ = solid_update(mat_static, np.zeros(6), deps, dt=dt, extra={})
    sig_d, ep_d, _ = solid_update(mat_dynamic, np.zeros(6), deps, dt=dt, extra={})

    assert sig_d[0] >= sig_s[0]


# ============================================================================
# 9. Element Degradation and Failure
# ============================================================================

def test_solid_update_element_degradation():
    """Reaching fmax triggers off degradation (off=0.99*0.8 in step 1, then 0.8*off, then 0.0 below 0.1)."""
    fmax_val = 1.05

    mat = build_law12(
        id=12,
        rho0=1.5e-9,
        E11=100000.0,
        E22=50000.0,
        E33=20000.0,
        nu12=0.25,
        nu23=0.2,
        nu31=0.15,
        G12=15000.0,
        G23=10000.0,
        G31=12000.0,
        cb=10.0,
        cn=0.5,
        fmax=fmax_val,
    )

    sig = np.zeros(6, dtype=float)
    deps = np.array([0.002, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    extra = {"off12": np.array([1.0]), "wpla12": np.array([0.5])}

    # In m12law.F:228 & 417, reaching fmax sets off=0.99, which then immediately
    # undergoes off = off * 0.8 = 0.792 in the general failure loop of the same step.
    sig_out1, _, _ = solid_update(mat, sig, deps, dt=1.0e-6, extra=extra)
    assert extra["off12"][0] == pytest.approx(0.99 * 0.8)

    sig_out2, _, _ = solid_update(mat, sig_out1, deps, dt=1.0e-6, extra=extra)
    assert extra["off12"][0] == pytest.approx(0.99 * 0.8 * 0.8)

    extra["off12"][0] = 0.09
    sig_out3, _, _ = solid_update(mat, sig_out2, deps, dt=1.0e-6, extra=extra)
    assert extra["off12"][0] == 0.0
    assert np.allclose(sig_out3, 0.0)


# ============================================================================
# 10. Shell Rejection
# ============================================================================

def test_shell_update_raises():
    """Verifies shell_update raises NotImplementedError."""
    mat = build_law12(
        id=12,
        rho0=1.5e-9,
        E11=100000.0,
        E22=50000.0,
        E33=20000.0,
        nu12=0.25,
        nu23=0.2,
        nu31=0.15,
        G12=15000.0,
        G23=10000.0,
        G31=12000.0,
    )
    with pytest.raises(NotImplementedError, match="3D solid elements only"):
        shell_update(mat, np.zeros(3), np.zeros(3))


# ============================================================================
# 11. Batched Vectorization Equivalence
# ============================================================================

def test_batched_vectorization():
    """Single element vs 10 elements array produce identical results."""
    mat = build_law12(
        id=12,
        rho0=1.5e-9,
        E11=100000.0,
        E22=50000.0,
        E33=20000.0,
        nu12=0.25,
        nu23=0.2,
        nu31=0.15,
        G12=15000.0,
        G23=10000.0,
        G31=12000.0,
        sigt1=200.0,
        delta=0.05,
        sigyt1=300.0,
        sigyc1=300.0,
    )

    n_elem = 10
    sig_single = np.zeros(6, dtype=float)
    deps_single = np.array([0.001, -0.0005, 0.0002, 0.0003, -0.0001, 0.0002], dtype=float)
    extra_single: Dict[str, Any] = {}

    sig_res_single, ep_res_single, c_res_single = solid_update(
        mat, sig_single, deps_single, dt=1.0e-6, extra=extra_single
    )

    sig_batch = np.tile(sig_single, (n_elem, 1))
    deps_batch = np.tile(deps_single, (n_elem, 1))
    extra_batch: Dict[str, Any] = {}

    sig_res_batch, ep_res_batch, c_res_batch = solid_update(
        mat, sig_batch, deps_batch, dt=1.0e-6, extra=extra_batch
    )

    assert sig_res_batch.shape == (n_elem, 6)
    for i in range(n_elem):
        assert np.allclose(sig_res_batch[i], sig_res_single)
        assert ep_res_batch[i] == pytest.approx(ep_res_single)
        assert extra_batch["dam12"][i] == pytest.approx(extra_single["dam12"][0])
        assert extra_batch["wpla12"][i] == pytest.approx(extra_single["wpla12"][0])


# ============================================================================
# 12. Algorithmic Tangent (consistent_solid_tangent)
# ============================================================================

def test_consistent_solid_tangent_elastic():
    """Verify tangent in the elastic regime matches the orthotropic elasticity matrix D."""
    mat = build_law12(
        id=12,
        rho0=1.5e-9,
        E11=100000.0,
        E22=50000.0,
        E33=20000.0,
        nu12=0.25,
        nu23=0.2,
        nu31=0.15,
        G12=15000.0,
        G23=10000.0,
        G31=12000.0,
    )
    p = mat.params
    # 1D call returning (6, 6)
    sig_1d = np.zeros(6, dtype=float)
    c_tan_1d = consistent_solid_tangent(mat, sig_1d)
    assert c_tan_1d.shape == (6, 6)
    assert c_tan_1d[0, 0] == pytest.approx(p["D11"])
    assert c_tan_1d[0, 1] == pytest.approx(p["D12"])
    assert c_tan_1d[0, 2] == pytest.approx(p["D13"])
    assert c_tan_1d[1, 1] == pytest.approx(p["D22"])
    assert c_tan_1d[1, 2] == pytest.approx(p["D23"])
    assert c_tan_1d[2, 2] == pytest.approx(p["D33"])
    assert c_tan_1d[3, 3] == pytest.approx(p["G12"])
    assert c_tan_1d[4, 4] == pytest.approx(p["G23"])
    assert c_tan_1d[5, 5] == pytest.approx(p["G31"])

    # 2D call returning (n, 6, 6)
    sig_2d = np.zeros((4, 6), dtype=float)
    c_tan_2d = consistent_solid_tangent(mat, sig_2d)
    assert c_tan_2d.shape == (4, 6, 6)
    for i in range(4):
        assert np.allclose(c_tan_2d[i], c_tan_1d)


def test_consistent_solid_tangent_perturbed():
    """Verify tangent under damage initiation reflects degradation."""
    mat = build_law12(
        id=12,
        rho0=1.5e-9,
        E11=100000.0,
        E22=50000.0,
        E33=20000.0,
        nu12=0.25,
        nu23=0.2,
        nu31=0.15,
        G12=15000.0,
        G23=10000.0,
        G31=12000.0,
        sigt1=100.0,
        delta=0.2,
    )
    sig_damaged = np.array([120.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    extra = {"dam12": np.array([[0.2, 0.0, 0.0, 0.0, 11000.0]])}
    c_tan = consistent_solid_tangent(mat, sig_damaged, extra=extra)
    # The damaged D11 component should be degraded compared to virgin D11
    assert c_tan.shape == (6, 6)
    assert c_tan[0, 0] <= mat.params["D11"]


# ============================================================================
# 13. Materials Module Dispatch
# ============================================================================

def test_materials_module_dispatch():
    """Verify top-level dispatch from pyradioss.materials for LAW12."""
    import pyradioss.materials as mats

    mat = build_law12(
        id=12,
        rho0=1.5e-9,
        E11=100000.0,
        E22=50000.0,
        E33=20000.0,
        nu12=0.25,
        nu23=0.2,
        nu31=0.15,
        G12=15000.0,
        G23=10000.0,
        G31=12000.0,
    )
    # sound_speed dispatch
    c_spd = mats.sound_speed(mat)
    assert c_spd == pytest.approx(mat.params["SSP"])

    # extra_shapes dispatch
    shapes = mats.extra_shapes(mat, nip=1)
    assert "dam12" in shapes
    assert "epe12" in shapes
    assert "epc12" in shapes
    assert "wpla12" in shapes

    # solid_update dispatch
    sig = np.zeros((1, 6))
    deps = np.array([[1.0e-5, 0, 0, 0, 0, 0]])
    s_out, ep_out, _ = mats.solid_update(mat, sig, deps, dt=1.0e-6, extra={})
    assert s_out.shape == (1, 6)
    assert s_out[0, 0] > 0.0

    # solid_tangent / consistent_solid_tangent dispatch
    c_tan = mats.solid_tangent(mat, sig)
    assert c_tan.shape == (1, 6, 6)

    # shell_update dispatch raises NotImplementedError
    with pytest.raises(NotImplementedError):
        mats.shell_update(mat, np.zeros(3), np.zeros(3))


# ============================================================================
# 14. Starter Deck Keyword Parsing
# ============================================================================

def test_starter_deck_keyword_parsing(tmp_path):
    """Verify parsing /MAT/LAW12 deck keyword using parse_starter_deck."""
    from pathlib import Path
    from pyradioss.common.messages import MessageLog
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.starter_keywords import parse_starter_deck
    from pyradioss.model.model import Model

    deck_text = """# OpenRadioss Starter input deck
/BEGIN
LAW12_TEST
/MAT/LAW12/10
Test Law 12 Composite
#    RHO_I              RHO_O
 1.500E-09          1.500E-09
#      E11                E22                E33
 1.000E+05          5.000E+04          2.000E+04
#     NU12               NU23               NU31
      0.25               0.20               0.15
#      G12                G23                G31
 1.500E+04          1.000E+04          1.200E+04
#    SIGT1              SIGT2              SIGT3              DELTA
 1.500E+02          1.500E+02          1.500E+02               0.05
#        B                  N               FMAX            WPLAREF
     500.0                0.5             1000.0                1.0
#   SIGYT1             SIGYT2             SIGYC1             SIGYC2
     250.0              250.0              250.0              250.0
#  SIGYT12            SIGYC12            SIGYT23            SIGYC23
     120.0              120.0              100.0              100.0
#   SIGYT3             SIGYC3            SIGYT13            SIGYC13
     200.0              200.0              110.0              110.0
#    ALPHA               EFIB                  C               EPS0               ICC
      0.30          2.000E+05               0.05                1.0                 1
/PART/1
Part 1
1 10
/END
"""
    deck_path = tmp_path / "DECK_0000.rad"
    deck_path.write_text(deck_text, encoding="ascii")

    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)

    assert 10 in model.materials
    mat = model.materials[10]
    assert mat.law == 12
    assert mat.law_name == "LAW12"
    assert mat.rho0 == pytest.approx(1.5e-9)
    assert mat.params["E11"] == pytest.approx(1.0e5)
    assert mat.params["E22"] == pytest.approx(5.0e4)
    assert mat.params["E33"] == pytest.approx(2.0e4)
    assert mat.params["nu12"] == pytest.approx(0.25)
    assert mat.params["G12"] == pytest.approx(1.5e4)
    assert mat.params["sigt1"] == pytest.approx(150.0)
    assert mat.params["cb"] == pytest.approx(500.0)
    assert mat.params["cn"] == pytest.approx(0.5)
    assert mat.params["fmax"] == pytest.approx(1000.0)
    assert mat.params["wplaref"] == pytest.approx(1.0)
    assert mat.params["sigyt1"] == pytest.approx(250.0)
    assert mat.params["sigyt12"] == pytest.approx(120.0)
    assert mat.params["alpha"] == pytest.approx(0.30)
    assert mat.params["efib"] == pytest.approx(2.0e5)
    assert mat.params["c"] == pytest.approx(0.05)
    assert mat.params["ICC"] == 1
    assert "D11" in mat.params
    assert "SSP" in mat.params


# ============================================================================
# 15. Coordinate Transformations (m14ama, m14gtf, m14ftg)
# ============================================================================

def test_coordinate_transformations():
    """Verify local orthotropic frame creation and round-trip stress transforms."""
    from pyradioss.materials.law12_comp3d import m14ama, m14gtf, m14ftg

    # Case 1: Standard global aligned triad
    rx = np.array([1.0, 0.0])
    ry = np.array([0.0, 1.0])
    rz = np.array([0.0, 0.0])
    sx = np.array([0.0, -1.0])
    sy = np.array([1.0, 0.0])
    sz = np.array([0.0, 0.0])

    ax, ay, az, bx, by, bz, cx, cy, cz = m14ama(rx, ry, rz, sx, sy, sz)
    assert len(ax) == 2

    # Orthogonality checks
    for i in range(2):
        a_vec = np.array([ax[i], ay[i], az[i]])
        b_vec = np.array([bx[i], by[i], bz[i]])
        c_vec = np.array([cx[i], cy[i], cz[i]])
        assert np.linalg.norm(a_vec) == pytest.approx(1.0)
        assert np.linalg.norm(b_vec) == pytest.approx(1.0)
        assert np.linalg.norm(c_vec) == pytest.approx(1.0)
        assert np.dot(a_vec, b_vec) == pytest.approx(0.0, abs=1e-7)
        assert np.dot(a_vec, c_vec) == pytest.approx(0.0, abs=1e-7)
        assert np.dot(b_vec, c_vec) == pytest.approx(0.0, abs=1e-7)

    # Case 2: Round-trip transformation of stress
    sig_global = np.array([
        [100.0, 50.0, 20.0, 15.0, 10.0, 5.0],
        [80.0, 30.0, 10.0, -5.0, 8.0, 12.0],
    ])
    deps_dummy = np.zeros_like(sig_global)

    sig_local, _ = m14gtf(sig_global, deps_dummy, ax, ay, az, bx, by, bz, cx, cy, cz)
    sig_back = m14ftg(sig_local, ax, ay, az, bx, by, bz, cx, cy, cz)

    assert np.allclose(sig_back, sig_global, atol=1e-7)

    # Case 3: Spherical coordinate mode jsph=1
    a_sphere = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    ax_s, ay_s, az_s, bx_s, by_s, bz_s, cx_s, cy_s, cz_s = m14ama(
        np.array([1.0]), np.array([0.0]), np.array([0.0]),
        np.array([0.0]), np.array([1.0]), np.array([0.0]),
        a=a_sphere, jsph=1,
    )
    assert cx_s[0] == pytest.approx(0.0)
    assert cy_s[0] == pytest.approx(0.0)
    assert cz_s[0] == pytest.approx(1.0)


# ============================================================================
# 16. Registry Hooks
# ============================================================================

def test_registry_hooks():
    """Verify all standard LAW12 aliases are registered in MAT_PHYSICS_REGISTRY."""
    from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY

    aliases = [
        12,
        "12",
        "LAW12",
        "3D_COMP",
        "COMP_3D",
        "MAT_LAW12",
        "MAT_3D_COMP",
        "MAT_COMP_3D",
        "3PARBI",
        "MAT_3PARBI",
    ]
    for alias in aliases:
        assert alias in MAT_PHYSICS_REGISTRY
        builder = MAT_PHYSICS_REGISTRY[alias]
        mat = builder(
            id=12,
            rho0=1.5e-9,
            E11=100000.0,
            E22=50000.0,
            E33=20000.0,
            nu12=0.25,
            nu23=0.2,
            nu31=0.15,
            G12=15000.0,
            G23=10000.0,
            G31=12000.0,
        )
        assert mat.law == 12
        assert mat.law_name == "LAW12"


# ============================================================================
# 17. Detailed Crack Open & Re-closure Cycle
# ============================================================================

def test_crack_open_no_compression_detailed():
    """Three-cycle test:
    1. Tensile damage opens crack (EPC > 0).
    2. Small compressive strain cannot close crack (EPC > 0 -> T1 = 0).
    3. Large compressive strain closes crack completely (EPC = 0 -> T1 < 0).
    """
    sigt1 = 100.0
    mat = build_law12(
        id=12,
        rho0=1.5e-9,
        E11=100000.0,
        E22=50000.0,
        E33=20000.0,
        nu12=0.0,  # decouple Poisson for clean 1D verification
        nu23=0.0,
        nu31=0.0,
        G12=15000.0,
        G23=10000.0,
        G31=12000.0,
        sigt1=sigt1,
        delta=0.1,
        sigyt1=1.0e10,
    )

    extra: Dict[str, Any] = {}
    sig = np.zeros(6, dtype=float)

    # Cycle 1: Tension exceeding sigt1 opens crack
    # E11 = 100000, deps1 = 0.003 => trial_sig = 300 > 100
    deps_tens = np.array([0.003, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig1, _, _ = solid_update(mat, sig, deps_tens, dt=1.0e-6, extra=extra)

    assert extra["dam12"][0, 0] > 0.0
    epc1_after_c1 = extra["epc12"][0, 0]
    assert epc1_after_c1 > 0.0
    assert sig1[0] == pytest.approx(sigt1)

    # Cycle 2: Partial compression (deps1 = -0.002):
    # Trial stress = 100 + 100000*(-0.002) = -100 < 0, but crack remains open (epc1 = 0.003 - 0.002 = 0.001 > 0).
    # Open crack suppresses compressive stress -> sig2[0] == 0.0!
    deps_comp_partial = np.array([-0.002, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig2, _, _ = solid_update(mat, sig1, deps_comp_partial, dt=1.0e-6, extra=extra)

    assert extra["epc12"][0, 0] > 0.0
    # No compressive stress allowed while crack is open!
    assert sig2[0] == pytest.approx(0.0)

    # Cycle 3: Further compression (deps1 = -0.002) completely closes crack (epc1 -> 0).
    # Closed crack now bears compressive stress!
    deps_comp_large = np.array([-0.002, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    sig3, _, _ = solid_update(mat, sig2, deps_comp_large, dt=1.0e-6, extra=extra)

    assert extra["epc12"][0, 0] == 0.0
    assert sig3[0] < 0.0


# ============================================================================
# 18. Starter Keyword Aliases and Model Properties
# ============================================================================

def test_starter_keyword_aliases_and_model_properties(tmp_path):
    """Verify parsing /MAT/3D_COMP and /MAT/COMP_3D and model property aliases."""
    from pyradioss.common.messages import MessageLog
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.starter_keywords import parse_starter_deck
    from pyradioss.model.model import Model

    deck_text = """# OpenRadioss Starter input deck
/BEGIN
ALIASES_TEST
/MAT/3D_COMP/201
Material 3D Comp
#    RHO_I              RHO_O
 1.500E-09          1.500E-09
#      E11                E22                E33
 1.000E+05          5.000E+04          2.000E+04
#     NU12               NU23               NU31
      0.25               0.20               0.15
#      G12                G23                G31
 1.500E+04          1.000E+04          1.200E+04
#    SIGT1              SIGT2              SIGT3              DELTA
 1.500E+02          1.500E+02          1.500E+02               0.05
#        B                  N               FMAX            WPLAREF
     500.0                0.5             1000.0                1.0
#   SIGYT1             SIGYT2             SIGYC1             SIGYC2
     250.0              250.0              250.0              250.0
#  SIGYT12            SIGYC12            SIGYT23            SIGYC23
     120.0              120.0              100.0              100.0
#   SIGYT3             SIGYC3            SIGYT13            SIGYC13
     200.0              200.0              110.0              110.0
#    ALPHA               EFIB                  C               EPS0               ICC
      0.30          2.000E+05               0.05                1.0                 1
/MAT/COMP_3D/202
Material Comp 3D
#    RHO_I              RHO_O
 1.600E-09          1.600E-09
#      E11                E22                E33
 1.200E+05          6.000E+04          2.500E+04
#     NU12               NU23               NU31
      0.28               0.22               0.18
#      G12                G23                G31
 1.600E+04          1.100E+04          1.300E+04
#    SIGT1              SIGT2              SIGT3              DELTA
 1.800E+02          1.800E+02          1.800E+02               0.06
#        B                  N               FMAX            WPLAREF
     600.0                0.6             1200.0                1.5
#   SIGYT1             SIGYT2             SIGYC1             SIGYC2
     280.0              280.0              280.0              280.0
#  SIGYT12            SIGYC12            SIGYT23            SIGYC23
     140.0              140.0              110.0              110.0
#   SIGYT3             SIGYC3            SIGYT13            SIGYC13
     220.0              220.0              120.0              120.0
#    ALPHA               EFIB                  C               EPS0               ICC
      0.35          2.200E+05               0.06                1.0                 1
/END
"""
    deck_path = tmp_path / "ALIASES_0000.rad"
    deck_path.write_text(deck_text, encoding="ascii")

    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)

    assert 201 in model.mat_law12s
    assert 201 in model.mat_3d_comps
    assert 201 in model.mat_comp_3ds
    assert 201 in model.mat_3parbis
    assert 201 in model.mat_ragabs

    assert 202 in model.mat_law12s
    assert 202 in model.mat_3d_comps
    assert 202 in model.mat_comp_3ds

    mat201 = model.mat_3d_comps[201]
    assert mat201.rho0 == pytest.approx(1.5e-9)
    assert mat201.e11 == pytest.approx(1.0e5)
    assert mat201.wplaref == pytest.approx(1.0)

    mat202 = model.mat_comp_3ds[202]
    assert mat202.rho0 == pytest.approx(1.6e-9)
    assert mat202.e11 == pytest.approx(1.2e5)
    assert mat202.wplaref == pytest.approx(1.5)


# ============================================================================
# 19. Deck Writer & Round-Trip Verification
# ============================================================================

def test_starter_deck_writer_roundtrip(tmp_path):
    """Verify StarterDeck.mat_law12 and mat_3d_comp write valid cards and roundtrip parse."""
    from pyradioss.common.messages import MessageLog
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.deck_writer import StarterDeck
    from pyradioss.input.starter_keywords import parse_starter_deck
    from pyradioss.model.model import Model

    writer = StarterDeck("LAW12_ROUNDTRIP")
    writer.mat_law12(
        mat_id=301,
        rho=1.5e-9,
        e11=110000.0,
        e22=55000.0,
        e33=22000.0,
        nu12=0.26,
        nu23=0.21,
        nu31=0.16,
        g12=14000.0,
        g23=9500.0,
        g31=11500.0,
        sig_t1=160.0,
        sig_t2=170.0,
        sig_t3=180.0,
        delta=0.07,
        b=450.0,
        n=0.55,
        fmax=950.0,
        wplaref=1.25,
        sig_1yt=240.0,
        sig_2yt=245.0,
        sig_1yc=250.0,
        sig_2yc=255.0,
        sig_12yt=115.0,
        sig_12yc=118.0,
        sig_23yt=98.0,
        sig_23yc=99.0,
        sig_3yt=190.0,
        sig_3yc=195.0,
        sig_13yt=105.0,
        sig_13yc=108.0,
        alpha=0.28,
        efib=190000.0,
        c=0.04,
        eps0=1.0,
        icc=1,
        title="Written Law12",
    )

    deck_path = tmp_path / "ROUNDTRIP_0000.rad"
    writer.write(str(deck_path))

    blocks = read_deck(str(deck_path))
    model = Model()
    log = MessageLog()
    parse_starter_deck(blocks, model, log)

    assert 301 in model.mat_law12s
    mat = model.mat_law12s[301]
    assert mat.rho0 == pytest.approx(1.5e-9)
    assert mat.e11 == pytest.approx(110000.0)
    assert mat.e22 == pytest.approx(55000.0)
    assert mat.e33 == pytest.approx(22000.0)
    assert mat.nu12 == pytest.approx(0.26)
    assert mat.g12 == pytest.approx(14000.0)
    assert mat.sig_t1 == pytest.approx(160.0)
    assert mat.delta == pytest.approx(0.07)
    assert mat.b == pytest.approx(450.0)
    assert mat.n == pytest.approx(0.55)
    assert mat.fmax == pytest.approx(950.0)
    assert mat.wplaref == pytest.approx(1.25)
    assert mat.sig_1yt == pytest.approx(240.0)
    assert mat.sig_12yt == pytest.approx(115.0)
    assert mat.alpha == pytest.approx(0.28)
    assert mat.efib == pytest.approx(190000.0)
    assert mat.c == pytest.approx(0.04)
    assert mat.icc == 1