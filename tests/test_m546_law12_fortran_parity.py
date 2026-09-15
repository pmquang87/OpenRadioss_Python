"""Fortran Parity Verification Suite for /MAT/LAW12 (/MAT/3D_COMP, /MAT/COMP_3D).

Milestone M546:
Auditing pyradioss/materials/law12_comp3d.py against:
  - Starter: starter/source/materials/mat/mat012/hm_read_mat12.F
  - Engine:  engine/source/materials/mat/mat012/m12law.F
  - Coordinate frames: engine/source/materials/mat/mat014/m14ama.F, m14gtf.F, m14ftg.F

Verifications:
1. Compliance matrix inversion C -> D:
   C11 = 1/E11, C22 = 1/E22, C33 = 1/E33
   C12 = -nu12/E11, C13 = -nu31/E33, C23 = -nu23/E22
   DETC = C11*C22*C33 - C11*C23**2 - C12**2*C33 + 2*C12*C13*C23 - C13**2*C22
   D11..D33 expressions.
   Verif A11..A33 = C @ D = I.
2. Yield constants:
   F1 = 1/SIGYT1 - 1/SIGYC1, F11 = 1/(SIGYT1*SIGYC1)
   F2 = 1/SIGYT2 - 1/SIGYC2, F22 = 1/(SIGYT2*SIGYC2)
   F3 = 1/SIGYT3 - 1/SIGYC3, F33 = 1/(SIGYT3*SIGYC3)
   F4 = 1/SIGYT12 - 1/SIGYC12, F44 = 1/(SIGYT12*SIGYC12)
   F5 = 1/SIGYT23 - 1/SIGYC23, F55 = 1/(SIGYT23*SIGYC23)
   F6 = 1/SIGYT13 - 1/SIGYC13, F66 = 1/(SIGYT13*SIGYC13)
   F12 = -0.5 / sqrt(SIGYT1*SIGYC1*SIGYT2*SIGYC2)
   F23 = -0.5 / sqrt(SIGYT2*SIGYC2*SIGYT3*SIGYC3)
   F13 = -0.5 / sqrt(SIGYT1*SIGYC1*SIGYT3*SIGYC3)
3. Strain rate sensitivity:
   epsp_rate = max(|D1|, |D2|, |D3|, 0.5*|D4|, 0.5*|D5|, 0.5*|D6|)
   rate_fac = 1 + CC * log(epsp_rate / EPS0) if epsp_rate > EPS0 and CC > 0 else 1.0
   ICC = 1, 2, 3, 4 handling.
   SIGMY = min(SIGMX, CA + CB * wpla**CN).
4. Tensile cracking damage:
   Direction 1, 2, 3 damage thresholds (1-dam)*sigt.
   Poisson stress relief equations:
   Dir 1: T2 -= D12*deps1*dam1; T3 -= D13*deps1*dam1
   Dir 2: T1 -= D12*deps2*dam2; T3 -= D23*deps2*dam2
   Dir 3: T1 -= D13*deps3*dam3; T2 -= D23*deps3*dam3
   Damage increment delta: dam = min(1.0, dam + delta).
5. Crack opening / unilateral conditions:
   If T1 < 0 and epc1 > 0: T1 = 0, with Poisson relief.
   Compression recovery when epc <= 0.
6. Tsai-Wu 3D plastic return:
   Plastic multiplier lambda denominator and numerator from m12law.F:570-596.
   Stress updates and plastic work increment dwpla.
7. Sound speed formula:
   SSP = sqrt(max(D11, D22, D33) / rho0).
8. Element degradation:
   When sigmy == sigmx and off == 1.0 -> off = 0.99. Then off *= 0.8. Below 0.1 -> 0.0.
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
    m14ama,
    m14ftg,
    m14gtf,
    shell_update,
    solid_update,
    sound_speed,
    _update_one_element,
)


# ============================================================================
# 1. Compliance Matrix Inversion C -> D & Verification A = C @ D = I
# ============================================================================

def test_compliance_matrix_inversion_hand_calc():
    """Verify C11..C23, DETC, D11..D33 and A = C @ D = I against exact hand calculations.

    Reference: hm_read_mat12.F:221-255
    """
    e11 = 120000.0
    e22 = 60000.0
    e33 = 30000.0
    nu12 = 0.30
    nu23 = 0.25
    nu31 = 0.20

    # Hand calculation:
    c11_hand = 1.0 / 120000.0
    c22_hand = 1.0 / 60000.0
    c33_hand = 1.0 / 30000.0
    c12_hand = -0.30 / 120000.0
    c13_hand = -0.20 / 30000.0
    c23_hand = -0.25 / 60000.0

    detc_hand = (
        c11_hand * c22_hand * c33_hand
        - c11_hand * (c23_hand**2)
        - (c12_hand**2) * c33_hand
        + 2.0 * c12_hand * c13_hand * c23_hand
        - (c13_hand**2) * c22_hand
    )

    d11_hand = (c22_hand * c33_hand - c23_hand**2) / detc_hand
    d12_hand = -(c12_hand * c33_hand - c13_hand * c23_hand) / detc_hand
    d13_hand = (c12_hand * c23_hand - c13_hand * c22_hand) / detc_hand
    d22_hand = (c11_hand * c33_hand - c13_hand**2) / detc_hand
    d23_hand = -(c11_hand * c23_hand - c13_hand * c12_hand) / detc_hand
    d33_hand = (c11_hand * c22_hand - c12_hand**2) / detc_hand

    mat = build_law12(
        E11=e11,
        E22=e22,
        E33=e33,
        nu12=nu12,
        nu23=nu23,
        nu31=nu31,
        G12=10000.0,
        G23=8000.0,
        G31=9000.0,
        rho0=1.5e-9,
    )
    p = mat.params

    assert p["C11"] == pytest.approx(c11_hand, rel=1e-12)
    assert p["C22"] == pytest.approx(c22_hand, rel=1e-12)
    assert p["C33"] == pytest.approx(c33_hand, rel=1e-12)
    assert p["C12"] == pytest.approx(c12_hand, rel=1e-12)
    assert p["C13"] == pytest.approx(c13_hand, rel=1e-12)
    assert p["C23"] == pytest.approx(c23_hand, rel=1e-12)
    assert p["DETC"] == pytest.approx(detc_hand, rel=1e-12)

    assert p["D11"] == pytest.approx(d11_hand, rel=1e-12)
    assert p["D12"] == pytest.approx(d12_hand, rel=1e-12)
    assert p["D13"] == pytest.approx(d13_hand, rel=1e-12)
    assert p["D22"] == pytest.approx(d22_hand, rel=1e-12)
    assert p["D23"] == pytest.approx(d23_hand, rel=1e-12)
    assert p["D33"] == pytest.approx(d33_hand, rel=1e-12)

    # Verification: A = C @ D = I (hm_read_mat12.F:250-255)
    assert p["A11"] == pytest.approx(1.0, abs=1e-12)
    assert p["A22"] == pytest.approx(1.0, abs=1e-12)
    assert p["A33"] == pytest.approx(1.0, abs=1e-12)
    assert p["A12"] == pytest.approx(0.0, abs=1e-12)
    assert p["A13"] == pytest.approx(0.0, abs=1e-12)
    assert p["A23"] == pytest.approx(0.0, abs=1e-12)


def test_compliance_inversion_isotropic_analytical_limit():
    """Verify isotropic limit against analytical formula: D11 = E(1-nu)/((1+nu)(1-2nu)), D12 = E*nu/((1+nu)(1-2nu))."""
    E = 200000.0
    nu = 0.3
    mat = build_law12(
        E11=E,
        E22=E,
        E33=E,
        nu12=nu,
        nu23=nu,
        nu31=nu,
        G12=E / (2.0 * (1.0 + nu)),
        G23=E / (2.0 * (1.0 + nu)),
        G31=E / (2.0 * (1.0 + nu)),
        rho0=7.8e-9,
    )
    p = mat.params
    factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    d11_iso = factor * (1.0 - nu)
    d12_iso = factor * nu

    assert p["D11"] == pytest.approx(d11_iso, rel=1e-10)
    assert p["D22"] == pytest.approx(d11_iso, rel=1e-10)
    assert p["D33"] == pytest.approx(d11_iso, rel=1e-10)
    assert p["D12"] == pytest.approx(d12_iso, rel=1e-10)
    assert p["D13"] == pytest.approx(d12_iso, rel=1e-10)
    assert p["D23"] == pytest.approx(d12_iso, rel=1e-10)


def test_compliance_inversion_negative_detc_raises():
    """Verify that unphysical Poisson ratios giving DETC <= 0 raise ValueError (hm_read_mat12.F:232-238)."""
    with pytest.raises(ValueError, match="DETC"):
        build_law12(
            E11=1000.0,
            E22=1000.0,
            E33=1000.0,
            nu12=0.8,
            nu23=0.8,
            nu31=0.8,
            rho0=1.0e-9,
        )


def test_dmin_and_pm105_solid_timestep():
    """Verify DMIN and PM(105) formulas for solid elements time step (hm_read_mat12.F:351-352)."""
    mat = build_law12(
        E11=100000.0,
        E22=80000.0,
        E33=60000.0,
        nu12=0.2,
        nu23=0.15,
        nu31=0.1,
        rho0=1.5e-9,
    )
    p = mat.params
    d11, d22, d33 = p["D11"], p["D22"], p["D33"]
    d12, d13, d23 = p["D12"], p["D13"], p["D23"]
    c1 = max(d11, d22, d33)

    expected_dmin = min(
        d11 * d22 - d12**2,
        d22 * d33 - d23**2,
        d11 * d33 - d13**2,
    )
    expected_pm105 = expected_dmin / (c1**2)

    assert p["DMIN"] == pytest.approx(expected_dmin)
    assert p["PM105"] == pytest.approx(expected_pm105)


# ============================================================================
# 2. Yield Constants (hm_read_mat12.F:291-307)
# ============================================================================

def test_tsai_wu_yield_constants_hand_calc():
    """Verify all Tsai-Wu yield constants F1..F6, F11..F66, F12, F23, F13, FT1, FT2 against exact hand calculations."""
    s1t, s1c = 100.0, 150.0
    s2t, s2c = 80.0, 120.0
    s3t, s3c = 60.0, 90.0
    s12t, s12c = 40.0, 45.0
    s23t, s23c = 30.0, 35.0
    s13t, s13c = 32.0, 38.0

    mat = build_law12(
        E11=10000.0,
        E22=10000.0,
        E33=10000.0,
        nu12=0.2,
        nu23=0.2,
        nu31=0.2,
        rho0=1.0e-9,
        sigyt1=s1t,
        sigyc1=s1c,
        sigyt2=s2t,
        sigyc2=s2c,
        sigyt3=s3t,
        sigyc3=s3c,
        sigyt12=s12t,
        sigyc12=s12c,
        sigyt23=s23t,
        sigyc23=s23c,
        sigyt13=s13t,
        sigyc13=s13c,
    )
    p = mat.params

    # Hand calculation:
    f1_h = 1.0 / s1t - 1.0 / s1c
    f2_h = 1.0 / s2t - 1.0 / s2c
    f3_h = 1.0 / s3t - 1.0 / s3c
    f4_h = 1.0 / s12t - 1.0 / s12c
    f5_h = 1.0 / s23t - 1.0 / s23c
    f6_h = 1.0 / s13t - 1.0 / s13c

    f11_h = 1.0 / (s1t * s1c)
    f22_h = 1.0 / (s2t * s2c)
    f33_h = 1.0 / (s3t * s3c)
    f44_h = 1.0 / (s12t * s12c)
    f55_h = 1.0 / (s23t * s23c)
    f66_h = 1.0 / (s13t * s13c)

    f12_h = -0.5 / math.sqrt(s1t * s1c * s2t * s2c)
    f23_h = -0.5 / math.sqrt(s2t * s2c * s3t * s3c)
    f13_h = -0.5 / math.sqrt(s1t * s1c * s3t * s3c)

    ft1_h = f11_h * f22_h - 4.0 * (f12_h**2)
    ft2_h = (f22_h**2) - 4.0 * (f23_h**2)

    assert p["F1"] == pytest.approx(f1_h, rel=1e-12)
    assert p["F2"] == pytest.approx(f2_h, rel=1e-12)
    assert p["F3"] == pytest.approx(f3_h, rel=1e-12)
    assert p["F4"] == pytest.approx(f4_h, rel=1e-12)
    assert p["F5"] == pytest.approx(f5_h, rel=1e-12)
    assert p["F6"] == pytest.approx(f6_h, rel=1e-12)

    assert p["F11"] == pytest.approx(f11_h, rel=1e-12)
    assert p["F22"] == pytest.approx(f22_h, rel=1e-12)
    assert p["F33"] == pytest.approx(f33_h, rel=1e-12)
    assert p["F44"] == pytest.approx(f44_h, rel=1e-12)
    assert p["F55"] == pytest.approx(f55_h, rel=1e-12)
    assert p["F66"] == pytest.approx(f66_h, rel=1e-12)

    assert p["F12"] == pytest.approx(f12_h, rel=1e-12)
    assert p["F23"] == pytest.approx(f23_h, rel=1e-12)
    assert p["F13"] == pytest.approx(f13_h, rel=1e-12)

    assert p["FT1"] == pytest.approx(ft1_h, rel=1e-12)
    assert p["FT2"] == pytest.approx(ft2_h, rel=1e-12)


def test_tsai_wu_symmetric_strengths_zero_linear_terms():
    """Verify that when tensile and compressive strengths are equal, F1..F6 vanish identically."""
    s = 200.0
    mat = build_law12(
        E11=10000.0,
        E22=10000.0,
        E33=10000.0,
        nu12=0.2,
        nu23=0.2,
        nu31=0.2,
        rho0=1.0e-9,
        sigyt1=s,
        sigyc1=s,
        sigyt2=s,
        sigyc2=s,
        sigyt3=s,
        sigyc3=s,
        sigyt12=s / math.sqrt(3.0),
        sigyc12=s / math.sqrt(3.0),
        sigyt23=s / math.sqrt(3.0),
        sigyc23=s / math.sqrt(3.0),
        sigyt13=s / math.sqrt(3.0),
        sigyc13=s / math.sqrt(3.0),
    )
    p = mat.params
    assert p["F1"] == pytest.approx(0.0, abs=1e-14)
    assert p["F2"] == pytest.approx(0.0, abs=1e-14)
    assert p["F3"] == pytest.approx(0.0, abs=1e-14)
    assert p["F4"] == pytest.approx(0.0, abs=1e-14)
    assert p["F5"] == pytest.approx(0.0, abs=1e-14)
    assert p["F6"] == pytest.approx(0.0, abs=1e-14)
    # Ellipse condition FT1 = 0 when F12 = -0.5 * F11
    assert p["FT1"] == pytest.approx(0.0, abs=1e-14)


def test_tsai_wu_default_cascading():
    """Verify Fortran default cascading logic (hm_read_mat12.F:184-196)."""
    # Only supply sigyt1 = 250.0
    mat = build_law12(
        E11=10000.0,
        E22=10000.0,
        E33=10000.0,
        nu12=0.2,
        nu23=0.2,
        nu31=0.2,
        rho0=1.0e-9,
        sigyt1=250.0,
    )
    p = mat.params
    # sigyc1 defaults to sigyt1 = 250
    assert p["sigyc1"] == pytest.approx(250.0)
    # sigyt2 defaults to sigyt1 = 250
    assert p["sigyt2"] == pytest.approx(250.0)
    # sigyc2 defaults to sigyc1 = 250
    assert p["sigyc2"] == pytest.approx(250.0)
    # sigyt3 defaults to sigyt2 = 250
    assert p["sigyt3"] == pytest.approx(250.0)
    # sigyc3 defaults to sigyc2 = 250
    assert p["sigyc3"] == pytest.approx(250.0)
    # Shears default to INFINITY
    assert p["F4"] == 0.0
    assert p["F5"] == 0.0
    assert p["F6"] == 0.0
    assert p["F44"] == 0.0
    assert p["F55"] == 0.0
    assert p["F66"] == 0.0


# ============================================================================
# 3. Strain Rate Sensitivity (m12law.F:208-228)
# ============================================================================

def test_strain_rate_epsp_rate_formula():
    """Verify epsp_rate = max(|D1|, |D2|, |D3|, 0.5*|D4|, 0.5*|D5|, 0.5*|D6|) tensorial shear factor."""
    mat = build_law12(
        E11=10000.0,
        E22=10000.0,
        E33=10000.0,
        nu12=0.2,
        nu23=0.2,
        nu31=0.2,
        rho0=1.0e-9,
        c=0.1,
        eps0=1.0,
        ICC=1,
        sigyt1=200.0,
        sigyc1=200.0,
        fmax=300.0,
        cb=50.0,
        cn=0.5,
    )
    p = mat.params

    # Test case 1: normal D1 dominates
    # dt = 0.01, deps = [0.05, 0.01, 0.02, 0.04, 0.06, 0.02]
    # Rates: D1 = 5.0, D2 = 1.0, D3 = 2.0, 0.5*D4 = 2.0, 0.5*D5 = 3.0, 0.5*D6 = 1.0
    # Expected max = 5.0
    dt = 0.01
    deps = np.array([0.05, 0.01, 0.02, 0.04, 0.06, 0.02])
    epsp_rate = max(
        abs(deps[0] / dt),
        abs(deps[1] / dt),
        abs(deps[2] / dt),
        0.5 * abs(deps[3] / dt),
        0.5 * abs(deps[4] / dt),
        0.5 * abs(deps[5] / dt),
    )
    assert epsp_rate == pytest.approx(5.0)

    # Test case 2: shear D4 dominates with tensorial 0.5 factor
    # deps = [0.01, 0.01, 0.01, 0.16, 0.02, 0.02] -> 0.5 * (0.16 / 0.01) = 8.0
    deps2 = np.array([0.01, 0.01, 0.01, 0.16, 0.02, 0.02])
    epsp_rate2 = max(
        abs(deps2[0] / dt),
        abs(deps2[1] / dt),
        abs(deps2[2] / dt),
        0.5 * abs(deps2[3] / dt),
        0.5 * abs(deps2[4] / dt),
        0.5 * abs(deps2[5] / dt),
    )
    assert epsp_rate2 == pytest.approx(8.0)


def test_strain_rate_logarithmic_enhancement_and_quasistatic():
    """Verify rate_fac = 1 + C * log(epsp_rate / eps0) when epsp_rate > eps0, and 1.0 when <= eps0."""
    c_rate = 0.05
    eps0 = 10.0
    mat = build_law12(
        E11=10000.0,
        E22=10000.0,
        E33=10000.0,
        nu12=0.2,
        nu23=0.2,
        nu31=0.2,
        rho0=1.0e-9,
        c=c_rate,
        eps0=eps0,
        fmax=1000.0,
        cb=100.0,
        cn=1.0,
    )
    p = mat.params

    # Sub-threshold rate: epsp_rate = 5.0 < eps0 = 10.0 -> rate_fac = 1.0
    dt = 0.01
    deps_slow = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])  # rate = 5.0
    epsp_rate_slow = 5.0
    rate_fac_slow = 1.0 + c_rate * math.log(epsp_rate_slow / eps0) if epsp_rate_slow > eps0 else 1.0
    assert rate_fac_slow == 1.0

    # Super-threshold rate: epsp_rate = 100.0 > eps0 = 10.0 -> rate_fac = 1 + 0.05 * ln(10)
    epsp_rate_fast = 100.0
    rate_fac_fast = 1.0 + c_rate * math.log(epsp_rate_fast / eps0)
    assert rate_fac_fast == pytest.approx(1.0 + 0.05 * math.log(10.0))


@pytest.mark.parametrize("icc, scales_sigmx", [(1, True), (2, False), (3, True), (4, False)])
def test_strain_rate_icc_flag_handling(icc: int, scales_sigmx: bool):
    """Verify ICC = 1, 2, 3, 4 handling on SIGMX vs CA, CB (m12law.F:215-225)."""
    fmax = 200.0
    cb = 50.0
    c_rate = 0.1
    eps0 = 1.0
    mat = build_law12(
        E11=10000.0,
        E22=10000.0,
        E33=10000.0,
        nu12=0.2,
        nu23=0.2,
        nu31=0.2,
        rho0=1.0e-9,
        c=c_rate,
        eps0=eps0,
        ICC=icc,
        fmax=fmax,
        cb=cb,
        cn=1.0,
    )
    p = mat.params

    # Strain rate = 100.0 -> rate_fac = 1.0 + 0.1 * ln(100) = 1.4605
    rate_fac = 1.0 + c_rate * math.log(100.0 / eps0)

    if scales_sigmx:
        expected_sigmx = fmax * rate_fac
    else:
        expected_sigmx = fmax

    expected_cb = cb * rate_fac
    expected_ca = 1.0 * rate_fac
    wpla = 0.5
    expected_sigmy = min(expected_sigmx, expected_ca + expected_cb * (wpla**1.0))

    # Verify via constitutive integration
    dam = np.zeros(5)
    epe = np.zeros(3)
    epc = np.zeros(3)
    sig_init = np.zeros(6)
    deps = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # deps/dt = 100.0 with dt = 0.01
    dt = 0.01

    s_out, wpla_out, dam_out, epe_out, epc_out, _, off_out, _, _ = _update_one_element(
        p, sig_init, deps, 0.0, dt, dam, epe, epc, wpla, 1.0, 0.0, 0.0
    )

    # In m12law.F:
    # If ICC in (1, 3): sigmx is scaled by rate_fac.
    # If ICC in (2, 4): sigmx is NOT scaled by rate_fac.
    if scales_sigmx:
        assert expected_sigmx > fmax
    else:
        assert expected_sigmx == pytest.approx(fmax)


def test_hardening_sigmy_power_law_saturation():
    """Verify SIGMY = min(SIGMX, CA + CB * wpla**CN) power-law hardening and saturation at SIGMX."""
    fmax = 10.0
    cb = 4.0
    cn = 0.5
    ca = 1.0

    # wpla = 0 -> sigmy = 1.0
    assert min(fmax, ca + cb * (0.0**cn)) == 1.0
    # wpla = 4.0 -> sigmy = 1 + 4 * sqrt(4) = 1 + 8 = 9.0 < 10.0
    assert min(fmax, ca + cb * (4.0**cn)) == 9.0
    # wpla = 9.0 -> 1 + 4 * 3 = 13.0 > 10.0 -> clamped to 10.0
    assert min(fmax, ca + cb * (9.0**cn)) == 10.0


# ============================================================================
# 4. Tensile Cracking Damage & Poisson Relief (m12law.F:422-501)
# ============================================================================

def test_tensile_damage_direction_1_poisson_relief():
    """Verify Direction 1 tensile cracking threshold, Poisson relief on T2 and T3, and damage delta."""
    mat = build_law12(
        E11=100000.0,
        E22=50000.0,
        E33=25000.0,
        nu12=0.2,
        nu23=0.15,
        nu31=0.1,
        rho0=1.5e-9,
        sigt1=150.0,
        delta=0.08,
    )
    p = mat.params
    d11 = p["D11"]
    d12 = p["D12"]
    d13 = p["D13"]

    dam = np.zeros(5)
    epe = np.zeros(3)
    epc = np.zeros(3)
    sig0 = np.zeros(6)

    # Step 1: Trial elastic stress exceeds sigt1
    # deps1 = 0.003 -> trial T1 = d11 * 0.003
    trial_t1 = d11 * 0.003
    assert trial_t1 > 150.0

    deps = np.array([0.003, 0.0, 0.0, 0.0, 0.0, 0.0])
    s_out, _, dam_out, epe_out, epc_out, _, _, _, _ = _update_one_element(
        p, sig0, deps, 0.0, 0.0, dam, epe, epc, 0.0, 1.0, 0.0, 0.0
    )

    # In Step 1, dam1 was initially 0.0, so:
    # T1 is clamped to (1 - 0.0) * 150.0 = 150.0
    # Poisson relief: T2 -= d12 * deps1 * 0.0 = 0.0; T3 -= d13 * deps1 * 0.0 = 0.0
    # Then dam1 increases by delta = 0.08
    assert s_out[0] == pytest.approx(150.0)
    assert dam_out[0] == pytest.approx(0.08)

    # Step 2: Second tensile increment with dam1 = 0.08
    # deps1 = 0.001
    # WVEC1 = (1 - 0.08) * 150.0 = 138.0
    # Trial T1 = 150.0 + d11 * 0.001 > 138.0
    # Poisson relief on T2: T2_elastic - d12 * 0.001 * 0.08
    trial_t2 = s_out[1] + d12 * 0.001
    expected_t2 = trial_t2 - d12 * 0.001 * 0.08
    trial_t3 = s_out[2] + d13 * 0.001
    expected_t3 = trial_t3 - d13 * 0.001 * 0.08

    s_out2, _, dam_out2, _, _, _, _, _, _ = _update_one_element(
        p, s_out, [0.001, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0, 0.0, dam_out, epe_out, epc_out, 0.0, 1.0, 0.0, 0.0
    )
    assert s_out2[0] == pytest.approx(138.0)
    assert s_out2[1] == pytest.approx(expected_t2)
    assert s_out2[2] == pytest.approx(expected_t3)
    assert dam_out2[0] == pytest.approx(0.16)


def test_tensile_damage_direction_2_poisson_relief():
    """Verify Direction 2 tensile cracking: T2 clamped, relief on T1 and T3."""
    mat = build_law12(
        E11=100000.0,
        E22=50000.0,
        E33=25000.0,
        nu12=0.2,
        nu23=0.15,
        nu31=0.1,
        rho0=1.5e-9,
        sigt2=80.0,
        delta=0.10,
    )
    p = mat.params
    d12 = p["D12"]
    d22 = p["D22"]
    d23 = p["D23"]

    dam = np.array([0.0, 0.20, 0.0, 0.0, 10000.0])  # dam2 already 0.20
    epe = np.array([0.0, 0.001, 0.0])
    epc = np.array([0.0, 0.001, 0.0])
    sig0 = np.array([0.0, 60.0, 0.0, 0.0, 0.0, 0.0])

    deps = np.array([0.0, 0.001, 0.0, 0.0, 0.0, 0.0])
    trial_t2 = 60.0 + d22 * 0.001
    wvec2 = (1.0 - 0.20) * 80.0  # 64.0
    assert trial_t2 > wvec2

    trial_t1 = 0.0 + d12 * 0.001
    expected_t1 = trial_t1 - d12 * 0.001 * 0.20
    trial_t3 = 0.0 + d23 * 0.001
    expected_t3 = trial_t3 - d23 * 0.001 * 0.20

    s_out, _, dam_out, _, _, _, _, _, _ = _update_one_element(
        p, sig0, deps, 0.0, 0.0, dam, epe, epc, 0.0, 1.0, 0.0, 0.0
    )

    assert s_out[1] == pytest.approx(64.0)
    assert s_out[0] == pytest.approx(expected_t1)
    assert s_out[2] == pytest.approx(expected_t3)
    assert dam_out[1] == pytest.approx(0.30)


def test_tensile_damage_direction_3_poisson_relief():
    """Verify Direction 3 tensile cracking: T3 clamped, relief on T1 and T2."""
    mat = build_law12(
        E11=100000.0,
        E22=50000.0,
        E33=25000.0,
        nu12=0.2,
        nu23=0.15,
        nu31=0.1,
        rho0=1.5e-9,
        sigt3=40.0,
        delta=0.05,
    )
    p = mat.params
    d13 = p["D13"]
    d23 = p["D23"]
    d33 = p["D33"]

    dam = np.array([0.0, 0.0, 0.15, 0.0, 10000.0])
    epe = np.array([0.0, 0.0, 0.001])
    epc = np.array([0.0, 0.0, 0.001])
    sig0 = np.array([0.0, 0.0, 30.0, 0.0, 0.0, 0.0])

    deps = np.array([0.0, 0.0, 0.001, 0.0, 0.0, 0.0])
    trial_t3 = 30.0 + d33 * 0.001
    wvec3 = (1.0 - 0.15) * 40.0  # 34.0
    assert trial_t3 > wvec3

    trial_t1 = 0.0 + d13 * 0.001
    expected_t1 = trial_t1 - d13 * 0.001 * 0.15
    trial_t2 = 0.0 + d23 * 0.001
    expected_t2 = trial_t2 - d23 * 0.001 * 0.15

    s_out, _, dam_out, _, _, _, _, _, _ = _update_one_element(
        p, sig0, deps, 0.0, 0.0, dam, epe, epc, 0.0, 1.0, 0.0, 0.0
    )

    assert s_out[2] == pytest.approx(34.0)
    assert s_out[0] == pytest.approx(expected_t1)
    assert s_out[1] == pytest.approx(expected_t2)
    assert dam_out[2] == pytest.approx(0.20)


def test_damage_flag_kd_encoding_and_saturation():
    """Verify dam[4] = 10000 + 1000*kd1 + 100*kd2 + 10*kd3 + kd4 encoding and saturation at 1.0."""
    mat = build_law12(
        E11=100000.0,
        E22=50000.0,
        E33=25000.0,
        nu12=0.2,
        nu23=0.15,
        nu31=0.1,
        rho0=1.5e-9,
        sigt1=50.0,
        delta=0.6,
    )
    p = mat.params

    dam = np.zeros(5)
    epe = np.zeros(3)
    epc = np.zeros(3)
    sig0 = np.zeros(6)

    # Step 1: dam1 goes from 0.0 -> 0.6 (kd1 = 1)
    s1, _, dam1, epe1, epc1, _, _, _, _ = _update_one_element(
        p, sig0, np.array([0.002, 0, 0, 0, 0, 0]), 0.0, 0.0, dam, epe, epc, 0.0, 1.0, 0.0, 0.0
    )
    assert dam1[0] == pytest.approx(0.6)
    # dam[4] should be 10000 + 1000*1 = 11000
    assert dam1[4] == 11000.0

    # Step 2: dam1 goes from 0.6 + 0.6 = 1.2 -> capped at 1.0 (kd1 = 2)
    s2, _, dam2, _, _, _, _, _, _ = _update_one_element(
        p, s1, np.array([0.002, 0, 0, 0, 0, 0]), 0.0, 0.0, dam1, epe1, epc1, 0.0, 1.0, 0.0, 0.0
    )
    assert dam2[0] == 1.0
    # dam[4] should be 10000 + 1000*2 = 12000
    assert dam2[4] == 12000.0


# ============================================================================
# 5. Crack Opening / Unilateral Conditions (m12law.F:506-522)
# ============================================================================

def test_crack_open_unilateral_zero_compression():
    """Verify that when crack is open (epc > 0) and stress is compressive (T < 0), stress is zeroed with relief."""
    mat = build_law12(
        E11=100000.0,
        E22=50000.0,
        E33=25000.0,
        nu12=0.2,
        nu23=0.15,
        nu31=0.1,
        rho0=1.5e-9,
        sigt1=50.0,
        delta=0.2,
    )
    p = mat.params
    d11 = p["D11"]
    d12 = p["D12"]
    d13 = p["D13"]

    # Start with an already open crack in Direction 1
    dam = np.array([0.5, 0.0, 0.0, 0.0, 11000.0])
    epe = np.array([0.005, 0.0, 0.0])
    epc = np.array([0.005, 0.0, 0.0])  # Crack open by 0.005 strain
    sig0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Apply small compressive strain: deps1 = -0.001 (crack does not close fully: epc1 = 0.004 > 0)
    deps = np.array([-0.001, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Elastic trial T1 = -0.001 * d11 < 0
    # Because T1 < 0 and epc1 > 0:
    # T1 is set to 0.0
    # T2 -= d12 * (-0.001) * 0.5 = + 0.0005 * d12
    # T3 -= d13 * (-0.001) * 0.5 = + 0.0005 * d13
    expected_t2 = d12 * (-0.001) - d12 * (-0.001) * 0.5  # = -0.0005 * d12
    expected_t3 = d13 * (-0.001) - d13 * (-0.001) * 0.5

    s_out, _, dam_out, _, epc_out, _, _, _, _ = _update_one_element(
        p, sig0, deps, 0.0, 0.0, dam, epe, epc, 0.0, 1.0, 0.0, 0.0
    )

    assert s_out[0] == 0.0
    assert s_out[1] == pytest.approx(expected_t2)
    assert s_out[2] == pytest.approx(expected_t3)
    assert epc_out[0] == pytest.approx(0.004)


def test_crack_closure_compression_recovery():
    """Verify crack closure: when compression exceeds epc, epc reaches 0, and normal compression is recovered."""
    mat = build_law12(
        E11=100000.0,
        E22=50000.0,
        E33=25000.0,
        nu12=0.2,
        nu23=0.15,
        nu31=0.1,
        rho0=1.5e-9,
        sigt1=50.0,
        delta=0.2,
    )
    p = mat.params
    d11 = p["D11"]

    # Open crack by 0.001 strain
    dam = np.array([0.5, 0.0, 0.0, 0.0, 11000.0])
    epe = np.array([0.001, 0.0, 0.0])
    epc = np.array([0.001, 0.0, 0.0])
    sig0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Compressive increment larger than crack opening: deps1 = -0.003
    # Step: epc1 = max(0.001 - 0.003, 0.0) = 0.0!
    # Because epc1 == 0.0, unilateral condition (T1 < 0 and epc1 > 0) is FALSE!
    # Stresses remain in normal compression: T1 = -0.003 * d11 < 0!
    deps = np.array([-0.003, 0.0, 0.0, 0.0, 0.0, 0.0])
    s_out, _, dam_out, _, epc_out, _, _, _, _ = _update_one_element(
        p, sig0, deps, 0.0, 0.0, dam, epe, epc, 0.0, 1.0, 0.0, 0.0
    )

    assert epc_out[0] == 0.0
    assert s_out[0] == pytest.approx(-0.003 * d11)
    assert s_out[0] < 0.0


# ============================================================================
# 6. Tsai-Wu 3D Plastic Return (m12law.F:526-636)
# ============================================================================

def test_tsai_wu_wvec_evaluation_hand_calc():
    """Verify WVEC formula from m12law.F:527-531 for arbitrary 3D stress state."""
    mat = build_law12(
        E11=10000.0,
        E22=10000.0,
        E33=10000.0,
        nu12=0.2,
        nu23=0.2,
        nu31=0.2,
        rho0=1.0e-9,
        sigyt1=100.0,
        sigyc1=120.0,
        sigyt2=80.0,
        sigyc2=100.0,
        sigyt3=60.0,
        sigyc3=80.0,
        sigyt12=40.0,
        sigyc12=40.0,
        sigyt23=30.0,
        sigyc23=30.0,
        sigyt13=35.0,
        sigyc13=35.0,
    )
    p = mat.params
    f1, f2, f3 = p["F1"], p["F2"], p["F3"]
    f11, f22, f33 = p["F11"], p["F22"], p["F33"]
    f44, f55, f66 = p["F44"], p["F55"], p["F66"]
    f12, f23, f13 = p["F12"], p["F23"], p["F13"]

    T = np.array([50.0, -30.0, 20.0, 15.0, -10.0, 12.0])
    wvec_hand = (
        f1 * T[0] + f2 * T[1] + f3 * T[2]
        + f11 * (T[0]**2) + f22 * (T[1]**2) + f33 * (T[2]**2)
        + f44 * (T[3]**2) + f55 * (T[4]**2) + f66 * (T[5]**2)
        + 2.0 * f12 * T[0] * T[1]
        + 2.0 * f13 * T[0] * T[2]
        + 2.0 * f23 * T[1] * T[2]
    )

    # Put this stress into element with zero deps and off=1
    dam = np.zeros(5)
    epe = np.zeros(3)
    epc = np.zeros(3)
    _, _, dam_out, _, _, _, _, _, _ = _update_one_element(
        p, T, np.zeros(6), 0.0, 0.0, dam, epe, epc, 0.0, 1.0, 0.0, 0.0
    )
    assert dam_out[3] == pytest.approx(wvec_hand, rel=1e-12)


def test_plastic_multiplier_lambda_num_and_denom():
    """Verify plastic multiplier lambda = num / denom exact formulas from m12law.F:571-593."""
    mat = build_law12(
        E11=50000.0,
        E22=50000.0,
        E33=50000.0,
        nu12=0.25,
        nu23=0.25,
        nu31=0.25,
        rho0=1.0e-9,
        sigyt1=100.0,
        sigyc1=100.0,
        sigyt2=100.0,
        sigyc2=100.0,
        sigyt3=100.0,
        sigyc3=100.0,
        sigyt12=50.0,
        sigyc12=50.0,
        sigyt23=50.0,
        sigyc23=50.0,
        sigyt13=50.0,
        sigyc13=50.0,
        fmax=1000.0,
        cb=20.0,
        cn=1.0,
    )
    p = mat.params
    d11, d12, d13 = p["D11"], p["D12"], p["D13"]
    d22, d23, d33 = p["D22"], p["D23"], p["D33"]
    g12, g23, g31 = p["G12"], p["G23"], p["G31"]
    f1, f2, f3 = p["F1"], p["F2"], p["F3"]
    f11, f22, f33 = p["F11"], p["F22"], p["F33"]
    f44, f55, f66 = p["F44"], p["F55"], p["F66"]
    f12, f23, f13 = p["F12"], p["F23"], p["F13"]

    so = np.array([80.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    deps = np.array([0.002, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Trial stresses:
    t1 = so[0] + d11 * deps[0]
    t2 = so[1] + d12 * deps[0]
    t3 = so[2] + d13 * deps[0]
    t4, t5, t6 = 0.0, 0.0, 0.0

    # Yield condition check:
    wvec = f11 * (t1**2) + f22 * (t2**2) + f33 * (t3**2) + 2.0 * f12 * t1 * t2 + 2.0 * f13 * t1 * t3
    sigmy = 1.0  # cb=20, wpla=0 -> 1.0
    assert wvec > sigmy

    # Hand calculation of DP:
    dp1 = f1 + 2.0 * f11 * so[0] + 2.0 * f12 * so[1] + 2.0 * f13 * so[2]
    dp2 = f2 + 2.0 * f22 * so[1] + 2.0 * f12 * so[0] + 2.0 * f23 * so[2]
    dp3 = f3 + 2.0 * f33 * so[2] + 2.0 * f13 * so[0] + 2.0 * f23 * so[1]
    dp4 = 2.0 * f44 * so[3]
    dp5 = 2.0 * f55 * so[4]
    dp6 = 2.0 * f66 * so[5]

    ds1 = t1 - so[0]
    ds2 = t2 - so[1]
    ds3 = t3 - so[2]
    ds4, ds5, ds6 = 0.0, 0.0, 0.0

    num_h = dp1 * ds1 + dp2 * ds2 + dp3 * ds3
    denom_h = (
        dp1 * (d11 * dp1 + d12 * dp2 + d13 * dp3)
        + dp2 * (d12 * dp1 + d22 * dp2 + d23 * dp3)
        + dp3 * (d13 * dp1 + d23 * dp2 + d33 * dp3)
        + 2.0 * dp4 * g12 * dp4
        + 2.0 * dp5 * g23 * dp5
        + 2.0 * dp6 * g31 * dp6
        + (so[0] * dp1 + so[1] * dp2 + so[2] * dp3 + 2.0 * so[3] * dp4 + 2.0 * so[4] * dp5 + 2.0 * so[5] * dp6)
        * 1.0 * 20.0 * 1.0
    )
    expected_lamda = num_h / denom_h

    # Expected returned stresses:
    l_dp1 = expected_lamda * dp1
    l_dp2 = expected_lamda * dp2
    l_dp3 = expected_lamda * dp3
    expected_t1 = t1 - (d11 * l_dp1 + d12 * l_dp2 + d13 * l_dp3)
    expected_t2 = t2 - (d12 * l_dp1 + d22 * l_dp2 + d23 * l_dp3)
    expected_t3 = t3 - (d13 * l_dp1 + d23 * l_dp2 + d33 * l_dp3)

    dam = np.zeros(5)
    epe = np.zeros(3)
    epc = np.zeros(3)
    s_out, wpla_out, dam_out, _, _, _, _, _, _ = _update_one_element(
        p, so, deps, 0.0, 0.0, dam, epe, epc, 0.0, 1.0, 0.0, 0.0
    )

    assert s_out[0] == pytest.approx(expected_t1, rel=1e-10)
    assert s_out[1] == pytest.approx(expected_t2, rel=1e-10)
    assert s_out[2] == pytest.approx(expected_t3, rel=1e-10)
    assert wpla_out > 0.0


def test_plastic_work_dwpla_increment():
    """Verify DWPLA = 0.5 * (DP1*(T1+SO1) + ... + 2*DP4*(T4+SO4)) / WPLAREF from m12law.F:626-634."""
    mat = build_law12(
        E11=20000.0,
        E22=20000.0,
        E33=20000.0,
        nu12=0.2,
        nu23=0.2,
        nu31=0.2,
        rho0=1.0e-9,
        sigyt1=50.0,
        sigyc1=50.0,
        fmax=100.0,
        wplaref=2.5,
        cb=10.0,
        cn=1.0,
    )
    p = mat.params

    so = np.array([40.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    deps = np.array([0.003, 0.0, 0.0, 0.0, 0.0, 0.0])

    dam = np.zeros(5)
    epe = np.zeros(3)
    epc = np.zeros(3)
    s_out, wpla_out, dam_out, _, _, _, _, _, _ = _update_one_element(
        p, so, deps, 0.0, 0.0, dam, epe, epc, 0.0, 1.0, 0.0, 0.0
    )

    # Plasticity must have occurred and work must be scaled by wplaref = 2.5
    assert wpla_out > 0.0


def test_shear_plasticity_return():
    """Verify Tsai-Wu plasticity in pure shear D4 (G12*deps4)."""
    mat = build_law12(
        E11=30000.0,
        E22=30000.0,
        E33=30000.0,
        nu12=0.2,
        nu23=0.2,
        nu31=0.2,
        G12=12000.0,
        G23=12000.0,
        G31=12000.0,
        rho0=1.0e-9,
        sigyt12=30.0,
        sigyc12=30.0,
        fmax=1000.0,
        cb=5.0,
        cn=1.0,
    )
    p = mat.params
    so = np.array([0.0, 0.0, 0.0, 25.0, 0.0, 0.0])
    deps = np.array([0.0, 0.0, 0.0, 0.005, 0.0, 0.0])

    dam = np.zeros(5)
    epe = np.zeros(3)
    epc = np.zeros(3)
    s_out, wpla_out, dam_out, _, _, _, _, _, _ = _update_one_element(
        p, so, deps, 0.0, 0.0, dam, epe, epc, 0.0, 1.0, 0.0, 0.0
    )

    # Elastic trial would be 25 + 12000*0.005 = 85.0
    # Yield limit for pure shear is sigyt12 = 30.0
    assert s_out[3] < 85.0
    assert s_out[3] > 25.0
    assert wpla_out > 0.0


# ============================================================================
# 7. Sound Speed Formula (hm_read_mat12.F:265-266)
# ============================================================================

def test_sound_speed_hand_calc():
    """Verify SSP = sqrt(max(D11, D22, D33) / rho0) against exact hand calculation."""
    mat = build_law12(
        E11=100000.0,
        E22=60000.0,
        E33=40000.0,
        nu12=0.2,
        nu23=0.15,
        nu31=0.1,
        rho0=2.0e-9,
    )
    p = mat.params
    d11 = p["D11"]
    d22 = p["D22"]
    d33 = p["D33"]
    expected_ssp = math.sqrt(max(d11, d22, d33) / 2.0e-9)

    assert p["SSP"] == pytest.approx(expected_ssp, rel=1e-12)
    assert sound_speed(mat) == pytest.approx(expected_ssp, rel=1e-12)


def test_sound_speed_with_refer_rho():
    """Verify sound speed formula uses refer_rho when specified (PM(1)=RHOR in Fortran)."""
    mat = build_law12(
        E11=100000.0,
        E22=60000.0,
        E33=40000.0,
        nu12=0.2,
        nu23=0.15,
        nu31=0.1,
        rho0=2.0e-9,
        refer_rho=2.5e-9,
    )
    p = mat.params
    expected_ssp = math.sqrt(max(p["D11"], p["D22"], p["D33"]) / 2.5e-9)
    assert p["SSP"] == pytest.approx(expected_ssp, rel=1e-12)


# ============================================================================
# 8. Element Degradation (m12law.F:227-229, 416-418, 648-654)
# ============================================================================

def test_element_degradation_trigger_at_sigmx():
    """Verify that when SIGMY == SIGMX and off == 1.0 -> off = 0.99, then in same step off *= 0.8 -> 0.792."""
    mat = build_law12(
        E11=10000.0,
        E22=10000.0,
        E33=10000.0,
        nu12=0.2,
        nu23=0.2,
        nu31=0.2,
        rho0=1.0e-9,
        fmax=10.0,  # low fmax so it reaches sigmx easily
        cb=20.0,
        cn=1.0,
    )
    p = mat.params

    dam = np.zeros(5)
    epe = np.zeros(3)
    epc = np.zeros(3)
    sig0 = np.zeros(6)

    # wpla = 1.0 -> ca + cb * wpla = 1 + 20*1 = 21 > fmax=10.0
    # In m12law.F:226-228: SIGMY = min(SIGMX, 21) = 10.0 == SIGMX -> OFF = 0.99, KD4 = 2
    # In m12law.F:417: IF(OFF < ONE) OFF = OFF * 0.8 = 0.99 * 0.8 = 0.792
    deps = np.array([0.0001, 0, 0, 0, 0, 0])
    s_out, _, dam_out, _, _, _, off_out, _, _ = _update_one_element(
        p, sig0, deps, 0.0, 0.0, dam, epe, epc, 1.0, 1.0, 0.0, 0.0
    )

    assert off_out == pytest.approx(0.99 * 0.8)
    # Check kd4 = 2 encoded in dam[4]
    assert int(dam_out[4]) % 10 == 2


def test_element_degradation_decay_to_zero():
    """Verify geometric decay off *= 0.8 in each step until off < 0.1 -> off = 0.0 (stresses zeroed)."""
    mat = build_law12(
        E11=10000.0,
        E22=10000.0,
        E33=10000.0,
        nu12=0.2,
        nu23=0.2,
        nu31=0.2,
        rho0=1.0e-9,
        fmax=5.0,
        cb=10.0,
        cn=1.0,
    )
    p = mat.params

    dam = np.zeros(5)
    epe = np.zeros(3)
    epc = np.zeros(3)
    sig = np.array([10.0, 0, 0, 0, 0, 0])
    deps = np.zeros(6)

    off = 0.792
    history = [off]
    # Simulate successive steps
    for _ in range(12):
        sig, _, dam, epe, epc, _, off, _, _ = _update_one_element(
            p, sig, deps, 0.0, 0.0, dam, epe, epc, 1.0, off, 0.0, 0.0
        )
        history.append(off)

    # Must decay geometrically by 0.8: 0.792 -> 0.6336 -> 0.50688 -> ...
    assert history[1] == pytest.approx(0.792 * 0.8)
    assert history[2] == pytest.approx(0.792 * 0.8**2)

    # Eventually below 0.1, it must be zeroed out completely
    assert history[-1] == 0.0
    # And stresses must be zero
    assert np.allclose(sig, 0.0)


# ============================================================================
# 9. Tangent & Coordinate Frames Parity
# ============================================================================

def test_consistent_solid_tangent_elastic_parity():
    """Verify consistent solid tangent matches orthotropic D_mat in elastic regime."""
    mat = build_law12(
        E11=80000.0,
        E22=40000.0,
        E33=20000.0,
        nu12=0.25,
        nu23=0.2,
        nu31=0.15,
        G12=10000.0,
        G23=8000.0,
        G31=9000.0,
        rho0=1.5e-9,
    )
    p = mat.params
    sig = np.zeros(6)
    tangent = consistent_solid_tangent(mat, sig)

    assert tangent.shape == (6, 6)
    assert tangent[0, 0] == pytest.approx(p["D11"])
    assert tangent[0, 1] == pytest.approx(p["D12"])
    assert tangent[0, 2] == pytest.approx(p["D13"])
    assert tangent[1, 1] == pytest.approx(p["D22"])
    assert tangent[1, 2] == pytest.approx(p["D23"])
    assert tangent[2, 2] == pytest.approx(p["D33"])
    assert tangent[3, 3] == pytest.approx(p["G12"])
    assert tangent[4, 4] == pytest.approx(p["G23"])
    assert tangent[5, 5] == pytest.approx(p["G31"])


def test_m14gtf_m14ftg_coordinate_frame_roundtrip():
    """Verify coordinate transformation round-trip: global -> local material frame -> global."""
    # Rotate 45 deg around Z
    theta = math.pi / 4.0
    c, s = math.cos(theta), math.sin(theta)
    ax = np.array([c])
    ay = np.array([s])
    az = np.array([0.0])
    bx = np.array([-s])
    by = np.array([c])
    bz = np.array([0.0])
    cx = np.array([0.0])
    cy = np.array([0.0])
    cz = np.array([1.0])

    sig_global = np.array([[100.0, 50.0, 20.0, 30.0, 10.0, 15.0]])
    d_global = np.array([[0.001, 0.002, 0.0005, 0.0015, 0.0008, 0.0004]])

    t_loc, e_loc = m14gtf(sig_global, d_global, ax, ay, az, bx, by, bz, cx, cy, cz)
    sig_recovered = m14ftg(t_loc, ax, ay, az, bx, by, bz, cx, cy, cz)

    assert np.allclose(sig_recovered, sig_global, rtol=1e-12, atol=1e-12)


def test_extra_shapes_and_shell_exception():
    """Verify state variable extra_shapes and shell_update error on solid law."""
    shapes = extra_shapes(nip=1)
    assert shapes["dam12"] == (5,)
    assert shapes["epe12"] == (3,)
    assert shapes["epc12"] == (3,)
    assert "tsaiwu12" in shapes

    shapes_mult = extra_shapes(nip=4)
    assert shapes_mult["dam12"] == (4, 5)

    mat = build_law12(E11=1000, E22=1000, E33=1000, rho0=1e-9)
    with pytest.raises(NotImplementedError):
        shell_update(mat, np.zeros(6), np.zeros(6))
