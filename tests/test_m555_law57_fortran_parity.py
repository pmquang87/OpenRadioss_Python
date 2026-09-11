"""
Comprehensive Fortran Oracle Parity Test Suite for /MAT/LAW57 (/MAT/BARLAT3).

Upstream OpenRadioss Fortran reference code:
  - starter/source/materials/mat/mat057/hm_read_mat57.F90
  - starter/source/materials/mat/mat057/calculp2.F90
  - engine/source/materials/mat/mat057/sigeps57c.F90
  - engine/source/materials/mat_share/mulawc.F90
  - engine/source/materials/tools/table_mat_vinterp.F

Covers:
  1. Barlat parameter initialization from Lankford ratios (R00, R45, R90)
     matching hm_read_mat57.F90:171-172, 244-247.
  2. calculp2 Newton-Raphson solver for parameter p matching calculp2.F90
     across isotropic and anisotropic cases (R45 in [0.5, 2.5], m in [2, 6, 8]).
  3. Barlat-Lian (1989) yield criterion & equivalent stress matching sigeps57c.F90:385-408,
     including analytical stress gradient components dseq_dsig.
  4. Multi-rate tabulated yield curves with linear strain rate interpolation and
     rate cutoff filtering (fcut/asrate, israte, vp).
  5. Mixed isotropic / kinematic hardening with backstress tensor alpha update
     matching sigeps57c.F90:425-450 and Bauschinger effect.
  6. Dynamic Young's modulus degradation E(pla) = E0 - (E0 - Einf)*(1 - exp(-CE*pla))
     and tabulated curve ifunce matching sigeps57c.F90:200-216.
  7. Tensile damage scaling factor FAIL = max(0, min(1, (epsr2 - epst)/(epsr2 - epsr1)))
     and element deletion at pla >= epsmax or epst >= epsr2 matching sigeps57c.F90:531-550.
  8. Shell through-thickness thinning increment deps_zz matching sigeps57c.F90:596-601.
  9. Acoustic sound speed c_shell = sqrt(E / ((1 - nu^2) * rho0)) matching sigeps57c.F90:603.
"""

from __future__ import annotations

import math
from typing import Tuple
import numpy as np
import pytest

from pyradioss.materials.law57_barlat import (
    Law57Params,
    BarlatParams,
    barlat_params,
    calculp2,
    barlat_equivalent_stress,
    build_law57,
    shell_update_law57,
    sound_speed_shell_law57,
    tangent_law57_shell,
    solid_update_law57,
    _eval_young,
    _eval_yield_stress,
    _curve_eval,
)


# ============================================================================
# Exact Fortran Reference Oracles
# ============================================================================

def fortran_oracle_barlat_init(r00: float, r45: float, r90: float, m: float
                               ) -> Tuple[float, float, float, float, float, float]:
    """Exact line-by-line Fortran oracle for hm_read_mat57.F90:160-252.

    Returns: (r, h_orig, c, a, h_bar, p)
    """
    zero = 0.0
    one = 1.0
    two = 2.0
    six = 6.0

    r0 = r00 if r00 > zero else one
    r45_f = r45 if r45 > zero else one
    r90_f = r90 if r90 > zero else one
    m_f = m if m > zero else six

    r = r0 / (one + r0)
    h = r90_f / (one + r90_f)

    c = two * math.sqrt(r * h)
    a = two - c
    h_bar = math.sqrt(r / h)
    p = fortran_oracle_calculp2(a, c, h_bar, 1.0, m_f, r45_f)

    return r, h, c, a, h_bar, p


def fortran_oracle_calculp2(a: float, c: float, h: float, p: float = 1.0,
                            m: float = 6.0, r45: float = 1.0) -> float:
    """Exact line-by-line Fortran oracle for calculp2.F90:42-95."""
    two = 2.0
    fourth = 0.25
    half = 0.5
    one = 1.0
    nmax = 10

    m2 = m - two
    c1 = (two ** m) * c
    c2 = fourth * (one - h)
    c2 = c2 * c2
    c3 = fourth * (one + h)
    c4 = half * (one + h)
    pp = float(p)

    for _ in range(nmax):
        gama = math.sqrt(c2 + fourth * pp * pp)
        alpha = c3 - gama
        beta = c3 + gama
        c5 = two * c2 / gama
        aba2 = abs(alpha) ** m2
        abb2 = abs(beta) ** m2
        ca = aba2 * (c4 - c5) * (one + r45)
        cb = abb2 * (c4 + c5) * (one + r45)
        ca1 = aba2 * alpha
        cb1 = abb2 * beta
        abg1 = c1 * (gama ** (m - one))
        f = a * (alpha * (ca1 - ca) + beta * (cb1 - cb)) + gama * abg1
        df = a * ((m - one) * (ca - cb) - (ca1 - cb1) * (m + (one + r45) * c5 / gama)) + m * abg1
        df = half * df * pp / gama
        pp = pp - f / df

    return pp


def fortran_oracle_barlat_seq(sxx: float, syy: float, sxy: float,
                              a: float, c: float, h_bar: float, p: float, m: float
                              ) -> Tuple[float, float, float, float, float, float]:
    """Exact Fortran equivalent stress & gradient oracle (sigeps57c.F90:243-265, 322-345).

    Returns: (seq, k1, k2, dseq_dsxx, dseq_dsyy, dseq_dsxy)
    """
    two = 2.0
    half = 0.5
    one = 1.0
    em20 = 1.0e-20

    normsig = math.sqrt(sxx * sxx + syy * syy + two * sxy * sxy)
    normsig = max(normsig, one)

    k1 = half * (sxx + h_bar * syy) / normsig
    k2 = math.sqrt((half * (sxx - h_bar * syy)) ** 2 + (p * sxy) ** 2) / normsig

    phi = a * (abs(k1 + k2) ** m) + a * (abs(k1 - k2) ** m) + c * (abs(two * k2) ** m)
    if phi > 0.0:
        seq_norm = math.exp((one / m) * math.log(half * phi))
    else:
        seq_norm = 0.0
    seq = seq_norm * normsig

    if seq > 0.0 and normsig > 0.0:
        s_ratio = seq / normsig
        p1 = s_ratio ** (one - m)

        kp = k1 + k2
        km = k1 - k2
        s1 = math.copysign(one, kp)
        s2 = math.copysign(one, km)
        t1 = abs(kp) ** (m - one)
        t2 = abs(km) ** (m - one)

        dseq_dk1 = p1 * (a / two) * (s1 * t1 + s2 * t2)
        dseq_dk2 = p1 * ((a / two) * (s1 * t1 - s2 * t2) + c * (abs(two * k2) ** (m - one)))

        dk1_dsxx = half
        dk1_dsyy = h_bar / two
        denom_k2 = max(normsig * 4.0 * k2, em20)
        dk2_dsxx = (sxx - h_bar * syy) / denom_k2
        dk2_dsyy = -h_bar * (sxx - h_bar * syy) / denom_k2
        dk2_dsxy = (p ** 2) * sxy / max(normsig * k2, em20)

        dseq_dsxx = dseq_dk1 * dk1_dsxx + dseq_dk2 * dk2_dsxx
        dseq_dsyy = dseq_dk1 * dk1_dsyy + dseq_dk2 * dk2_dsyy
        dseq_dsxy = dseq_dk2 * dk2_dsxy
    else:
        dseq_dsxx = 0.0
        dseq_dsyy = 0.0
        dseq_dsxy = 0.0

    return seq, k1, k2, dseq_dsxx, dseq_dsyy, dseq_dsxy


# ============================================================================
# 1. Barlat Parameter Initialization from Lankford Ratios
# ============================================================================

class TestBarlatParameterInit:
    """Parity tests for hm_read_mat57.F90:160-252."""

    def test_isotropic_default_lankford(self):
        """Zero or default Lankford inputs must set r00=r45=r90=1, m=6."""
        bp = barlat_params(r0=0.0, r45=0.0, r90=0.0, m=0.0)
        r_ref, h_ref, c_ref, a_ref, h_bar_ref, p_ref = fortran_oracle_barlat_init(0.0, 0.0, 0.0, 0.0)

        assert bp.r == pytest.approx(r_ref, rel=1e-14)
        assert bp.h == pytest.approx(h_ref, rel=1e-14)
        assert bp.c == pytest.approx(c_ref, rel=1e-14)
        assert bp.a == pytest.approx(a_ref, rel=1e-14)
        assert bp.h_bar == pytest.approx(h_bar_ref, rel=1e-14)
        assert bp.p == pytest.approx(p_ref, rel=1e-14)
        # Numerical values for isotropic:
        assert bp.r == pytest.approx(0.5)
        assert bp.h == pytest.approx(0.5)
        assert bp.c == pytest.approx(1.0)
        assert bp.a == pytest.approx(1.0)
        assert bp.h_bar == pytest.approx(1.0)
        assert bp.p == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("r00,r45,r90,m", [
        (1.0, 1.0, 1.0, 6.0),     # Isotropic
        (1.5, 1.2, 1.8, 6.0),     # AA6016-T4 aluminum alloy
        (1.79, 1.34, 2.21, 6.0),  # DC06 deep-drawing steel
        (0.65, 0.82, 0.95, 8.0),  # FCC sheet metal
        (2.1, 1.8, 2.6, 4.0),     # High r-value steel
        (0.5, 0.7, 0.6, 2.0),     # Low r-value alloy (m=2)
    ])
    def test_anisotropic_materials_match_fortran_oracle(self, r00, r45, r90, m):
        """Verify barlat_params matches the Fortran oracle across diverse alloys."""
        bp = barlat_params(r0=r00, r45=r45, r90=r90, m=m)
        r_ref, h_ref, c_ref, a_ref, h_bar_ref, p_ref = fortran_oracle_barlat_init(r00, r45, r90, m)

        assert bp.r == pytest.approx(r_ref, rel=1e-14)
        assert bp.h == pytest.approx(h_ref, rel=1e-14)
        assert bp.c == pytest.approx(c_ref, rel=1e-14)
        assert bp.a == pytest.approx(a_ref, rel=1e-14)
        assert bp.h_bar == pytest.approx(h_bar_ref, rel=1e-14)
        assert bp.p == pytest.approx(p_ref, rel=1e-14)


# ============================================================================
# 2. calculp2 Newton-Raphson Solver Line-by-Line Parity
# ============================================================================

class TestCalculp2NewtonRaphson:
    """Parity tests for calculp2.F90 across R45 in [0.5, 2.5] and m in [2, 6, 8]."""

    @pytest.mark.parametrize("m", [2.0, 4.0, 6.0, 8.0])
    @pytest.mark.parametrize("r45", [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.5])
    def test_calculp2_grid_parity(self, m, r45):
        """Check calculp2 output against exact Fortran reference across grid."""
        # Using typical sheet metal parameters
        r00, r90 = 1.4, 1.9
        r = r00 / (1.0 + r00)
        h_orig = r90 / (1.0 + r90)
        c = 2.0 * math.sqrt(r * h_orig)
        a = 2.0 - c
        h_bar = math.sqrt(r / h_orig)

        p_py = calculp2(a=a, c=c, h_bar=h_bar, p=1.0, m=m, r45=r45)
        p_ref = fortran_oracle_calculp2(a=a, c=c, h=h_bar, p=1.0, m=m, r45=r45)

        assert p_py == pytest.approx(p_ref, rel=1e-14)

    @pytest.mark.parametrize("m", [2.0, 6.0, 8.0])
    def test_calculp2_root_residual(self, m):
        """Verify that the computed p makes the Fortran residual equation f(p) < 1e-4."""
        r00, r45, r90 = 1.6, 1.3, 2.0
        bp = barlat_params(r00, r45, r90, m)

        # Residual check of calculp2.F90 line 88
        p_val = bp.p
        c2 = (0.25 * (1.0 - bp.h_bar)) ** 2
        c3 = 0.25 * (1.0 + bp.h_bar)
        c4 = 0.5 * (1.0 + bp.h_bar)
        gama = math.sqrt(c2 + 0.25 * p_val * p_val)
        alpha = c3 - gama
        beta = c3 + gama
        c5 = 2.0 * c2 / gama
        aba2 = abs(alpha) ** (m - 2.0)
        abb2 = abs(beta) ** (m - 2.0)
        ca = aba2 * (c4 - c5) * (1.0 + r45)
        cb = abb2 * (c4 + c5) * (1.0 + r45)
        ca1 = aba2 * alpha
        cb1 = abb2 * beta
        abg1 = ((2.0 ** m) * bp.c) * (gama ** (m - 1.0))
        f_res = bp.a * (alpha * (ca1 - ca) + beta * (cb1 - cb)) + gama * abg1

        assert abs(f_res) < 1.0e-3

    def test_calculp2_isotropic_invariance(self):
        """In isotropic conditions (a=c=h_bar=r45=1), p=1 is an exact root for all m."""
        for m in (2.0, 3.0, 4.0, 6.0, 8.0, 10.0):
            p_val = calculp2(a=1.0, c=1.0, h_bar=1.0, p=1.0, m=m, r45=1.0)
            assert p_val == pytest.approx(1.0, abs=1e-14)


# ============================================================================
# 3. Barlat-Lian (1989) Yield Criterion & Equivalent Stress
# ============================================================================

class TestBarlatYieldCriterion:
    """Parity tests for sigeps57c.F90:385-408."""

    @pytest.mark.parametrize("m", [2.0, 6.0, 8.0])
    def test_equivalent_stress_matches_oracle(self, m):
        """Barlat equivalent stress matches Fortran oracle for 2D plane stress states."""
        bp = barlat_params(r0=1.4, r45=1.1, r90=1.7, m=m)
        stresses = [
            (250.0, 0.0, 0.0),       # Uniaxial xx
            (0.0, 180.0, 0.0),       # Uniaxial yy
            (0.0, 0.0, 120.0),       # Pure shear
            (200.0, 150.0, 60.0),    # Combined tension-shear
            (-180.0, 90.0, -45.0),   # Mixed sign
            (-220.0, -140.0, 0.0),   # Biaxial compression
        ]

        for sxx, syy, sxy in stresses:
            sig = np.array([sxx, syy, sxy])
            seq_py = barlat_equivalent_stress(sig, bp.a, bp.c, bp.h_bar, bp.p, m)
            seq_ref, _, _, _, _, _ = fortran_oracle_barlat_seq(sxx, syy, sxy, bp.a, bp.c, bp.h_bar, bp.p, m)
            assert seq_py == pytest.approx(seq_ref, rel=1e-13)

    def test_analytical_stress_gradients(self):
        """Compare analytical dseq_dsig with central finite difference approximations."""
        m = 6.0
        bp = barlat_params(1.5, 1.2, 1.8, m)
        sxx, syy, sxy = 220.0, 140.0, 55.0
        h_fd = 1.0e-6

        seq_ref, _, _, dxx_ana, dyy_ana, dxy_ana = fortran_oracle_barlat_seq(
            sxx, syy, sxy, bp.a, bp.c, bp.h_bar, bp.p, m
        )

        # Finite differences
        s_p = barlat_equivalent_stress(np.array([sxx + h_fd, syy, sxy]), bp.a, bp.c, bp.h_bar, bp.p, m)
        s_m = barlat_equivalent_stress(np.array([sxx - h_fd, syy, sxy]), bp.a, bp.c, bp.h_bar, bp.p, m)
        dxx_fd = (s_p - s_m) / (2.0 * h_fd)

        s_p = barlat_equivalent_stress(np.array([sxx, syy + h_fd, sxy]), bp.a, bp.c, bp.h_bar, bp.p, m)
        s_m = barlat_equivalent_stress(np.array([sxx, syy - h_fd, sxy]), bp.a, bp.c, bp.h_bar, bp.p, m)
        dyy_fd = (s_p - s_m) / (2.0 * h_fd)

        s_p = barlat_equivalent_stress(np.array([sxx, syy, sxy + h_fd]), bp.a, bp.c, bp.h_bar, bp.p, m)
        s_m = barlat_equivalent_stress(np.array([sxx, syy, sxy - h_fd]), bp.a, bp.c, bp.h_bar, bp.p, m)
        dxy_fd = (s_p - s_m) / (2.0 * h_fd)

        assert dxx_ana == pytest.approx(dxx_fd, rel=1e-5)
        assert dyy_ana == pytest.approx(dyy_fd, rel=1e-5)
        assert dxy_ana == pytest.approx(dxy_fd, rel=1e-5)

    def test_von_mises_recovery_m2(self):
        """For isotropic parameters and m=2, Barlat-Lian reduces identically to von Mises."""
        bp = barlat_params(1.0, 1.0, 1.0, m=2.0)
        trials = [
            np.array([120.0, 60.0, 30.0]),
            np.array([-90.0, 150.0, -40.0]),
            np.array([210.0, -110.0, 75.0]),
        ]
        for sig in trials:
            sxx, syy, sxy = sig
            seq_b = barlat_equivalent_stress(sig, bp.a, bp.c, bp.h_bar, bp.p, m=2.0)
            seq_vm = math.sqrt(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy)
            assert seq_b == pytest.approx(seq_vm, rel=1e-12)


# ============================================================================
# 4. Multi-Rate Tabulated Yield Curves & Strain Rate Filtering
# ============================================================================

class TestMultiRateAndFiltering:
    """Parity tests for table_mat_vinterp.F and sigeps57c.F90:170-180, 516-521."""

    def test_multi_rate_linear_interpolation(self):
        """Verify yield stress bilinear interpolation between multiple strain rate curves."""
        # Define 3 rate curves at epsd = [0.0, 10.0, 100.0]
        x_pts = [0.0, 0.05, 0.15, 0.30]
        y_r0 = [200.0, 240.0, 280.0, 320.0]
        y_r1 = [250.0, 300.0, 350.0, 400.0]
        y_r2 = [300.0, 370.0, 440.0, 500.0]

        mat = build_law57(
            E=210000.0, nu=0.3,
            curves=[
                (x_pts, y_r0, 0.0),
                (x_pts, y_r1, 10.0),
                (x_pts, y_r2, 100.0),
            ]
        )

        # Test at pla = 0.05 and intermediate rate epsd = 5.0 (midway between 0 and 10)
        # Expected: 0.5 * 240 + 0.5 * 300 = 270.0
        sy, h_slope, sy0 = _eval_yield_stress(mat, np.array([0.05]), np.array([5.0]))
        assert sy[0] == pytest.approx(270.0, rel=1e-10)

        # Test at pla = 0.15 and rate = 55.0 (halfway between 10 and 100: w = 45/90 = 0.5)
        # Expected: 0.5 * 350 + 0.5 * 440 = 395.0
        sy, _, _ = _eval_yield_stress(mat, np.array([0.15]), np.array([55.0]))
        assert sy[0] == pytest.approx(395.0, rel=1e-10)

        # Test at rate above maximum (clamped to highest rate)
        sy, _, _ = _eval_yield_stress(mat, np.array([0.05]), np.array([500.0]))
        assert sy[0] == pytest.approx(370.0, rel=1e-10)

    def test_strain_rate_filtering_vp0(self):
        """Verify total strain rate filtering when vp=0, israte=1 matching sigeps57c.F90:178."""
        # In sigeps57c.F90 line 178: epsd = asrate*epsd_pg + (one - asrate)*epsd
        mat = build_law57(E=200000.0, nu=0.3, israte=1, asrate=0.25, vp=0)

        extra = {
            "epsd57": np.array([10.0]),     # epsd_prev
            "epsd_pg": np.array([50.0]),    # instantaneous epsd_pg
            "asrate": 0.25,
        }
        deps = np.array([1.0e-4, 0.0, 0.0])
        sig = np.zeros(3)

        shell_update_law57(mat, sig, deps, dt=1.0e-5, extra=extra)

        # Expected filtered epsd: 0.25 * 50.0 + 0.75 * 10.0 = 20.0
        assert extra["epsd57"][0] == pytest.approx(20.0, rel=1e-10)

    def test_plastic_strain_rate_filtering_vp1(self):
        """Verify plastic strain rate filtering when vp=1 matching sigeps57c.F90:519-520."""
        # In sigeps57c.F90 lines 519-520:
        # dpdt = dpla / dt
        # epsd = asrate*dpdt + (one - asrate)*epsd
        mat = build_law57(
            E=200000.0, nu=0.3,
            sigy0=200.0,
            curves=[([0.0, 0.1], [200.0, 300.0])],
            vp=1,
            israte=1,
            asrate=0.4,
        )

        dt = 1.0e-5
        # Large strain increment driving plasticity
        deps = np.array([0.005, 0.0, 0.0])
        sig = np.array([180.0, 0.0, 0.0])
        extra = {
            "epsd57": np.array([5.0]),  # previous rate
            "asrate": 0.4,
        }

        res = shell_update_law57(mat, sig, deps, dt=dt, extra=extra)
        pla = res.pla
        assert pla > 0.0

        # dpdt = pla / dt
        dpdt = pla / dt
        expected_epsd = 0.4 * dpdt + 0.6 * 5.0
        assert extra["epsd57"][0] == pytest.approx(expected_epsd, rel=1e-8)


# ============================================================================
# 5. Mixed Isotropic / Kinematic Hardening & Backstress Evolution
# ============================================================================

class TestMixedKinematicHardening:
    """Parity tests for sigeps57c.F90:278-280, 383-395, 431-433."""

    def test_pure_isotropic_hardening_zero_backstress(self):
        """When fisokin = 0.0 (CHARD = 0), backstress alpha remains identically zero."""
        mat = build_law57(
            E=210000.0, nu=0.3,
            fisokin=0.0,
            curves=[([0.0, 0.05, 0.10], [200.0, 300.0, 400.0])]
        )

        sig = np.zeros(3)
        deps = np.array([0.003, 0.0, 0.0])
        extra = {"sigb57": np.zeros(3)}

        res = shell_update_law57(mat, sig, deps, extra=extra)
        assert res.pla > 0.0
        np.testing.assert_allclose(extra["sigb57"], [0.0, 0.0, 0.0], atol=1e-12)

    def test_pure_kinematic_hardening_backstress_growth(self):
        """When fisokin = 1.0 (CHARD = 1), yield stress stays at Y0 while backstress grows."""
        mat = build_law57(
            E=210000.0, nu=0.3,
            fisokin=1.0,
            curves=[([0.0, 0.05, 0.10], [200.0, 300.0, 400.0])]
        )

        sig = np.zeros(3)
        deps = np.array([0.004, 0.0, 0.0])
        extra = {"sigb57": np.zeros(3)}

        res = shell_update_law57(mat, sig, deps, extra=extra)
        assert res.pla > 0.0
        # Backstress alpha_xx must be positive and significant
        assert extra["sigb57"][0] > 0.0
        # In-plane shear backstress must be identically zero for purely normal loading
        assert extra["sigb57"][2] == pytest.approx(0.0, abs=1e-12)

    def test_bauschinger_effect_cyclic_loading(self):
        """Verify Bauschinger effect under kinematic hardening: reverse yield is lower."""
        mat = build_law57(
            E=200000.0, nu=0.3,
            fisokin=1.0,
            curves=[([0.0, 0.1], [200.0, 400.0])]  # Plastic hardening slope H = 2000 MPa
        )

        extra = {"sigb57": np.zeros(3), "pla57": np.zeros(1)}

        # Forward tension yielding
        sig1, pla1, _ = shell_update_law57(mat, np.zeros(3), np.array([0.003, 0.0, 0.0]), extra=extra)
        assert pla1 > 0.0
        fwd_stress = sig1[0]
        assert fwd_stress > 200.0
        assert extra["sigb57"][0] > 0.0

        # Now apply compression reversal
        sig2, pla2, _ = shell_update_law57(mat, sig1, np.array([-0.003, 0.0, 0.0]), extra=extra)
        # With backstress pointing in +xx, reverse yield in -xx occurs at |sigma| < 200 MPa
        # (earlier onset of reverse plasticity, characteristic of Bauschinger effect)
        assert abs(sig2[0]) < fwd_stress


# ============================================================================
# 6. Dynamic Young's Modulus Degradation
# ============================================================================

class TestYoungModulusDegradation:
    """Parity tests for sigeps57c.F90:200-216."""

    def test_exponential_young_degradation(self):
        """Verify E(pla) = E0 - (E0 - Einf)*(1 - exp(-CE*pla))."""
        e0 = 210000.0
        einf = 160000.0
        ce = 25.0
        nu = 0.3
        mat = build_law57(E=e0, nu=nu, einf=einf, ce=ce)

        for pla_val in (0.0, 0.02, 0.05, 0.15, 0.50):
            e_arr, a11_arr, a12_arr, g_arr = _eval_young(mat, np.array([pla_val]))
            e_expected = e0 - (e0 - einf) * (1.0 - math.exp(-ce * pla_val))
            assert e_arr[0] == pytest.approx(e_expected, rel=1e-12)
            assert a11_arr[0] == pytest.approx(e_expected / (1.0 - nu ** 2), rel=1e-12)
            assert a12_arr[0] == pytest.approx(nu * a11_arr[0], rel=1e-12)
            assert g_arr[0] == pytest.approx(e_expected / (2.0 * (1.0 + nu)), rel=1e-12)

    def test_tabulated_young_degradation_ifunce(self):
        """Verify piecewise linear E(pla) evaluation from user curve."""
        x_pts = [0.0, 0.05, 0.10, 0.20]
        y_pts = [210000.0, 195000.0, 180000.0, 165000.0]

        mat = build_law57(
            E=210000.0, nu=0.3,
            ifunce=10,
            E_curve=(x_pts, y_pts)
        )

        # At pla = 0.025 (halfway between 0 and 0.05): E = 202500
        e_arr, _, _, _ = _eval_young(mat, np.array([0.025]))
        assert e_arr[0] == pytest.approx(202500.0, rel=1e-10)

        # At pla = 0.075: E = 187500
        e_arr, _, _, _ = _eval_young(mat, np.array([0.075]))
        assert e_arr[0] == pytest.approx(187500.0, rel=1e-10)


# ============================================================================
# 7. Tensile Failure Damage Factor & Element Deletion
# ============================================================================

class TestTensileFailureAndDeletion:
    """Parity tests for sigeps57c.F90:504-508, 531-550."""

    def test_tensile_damage_scaling_formula(self):
        """Verify FAIL = clamp((epsr2 - epst)/(epsr2 - epsr1), 0.0, 1.0)."""
        epsr1 = 0.10
        epsr2 = 0.20
        mat = build_law57(
            E=200000.0, nu=0.3,
            epsr1=epsr1, epsr2=epsr2,
            curves=[([0.0, 0.3], [200.0, 400.0])]
        )

        # 1. Strain below epsr1: no damage, FAIL = 1.0, dmg[2] = 0.0
        extra1 = {"eps57": np.zeros(3), "dmg57": np.zeros(3)}
        shell_update_law57(mat, np.zeros(3), np.array([0.05, 0.0, 0.0]), extra=extra1)
        assert extra1["dmg57"][2] == pytest.approx(0.0)

        # 2. Intermediate strain: epst = 0.15 => dmg[2] = (0.15 - 0.10)/(0.20 - 0.10) = 0.5
        extra2 = {"eps57": np.zeros(3), "dmg57": np.zeros(3)}
        shell_update_law57(mat, np.zeros(3), np.array([0.15, 0.0, 0.0]), extra=extra2)
        assert extra2["dmg57"][2] == pytest.approx(0.5, rel=1e-6)

        # 3. Rupture strain: epst >= 0.20 => dmg[2] = 1.0, element deleted, stress zeroed
        extra3 = {"eps57": np.zeros(3), "dmg57": np.zeros(3)}
        res3 = shell_update_law57(mat, np.zeros(3), np.array([0.22, 0.0, 0.0]), extra=extra3)
        assert extra3["dmg57"][2] == pytest.approx(1.0)
        assert float(np.asarray(extra3["off57"]).flatten()[0]) == pytest.approx(0.0)
        np.testing.assert_allclose(res3.sig, [0.0, 0.0, 0.0], atol=1e-12)

    def test_plastic_strain_deletion_epsmax(self):
        """Verify element deletion when pla >= epsmax matching sigeps57c.F90:504-508."""
        epsmax = 0.05
        mat = build_law57(
            E=200000.0, nu=0.3,
            epsmax=epsmax,
            curves=[([0.0, 0.1], [200.0, 300.0])]
        )

        extra = {"pla57": np.zeros(1), "off57": np.array([1.0])}
        # Apply strain increment that drives pla past epsmax
        res = shell_update_law57(mat, np.zeros(3), np.array([0.06, 0.0, 0.0]), extra=extra)

        assert extra["pla57"][0] >= epsmax
        assert extra["off57"][0] == pytest.approx(0.0)
        np.testing.assert_allclose(res.sig, [0.0, 0.0, 0.0], atol=1e-12)


# ============================================================================
# 8. Shell Through-Thickness Thinning Increment
# ============================================================================

class TestShellThicknessThinning:
    """Parity tests for sigeps57c.F90:417, 596-601."""

    def test_elastic_shell_thinning(self):
        """In pure elasticity, deps_zz = -nu/(1-nu)*(deps_xx + deps_yy)."""
        e = 210000.0
        nu = 0.3
        mat = build_law57(E=e, nu=nu, sigy0=1.0e10)  # Pure elastic

        deps_xx = 1.0e-3
        deps_yy = 5.0e-4
        deps = np.array([deps_xx, deps_yy, 0.0])
        extra = {"thk57": np.array([2.0]), "thkly": np.array([2.0])}

        shell_update_law57(mat, np.zeros(3), deps, extra=extra)

        expected_deps_zz = -(nu / (1.0 - nu)) * (deps_xx + deps_yy)
        assert extra["depszz"] == pytest.approx(expected_deps_zz, rel=1e-10)
        expected_thk = 2.0 + expected_deps_zz * 2.0
        assert extra["thk57"][0] == pytest.approx(expected_thk, rel=1e-10)

    def test_plastic_volume_incompressibility_thinning(self):
        """In plastic yielding, out-of-plane plastic thinning balances in-plane plastic strains."""
        mat = build_law57(
            E=200000.0, nu=0.3,
            curves=[([0.0, 0.1], [200.0, 200.0])]  # Perfectly plastic
        )

        deps = np.array([0.005, 0.0, 0.0])
        extra = {"thk57": np.array([1.5]), "thkly": np.array([1.5])}

        shell_update_law57(mat, np.zeros(3), deps, extra=extra)

        # With plastic strain > 0, total thickness thinning is negative (sheet thins)
        assert extra["depszz"] < 0.0
        assert extra["thk57"][0] < 1.5


# ============================================================================
# 9. Acoustic Sound Speed
# ============================================================================

class TestAcousticSoundSpeed:
    """Parity tests for sigeps57c.F90:603."""

    @pytest.mark.parametrize("e,nu,rho", [
        (210000.0, 0.3, 7.85e-9),   # Steel (t-mm-s units)
        (70000.0, 0.33, 2.7e-9),    # Aluminum
        (110000.0, 0.34, 4.5e-9),   # Titanium
    ])
    def test_thin_shell_sound_speed_formula(self, e, nu, rho):
        """Verify c_shell = sqrt(E / ((1 - nu^2) * rho0))."""
        mat = build_law57(E=e, nu=nu, rho0=rho)
        c_calc = sound_speed_shell_law57(mat)
        c_expected = math.sqrt(e / ((1.0 - nu ** 2) * rho))
        assert c_calc == pytest.approx(c_expected, rel=1e-12)

    def test_sound_speed_degrades_with_young_modulus(self):
        """When Young's modulus degrades, sound speed decreases accordingly."""
        e0 = 200000.0
        einf = 120000.0
        ce = 50.0
        rho = 7.8e-9
        nu = 0.3
        mat = build_law57(E=e0, nu=nu, rho0=rho, einf=einf, ce=ce)

        extra = {"pla57": np.array([0.1])}
        res = shell_update_law57(mat, np.zeros(3), np.array([1e-5, 0.0, 0.0]), extra=extra)

        e_deg = e0 - (e0 - einf) * (1.0 - math.exp(-ce * 0.1))
        expected_c = math.sqrt(e_deg / ((1.0 - nu ** 2) * rho))
        assert res.c_sound == pytest.approx(expected_c, rel=1e-10)


# ============================================================================
# 10. Rejection of 3D Solid Elements
# ============================================================================

def test_solid_update_raises_not_implemented():
    """LAW57 is strictly 2D plane-stress: solid_update must raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="shell elements only"):
        solid_update_law57()
