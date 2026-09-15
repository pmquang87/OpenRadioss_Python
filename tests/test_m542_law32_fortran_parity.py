"""
Exhaustive Fortran Parity Audit Suite for LAW32 (/MAT/LAW32, /MAT/HILL).

Audited against upstream OpenRadioss source files:
  - starter/source/materials/mat/mat032/hm_read_mat32.F
  - engine/source/materials/mat/mat032/m32elas.F
  - engine/source/materials/mat/mat032/m32plas.F
  - engine/source/materials/mat/mat032/sigeps32c.F
  - engine/source/airbag/uroto.F

Covers 8 critical physics parity checkpoints:
  1. Default values & error checking in hm_read_mat32.F
  2. Hill 1948 anisotropy coefficients calculation
  3. Orthotropic elastic predictor (m32elas.F)
  4. Coordinate rotation to material axes (m32plas.F:108-114, uroto.F)
  5. Strain rate measure and dynamic yield (m32plas.F:118-129)
  6. Three plastic return algorithms (IPLA=0, IPLA=1, IPLA=2)
  7. Through-thickness strain increment Ezz and thickness update
  8. Sound speed SDSP = sqrt(YOUNG / RHO0) (hm_read_mat32.F:155)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials import law32_hill
from pyradioss.materials.law32_hill import (
    build_law32,
    shell_update,
    sound_speed,
    _rot_elem_to_mat,
    _rot_mat_to_elem,
    _get_dir_cosines,
)


# ============================================================================
# 1. Default Values & Error Checking in hm_read_mat32.F
# ============================================================================

class TestHmReadMat32Defaults:
    """Parity audit of starter parameter processing in hm_read_mat32.F."""

    def test_anu_clamping_at_half(self):
        """hm_read_mat32.F:139: IF(ANU==HALF) ANU=ZEP499 (0.499)."""
        mat = build_law32(id=1, E=2.1e11, nu=0.5, A=200e6)
        assert mat.params["nu"] == pytest.approx(0.499)

        # Also for nu > 0.5
        mat_gt = build_law32(id=2, E=2.1e11, nu=0.52, A=200e6)
        assert mat_gt.params["nu"] == pytest.approx(0.499)

        # Negative nu rejected
        with pytest.raises(ValueError, match="Poisson's ratio nu must be >= 0"):
            build_law32(id=3, E=2.1e11, nu=-0.1, A=200e6)

    def test_r00_r45_r90_default_to_one(self):
        """hm_read_mat32.F:140-142: IF(R00==ZERO) R00=ONE, etc."""
        mat = build_law32(id=1, E=2.0e11, nu=0.3, A=300e6, r00=0.0, r45=0.0, r90=0.0)
        assert mat.params["r00"] == pytest.approx(1.0)
        assert mat.params["r45"] == pytest.approx(1.0)
        assert mat.params["r90"] == pytest.approx(1.0)
        # Yields exact von Mises coefficients
        assert mat.params["A11"] == pytest.approx(1.0)
        assert mat.params["A22"] == pytest.approx(1.0)
        assert mat.params["A1122"] == pytest.approx(1.0)
        assert mat.params["A12"] == pytest.approx(3.0)

    def test_ca_zero_defaults_to_infinity(self):
        """hm_read_mat32.F:143: IF(CA==ZERO) CA=INFINITY."""
        mat = build_law32(id=1, E=2.0e11, nu=0.3, A=0.0)
        assert mat.params["A"] >= 1e30

    def test_cn_zero_defaults_to_one_and_error_if_gt_one(self):
        """hm_read_mat32.F:144: IF(CN==ZERO) CN=ONE; line 219: IF(CN>1.) ERROR 213."""
        mat = build_law32(id=1, E=2.0e11, nu=0.3, A=200e6, n=0.0)
        assert mat.params["n"] == pytest.approx(1.0)

        # Error if CN > 1.0
        with pytest.raises(ValueError, match="upstream error 213"):
            build_law32(id=2, E=2.0e11, nu=0.3, A=200e6, n=1.05)

    def test_epsm_sigm_zero_default_to_infinity(self):
        """hm_read_mat32.F:145-146: IF(EPSM==ZERO) EPSM=INFINITY; IF(SIGM==ZERO) SIGM=INFINITY."""
        mat = build_law32(id=1, E=2.0e11, nu=0.3, A=200e6, eps_max=0.0, sig_max=0.0)
        assert mat.params["eps_max"] >= 1e30
        assert mat.params["sig_max"] >= 1e30

    def test_cm_zero_sets_eps0_one(self):
        """hm_read_mat32.F:147: IF(CM==ZERO) EPS0=ONE."""
        # Even if MAT_SRP was given as 0 or not given, CM==0 forces EPS0 = 1.0
        mat = build_law32(id=1, E=2.0e11, nu=0.3, A=200e6, m=0.0, eps0=0.0)
        assert mat.params["m"] == pytest.approx(0.0)
        assert mat.params["eps0"] == pytest.approx(1.0)

    def test_eps0_error_if_le_zero_when_cm_nonzero(self):
        """hm_read_mat32.F:224-231: IF(EPS0<=ZERO) ERROR 207."""
        with pytest.raises(ValueError, match="upstream error 207"):
            build_law32(id=1, E=2.0e11, nu=0.3, A=200e6, m=0.1, eps0=0.0)

        with pytest.raises(ValueError, match="upstream error 207"):
            build_law32(id=2, E=2.0e11, nu=0.3, A=200e6, m=0.1, eps0=-1.5)


# ============================================================================
# 2. Hill 1948 Anisotropy Coefficients Calculation
# ============================================================================

class TestHill1948Coefficients:
    """Parity audit of Hill coefficients in hm_read_mat32.F:157-168."""

    def test_hill_coefficients_formulas_ir0_zero(self):
        """Verify Hill 1948 formulas without IR0 normalization.
        R = 0.25 * (R00 + 2*R45 + R90)
        H = R / (1 + R)
        A11 = H * (1 + 1/R00)
        A22 = H * (1 + 1/R90)
        A1122 = 2 * H
        A12 = 2 * H * (R45 + 0.5) * (1/R00 + 1/R90)
        """
        r00, r45, r90 = 1.6, 1.1, 2.4
        r = 0.25 * (r00 + 2.0 * r45 + r90)  # 0.25 * (1.6 + 2.2 + 2.4) = 1.55
        h = r / (1.0 + r)                    # 1.55 / 2.55 = 0.607843137...
        exp_a11 = h * (1.0 + 1.0 / r00)
        exp_a22 = h * (1.0 + 1.0 / r90)
        exp_a1122 = 2.0 * h
        exp_a12 = 2.0 * h * (r45 + 0.5) * (1.0 / r00 + 1.0 / r90)

        mat = build_law32(id=1, E=2.0e11, nu=0.3, A=300e6, r00=r00, r45=r45, r90=r90, i_yield=0)
        p = mat.params
        assert p["A11"] == pytest.approx(exp_a11, rel=1e-14)
        assert p["A22"] == pytest.approx(exp_a22, rel=1e-14)
        assert p["A1122"] == pytest.approx(exp_a1122, rel=1e-14)
        assert p["A12"] == pytest.approx(exp_a12, rel=1e-14)

    def test_hill_coefficients_ir0_nonzero(self):
        """hm_read_mat32.F:163-168:
        IF (IR0 > 0) THEN
          A22=A22/A11; A1122=A1122/A11; A12=A12/A11; A11=1.0
        END IF
        """
        r00, r45, r90 = 1.6, 1.1, 2.4
        r = 0.25 * (r00 + 2.0 * r45 + r90)
        h = r / (1.0 + r)
        raw_a11 = h * (1.0 + 1.0 / r00)
        raw_a22 = h * (1.0 + 1.0 / r90)
        raw_a1122 = 2.0 * h
        raw_a12 = 2.0 * h * (r45 + 0.5) * (1.0 / r00 + 1.0 / r90)

        mat = build_law32(id=1, E=2.0e11, nu=0.3, A=300e6, r00=r00, r45=r45, r90=r90, i_yield=1)
        p = mat.params
        assert p["A11"] == pytest.approx(1.0, rel=1e-14)
        assert p["A22"] == pytest.approx(raw_a22 / raw_a11, rel=1e-14)
        assert p["A1122"] == pytest.approx(raw_a1122 / raw_a11, rel=1e-14)
        assert p["A12"] == pytest.approx(raw_a12 / raw_a11, rel=1e-14)


# ============================================================================
# 3. Orthotropic Elastic Predictor (m32elas.F)
# ============================================================================

class TestOrthotropicElasticPredictorM32elas:
    """Parity audit of m32elas.F:65-75:
    G11 = YOUNG / (2 * (1 + ANU))
    A11 = YOUNG / (1 - ANU^2)
    A21 = ANU * A11
    SIGNXX = SIGOXX + A11*DEPSXX + A21*DEPSYY
    SIGNYY = SIGOYY + A21*DEPSXX + A11*DEPSYY
    SIGNXY = SIGOXY + G11*DEPSXY
    SIGNYZ = SIGOYZ + GS*DEPSYZ
    SIGNZX = SIGOZX + GS*DEPSZX
    """

    def test_elastic_trial_stresses(self):
        young = 206000.0
        anu = 0.28
        mat = build_law32(id=1, E=young, nu=anu, A=1000.0, B=1.0, n=0.0)

        a11 = young / (1.0 - anu * anu)
        a21 = anu * a11
        g11 = young / (2.0 * (1.0 + anu))
        gs = 5.0 / 6.0 * g11

        sig_old = np.array([[50.0, -30.0, 20.0, 10.0, -5.0]])
        deps = np.array([[0.001, -0.0005, 0.0008, 0.0002, -0.0003]])

        exp_signxx = sig_old[0, 0] + a11 * deps[0, 0] + a21 * deps[0, 1]
        exp_signyy = sig_old[0, 1] + a21 * deps[0, 0] + a11 * deps[0, 1]
        exp_signxy = sig_old[0, 2] + g11 * deps[0, 2]
        exp_signyz = sig_old[0, 3] + gs * deps[0, 3]
        exp_signzx = sig_old[0, 4] + gs * deps[0, 4]

        sig_new, epsp_new = shell_update(mat, sig_old.copy(), deps, dt=1e-5)

        assert sig_new[0, 0] == pytest.approx(exp_signxx, rel=1e-12)
        assert sig_new[0, 1] == pytest.approx(exp_signyy, rel=1e-12)
        assert sig_new[0, 2] == pytest.approx(exp_signxy, rel=1e-12)
        assert sig_new[0, 3] == pytest.approx(exp_signyz, rel=1e-12)
        assert sig_new[0, 4] == pytest.approx(exp_signzx, rel=1e-12)
        assert epsp_new[0] == pytest.approx(0.0)


# ============================================================================
# 4. Coordinate Rotation to Material Axes (m32plas.F:108-114, uroto.F)
# ============================================================================

class TestCoordinateRotationM32plas:
    """Parity audit of tensor coordinate rotation."""

    def test_rotation_forward_and_backward(self):
        """m32plas.F:108-114 & uroto.F:45-62:
        D11 = DIR(1)*DIR(1), D22 = DIR(2)*DIR(2), D12 = DIR(1)*DIR(2)
        S11 = D11*SIGNXX + D22*SIGNYY + 2*D12*SIGNXY
        S22 = D22*SIGNXX + D11*SIGNYY - 2*D12*SIGNXY
        S12 = D12*(SIGNYY - SIGNXX) + (D11 - D22)*SIGNXY

        uroto.F:
        SXX = D11*S11 + D22*S22 - 2*D12*S12
        SYY = D22*S11 + D11*S22 + 2*D12*S12
        SXY = D12*(S11 - S22) + (D11 - D22)*S12
        """
        theta = math.radians(37.5)
        c = math.cos(theta)
        s = math.sin(theta)
        d11 = np.array([c * c])
        d22 = np.array([s * s])
        d12 = np.array([c * s])

        sxx = np.array([125.0])
        syy = np.array([-45.0])
        sxy = np.array([60.0])

        s11, s22, s12 = _rot_elem_to_mat(sxx, syy, sxy, d11, d22, d12)

        exp_s11 = d11[0] * sxx[0] + d22[0] * syy[0] + 2.0 * d12[0] * sxy[0]
        exp_s22 = d22[0] * sxx[0] + d11[0] * syy[0] - 2.0 * d12[0] * sxy[0]
        exp_s12 = d12[0] * (syy[0] - sxx[0]) + (d11[0] - d22[0]) * sxy[0]

        assert s11[0] == pytest.approx(exp_s11, rel=1e-14)
        assert s22[0] == pytest.approx(exp_s22, rel=1e-14)
        assert s12[0] == pytest.approx(exp_s12, rel=1e-14)

        # Invariance of trace: S11 + S22 == SXX + SYY
        assert (s11[0] + s22[0]) == pytest.approx(sxx[0] + syy[0], rel=1e-14)

        # Reverse rotation reconstructs original stresses
        sxx_rec, syy_rec, sxy_rec = _rot_mat_to_elem(s11, s22, s12, d11, d22, d12)
        assert sxx_rec[0] == pytest.approx(sxx[0], rel=1e-14)
        assert syy_rec[0] == pytest.approx(syy[0], rel=1e-14)
        assert sxy_rec[0] == pytest.approx(sxy[0], rel=1e-14)


# ============================================================================
# 5. Strain Rate Measure & Dynamic Yield (m32plas.F:118-129)
# ============================================================================

class TestStrainRateDynamicYield:
    """Parity audit of strain rate calculation and dynamic yield in m32plas.F."""

    def test_epsp_rate_formula_and_epdr_clamp(self):
        """m32plas.F:118-121:
        EPSP = MAX(ABS(DEPSXX), ABS(DEPSYY), HALF*ABS(DEPSXY)) * DTINV
        EPSP = MAX(EPSP, EPDR)
        YLD = CA*(CE+EPSEQ)**CN * EPSP**CM
        """
        deps_xx = 0.002
        deps_yy = -0.003
        deps_xy = 0.008  # half is 0.004 -> max is 0.004
        dt = 2e-4
        raw_rate = max(abs(deps_xx), abs(deps_yy), 0.5 * abs(deps_xy)) / dt  # 0.004 / 2e-4 = 20.0

        epdr = 5.0
        cm = 0.15
        ca = 400.0
        ce = 0.01
        cn = 0.2

        mat = build_law32(id=1, E=210000.0, nu=0.3, A=ca, B=ce, n=cn, m=cm, eps0=epdr)

        sig = np.zeros((1, 3))
        deps = np.array([[deps_xx, deps_yy, deps_xy]])
        epsp = np.array([0.05])

        # Expected flow stress:
        # EPSP = max(20.0, 5.0) = 20.0
        # YLD = CA * (CE + EPSEQ)^CN * EPSP^CM
        exp_yld = ca * ((ce + 0.05) ** cn) * (20.0 ** cm)

        sig_out, epsp_out = shell_update(mat, sig, deps, epsp, dt=dt)
        # Verify equivalent stress on output equals exp_yld
        seq = math.sqrt(sig_out[0, 0]**2 + sig_out[0, 1]**2 - sig_out[0, 0]*sig_out[0, 1] + 3.0*sig_out[0, 2]**2)
        assert seq == pytest.approx(exp_yld, rel=1e-4)

    def test_rate_clamped_at_floor_epdr(self):
        """When strain rate < EPDR, EPSP is clamped to EPDR (m32plas.F:121)."""
        dt = 1.0
        deps = np.array([[0.005, 0.0, 0.0]])  # rate = 0.005 << 50.0, trial stress > 1000 > exp_yld
        epdr = 50.0
        cm = 0.1
        ca = 300.0
        ce = 1.0
        cn = 0.0

        mat = build_law32(id=1, E=210000.0, nu=0.3, A=ca, B=ce, n=cn, m=cm, eps0=epdr)

        sig_out, _ = shell_update(mat, np.zeros((1, 3)), deps, dt=dt)
        exp_yld = ca * (epdr ** cm)
        seq = math.sqrt(sig_out[0, 0]**2 + sig_out[0, 1]**2 - sig_out[0, 0]*sig_out[0, 1] + 3.0*sig_out[0, 2]**2)
        assert seq == pytest.approx(exp_yld, rel=1e-4)


# ============================================================================
# 6. Three Plastic Return Algorithms (m32plas.F: IPLA=0, 1, 2)
# ============================================================================

class TestPlasticReturnAlgorithms:
    """Parity audit of the three return mapping algorithms against exact Fortran arithmetic."""

    def test_radial_return_ipla_0_exact_fortran(self):
        """m32plas.F:103-143: IPLA=0 radial projection:
        SEQ = SQRT(A11*S11^2 + A22*S22^2 - A1122*S11*S22 + A12*S12^2)
        SCALE = MIN(1, YLD / SEQ)
        SIGNXX = SIGNXX * SCALE
        SIGNYY = SIGNYY * SCALE
        SIGNXY = SIGNXY * SCALE
        DPLA = (SEQ - YLD) / E
        """
        e = 200000.0
        nu = 0.3
        mat = build_law32(id=1, E=e, nu=nu, A=300.0, B=1.0, n=0.0, r00=1.6, r45=1.2, r90=2.0, ipla=0)
        p = mat.params

        deps = np.array([[0.003, 0.001, 0.002]])
        a11_el = e / (1.0 - nu**2)
        a21_el = nu * a11_el
        g11_el = e / (2.0 * (1.0 + nu))

        sxx_tr = a11_el * deps[0, 0] + a21_el * deps[0, 1]
        syy_tr = a21_el * deps[0, 0] + a11_el * deps[0, 1]
        sxy_tr = g11_el * deps[0, 2]

        seq_tr = math.sqrt(p["A11"] * sxx_tr**2 + p["A22"] * syy_tr**2
                           - p["A1122"] * sxx_tr * syy_tr + p["A12"] * sxy_tr**2)
        yld = 300.0
        scale = min(1.0, yld / seq_tr)
        exp_dpla = (seq_tr - yld) / e

        sig_out, epsp_out = shell_update(mat, np.zeros((1, 3)), deps, dt=1e-5)

        assert sig_out[0, 0] == pytest.approx(sxx_tr * scale, rel=1e-12)
        assert sig_out[0, 1] == pytest.approx(syy_tr * scale, rel=1e-12)
        assert sig_out[0, 2] == pytest.approx(sxy_tr * scale, rel=1e-12)
        assert epsp_out[0] == pytest.approx(exp_dpla, rel=1e-12)

    def test_plane_stress_projection_ipla_2_exact_fortran(self):
        """m32plas.F:144-196: IPLA=2 projection with p=const + s33=0:
        NU1 = NU / (1 - NU)
        P = -(S11 + S22) / 3
        Q = (1 - NU1) * P
        S11 = S11 + Q, S22 = S22 + Q
        A = A11*S11^2 + A22*S22^2 - A1122*S11*S22 + A12*S12^2
        B = -Q * (A11*S11 + A22*S22 - 0.5*A1122*(S11+S22))
        C = (A11 + A22 - A1122)*Q^2 - YLD^2
        SCALE = MIN(1, (-B + SQRT(B^2 - A*C)) / A)
        UMR = 1 - SCALE
        Q = Q * UMR
        SIGNXX = SIGNXX*SCALE - Q
        SIGNYY = SIGNYY*SCALE - Q
        SIGNXY = SIGNXY*SCALE
        DPLA = SEQ * UMR / G3, G3 = 1.5*E / (1+NU)
        """
        e = 200000.0
        nu = 0.3
        mat = build_law32(id=1, E=e, nu=nu, A=300.0, B=1.0, n=0.0, r00=1.6, r45=1.2, r90=2.0, ipla=2)
        p = mat.params

        deps = np.array([[0.003, -0.001, 0.002]])
        a11_el = e / (1.0 - nu**2)
        a21_el = nu * a11_el
        g11_el = e / (2.0 * (1.0 + nu))

        sxx_tr = a11_el * deps[0, 0] + a21_el * deps[0, 1]
        syy_tr = a21_el * deps[0, 0] + a11_el * deps[0, 1]
        sxy_tr = g11_el * deps[0, 2]

        nu1 = nu / (1.0 - nu)
        p_hydro = -(sxx_tr + syy_tr) / 3.0
        q = (1.0 - nu1) * p_hydro
        s11 = sxx_tr + q
        s22 = syy_tr + q
        s12 = sxy_tr

        a_quad = p["A11"] * s11**2 + p["A22"] * s22**2 - p["A1122"] * s11 * s22 + p["A12"] * s12**2
        b_quad = -q * (p["A11"] * s11 + p["A22"] * s22 - 0.5 * p["A1122"] * (s11 + s22))
        c_quad = (p["A11"] + p["A22"] - p["A1122"]) * q**2
        seq_p = math.sqrt(a_quad + 2.0 * b_quad + c_quad)
        c_quad -= 300.0**2

        scale = (-b_quad + math.sqrt(b_quad**2 - a_quad * c_quad)) / a_quad
        umr = 1.0 - scale
        q_corr = q * umr
        g3 = 1.5 * e / (1.0 + nu)
        exp_dpla = seq_p * umr / g3

        sig_out, epsp_out = shell_update(mat, np.zeros((1, 3)), deps, dt=1e-5)

        assert sig_out[0, 0] == pytest.approx(sxx_tr * scale - q_corr, rel=1e-12)
        assert sig_out[0, 1] == pytest.approx(syy_tr * scale - q_corr, rel=1e-12)
        assert sig_out[0, 2] == pytest.approx(sxy_tr * scale, rel=1e-12)
        assert epsp_out[0] == pytest.approx(exp_dpla, rel=1e-12)

    def test_newton_raphson_ipla_1_exact_fortran(self):
        """m32plas.F:198-350: IPLA=1 iterative Newton-Raphson return."""
        e = 200000.0
        nu = 0.3
        mat = build_law32(id=1, E=e, nu=nu, A=300.0, B=0.01, n=0.2, r00=1.5, r45=1.2, r90=1.8, ipla=1)

        deps = np.array([[0.005, -0.001, 0.003]])
        sig_out, epsp_out = shell_update(mat, np.zeros((1, 3)), deps, dt=1e-5)

        # After return mapping, the stress lies on the linearized yield surface per m32plas.F:304,331:
        # YLD_I = YLDP + H * DPLA
        p = mat.params
        a11, a22, a1122, a12 = p["A11"], p["A22"], p["A1122"], p["A12"]
        seq_final = math.sqrt(a11 * sig_out[0, 0]**2 + a22 * sig_out[0, 1]**2
                              - a1122 * sig_out[0, 0] * sig_out[0, 1] + a12 * sig_out[0, 2]**2)
        yld_p = 300.0 * (0.01 ** 0.2)
        h_slope = 300.0 * 0.2 * (0.01 ** (0.2 - 1.0))
        yld_linear = yld_p + h_slope * epsp_out[0]
        assert seq_final == pytest.approx(yld_linear, rel=1e-4)


# ============================================================================
# 7. Through-Thickness Strain Increment Ezz & Thickness Update
# ============================================================================

class TestThroughThicknessStrainEzz:
    """Parity audit of through-thickness strain Ezz and thickness update:
    m32plas.F:140, 193: EZZ = -NU5 * DPLA * HALF * (SIGNXX + SIGNYY) / YLD
    m32plas.F:344: EZZ = -NU5 * DPLA * S1 / YLD
    sigeps32c.F:190: EZZ = -(DEPSXX + DEPSYY)*(NU/(1-NU)) + EZZ
    sigeps32c.F:192: THK = THK + EZZ * THKLYL * OFF
    """

    def test_ezz_ipla_0(self):
        """Verify Ezz for IPLA=0."""
        e = 200000.0
        nu = 0.3
        mat = build_law32(id=1, E=e, nu=nu, A=300.0, B=1.0, n=0.0, ipla=0)

        deps = np.array([[0.004, 0.001, 0.0]])
        extra = {"ezz": 0.0}
        sig_out, epsp_out = shell_update(mat, np.zeros((1, 3)), deps, dt=1e-5, extra=extra)

        nu1 = nu / (1.0 - nu)
        nu5 = 1.0 - nu1
        dpla = epsp_out[0]
        yld = 300.0

        ezz_el = -(deps[0, 0] + deps[0, 1]) * nu1
        ezz_pl = -nu5 * dpla * 0.5 * (sig_out[0, 0] + sig_out[0, 1]) / yld
        exp_ezz = ezz_el + ezz_pl

        assert extra["ezz"][0] == pytest.approx(exp_ezz, rel=1e-12)

    def test_ezz_ipla_1_anisotropic_material_axes(self):
        """m32plas.F:342-344: IPLA=1 uses S1 = A11*S11 + A22*S22 - 0.5*A1122*(S11+S22)."""
        e = 200000.0
        nu = 0.3
        r00, r45, r90 = 1.6, 1.1, 2.4
        mat = build_law32(id=1, E=e, nu=nu, A=300.0, B=1.0, n=0.0, r00=r00, r45=r45, r90=r90, ipla=1)
        p = mat.params

        deps = np.array([[0.004, -0.001, 0.002]])
        extra = {"ezz": 0.0, "theta": math.radians(30.0)}
        sig_out, epsp_out = shell_update(mat, np.zeros((1, 3)), deps, dt=1e-5, extra=extra)

        nu1 = nu / (1.0 - nu)
        nu5 = 1.0 - nu1
        dpla = epsp_out[0]
        yld = 300.0

        # Rotate final stress to material frame
        d11, d22, d12 = _get_dir_cosines(extra, 1)
        s11_m, s22_m, _ = _rot_elem_to_mat(sig_out[:, 0], sig_out[:, 1], sig_out[:, 2], d11, d22, d12)
        s1 = p["A11"] * s11_m[0] + p["A22"] * s22_m[0] - 0.5 * p["A1122"] * (s11_m[0] + s22_m[0])

        ezz_el = -(deps[0, 0] + deps[0, 1]) * nu1
        ezz_pl = -nu5 * dpla * s1 / yld
        exp_ezz = ezz_el + ezz_pl

        assert extra["ezz"][0] == pytest.approx(exp_ezz, rel=1e-12)

    def test_thickness_update_parity(self):
        """sigeps32c.F:192: THK = THK + EZZ * THKLYL * OFF."""
        e = 200000.0
        nu = 0.3
        mat = build_law32(id=1, E=e, nu=nu, A=300.0, B=1.0, n=0.0, ipla=0)

        initial_thk = np.array([2.5])
        thklyl = np.array([2.5])
        extra = {"ezz": 0.0, "thk": initial_thk.copy(), "thklyl": thklyl}
        deps = np.array([[0.004, 0.001, 0.0]])

        shell_update(mat, np.zeros((1, 3)), deps, dt=1e-5, extra=extra)
        exp_thk = initial_thk[0] + extra["ezz"][0] * thklyl[0]
        assert extra["thk"][0] == pytest.approx(exp_thk, rel=1e-12)


# ============================================================================
# 8. Sound Speed SDSP = sqrt(YOUNG / RHO0) (hm_read_mat32.F:155)
# ============================================================================

class TestSoundSpeedSDSP:
    """Parity audit of sound speed SDSP in hm_read_mat32.F:155:
    SDSP = SQRT(YOUNG / MAX(PM(1), EM20))
    PM(27) = SDSP
    """

    def test_sound_speed_sdsp_exact_match(self):
        young = 2.1e11
        rho0 = 7850.0
        mat = build_law32(id=1, rho0=rho0, E=young, nu=0.3, A=300e6)
        exp_sdsp = math.sqrt(young / rho0)

        assert sound_speed(mat) == pytest.approx(exp_sdsp, rel=1e-14)
        assert mat.params["c"] == pytest.approx(exp_sdsp, rel=1e-14)

    def test_sound_speed_with_dynamic_rho(self):
        young = 2.1e11
        rho0 = 7850.0
        mat = build_law32(id=1, rho0=rho0, E=young, nu=0.3, A=300e6)
        rho_current = 7900.0
        exp_sdsp = math.sqrt(young / rho_current)

        assert sound_speed(mat, rho=rho_current) == pytest.approx(exp_sdsp, rel=1e-14)
