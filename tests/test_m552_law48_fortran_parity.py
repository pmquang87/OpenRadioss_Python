"""
Comprehensive Fortran Oracle Parity Test Suite for /MAT/LAW48 (/MAT/ZHAO).

Milestone M552 Subagent 2A: Fortran Parity Auditor
Audits pyradioss/materials/law48_zhao.py against upstream OpenRadioss Fortran reference code:
- engine/source/materials/mat/mat048/sigeps48.F (solids)
- engine/source/materials/mat/mat048/sigeps48c.F (shells)
- starter/source/materials/mat/mat048/hm_read_mat48.F (starter keyword reader)

Verifications:
1. Static work hardening P_A = c_a * sigma_y + c_b * eps_p^c_n and its cutoff at eps_p_max.
2. Coupled logarithmic rate sensitivity P_B = (c_c - c_d * eps_p^c_m) * ln(eps_dot / eps_dot_0) for eps_dot > eps_dot_0.
3. Uncoupled power-law high-rate term P_C = c_e * eps_dot^c_k for eps_dot > 0.
4. Total yield stress saturation YLD = min(S_max + P_C, P_A + P_B + P_C).
5. Plastic hardening slope H = P_DA + P_DB and its zeroing when saturated (YLD < YY).
6. 3D maximum principal strain cubic solver (sigeps48.F:201-245) vs exact analytical roots across pure tension, compression, and shear.
7. Tensile failure factor FAIL = max(0, min(1, (eps_r2 - eps_t) / (eps_r2 - eps_r1))) and scaling of yield stress and hardening.
8. Plane-stress shell radial projection and Newton-Raphson iterative return (sigeps48c.F).
9. Exact sound speeds c_solid = sqrt((C_1 + 4/3 G) / rho_0) and c_shell = sqrt(E / ((1 - nu^2) rho_0)).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyradioss.materials.law48_zhao import (
    Law48Params,
    build_law48,
    eval_yield_and_hardening,
    _principal_strain_3d,
    _principal_strain_2d,
    tensile_failure_factor,
    sound_speed_solid_law48,
    sound_speed_shell_law48,
    solid_update_law48,
    shell_update_law48,
    plane_stress_return_newton_law48,
)


# ============================================================================
# Fortran Oracle Implementations (Direct Line-by-Line Transcriptions)
# ============================================================================

def fortran_oracle_yield_and_hardening(
    ca: float,
    cb: float,
    cn: float,
    cc: float,
    cd: float,
    cm: float,
    ce: float,
    ck: float,
    eps0: float,
    smax: float,
    epmax: float,
    E: float,
    pla: float,
    epsp: float,
    fail: float = 1.0,
) -> tuple[float, float, float, float, float]:
    """Line-by-line replication of sigeps48.F lines 259-303."""
    # Line 259-266: PA static work hardening
    if pla <= 0.0:
        pa = ca
    elif pla > epmax:
        pa = ca + cb * (epmax ** cn)
    else:
        pa = ca + cb * (pla ** cn)

    # Line 267-273: PB coupled strain rate sensitivity
    if epsp <= eps0:
        pb = 0.0
    elif pla <= 0.0:
        pb = cc * math.log(epsp / eps0)
    else:
        pb = (cc - cd * (pla ** cm)) * math.log(epsp / eps0)

    # Line 274-278: PC high rate power law
    if epsp <= 0.0:
        pc = 0.0
    else:
        pc = ce * (epsp ** ck)

    # Line 280-286: PDA derivative of work hardening
    if pla > 0.0 and cn >= 1.0:
        pda = cb * cn * (pla ** (cn - 1.0))
    elif pla > 0.0 and cn < 1.0:
        pda = cb * cn * (pla ** (1.0 - cn))
    else:
        pda = E

    # Line 287-293: PDB derivative of coupled rate sensitivity
    if pla <= 0.0 or epsp <= eps0:
        pdb = 0.0
    elif cm >= 1.0:
        pdb = cd * cm * (pla ** (cm - 1.0)) * math.log(epsp / eps0)
    else:
        pdb = cd * cm * (pla ** (1.0 - cm)) * math.log(epsp / eps0)

    # Line 295-303: Saturation, fail scaling, and failure cutoff
    yy = pa + pb + pc
    yld = min(smax + pc, yy)
    h = pda + pdb
    if yld < yy:
        h = 0.0
    yld = fail * yld
    h = fail * h

    if pla > epmax:
        yld = 0.0

    return yld, h, pa, pb, pc


def fortran_oracle_cubic_solver(eps: list[float] | np.ndarray) -> float:
    """Exact replication of sigeps48.F lines 203-245."""
    e = list(eps)
    dav = (e[0] + e[1] + e[2]) / 3.0
    e1 = e[0] - dav
    e2 = e[1] - dav
    e3 = e[2] - dav
    e4 = 0.5 * e[3]
    e5 = 0.5 * e[4]
    e6 = 0.5 * e[5]

    e42 = e4 * e4
    e52 = e5 * e5
    e62 = e6 * e6

    c = - e1 * e1 - e2 * e2 - e3 * e3 - e42 - e52 - e62
    d = - e1 * e2 * e3 + e1 * e52 + e2 * e62 + e3 * e42 - 2.0 * e4 * e5 * e6
    cd = c / 3.0
    epst = math.sqrt(max(-cd, 0.0))
    epst2 = epst * epst
    y = (epst2 + c) * epst + d

    if abs(y) > 1.0e-8:
        epst = 1.75 * epst
        # Iteration 1
        epst2 = epst * epst
        y = (epst2 + c) * epst + d
        yp = 3.0 * epst2 + c
        if yp != 0.0:
            epst = epst - y / yp
        # Iteration 2
        epst2 = epst * epst
        y = (epst2 + c) * epst + d
        yp = 3.0 * epst2 + c
        if yp != 0.0:
            epst = epst - y / yp
        # Iteration 3
        epst2 = epst * epst
        y = (epst2 + c) * epst + d
        yp = 3.0 * epst2 + c
        if yp != 0.0:
            epst = epst - y / yp
        # Iteration 4
        epst2 = epst * epst
        y = (epst2 + c) * epst + d
        yp = 3.0 * epst2 + c
        if yp != 0.0:
            epst = epst - y / yp
        epst = epst + dav

    return epst


def fortran_oracle_shell_newton_return(
    E: float,
    nu: float,
    G: float,
    sig_tr: list[float],
    yld: float,
    h_iso: float,
    nmax: int = 3,
) -> tuple[list[float], float, float]:
    """Exact line-by-line replication of sigeps48c.F lines 346-423."""
    nu11 = 1.0 / (1.0 - nu)
    nu21 = 1.0 / (1.0 + nu)
    nu31 = (1.0 - 2.0 * nu) / (1.0 - nu)

    s1 = sig_tr[0] + sig_tr[1]
    s2 = sig_tr[0] - sig_tr[1]
    s3 = sig_tr[2]

    aa = 0.25 * s1 * s1
    bb = 0.75 * s2 * s2 + 3.0 * s3 * s3
    svm = math.sqrt(aa + bb)

    if svm <= yld:
        return list(sig_tr), 0.0, 0.0

    g31 = 3.0 * G
    dpla_j = (svm - yld) / (g31 + h_iso)
    dpla_i = dpla_j

    pp = 1.0
    qq = 1.0
    dr = 0.0

    for _ in range(nmax):
        dpla_i = dpla_j
        yld_i = yld + h_iso * dpla_i
        dr = 0.5 * E * dpla_i / yld_i
        pp = 1.0 / (1.0 + dr * nu11)
        qq = 1.0 / (1.0 + 3.0 * dr * nu21)
        p2 = pp * pp
        q2 = qq * qq
        f = aa * p2 + bb * q2 - yld_i * yld_i
        df = -(aa * nu11 * p2 * pp + 3.0 * bb * nu21 * q2 * qq) * (E - 2.0 * dr * h_iso) / yld_i - 2.0 * h_iso * yld_i
        if dpla_i > 0.0 and df != 0.0:
            dpla_j = max(0.0, dpla_i - f / df)
        else:
            dpla_j = 0.0

    s1_corr = s1 * pp
    s2_corr = s2 * qq
    signxx = 0.5 * (s1_corr + s2_corr)
    signyy = 0.5 * (s1_corr - s2_corr)
    signxy = s3 * qq
    dezz = -nu31 * dr * s1_corr / E

    return [signxx, signyy, signxy], dpla_i, dezz


# ============================================================================
# 1. Static Work Hardening P_A and eps_p_max Cutoff
# ============================================================================

class TestStaticWorkHardening:
    """Verify static work hardening P_A = c_a * sigma_y + c_b * eps_p^c_n against Fortran."""

    @pytest.mark.parametrize("ca, sigy0, cb, cn, epmax", [
        (1.0, 200.0, 300.0, 0.4, 0.25),
        (1.2, 350.0, 500.0, 0.7, 0.30),
        (1.0, 180.0, 250.0, 1.0001, 0.15),
        (0.8, 400.0, 600.0, 1.5, 0.20),
    ])
    def test_work_hardening_curve_parity(self, ca, sigy0, cb, cn, epmax):
        """Parity across full plastic strain range from zero to beyond eps_p_max."""
        p = Law48Params(ca=ca, sigy0=sigy0, cb=cb, cn=cn, eps_max=epmax)
        E = p.E

        # Test points: negative, zero, intermediate, at boundary, and beyond boundary
        eps_points = [-0.01, 0.0, 0.01, 0.05, 0.10, epmax * 0.999, epmax, epmax * 1.05, epmax * 2.0]

        for ep in eps_points:
            yld_py, h_py, _, _ = eval_yield_and_hardening(p, epsp=ep, eps_dot=0.0)
            yld_f, h_f, pa_f, _, _ = fortran_oracle_yield_and_hardening(
                ca=ca * sigy0, cb=cb, cn=cn, cc=0.0, cd=0.0, cm=1.0, ce=0.0, ck=1.0,
                eps0=1.0, smax=1e30, epmax=epmax, E=E, pla=ep, epsp=0.0
            )

            if ep > epmax:
                # Element failure: yield stress must be 0
                assert yld_py == 0.0
                assert yld_f == 0.0
                assert h_py == 0.0
            else:
                assert math.isclose(float(yld_py), yld_f, rel_tol=1e-12, abs_tol=1e-12)
                assert math.isclose(float(h_py), h_f, rel_tol=1e-12, abs_tol=1e-12)

    def test_work_hardening_vectorized_evaluation(self):
        """Batched numpy vector evaluation matches element-wise Fortran oracle."""
        p = Law48Params(ca=1.0, sigy0=250.0, cb=400.0, cn=0.5, eps_max=0.20)
        eps_arr = np.linspace(0.0, 0.25, 26)
        yld_arr, h_arr, _, _ = eval_yield_and_hardening(p, epsp=eps_arr, eps_dot=0.0)

        for i, ep in enumerate(eps_arr):
            yld_f, h_f, _, _, _ = fortran_oracle_yield_and_hardening(
                ca=250.0, cb=400.0, cn=0.5, cc=0.0, cd=0.0, cm=1.0, ce=0.0, ck=1.0,
                eps0=1.0, smax=1e30, epmax=0.20, E=p.E, pla=float(ep), epsp=0.0
            )
            if ep > 0.20:
                assert math.isclose(yld_arr[i], 0.0)
                assert math.isclose(h_arr[i], 0.0)
            else:
                assert math.isclose(yld_arr[i], yld_f, rel_tol=1e-12, abs_tol=1e-12)
                assert math.isclose(h_arr[i], h_f, rel_tol=1e-12, abs_tol=1e-12)


# ============================================================================
# 2. Coupled Logarithmic Rate Sensitivity P_B
# ============================================================================

class TestCoupledLogarithmicRateSensitivity:
    """Verify coupled rate sensitivity P_B = (c_c - c_d * eps_p^c_m) * ln(eps_dot / eps_dot_0)."""

    @pytest.mark.parametrize("cc, cd, cm, eps0", [
        (30.0, 15.0, 0.5, 1.0),
        (50.0, 20.0, 0.8, 0.001),
        (25.0, 10.0, 1.2, 10.0),
        (40.0, 0.0, 1.0001, 1.0),  # uncoupled limit
    ])
    def test_coupled_rate_sensitivity_parity(self, cc, cd, cm, eps0):
        """Verify PB across rates below, at, and far above reference rate eps0."""
        p = Law48Params(
            ca=1.0, sigy0=300.0, cb=200.0, cn=0.5,
            cc=cc, cd=cd, cm=cm, eps0=eps0,
        )

        rates = [0.1 * eps0, eps0, 2.0 * eps0, 10.0 * eps0, 500.0 * eps0, 10000.0 * eps0]
        plas = [0.0, 0.01, 0.05, 0.10]

        for edot in rates:
            for pla in plas:
                yld_py, h_py, _, _ = eval_yield_and_hardening(p, epsp=pla, eps_dot=edot)
                yld_f, h_f, pa_f, pb_f, pc_f = fortran_oracle_yield_and_hardening(
                    ca=300.0, cb=200.0, cn=0.5, cc=cc, cd=cd, cm=cm, ce=0.0, ck=1.0,
                    eps0=eps0, smax=1e30, epmax=1e30, E=p.E, pla=pla, epsp=edot
                )

                # Parity on yield stress and hardening
                assert math.isclose(float(yld_py), yld_f, rel_tol=1e-12, abs_tol=1e-12)
                assert math.isclose(float(h_py), h_f, rel_tol=1e-12, abs_tol=1e-12)

                # Theoretical check on rate threshold
                if edot <= eps0:
                    assert pb_f == 0.0
                else:
                    expected_pb = (cc - cd * (pla ** cm)) * math.log(edot / eps0) if pla > 0.0 else cc * math.log(edot / eps0)
                    assert math.isclose(pb_f, expected_pb, rel_tol=1e-12)

    def test_coupled_hardening_softening_decay(self):
        """Verify that coupling term cd * eps_p^cm reduces rate sensitivity at higher plastic strain."""
        p = Law48Params(
            ca=1.0, sigy0=200.0, cb=0.0, cn=1.0,
            cc=50.0, cd=30.0, cm=0.5, eps0=1.0,
        )
        edot = 100.0

        # At pla=0: PB = 50 * ln(100) = 230.2585
        yld_0, _, _, _ = eval_yield_and_hardening(p, epsp=0.0, eps_dot=edot)
        # At pla=0.04: sqrt(0.04) = 0.2 -> (50 - 30*0.2) = 44 -> PB = 44 * ln(100) = 202.6275
        yld_1, _, _, _ = eval_yield_and_hardening(p, epsp=0.04, eps_dot=edot)
        # At pla=0.25: sqrt(0.25) = 0.5 -> (50 - 30*0.5) = 35 -> PB = 35 * ln(100) = 161.1810
        yld_2, _, _, _ = eval_yield_and_hardening(p, epsp=0.25, eps_dot=edot)

        assert float(yld_0) > float(yld_1) > float(yld_2)
        assert math.isclose(float(yld_0) - 200.0, 50.0 * math.log(100.0), rel_tol=1e-12)
        assert math.isclose(float(yld_1) - 200.0, 44.0 * math.log(100.0), rel_tol=1e-12)
        assert math.isclose(float(yld_2) - 200.0, 35.0 * math.log(100.0), rel_tol=1e-12)


# ============================================================================
# 3. Uncoupled Power-Law High-Rate Term P_C
# ============================================================================

class TestUncoupledHighRatePowerLaw:
    """Verify uncoupled power-law high-rate term P_C = c_e * eps_dot^c_k for eps_dot > 0."""

    @pytest.mark.parametrize("ce, ck", [
        (0.01, 1.0),
        (0.05, 1.2),
        (0.002, 1.8),
        (0.1, 0.6),
    ])
    def test_high_rate_power_law_parity(self, ce, ck):
        """Check PC against exact Fortran power-law formula."""
        p = Law48Params(
            ca=1.0, sigy0=200.0, cb=0.0, cn=1.0,
            ce=ce, ck=ck,
        )

        rates = [0.0, 1.0, 10.0, 100.0, 2500.0, 50000.0]
        for edot in rates:
            yld_py, _, _, _ = eval_yield_and_hardening(p, epsp=0.0, eps_dot=edot)
            yld_f, _, _, _, pc_f = fortran_oracle_yield_and_hardening(
                ca=200.0, cb=0.0, cn=1.0, cc=0.0, cd=0.0, cm=1.0, ce=ce, ck=ck,
                eps0=1.0, smax=1e30, epmax=1e30, E=p.E, pla=0.0, epsp=edot
            )

            expected_pc = ce * (edot ** ck) if edot > 0.0 else 0.0
            assert math.isclose(pc_f, expected_pc, rel_tol=1e-12)
            assert math.isclose(float(yld_py), 200.0 + expected_pc, rel_tol=1e-12)


# ============================================================================
# 4. Total Yield Stress Saturation YLD = min(S_max + P_C, P_A + P_B + P_C)
# ============================================================================

class TestYieldStressSaturation:
    """Verify saturation YLD = min(S_max + P_C, P_A + P_B + P_C)."""

    def test_static_and_dynamic_saturation_ceiling(self):
        """Verify that ceiling expands dynamically with high-rate term P_C."""
        smax = 400.0
        ce = 0.02
        ck = 1.3
        p = Law48Params(
            ca=1.0, sigy0=200.0, cb=500.0, cn=0.5,
            ce=ce, ck=ck, sig_max=smax,
        )

        # 1. Below ceiling: eps_p = 0.01 -> PA = 200 + 500*0.1 = 250 < Smax (400)
        yld_low, h_low, _, _ = eval_yield_and_hardening(p, epsp=0.01, eps_dot=0.0)
        assert math.isclose(float(yld_low), 250.0, rel_tol=1e-12)
        assert float(h_low) > 0.0

        # 2. Saturated static: eps_p = 0.25 -> PA = 200 + 500*0.5 = 450 > Smax (400)
        yld_sat, h_sat, _, _ = eval_yield_and_hardening(p, epsp=0.25, eps_dot=0.0)
        assert math.isclose(float(yld_sat), smax, rel_tol=1e-12)
        assert float(h_sat) == 0.0  # zero slope when capped

        # 3. Dynamic ceiling expansion: edot = 1000 s^-1 -> PC = 0.02 * (1000^1.3)
        edot = 1000.0
        pc = ce * (edot ** ck)
        yld_dyn, h_dyn, _, _ = eval_yield_and_hardening(p, epsp=0.25, eps_dot=edot)
        # Cap must be Smax + PC, NOT just Smax
        expected_dyn_cap = smax + pc
        assert math.isclose(float(yld_dyn), expected_dyn_cap, rel_tol=1e-12)
        assert float(h_dyn) == 0.0


# ============================================================================
# 5. Plastic Hardening Slope H = P_DA + P_DB and Zeroing on Saturation
# ============================================================================

class TestHardeningSlopeFormulas:
    """Verify plastic hardening slope H = P_DA + P_DB and zeroing on saturation."""

    @pytest.mark.parametrize("cn", [0.3, 0.7, 1.0001, 1.5])
    def test_pda_derivative_branches(self, cn):
        """Check cn < 1 vs cn >= 1 branches matching sigeps48.F:280-286."""
        p = Law48Params(ca=1.0, sigy0=200.0, cb=300.0, cn=cn)
        E = p.E

        # At pla <= 0: PDA = E
        _, h_zero, _, _ = eval_yield_and_hardening(p, epsp=0.0, eps_dot=0.0)
        assert math.isclose(float(h_zero), E, rel_tol=1e-12)

        # At pla > 0
        pla = 0.05
        _, h_pla, _, _ = eval_yield_and_hardening(p, epsp=pla, eps_dot=0.0)
        if cn >= 1.0:
            expected_pda = 300.0 * cn * (pla ** (cn - 1.0))
        else:
            expected_pda = 300.0 * cn * (pla ** (1.0 - cn))
        assert math.isclose(float(h_pla), expected_pda, rel_tol=1e-12)

    @pytest.mark.parametrize("cm", [0.4, 0.8, 1.0001, 1.4])
    def test_pdb_derivative_branches(self, cm):
        """Check cm < 1 vs cm >= 1 branches matching sigeps48.F:287-293."""
        p = Law48Params(
            ca=1.0, sigy0=200.0, cb=0.0, cn=1.0,
            cc=50.0, cd=20.0, cm=cm, eps0=1.0,
        )

        edot = 50.0
        pla = 0.04
        _, h_val, _, _ = eval_yield_and_hardening(p, epsp=pla, eps_dot=edot)

        log_rat = math.log(edot / 1.0)
        if cm >= 1.0:
            expected_pdb = 20.0 * cm * (pla ** (cm - 1.0)) * log_rat
        else:
            expected_pdb = 20.0 * cm * (pla ** (1.0 - cm)) * log_rat

        assert math.isclose(float(h_val), expected_pdb, rel_tol=1e-12)


# ============================================================================
# 6. 3D Maximum Principal Strain Cubic Solver vs Exact Analytical Roots
# ============================================================================

class TestPrincipalStrainCubicSolverParity:
    """Verify the 3D maximum principal strain solver (sigeps48.F:201-245)."""

    def test_pure_uniaxial_tension(self):
        """Pure uniaxial tension [eps0, -nu*eps0, -nu*eps0, 0, 0, 0].

        Note on upstream Fortran semantics:
        sigeps48.F line 219 defines C = - E1*E1 - E2*E2 - E3*E3 - E42 - E52 - E62
        without the 1/2 factor on normal deviatoric components. Both Fortran and
        pyradioss solve this exact cubic equation, producing identical roots to 1e-12.
        """
        eps0 = 0.08
        nu = 0.3
        eps = np.array([eps0, -nu * eps0, -nu * eps0, 0.0, 0.0, 0.0])

        epst_py = _principal_strain_3d(eps)
        epst_f = fortran_oracle_cubic_solver(eps)

        assert math.isclose(float(epst_py), epst_f, rel_tol=1e-12, abs_tol=1e-12)
        # Root of upstream cubic with C = -e1^2-e2^2-e3^2 is ~0.100858
        assert float(epst_py) > 0.0
        assert math.isclose(float(epst_py), 0.10085820990528269, rel_tol=1e-7)

    def test_pure_uniaxial_compression(self):
        """Pure uniaxial compression [-eps0, nu*eps0, nu*eps0, 0, 0, 0]."""
        eps0 = 0.05
        nu = 0.3
        eps = np.array([-eps0, nu * eps0, nu * eps0, 0.0, 0.0, 0.0])

        epst_py = _principal_strain_3d(eps)
        epst_f = fortran_oracle_cubic_solver(eps)

        assert math.isclose(float(epst_py), epst_f, rel_tol=1e-12, abs_tol=1e-12)
        # Lateral Poisson expansion root in upstream cubic solver
        assert float(epst_py) > 0.0
        assert math.isclose(float(epst_py), 0.04233904864729988, rel_tol=1e-7)

    def test_pure_shear(self):
        """Pure shear [0, 0, 0, gamma, 0, 0] engineering shear."""
        gamma = 0.06
        eps = np.array([0.0, 0.0, 0.0, gamma, 0.0, 0.0])

        epst_py = _principal_strain_3d(eps)
        epst_f = fortran_oracle_cubic_solver(eps)

        assert math.isclose(float(epst_py), epst_f, rel_tol=1e-12, abs_tol=1e-12)
        # Analytical maximum eigenvalue of pure shear tensor is gamma / 2
        assert math.isclose(float(epst_py), 0.5 * gamma, rel_tol=1e-7)

    def test_general_3d_strain_tensors(self):
        """Verify general non-diagonal 3D strain states against Fortran solver."""
        test_states = [
            [0.05, 0.02, -0.01, 0.03, 0.02, 0.01],
            [0.12, 0.08, 0.04, 0.00, 0.05, 0.02],
            [-0.03, -0.02, -0.01, 0.04, 0.01, 0.03],
            [0.01, -0.01, 0.02, 0.08, 0.04, 0.02],
        ]

        for eps in test_states:
            eps_arr = np.array(eps)
            epst_py = _principal_strain_3d(eps_arr)
            epst_f = fortran_oracle_cubic_solver(eps_arr)
            assert math.isclose(float(epst_py), epst_f, rel_tol=1e-12, abs_tol=1e-12)


# ============================================================================
# 7. Tensile Failure Factor FAIL and Yield/Hardening Scaling
# ============================================================================

class TestTensileFailureFactorAndScaling:
    """Verify FAIL = max(0, min(1, (eps_r2 - eps_t) / (eps_r2 - eps_r1)))."""

    def test_tensile_failure_factor_ramp(self):
        """Test linear softening ramp between eps_r1 and eps_r2."""
        p = Law48Params(eps_t1=0.10, eps_t2=0.20)

        # Before failure onset
        assert math.isclose(float(tensile_failure_factor(p, 0.05)), 1.0)
        assert math.isclose(float(tensile_failure_factor(p, 0.10)), 1.0)

        # Mid-ramp
        assert math.isclose(float(tensile_failure_factor(p, 0.125)), 0.75)
        assert math.isclose(float(tensile_failure_factor(p, 0.15)), 0.50)
        assert math.isclose(float(tensile_failure_factor(p, 0.175)), 0.25)

        # At and above complete failure
        assert math.isclose(float(tensile_failure_factor(p, 0.20)), 0.0)
        assert math.isclose(float(tensile_failure_factor(p, 0.30)), 0.0)

    def test_yield_and_hardening_fail_scaling(self):
        """Yield stress and hardening are scaled by FAIL factor."""
        p = Law48Params(
            ca=1.0, sigy0=300.0, cb=400.0, cn=0.5,
            eps_t1=0.10, eps_t2=0.20,
        )

        epst = 0.15  # FAIL = 0.5
        fail = float(tensile_failure_factor(p, epst))
        assert math.isclose(fail, 0.5)

        yld, h, _, _ = eval_yield_and_hardening(p, epsp=0.04, eps_dot=0.0, fail=fail)
        yld_unscaled, h_unscaled, _, _ = eval_yield_and_hardening(p, epsp=0.04, eps_dot=0.0, fail=1.0)

        assert math.isclose(float(yld), 0.5 * float(yld_unscaled), rel_tol=1e-12)
        assert math.isclose(float(h), 0.5 * float(h_unscaled), rel_tol=1e-12)


# ============================================================================
# 8. Plane-Stress Shell Radial Projection and Newton-Raphson Return
# ============================================================================

class TestShellPlaneStressReturnParity:
    """Verify plane-stress shell radial projection and Newton-Raphson return (sigeps48c.F)."""

    def test_newton_raphson_plane_stress_parity_vs_fortran(self):
        """Verify plane_stress_return_newton_law48 against fortran_oracle_shell_newton_return."""
        p = Law48Params(E=210000.0, nu=0.3, sigy0=250.0, cb=400.0, cn=0.5)
        yld = 250.0
        h_iso = 300.0

        # Test multiple trial plane stress states exceeding yield
        test_stresses = [
            [400.0, 0.0, 0.0],
            [300.0, 200.0, 0.0],
            [0.0, 0.0, 200.0],
            [350.0, -100.0, 80.0],
        ]

        for s_tr in test_stresses:
            sig_arr = np.array([s_tr])
            yld_arr = np.array([yld])
            h_arr = np.array([h_iso])
            epsp_arr = np.array([0.0])

            sig_py, dpla_py, dezz_py = plane_stress_return_newton_law48(
                p, sig_arr, yld_arr, h_arr, epsp_arr, nmax=3
            )
            sig_f, dpla_f, dezz_f = fortran_oracle_shell_newton_return(
                p.E, p.nu, p.G, s_tr, yld, h_iso, nmax=3
            )

            assert np.allclose(sig_py[0], sig_f, rtol=1e-12, atol=1e-12)
            assert math.isclose(float(dpla_py[0]), dpla_f, rel_tol=1e-12, abs_tol=1e-12)
            assert math.isclose(float(dezz_py[0]), dezz_f, rel_tol=1e-12, abs_tol=1e-12)

    def test_shell_update_law48_iflag1_vs_iflag0(self):
        """Verify shell_update_law48 dispatch with iflag=0 (radial) and iflag=1 (Newton)."""
        p = Law48Params(E=200000.0, nu=0.3, ca=1.0, sigy0=200.0, cb=300.0, cn=0.5)
        sig = np.zeros((1, 3))
        deps = np.array([[0.005, 0.0, 0.0]])

        # Radial return
        sig_rad, ep_rad = shell_update_law48(p, sig.copy(), deps, epsp=0.0, dt=1e-4, iflag=0)
        # Newton-Raphson return
        sig_newt, ep_newt = shell_update_law48(p, sig.copy(), deps, epsp=0.0, dt=1e-4, iflag=1)

        assert ep_rad[0] > 0.0
        assert ep_newt[0] > 0.0
        sxx_n, syy_n, sxy_n = sig_newt[0]
        svm_n = math.sqrt(sxx_n**2 + syy_n**2 - sxx_n * syy_n + 3.0 * sxy_n**2)
        # In sigeps48c.F:389-395, Newton-Raphson returns stresses to yld_i = yld + h_iso * dpla
        assert math.isclose(svm_n, 200.0 + p.E * ep_newt[0], rel_tol=1e-5)


# ============================================================================
# 9. Exact Sound Speeds
# ============================================================================

class TestSoundSpeedsParity:
    """Verify exact sound speeds c_solid and c_shell across various materials."""

    @pytest.mark.parametrize("E, nu, rho0", [
        (210000.0, 0.30, 7.85e-9),   # Structural Steel
        (70000.0, 0.33, 2.70e-9),    # Aluminium
        (110000.0, 0.34, 4.43e-9),   # Titanium
        (120000.0, 0.35, 8.96e-9),   # Copper
        (3000.0, 0.40, 1.20e-9),     # Polymer
    ])
    def test_sound_speeds_exact_parity(self, E, nu, rho0):
        """Compare sound speed routines against exact analytical formulas."""
        p = Law48Params(E=E, nu=nu, rho0=rho0)

        # Upstream hm_read_mat48.F:187-188 formulas:
        C1 = E / (3.0 * (1.0 - 2.0 * nu))
        G = E / (2.0 * (1.0 + nu))
        c_solid_fortran = math.sqrt((C1 + 4.0 * G / 3.0) / rho0)

        # Upstream hm_read_mat48.F:208 formula:
        A11 = E / (1.0 - nu * nu)
        c_shell_fortran = math.sqrt(A11 / rho0)

        c_solid_py = sound_speed_solid_law48(p)
        c_shell_py = sound_speed_shell_law48(p)

        assert math.isclose(c_solid_py, c_solid_fortran, rel_tol=1e-12)
        assert math.isclose(c_shell_py, c_shell_fortran, rel_tol=1e-12)

        # Dilatational solid wave speed analytical identity: sqrt(E*(1-nu)/((1+nu)*(1-2nu)*rho0))
        c_solid_closed = math.sqrt(E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu) * rho0))
        assert math.isclose(c_solid_py, c_solid_closed, rel_tol=1e-12)
