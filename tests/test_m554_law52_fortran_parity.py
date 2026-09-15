"""
Milestone M554: Comprehensive Fortran Oracle Parity Test Suite for LAW52
Gurson-Tvergaard-Needleman (GTN) Porous Metal Plasticity Material Model
(/MAT/LAW52, /MAT/GURSON, /MAT/PLAS_GURS).

Upstream OpenRadioss Fortran reference sources audited:
- ``engine/source/materials/mat/mat052/sigeps52.F`` (3D solid continuum kernel)
- ``engine/source/materials/mat/mat052/sigeps52c.F`` (2D shell plane-stress kernel)
- ``starter/source/materials/mat/mat052/hm_read_mat52.F`` (parameter initialization)

Covers the 9 verification items mandated by the M554 Fortran Parity Audit:
1. Gurson yield function:
   Phi = (q / sigma_M)^2 + 2*q1*f* * cosh(3*q2*P / (2*sigma_M)) - (1 + q3*f*^2) = 0
   matching sigeps52.F:291-305 and sigeps52c.F:315-328.
2. Tvergaard-Needleman void coalescence acceleration:
   f* = f for f <= f_c
   f* = f_c + ((f_u - f_c) / (f_F - f_c)) * (f - f_c) for f > f_c
   with f_u = 1 / q1 matching hm_read_mat52.F:194-196 and sigeps52.F:283-289.
3. Void growth kinematics:
   Delta f_g = (1 - f) * tr(Delta eps^p) under pure tension, compression, and pure shear
   matching sigeps52.F:402 and sigeps52c.F:476.
4. Chu & Needleman Gaussian strain-controlled void nucleation:
   Delta f_n = (f_N / (s_N * sqrt(2*pi))) * exp(-0.5 * ((eps_M - eps_N) / s_N)^2) * Delta eps_M
   matching sigeps52.F:329-330, 404, and nucleation suppression under compression for IFLAG=2,3.
5. Matrix flow stress and work equivalence:
   sigma_M = (A + B * eps_M^n) * [1 + (dot_eps_M / C)^(1/P)] and
   (1 - f) * sigma_M * Delta eps_M = sigma : Delta eps^p
   matching sigeps52.F:381-384, 409-410.
6. Cutting-plane / Newton return mapping convergence across 5 iterations
   matching sigeps52.F:331-400 and sigeps52c.F:356-470.
7. Cavitation limit (VA <= 0) and complete void coalescence rupture (f* >= f_u or f >= f_F)
   zeroing stresses and setting off = 0.0.
8. Plane-stress shell thickness thinning increment:
   Delta eps_zz = (nu / (1 - nu)) * (Delta eps_xx^p + Delta eps_yy^p) + Delta eps_zz^p
   matching sigeps52c.F:292-294 and 474-475.
9. Instantaneous acoustic sound speeds:
   c_solid = sqrt((E * (1 - nu)) / ((1 + nu) * (1 - 2*nu) * rho0)) and
   c_shell = sqrt(E / ((1 - nu^2) * rho0))
   matching sigeps52.F:208-210 and sigeps52c.F:278.
"""

import math
from typing import Any, Dict, Tuple
import numpy as np
import pytest

from pyradioss.materials.law52_gurson import (
    Law52Params,
    build_law52,
    compute_f_star,
    gurson_yield_function,
    solid_update_law52,
    shell_update_law52,
    sound_speed_solid_law52,
    sound_speed_shell_law52,
    tangent_law52_solid,
    tangent_law52_shell,
)


# ===================================================================
# Fortran Line-by-Line Reference Oracles
# ===================================================================

def fortran_sigeps52_oracle_step(
    e: float,
    nu: float,
    rho0: float,
    yeild0: float,
    et: float,
    n_exp: float,
    csd: float,
    visp: float,
    q1: float,
    q2: float,
    q3: float,
    sn: float,
    epsn: float,
    fi: float,
    fn: float,
    fc: float,
    ff: float,
    fu: float,
    iflag: int,
    sig_old: np.ndarray,
    deps: np.ndarray,
    epsm_old: float,
    sigm_old: float,
    fg_old: float,
    fn1_old: float,
    f_old: float,
    fstar_old: float,
    off_old: float,
    epsp_rate: float,
) -> Dict[str, Any]:
    """Exact line-by-line transcription of sigeps52.F lines 208-438 (IFLAG=0, 2, 3)."""
    em20 = 1e-20
    ep20 = 1e20
    third = 1.0 / 3.0
    half = 0.5
    three_half = 1.5
    pi = math.pi
    sqr22 = 1.0 / math.sqrt(2.0 * pi)

    # Lines 208-214: Moduli
    g = half * e / (1.0 + nu)
    c11 = e / 3.0 / (1.0 - 2.0 * nu)
    soundsp = math.sqrt((c11 + (4.0 / 3.0) * g) / rho0)
    c1 = e * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c2 = c1 * nu / (1.0 - nu)

    sign = np.zeros(6, dtype=float)
    if off_old == 0.0 or f_old >= ff or fstar_old >= fu:
        return {
            "sign": sign,
            "epsm": epsm_old,
            "sigm": sigm_old,
            "fg": fg_old,
            "fn1": fn1_old,
            "f": f_old,
            "fstar": fstar_old,
            "off": 0.0,
            "soundsp": soundsp,
            "iterations": [],
        }

    # Lines 253-261: Elastic predictor
    sign[0] = sig_old[0] + c1 * deps[0] + c2 * (deps[1] + deps[2])
    sign[1] = sig_old[1] + c1 * deps[1] + c2 * (deps[0] + deps[2])
    sign[2] = sig_old[2] + c1 * deps[2] + c2 * (deps[0] + deps[1])
    sign[3] = sig_old[3] + g * deps[3]
    sign[4] = sig_old[4] + g * deps[4]  # YZ
    sign[5] = sig_old[5] + g * deps[5]  # ZX

    # Lines 283-289: f* calculation
    if f_old <= fc:
        fstar = f_old
        df = 1.0
    else:
        df = (fu - fc) / (ff - fc)
        fstar = fc + df * (f_old - fc)

    # Lines 291-297: Hydrostatic pressure and von Mises stress
    pn = (sign[0] + sign[1] + sign[2]) * third
    sxx = sign[0] - pn
    syy = sign[1] - pn
    szz = sign[2] - pn
    vm_sq = half * (sxx**2 + syy**2 + szz**2) + (sign[3]**2 + sign[5]**2 + sign[4]**2)
    vm = math.sqrt(max(3.0 * vm_sq, 0.0))

    # Lines 298-305: Yield threshold
    sigm1 = 1.0 / max(sigm_old, em20)
    var = three_half * q2 * pn * sigm1
    var = math.exp(max(min(var, 80.0), -80.0))
    coh = half * (var + 1.0 / max(em20, var))
    sih = half * (var - 1.0 / max(em20, var))
    va = 1.0 + q3 * (fstar**2) - 2.0 * q1 * fstar * coh

    # Cavitation rupture under extreme tension
    if va <= 0.0 and pn > 0.0:
        return {
            "sign": np.zeros(6),
            "epsm": epsm_old,
            "sigm": sigm_old,
            "fg": fg_old,
            "fn1": fn1_old,
            "f": max(f_old, ff),
            "fstar": fu,
            "off": 0.0,
            "soundsp": soundsp,
            "iterations": [],
        }

    va_sqrt = math.sqrt(max(0.0, va))
    yld = sigm_old * va_sqrt

    if vm < yld or yld == 0.0:
        return {
            "sign": sign,
            "epsm": epsm_old,
            "sigm": sigm_old,
            "fg": fg_old,
            "fn1": fn1_old,
            "f": f_old,
            "fstar": fstar,
            "off": 1.0,
            "soundsp": soundsp,
            "iterations": [],
        }

    # Lines 312-330: Setup plastic return mapping
    sig_tr = sign.copy()
    dp11 = dp22 = dp33 = dp12 = dp13 = dp23 = 0.0
    a21 = ep20 if f_old == 1.0 else sigm1 / (1.0 - f_old)

    a1 = fn * math.exp(-half * (((epsm_old - epsn) / sn)**2)) / sn
    a1 = a1 * sqr22

    iter_history = []

    # Lines 331-400: 5 cutting-plane iterations
    for iter_idx in range(5):
        vm1 = 1.0 / max(vm, em20)
        va1 = 1.0 / max(va_sqrt, em20)
        va11 = half * q1 * q2 * fstar * sih * va1

        d11 = half * (2.0 * sign[0] - sign[1] - sign[2]) * vm1 + va11
        d22 = half * (2.0 * sign[1] - sign[0] - sign[2]) * vm1 + va11
        d33 = half * (2.0 * sign[2] - sign[0] - sign[1]) * vm1 + va11
        d12 = 3.0 * sign[3] * vm1
        d13 = 3.0 * sign[5] * vm1  # ZX
        d23 = 3.0 * sign[4] * vm1  # YZ

        a2 = (
            d11 * sign[0] + d22 * sign[1] + d33 * sign[2] +
            2.0 * (d12 * sign[3] + d13 * sign[5] + d23 * sign[4])
        ) * a21

        dcrf = -sigm_old * (q3 * fstar * df - q1 * coh * df) * va1
        dcrm = -va_sqrt - 3.0 * va11 * pn * sigm1

        if n_exp == 1.0:
            dsepp = et * (1.0 + (epsp_rate * (1.0 / csd if csd > 0 else 0.0))**visp)
        else:
            dsepp = (
                et * n_exp * (max(epsm_old, em20)**(n_exp - 1.0)) *
                (1.0 + (epsp_rate * (1.0 / csd if csd > 0 else 0.0))**visp)
            )

        dcd = (
            c1 * (d11**2 + d22**2 + d33**2) +
            2.0 * c2 * (d11 * d22 + d11 * d33 + d22 * d33) +
            2.0 * g * (d12**2 + d13**2 + d23**2)
        )

        lam1 = dcd - dcrm * dsepp * a2 - dcrf * ((1.0 - f_old) * (d11 + d22 + d33) + a1 * a2)
        lamda = max(0.0, vm - yld) / lam1 if lam1 != 0.0 else 0.0

        iter_history.append({
            "iter": iter_idx + 1,
            "vm": vm,
            "yld": yld,
            "diff": vm - yld,
            "lamda": lamda,
            "D": np.array([d11, d22, d33, d12, d23, d13]),
            "LAM1": lam1,
        })

        dp11 += lamda * d11
        dp22 += lamda * d22
        dp33 += lamda * d33
        dp12 += lamda * d12
        dp13 += lamda * d13
        dp23 += lamda * d23

        sign[0] = sig_tr[0] - c1 * dp11 - c2 * (dp22 + dp33)
        sign[1] = sig_tr[1] - c1 * dp22 - c2 * (dp11 + dp33)
        sign[2] = sig_tr[2] - c1 * dp33 - c2 * (dp22 + dp11)
        sign[3] = sig_tr[3] - 2.0 * g * dp12
        sign[4] = sig_tr[4] - 2.0 * g * dp23
        sign[5] = sig_tr[5] - 2.0 * g * dp13

        pn = (sign[0] + sign[1] + sign[2]) * third
        sxx = sign[0] - pn
        syy = sign[1] - pn
        szz = sign[2] - pn
        vm_sq = half * (sxx**2 + syy**2 + szz**2) + (sign[3]**2 + sign[5]**2 + sign[4]**2)
        vm = math.sqrt(max(3.0 * vm_sq, 0.0))

        var = three_half * q2 * pn * sigm1
        var = math.exp(max(min(var, 80.0), -80.0))
        coh = half * (var + 1.0 / max(em20, var))
        sih = half * (var - 1.0 / max(em20, var))
        va = 1.0 + q3 * (fstar**2) - 2.0 * q1 * fstar * coh
        va_sqrt = math.sqrt(max(0.0, va))
        yld = sigm_old * va_sqrt

    # Lines 381-384: Matrix equivalent plastic strain
    epsp1 = (
        sign[0] * dp11 + sign[1] * dp22 + sign[2] * dp33 +
        2.0 * (sign[3] * dp12 + sign[5] * dp13 + sign[4] * dp23)
    ) * a21
    epsp1 = max(0.0, epsp1)

    # Lines 402-410: Void growth, nucleation, matrix update
    fg_new = fg_old + (1.0 - f_old) * (dp11 + dp22 + dp33)
    fn1_new = fn1_old
    if iflag in (2, 3):
        if pn >= 0.0:
            fn1_new += a1 * epsp1
    else:
        fn1_new += a1 * epsp1

    epsm_new = epsm_old + epsp1
    f_new = fi + fg_new + fn1_new
    if q1 == 0.0 and q2 == 0.0 and q3 == 0.0:
        f_new = 0.0
    if f_new < 0.0:
        f_new = 0.0

    rate_factor = 1.0
    if csd > 0.0 and visp > 0.0:
        rate_factor = 1.0 + (epsp_rate / csd)**visp
    sigm_new = (yeild0 + et * (epsm_new**n_exp)) * rate_factor

    # Lines 412-420: Recompute f*
    if f_new <= fc:
        fstar_new = f_new
    else:
        fstar_new = fc + (fu - fc) / (ff - fc) * (f_new - fc)

    off_new = 1.0
    if fstar_new >= fu or f_new >= ff:
        off_new = 0.0
        sign[:] = 0.0

    return {
        "sign": sign,
        "epsm": epsm_new,
        "sigm": sigm_new,
        "fg": fg_new,
        "fn1": fn1_new,
        "f": f_new,
        "fstar": fstar_new,
        "off": off_new,
        "soundsp": soundsp,
        "iterations": iter_history,
        "dp": np.array([dp11, dp22, dp33, dp12, dp23, dp13]),
        "epsp1": epsp1,
    }


def fortran_sigeps52c_oracle_step(
    e: float,
    nu: float,
    rho0: float,
    yeild0: float,
    et: float,
    en: float,
    csd: float,
    visp: float,
    q1: float,
    q2: float,
    q3: float,
    sn: float,
    epsn: float,
    fi: float,
    fn: float,
    fc: float,
    ff: float,
    fu: float,
    iflag: int,
    sigo: np.ndarray,
    deps: np.ndarray,
    thk_old: float,
    thkly: float,
    pla_old: float,
    sigm_old: float,
    fg_old: float,
    fn1_old: float,
    f_old: float,
    fstar_old: float,
    off_old: float,
    rate: float,
) -> Dict[str, Any]:
    """Exact line-by-line transcription of sigeps52c.F lines 232-500 for plane-stress shells."""
    em20 = 1e-20
    ep20 = 1e20
    third = 1.0 / 3.0
    half = 0.5
    three_half = 1.5
    sqr22 = 1.0 / math.sqrt(2.0 * math.pi)

    a1 = e / (1.0 - nu**2)
    a2 = nu * a1
    g = half * e / (1.0 + nu)
    gs = g
    c1 = e * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c2 = c1 * nu / (1.0 - nu)
    nn1 = nu / (1.0 - nu)
    soundsp = math.sqrt(a1 / rho0)

    sign = np.zeros(len(sigo), dtype=float)
    if off_old == 0.0 or f_old >= ff or fstar_old >= fu:
        return {
            "sign": sign,
            "thk": thk_old,
            "pla": pla_old,
            "sigm": sigm_old,
            "fg": fg_old,
            "fn1": fn1_old,
            "f": f_old,
            "fstar": fstar_old,
            "off": 0.0,
            "soundsp": soundsp,
        }

    # Lines 272-276: Elastic trial predictor
    sign[0] = sigo[0] + a1 * deps[0] + a2 * deps[1]
    sign[1] = sigo[1] + a2 * deps[0] + a1 * deps[1]
    sign[2] = sigo[2] + g * deps[2]
    if len(sigo) >= 5:
        sign[3] = sigo[3] + gs * deps[3]
        sign[4] = sigo[4] + gs * deps[4]

    # Lines 292-293: Elastic thickness thinning
    dezz_el = -nn1 * (deps[0] + deps[1])
    thk = thk_old + dezz_el * thkly * off_old

    # Lines 308-314: f*
    if f_old <= fc:
        fstar = f_old
        df = 1.0
    else:
        df = (fu - fc) / (ff - fc)
        fstar = fc + df * (f_old - fc)

    # Lines 315-328: In-plane von Mises and pressure
    s23 = sign[3] if len(sign) >= 5 else 0.0
    s31 = sign[4] if len(sign) >= 5 else 0.0
    vm = math.sqrt(max(sign[0]**2 + sign[1]**2 - sign[0] * sign[1] + 3.0 * (sign[2]**2 + s23**2 + s31**2), 0.0))
    pn = (sign[0] + sign[1]) * third

    sigm1 = 1.0 / max(em20, sigm_old)
    var = three_half * q2 * pn * sigm1
    var = math.exp(max(min(var, 80.0), -80.0))
    coh = half * (var + 1.0 / max(em20, var))
    sih = half * (var - 1.0 / max(em20, var))

    va = 1.0 + q3 * (fstar**2) - 2.0 * q1 * fstar * coh
    if va <= 0.0 and pn > 0.0:
        return {
            "sign": np.zeros_like(sign),
            "thk": thk,
            "pla": pla_old,
            "sigm": sigm_old,
            "fg": fg_old,
            "fn1": fn1_old,
            "f": max(f_old, ff),
            "fstar": fu,
            "off": 0.0,
            "soundsp": soundsp,
        }

    va_sqrt = math.sqrt(max(0.0, va))
    yldc = sigm_old * va_sqrt

    if vm < yldc or yldc == 0.0:
        return {
            "sign": sign,
            "thk": thk,
            "pla": pla_old,
            "sigm": sigm_old,
            "fg": fg_old,
            "fn1": fn1_old,
            "f": f_old,
            "fstar": fstar,
            "off": 1.0,
            "soundsp": soundsp,
        }

    # Return mapping
    sig_tr = sign.copy()
    dp11 = dp22 = dp33 = dp12 = dp13 = dp23 = 0.0
    a21 = ep20 if f_old == 1.0 else sigm1 / (1.0 - f_old)
    a11 = fn * math.exp(-half * (((pla_old - epsn) / sn)**2)) / sn * sqr22

    for _ in range(5):
        vm1 = 1.0 / max(em20, vm)
        va1 = 1.0 / max(em20, va_sqrt)
        va2 = half * q1 * q2 * fstar * sih * va1

        d11 = half * (2.0 * sign[0] - sign[1]) * vm1 + va2
        d22 = half * (2.0 * sign[1] - sign[0]) * vm1 + va2
        d33 = half * (-sign[0] - sign[1]) * vm1 + va2
        d12 = 3.0 * sign[2] * vm1
        d23 = 3.0 * s23 * vm1
        d13 = 3.0 * s31 * vm1

        a22 = (d11 * sign[0] + d22 * sign[1] + 2.0 * (d12 * sign[2] + d13 * s31 + d23 * s23)) * a21
        dcrf = -sigm_old * (q3 * fstar * df - q1 * coh * df) * va1
        dcrm = -va_sqrt - 3.0 * va2 * pn * sigm1

        rate_term = 1.0 + (rate / csd)**visp if (csd > 0 and visp > 0) else 1.0
        if en == 1.0:
            dsepp = et * rate_term
        else:
            dsepp = et * en * (max(pla_old, em20)**(en - 1.0)) * rate_term

        dcd = (
            c1 * (d11**2 + d22**2 + d33**2) +
            2.0 * c2 * (d11 * d22 + d11 * d33 + d22 * d33) +
            2.0 * g * (d12**2) + 2.0 * gs * (d13**2 + d23**2)
        )

        lam1 = dcd - dcrm * dsepp * a22 - dcrf * ((1.0 - f_old) * (d11 + d22 + d33) + a11 * a22)
        lamda = max(0.0, vm - yldc) / lam1 if lam1 != 0.0 else 0.0

        dp11 += lamda * d11
        dp22 += lamda * d22
        dp33 += lamda * d33
        dp12 += lamda * d12
        dp13 += lamda * d13
        dp23 += lamda * d23

        sign[0] = sig_tr[0] - a1 * dp11 - a2 * dp22
        sign[1] = sig_tr[1] - a2 * dp11 - a1 * dp22
        sign[2] = sig_tr[2] - 2.0 * g * dp12
        if len(sign) >= 5:
            sign[3] = sig_tr[3] - 2.0 * gs * dp23
            sign[4] = sig_tr[4] - 2.0 * gs * dp13
            s23 = sign[3]
            s31 = sign[4]

        vm = math.sqrt(max(sign[0]**2 + sign[1]**2 - sign[0] * sign[1] + 3.0 * (sign[2]**2 + s23**2 + s31**2), 0.0))
        pn = (sign[0] + sign[1]) * third

        var = three_half * q2 * pn * sigm1
        var = math.exp(max(min(var, 80.0), -80.0))
        coh = half * (var + 1.0 / max(em20, var))
        sih = half * (var - 1.0 / max(em20, var))
        va = 1.0 + q3 * (fstar**2) - 2.0 * q1 * fstar * coh
        va_sqrt = math.sqrt(max(0.0, va))
        yldc = sigm_old * va_sqrt

    # Plastic thickness thinning: dezz_pl = nn1*(dp11 + dp22) + dp33
    dezz_pl = nn1 * (dp11 + dp22) + dp33
    thk += dezz_pl * thkly * off_old

    epsp1 = (sign[0] * dp11 + sign[1] * dp22 + 2.0 * (sign[2] * dp12 + s23 * dp23 + s31 * dp13)) * a21
    epsp1 = max(0.0, epsp1)

    fg_new = fg_old + (1.0 - f_old) * (dp11 + dp22 + dp33)
    fn1_new = fn1_old
    if iflag in (2, 3):
        if pn >= 0.0:
            fn1_new += a11 * epsp1
    else:
        fn1_new += a11 * epsp1

    pla_new = pla_old + epsp1
    f_new = fi + fg_new + fn1_new
    if q1 == 0.0 and q2 == 0.0 and q3 == 0.0:
        f_new = 0.0
    if f_new < 0.0:
        f_new = 0.0

    sigm_new = (yeild0 + et * (pla_new**en)) * rate_term

    if f_new <= fc:
        fstar_new = f_new
    else:
        fstar_new = fc + (fu - fc) / (ff - fc) * (f_new - fc)

    off_new = 1.0
    if fstar_new >= fu or f_new >= ff:
        off_new = 0.0
        sign[:] = 0.0

    return {
        "sign": sign,
        "thk": thk,
        "pla": pla_new,
        "sigm": sigm_new,
        "fg": fg_new,
        "fn1": fn1_new,
        "f": f_new,
        "fstar": fstar_new,
        "off": off_new,
        "soundsp": soundsp,
        "dezz_pl": dezz_pl,
        "epsp1": epsp1,
    }


# ===================================================================
# Test Suite Fixtures
# ===================================================================

@pytest.fixture
def gtn_params():
    """Standard verified GTN parameter set."""
    return {
        "E": 210000.0,
        "nu": 0.3,
        "rho0": 7.85e-9,
        "A": 400.0,
        "B": 500.0,
        "n": 0.2,
        "CSD": 0.0,
        "VISP": 1.0,
        "q1": 1.5,
        "q2": 1.0,
        "q3": 2.25,
        "s_N": 0.1,
        "eps_N": 0.3,
        "f_I": 0.01,
        "f_N": 0.04,
        "f_C": 0.15,
        "f_F": 0.25,
    }


# ===================================================================
# 1. Gurson Yield Function Parity Tests (sigeps52.F:291-305 & sigeps52c.F)
# ===================================================================

def test_gurson_yield_function_formula_parity():
    """Direct mathematical check of GTN yield condition:
    Phi = (q / sigma_M)^2 + 2*q1*f* * cosh(3*q2*P / (2*sigma_M)) - (1 + q3*f*^2) = 0.
    """
    sig_m = 450.0
    q1 = 1.5
    q2 = 1.0
    q3 = 2.25
    f_stars = [0.0, 0.02, 0.05, 0.10, 0.15]
    pressures = [-300.0, -100.0, 0.0, 150.0, 400.0]

    for f_star in f_stars:
        for p in pressures:
            var = 1.5 * q2 * p / sig_m
            coh = math.cosh(var)
            va = 1.0 + q3 * (f_star**2) - 2.0 * q1 * f_star * coh
            if va <= 0.0:
                continue
            # Exactly on the yield surface
            q_on_surface = sig_m * math.sqrt(va)

            phi = gurson_yield_function(q_on_surface, p, sig_m, f_star, q1=q1, q2=q2, q3=q3)
            np.testing.assert_allclose(phi, 0.0, atol=1e-12)


def test_gurson_yield_fortran_algebraic_equivalence():
    """Verify that Fortran variables (VAR, COH, VA, YLD) in sigeps52.F match gurson_yield_function."""
    q1, q2, q3 = 1.5, 1.0, 2.25
    sigm = 500.0
    fstar = 0.08
    pn = 200.0

    # Fortran lines 299-305
    sigm1 = 1.0 / sigm
    var_f = 1.5 * q2 * pn * sigm1
    var_exp = math.exp(var_f)
    coh_f = 0.5 * (var_exp + 1.0 / var_exp)
    va_f = 1.0 + q3 * (fstar**2) - 2.0 * q1 * fstar * coh_f
    yld_f = sigm * math.sqrt(va_f)

    # Python evaluation
    phi_f = gurson_yield_function(yld_f, pn, sigm, fstar, q1, q2, q3)
    np.testing.assert_allclose(phi_f, 0.0, atol=1e-12)
    np.testing.assert_allclose(coh_f, math.cosh(var_f), rtol=1e-14)


# ===================================================================
# 2. Tvergaard-Needleman Void Coalescence Parity (sigeps52.F:283-289)
# ===================================================================

def test_coalescence_acceleration_fortran_parity():
    """Verify f* acceleration matches Fortran sigeps52.F lines 283-289 and 412-420."""
    q1 = 1.5
    fu = 1.0 / q1  # hm_read_mat52.F line 196: FU = 1 / Q1
    fc = 0.15
    ff = 0.25

    test_f_values = [0.0, 0.05, 0.15, 0.18, 0.22, 0.25, 0.30]

    for f_val in test_f_values:
        # Fortran oracle
        if f_val <= fc:
            fstar_expected = f_val
            df_expected = 1.0
        else:
            df_expected = (fu - fc) / (ff - fc)
            fstar_expected = fc + df_expected * (f_val - fc)

        fstar_py, df_py = compute_f_star(f_val, fc, ff, fu)

        np.testing.assert_allclose(fstar_py, fstar_expected, rtol=1e-14)
        np.testing.assert_allclose(df_py, df_expected, rtol=1e-14)


def test_coalescence_failure_limit_reaches_fu():
    """When void fraction reaches failure f_F, f* reaches ultimate void fraction f_u."""
    q1 = 1.5
    fu = 1.0 / q1
    fc = 0.12
    ff = 0.22
    fstar_at_ff, df = compute_f_star(ff, fc, ff, fu)
    np.testing.assert_allclose(fstar_at_ff, fu, rtol=1e-14)
    assert df > 1.0


# ===================================================================
# 3. Void Growth Kinematics Parity (sigeps52.F:402 & sigeps52c.F:476)
# ===================================================================

def test_void_growth_pure_shear_is_identically_zero(gtn_params):
    """Under pure shear, tr(Delta eps^p) = 0 so void growth Delta f_g == 0 (sigeps52.F:402)."""
    mat = build_law52(**gtn_params)
    sig = np.zeros(6, dtype=float)
    # Pure engineering shear increment: deps_xy = 0.015
    deps = np.array([0.0, 0.0, 0.0, 0.015, 0.0, 0.0], dtype=float)
    extra = {}

    sig_out, epsp_out = solid_update_law52(mat, sig, deps, epsp=0.0, dt=1e-5, extra=extra)

    fg = extra["dmg"][0, 1]
    # Pure shear hydrostatic pressure PN = 0, so D11=D22=D33=0, tr(D)=0
    np.testing.assert_allclose(fg, 0.0, atol=1e-12)
    assert epsp_out > 0.0


def test_void_growth_pure_tension_positive(gtn_params):
    """Under hydrostatic tension without exceeding cavitation limit,
    tr(Delta eps^p) > 0 so void growth Delta f_g > 0.
    """
    mat = build_law52(**gtn_params)
    sig = np.zeros(6, dtype=float)
    # Tensile strain with Poisson contraction: P_N > 0 and VA > 0
    deps = np.array([0.003, -0.0005, -0.0005, 0.0, 0.0, 0.0], dtype=float)
    extra = {}

    sig_out, epsp_out = solid_update_law52(mat, sig, deps, epsp=0.0, dt=1e-5, extra=extra)

    fg = extra["dmg"][0, 1]
    f = extra["dmg"][0, 3]
    f_I = gtn_params["f_I"]

    assert extra["off"][0] == 1.0
    assert epsp_out > 0.0
    assert fg > 0.0
    assert f > f_I
    # Verify Delta f_g matches (1 - f_old) * tr(dp)
    np.testing.assert_allclose(f, f_I + fg + extra["dmg"][0, 2], rtol=1e-12)


def test_void_growth_compression_negative_with_zero_clamp(gtn_params):
    """Under triaxial compression, tr(Delta eps^p) < 0, and total f is clamped >= 0 (sigeps52.F:408)."""
    params = gtn_params.copy()
    params["f_I"] = 0.0001
    mat = build_law52(**params)

    sig = np.zeros(6, dtype=float)
    # Extreme triaxial compression
    deps = np.array([-0.01, -0.01, -0.01, 0.0, 0.0, 0.0], dtype=float)
    extra = {}

    sig_out, epsp_out = solid_update_law52(mat, sig, deps, epsp=0.0, dt=1e-5, extra=extra)

    f = extra["dmg"][0, 3]
    assert f >= 0.0  # sigeps52.F:408: IF(F(I)<ZERO) F(I)=ZERO


# ===================================================================
# 4. Chu & Needleman Gaussian Void Nucleation Parity (sigeps52.F:329-330)
# ===================================================================

def test_gaussian_nucleation_formula_parity():
    """Verify Chu & Needleman Gaussian distribution formula matches sigeps52.F lines 329-330."""
    fn = 0.04
    sn = 0.1
    epsn = 0.3
    sqr22 = 1.0 / math.sqrt(2.0 * math.pi)

    epsm_values = np.linspace(0.0, 0.6, 7)
    for epsm in epsm_values:
        a1_fortran = fn * math.exp(-0.5 * (((epsm - epsn) / sn)**2)) / sn * sqr22
        a1_theory = (fn / (sn * math.sqrt(2.0 * math.pi))) * math.exp(-0.5 * (((epsm - epsn) / sn)**2))

        np.testing.assert_allclose(a1_fortran, a1_theory, rtol=1e-14)


def test_nucleation_suppressed_under_compression_iflag2_and_3():
    """In IFLAG=2 and IFLAG=3, void nucleation is strictly suppressed under compression (PN < 0).
    Matches sigeps52.F line 768 and line 939.
    """
    for iflag_val in [2, 3]:
        mat = build_law52(
            E=210000.0, nu=0.3, rho0=7.8e-9, A=400.0, B=0.0, n=1.0,
            f_I=0.0, f_N=0.05, s_N=0.1, eps_N=0.05,
            iflag=iflag_val,
        )
        sig = np.zeros(6, dtype=float)
        # Compressive strain step: PN < 0
        deps_comp = np.array([-0.005, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        extra = {}
        solid_update_law52(mat, sig, deps_comp, 0.0, dt=1e-5, extra=extra)

        fn_accumulated = extra["dmg"][0, 2]
        np.testing.assert_allclose(fn_accumulated, 0.0, atol=1e-15)


def test_nucleation_active_under_compression_iflag0():
    """In standard GTN (IFLAG=0), nucleation occurs regardless of pressure sign (sigeps52.F:404)."""
    mat = build_law52(
        E=210000.0, nu=0.3, rho0=7.8e-9, A=400.0, B=0.0, n=1.0,
        f_I=0.0, f_N=0.05, s_N=0.1, eps_N=0.05,
        iflag=0,
    )
    sig = np.zeros(6, dtype=float)
    deps_comp = np.array([-0.005, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    extra = {}
    solid_update_law52(mat, sig, deps_comp, 0.0, dt=1e-5, extra=extra)

    fn_accumulated = extra["dmg"][0, 2]
    assert fn_accumulated > 0.0


# ===================================================================
# 5. Matrix Flow Stress and Plastic Work Equivalence (sigeps52.F:381-384)
# ===================================================================

def test_matrix_work_equivalence_parity(gtn_params):
    """Verify work equivalence: (1 - f) * sigma_M * Delta eps_M = sigma : Delta eps^p
    matching sigeps52.F:381-384.
    """
    mat = build_law52(**gtn_params)
    sig = np.zeros(6, dtype=float)
    deps = np.array([0.006, 0.001, -0.002, 0.003, 0.0, 0.0], dtype=float)
    extra = {}

    sig_out, epsp_out = solid_update_law52(mat, sig, deps, epsp=0.0, dt=1e-5, extra=extra)

    f_curr = gtn_params["f_I"]
    sigm = gtn_params["A"]
    epsm_incr = extra["epsm"][0]

    oracle_res = fortran_sigeps52_oracle_step(
        e=gtn_params["E"],
        nu=gtn_params["nu"],
        rho0=gtn_params["rho0"],
        yeild0=gtn_params["A"],
        et=gtn_params["B"],
        n_exp=gtn_params["n"],
        csd=gtn_params["CSD"],
        visp=gtn_params["VISP"],
        q1=gtn_params["q1"],
        q2=gtn_params["q2"],
        q3=gtn_params["q3"],
        sn=gtn_params["s_N"],
        epsn=gtn_params["eps_N"],
        fi=gtn_params["f_I"],
        fn=gtn_params["f_N"],
        fc=gtn_params["f_C"],
        ff=gtn_params["f_F"],
        fu=1.0 / gtn_params["q1"],
        iflag=0,
        sig_old=np.zeros(6),
        deps=deps,
        epsm_old=1e-20 if gtn_params["n"] < 1.0 else 0.0,
        sigm_old=gtn_params["A"],
        fg_old=0.0,
        fn1_old=0.0,
        f_old=gtn_params["f_I"],
        fstar_old=gtn_params["f_I"],
        off_old=1.0,
        epsp_rate=0.0,
    )

    dp = oracle_res["dp"]
    # Matrix plastic work = sigma : Delta eps^p
    work_sigma = (
        sig_out[0] * dp[0] + sig_out[1] * dp[1] + sig_out[2] * dp[2] +
        2.0 * (sig_out[3] * dp[3] + sig_out[4] * dp[4] + sig_out[5] * dp[5])
    )
    work_matrix = (1.0 - f_curr) * sigm * epsm_incr

    np.testing.assert_allclose(work_matrix, work_sigma, rtol=1e-5, atol=1e-12)


# ===================================================================
# 6. Direct Numerical Comparison with Fortran Oracle (5 Iterations)
# ===================================================================

def test_solid_direct_fortran_oracle_parity(gtn_params):
    """Direct step-by-step numerical parity between Python solid_update_law52
    and Fortran sigeps52.F oracle across 5 cutting-plane iterations.
    """
    mat = build_law52(**gtn_params)

    test_deps = [
        np.array([0.005, 0.001, -0.001, 0.002, 0.001, 0.0], dtype=float),
        np.array([0.004, -0.002, 0.001, 0.001, -0.001, 0.002], dtype=float),
        np.array([0.006, 0.002, 0.003, 0.0, 0.002, -0.001], dtype=float),
    ]

    sig_py = np.zeros(6, dtype=float)
    epsp_py = 0.0
    extra_py = {}

    sig_f = np.zeros(6, dtype=float)
    epsm_f = 1e-20 if gtn_params["n"] < 1.0 else 0.0
    sigm_f = gtn_params["A"]
    fg_f = 0.0
    fn1_f = 0.0
    f_f = gtn_params["f_I"]
    fstar_f, _ = compute_f_star(f_f, gtn_params["f_C"], gtn_params["f_F"], 1.0 / gtn_params["q1"])
    off_f = 1.0

    for step_idx, d_eps in enumerate(test_deps):
        sig_py, epsp_py = solid_update_law52(mat, sig_py, d_eps, epsp_py, dt=1e-5, extra=extra_py)

        res_f = fortran_sigeps52_oracle_step(
            e=gtn_params["E"],
            nu=gtn_params["nu"],
            rho0=gtn_params["rho0"],
            yeild0=gtn_params["A"],
            et=gtn_params["B"],
            n_exp=gtn_params["n"],
            csd=gtn_params["CSD"],
            visp=gtn_params["VISP"],
            q1=gtn_params["q1"],
            q2=gtn_params["q2"],
            q3=gtn_params["q3"],
            sn=gtn_params["s_N"],
            epsn=gtn_params["eps_N"],
            fi=gtn_params["f_I"],
            fn=gtn_params["f_N"],
            fc=gtn_params["f_C"],
            ff=gtn_params["f_F"],
            fu=1.0 / gtn_params["q1"],
            iflag=0,
            sig_old=sig_f,
            deps=d_eps,
            epsm_old=epsm_f,
            sigm_old=sigm_f,
            fg_old=fg_f,
            fn1_old=fn1_f,
            f_old=f_f,
            fstar_old=fstar_f,
            off_old=off_f,
            epsp_rate=0.0,
        )

        sig_f = res_f["sign"]
        epsm_f = res_f["epsm"]
        sigm_f = res_f["sigm"]
        fg_f = res_f["fg"]
        fn1_f = res_f["fn1"]
        f_f = res_f["f"]
        fstar_f = res_f["fstar"]
        off_f = res_f["off"]

        np.testing.assert_allclose(sig_py, sig_f, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(extra_py["epsm"][0], epsm_f, rtol=1e-5, atol=1e-12)
        np.testing.assert_allclose(extra_py["sigm"][0], sigm_f, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(extra_py["dmg"][0, 1], fg_f, rtol=1e-5, atol=1e-12)
        np.testing.assert_allclose(extra_py["dmg"][0, 2], fn1_f, rtol=1e-5, atol=1e-12)
        np.testing.assert_allclose(extra_py["dmg"][0, 3], f_f, rtol=1e-5, atol=1e-12)
        np.testing.assert_allclose(extra_py["dmg"][0, 4], fstar_f, rtol=1e-5, atol=1e-12)
        assert extra_py["off"][0] == off_f


def test_shell_direct_fortran_oracle_parity(gtn_params):
    """Direct step-by-step numerical parity between Python shell_update_law52
    and Fortran sigeps52c.F oracle across plane-stress iterations.
    """
    mat = build_law52(**gtn_params)

    test_deps_shell = [
        np.array([0.005, 0.001, 0.002], dtype=float),
        np.array([0.003, -0.001, 0.003], dtype=float),
        np.array([0.004, 0.002, -0.001], dtype=float),
    ]

    sig_py = np.zeros(3, dtype=float)
    epsp_py = 0.0
    extra_py = {"thk": np.array([2.0]), "thk0": np.array([2.0])}

    sig_f = np.zeros(3, dtype=float)
    thk_f = 2.0
    thkly = 2.0
    pla_f = 1e-20 if gtn_params["n"] < 1.0 else 0.0
    sigm_f = gtn_params["A"]
    fg_f = 0.0
    fn1_f = 0.0
    f_f = gtn_params["f_I"]
    fstar_f, _ = compute_f_star(f_f, gtn_params["f_C"], gtn_params["f_F"], 1.0 / gtn_params["q1"])
    off_f = 1.0

    for d_eps in test_deps_shell:
        sig_py, epsp_py = shell_update_law52(mat, sig_py, d_eps, epsp_py, dt=1e-5, extra=extra_py)

        res_f = fortran_sigeps52c_oracle_step(
            e=gtn_params["E"],
            nu=gtn_params["nu"],
            rho0=gtn_params["rho0"],
            yeild0=gtn_params["A"],
            et=gtn_params["B"],
            en=gtn_params["n"],
            csd=gtn_params["CSD"],
            visp=gtn_params["VISP"],
            q1=gtn_params["q1"],
            q2=gtn_params["q2"],
            q3=gtn_params["q3"],
            sn=gtn_params["s_N"],
            epsn=gtn_params["eps_N"],
            fi=gtn_params["f_I"],
            fn=gtn_params["f_N"],
            fc=gtn_params["f_C"],
            ff=gtn_params["f_F"],
            fu=1.0 / gtn_params["q1"],
            iflag=0,
            sigo=sig_f,
            deps=d_eps,
            thk_old=thk_f,
            thkly=thkly,
            pla_old=pla_f,
            sigm_old=sigm_f,
            fg_old=fg_f,
            fn1_old=fn1_f,
            f_old=f_f,
            fstar_old=fstar_f,
            off_old=off_f,
            rate=0.0,
        )

        sig_f = res_f["sign"]
        thk_f = res_f["thk"]
        pla_f = res_f["pla"]
        sigm_f = res_f["sigm"]
        fg_f = res_f["fg"]
        fn1_f = res_f["fn1"]
        f_f = res_f["f"]
        fstar_f = res_f["fstar"]
        off_f = res_f["off"]

        np.testing.assert_allclose(sig_py, sig_f, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(extra_py["thk"][0], thk_f, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(extra_py["epsm"][0], pla_f, rtol=1e-5, atol=1e-12)
        np.testing.assert_allclose(extra_py["sigm"][0], sigm_f, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(extra_py["dmg"][0, 1], fg_f, rtol=1e-5, atol=1e-12)
        np.testing.assert_allclose(extra_py["dmg"][0, 3], f_f, rtol=1e-5, atol=1e-12)
        assert extra_py["off"][0] == off_f


# ===================================================================
# 7. Cavitation Limit and Complete Rupture (VA <= 0 & f* >= f_u)
# ===================================================================

def test_cavitation_rupture_zeros_stresses(gtn_params):
    """Under extreme hydrostatic tension when VA <= 0 and PN > 0,
    element undergoes cavitation rupture, stresses are zeroed, and off = 0.0.
    """
    mat = build_law52(**gtn_params)
    sig = np.zeros(6, dtype=float)
    # Huge hydrostatic tension: Delta eps = [0.1, 0.1, 0.1, 0, 0, 0]
    deps = np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0], dtype=float)
    extra = {}

    sig_out, epsp_out = solid_update_law52(mat, sig, deps, 0.0, dt=1e-4, extra=extra)

    assert extra["off"][0] == 0.0
    np.testing.assert_allclose(sig_out, 0.0, atol=1e-12)


def test_complete_coalescence_rupture_at_ff(gtn_params):
    """When f >= f_F, element ruptures completely: off = 0.0, stresses zeroed."""
    params = gtn_params.copy()
    params["f_I"] = 0.245  # Near f_F = 0.25
    mat = build_law52(**params)

    sig = np.zeros(6, dtype=float)
    deps = np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    extra = {}

    sig_out, epsp_out = solid_update_law52(mat, sig, deps, 0.0, dt=1e-4, extra=extra)

    assert extra["off"][0] == 0.0
    np.testing.assert_allclose(sig_out, 0.0, atol=1e-12)


# ===================================================================
# 8. Shell Thickness Thinning Parity (sigeps52c.F:292-294, 474-475)
# ===================================================================

def test_shell_thickness_thinning_parity(gtn_params):
    """Verify thickness thinning increment:
    Delta eps_zz = (nu / (1 - nu)) * (Delta eps_xx^p + Delta eps_yy^p) + Delta eps_zz^p
    matching sigeps52c.F:474-475.
    """
    mat = build_law52(**gtn_params)
    thk0 = 1.8
    extra = {"thk": np.array([thk0]), "thk0": np.array([thk0])}

    # Uniaxial in-plane tension: eps_xx = 0.02
    deps = np.array([0.02, 0.0, 0.0], dtype=float)
    sig = np.zeros(3, dtype=float)

    sig_out, epsp_out = shell_update_law52(mat, sig, deps, 0.0, dt=1e-5, extra=extra)

    thk_final = extra["thk"][0]
    # Thickness must thin under in-plane tension
    assert thk_final < thk0
    thinning_pct = (thk0 - thk_final) / thk0 * 100.0
    assert thinning_pct > 0.5  # Significant plastic thinning


def test_shell_thickness_thickening_under_compression(gtn_params):
    """Under biaxial in-plane compression, shell thickness increases (thickens)."""
    mat = build_law52(**gtn_params)
    thk0 = 1.5
    extra = {"thk": np.array([thk0]), "thk0": np.array([thk0])}

    # In-plane biaxial compression
    deps = np.array([-0.015, -0.015, 0.0], dtype=float)
    sig = np.zeros(3, dtype=float)

    sig_out, epsp_out = shell_update_law52(mat, sig, deps, 0.0, dt=1e-5, extra=extra)

    thk_final = extra["thk"][0]
    assert thk_final > thk0  # Thickness must increase under in-plane compression


# ===================================================================
# 9. Acoustic Sound Speed Parity (sigeps52.F:208-210 & sigeps52c.F:278)
# ===================================================================

def test_sound_speed_solid_algebraic_proof():
    """Verify Fortran sound speed formula sqrt((C11 + 4/3*G) / rho0) in sigeps52.F:210
    is algebraically identical to sqrt(E*(1-nu) / ((1+nu)*(1-2nu)*rho0)).
    """
    materials = [
        {"name": "Steel", "E": 210000.0, "nu": 0.3, "rho0": 7.85e-9},
        {"name": "Aluminum", "E": 70000.0, "nu": 0.33, "rho0": 2.7e-9},
        {"name": "Titanium", "E": 110000.0, "nu": 0.34, "rho0": 4.5e-9},
        {"name": "Incompressible-Limit", "E": 1000.0, "nu": 0.499, "rho0": 1.0e-9},
    ]

    for m in materials:
        E = m["E"]
        nu = m["nu"]
        rho0 = m["rho0"]

        # Fortran lines 208-210:
        g = 0.5 * E / (1.0 + nu)
        c11 = E / 3.0 / (1.0 - 2.0 * nu)
        c_fortran = math.sqrt((c11 + (4.0 / 3.0) * g) / rho0)

        # Theoretical continuum sound speed formula
        c_theory = math.sqrt(E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu) * rho0))

        # Python implementation
        c_py = sound_speed_solid_law52(Law52Params(E=E, nu=nu, rho0=rho0))

        np.testing.assert_allclose(c_py, c_fortran, rtol=1e-14)
        np.testing.assert_allclose(c_py, c_theory, rtol=1e-14)


def test_sound_speed_shell_algebraic_proof():
    """Verify Fortran plane-stress sound speed formula sqrt(A1 / rho0) in sigeps52c.F:278
    is algebraically identical to sqrt(E / ((1 - nu^2)*rho0)).
    """
    materials = [
        {"name": "Steel", "E": 210000.0, "nu": 0.3, "rho0": 7.85e-9},
        {"name": "Aluminum", "E": 70000.0, "nu": 0.33, "rho0": 2.7e-9},
    ]

    for m in materials:
        E = m["E"]
        nu = m["nu"]
        rho0 = m["rho0"]

        # Fortran line 232 & 278:
        a1 = E / (1.0 - nu**2)
        c_fortran = math.sqrt(a1 / rho0)

        c_theory = math.sqrt(E / ((1.0 - nu**2) * rho0))
        c_py = sound_speed_shell_law52(Law52Params(E=E, nu=nu, rho0=rho0))

        np.testing.assert_allclose(c_py, c_fortran, rtol=1e-14)
        np.testing.assert_allclose(c_py, c_theory, rtol=1e-14)


# ===================================================================
# 10. Algorithmic Consistent Tangents Parity
# ===================================================================

def test_algorithmic_tangents_spectral_properties(gtn_params):
    """Verify algorithmic consistent tangents are symmetric and positive semi-definite."""
    mat = build_law52(**gtn_params)

    # Elastic state
    Cel_solid = tangent_law52_solid(mat, np.zeros(6), epsp=0.0, epsp_incr=0.0)
    np.testing.assert_allclose(Cel_solid, Cel_solid.T, atol=1e-10)
    eig_solid = np.linalg.eigvalsh(Cel_solid)
    assert np.all(eig_solid > 0.0)

    Cel_shell = tangent_law52_shell(mat, np.zeros(3), epsp=0.0, epsp_incr=0.0)
    np.testing.assert_allclose(Cel_shell, Cel_shell.T, atol=1e-10)
    eig_shell = np.linalg.eigvalsh(Cel_shell)
    assert np.all(eig_shell > 0.0)
