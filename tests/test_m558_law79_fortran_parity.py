"""Fortran Parity & Physics Oracle Verifier for Milestone M558: /MAT/LAW79 (Johnson-Holmquist JH-2).

Directly cites and mirrors:
- Upstream Fortran starter reader:
  ``starter/source/materials/mat/mat079/hm_read_mat79.F``
- Upstream Fortran engine physics:
  ``engine/source/materials/mat/mat079/sigeps79.F`` (lines 76-295)

Tests:
1. Exact Python oracle reproducing sigeps79.F lines 76-295.
2. Elastic deviatoric stresses and equivalent stress J2, VM (lines 128-140).
3. EOS pressure with cubic terms K1, K2, K3, bulking pressure deltap, and tensile cutoff (lines 145-154).
4. Intact yield stress sigyi = a * (pstar + tstar)**n (lines 159-166).
5. Fractured yield stress sigyf = b * (pstar)**m capped at sigfmax (lines 168-174, 184).
6. Strain rate scaling ce = 1 + c * log(max(epsd, eps0)/eps0) (lines 176-180).
7. Yield stress interpolation sigy = (1 - dmg)*sigyi + dmg*sigyf (line 185).
8. Radial return scale and deviatoric stress scaling (lines 191-208).
9. Failure strain epfail = d1 * (pstar + tstar)**d2 and damage evolution (lines 218-237).
10. Bulking pressure deltau and deltap incrementation matching lines 265-276.
11. All 4 element deletion modes idel = 0, 1, 2, 3 (lines 239-259).
12. Acoustic sound speed calculation matching lines 287-292.
13. Comprehensive parity check: solid_update vs Fortran oracle across 60+ diverse physical states (rtol=1e-12, atol=1e-12).
"""

import math
import numpy as np
import pytest

from pyradioss.materials.law79_john_holm import (
    Law79Params,
    build_law79,
    solid_update,
    sound_speed_solid,
)


# =============================================================================
# 1. Exact Fortran Oracle Function (sigeps79.F lines 76-295)
# =============================================================================

def sigeps79_oracle(
    uparam: np.ndarray,
    rho0: np.ndarray,
    deps: np.ndarray,       # shape (nel, 6): xx, yy, zz, xy, yz, zx
    sigo: np.ndarray,       # shape (nel, 6): xx, yy, zz, xy, yz, zx
    epsd: np.ndarray,       # shape (nel,)
    amu: np.ndarray,        # shape (nel,)
    uvar: np.ndarray,       # shape (nel, 2): [deltap, sigy * shel]
    off: np.ndarray,        # shape (nel,)
    dmg: np.ndarray,        # shape (nel,)
    defp: np.ndarray,       # shape (nel,)
) -> dict:
    """Exact python oracle reproduction of OpenRadioss sigeps79.F lines 76-295.

    Parameters
    ----------
    uparam : ndarray of length 20
        Packed model parameters from hm_read_mat79.F lines 202-221.
    rho0 : ndarray of length nel
        Reference density.
    deps : ndarray of shape (nel, 6)
        Strain increment tensor (normal and engineering shear).
    sigo : ndarray of shape (nel, 6)
        Old Cauchy stress tensor.
    epsd : ndarray of shape (nel,)
        Equivalent strain rate.
    amu : ndarray of shape (nel,)
        Volumetric strain mu = rho / rho0 - 1.
    uvar : ndarray of shape (nel, 2)
        Internal state variables [uvar1 = deltap, uvar2 = sigy * shel].
    off : ndarray of shape (nel,)
        Element activation flag.
    dmg : ndarray of shape (nel,)
        Damage variable in [0, 1].
    defp : ndarray of shape (nel,)
        Accumulated equivalent plastic strain.

    Returns
    -------
    dict
        Dictionary containing updated outputs:
        'sign', 'sigy', 'dpla', 'soundsp', 'uvar', 'off', 'dmg', 'defp',
        'pnew', 'pstar', 'vm', 'scale', 'deltap'
    """
    g       = float(uparam[0])
    g2      = float(uparam[1])
    aa      = float(uparam[2])
    bb      = float(uparam[3])
    mm      = float(uparam[4])
    nn      = float(uparam[5])
    cc      = float(uparam[6])
    eps0    = float(uparam[7])
    sigfmax = float(uparam[8])
    tstar   = float(uparam[9])
    phel    = float(uparam[10])
    shel    = float(uparam[11])
    d1      = float(uparam[12])
    d2      = float(uparam[13])
    k1      = float(uparam[14])
    k2      = float(uparam[15])
    k3      = float(uparam[16])
    beta    = float(uparam[17])
    idel    = int(round(float(uparam[18])))
    epsmax  = float(uparam[19])

    nel = len(rho0)
    sign = np.zeros((nel, 6), dtype=np.float64)
    sigy = np.zeros(nel, dtype=np.float64)
    dpla = np.zeros(nel, dtype=np.float64)
    soundsp = np.zeros(nel, dtype=np.float64)
    scale_out = np.ones(nel, dtype=np.float64)
    vm_out = np.zeros(nel, dtype=np.float64)
    pnew_out = np.zeros(nel, dtype=np.float64)
    pstar_out = np.zeros(nel, dtype=np.float64)

    uvar_ret = np.asarray(uvar, dtype=np.float64).copy()
    off_ret = np.asarray(off, dtype=np.float64).copy()
    dmg_ret = np.asarray(dmg, dtype=np.float64).copy()
    defp_ret = np.asarray(defp, dtype=np.float64).copy()

    for i in range(nel):
        # Lines 115-123: Recovering internal variables
        if off_ret[i] < 0.1:     # EM01 = 0.1
            off_ret[i] = 0.0
        if off_ret[i] < 1.0:     # FOUR_OVER_5 = 0.8
            off_ret[i] = off_ret[i] * 0.8

        deltap = uvar_ret[i, 0]
        sigyold = uvar_ret[i, 1] / shel if shel > 0.0 else 0.0
        dmg_old = dmg_ret[i]
        mu = amu[i]
        mu2 = mu * mu

        # Lines 128-140: Computation of elastic deviatoric stresses and equivalent stress
        dav = (deps[i, 0] + deps[i, 1] + deps[i, 2]) / 3.0
        pold = -(sigo[i, 0] + sigo[i, 1] + sigo[i, 2]) / 3.0

        sign[i, 0] = sigo[i, 0] + pold + g2 * (deps[i, 0] - dav)
        sign[i, 1] = sigo[i, 1] + pold + g2 * (deps[i, 1] - dav)
        sign[i, 2] = sigo[i, 2] + pold + g2 * (deps[i, 2] - dav)
        sign[i, 3] = sigo[i, 3] + g * deps[i, 3]
        sign[i, 4] = sigo[i, 4] + g * deps[i, 4]
        sign[i, 5] = sigo[i, 5] + g * deps[i, 5]

        j2 = (
            0.5 * (sign[i, 0] ** 2 + sign[i, 1] ** 2 + sign[i, 2] ** 2)
            + sign[i, 3] ** 2
            + sign[i, 4] ** 2
            + sign[i, 5] ** 2
        )
        vm = math.sqrt(3.0 * j2)
        vm_out[i] = vm

        # Lines 145-154: Computation of pressure
        pnew = k1 * mu + deltap
        if mu > 0.0:
            pnew = pnew + k2 * mu2 + k3 * mu2 * mu
        elif idel != 1:
            pmin = -tstar * phel * (1.0 - dmg_ret[i])
            pnew = max(pnew, pmin)
        pstar = pnew / phel if phel > 0.0 else 0.0
        pnew_out[i] = pnew
        pstar_out[i] = pstar

        # Lines 159-186: Computation of yield stress
        if nn == 0.0:
            sigyi = aa
        elif (pstar + tstar) > 0.0:
            sigyi = aa * (pstar + tstar) ** nn
        else:
            sigyi = 0.0

        if mm == 0.0:
            sigyf = bb
        elif pstar > 0.0:
            sigyf = bb * (pstar) ** mm
        else:
            sigyf = 0.0

        if epsd[i] <= eps0:
            ce = 1.0
        else:
            ce = 1.0 + cc * math.log(epsd[i] / eps0)

        sigyi = ce * sigyi
        sigyf = ce * sigyf
        sigyf = min(sigyf, sigfmax)
        sigy[i] = (1.0 - dmg_ret[i]) * sigyi + dmg_ret[i] * sigyf

        # Lines 191-208: Radial return
        scale = 1.0
        if off_ret[i] == 1.0:
            sigstar = vm / shel if shel > 0.0 else 0.0
            if sigstar < sigy[i]:
                scale = 1.0
            elif vm > 0.0:
                scale = sigy[i] / sigstar
            else:
                scale = 0.0
            sign[i, 0] = scale * sign[i, 0]
            sign[i, 1] = scale * sign[i, 1]
            sign[i, 2] = scale * sign[i, 2]
            sign[i, 3] = scale * sign[i, 3]
            sign[i, 4] = scale * sign[i, 4]
            sign[i, 5] = scale * sign[i, 5]
        scale_out[i] = scale

        # Lines 215-260: Update plastic strain and damage
        if off_ret[i] == 1.0:
            # Compute plastic strain at failure
            if d2 == 0.0:
                epfail = d1
            elif (pstar + tstar) >= 0.0:
                epfail = d1 * (pstar + tstar) ** d2
            else:
                epfail = 0.0

            # Update plastic strain and damage
            if epfail > 0.0:
                dpla[i] = (1.0 - scale) * vm / (3.0 * math.sqrt(3.0) * g)
                defp_ret[i] = defp_ret[i] + dpla[i]
                dmg_ret[i] = dmg_ret[i] + dpla[i] / epfail
                dmg_ret[i] = min(dmg_ret[i], 1.0)
            elif scale < 1.0:
                dmg_ret[i] = 1.0

            # Check element deletion
            if idel == 1:
                if (pstar + tstar) < 0.0:
                    off_ret[i] = 0.8
            elif idel == 2:
                if defp_ret[i] > epsmax:
                    off_ret[i] = 0.8
            elif idel == 3:
                if dmg_ret[i] == 1.0:
                    off_ret[i] = 0.8

        # Lines 265-276: Compute pressure increment
        if (dmg_ret[i] > dmg_old) and (mu > 0.0) and (off_ret[i] == 1.0):
            p1 = k1 * mu
            yield_curr = (1.0 - dmg_ret[i]) * sigyi + dmg_ret[i] * sigyf
            deltau = (sigyold * sigyold - yield_curr * yield_curr) / (6.0 * g)
            if deltau > 0.0:
                deltau = deltau * shel * shel
                deltap = -p1 + math.sqrt((deltap + p1) ** 2 + 2.0 * beta * k1 * deltau)

        # Lines 281-295: Update stress tensor and sound speed
        uvar_ret[i, 0] = deltap
        uvar_ret[i, 1] = sigy[i] * shel
        sign[i, 0] = sign[i, 0] - pnew
        sign[i, 1] = sign[i, 1] - pnew
        sign[i, 2] = sign[i, 2] - pnew

        if mu > 0.0:
            dpdmu = k1 + 2.0 * k2 * mu + 3.0 * k3 * mu2
        else:
            dpdmu = k1
        soundsp[i] = math.sqrt((dpdmu + (4.0 / 3.0) * g) / rho0[i])

    return {
        "sign": sign,
        "sigy": sigy,
        "dpla": dpla,
        "soundsp": soundsp,
        "uvar": uvar_ret,
        "off": off_ret,
        "dmg": dmg_ret,
        "defp": defp_ret,
        "pnew": pnew_out,
        "pstar": pstar_out,
        "vm": vm_out,
        "scale": scale_out,
        "deltap": uvar_ret[:, 0],
    }


def make_uparam(
    shear: float = 100.0,
    a: float = 0.93,
    b: float = 0.31,
    m: float = 0.6,
    n: float = 0.65,
    c: float = 0.007,
    eps0: float = 1.0,
    sigfmax: float = 0.8,
    tmax: float = 0.37,
    hel: float = 14.5,
    phel: float = 5.13,
    d1: float = 0.48,
    d2: float = 0.48,
    k1: float = 220.0,
    k2: float = 0.0,
    k3: float = 0.0,
    beta: float = 1.0,
    idel: int = 0,
    epsmax: float = 1.0e20,
) -> np.ndarray:
    """Construct Fortran UPARAM(20) array matching hm_read_mat79.F lines 202-221."""
    uparam = np.zeros(20, dtype=np.float64)
    uparam[0] = shear
    uparam[1] = 2.0 * shear
    uparam[2] = a
    uparam[3] = b
    uparam[4] = m
    uparam[5] = n
    uparam[6] = c
    uparam[7] = eps0
    uparam[8] = sigfmax
    uparam[9] = tmax / phel if phel != 0.0 else 0.0
    uparam[10] = phel
    uparam[11] = 1.5 * (hel - phel)
    uparam[12] = d1
    uparam[13] = d2
    uparam[14] = k1
    uparam[15] = k2
    uparam[16] = k3
    uparam[17] = beta
    uparam[18] = float(idel)
    uparam[19] = epsmax
    return uparam


# =============================================================================
# 2. Physics Sub-component Unit Tests
# =============================================================================

def test_elastic_deviatoric_stresses_and_vm():
    """Verify lines 128-140: dav, pold, deviatoric trial stresses, J2 and VM."""
    g = 120.0
    g2 = 2.0 * g
    up = make_uparam(shear=g)
    rho0 = np.array([3.0])

    sigo = np.array([[10.0, 4.0, -2.0, 5.0, -3.0, 1.5]])
    deps = np.array([[0.002, -0.001, 0.0005, 0.004, -0.002, 0.001]])

    # Hand computation matching Fortran lines 129-140
    dav = (deps[0, 0] + deps[0, 1] + deps[0, 2]) / 3.0
    pold = -(sigo[0, 0] + sigo[0, 1] + sigo[0, 2]) / 3.0
    s_dev_xx = sigo[0, 0] + pold + g2 * (deps[0, 0] - dav)
    s_dev_yy = sigo[0, 1] + pold + g2 * (deps[0, 1] - dav)
    s_dev_zz = sigo[0, 2] + pold + g2 * (deps[0, 2] - dav)
    s_xy = sigo[0, 3] + g * deps[0, 3]
    s_yz = sigo[0, 4] + g * deps[0, 4]
    s_zx = sigo[0, 5] + g * deps[0, 5]

    # Trace of normal deviatoric stresses must be exactly 0
    assert math.isclose(s_dev_xx + s_dev_yy + s_dev_zz, 0.0, abs_tol=1e-12)

    j2 = 0.5 * (s_dev_xx**2 + s_dev_yy**2 + s_dev_zz**2) + s_xy**2 + s_yz**2 + s_zx**2
    vm = math.sqrt(3.0 * j2)

    res = sigeps79_oracle(
        up, rho0, deps, sigo,
        epsd=np.array([0.0]), amu=np.array([0.0]),
        uvar=np.array([[0.0, 100.0]]), off=np.array([1.0]),
        dmg=np.array([0.0]), defp=np.array([0.0])
    )

    assert math.isclose(res["vm"][0], vm, rel_tol=1e-12, abs_tol=1e-12)


def test_eos_pressure_and_tensile_cutoff():
    """Verify lines 145-154: compression cubic polynomial, tension cutoff PMIN, and IDEL=1 bypass."""
    k1, k2, k3 = 200.0, 50.0, 20.0
    phel, hel, tmax = 5.0, 15.0, 0.4
    tstar = tmax / phel

    # Case A: Compression mu > 0
    up_comp = make_uparam(k1=k1, k2=k2, k3=k3, phel=phel, hel=hel, tmax=tmax, idel=0)
    mu_comp = 0.03
    deltap = 0.5
    expected_p = k1 * mu_comp + deltap + k2 * (mu_comp**2) + k3 * (mu_comp**3)
    expected_pstar = expected_p / phel

    res_comp = sigeps79_oracle(
        up_comp, np.array([3.0]), np.zeros((1, 6)), np.zeros((1, 6)),
        epsd=np.array([0.0]), amu=np.array([mu_comp]),
        uvar=np.array([[deltap, 10.0]]), off=np.array([1.0]),
        dmg=np.array([0.2]), defp=np.array([0.0])
    )
    assert math.isclose(res_comp["pnew"][0], expected_p, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(res_comp["pstar"][0], expected_pstar, rel_tol=1e-12, abs_tol=1e-12)

    # Case B: Tension mu < 0 with IDEL != 1 -> PMIN cutoff applied
    mu_tens = -0.05
    dmg_val = 0.4
    pmin = -tstar * phel * (1.0 - dmg_val)
    raw_p = k1 * mu_tens + 0.0  # -10.0 < pmin (-0.24)
    assert raw_p < pmin

    res_tens_cutoff = sigeps79_oracle(
        up_comp, np.array([3.0]), np.zeros((1, 6)), np.zeros((1, 6)),
        epsd=np.array([0.0]), amu=np.array([mu_tens]),
        uvar=np.array([[0.0, 10.0]]), off=np.array([1.0]),
        dmg=np.array([dmg_val]), defp=np.array([0.0])
    )
    assert math.isclose(res_tens_cutoff["pnew"][0], pmin, rel_tol=1e-12, abs_tol=1e-12)

    # Case C: Tension mu < 0 with IDEL = 1 -> PMIN NOT applied, pure K1*mu + deltap
    up_idel1 = make_uparam(k1=k1, k2=k2, k3=k3, phel=phel, hel=hel, tmax=tmax, idel=1)
    res_tens_idel1 = sigeps79_oracle(
        up_idel1, np.array([3.0]), np.zeros((1, 6)), np.zeros((1, 6)),
        epsd=np.array([0.0]), amu=np.array([mu_tens]),
        uvar=np.array([[0.0, 10.0]]), off=np.array([1.0]),
        dmg=np.array([dmg_val]), defp=np.array([0.0])
    )
    assert math.isclose(res_tens_idel1["pnew"][0], raw_p, rel_tol=1e-12, abs_tol=1e-12)


def test_intact_yield_stress():
    """Verify lines 160-166: sigyi = a * (pstar + tstar)**n and boundaries."""
    a = 0.95
    n = 0.7
    tmax, phel = 0.5, 5.0
    tstar = tmax / phel
    up = make_uparam(a=a, n=n, tmax=tmax, phel=phel, hel=15.0)

    # Sub-case 1: Standard positive pressure
    pstar = 0.4
    pnew = pstar * phel
    mu = pnew / up[14]  # mu = p / k1
    res1 = sigeps79_oracle(
        up, np.array([3.0]), np.zeros((1, 6)), np.zeros((1, 6)),
        epsd=np.array([0.0]), amu=np.array([mu]),
        uvar=np.array([[0.0, 10.0]]), off=np.array([1.0]),
        dmg=np.array([0.0]), defp=np.array([0.0])
    )
    expected_sigyi = a * ((pstar + tstar) ** n)
    assert math.isclose(res1["sigy"][0], expected_sigyi, rel_tol=1e-12, abs_tol=1e-12)

    # Sub-case 2: n = 0 exponent -> sigyi = a unconditionally
    up_n0 = make_uparam(a=a, n=0.0, tmax=tmax, phel=phel, hel=15.0)
    res2 = sigeps79_oracle(
        up_n0, np.array([3.0]), np.zeros((1, 6)), np.zeros((1, 6)),
        epsd=np.array([0.0]), amu=np.array([mu]),
        uvar=np.array([[0.0, 10.0]]), off=np.array([1.0]),
        dmg=np.array([0.0]), defp=np.array([0.0])
    )
    assert math.isclose(res2["sigy"][0], a, rel_tol=1e-12, abs_tol=1e-12)

    # Sub-case 3: pstar + tstar <= 0 (under idel=1 extreme tension) -> sigyi = 0
    up_tens = make_uparam(a=a, n=n, tmax=tmax, phel=phel, hel=15.0, idel=1)
    pstar_neg = -2.0 * tstar  # pstar + tstar = -tstar < 0
    mu_neg = (pstar_neg * phel) / up[14]
    res3 = sigeps79_oracle(
        up_tens, np.array([3.0]), np.zeros((1, 6)), np.zeros((1, 6)),
        epsd=np.array([0.0]), amu=np.array([mu_neg]),
        uvar=np.array([[0.0, 10.0]]), off=np.array([1.0]),
        dmg=np.array([0.0]), defp=np.array([0.0])
    )
    assert res3["sigy"][0] == 0.0


def test_fractured_yield_stress_and_capping():
    """Verify lines 168-174 and 184: sigyf = b * (pstar)**m capped at sigfmax."""
    b = 0.4
    m = 0.8
    sigfmax = 0.5
    phel = 5.0
    up = make_uparam(b=b, m=m, sigfmax=sigfmax, phel=phel, hel=15.0)

    # Sub-case 1: Standard positive pressure below cap
    pstar1 = 0.5
    mu1 = (pstar1 * phel) / up[14]
    res1 = sigeps79_oracle(
        up, np.array([3.0]), np.zeros((1, 6)), np.zeros((1, 6)),
        epsd=np.array([0.0]), amu=np.array([mu1]),
        uvar=np.array([[0.0, 10.0]]), off=np.array([1.0]),
        dmg=np.array([1.0]), defp=np.array([0.0])  # D=1 isolates sigyf
    )
    expected_sigyf1 = b * (pstar1 ** m)
    assert expected_sigyf1 < sigfmax
    assert math.isclose(res1["sigy"][0], expected_sigyf1, rel_tol=1e-12, abs_tol=1e-12)

    # Sub-case 2: High pressure exceeding sigfmax -> capped at sigfmax
    pstar2 = 3.0
    mu2 = (pstar2 * phel) / up[14]
    res2 = sigeps79_oracle(
        up, np.array([3.0]), np.zeros((1, 6)), np.zeros((1, 6)),
        epsd=np.array([0.0]), amu=np.array([mu2]),
        uvar=np.array([[0.0, 10.0]]), off=np.array([1.0]),
        dmg=np.array([1.0]), defp=np.array([0.0])
    )
    uncapped = b * (pstar2 ** m)
    assert uncapped > sigfmax
    assert math.isclose(res2["sigy"][0], sigfmax, rel_tol=1e-12, abs_tol=1e-12)

    # Sub-case 3: pstar <= 0 -> sigyf = 0
    mu_zero = 0.0
    res3 = sigeps79_oracle(
        up, np.array([3.0]), np.zeros((1, 6)), np.zeros((1, 6)),
        epsd=np.array([0.0]), amu=np.array([mu_zero]),
        uvar=np.array([[0.0, 10.0]]), off=np.array([1.0]),
        dmg=np.array([1.0]), defp=np.array([0.0])
    )
    assert res3["sigy"][0] == 0.0


def test_strain_rate_scaling():
    """Verify lines 176-180: ce = 1 + c * log(max(epsd, eps0)/eps0) if epsd > eps0 else 1."""
    c = 0.015
    eps0 = 1.0
    a = 1.0
    up = make_uparam(a=a, n=0.0, c=c, eps0=eps0)

    # Sub-case 1: epsd <= eps0 -> ce = 1.0
    res1 = sigeps79_oracle(
        up, np.array([3.0]), np.zeros((1, 6)), np.zeros((1, 6)),
        epsd=np.array([0.5]), amu=np.array([0.0]),
        uvar=np.array([[0.0, 10.0]]), off=np.array([1.0]),
        dmg=np.array([0.0]), defp=np.array([0.0])
    )
    assert math.isclose(res1["sigy"][0], a, rel_tol=1e-12, abs_tol=1e-12)

    # Sub-case 2: epsd > eps0 -> ce = 1 + c * ln(epsd / eps0)
    epsd_val = 500.0
    expected_ce = 1.0 + c * math.log(epsd_val / eps0)
    res2 = sigeps79_oracle(
        up, np.array([3.0]), np.zeros((1, 6)), np.zeros((1, 6)),
        epsd=np.array([epsd_val]), amu=np.array([0.0]),
        uvar=np.array([[0.0, 10.0]]), off=np.array([1.0]),
        dmg=np.array([0.0]), defp=np.array([0.0])
    )
    assert math.isclose(res2["sigy"][0], a * expected_ce, rel_tol=1e-12, abs_tol=1e-12)


def test_yield_stress_interpolation():
    """Verify line 185: sigy = (1 - dmg) * sigyi + dmg * sigyf across full dmg range."""
    a, b = 1.0, 0.2
    up = make_uparam(a=a, b=b, n=0.0, m=0.0)

    for dmg_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        res = sigeps79_oracle(
            up, np.array([3.0]), np.zeros((1, 6)), np.zeros((1, 6)),
            epsd=np.array([0.0]), amu=np.array([0.01]),  # pstar > 0 so sigyf=b
            uvar=np.array([[0.0, 10.0]]), off=np.array([1.0]),
            dmg=np.array([dmg_val]), defp=np.array([0.0])
        )
        expected = (1.0 - dmg_val) * a + dmg_val * b
        assert math.isclose(res["sigy"][0], expected, rel_tol=1e-12, abs_tol=1e-12)


def test_radial_return_scale_and_deviatoric_scaling():
    """Verify lines 191-208: scale = min(1.0, sigy / sigstar) and deviatoric return."""
    g = 100.0
    hel, phel = 10.0, 4.0
    shel = 1.5 * (hel - phel)  # 9.0
    a = 1.0
    up = make_uparam(shear=g, a=a, n=0.0, hel=hel, phel=phel)

    # Elastic case: vm < sigy * shel
    deps_el = np.array([[0.0, 0.0, 0.0, 0.01, 0.0, 0.0]])  # tau = 100 * 0.01 = 1.0, vm = sqrt(3) ~ 1.732 < 9.0
    res_el = sigeps79_oracle(
        up, np.array([3.0]), deps_el, np.zeros((1, 6)),
        epsd=np.array([0.0]), amu=np.array([0.0]),
        uvar=np.array([[0.0, 10.0]]), off=np.array([1.0]),
        dmg=np.array([0.0]), defp=np.array([0.0])
    )
    assert res_el["scale"][0] == 1.0
    assert math.isclose(res_el["sign"][0, 3], 1.0, rel_tol=1e-12)

    # Plastic case: vm > sigy * shel -> scale = (sigy * shel) / vm
    deps_pl = np.array([[0.0, 0.0, 0.0, 0.15, 0.0, 0.0]])  # tau = 15.0, vm = 15 * sqrt(3) ~ 25.98 > 9.0
    res_pl = sigeps79_oracle(
        up, np.array([3.0]), deps_pl, np.zeros((1, 6)),
        epsd=np.array([0.0]), amu=np.array([0.0]),
        uvar=np.array([[0.0, 10.0]]), off=np.array([1.0]),
        dmg=np.array([0.0]), defp=np.array([0.0])
    )
    vm_tr = 15.0 * math.sqrt(3.0)
    sigstar = vm_tr / shel
    expected_scale = a / sigstar
    assert math.isclose(res_pl["scale"][0], expected_scale, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(res_pl["sign"][0, 3], expected_scale * 15.0, rel_tol=1e-12, abs_tol=1e-12)


def test_failure_strain_and_damage_evolution():
    """Verify lines 218-237: epfail = d1*(pstar+tstar)**d2, dpla, and dmg accumulation."""
    d1, d2 = 0.05, 0.8
    g = 100.0
    hel, phel, tmax = 10.0, 4.0, 0.4
    tstar = tmax / phel
    up = make_uparam(shear=g, a=1.0, n=0.0, d1=d1, d2=d2, hel=hel, phel=phel, tmax=tmax)

    deps = np.array([[0.0, 0.0, 0.0, 0.1, 0.0, 0.0]])  # plastic yielding
    pstar = 0.2
    mu = (pstar * phel) / up[14]

    res = sigeps79_oracle(
        up, np.array([3.0]), deps, np.zeros((1, 6)),
        epsd=np.array([0.0]), amu=np.array([mu]),
        uvar=np.array([[0.0, 10.0]]), off=np.array([1.0]),
        dmg=np.array([0.1]), defp=np.array([0.005])
    )

    expected_epfail = d1 * ((pstar + tstar) ** d2)
    scale = res["scale"][0]
    vm = res["vm"][0]
    expected_dpla = (1.0 - scale) * vm / (3.0 * math.sqrt(3.0) * g)
    expected_defp = 0.005 + expected_dpla
    expected_dmg = min(1.0, 0.1 + expected_dpla / expected_epfail)

    assert math.isclose(res["dpla"][0], expected_dpla, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(res["defp"][0], expected_defp, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(res["dmg"][0], expected_dmg, rel_tol=1e-12, abs_tol=1e-12)


def test_bulking_pressure_incrementation():
    """Verify lines 265-276: deltau and deltap incrementation upon damage accumulation under compression."""
    g = 100.0
    k1 = 150.0
    beta = 0.85
    hel, phel = 10.0, 4.0
    shel = 1.5 * (hel - phel)  # 9.0
    up = make_uparam(shear=g, k1=k1, beta=beta, a=1.0, b=0.2, n=0.0, m=0.0, d1=0.05, hel=hel, phel=phel)

    mu = 0.02
    p1 = k1 * mu
    sigyold = 1.0  # intact normalized strength from previous step
    uvar_init = np.array([[0.1, sigyold * shel]])  # deltap = 0.1

    # Large plastic shear strain to cause damage increase
    deps = np.array([[0.0, 0.0, 0.0, 0.15, 0.0, 0.0]])
    res = sigeps79_oracle(
        up, np.array([3.0]), deps, np.zeros((1, 6)),
        epsd=np.array([0.0]), amu=np.array([mu]),
        uvar=uvar_init, off=np.array([1.0]),
        dmg=np.array([0.0]), defp=np.array([0.0])
    )

    new_dmg = res["dmg"][0]
    assert new_dmg > 0.0
    yield_curr = (1.0 - new_dmg) * 1.0 + new_dmg * 0.2
    deltau = (sigyold**2 - yield_curr**2) / (6.0 * g) * (shel**2)
    assert deltau > 0.0

    expected_deltap = -p1 + math.sqrt((0.1 + p1)**2 + 2.0 * beta * k1 * deltau)
    assert math.isclose(res["uvar"][0, 0], expected_deltap, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(res["deltap"][0], expected_deltap, rel_tol=1e-12, abs_tol=1e-12)


def test_all_four_element_deletion_modes():
    """Verify all 4 deletion modes: IDEL = 0 (none), 1 (tension), 2 (plastic strain), 3 (full damage)."""
    phel, hel, tmax = 5.0, 15.0, 0.4
    tstar = tmax / phel
    epsmax = 0.05

    # 1. IDEL = 0: No deletion even with severe conditions
    up0 = make_uparam(idel=0, epsmax=epsmax, tmax=tmax, phel=phel, hel=hel)
    res0 = sigeps79_oracle(
        up0, np.array([3.0]), np.zeros((1, 6)), np.zeros((1, 6)),
        epsd=np.array([0.0]), amu=np.array([-0.05]),  # tension
        uvar=np.array([[0.0, 10.0]]), off=np.array([1.0]),
        dmg=np.array([1.0]), defp=np.array([0.1])  # D=1, defp > epsmax
    )
    assert res0["off"][0] == 1.0

    # 2. IDEL = 1: Tension deletion when P* + T* < 0
    up1 = make_uparam(idel=1, epsmax=epsmax, tmax=tmax, phel=phel, hel=hel)
    # Under IDEL=1, mu < 0 gives pnew = K1*mu. If pnew < -tstar*phel -> pstar + tstar < 0
    mu_tens = -0.01  # K1*mu = 220 * -0.01 = -2.2 < -0.4
    res1 = sigeps79_oracle(
        up1, np.array([3.0]), np.zeros((1, 6)), np.zeros((1, 6)),
        epsd=np.array([0.0]), amu=np.array([mu_tens]),
        uvar=np.array([[0.0, 10.0]]), off=np.array([1.0]),
        dmg=np.array([0.0]), defp=np.array([0.0])
    )
    assert res1["off"][0] == 0.8  # Element deletion triggered

    # 3. IDEL = 2: Deletion when defp > epsmax
    up2 = make_uparam(idel=2, epsmax=epsmax, tmax=tmax, phel=phel, hel=hel, d1=0.1)
    res2 = sigeps79_oracle(
        up2, np.array([3.0]), np.zeros((1, 6)), np.zeros((1, 6)),
        epsd=np.array([0.0]), amu=np.array([0.01]),
        uvar=np.array([[0.0, 10.0]]), off=np.array([1.0]),
        dmg=np.array([0.0]), defp=np.array([0.06])  # defp > epsmax
    )
    assert res2["off"][0] == 0.8

    # 4. IDEL = 3: Deletion when dmg == 1.0
    up3 = make_uparam(idel=3, epsmax=epsmax, tmax=tmax, phel=phel, hel=hel)
    res3 = sigeps79_oracle(
        up3, np.array([3.0]), np.zeros((1, 6)), np.zeros((1, 6)),
        epsd=np.array([0.0]), amu=np.array([0.01]),
        uvar=np.array([[0.0, 10.0]]), off=np.array([1.0]),
        dmg=np.array([1.0]), defp=np.array([0.0])  # dmg = 1
    )
    assert res3["off"][0] == 0.8

    # 5. Decay of OFF over subsequent time steps
    off_step = 0.8
    expected_decay = []
    while off_step > 0.0:
        if off_step < 0.1:
            off_step = 0.0
        else:
            off_step *= 0.8
        expected_decay.append(off_step)

    curr_off = np.array([0.8])
    actual_decay = []
    for _ in range(len(expected_decay)):
        res_decay = sigeps79_oracle(
            up0, np.array([3.0]), np.zeros((1, 6)), np.zeros((1, 6)),
            epsd=np.array([0.0]), amu=np.array([0.0]),
            uvar=np.array([[0.0, 10.0]]), off=curr_off,
            dmg=np.array([0.0]), defp=np.array([0.0])
        )
        curr_off = res_decay["off"]
        actual_decay.append(float(curr_off[0]))

    np.testing.assert_allclose(actual_decay, expected_decay, rtol=1e-12, atol=1e-12)


def test_sound_speed_calculation():
    """Verify lines 287-292: sound speed with K1, K2, K3 for mu > 0 and K1 for mu <= 0."""
    k1, k2, k3 = 200.0, 60.0, 30.0
    g = 120.0
    rho0 = 3.2
    up = make_uparam(shear=g, k1=k1, k2=k2, k3=k3)

    # Compression: mu > 0
    mu_comp = 0.02
    dpdmu_comp = k1 + 2.0 * k2 * mu_comp + 3.0 * k3 * (mu_comp**2)
    expected_c_comp = math.sqrt((dpdmu_comp + (4.0 / 3.0) * g) / rho0)

    res_comp = sigeps79_oracle(
        up, np.array([rho0]), np.zeros((1, 6)), np.zeros((1, 6)),
        epsd=np.array([0.0]), amu=np.array([mu_comp]),
        uvar=np.array([[0.0, 10.0]]), off=np.array([1.0]),
        dmg=np.array([0.0]), defp=np.array([0.0])
    )
    assert math.isclose(res_comp["soundsp"][0], expected_c_comp, rel_tol=1e-12, abs_tol=1e-12)

    # Tension: mu <= 0
    mu_tens = -0.01
    dpdmu_tens = k1
    expected_c_tens = math.sqrt((dpdmu_tens + (4.0 / 3.0) * g) / rho0)

    res_tens = sigeps79_oracle(
        up, np.array([rho0]), np.zeros((1, 6)), np.zeros((1, 6)),
        epsd=np.array([0.0]), amu=np.array([mu_tens]),
        uvar=np.array([[0.0, 10.0]]), off=np.array([1.0]),
        dmg=np.array([0.0]), defp=np.array([0.0])
    )
    assert math.isclose(res_tens["soundsp"][0], expected_c_tens, rel_tol=1e-12, abs_tol=1e-12)


# =============================================================================
# 3. Exhaustive Parity Verification: solid_update vs Fortran Oracle (50+ States)
# =============================================================================

@pytest.mark.parametrize("case_idx", range(60))
def test_solid_update_vs_fortran_oracle_exhaustive_50_states(case_idx: int):
    """Exhaustively verify pyradioss.materials.law79_john_holm.solid_update against sigeps79_oracle.

    Evaluates 60 diverse, independent physical parameter configurations covering:
    - Pure elastic, yielding, post-yield hardening and softening
    - Compression cubic EOS, tension with/without cutoff
    - Strain rate enhancement (low to extreme rates)
    - Full damage range D in [0, 1]
    - Bulking pressure activation under compression
    - Element deletion modes IDEL in {0, 1, 2, 3}
    - Tolerance: assert_allclose(rtol=1e-12, atol=1e-12).
    """
    rng = np.random.default_rng(1000 + case_idx)

    # Randomized realistic ceramic / brittle parameters
    shear = float(rng.uniform(50.0, 250.0))
    a = float(rng.uniform(0.5, 1.5))
    b = float(rng.uniform(0.1, 0.6))
    m = float(rng.choice([0.0, 0.5, 0.8, 1.0]))
    n = float(rng.choice([0.0, 0.5, 0.65, 1.0]))
    c = float(rng.choice([0.0, 0.005, 0.015, 0.03]))
    eps0 = float(rng.uniform(0.5, 2.0))
    sigfmax = float(rng.uniform(0.4, 1.5))
    tmax = float(rng.uniform(0.1, 0.8))
    hel = float(rng.uniform(8.0, 20.0))
    phel = float(rng.uniform(2.0, hel - 1.0))
    d1 = float(rng.choice([0.0, 0.05, 0.2, 0.5]))
    d2 = float(rng.choice([0.0, 0.5, 0.8, 1.0]))
    k1 = float(rng.uniform(100.0, 300.0))
    k2 = float(rng.uniform(0.0, 80.0))
    k3 = float(rng.uniform(0.0, 40.0))
    beta = float(rng.uniform(0.0, 1.0))
    idel = int(rng.choice([0, 1, 2, 3]))
    epsmax = float(rng.choice([0.02, 0.05, 1.0e20]))
    rho0 = float(rng.uniform(2.0, 4.0))

    # Construct material object for pyradioss
    mat_dict = {
        "id": case_idx + 1,
        "rho": rho0,
        "shear": shear,
        "a": a,
        "b": b,
        "m": m,
        "n": n,
        "c": c,
        "eps0": eps0,
        "sigfmax": sigfmax,
        "t": tmax,
        "hel": hel,
        "phel": phel,
        "d1": d1,
        "d2": d2,
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "beta": beta,
        "idel": idel,
        "epsmax": epsmax,
    }
    mat = build_law79(mat_dict)

    # Construct Fortran UPARAM(20) array
    uparam = make_uparam(
        shear=shear, a=a, b=b, m=m, n=n, c=c, eps0=eps0, sigfmax=sigfmax,
        tmax=tmax, hel=hel, phel=phel, d1=d1, d2=d2, k1=k1, k2=k2, k3=k3,
        beta=beta, idel=idel, epsmax=epsmax
    )

    # Generate state variables
    sigo = rng.uniform(-10.0, 10.0, size=6)
    deps = rng.uniform(-0.01, 0.01, size=6)
    # Add substantial shear strain to trigger plasticity in a subset of cases
    if case_idx % 2 == 0:
        deps[3] = rng.uniform(0.05, 0.2)

    mu_val = float(rng.choice([
        rng.uniform(0.001, 0.05),     # compression
        rng.uniform(-0.02, -0.0001),  # tension
        0.0,                          # neutral
    ]))
    epsd_val = float(rng.choice([
        0.0,
        rng.uniform(0.1, 1000.0),
    ]))
    deltap_init = float(rng.choice([0.0, rng.uniform(0.01, 2.0)]))
    sigyold_val = float(rng.uniform(0.2, 1.2))
    shel = 1.5 * (hel - phel)
    uvar_init = np.array([deltap_init, sigyold_val * shel])

    dmg_init = float(rng.choice([0.0, rng.uniform(0.01, 0.95)]))
    defp_init = float(rng.uniform(0.0, 0.03))
    off_init = 1.0

    # 1. Run exact Fortran oracle
    oracle_res = sigeps79_oracle(
        uparam=uparam,
        rho0=np.array([rho0]),
        deps=deps.reshape(1, 6),
        sigo=sigo.reshape(1, 6),
        epsd=np.array([epsd_val]),
        amu=np.array([mu_val]),
        uvar=uvar_init.reshape(1, 2),
        off=np.array([off_init]),
        dmg=np.array([dmg_init]),
        defp=np.array([defp_init]),
    )

    # 2. Run pyradioss solid_update
    extra_py = {
        "mu": mu_val,
        "amu": mu_val,
        "epsd": epsd_val,
        "uvar": uvar_init.copy(),
        "deltap": deltap_init,
        "sigy_old": sigyold_val,
        "off": off_init,
        "dmg": dmg_init,
    }
    epsp_py = np.array([defp_init])

    sig_py = solid_update(mat, sigo.copy(), deps.copy(), extra=extra_py, epsp=epsp_py)

    # 3. Assert exact mathematical parity (rtol=1e-12, atol=1e-12)
    oracle_sign = oracle_res["sign"][0]
    oracle_sigy = oracle_res["sigy"][0]
    oracle_dpla = oracle_res["dpla"][0]
    oracle_soundsp = oracle_res["soundsp"][0]
    oracle_uvar = oracle_res["uvar"][0]
    oracle_off = oracle_res["off"][0]
    oracle_dmg = oracle_res["dmg"][0]
    oracle_defp = oracle_res["defp"][0]
    oracle_scale = oracle_res["scale"][0]
    oracle_pnew = oracle_res["pnew"][0]
    oracle_pstar = oracle_res["pstar"][0]

    np.testing.assert_allclose(sig_py, oracle_sign, rtol=1e-12, atol=1e-12, err_msg=f"Case {case_idx}: sig mismatch")
    np.testing.assert_allclose(extra_py["dmg"], oracle_dmg, rtol=1e-12, atol=1e-12, err_msg=f"Case {case_idx}: dmg mismatch")
    np.testing.assert_allclose(extra_py["off"], oracle_off, rtol=1e-12, atol=1e-12, err_msg=f"Case {case_idx}: off mismatch")
    np.testing.assert_allclose(extra_py["uvar"], oracle_uvar, rtol=1e-12, atol=1e-12, err_msg=f"Case {case_idx}: uvar mismatch")
    np.testing.assert_allclose(extra_py["deltap"], oracle_uvar[0], rtol=1e-12, atol=1e-12, err_msg=f"Case {case_idx}: deltap mismatch")
    np.testing.assert_allclose(epsp_py[0], oracle_defp, rtol=1e-12, atol=1e-12, err_msg=f"Case {case_idx}: defp mismatch")
    np.testing.assert_allclose(extra_py["sound_speed"], oracle_soundsp, rtol=1e-12, atol=1e-12, err_msg=f"Case {case_idx}: soundsp mismatch")
    np.testing.assert_allclose(extra_py["scale"], oracle_scale, rtol=1e-12, atol=1e-12, err_msg=f"Case {case_idx}: scale mismatch")
    np.testing.assert_allclose(extra_py["p"], oracle_pnew, rtol=1e-12, atol=1e-12, err_msg=f"Case {case_idx}: pnew mismatch")
    np.testing.assert_allclose(extra_py["pstar"], oracle_pstar, rtol=1e-12, atol=1e-12, err_msg=f"Case {case_idx}: pstar mismatch")
