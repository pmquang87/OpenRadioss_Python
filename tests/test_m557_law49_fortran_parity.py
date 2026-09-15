r"""Exhaustive Fortran Parity & Physics Oracle verification for /MAT/LAW49 (Steinberg-Guinan shock plasticity model).

Cites upstream reference files:
- ``C:\OpenRadioss\source\OpenRadioss-latest-20260520\starter\source\materials\mat\mat049\hm_read_mat49.F``
- ``C:\OpenRadioss\source\OpenRadioss-latest-20260520\engine\source\materials\mat\mat049\m49law.F``
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import pytest

from pyradioss.materials.law49_steinb import (
    Law49Params,
    build_law49,
    solid_update,
    solid_update_law49,
    sound_speed_solid,
)


# =============================================================================
# 1. UPSTREAM FORTRAN ENGINE ORACLE: M49LAW (lines 73 - 207)
# =============================================================================

def m49law_fortran_oracle(
    sig_in: np.ndarray,
    deps_in: np.ndarray,
    epxe_in: np.ndarray,
    theta_in: np.ndarray,
    df_in: np.ndarray,
    espe_in: np.ndarray,
    rho0_in: np.ndarray,
    dpdm_in: np.ndarray,
    off_in: np.ndarray,
    # Material properties from PM(1:NPROPM, MX)
    g0: float,
    sig0: float,
    cb: float,
    cn: float,
    epmx: float,
    sigmx: float,
    cb1: float,
    cb2: float,
    ch: float,
    sph: float,
    cf: float,
    t0: float,
    tmelt: float,
    rhocp: float,
    dt1: float = 1.0,
    vol: np.ndarray | None = None,
    jlag: int = 0,
    jthe: int = 0,
    fheat_in: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Exact, faithful Python replica of OpenRadioss m49law.F (lines 73 - 207).

    Parameters and variables follow exact upstream Fortran naming and operations:
    - PM(22) = G0, PM(38) = SIG0, PM(39) = CB, PM(40) = CN, PM(41) = EPMX, PM(42) = SIGMX
    - PM(43) = CB1, PM(44) = CB2, PM(45) = CH, PM(69) = SPH = RHOCP, PM(77) = CF, PM(78) = T0, PM(46) = TMELT
    """
    sig = np.array(sig_in, dtype=np.float64).copy()
    deps = np.array(deps_in, dtype=np.float64).copy()
    if sig.ndim == 1:
        sig = sig.reshape(1, 6)
    if deps.ndim == 1:
        deps = deps.reshape(1, 6)

    nel = sig.shape[0]

    def _prep_1d(arr_in: Any, default: float) -> np.ndarray:
        if arr_in is None:
            return np.full(nel, default, dtype=np.float64)
        arr = np.atleast_1d(np.array(arr_in, dtype=np.float64)).copy()
        if len(arr) == 1 and nel > 1:
            return np.full(nel, arr[0], dtype=np.float64)
        return arr

    epxe = _prep_1d(epxe_in, 0.0)
    theta = _prep_1d(theta_in, t0)
    df = _prep_1d(df_in, 1.0)
    espe = _prep_1d(espe_in, 0.0)
    rho0 = _prep_1d(rho0_in, 1.0)
    dpdm = _prep_1d(dpdm_in, 0.0)
    off = _prep_1d(off_in, 1.0)

    fheat = np.zeros(nel, dtype=np.float64) if fheat_in is None else np.array(fheat_in, dtype=np.float64).copy()
    vol_arr = np.ones(nel, dtype=np.float64) if vol is None else np.array(vol, dtype=np.float64).copy()

    # m49law.F lines 89-93
    THIRD = 1.0 / 3.0
    P = np.zeros(nel, dtype=np.float64)
    DAV = np.zeros(nel, dtype=np.float64)
    EMELT = np.zeros(nel, dtype=np.float64)

    for i in range(nel):
        P[i] = -(sig[i, 0] + sig[i, 1] + sig[i, 2]) * THIRD
        DAV[i] = -(deps[i, 0] + deps[i, 1] + deps[i, 2]) * THIRD
        EMELT[i] = sph * tmelt

    # m49law.F lines 97-117: SHEAR MODULUS & YIELD
    QA = np.zeros(nel, dtype=np.float64)
    QB = np.zeros(nel, dtype=np.float64)
    QC = np.zeros(nel, dtype=np.float64)
    G = np.zeros(nel, dtype=np.float64)
    QD = np.zeros(nel, dtype=np.float64)
    QE = np.zeros(nel, dtype=np.float64)
    YLD = np.zeros(nel, dtype=np.float64)

    for i in range(nel):
        # Line 98: QA = P(I)*DF(I)**THIRD (real cube root for signed inputs)
        QA[i] = P[i] * np.cbrt(df[i])
        # Line 99: QB = ONE - CH*(THETA(I)-T0)
        QB[i] = 1.0 - ch * (theta[i] - t0)

        # Lines 100-106: QC
        if EMELT[i] <= 0.0 or cf <= 0.0:
            QC[i] = 1.0
        elif espe[i] >= EMELT[i]:
            QC[i] = 0.0
        else:
            QC[i] = math.exp(cf * espe[i] / (espe[i] - EMELT[i]))

        # Lines 107-108: G(I) and QD(I)
        G[i] = g0 * (cb1 * QA[i] + QB[i]) * QC[i]
        QD[i] = (cb2 * QA[i] + QB[i]) * QC[i]

        # Lines 109-115: QE
        if epxe[i] <= 0.0:
            QE[i] = sig0
        elif epxe[i] > epmx:
            QE[i] = sig0 * ((1.0 + cb * epmx) ** cn)
        else:
            QE[i] = sig0 * ((1.0 + cb * epxe[i]) ** cn)

        # Line 116: YLD(I) = MIN(SIGMX, QE) * QD(I)
        YLD[i] = min(sigmx, QE[i]) * QD[i]

    YLD_nom = YLD.copy()

    # m49law.F lines 121-130: DEVIATORIC STRESS ESTIMATE
    G1 = np.zeros(nel, dtype=np.float64)
    G2 = np.zeros(nel, dtype=np.float64)
    for i in range(nel):
        G1[i] = dt1 * G[i] * off[i]
        G2[i] = 2.0 * G1[i] * off[i]
        sig[i, 0] = sig[i, 0] + P[i] + G2[i] * (deps[i, 0] + DAV[i])
        sig[i, 1] = sig[i, 1] + P[i] + G2[i] * (deps[i, 1] + DAV[i])
        sig[i, 2] = sig[i, 2] + P[i] + G2[i] * (deps[i, 2] + DAV[i])
        sig[i, 3] = sig[i, 3] + G1[i] * deps[i, 3]
        sig[i, 4] = sig[i, 4] + G1[i] * deps[i, 4]
        sig[i, 5] = sig[i, 5] + G1[i] * deps[i, 5]

    # m49law.F lines 132-135: dP/dRHO & CXX
    FOUR_OVER_3 = 4.0 / 3.0
    CXX = np.zeros(nel, dtype=np.float64)
    for i in range(nel):
        dpdm[i] = dpdm[i] + FOUR_OVER_3 * G[i]
        CXX[i] = math.sqrt(abs(dpdm[i]) / rho0[i])

    # m49law.F lines 137-141: EPD & AJ2
    EPD = np.zeros(nel, dtype=np.float64)
    AJ2 = np.zeros(nel, dtype=np.float64)
    for i in range(nel):
        EPD[i] = off[i] * max(
            abs(deps[i, 0]),
            abs(deps[i, 1]),
            abs(deps[i, 2]),
            0.5 * abs(deps[i, 3]),
            0.5 * abs(deps[i, 4]),
            0.5 * abs(deps[i, 5]),
        )
        j2_val = 0.5 * (sig[i, 0] ** 2 + sig[i, 1] ** 2 + sig[i, 2] ** 2) + sig[i, 3] ** 2 + sig[i, 4] ** 2 + sig[i, 5] ** 2
        AJ2[i] = math.sqrt(3.0 * j2_val)

    # m49law.F lines 143-169: Melting & Radial Return
    QH = np.zeros(nel, dtype=np.float64)
    SCALE = np.zeros(nel, dtype=np.float64)

    for i in range(nel):
        if theta[i] >= tmelt:
            QH[i] = 0.0
            G[i] = 0.0
            YLD[i] = 0.0
            AJ2[i] = 0.0
            SCALE[i] = 0.0
        else:
            if cn >= 1.0:
                QH[i] = QD[i] * sig0 * cb * cn * ((1.0 + cb * epxe[i]) ** (cn - 1.0))
            elif epxe[i] > 0.0:
                QH[i] = QD[i] * sig0 * cb * cn / ((1.0 + cb * epxe[i]) ** (1.0 - cn))
            else:
                QH[i] = 0.0

            if AJ2[i] <= YLD[i]:
                SCALE[i] = 1.0
            elif AJ2[i] != 0.0:
                SCALE[i] = YLD[i] / AJ2[i]
            else:
                SCALE[i] = 0.0

    # m49law.F lines 171-189: Radial Return, DPLA, YLD actual
    EM15 = 1e-15
    DPLA = np.zeros(nel, dtype=np.float64)
    for i in range(nel):
        DPLA[i] = (1.0 - SCALE[i]) * AJ2[i] / max(3.0 * G[i] + QH[i], EM15)
        YLD[i] = YLD[i] + DPLA[i] * QH[i]
        sig[i, 0] = SCALE[i] * sig[i, 0] * off[i]
        sig[i, 1] = SCALE[i] * sig[i, 1] * off[i]
        sig[i, 2] = SCALE[i] * sig[i, 2] * off[i]
        sig[i, 3] = SCALE[i] * sig[i, 3] * off[i]
        sig[i, 4] = SCALE[i] * sig[i, 4] * off[i]
        sig[i, 5] = SCALE[i] * sig[i, 5] * off[i]
        epxe[i] = (epxe[i] + DPLA[i]) * off[i]

    SIGY = YLD.copy()
    DEFP = epxe.copy()

    # m49law.F lines 191-207: Temperature heating due to plastic work
    if jthe != 0 and jlag != 0:
        for i in range(nel):
            fheat[i] = fheat[i] + SIGY[i] * DPLA[i] * vol_arr[i]
    elif rhocp > 0.0:
        for i in range(nel):
            theta[i] = theta[i] + SIGY[i] * DPLA[i] / rhocp

    return {
        "sig": sig,          # Final deviatoric stress tensor
        "epxe": epxe,        # DEFP: accumulated plastic strain
        "theta": theta,      # Updated temperature
        "cxx": CXX,          # Sound speed
        "sigy": SIGY,        # Actual yield stress (after return increment)
        "dpla": DPLA,        # Plastic strain increment
        "g": G,              # Shear modulus
        "qd": QD,            # Yield scaling factor
        "qc": QC,            # Melt factor
        "qe": QE,            # Cold work hardening function
        "qh": QH,            # Hardening slope
        "scale": SCALE,      # Radial return scaling factor
        "aj2": AJ2,          # von Mises equivalent trial stress
        "qa": QA,            # Pressure scaling factor
        "qb": QB,            # Temperature scaling factor
        "p": P,              # Hydrostatic pressure
        "yld": YLD_nom,      # Nominal yield stress
        "yld_nom": YLD_nom,  # Nominal yield stress
        "fheat": fheat,
    }


# =============================================================================
# 2. UNIT ATOMIC ORACLE TESTS (EXACT FORTRAN EXPRESSIONS)
# =============================================================================

@pytest.mark.parametrize("p_val", [-5.0e3, -1.0, 0.0, 1.0, 2.5e4])
@pytest.mark.parametrize("df_val", [-8.0, -1.0, -0.125, 0.125, 0.8, 1.0, 1.25, 8.0])
def test_oracle_qa_pressure_density(p_val: float, df_val: float):
    """Test QA = P * DF**(1/3) across positive and negative pressures and densities (line 98)."""
    expected_qa = p_val * np.cbrt(df_val)

    # Evaluate via oracle
    res = m49law_fortran_oracle(
        sig_in=np.array([-p_val, -p_val, -p_val, 0.0, 0.0, 0.0]),
        deps_in=np.zeros(6),
        epxe_in=np.zeros(1),
        theta_in=np.array([300.0]),
        df_in=np.array([df_val]),
        espe_in=np.zeros(1),
        rho0_in=np.array([1.0]),
        dpdm_in=np.zeros(1),
        off_in=np.array([1.0]),
        g0=1.0e4,
        sig0=100.0,
        cb=10.0,
        cn=0.5,
        epmx=1e20,
        sigmx=1e20,
        cb1=1e-4,
        cb2=1e-4,
        ch=1e-4,
        sph=1.0,
        cf=0.0,
        t0=300.0,
        tmelt=1e20,
        rhocp=1.0,
    )
    assert math.isclose(res["qa"][0], expected_qa, rel_tol=1e-14, abs_tol=1e-14)


@pytest.mark.parametrize("theta_val", [100.0, 200.0, 300.0, 500.0, 1200.0])
@pytest.mark.parametrize("h_val", [0.0, 1e-5, 3.8e-4, 1e-3])
def test_oracle_qb_varying_temperatures(theta_val: float, h_val: float):
    """Test QB = 1 - h * (theta - t0) across varying temperatures (line 99)."""
    t0_ref = 300.0
    expected_qb = 1.0 - h_val * (theta_val - t0_ref)

    res = m49law_fortran_oracle(
        sig_in=np.zeros(6),
        deps_in=np.zeros(6),
        epxe_in=np.zeros(1),
        theta_in=np.array([theta_val]),
        df_in=np.array([1.0]),
        espe_in=np.zeros(1),
        rho0_in=np.array([1.0]),
        dpdm_in=np.zeros(1),
        off_in=np.array([1.0]),
        g0=1.0e4,
        sig0=100.0,
        cb=10.0,
        cn=0.5,
        epmx=1e20,
        sigmx=1e20,
        cb1=1e-4,
        cb2=1e-4,
        ch=h_val,
        sph=1.0,
        cf=0.0,
        t0=t0_ref,
        tmelt=1e20,
        rhocp=1.0,
    )
    assert math.isclose(res["qb"][0], expected_qb, rel_tol=1e-14, abs_tol=1e-14)


@pytest.mark.parametrize(
    "emelt, cf, espe, expected_qc",
    [
        # Emelt <= 0 or cf <= 0 -> QC = 1.0
        (0.0, 1.0, 50.0, 1.0),
        (-100.0, 1.0, 50.0, 1.0),
        (500.0, 0.0, 50.0, 1.0),
        (500.0, -2.0, 50.0, 1.0),
        (0.0, 0.0, 0.0, 1.0),
        # espe >= Emelt -> QC = 0.0
        (500.0, 1.0, 500.0, 0.0),
        (500.0, 2.5, 600.0, 0.0),
        (100.0, 0.5, 100.0001, 0.0),
        # 0 < espe < Emelt -> QC = exp(cf * espe / (espe - Emelt))
        (500.0, 1.0, 250.0, math.exp(1.0 * 250.0 / (250.0 - 500.0))),
        (1000.0, 2.0, 100.0, math.exp(2.0 * 100.0 / (100.0 - 1000.0))),
        (1000.0, 0.5, 800.0, math.exp(0.5 * 800.0 / (800.0 - 1000.0))),
        # espe == 0 -> exp(0) = 1.0
        (500.0, 1.0, 0.0, 1.0),
    ],
)
def test_oracle_qc_melt_softening(emelt: float, cf: float, espe: float, expected_qc: float):
    """Test QC conditions strictly adhering to m49law.F lines 100-106."""
    # sph * tmelt = emelt
    sph_val = 1.0
    tmelt_val = emelt if emelt > 0 else 1000.0
    if emelt <= 0:
        sph_val = emelt / tmelt_val

    res = m49law_fortran_oracle(
        sig_in=np.zeros(6),
        deps_in=np.zeros(6),
        epxe_in=np.zeros(1),
        theta_in=np.array([300.0]),
        df_in=np.array([1.0]),
        espe_in=np.array([espe]),
        rho0_in=np.array([1.0]),
        dpdm_in=np.zeros(1),
        off_in=np.array([1.0]),
        g0=1.0e4,
        sig0=100.0,
        cb=10.0,
        cn=0.5,
        epmx=1e20,
        sigmx=1e20,
        cb1=1e-4,
        cb2=1e-4,
        ch=0.0,
        sph=sph_val,
        cf=cf,
        t0=300.0,
        tmelt=tmelt_val,
        rhocp=1.0,
    )
    assert math.isclose(res["qc"][0], expected_qc, rel_tol=1e-12, abs_tol=1e-14)


def test_oracle_g_and_qd_expressions():
    """Test G = G0 * (b1 * QA + QB) * QC and QD = (b2 * QA + QB) * QC (lines 107-108)."""
    g0 = 4.5e4
    b1 = 2.5e-5
    b2 = 1.8e-5
    qa = 500.0
    qb = 0.92
    qc = 0.85

    expected_g = g0 * (b1 * qa + qb) * qc
    expected_qd = (b2 * qa + qb) * qc

    # Construct state matching qa, qb, qc
    t0 = 300.0
    ch = 1e-4
    theta = t0 + (1.0 - qb) / ch
    p_val = qa
    ln_qc = math.log(qc)
    emelt = 2000.0
    espe = emelt * ln_qc / (ln_qc - 1.0)

    res = m49law_fortran_oracle(
        sig_in=np.array([-p_val, -p_val, -p_val, 0.0, 0.0, 0.0]),
        deps_in=np.zeros(6),
        epxe_in=np.zeros(1),
        theta_in=np.array([theta]),
        df_in=np.array([1.0]),
        espe_in=np.array([espe]),
        rho0_in=np.array([1.0]),
        dpdm_in=np.zeros(1),
        off_in=np.array([1.0]),
        g0=g0,
        sig0=100.0,
        cb=10.0,
        cn=0.5,
        epmx=1e20,
        sigmx=1e20,
        cb1=b1,
        cb2=b2,
        ch=ch,
        sph=1.0,
        cf=1.0,
        t0=t0,
        tmelt=emelt,
        rhocp=1.0,
    )
    assert math.isclose(res["g"][0], expected_g, rel_tol=1e-12)
    assert math.isclose(res["qd"][0], expected_qd, rel_tol=1e-12)


@pytest.mark.parametrize(
    "epxe_val, expected_qe",
    [
        # epxe <= 0 -> QE = sig0
        (-0.5, 200.0),
        (0.0, 200.0),
        # 0 < epxe <= epmx -> QE = sig0 * (1 + cb*epxe)**cn
        (0.02, 200.0 * ((1.0 + 50.0 * 0.02) ** 0.35)),
        (0.05, 200.0 * ((1.0 + 50.0 * 0.05) ** 0.35)),
        (0.10, 200.0 * ((1.0 + 50.0 * 0.10) ** 0.35)),  # exactly epmx
        # epxe > epmx -> QE = sig0 * (1 + cb*epmx)**cn
        (0.15, 200.0 * ((1.0 + 50.0 * 0.10) ** 0.35)),
        (1.0, 200.0 * ((1.0 + 50.0 * 0.10) ** 0.35)),
    ],
)
def test_oracle_qe_cold_work_hardening(epxe_val: float, expected_qe: float):
    """Test QE cold work hardening: epsp<=0, 0<epsp<=eps_max, epsp>eps_max (lines 109-115)."""
    res = m49law_fortran_oracle(
        sig_in=np.zeros(6),
        deps_in=np.zeros(6),
        epxe_in=np.array([epxe_val]),
        theta_in=np.array([300.0]),
        df_in=np.array([1.0]),
        espe_in=np.zeros(1),
        rho0_in=np.array([1.0]),
        dpdm_in=np.zeros(1),
        off_in=np.array([1.0]),
        g0=1.0e4,
        sig0=200.0,
        cb=50.0,
        cn=0.35,
        epmx=0.10,
        sigmx=1e20,
        cb1=0.0,
        cb2=0.0,
        ch=0.0,
        sph=1.0,
        cf=0.0,
        t0=300.0,
        tmelt=1e20,
        rhocp=1.0,
    )
    assert math.isclose(res["qe"][0], expected_qe, rel_tol=1e-12)


@pytest.mark.parametrize("sigmx_val", [150.0, 250.0, 1e20])
def test_oracle_yld_capping(sigmx_val: float):
    """Test YLD = min(sigma_max, QE) * QD (line 116)."""
    sig0 = 100.0
    cb = 20.0
    cn = 0.5
    epxe = 0.05
    qe = sig0 * ((1.0 + cb * epxe) ** cn)

    res = m49law_fortran_oracle(
        sig_in=np.zeros(6),
        deps_in=np.zeros(6),
        epxe_in=np.array([epxe]),
        theta_in=np.array([300.0]),
        df_in=np.array([1.0]),
        espe_in=np.zeros(1),
        rho0_in=np.array([1.0]),
        dpdm_in=np.zeros(1),
        off_in=np.array([1.0]),
        g0=1.0e4,
        sig0=sig0,
        cb=cb,
        cn=cn,
        epmx=1e20,
        sigmx=sigmx_val,
        cb1=0.0,
        cb2=0.0,
        ch=0.0,
        sph=1.0,
        cf=0.0,
        t0=300.0,
        tmelt=1e20,
        rhocp=1.0,
    )
    assert math.isclose(res["yld"][0], min(sigmx_val, qe) * 1.0, rel_tol=1e-12)


@pytest.mark.parametrize(
    "cn_val, epxe_val",
    [
        # CN >= 1.0, epxe <= 0
        (1.0, 0.0),
        (1.0, -0.05),
        (1.5, 0.0),
        (2.0, -0.1),
        # CN >= 1.0, epxe > 0
        (1.0, 0.02),
        (1.5, 0.04),
        (2.0, 0.08),
        # CN < 1.0, epxe <= 0 -> QH = 0.0
        (0.5, 0.0),
        (0.5, -0.01),
        (0.2, -0.1),
        # CN < 1.0, epxe > 0
        (0.5, 0.01),
        (0.3, 0.05),
    ],
)
def test_oracle_qh_hardening_slope(cn_val: float, epxe_val: float):
    """Test hardening slope QH for n >= 1, n < 1, and epsp <= 0 (m49law.F lines 153-159)."""
    sig0 = 120.0
    cb = 30.0
    qd = 1.0

    if cn_val >= 1.0:
        expected_qh = qd * sig0 * cb * cn_val * ((1.0 + cb * epxe_val) ** (cn_val - 1.0))
    elif epxe_val > 0.0:
        expected_qh = qd * sig0 * cb * cn_val / ((1.0 + cb * epxe_val) ** (1.0 - cn_val))
    else:
        expected_qh = 0.0

    res = m49law_fortran_oracle(
        sig_in=np.zeros(6),
        deps_in=np.zeros(6),
        epxe_in=np.array([epxe_val]),
        theta_in=np.array([300.0]),
        df_in=np.array([1.0]),
        espe_in=np.zeros(1),
        rho0_in=np.array([1.0]),
        dpdm_in=np.zeros(1),
        off_in=np.array([1.0]),
        g0=1.0e4,
        sig0=sig0,
        cb=cb,
        cn=cn_val,
        epmx=1e20,
        sigmx=1e20,
        cb1=0.0,
        cb2=0.0,
        ch=0.0,
        sph=1.0,
        cf=0.0,
        t0=300.0,
        tmelt=1e20,
        rhocp=1.0,
    )
    assert math.isclose(res["qh"][0], expected_qh, rel_tol=1e-12, abs_tol=1e-14)


def test_oracle_radial_return_and_dpla():
    """Test radial return scale and DPLA = (1 - scale) * AJ2 / max(3*G + QH, 1e-15) (lines 161-173)."""
    g0 = 5.0e4
    sig0 = 200.0
    deps = np.array([0.0, 0.0, 0.0, 0.01, 0.0, 0.0])
    res = m49law_fortran_oracle(
        sig_in=np.zeros(6),
        deps_in=deps,
        epxe_in=np.zeros(1),
        theta_in=np.array([300.0]),
        df_in=np.array([1.0]),
        espe_in=np.zeros(1),
        rho0_in=np.array([1.0]),
        dpdm_in=np.zeros(1),
        off_in=np.array([1.0]),
        g0=g0,
        sig0=sig0,
        cb=0.0,
        cn=0.0,
        epmx=1e20,
        sigmx=1e20,
        cb1=0.0,
        cb2=0.0,
        ch=0.0,
        sph=1.0,
        cf=0.0,
        t0=300.0,
        tmelt=1e20,
        rhocp=1.0,
    )
    aj2 = math.sqrt(3.0) * (g0 * 0.01)
    expected_scale = sig0 / aj2
    expected_dpla = (1.0 - expected_scale) * aj2 / (3.0 * g0)
    assert math.isclose(res["scale"][0], expected_scale, rel_tol=1e-12)
    assert math.isclose(res["dpla"][0], expected_dpla, rel_tol=1e-12)

    # Elastic case: small strain
    deps_el = np.array([0.0, 0.0, 0.0, 1e-4, 0.0, 0.0])
    res_el = m49law_fortran_oracle(
        sig_in=np.zeros(6),
        deps_in=deps_el,
        epxe_in=np.zeros(1),
        theta_in=np.array([300.0]),
        df_in=np.array([1.0]),
        espe_in=np.zeros(1),
        rho0_in=np.array([1.0]),
        dpdm_in=np.zeros(1),
        off_in=np.array([1.0]),
        g0=g0,
        sig0=sig0,
        cb=0.0,
        cn=0.0,
        epmx=1e20,
        sigmx=1e20,
        cb1=0.0,
        cb2=0.0,
        ch=0.0,
        sph=1.0,
        cf=0.0,
        t0=300.0,
        tmelt=1e20,
        rhocp=1.0,
    )
    assert res_el["scale"][0] == 1.0
    assert res_el["dpla"][0] == 0.0


def test_oracle_actual_yield_stress_update():
    """Test actual yield stress update: YLD = YLD + DPLA * QH (line 175)."""
    g0 = 4.0e4
    sig0 = 150.0
    cb = 20.0
    cn = 1.0  # constant QH = sig0 * cb * cn = 3000.0
    deps = np.array([0.0, 0.0, 0.0, 0.02, 0.0, 0.0])

    res = m49law_fortran_oracle(
        sig_in=np.zeros(6),
        deps_in=deps,
        epxe_in=np.zeros(1),
        theta_in=np.array([300.0]),
        df_in=np.array([1.0]),
        espe_in=np.zeros(1),
        rho0_in=np.array([1.0]),
        dpdm_in=np.zeros(1),
        off_in=np.array([1.0]),
        g0=g0,
        sig0=sig0,
        cb=cb,
        cn=cn,
        epmx=1e20,
        sigmx=1e20,
        cb1=0.0,
        cb2=0.0,
        ch=0.0,
        sph=1.0,
        cf=0.0,
        t0=300.0,
        tmelt=1e20,
        rhocp=1.0,
    )
    qh = sig0 * cb * cn
    yld_init = sig0
    aj2 = math.sqrt(3.0) * (g0 * 0.02)
    scale = yld_init / aj2
    dpla = (1.0 - scale) * aj2 / (3.0 * g0 + qh)
    expected_actual_yld = yld_init + dpla * qh

    assert math.isclose(res["sigy"][0], expected_actual_yld, rel_tol=1e-12)


@pytest.mark.parametrize("theta_test", [1356.0, 1500.0, 3000.0])
def test_oracle_melting_at_tmelt(theta_test: float):
    """Test melting at theta >= tmelt: G=0, YLD=0, deviatoric stress zeroed (lines 145-150, 176-181)."""
    tmelt = 1356.0
    deps = np.array([0.01, -0.005, -0.005, 0.02, 0.01, -0.01])

    res = m49law_fortran_oracle(
        sig_in=np.array([100.0, -50.0, -50.0, 20.0, 10.0, -5.0]),
        deps_in=deps,
        epxe_in=np.array([0.05]),
        theta_in=np.array([theta_test]),
        df_in=np.array([1.0]),
        espe_in=np.zeros(1),
        rho0_in=np.array([1.0]),
        dpdm_in=np.zeros(1),
        off_in=np.array([1.0]),
        g0=4.5e4,
        sig0=200.0,
        cb=10.0,
        cn=0.5,
        epmx=1e20,
        sigmx=1e20,
        cb1=1e-4,
        cb2=1e-4,
        ch=1e-4,
        sph=1.0,
        cf=0.0,
        t0=300.0,
        tmelt=tmelt,
        rhocp=1.0,
    )
    assert res["g"][0] == 0.0
    assert res["sigy"][0] == 0.0
    assert res["scale"][0] == 0.0
    assert res["qh"][0] == 0.0
    assert res["dpla"][0] == 0.0
    # Deviatoric stress is zeroed
    assert np.allclose(res["sig"], 0.0, atol=1e-15)


def test_oracle_thermal_heating_update():
    """Test thermal heating update: theta = theta + YLD * DPLA / rhoc_p (m49law.F line 200)."""
    theta_0 = 300.0
    rhocp = 3.5e3
    deps = np.array([0.0, 0.0, 0.0, 0.02, 0.0, 0.0])

    res = m49law_fortran_oracle(
        sig_in=np.zeros(6),
        deps_in=deps,
        epxe_in=np.zeros(1),
        theta_in=np.array([theta_0]),
        df_in=np.array([1.0]),
        espe_in=np.zeros(1),
        rho0_in=np.array([1.0]),
        dpdm_in=np.zeros(1),
        off_in=np.array([1.0]),
        g0=4.0e4,
        sig0=150.0,
        cb=10.0,
        cn=0.5,
        epmx=1e20,
        sigmx=1e20,
        cb1=0.0,
        cb2=0.0,
        ch=0.0,
        sph=rhocp,
        cf=0.0,
        t0=300.0,
        tmelt=1e20,
        rhocp=rhocp,
    )
    dpla = res["dpla"][0]
    sigy = res["sigy"][0]
    expected_theta = theta_0 + sigy * dpla / rhocp
    assert math.isclose(res["theta"][0], expected_theta, rel_tol=1e-12)


@pytest.mark.parametrize("bulk_val", [5.0e4, 1.3e5, 2.0e5])
@pytest.mark.parametrize("g_val", [0.0, 2.0e4, 4.5e4, 8.0e4])
@pytest.mark.parametrize("rho0_val", [2.7, 7.85, 8.96, 19.3])
def test_oracle_acoustic_sound_speed(bulk_val: float, g_val: float, rho0_val: float):
    """Test acoustic sound speed c = sqrt(|bulk + 4/3*G| / rho0) (lines 133-134)."""
    expected_c = math.sqrt(abs(bulk_val + (4.0 / 3.0) * g_val) / rho0_val)

    res = m49law_fortran_oracle(
        sig_in=np.zeros(6),
        deps_in=np.zeros(6),
        epxe_in=np.zeros(1),
        theta_in=np.array([300.0]),
        df_in=np.array([1.0]),
        espe_in=np.zeros(1),
        rho0_in=np.array([rho0_val]),
        dpdm_in=np.array([bulk_val]),
        off_in=np.array([1.0]),
        g0=g_val,
        sig0=100.0,
        cb=0.0,
        cn=0.0,
        epmx=1e20,
        sigmx=1e20,
        cb1=0.0,
        cb2=0.0,
        ch=0.0,
        sph=1.0,
        cf=0.0,
        t0=300.0,
        tmelt=1e20,
        rhocp=1.0,
    )
    assert math.isclose(res["cxx"][0], expected_c, rel_tol=1e-12)


# =============================================================================
# 3. COMPREHENSIVE DIVERSE-STATE BENCHMARK: ORACLE vs solid_update (50+ states)
# =============================================================================

@pytest.fixture(scope="module")
def diverse_states():
    """Generates 55+ distinct physical and mathematical states covering all regimes of LAW49."""
    states = []
    materials_list = [
        # Mat 0: Copper-OFHC standard shock parameters
        Law49Params(
            rho0=8.96,
            e0=1.24e5,
            nu=0.34,
            g0=4.6e4,
            bulk=1.3e5,
            sig0=120.0,
            beta=36.0,
            n=0.45,
            eps_max=1e20,
            sigma_max=640.0,
            t0=300.0,
            tmelt=1356.0,
            rhoc_p=3.45e3,
            b1=2.8e-5,
            b2=2.8e-5,
            h=3.8e-4,
            f=0.0,
            title="Copper-OFHC",
        ),
        # Mat 1: Steel with n >= 1.0 and melt energy softening
        Law49Params(
            rho0=7.85,
            e0=2.1e5,
            nu=0.28,
            g0=8.2e4,
            bulk=1.6e5,
            sig0=350.0,
            beta=15.0,
            n=1.2,
            eps_max=0.5,
            sigma_max=1200.0,
            t0=293.0,
            tmelt=1793.0,
            rhoc_p=3.75e3,
            b1=1.5e-5,
            b2=1.5e-5,
            h=2.5e-4,
            f=0.6,
            title="Alloy-Steel",
        ),
        # Mat 2: Aluminum with low yield and high thermal sensitivity
        Law49Params(
            rho0=2.70,
            e0=7.0e4,
            nu=0.33,
            g0=2.63e4,
            bulk=6.86e4,
            sig0=90.0,
            beta=80.0,
            n=0.25,
            eps_max=0.2,
            sigma_max=320.0,
            t0=295.0,
            tmelt=933.0,
            rhoc_p=2.43e3,
            b1=6.5e-5,
            b2=6.5e-5,
            h=6.2e-4,
            f=1.0,
            title="Al-6061",
        ),
        # Mat 3: Tantalum with strong pressure hardening
        Law49Params(
            rho0=16.65,
            e0=1.78e5,
            nu=0.34,
            g0=6.64e4,
            bulk=1.85e5,
            sig0=377.0,
            beta=10.0,
            n=0.10,
            eps_max=1e20,
            sigma_max=1100.0,
            t0=300.0,
            tmelt=3269.0,
            rhoc_p=2.33e3,
            b1=1.45e-5,
            b2=1.45e-5,
            h=1.3e-4,
            f=0.0,
            title="Tantalum",
        ),
    ]

    # 1-10: Pure elastic states
    for idx, strain_mag in enumerate([1e-6, 5e-6, 1e-5, 2e-5, 5e-5]):
        for mat_idx in [0, 1]:
            mat_p = materials_list[mat_idx]
            deps = np.array([strain_mag, -0.5 * strain_mag, -0.5 * strain_mag, strain_mag, 0.0, 0.0])
            sig_old = np.array([10.0, 5.0, -15.0, 2.0, -1.0, 3.0])
            states.append((mat_p, sig_old, deps, 0.0, 300.0, 1.0, 0.0, 1.0, f"Elastic_{idx}_mat{mat_idx}"))

    # 11-20: Plastic flow & hardening
    for idx, strain_mag in enumerate([0.005, 0.01, 0.02, 0.05, 0.08]):
        for mat_idx in [0, 2]:
            mat_p = materials_list[mat_idx]
            deps = np.array([strain_mag, -0.3 * strain_mag, -0.2 * strain_mag, 2.0 * strain_mag, -strain_mag, 0.5 * strain_mag])
            sig_old = np.array([50.0, -20.0, -30.0, 40.0, -10.0, 15.0])
            epsp_old = 0.01 * idx
            states.append((mat_p, sig_old, deps, epsp_old, 350.0, 1.02, 10.0, 1.0, f"Plastic_{idx}_mat{mat_idx}"))

    # 21-28: Saturated hardening (eps_max & sigma_max)
    for idx, epsp_val in enumerate([0.15, 0.25, 0.6, 1.2]):
        for mat_idx in [1, 2]:
            mat_p = materials_list[mat_idx]
            deps = np.array([0.02, 0.01, -0.03, 0.04, -0.02, 0.01])
            sig_old = np.array([100.0, 100.0, -200.0, 80.0, 0.0, 0.0])
            states.append((mat_p, sig_old, deps, epsp_val, 400.0, 1.05, 50.0, 1.0, f"Saturation_{idx}_mat{mat_idx}"))

    # 29-34: Cryogenic temperatures (theta < t0)
    for idx, temp in enumerate([50.0, 120.0, 200.0]):
        for mat_idx in [0, 3]:
            mat_p = materials_list[mat_idx]
            deps = np.array([0.01, -0.005, -0.005, 0.015, 0.0, 0.0])
            sig_old = np.zeros(6)
            states.append((mat_p, sig_old, deps, 0.02, temp, 1.01, 0.0, 1.0, f"Cryogenic_{idx}_mat{mat_idx}"))

    # 35-42: High temperature & Melting (near tmelt, at tmelt, above tmelt)
    for mat_idx in [0, 2]:
        mat_p = materials_list[mat_idx]
        tm = mat_p.tmelt
        for idx, temp in enumerate([tm - 50.0, tm - 1.0, tm, tm + 100.0]):
            deps = np.array([0.02, -0.01, -0.01, 0.03, 0.01, -0.02])
            sig_old = np.array([30.0, 20.0, -50.0, 15.0, 5.0, -10.0])
            states.append((mat_p, sig_old, deps, 0.05, temp, 0.95, 100.0, 1.0, f"ThermalMelt_{idx}_mat{mat_idx}"))

    # 43-48: Specific internal energy melt softening (QC)
    mat_p = materials_list[1]  # f = 0.6
    emelt = mat_p.rhoc_p * mat_p.tmelt
    for idx, factor in enumerate([0.1, 0.5, 0.9, 1.0, 1.2]):
        espe_val = factor * emelt
        deps = np.array([0.01, -0.005, -0.005, 0.02, 0.0, 0.0])
        sig_old = np.array([40.0, -20.0, -20.0, 25.0, 0.0, 0.0])
        states.append((mat_p, sig_old, deps, 0.03, 500.0, 1.0, espe_val, 1.0, f"EnergyMelt_{idx}"))

    # 49-54: High pressure shock compression & negative pressure tension
    for idx, p_level in enumerate([-2000.0, -500.0, 2000.0, 10000.0, 50000.0]):
        mat_p = materials_list[0]
        sig_old = np.array([-p_level, -p_level, -p_level, 10.0, -5.0, 5.0])
        df_val = 1.0 + p_level / (3.0 * mat_p.bulk)
        deps = np.array([-0.01, -0.01, -0.01, 0.01, 0.0, 0.0])
        states.append((mat_p, sig_old, deps, 0.02, 320.0, df_val, 20.0, 1.0, f"ShockPressure_{idx}"))

    # 55-58: Inactive / deleted elements (off = 0.0)
    for idx, mat_idx in enumerate([0, 1]):
        mat_p = materials_list[mat_idx]
        deps = np.array([0.05, -0.02, -0.03, 0.05, 0.02, -0.01])
        sig_old = np.array([100.0, -50.0, -50.0, 30.0, 10.0, -10.0])
        states.append((mat_p, sig_old, deps, 0.1, 300.0, 1.0, 0.0, 0.0, f"Inactive_{idx}"))

    return states


def test_parity_oracle_vs_solid_update_all_states(diverse_states):
    """Exhaustive differential parity test: law49_steinb.solid_update vs m49law_fortran_oracle across 55+ states.

    Verifies at rtol=1e-12, atol=1e-12:
    - Deviatoric stress tensor components
    - Accumulated equivalent plastic strain (defp / epsp)
    - Updated temperature theta
    - Actual yield stress (sigy)
    - Plastic strain increment (dpla)
    - Shear modulus G and scaling factor QD
    - Melt factor QC and hardening slope QH
    - Longitudinal acoustic wave speed
    """
    assert len(diverse_states) >= 50, f"Expected 50+ diverse states, got {len(diverse_states)}"

    for state in diverse_states:
        mat_p, sig_old, deps, epsp_old, theta_old, df_val, espe_val, off_val, desc = state
        mat = build_law49(mat_p)

        # 1. Run upstream Fortran oracle
        oracle_res = m49law_fortran_oracle(
            sig_in=sig_old,
            deps_in=deps,
            epxe_in=np.array([epsp_old]),
            theta_in=np.array([theta_old]),
            df_in=np.array([df_val]),
            espe_in=np.array([espe_val]),
            rho0_in=np.array([mat_p.rho0]),
            dpdm_in=np.array([mat_p.bulk]),
            off_in=np.array([off_val]),
            g0=mat_p.g0,
            sig0=mat_p.sig0,
            cb=mat_p.beta,
            cn=mat_p.n,
            epmx=mat_p.eps_max,
            sigmx=mat_p.sigma_max,
            cb1=mat_p.b1,
            cb2=mat_p.b2,
            ch=mat_p.h,
            sph=mat_p.rhoc_p,
            cf=mat_p.f,
            t0=mat_p.t0,
            tmelt=mat_p.tmelt,
            rhocp=mat_p.rhoc_p,
            dt1=1.0,
        )

        # 2. Run pyradioss solid_update
        extra = {
            "theta": theta_old,
            "df": df_val,
            "espe": espe_val,
            "off": off_val,
        }
        sig_py, epsp_py, c_py = solid_update(
            mat,
            sig_old.copy(),
            deps.copy(),
            epsp=epsp_old,
            dt=1.0,
            extra=extra,
            return_tuple=True,
        )

        # 3. Deviatoric stress comparison:
        s_py = extra["s"][0] if extra["s"].ndim > 1 else extra["s"]
        s_oracle = oracle_res["sig"][0]
        np.testing.assert_allclose(
            s_py,
            s_oracle,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Deviatoric stress mismatch in state {desc}",
        )

        # 4. Equivalent plastic strain comparison (defp / epsp)
        epsp_oracle = oracle_res["epxe"][0]
        np.testing.assert_allclose(
            epsp_py,
            epsp_oracle,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Plastic strain mismatch in state {desc}",
        )

        # 5. Temperature comparison
        theta_oracle = oracle_res["theta"][0]
        np.testing.assert_allclose(
            extra["theta"][0],
            theta_oracle,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Temperature mismatch in state {desc}",
        )

        # 6. Actual yield stress comparison (sigy)
        sigy_oracle = oracle_res["sigy"][0]
        np.testing.assert_allclose(
            extra["sigy"][0],
            sigy_oracle,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Yield stress mismatch in state {desc}",
        )

        # 7. Plastic strain increment comparison (dpla)
        dpla_oracle = oracle_res["dpla"][0]
        np.testing.assert_allclose(
            extra["dpla"][0],
            dpla_oracle,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"DPLA mismatch in state {desc}",
        )

        # 8. Shear modulus G comparison
        g_oracle = oracle_res["g"][0]
        np.testing.assert_allclose(
            extra["g"][0],
            g_oracle,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Shear modulus G mismatch in state {desc}",
        )

        # 9. Yield factor QD comparison
        qd_oracle = oracle_res["qd"][0]
        np.testing.assert_allclose(
            extra["qd"][0],
            qd_oracle,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"QD factor mismatch in state {desc}",
        )

        # 10. Melt factor QC comparison
        qc_oracle = oracle_res["qc"][0]
        np.testing.assert_allclose(
            extra["qc"][0],
            qc_oracle,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"QC factor mismatch in state {desc}",
        )

        # 11. Hardening slope QH comparison
        qh_oracle = oracle_res["qh"][0]
        np.testing.assert_allclose(
            extra["qh"][0],
            qh_oracle,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Hardening slope QH mismatch in state {desc}",
        )

        # 12. Acoustic sound speed comparison
        c_oracle = oracle_res["cxx"][0]
        np.testing.assert_allclose(
            c_py,
            c_oracle,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Acoustic sound speed mismatch in state {desc}",
        )


def test_vectorized_multi_element_oracle_parity(diverse_states):
    """Test batched 2D multi-element array updates against the vectorized oracle."""
    nel = 20
    selected_states = diverse_states[:nel]
    mat_p = selected_states[0][0]
    mat = build_law49(mat_p)

    sig_arr = np.array([s[1] for s in selected_states], dtype=np.float64)
    deps_arr = np.array([s[2] for s in selected_states], dtype=np.float64)
    epsp_arr = np.array([s[3] for s in selected_states], dtype=np.float64)
    theta_arr = np.array([s[4] for s in selected_states], dtype=np.float64)
    df_arr = np.array([s[5] for s in selected_states], dtype=np.float64)
    espe_arr = np.array([s[6] for s in selected_states], dtype=np.float64)
    off_arr = np.array([s[7] for s in selected_states], dtype=np.float64)

    # Run batched oracle
    oracle_res = m49law_fortran_oracle(
        sig_in=sig_arr,
        deps_in=deps_arr,
        epxe_in=epsp_arr,
        theta_in=theta_arr,
        df_in=df_arr,
        espe_in=espe_arr,
        rho0_in=np.full(nel, mat_p.rho0),
        dpdm_in=np.full(nel, mat_p.bulk),
        off_in=off_arr,
        g0=mat_p.g0,
        sig0=mat_p.sig0,
        cb=mat_p.beta,
        cn=mat_p.n,
        epmx=mat_p.eps_max,
        sigmx=mat_p.sigma_max,
        cb1=mat_p.b1,
        cb2=mat_p.b2,
        ch=mat_p.h,
        sph=mat_p.rhoc_p,
        cf=mat_p.f,
        t0=mat_p.t0,
        tmelt=mat_p.tmelt,
        rhocp=mat_p.rhoc_p,
    )

    # Run batched solid_update
    extra = {
        "theta": theta_arr,
        "df": df_arr,
        "espe": espe_arr,
        "off": off_arr,
    }
    sig_py, epsp_py, c_py = solid_update(
        mat,
        sig_arr.copy(),
        deps_arr.copy(),
        epsp=epsp_arr.copy(),
        extra=extra,
        return_tuple=True,
    )

    np.testing.assert_allclose(extra["s"], oracle_res["sig"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(epsp_py, oracle_res["epxe"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(extra["theta"], oracle_res["theta"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(extra["sigy"], oracle_res["sigy"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(extra["dpla"], oracle_res["dpla"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(c_py, oracle_res["cxx"], rtol=1e-12, atol=1e-12)
