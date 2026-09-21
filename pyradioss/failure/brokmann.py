"""
Brokmann subcritical crack growth failure model for glass (/FAIL/ALTER Brokmann).

Fortran origin:
  ``engine/source/materials/fail/alter/fail_brokmann.F``          (engine step)
  ``common_source/fail/newman_raju.F90``                          (geometry factor)
  ``starter/source/materials/fail/windshield_alter/brokmann_crack_init.F90`` (init)

Theory
------
Fracture model for glass based on Paris–Erdogan subcritical crack growth.
Each element carries a statistically distributed surface micro-crack (half-ellipse)
characterised by its length *c* and depth *a*.  At each time step the stress
intensity factor at the crack tip is evaluated using the Newman-Raju (1981)
geometry correction and the crack is grown using the Paris law.  Failure is
triggered when K₁ ≥ K_IC.

Newman-Raju geometry correction factor
(``common_source/fail/newman_raju.F90`` lines 51-102):

    q    = 1 + 1.464 * (a/c)^1.65
    m1   = 1.13 - 0.09*(a/c)
    m2   = -0.54 + 0.89 / (0.2 + a/c)
    m3   = 0.5 - 1/(0.65 + a/c) + 14*(1 - a/c)^24
    g    = 1 + (0.1 + 0.35*(a/t)^2) * (1 - sin(phi*pi))^2
    fphi = ((a/c)^2 * cos^2(phi*pi) + sin^2(phi*pi))^0.25
    fw   = cos(pi*c/(2b) * sqrt(a/t))
    f    = (m1 + m2*(a/t)^2 + m3*(a/t)^4) * fphi * g / sqrt(|fw|)
    Y    = sqrt(1/q) * f

where phi = 0.5 gives the deepest point (tip in depth direction, used for V_A)
and phi = 0.0 gives the surface point (used for V_C).

Paris–Erdogan crack growth (``fail_brokmann.F`` lines 151-158):

    V_A = V0 * (K1_A / K_CM)^EXP_N   if K_TH_M <= K1_A < K_CM
    V_C = V0 * (K1_C / K_CM)^EXP_N   if K_TH_M <= K1_C and K1_A < K_CM
    DA  = V_A * dt * FAC_LENM          (growth in depth, micrometers)
    DC  = V_C * dt * FAC_LENM          (growth in length)

Stress intensity factor (``fail_brokmann.F`` lines 138-139):

    K1_A = Y_A * sigma_eff * sqrt(pi * a * 1e-6)   [Pa*sqrt(m)]
    K1_C = Y_C * sigma_eff * sqrt(pi * c * 1e-6)

State variable layout (Python 0-based, Fortran +1):
  uvar[14] = FAIL_B   : failure flag (1 = failed, 0 = growing)
  uvar[15] = CR_LEN   : crack half-length c  [micrometers]
  uvar[16] = CR_DEPTH : crack depth      a  [micrometers]
  uvar[17] = CR_ANG   : crack angle         [radians]
  uvar[18] = THK0     : initial thickness   [micrometers]
  uvar[19] = ALDT0    : initial width       [micrometers]
  uvar[20] = SIG_COS  : filtered crack-opening stress [model units]

UPARAM layout (Python 0-based, Fortran UPARAM(i) → uparam[i-1]):
  uparam[0]  = EXP_N   : Paris exponent
  uparam[5]  = K_IC    : fracture toughness  [model units: stress*sqrt(L)]
  uparam[6]  = K_TH    : threshold SIF       [model units]
  uparam[7]  = V0      : reference crack velocity [model units: L/T]
  uparam[9]  = ALPHA   : exponential-average filter weight
  uparam[29] = SIG_INI : initial surface residual stress [model units]
  uparam[32] = FAC_M   : unit mass factor
  uparam[33] = FAC_L   : unit length factor
  uparam[34] = FAC_T   : unit time factor
"""

from __future__ import annotations

import math
import sys
from typing import Any
import numpy as np

_TINY = 1.0e-20
_EP06 = 1.0e6   # → micrometers  (EP06 in Fortran)
_EM6  = 1.0e-6  # ← micrometers  (EM6  in Fortran)

# Hard-coded constants matching fail_brokmann.F lines 102-103
_DSIG_INI = 2.0   # [MPa/s] – stress-rate threshold
_XDAMP    = 25.0  # damping exponent for high stress rates


# ---------------------------------------------------------------------------
# Newman-Raju geometry factor
# Source: common_source/fail/newman_raju.F90  lines 51-102
# ---------------------------------------------------------------------------

def newman_raju(c: float, a: float, t: float, b: float, fpi: float) -> float:
    """Geometry correction factor for a semi-elliptical surface crack.

    Implements the Newman-Raju (1981) formula exactly as coded in
    ``common_source/fail/newman_raju.F90`` lines 73-99.

    Parameters
    ----------
    c : float
        Crack half-length (surface direction) [micrometers].
    a : float
        Crack depth [micrometers].
    t : float
        Plate thickness [micrometers].
    b : float
        Plate half-width [micrometers].
    fpi : float
        Angular position parameter:
          0.5 → deepest point (phi = pi/2, sin=1, cos=0),
          0.0 → surface point  (phi = 0,   sin=0, cos=1).

    Returns
    -------
    float
        Geometry correction factor Y (dimensionless).

    Notes
    -----
    Fortran: ``common_source/fail/newman_raju.F90`` lines 73-99.
    """
    # angular position (newman_raju.F90 lines 73-82)
    if fpi == 0.5:
        sinp = 1.0
        cosp = 0.0
    elif fpi == 0.0:
        sinp = 0.0
        cosp = 1.0
    else:
        sinp = math.sin(fpi * math.pi)
        cosp = math.cos(fpi * math.pi)

    ac = a / c                       # newman_raju.F90 line 84
    at = a / t                       # newman_raju.F90 line 85
    q  = 1.0 + 1.464 * ac ** 1.65   # newman_raju.F90 line 86

    m1 = 1.13 - 0.09 * ac           # newman_raju.F90 line 88
    m2 = -0.54 + 0.89 / (0.2 + ac)  # newman_raju.F90 line 89
    m3 = 0.5 - 1.0 / (0.65 + ac) + 14.0 * (1.0 - ac) ** 24  # line 90
    g  = 1.0 + (0.1 + 0.35 * at ** 2) * (1.0 - sinp) ** 2   # line 91

    fphi = (ac ** 2 * cosp ** 2 + sinp ** 2) ** 0.25          # line 93
    # fb = math.pi * c * math.sqrt(at)                        # line 94  (unused in Y)

    # finite-width correction  (newman_raju.F90 line 96)
    fw = math.cos(math.pi * c / (2.0 * b) * math.sqrt(at))

    f = (m1 + m2 * at ** 2 + m3 * at ** 4) * fphi * g / math.sqrt(abs(fw) + _TINY)
    y = math.sqrt(1.0 / q) * f      # newman_raju.F90 line 99
    return y


def newman_raju_vec(
    c: np.ndarray,
    a: np.ndarray,
    t: np.ndarray,
    b: np.ndarray,
    fpi: float,
) -> np.ndarray:
    """Vectorised Newman-Raju geometry factor over element arrays.

    Same formulae as :func:`newman_raju` but accepts NumPy arrays.
    """
    if fpi == 0.5:
        sinp = 1.0
        cosp = 0.0
    elif fpi == 0.0:
        sinp = 0.0
        cosp = 1.0
    else:
        sinp = math.sin(fpi * math.pi)
        cosp = math.cos(fpi * math.pi)

    ac = a / np.maximum(c, _TINY)
    at = a / np.maximum(t, _TINY)
    q  = 1.0 + 1.464 * ac ** 1.65

    m1 = 1.13 - 0.09 * ac
    m2 = -0.54 + 0.89 / (0.2 + ac)
    m3 = 0.5 - 1.0 / (0.65 + ac) + 14.0 * (1.0 - ac) ** 24
    g  = 1.0 + (0.1 + 0.35 * at ** 2) * (1.0 - sinp) ** 2

    fphi = (ac ** 2 * cosp ** 2 + sinp ** 2) ** 0.25
    fw   = np.cos(math.pi * c / (2.0 * np.maximum(b, _TINY)) * np.sqrt(np.maximum(at, 0.0)))

    f  = (m1 + m2 * at ** 2 + m3 * at ** 4) * fphi * g / np.sqrt(np.abs(fw) + _TINY)
    return np.sqrt(1.0 / np.maximum(q, _TINY)) * f


def newman_raju_K(
    a: float | np.ndarray,
    c: float | np.ndarray,
    t: float | np.ndarray,
    b: float | np.ndarray,
    sigma: float | np.ndarray,
    phi: float | np.ndarray = math.pi / 2,
) -> float | np.ndarray:
    """Mode I stress intensity factor K_I for a semi-elliptical surface crack.

    Implements the Newman-Raju (1981) stress intensity factor formula.

    Upstream Fortran source:
      ``common_source/fail/newman_raju.F90`` lines 51-102
      ``engine/source/materials/fail/alter/fail_brokmann.F`` lines 136-139

    Theory:
      ac = a / c
      at = a / t
      Q = 1 + 1.464 * (a/c)^1.65
      m1 = 1.13 - 0.09 * (a/c)
      m2 = -0.54 + 0.89 / (0.2 + a/c)
      m3 = 0.5 - 1 / (0.65 + a/c) + 14 * (1 - a/c)^24
      g = 1 + (0.1 + 0.35 * (a/t)^2) * (1 - sin(phi))^2
      fphi = ((a/c)^2 * cos^2(phi) + sin^2(phi))^0.25
      fw = cos(pi * c / (2 * b) * sqrt(a/t))
      F = sqrt(1 / Q) * (m1 + m2 * (a/t)^2 + m3 * (a/t)^4) * g * fphi / sqrt(|fw|)
      K_I = sigma * sqrt(pi * a) * F

    Parameters
    ----------
    a : float or np.ndarray
        Crack depth [L].
    c : float or np.ndarray
        Crack half-length [L].
    t : float or np.ndarray
        Plate thickness [L].
    b : float or np.ndarray
        Plate half-width [L].
    sigma : float or np.ndarray
        Applied tensile / opening stress [stress].
    phi : float or np.ndarray, optional
        Parametric angle along the crack border [radians].
        Default is pi/2 (deepest point). phi=0 corresponds to the surface point.

    Returns
    -------
    float or np.ndarray
        Stress intensity factor K_I [stress * sqrt(L)].
    """
    is_array = any(isinstance(v, np.ndarray) for v in (a, c, t, b, sigma, phi))
    if not is_array:
        a_f = float(a)
        c_f = float(c)
        t_f = float(t)
        b_f = float(b)
        sig_f = float(sigma)
        phi_f = float(phi)

        if a_f <= 0.0 or c_f <= 0.0:
            return 0.0

        if abs(phi_f - math.pi / 2.0) < 1e-7 or phi_f == 0.5:
            sinp = 1.0
            cosp = 0.0
        elif abs(phi_f) < 1e-7:
            sinp = 0.0
            cosp = 1.0
        else:
            sinp = math.sin(phi_f)
            cosp = math.cos(phi_f)

        ac = a_f / c_f
        at = a_f / t_f
        q = 1.0 + 1.464 * (ac ** 1.65)

        m1 = 1.13 - 0.09 * ac
        m2 = -0.54 + 0.89 / (0.2 + ac)
        m3 = 0.5 - 1.0 / (0.65 + ac) + 14.0 * ((1.0 - ac) ** 24)
        g = 1.0 + (0.1 + 0.35 * (at ** 2)) * ((1.0 - sinp) ** 2)

        fphi = (ac ** 2 * cosp ** 2 + sinp ** 2) ** 0.25
        fw = math.cos(math.pi * c_f / (2.0 * b_f) * math.sqrt(max(at, 0.0)))

        f = (m1 + m2 * (at ** 2) + m3 * (at ** 4)) * fphi * g / math.sqrt(abs(fw) + _TINY)
        F = math.sqrt(1.0 / max(q, _TINY)) * f
        return sig_f * math.sqrt(math.pi * a_f) * F

    # Vectorized path
    a_arr = np.asarray(a, dtype=float)
    c_arr = np.asarray(c, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    sig_arr = np.asarray(sigma, dtype=float)
    phi_arr = np.asarray(phi, dtype=float)

    sinp = np.where(np.isclose(phi_arr, math.pi / 2.0) | np.isclose(phi_arr, 0.5), 1.0,
           np.where(np.isclose(phi_arr, 0.0), 0.0, np.sin(phi_arr)))
    cosp = np.where(np.isclose(phi_arr, math.pi / 2.0) | np.isclose(phi_arr, 0.5), 0.0,
           np.where(np.isclose(phi_arr, 0.0), 1.0, np.cos(phi_arr)))

    ac = a_arr / np.maximum(c_arr, _TINY)
    at = a_arr / np.maximum(t_arr, _TINY)
    q = 1.0 + 1.464 * (ac ** 1.65)

    m1 = 1.13 - 0.09 * ac
    m2 = -0.54 + 0.89 / (0.2 + ac)
    m3 = 0.5 - 1.0 / (0.65 + ac) + 14.0 * ((1.0 - ac) ** 24)
    g = 1.0 + (0.1 + 0.35 * (at ** 2)) * ((1.0 - sinp) ** 2)

    fphi = (ac ** 2 * cosp ** 2 + sinp ** 2) ** 0.25
    fw = np.cos(math.pi * c_arr / (2.0 * np.maximum(b_arr, _TINY)) * np.sqrt(np.maximum(at, 0.0)))

    f = (m1 + m2 * (at ** 2) + m3 * (at ** 4)) * g * fphi / np.sqrt(np.abs(fw) + _TINY)
    F = np.sqrt(1.0 / np.maximum(q, _TINY)) * f
    return sig_arr * np.sqrt(np.maximum(math.pi * a_arr, 0.0)) * F


# ---------------------------------------------------------------------------
# Brokmann step (vectorised, element-group level)
# Source: engine/source/materials/fail/alter/fail_brokmann.F  lines 108-172
# ---------------------------------------------------------------------------

def brokmann_step(
    uvar: np.ndarray,
    off: np.ndarray,
    signxx: np.ndarray,
    signyy: np.ndarray,
    signxy: np.ndarray,
    uparam: np.ndarray,
    timestep: float,
    time: float,
    tdel: np.ndarray,
) -> np.ndarray:
    """Advance the Brokmann subcritical crack growth model for one time step.

    Mirrors ``SUBROUTINE FAIL_BROKMANN`` in
    ``engine/source/materials/fail/alter/fail_brokmann.F`` lines 108-172,
    translated to Python/NumPy (vectorised over elements).

    Parameters
    ----------
    uvar : ndarray, shape (nel, ≥21)
        State variable array (modified in-place).  0-based Python indices:
        [14]=FAIL_B, [15]=CR_LEN(c), [16]=CR_DEPTH(a), [17]=CR_ANG,
        [18]=THK0, [19]=ALDT0, [20]=SIG_COS.
    off : ndarray, shape (nel,)
        Element alive flag (1.0 = alive).
    signxx, signyy, signxy : ndarray, shape (nel,)
        In-plane stress components [model units].
    uparam : ndarray, shape (≥35,)
        Failure model parameters (0-based Python, i.e. Fortran index -1).
    timestep : float
        Current time step [model units].
    time : float
        Current simulation time [model units].
    tdel : ndarray, shape (nel,)
        Failure time tag array (modified in-place).

    Returns
    -------
    ndarray, shape (nel,)
        Boolean mask: True where element has just been flagged as failed.

    Notes
    -----
    Fortran source: ``fail_brokmann.F`` lines 80-172.
    UPARAM Fortran→Python index mapping (1-based→0-based):
      UPARAM(1)→[0]: EXP_N, UPARAM(6)→[5]: K_IC, UPARAM(7)→[6]: K_TH,
      UPARAM(8)→[7]: V0, UPARAM(10)→[9]: ALPHA, UPARAM(30)→[29]: SIG_INI,
      UPARAM(33)→[32]: FAC_M, UPARAM(34)→[33]: FAC_L, UPARAM(35)→[34]: FAC_T.
    """
    # --- read parameters (fail_brokmann.F lines 80-89, 1-based Fortran → 0-based) ---
    exp_n   = float(uparam[0])   # UPARAM(1)
    k_ic    = float(uparam[5])   # UPARAM(6)
    k_th    = float(uparam[6])   # UPARAM(7)
    v0      = float(uparam[7])   # UPARAM(8)
    alpha   = float(uparam[9])   # UPARAM(10)
    sig_ini = float(uparam[29])  # UPARAM(30) – residual surface stress
    fac_m   = float(uparam[32])  # UPARAM(33)
    fac_l   = float(uparam[33])  # UPARAM(34)
    fac_t   = float(uparam[34])  # UPARAM(35)

    # --- unit conversions (fail_brokmann.F lines 93-97) ---
    fac_lenm = _EP06 * fac_l                         # → micrometers
    fac_mpa  = _EM6  * fac_m / (fac_l * fac_t ** 2) # stress → MPa
    kcm      = k_ic  * math.sqrt(fac_l)              # K_IC in [Pa·√m]
    ktm      = k_th  * math.sqrt(fac_l)              # K_TH in [Pa·√m]

    # --- derived constants (fail_brokmann.F lines 99-106) ---
    alphai  = 1.0 - alpha
    exp_m   = 1.0 / (1.0 + _XDAMP)   # fail_brokmann.F line 104

    # --- masks for active (alive, not yet failed, crack present) elements
    # fail_brokmann.F line 109: OFF==1 .and. UVAR(15)==0 .and. UVAR(16)>0
    active = (off == 1.0) & (uvar[:, 14] == 0.0) & (uvar[:, 15] > 0.0)
    if not np.any(active):
        return np.zeros(off.shape[0], dtype=bool)

    idx = np.where(active)[0]

    # --- work on active subset ---
    cr_len   = uvar[idx, 15].copy()   # c  [μm]  (fail_brokmann.F UVAR(17))
    cr_depth = uvar[idx, 16].copy()   # a  [μm]  (fail_brokmann.F UVAR(16)…wait…)
    # NOTE: In fail_brokmann.F the call is NEWMAN_RAJU(UVAR(I,17),UVAR(I,16),...)
    # i.e., first arg is c=UVAR(17)=CR_DEPTH slot and second a=UVAR(16)=CR_LEN slot.
    # Mapping (Fortran 1-based → Python 0-based):
    #   UVAR(16)=CR_LEN=c, UVAR(17)=CR_DEPTH=a
    # Call: NEWMAN_RAJU(UVAR(I,17)=c_arg, UVAR(I,16)=a_arg, UVAR(I,19)=t, UVAR(I,20)=b, ...)
    # Signature: subroutine newman_raju(c, a, t, b, fpi, y)
    # So: c_arg=UVAR(17)=CR_DEPTH[idx,16 py], a_arg=UVAR(16)=CR_LEN[idx,15 py]
    # Matching exactly: c=CR_DEPTH, a=CR_LEN in Newman-Raju call.
    # This is consistent: c is the larger half-axis (surface), a is depth.
    # Fortran names: CR_LEN=crack length (surface half-axis, big), CR_DEPTH=crack depth (small)
    # But the call order (c,a)=(UVAR17,UVAR16) means c=CR_DEPTH, a=CR_LEN
    # Let us faithfully follow the Fortran call (fail_brokmann.F lines 136-137):
    #   CALL NEWMAN_RAJU(UVAR(I,17), UVAR(I,16), UVAR(I,19), UVAR(I,20), FAC_PI2, YA)
    #   CALL NEWMAN_RAJU(UVAR(I,17), UVAR(I,16), UVAR(I,19), UVAR(I,20), FAC_PI0, YC)
    # Python 0-based: uvar[:,16] = UVAR(17), uvar[:,15] = UVAR(16)
    nr_c  = uvar[idx, 16]  # UVAR(17) Fortran → Python [16]  (first NR arg = c)
    nr_a  = uvar[idx, 15]  # UVAR(16) Fortran → Python [15]  (second NR arg = a)
    nr_t  = uvar[idx, 18]  # UVAR(19) → Python [18] = THK0
    nr_b  = uvar[idx, 19]  # UVAR(20) → Python [19] = ALDT0
    sig_cos_prev = uvar[idx, 20].copy()  # UVAR(21) → Python [20]
    cr_ang_2     = uvar[idx, 17] * 2.0  # UVAR(18)*2 → Python [17]

    # --- crack opening stress (fail_brokmann.F lines 114-127) ---
    sig_cos = (
        0.5 * (signxx[idx] + signyy[idx])
        + np.cos(cr_ang_2) * 0.5 * (signxx[idx] - signyy[idx])
        + np.sin(cr_ang_2) * 0.5 * signxy[idx]
    )
    # exponential average filter (fail_brokmann.F line 118)
    sig_cos = sig_cos * alpha + sig_cos_prev * alphai

    # stress rate in MPa/s (fail_brokmann.F lines 121-122)
    dsig_n = (sig_cos - sig_cos_prev) / max(timestep, _TINY)
    dsig_n_mpa_s = dsig_n * fac_mpa / fac_t

    # high-stress-rate correction (fail_brokmann.F lines 125-127)
    sig_cos_filt = sig_cos.copy()
    hi_rate = dsig_n_mpa_s > _DSIG_INI
    if np.any(hi_rate):
        sig_cos_filt[hi_rate] = (
            sig_cos[hi_rate] * (_DSIG_INI / np.abs(dsig_n_mpa_s[hi_rate])) ** exp_m
        )

    # save filtered SIG_COS (fail_brokmann.F line 124)
    uvar[idx, 20] = sig_cos

    # --- residual stress subtraction (fail_brokmann.F lines 131-132) ---
    sig_cosw = np.maximum(0.0, sig_cos_filt - sig_ini)

    # --- Newman-Raju geometry factors (fail_brokmann.F lines 136-137) ---
    ya = newman_raju_vec(nr_c, nr_a, nr_t, nr_b, 0.5)  # FAC_PI2 = 0.5
    yc = newman_raju_vec(nr_c, nr_a, nr_t, nr_b, 0.0)  # FAC_PI0 = 0.0

    # --- stress intensity factors (fail_brokmann.F lines 138-139) ---
    # K1_A = YA * sig_cosw * sqrt(pi * UVAR(16) * 1e-6)
    # K1_C = YC * sig_cosw * sqrt(pi * UVAR(17) * 1e-6)
    # UVAR(16)=CR_LEN (Python [15]=nr_a), UVAR(17)=CR_DEPTH (Python [16]=nr_c)
    k1_a = ya * sig_cosw * np.sqrt(math.pi * nr_a * _EM6)
    k1_c = yc * sig_cosw * np.sqrt(math.pi * nr_c * _EM6)

    # --- rupture criterion #1 (fail_brokmann.F lines 142-149) ---
    just_failed = np.zeros(len(idx), dtype=bool)
    fail1 = k1_a >= kcm
    if np.any(fail1):
        tdel[idx[fail1]] = time
        uvar[idx[fail1], 14] = 1.0
        just_failed |= fail1

    # --- crack growth velocities (fail_brokmann.F lines 151-152) ---
    v_a = np.zeros(len(idx))
    v_c = np.zeros(len(idx))
    grow_a = (k1_a >= ktm) & (k1_a < kcm)
    grow_c = (k1_c >= ktm) & (k1_a < kcm)
    if np.any(grow_a):
        v_a[grow_a] = v0 * (k1_a[grow_a] / kcm) ** exp_n
    if np.any(grow_c):
        v_c[grow_c] = v0 * (k1_c[grow_c] / kcm) ** exp_n

    # --- crack growth integration (fail_brokmann.F lines 155-158) ---
    da = v_a * timestep * fac_lenm
    dc = v_c * timestep * fac_lenm
    uvar[idx, 15] += da   # UVAR(16) += DA  (CR_LEN, Python [15])
    uvar[idx, 16] += dc   # UVAR(17) += DC  (CR_DEPTH, Python [16])

    # --- rupture criterion #2 (fail_brokmann.F lines 161-169) ---
    # Re-evaluate K1_A with updated crack length (still uses sig_cos unfiltered)
    nr_a_new = uvar[idx, 15]
    k1_a2 = ya * sig_cos * np.sqrt(math.pi * nr_a_new * _EM6)
    not_yet_failed = uvar[idx, 14] == 0.0
    fail2 = (k1_a2 >= kcm) & not_yet_failed
    if np.any(fail2):
        tdel[idx[fail2]] = time
        uvar[idx[fail2], 14] = 1.0
        just_failed |= fail2

    # Map back to full-element result array
    result = np.zeros(off.shape[0], dtype=bool)
    result[idx] = just_failed
    return result


# ---------------------------------------------------------------------------
# Dispatch-compatible shell_step / solid_step wrappers
# ---------------------------------------------------------------------------

def _extract_params(fail) -> np.ndarray:
    """Build a ≥35-element uparam array from the fail object's params dict."""
    p   = fail.params
    out = np.zeros(35, dtype=float)
    # Fortran index (1-based) → Python 0-based
    out[0]  = float(p.get("exp_n",   p.get("EXP_N",   16.0)))  # UPARAM(1)
    out[5]  = float(p.get("k_ic",    p.get("K_IC",     1.0)))   # UPARAM(6)
    out[6]  = float(p.get("k_th",    p.get("K_TH",     0.0)))   # UPARAM(7)
    out[7]  = float(p.get("v0",      p.get("V0",       1.0)))   # UPARAM(8)
    out[9]  = float(p.get("alpha",   p.get("ALPHA",    0.9)))   # UPARAM(10)
    out[29] = float(p.get("sig_ini", p.get("SIG_INI",  0.0)))   # UPARAM(30)
    out[32] = float(p.get("fac_m",   p.get("FAC_M",    1.0)))   # UPARAM(33)
    out[33] = float(p.get("fac_l",   p.get("FAC_L",    1.0)))   # UPARAM(34)
    out[34] = float(p.get("fac_t",   p.get("FAC_T",    1.0)))   # UPARAM(35)
    return out


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Brokmann shell-layer damage step; returns broken mask.

    ``fail`` must carry a ``state`` dict with the Brokmann UVAR arrays and
    a ``params`` dict with the material parameters.

    Notes
    -----
    Fortran: ``fail_brokmann.F`` — shell-specific entry point.
    """
    uparam = _extract_params(fail)
    sig_arr = np.asarray(sig, dtype=float)
    nel = sig_arr.shape[0]

    state = getattr(fail, "state", {})
    uvar  = state.get("uvar", np.zeros((nel, 21)))
    off   = state.get("off",  np.ones(nel))
    tdel  = state.get("tdel", np.zeros(nel))
    time  = state.get("time", 0.0)

    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2] if sig_arr.shape[1] > 2 else np.zeros(nel)

    broken = brokmann_step(uvar, off, sxx, syy, sxy, uparam, dt, time, tdel)

    # Sync off-flag: elements that just failed lose their alive flag
    off[broken] = 0.0
    # Update damage proxy: failed → dama = 1
    dama[broken] = 1.0
    return broken


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Brokmann solid damage step (uses first 3 stress components)."""
    sig_arr = np.asarray(sig, dtype=float)
    return shell_step(fail, sig_arr[:, :3], d_epsp, deps, dt, dama, tstar=tstar)


# ---------------------------------------------------------------------------
# Brokmann failure check and Paris law crack growth
# Upstream Fortran source:
#   engine/source/materials/fail/alter/fail_brokmann.F lines 136-169
#   common_source/fail/newman_raju.F90 lines 51-102
# ---------------------------------------------------------------------------

def _get_group_attr(group: Any, names: list[str], default: Any = None) -> Any:
    """Extract an attribute or dictionary key from group/dict/state."""
    if isinstance(group, dict):
        for name in names:
            if name in group and group[name] is not None:
                return group[name]
        for sub in ("params", "state", "props", "fail"):
            if sub in group and isinstance(group[sub], dict):
                for name in names:
                    if name in group[sub] and group[sub][name] is not None:
                        return group[sub][name]
        return default

    for name in names:
        if hasattr(group, name):
            val = getattr(group, name)
            if val is not None:
                return val

    for sub in ("params", "state", "props", "fail", "mat"):
        if hasattr(group, sub):
            sub_obj = getattr(group, sub)
            if isinstance(sub_obj, dict):
                for name in names:
                    if name in sub_obj and sub_obj[name] is not None:
                        return sub_obj[name]
            elif sub_obj is not None:
                for name in names:
                    if hasattr(sub_obj, name):
                        val = getattr(sub_obj, name)
                        if val is not None:
                            return val

    return default


def brokmann_check(
    group: Any,
    stress_eq: float | np.ndarray,
    phi: float = math.pi / 2,
) -> bool | np.ndarray:
    """Check Brokmann failure criterion for crack growth / critical crack size.

    Upstream Fortran source:
      ``engine/source/materials/fail/alter/fail_brokmann.F`` lines 136-149, 161-169
      ``common_source/fail/newman_raju.F90`` lines 51-102

    Checks whether:
      1. Crack depth reaches or exceeds critical crack size: a >= a_crit
      2. Stress intensity factor reaches or exceeds fracture toughness: K_I >= K_IC
      3. Crack penetrates plate thickness: a >= t

    Parameters
    ----------
    group : Any
        Crack group, element group, or dictionary with crack and material properties.
    stress_eq : float or np.ndarray
        Applied equivalent tensile / opening stress.
    phi : float, optional
        Parametric angle along crack front [radians]. Default is pi/2 (deepest point).

    Returns
    -------
    bool or np.ndarray
        True where the failure criterion is met.
    """
    a = _get_group_attr(group, ["a", "cr_depth", "CR_DEPTH", "crack_depth", "a0", "crack_a", "depth"], default=None)
    c = _get_group_attr(group, ["c", "cr_len", "CR_LEN", "crack_length", "c0", "crack_c", "length"], default=None)
    t = _get_group_attr(group, ["t", "thick", "thickness", "THK0", "thk0", "thk"], default=None)
    b = _get_group_attr(group, ["b", "width", "half_width", "ALDT0", "aldt0", "w"], default=None)
    k_ic = _get_group_attr(group, ["k_ic", "K_IC", "kic", "KIC", "K_c", "k_c", "Kc", "KC", "kcm", "KCM"], default=None)
    a_crit = _get_group_attr(group, ["a_crit", "acrit", "a_c", "critical_a", "A_CRIT", "ACRIT"], default=None)

    # UVAR state array fallback if present
    uvar = _get_group_attr(group, ["uvar", "UVAR"], default=None)
    if uvar is not None:
        uvar_arr = np.asarray(uvar)
        if uvar_arr.ndim == 1 and uvar_arr.shape[0] >= 17:
            if a is None:
                a = float(uvar_arr[15])
            if c is None:
                c = float(uvar_arr[16])
            if t is None and uvar_arr.shape[0] >= 19:
                t = float(uvar_arr[18])
            if b is None and uvar_arr.shape[0] >= 20:
                b = float(uvar_arr[19])
        elif uvar_arr.ndim == 2 and uvar_arr.shape[1] >= 17:
            if a is None:
                a = uvar_arr[:, 15]
            if c is None:
                c = uvar_arr[:, 16]
            if t is None and uvar_arr.shape[1] >= 19:
                t = uvar_arr[:, 18]
            if b is None and uvar_arr.shape[1] >= 20:
                b = uvar_arr[:, 19]

    if a is None:
        a = 0.001
    if c is None:
        c = a * 2.0 if not isinstance(a, np.ndarray) else a * 2.0
    if t is None:
        t = 10.0 * np.max(a) if isinstance(a, np.ndarray) else 10.0 * a
    if b is None:
        b = 10.0 * np.max(c) if isinstance(c, np.ndarray) else 10.0 * c

    is_scalar = (np.ndim(stress_eq) == 0 and np.ndim(a) == 0)

    if is_scalar:
        a_val = float(a)
        c_val = float(c)
        t_val = float(t)
        b_val = float(b)
        sig_val = float(stress_eq)
        failed = False

        if a_crit is not None and a_val >= float(a_crit):
            failed = True
        if t_val > 0.0 and a_val >= t_val:
            failed = True

        K = 0.0
        if sig_val > 0.0 and a_val > 0.0:
            K = float(newman_raju_K(a_val, c_val, t_val, b_val, sig_val, phi=phi))
            if k_ic is not None and K >= float(k_ic):
                failed = True

        # Store results on group if mutable
        if hasattr(group, "failed"):
            group.failed = bool(failed)
        elif hasattr(group, "is_failed"):
            group.is_failed = bool(failed)
        elif isinstance(group, dict):
            group["failed"] = bool(failed)

        if hasattr(group, "K"):
            group.K = K
        elif isinstance(group, dict):
            group["K"] = K

        return bool(failed)

    # Vectorized path
    a_arr = np.asarray(a, dtype=float)
    c_arr = np.asarray(c, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    sig_arr = np.asarray(stress_eq, dtype=float)

    shape = np.broadcast_shapes(a_arr.shape, sig_arr.shape)
    failed = np.zeros(shape, dtype=bool)

    if a_crit is not None:
        failed |= (a_arr >= float(a_crit))
    failed |= (a_arr >= t_arr)

    pos_sig = sig_arr > 0.0
    if np.any(pos_sig):
        K_arr = newman_raju_K(a_arr, c_arr, t_arr, b_arr, sig_arr, phi=phi)
        if k_ic is not None:
            failed |= ((K_arr >= float(k_ic)) & pos_sig)
        if hasattr(group, "K"):
            group.K = K_arr
        elif isinstance(group, dict):
            group["K"] = K_arr

    if hasattr(group, "failed"):
        group.failed = failed
    elif hasattr(group, "is_failed"):
        group.is_failed = failed
    elif isinstance(group, dict):
        group["failed"] = failed

    return failed


def paris_law_da(delta_K: float, C: float, m: float, dN: float = 1.0) -> float:
    """Paris–Erdogan crack growth increment da = C * (Delta_K)^m * dN.

    Upstream Fortran source:
      ``engine/source/materials/fail/alter/fail_brokmann.F`` lines 151-158
    """
    if delta_K <= 0.0:
        return 0.0
    return float(C) * (float(delta_K) ** float(m)) * float(dN)


class BrokmannGroup:
    """Element or material group state for the Brokmann sub-critical crack growth model.

    Tracks crack depth a, half-length c, plate geometry (t, b), and fracture parameters.
    """

    def __init__(
        self,
        a: float = 0.001,
        c: float = 0.002,
        t: float = 0.010,
        b: float = 0.050,
        k_ic: float | None = 25.0e6,
        a_crit: float | None = None,
        C: float = 1.0e-11,
        m: float = 3.0,
        phi: float = math.pi / 2,
    ) -> None:
        self.a = float(a)
        self.c = float(c)
        self.t = float(t)
        self.b = float(b)
        self.k_ic = float(k_ic) if k_ic is not None else None
        self.a_crit = float(a_crit) if a_crit is not None else None
        self.C = float(C)
        self.m = float(m)
        self.phi = float(phi)
        self.failed = False
        self.K = 0.0

    def check(self, stress_eq: float) -> bool:
        """Check failure criterion under given equivalent stress."""
        self.failed = bool(brokmann_check(self, stress_eq, phi=self.phi))
        return self.failed

    def grow(self, delta_stress: float, dN: float = 1.0) -> tuple[float, float]:
        """Advance crack dimensions (a, c) by dN cycles using Paris law."""
        if delta_stress <= 0.0 or self.failed:
            return self.a, self.c

        # Deepest point (phi = pi/2) governs depth growth da
        k_a = float(newman_raju_K(self.a, self.c, self.t, self.b, delta_stress, phi=math.pi / 2))
        da = paris_law_da(k_a, self.C, self.m, dN=dN)

        # Surface point (phi = 0) governs length growth dc
        k_c = float(newman_raju_K(self.a, self.c, self.t, self.b, delta_stress, phi=0.0))
        dc = paris_law_da(k_c, self.C, self.m, dN=dN)

        self.a += da
        self.c += dc

        self.check(delta_stress)
        return self.a, self.c


def register(dispatcher_dict: dict | None = None) -> None:
    """Register Brokmann failure model with a failure model dispatcher.

    Upstream Fortran source:
      ``engine/source/materials/fail/alter/fail_brokmann.F``
    """
    mod = sys.modules[__name__]
    if dispatcher_dict is not None:
        dispatcher_dict["BROKMANN"] = mod
        dispatcher_dict["FAIL_BROKMANN"] = mod
        dispatcher_dict["ALTER_BROKMANN"] = mod
    try:
        from pyradioss import failure
        if hasattr(failure, "FAILURE_MODELS") and isinstance(failure.FAILURE_MODELS, dict):
            failure.FAILURE_MODELS["BROKMANN"] = mod
            failure.FAILURE_MODELS["FAIL_BROKMANN"] = mod
            failure.FAILURE_MODELS["ALTER_BROKMANN"] = mod
    except ImportError:
        pass


__all__ = [
    "newman_raju",
    "newman_raju_vec",
    "newman_raju_K",
    "brokmann_step",
    "brokmann_check",
    "BrokmannGroup",
    "paris_law_da",
    "register",
    "shell_step",
    "solid_step",
]
