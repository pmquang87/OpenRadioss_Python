"""
Friction MODELS (Ifric > 0) — the MFROT mu(p, v) laws and the IFQ force
filter of /INTER/TYPE7 (M15).

Fortran origin
--------------
``engine/source/interfaces/int07/i7for3.F`` — the "Friction coefficient
computation" block (isotropic branch, IORTHFRIC = 0) evaluates, per
active pair, a friction coefficient from the contact PRESSURE and the
tangential SLIP SPEED before the tangential-force block applies it:

    p  = -FNI / AREA          FNI = -STIF*PENE (+ the damper part), so
                              p is the (positive) normal force over the
                              CURRENT area of the main segment,
                              AREA = 1/2 |(x3-x1) x (x4-x2)|
    v  = |v_rel - (v_rel.n) n|    tangential relative speed

    MFROT = 0   Coulomb        mu = Fric
    MFROT = 1   "viscous"      mu = Fric + (C1 + C4 p) p
                (generalized        + (C2 + C3 p) v + C5 v^2
                 polynomial)
    MFROT = 2   Darmstadt      mu = Fric + C1 e^(C2 v) p^2
                                        + C3 e^(C4 v) p + C5 e^(C6 v)
    MFROT = 3   Renard         piecewise in v (C5 = Vcr1, C6 = Vcr2):
                  0 <= v <= C5 : mu = C1 + (C3-C1) (v/C5)(2 - v/C5)
                  C5 < v < C6  : mu = C3 + (C4-C3) xi^2 (3 - 2 xi),
                                 xi = (v-C5)/(C6-C5)
                  v >= C6      : mu = C2 - (C2-C4)/(1 + (C2-C4)(v-C6)^2)
    MFROT = 4   exp. decay     mu = C1 + (Fric - C1) e^(-C2 v)

    and always  mu = max(mu, 1e-30)   (the EM30 floor of i7for3.F)

The reader (``hm_read_inter_type07.F``, checked, not the docs) maps the
card fields as: ``Ifric`` -> MFROT (IPARI(30)), ``Ifiltr`` -> IFQ
(IPARI(31)), ``Xfreq`` -> ALPHA -> XFILTR with

    IFQ = 1 : XFILTR = Xfreq            (a direct filter coefficient)
    IFQ = 2 : XFILTR = 2 pi / Xfreq     (Xfreq = a period in CYCLES)
    IFQ = 3 : XFILTR = 2 pi * Xfreq     (Xfreq = a cutoff FREQUENCY;
                                         the per-cycle coefficient is
                                         XFILTR * dt)

(``FOUR*ATAN2(ONE,ZERO)`` is 2 pi), erroring on XFILTR < 0 and on
XFILTR > 1 for IFQ <= 2 — mirrored by the port's reader. C1..C5 are read
when Ifric > 0 and C6 when Ifric > 1 (FRIC_P(1..6)).

The IFQ filter itself (the "TOTAL (VISCOUS) FORMULATION + FRICTION
FILTERING" block of i7for3.F) is a first-order low-pass — an exponential
moving average on the tangential force with the per-pair anchor CAND_F:

    F_t^filtered = ALPHA * F_t^target + (1 - ALPHA) * F_t^previous

with ALPHA = XFILTR for IFQ 1/2 and ALPHA from the cutoff frequency for
IFQ 3. DOCUMENTED DEVIATION: the fetched source computes the IFQ = 3
per-cycle coefficient as ``ALPHA = MAX(ONE, ALPHA0*DT12)`` — which is
always 1 (ALPHA0*dt << 1 on any resolved run), i.e. it disables the
filter the card asked for; the documented behaviour (and the obvious
intent of the surrounding code) is the cutoff-frequency first-order
filter, so the port uses ``ALPHA = MIN(ONE, XFILTR*dt)`` and records the
difference here rather than reproducing what looks like an upstream MAX/
MIN slip. IFQ >= 10 selects the INCREMENTAL (stiffness) tangential
formulation (MODFR = 2, Iform = 2) — ``apply_incremental_stiffness``
below (M477).  The Fortran reference is i7for3.F lines 2310-2356:

    FX = CAND_FX + ALPHA * STIF0 * VX * DT12
    FTN = FX*N1 + FY*N2 + FZ*N3       (normal projection)
    FX = FX - FTN*N1                   (tangential plane)
    BETA = min(1, XMU * sqrt(FN/FT))  (Coulomb return mapping)
    FXT = FX * BETA; CAND_FX = FXT     (store return-mapped force)

Note: the Fortran uses DT12 (= (DT1+DT2)/2, the leapfrog midstep) while
the port uses dt (= DT1, the cycle timestep).  For constant dt these are
identical; for variable dt the difference is second-order.

TYPE11 (a DOCUMENTED PORT EXTENSION — checked against the source): the
original engine never evaluates friction models for edge-to-edge
contact. ``i11mainf.F`` hardcodes ``MFROT = 0`` before the force call
and ``I11FOR3`` receives only the constant per-pair FRICC — no
FRIC_COEFS at all; the TYPE11 reader has no Ifric/Ifiltr/C fields
either. The port EXTENDS the TYPE7 evaluation to TYPE11 deliberately
(both solvers), defining the edge-pair contact pressure as

    p = f_n / (L_main * gap_pair)

— the current main-edge length times the pair gap, i.e. the tributary
contact strip of a line contact whose half-width is the physical gap
(for Igap = 1 shells the gap IS the surface half-thickness sum). An edge
pair has no segment area, so SOME definition is required; this one is
computable per pair from quantities the kernel already carries, reduces
to sensible magnitudes for shell-edge decks, and is stated here so a
calibrated C1 can be converted. Decks that must match OpenRadioss
byte-for-byte should keep Ifric = 0 on TYPE11.

The implicit solver evaluates the SAME laws at their STATIC LIMIT — see
``mu_static`` below and implicit/contact.py (M15).
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20

#: the mu floor of i7for3.F (XMU = MAX(XMU, EM30))
_MU_FLOOR = 1e-30


def mu_kinetic(mfrot, mu0, c, p, v):
    """The MFROT friction coefficient mu(p, v) of i7for3.F (module
    docstring), vectorized over pairs.

    ``mfrot`` 1..4, ``mu0`` the card Fric, ``c`` the C1..C6 array (6,),
    ``p`` the contact pressure (>= 0) per pair, ``v`` the tangential slip
    speed per pair. Returns mu floored at 1e-30 exactly like the
    original. (mfrot = 0 never reaches here — the constant-mu fast path
    is kept verbatim in the interfaces, bit-identical to M4.)"""
    if mfrot == 1:
        # generalized polynomial ("viscous friction") — i7for3 MFROT==1
        mu = mu0 + (c[0] + c[3] * p) * p + (c[1] + c[2] * p) * v \
            + c[4] * v * v
    elif mfrot == 2:
        # Darmstadt law — i7for3 MFROT==2 (note the original's implicit
        # twin i7keg3.F drops the Fric term here; the engine i7for3 form
        # with Fric is the one both port solvers use — consistency of the
        # two solvers wins, recorded in the module docstring)
        mu = mu0 + c[0] * np.exp(c[1] * v) * p * p \
            + c[2] * np.exp(c[3] * v) * p + c[4] * np.exp(c[5] * v)
    elif mfrot == 3:
        # Renard law — piecewise in v, C5/C6 the critical speeds
        v = np.asarray(v, dtype=float)
        vc1 = max(c[4], 1e-30)
        vc2 = max(c[5], vc1 + 1e-30)
        mu = np.empty_like(v)
        lo = v <= vc1
        mid = (~lo) & (v < vc2)
        hi = (~lo) & (~mid)
        vv1 = v[lo] / vc1
        mu[lo] = c[0] + (c[2] - c[0]) * vv1 * (2.0 - vv1)
        xi = (v[mid] - vc1) / (vc2 - vc1)
        mu[mid] = c[2] + (c[3] - c[2]) * (3.0 - 2.0 * xi) * xi * xi
        dmu = c[1] - c[3]
        denom = 1.0 + dmu * (v[hi] - vc2) ** 2
        safe_denom = np.where(np.abs(denom) > 1e-15, denom, 1e-15)
        mu[hi] = c[1] - dmu / safe_denom
    elif mfrot == 4:
        # exponential decay (static -> dynamic) — i7for3 MFROT==4
        mu = c[0] + (mu0 - c[0]) * np.exp(-c[1] * v)
    else:
        raise ValueError(f"MFROT={mfrot} (Ifric 1..4)")
    return np.maximum(mu, _MU_FLOOR)


def mu_static(mfrot, mu0, c, p):
    """The STATIC LIMIT mu(p, v=0) of the MFROT laws and its pressure
    derivative — what the IMPLICIT return mapping uses for the Coulomb
    cone (M15; the port convention: rate devices under implicit reduce
    LOUDLY to their static limit, never fed the pseudo-velocity du/1 —
    the original's I7KFOR3 does feed its mu(p,v) the increment fields,
    i.e. a step-size-dependent pseudo-rate, which the port deliberately
    does not reproduce; see implicit/contact.py).

    Setting v = 0 (e^0 = 1, the Renard law lands on its static branch
    value C1):

        MFROT 1 : mu = mu0 + C1 p + C4 p^2      mu' = C1 + 2 C4 p
        MFROT 2 : mu = mu0 + C1 p^2 + C3 p + C5 mu' = 2 C1 p + C3
        MFROT 3 : mu = C1 (static coefficient)  mu' = 0
        MFROT 4 : mu = mu0                      mu' = 0

    Returns (mu, dmu_dp); where the 1e-30 floor clamps, dmu_dp = 0 (the
    clamped branch is constant)."""
    p = np.asarray(p, dtype=float)
    if mfrot == 1:
        mu = mu0 + (c[0] + c[3] * p) * p
        dmu = c[0] + 2.0 * c[3] * p
    elif mfrot == 2:
        mu = mu0 + c[0] * p * p + c[2] * p + c[4]
        dmu = 2.0 * c[0] * p + c[2]
    elif mfrot == 3:
        mu = np.full_like(p, c[0])
        dmu = np.zeros_like(p)
    elif mfrot == 4:
        mu = np.full_like(p, mu0)
        dmu = np.zeros_like(p)
    else:
        raise ValueError(f"MFROT={mfrot} (Ifric 1..4)")
    clamped = mu < _MU_FLOOR
    return np.where(clamped, _MU_FLOOR, mu), np.where(clamped, 0.0, dmu)


def has_velocity_terms(mfrot, c) -> bool:
    """True when the MFROT law actually depends on the slip speed for
    the given coefficients — drives the one-time implicit static-limit
    warning (a purely pressure-dependent MFROT 1/2 deck needs none)."""
    if mfrot == 1:
        return bool(c[1] != 0.0 or c[2] != 0.0 or c[4] != 0.0)
    if mfrot == 2:
        return bool(c[1] != 0.0 or c[3] != 0.0 or c[5] != 0.0)
    return mfrot in (3, 4)


def filter_alpha(ifq, xfiltr, dt):
    """The per-cycle IFQ filter coefficient ALPHA (module docstring).
    IFQ 1/2/11/12: the constant XFILTR; IFQ 3/13: min(1, XFILTR*dt) — the
    cutoff-frequency form (the documented MAX->MIN deviation)."""
    if ifq in (3, 13):
        return min(1.0, xfiltr * dt)
    return xfiltr


def apply_filter(keys, ft_target, alpha, filt_keys, filt_vals):
    """One IFQ filter step over the active pairs (the CAND_F exponential
    moving average of i7for3.F):

        F = alpha * F_target + (1 - alpha) * F_prev(key)

    ``keys`` (m,) int64 pair keys of the ACTIVE pairs, ``ft_target``
    (m, 3) the unfiltered tangential force, ``filt_keys``/``filt_vals``
    the SORTED anchor store from the previous cycle (missing key =>
    F_prev = 0, the IFPEN reset semantics: a pair that separated starts
    its history afresh). Returns (ft, new_keys, new_vals) with the new
    store sorted — pairs not active this cycle are dropped, exactly the
    scope of the original's IFPEN bookkeeping."""
    prev = np.zeros_like(ft_target)
    if len(filt_keys):
        pos = np.searchsorted(filt_keys, keys)
        pos = np.minimum(pos, len(filt_keys) - 1)
        hit = filt_keys[pos] == keys
        prev[hit] = filt_vals[pos[hit]]
    ft = alpha * ft_target + (1.0 - alpha) * prev
    order = np.argsort(keys)
    return ft, keys[order], ft[order]


def apply_incremental_stiffness(keys, k, v_rel, dt, normal, mu, fn, alpha, filt_keys, filt_vals):
    """One incremental stiffness tangential force step (MODFR=2 / IFQ>=10)
    over the active pairs — the CAND_FX/FY/FZ explicit path in i7for3.F
    lines 2310-2356 (M477).

    Returns (ft, new_keys, new_vals), where ``ft`` is the tangential force
    vector.  The incremental formula (Fortran lines 2320-2340):

        F_trial = CAND_F + alpha * k * v_rel * dt
        F_trial_tan = F_trial - (F_trial . normal) * normal
        beta = min(1, mu * |Fn| / |F_trial_tan|)
        ft = F_trial_tan * beta
        CAND_F = ft  (stored for next cycle — the return-mapped value)

    This implicitly handles stick/slip transitions: beta=1 in stick
    (elastic accumulation), beta<1 in slip (Coulomb saturation at mu*Fn).
    """
    prev = np.zeros_like(v_rel)
    if len(filt_keys):
        pos = np.searchsorted(filt_keys, keys)
        pos = np.minimum(pos, len(filt_keys) - 1)
        hit = filt_keys[pos] == keys
        prev[hit] = filt_vals[pos[hit]]

    # F_trial = CAND_F + alpha * k * v_rel * dt
    f_trial = prev + (alpha * k * dt)[:, None] * v_rel
    
    # Project to tangential plane (F_trial_tan)
    ftn = np.einsum('ij,ij->i', f_trial, normal)
    f_trial_tan = f_trial - ftn[:, None] * normal
    
    # Frictional limit
    ft_mag = np.linalg.norm(f_trial_tan, axis=1)
    
    # beta = min(1.0, mu * fn / ft_mag)
    # Handle division by zero where ft_mag is extremely small
    safe_mag = np.maximum(ft_mag, 1e-30)
    beta = np.clip((mu * np.maximum(fn, 0.0)) / safe_mag, 0.0, 1.0)
    
    ft = f_trial_tan * beta[:, None]
    
    order = np.argsort(keys)
    return ft, keys[order], ft[order]
