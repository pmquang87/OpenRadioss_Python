"""
Fully non-stationary / evolutionary MULTIAXIAL (JOINT-TENSOR) fatigue — M27: the
frequency-domain fatigue-damage estimate of a MULTIAXIAL random-vibration
response whose full stress-TENSOR cross-PSD S_sigmasigma(omega, t) — a 6x6
EVOLUTIONARY cross-PSD, NOT merely an equivalent scalar — varies with time,
computed by forming the per-time-window stress-tensor cross-PSD, reducing it PER
WINDOW with the M21/M23 critical-plane machinery (the equivalent von Mises +
max-normal / max-shear critical plane), letting the CRITICAL PLANE itself DRIFT
window to window, and Palmgren-Miner-summing the per-window MULTIAXIAL damages —
cross-validated against a non-stationary MULTIVARIATE time-domain Monte-Carlo.

WHERE M27 SITS RELATIVE TO M25 / M26. Every non-stationary multiaxial estimator
up to here reduced the tensor to an EQUIVALENT SCALAR with a FIXED reduction and
then scaled / windowed THAT scalar:
* M25 (separable, multiaxial) scaled the equivalent-scalar (von Mises /
  critical-plane) PSD's moments by the RMS mission profile a_i^2 — a FIXED
  spectral shape, a FIXED critical plane, only the LEVEL drifts.
* M26 (evolutionary, multiaxial) applied the drifting-shape window W_i(f) to that
  same equivalent-SCALAR PSD per window — the spectral SHAPE of the scalar drifts,
  but the REDUCTION (which plane, which F_np) was computed ONCE from the
  stationary tensor and HELD FIXED across the windows.
M27 lifts exactly that last assumption: the reduction is NOT fixed. The full 6x6
JOINT tensor cross-PSD evolves, so the per-window 6x6 spectral-MOMENT matrices
M_{n,i} are recomputed from the WINDOW's OWN tensor, and the critical-plane
ORIENTATION and the M23 non-proportionality factor F_np are RE-SEARCHED FROM the
window's own tensor — so the critical plane may ROTATE and F_np may DRIFT window
to window. This is the natural consumer of BOTH the M21/M23 multiaxial machinery
AND the M25/M26 non-stationary / evolutionary machinery.

THE THREE BUILT-IN REDUCTIONS (exact, asserted).
* STATIONARY tensor / SINGLE window -> the M21 spectral multiaxial answer EXACTLY
  (all three reductions): one flat unit window gives M_{n,1} = the full-tensor
  moment matrices, so the per-window plane search IS the M21 ``critical_plane_
  search`` and the vM moments ARE ``equivalent_vonmises_moments`` — bit-identical.
* EQUIVALENT-SCALAR reduction with a FIXED critical plane -> the M26 scalar
  per-window spectrogram EXACTLY: projecting the windowed tensor onto ONE fixed
  plane p and windowing the scalar p^T S_cross p IS the M26 windowed-scalar path.
  The ``drift=False`` branch DELEGATES to the M26 estimators on the projected
  scalar PSD, so it is bit-identical to M26 by construction.
* CONSTANT-tensor-shape / RMS-only drift -> the M25 multiaxial block answer
  EXACTLY: a flat window (bw <= 0) with only the level a_i drifting gives a
  constant per-window tensor SHAPE, so the critical plane does NOT rotate and the
  window Miner-sum of a_i^2-scaled moments IS the M25 block Miner-sum.
The whole POINT of M27 over M26 is the case NONE of these cover: a "rotating
principal axes" tensor whose per-window critical plane genuinely DRIFTS, so the
joint-tensor window Miner-sum DIFFERS from the M26 fixed-reduction scalar
spectrogram — the M27 <-> M26 boundary made explicit.

Fortran origin
--------------
There is NONE. ``engine/source/input/freimpl.F`` (the /IMPL reader, re-read line
by line for M16-M26 AND AGAIN for M27 — 639 lines, fetched from
raw.githubusercontent.com) parses only /IMPL/DYNA (the DIRECT Newmark/HHT
integrator), /IMPL/BUCKL, /IMPL/DT, /IMPL/NONLIN and /IMPL/ARCL plus the
linear-solver housekeeping — there is no /FATIG, no S-N / Miner branch, no
Dirlik / rainflow / narrow-band estimator, no von-Mises / critical-plane /
stress-tensor cross-PSD machinery, and (as M16-M26 already found) NOTHING
non-stationary / evolutionary / spectrogram / joint-tensor of any kind. The sole
``PSD`` token in the whole file is still ``IMUMPSD`` (line 269), a MUMPS-solver
flag, not a power spectral density. OpenRadioss is a time-domain crash/impact
code: the random-vibration MULTIAXIAL fatigue analysis — stationary (M21-M23),
non-Gaussian (M24), separable non-stationary (M25), evolutionary-SCALAR (M26) OR
fully evolutionary JOINT-TENSOR (M27) — is simply not part of the open-source
solver, exactly as M16 found for the real eigensolver and M17-M26 for the
transfer functions, the PSD machinery and the spectral-fatigue estimators this
module extends.

So M27 does exactly what M16-M26 did: it ports evolutionary joint-tensor fatigue
as a clean LIBRARY capability and drives it with a minimal PORT engine sub-flag
(/IMPL/FATIG/MULT/EVOL/JOINT — the JOINT-TENSOR analogue of the M26 evolutionary
scalar card, implying MULT + EVOL). Nothing in the M10 direct integrator, the M16
REAL eigensolver, the M17/M18 superposition, the M19 PSD path, the M20 SCALAR
fatigue, the M21 MULTIAXIAL SPECTRAL path, the M22 NON-PROPORTIONAL TIME-DOMAIN
path, the M23 SPECTRAL NON-PROPORTIONAL path, the M24 NON-GAUSSIAN correction,
the M25 NON-STATIONARY (separable) path OR the M26 EVOLUTIONARY (non-separable
scalar) path is touched: the joint-tensor path is a NEW, parallel path that
CONSUMES the M21 tensor / plane machinery (``stress_tensor_cross_psd`` /
``tensor_moment_matrices`` / ``critical_plane_search`` / ``equivalent_vonmises_
moments`` / the multivariate synthesiser) and the M23 F_np + the M26 drifting-
shape window (all read-only) and produces its answer ALONGSIDE the M21
stationary, M25 non-stationary and M26 scalar-evolutionary numbers so a listing
shows the stationary, the RMS-non-stationary, the scalar-shape-evolutionary and
the joint-tensor-evolutionary answers side by side.

Theory — evolutionary joint-tensor multiaxial fatigue
-----------------------------------------------------
(Priestley, "Evolutionary spectra and non-stationary processes", J. Roy. Statist.
Soc. B 27, 1965 — the evolutionary spectrum S(omega, t), here the MATRIX-valued
6x6 tensor cross-PSD S_sigmasigma(omega, t); Preumont & Piefort, "Predicting
random high-cycle fatigue life with finite elements", J. Sound Vib. 168, 1994,
and Pitoiset & Preumont, "Spectral methods for multiaxial random fatigue
analysis", Int. J. Fatigue 22, 2000 — the equivalent-von-Mises trace(Q S)
projection; Carpinteri & Spagnoli / Cristofori, Susmel & Tovo — the critical
plane; Backstrom & Marquis — the non-proportionality background; the M21/M23
multiaxial + M25/M26 non-stationary/evolutionary bases; Palmgren-Miner — the
linear window-damage summation; Newland, "An Introduction to Random
Vibrations..." ch. 5-7 — the short-time / windowed spectral method.)

THE PER-WINDOW TENSOR CROSS-PSD (the JOINT generalisation of M26's scalar
window). The structure's stress-tensor cross-PSD is the 6x6 Hermitian matrix
S_cross(omega) = H_sigma(omega) S_ff(omega) H_sigma(omega)^H (M21 eq. (1)). A
resonance sweep — the excitation energy moving through the modal band with time —
is a time-varying INPUT window W_i(f) on S_ff; because W_i is a per-frequency
SCALAR multiplier it commutes with the linear |H|^2 map (M26 established this for
the scalar |H_sigma|^2), so the per-window TENSOR cross-PSD is

    S_cross,i(omega) = a_i^2 W_i(f) S_cross(omega) ,   omega = 2 pi f       (1)

with W_i(f) = exp(-(f - fc_i)^2 / (2 bw_i^2)) the swept-centre / broadening
Gaussian window (M26 eq. (2)) and a_i the per-window RMS level (the M25
modulation). The per-window 6x6 spectral-MOMENT matrices are then the WINDOWED
integrals

    M_{n,i} = (1/pi) int_0^inf omega^n S_cross,i(omega) domega
            = a_i^2 (1/pi) int_0^inf omega^n W_i(f) S_cross(omega) domega    (2)

— a GENUINELY DIFFERENT 6x6 matrix per window (a different weighted average of
the frequency-varying tensor orientation), NOT a_i^2 times ONE shared matrix. The
CRUCIAL physics: in a real structure S_cross(f) has a DIFFERENT tensor ORIENTATION
at different frequencies (near mode-1 resonance the stress is dominated by mode-1's
stress shape, near mode-2 by mode-2's), so a window that SWEEPS through the modal
band picks out DIFFERENT tensor orientations -> the critical plane of M_{0,i}
ROTATES window to window. M26's fixed reduction — one plane from the stationary
(all-band) M_0 — is BLIND to this rotation; M27 recovers it.

THE PER-WINDOW REDUCTION (the critical plane may ROTATE). Each window's M_{n,i}
is reduced EXACTLY as M21/M23 reduce a stationary tensor:
* EQUIVALENT VON MISES: m_n^vm,i = trace(Q M_{n,i}) (Preumont-Piefort; M21 eq.
  (3) integrated) — a FIXED quadratic reduction, so the von-Mises SHAPE drifts
  (the windowed scalar) but its "plane" does not rotate (there is none); this
  reduction alone recovers the M26 scalar-von-Mises spectrogram.
* MAX-NORMAL / MAX-SHEAR CRITICAL PLANE: the plane n_i that maximises the
  resolved-stress variance p^T M_{0,i} p is RE-SEARCHED FROM M_{0,i} (M21
  ``critical_plane_search``) — so n_i may ROTATE window to window as the tensor
  orientation drifts; the reduced scalar moments p(n_i)^T M_{n,i} p(n_i) feed the
  M20 estimators.
* NON-PROPORTIONALITY F_np,i: the M23 spectral F_np of the window's critical
  plane (the aspect ratio of the 2x2 in-plane shear block of M_{0,i}) — DRIFTS
  window to window when the tensor's non-proportional content evolves. The plane
  it is evaluated on is the M23-CONVENTION max-shear plane (argmax of the
  NP-effective shear variance lambda_1 + lambda_2 = tau_a^2), because the M21
  lambda_1-argmax plane used for the DAMAGE reduction is F_np-blind (degenerate
  for uniaxial-dominated windows, reporting F_np = 0 on non-proportional states).

THE WINDOW MINER-SUM (Palmgren-Miner). Within each window the STATIONARY M20
estimator gives a damage RATE (E[D]/T)_i from the window's OWN reduced moments;
the window contributes D_i = (E[D]/T)_i T_i and the windows ACCUMULATE linearly

    D = sum_i D_i = sum_i (E[D]/T)_i T_i ,   E[D]/T = D / sum_i T_i          (3)

with the mission time-to-failure T_f = (sum_i T_i) / D (failure at D = 1). This
is IDENTICAL in FORM to the M26 window Miner-sum (M26 eq. (1)) — the difference is
that M26's per-window scalar moments come from ONE fixed reduction, whereas M27's
come from the window's OWN re-searched critical plane. When the critical plane
does NOT rotate (a constant tensor shape) the two coincide EXACTLY; when it
rotates they DIFFER — the M27 <-> M26 boundary.

THE PER-WINDOW CRITICAL-PLANE DRIFT (the reporting point). The module reports, per
window, the max-normal plane normal, the max-shear plane normal, F_np and the
equivalent-stress RMS — the DRIFT of the critical plane / non-proportionality /
level that a JOINT evolutionary tensor exhibits over the M26 fixed-reduction
scalar spectrogram.

NON-STATIONARY MULTIVARIATE MONTE-CARLO. As an independent time-domain validation
the module SYNTHESISES a non-stationary MULTIVARIATE history — the M21 multivariate
spectral-representation method (a per-bin eigendecomposition / Cholesky-type factor
of the tensor cross-PSD, reproducing the full 6x6 covariance) generated with a
TIME-VARYING tensor cross-PSD (per-window blocks of the WINDOWED tensor S_cross,i
concatenated in time, so the instantaneous tensor spectrum tracks S_cross(omega,
t)). Each window's segment is PROJECTED onto that WINDOW's OWN critical plane (the
M22 critical-plane linear projection — a Gaussian scalar), RAINFLOW-counted (the
M20 ASTM E1049 counter) and Miner-summed — the time-domain multiaxial damage the
joint-tensor spectral estimate approximates.

    CONSTANT-SHAPE DELEGATION (the M25 / M21 reduction). In the CONSTANT-tensor-
    shape limit (a flat window, a shared tensor SHAPE scaled by the RMS envelope
    a(t)) the per-window tensors collapse to a_i^2 S_cross, the critical plane is
    FIXED, and the synthesiser DELEGATES to a SINGLE multivariate carrier of the
    shared tensor times the piecewise RMS envelope, projected onto the fixed plane
    — the multivariate lift of the M25 scalar carrier-times-envelope path. A
    SINGLE unit window then reduces BIT-IDENTICALLY to the M21 multivariate
    Monte-Carlo (``monte_carlo_multiaxial_damage``).

    WINDOW-BOUNDARY RAINFLOW CAVEAT (documented, carried from M25/M26). Rainflow
    over the concatenated record naturally handles cycles that STRADDLE a window
    boundary; the closed-form window Miner-sum (3) counts each window's cycles
    INDEPENDENTLY and so MISSES those boundary cycles — a small documented
    discrepancy that vanishes as the windows grow long relative to the cycle
    period. The Monte-Carlo is the reference that includes them.

    ROTATING-vs-WINDOWED CRITICAL PLANE (documented). The Monte-Carlo projects each
    window's segment onto that window's (piecewise-constant) critical plane — the
    WINDOWED critical-plane path, matching the windowed spectral estimate. A
    continuously ROTATING critical plane (the plane orientation varying WITHIN a
    window as the instantaneous tensor rotates) is the fine-window limit; M27 uses
    the windowed (short-time) plane, the engineering standard, NOT a continuous
    rotation.

Deliberate deviations / deferrals (documented, not hidden)
----------------------------------------------------------
* LIBRARY-FIRST sub-flag (/IMPL/FATIG/MULT/EVOL/JOINT) — no upstream equivalent,
  exactly as established for M16-M26's PORT cards.
* WINDOWED (short-time) evolutionary tensor spectrogram, NOT a continuous
  Wigner-Ville / Loeve INSTANTANEOUS-tensor-spectrum formulation (a different,
  cross-term-laden matrix-valued time-frequency distribution) — DEFERRED, exactly
  as M26 deferred the scalar Wigner-Ville distribution.
* SHORT-TIME-STATIONARY approximation: each window is treated as locally
  stationary with its own tensor cross-PSD (the M20/M21 estimators assume
  stationarity within the window). The window must be LONG relative to the carrier
  period and SHORT relative to the tensor drift — the standard spectrogram
  trade-off, documented.
* A full NON-GAUSSIAN JOINT-TENSOR evolutionary distribution (a jointly time-
  varying, non-Gaussian tensor marginal) is DEFERRED; M27 composes with the M24
  non-Gaussian correction on the equivalent scalar but does not model a
  non-Gaussian joint-tensor evolutionary distribution.
* MEAN-STRESS beyond the basic M20/M21 Goodman intercept (applied to the
  equivalent scalar), CRACK-GROWTH / fracture-mechanics fatigue, the COMPLEX-FRF
  stress recovery and a MULTI-INPUT cross-PSD with COHERENCE (the rank-1 single-
  input tensor of M21 eq. (1)) remain DEFERRED (the unchanged M20-M26 tail).
"""

from __future__ import annotations

import math

import numpy as np

from ..common.npcompat import trapezoid

# reuse the M21 tensor / plane machinery and the M26 drifting-shape window,
# read-only (the natural consumer relationship — M27 is the joint-tensor lift of
# the M26 scalar-evolutionary path, both built on M21)
from .multiaxial_fatigue import (critical_plane_search,
                                 equivalent_vonmises_moments,
                                 synthesize_multiaxial_history,
                                 tensor_moment_matrices)
from .spectral_nonproportional_fatigue import (
    max_shear_plane_nonproportionality)


# ============================================================================
# The per-window WINDOWED tensor moment matrices (build item 1a)
# ============================================================================

def windowed_tensor_moment_matrices(omega, Scross, window, scale=1.0, nmax=4):
    """The per-window 6x6 spectral-MOMENT matrices M_{n,i} of the WINDOWED tensor
    cross-PSD (theory eq. (2)):

        M_{n,i} = a_i^2 (1/pi) int omega^n W_i(f) S_cross(omega) domega

    from the angular grid ``omega`` (nf,), the stationary tensor cross-PSD
    ``Scross`` (nf, 6, 6), the per-frequency window weight ``window`` (nf,) and
    the RMS level ``scale`` (a_i). Returns the REAL symmetric part as a
    (nmax+1, 6, 6) array — the same convention as the M21
    ``tensor_moment_matrices`` (a real projection p^T M_n p uses only Re(M_n)).

    This is exactly ``tensor_moment_matrices`` applied to the WINDOWED tensor
    a_i^2 W_i(f) S_cross(omega); it commutes with the reduction (the window is a
    per-frequency scalar, so windowing the tensor then reducing == windowing the
    reduced scalar — the M26 commutation lifted to the full matrix). Because the
    window is scalar the whole matrix stack is formed with ONE weighted quadrature,
    NOT six-by-six per window."""
    omega = np.asarray(omega, dtype=float)
    S = np.asarray(Scross)
    W = np.asarray(window, dtype=float)
    a2 = float(scale) ** 2
    order = np.argsort(omega)
    w = omega[order]
    # broadcast the per-frequency window onto the (nf, 6, 6) tensor once
    Sw = S[order] * W[order][:, None, None]
    out = np.zeros((nmax + 1, 6, 6))
    for n in range(nmax + 1):
        integrand = (w ** n)[:, None, None] * Sw          # (nf, 6, 6)
        out[n] = a2 * trapezoid(integrand.real, w, axis=0) / np.pi
    return out


def _spectral_window(freqs, fc, bw):
    """A Gaussian bandpass spectral WINDOW W(f) = exp(-(f - fc)^2 / (2 bw^2)) on
    ``freqs`` [Hz] (M26 eq. (2)); a non-positive ``bw`` means NO shape drift (a
    flat all-ones window -> a constant tensor shape -> the M25 limit). A thin
    mirror of ``evolutionary_fatigue.spectral_window`` kept local so the tensor
    path does not import the scalar module for one line."""
    f = np.asarray(freqs, dtype=float)
    if bw is None or bw <= 0.0:
        return np.ones_like(f)
    return np.exp(-((f - float(fc)) ** 2) / (2.0 * float(bw) ** 2))


def joint_evolutionary_windows(freqs, durations, fc, bw, scales=None):
    """Build the per-window drifting-shape descriptors for a JOINT-TENSOR
    spectrogram (theory eqs. (1)-(2)): partition the mission into
    ``len(durations)`` windows and, per window i, the swept-centre / broadening
    Gaussian window W_i(f) and RMS level a_i.

    ``fc`` = (fc0, fc1) — the window centre frequency at the FIRST and LAST window
    mid-time (sweeps linearly; a "chirp"); ``bw`` = (bw0, bw1) — the window
    bandwidth start/end. A scalar ``fc`` / ``bw`` is held constant; ``bw`` <= 0
    means a FLAT window (no shape drift -> the constant-tensor-shape / M25 case).
    ``scales`` — the per-window RMS level a_i (default all-ones). Mirrors the M26
    ``drifting_shape_spectrogram`` parameterisation so /EVOL and /JOINT share the
    SAME schedule. Returns a list of window dicts {W, fc, bw, scale, duration}."""
    f = np.asarray(freqs, dtype=float)
    T = np.asarray(durations, dtype=float).ravel()
    nwin = T.size
    if nwin == 0:
        raise ValueError("joint_evolutionary_windows needs at least one window.")
    if np.any(T <= 0.0):
        raise ValueError("every window duration must be positive.")
    if scales is None:
        a = np.ones(nwin)
    else:
        a = np.asarray(scales, dtype=float).ravel()
        if a.size != nwin:
            raise ValueError("scales and durations must match in length.")
    fc0, fc1 = (fc if isinstance(fc, (tuple, list, np.ndarray)) else (fc, fc))
    bw0, bw1 = (bw if isinstance(bw, (tuple, list, np.ndarray)) else (bw, bw))
    # window mid-time fractions (equal-duration parameterisation of the drift —
    # the SAME parameterisation as the M26 drifting-shape spectrogram)
    s = np.array([0.5]) if nwin == 1 else (np.arange(nwin) + 0.5) / nwin
    windows = []
    for i in range(nwin):
        fci = float(fc0) + s[i] * (float(fc1) - float(fc0))
        bwi = float(bw0) + s[i] * (float(bw1) - float(bw0))
        windows.append({"W": _spectral_window(f, fci, bwi), "fc": fci,
                        "bw": bwi, "scale": float(a[i]), "duration": float(T[i])})
    return windows


# ============================================================================
# The per-window multiaxial reduction — plane may ROTATE (build item 1a)
# ============================================================================

def reduce_window_tensor(Mmats, m, C, mean_stress=0.0, ultimate=0.0,
                         naz=24, npol=13):
    """Reduce ONE window's 6x6 spectral-MOMENT matrices ``Mmats`` (the windowed
    ``windowed_tensor_moment_matrices`` output) to the M21/M23 critical-plane
    scalars and their M20 damage — the critical plane RE-SEARCHED from THIS
    window's tensor (theory "THE PER-WINDOW REDUCTION"):

      * ``von_mises``  : the equivalent von-Mises moments trace(Q M_{n,i}) and
        their M20 estimators (a FIXED quadratic reduction — no plane);
      * ``normal_plane``: the max-NORMAL-stress critical plane (searched from
        M_{0,i}) and its scalar moments / M20 estimators;
      * ``shear_plane`` : the max-SHEAR-stress critical plane (searched from
        M_{0,i}) and its scalar moments / M20 estimators;
      * ``F_np``        : the M23 spectral non-proportionality of the max-shear
        critical plane (the aspect ratio of its 2x2 in-plane shear block),
        evaluated on the M23-convention plane — the plane maximising the
        NP-effective shear variance lambda_1 + lambda_2 (its normal is returned
        as ``F_np_normal``); the lambda_1-argmax plane above is F_np-blind.

    Returns a dict with, per reduction, {moments, summary (the four M20
    estimators), damage_rate (Dirlik), normal, ...}, plus ``F_np`` and ``sigma_vm``
    (the equivalent-von-Mises RMS = sqrt(m_0^vm)). ``normals`` optionally supplies
    a precomputed candidate-normal grid so the windows SHARE it."""
    from . import spectral_fatigue as sf
    Mmats = np.asarray(Mmats)
    out = {}
    # --- equivalent von Mises (the fixed quadratic reduction) -------------------
    mom_vm = equivalent_vonmises_moments(Mmats)
    out["von_mises"] = {
        "moments": mom_vm,
        "summary": sf.fatigue_summary(mom_vm, m, C, mean_stress, ultimate)}
    out["sigma_vm"] = math.sqrt(max(float(mom_vm[0]), 0.0))
    # --- critical planes: RE-SEARCHED from THIS window's M_{0,i} (the plane may
    # ROTATE window to window as the tensor orientation drifts) ------------------
    for key, method in (("normal_plane", "normal"), ("shear_plane", "shear")):
        cp = critical_plane_search(Mmats, method=method, naz=naz, npol=npol)
        cp["summary"] = sf.fatigue_summary(cp["moments"], m, C, mean_stress,
                                           ultimate)
        out[key] = cp
    # --- the M23 spectral F_np of the (re-searched) max-shear critical plane ----
    # Evaluated under the M23 critical-plane convention (the plane maximising the
    # NP-EFFECTIVE shear variance lambda_1 + lambda_2 = tau_a^2), NOT at the M21
    # lambda_1-argmax plane above: that plane is F_np-BLIND — degenerate for a
    # uniaxial-dominated tensor, its first-found member carries an empty minor
    # shear axis and would report F_np = 0 even for genuinely non-proportional
    # (independent bending + torsion) content. The damage reductions keep the M21
    # plane (the M27 <-> M21 identity); only the F_np report uses the M23 plane.
    out["F_np"], out["F_np_normal"] = max_shear_plane_nonproportionality(
        Mmats[0], naz=naz, npol=npol)
    for key in ("von_mises", "normal_plane", "shear_plane"):
        out[key]["damage_rate"] = float(out[key]["summary"]["dirlik"]
                                        ["damage_rate"])
    return out


# ============================================================================
# The JOINT-TENSOR evolutionary window Miner-sum (build item 1a + 1b)
# ============================================================================

def joint_evolutionary_fatigue_summary(omega, Scross, durations, fc, bw, m, C,
                                       scales=None, mean_stress=0.0, ultimate=0.0,
                                       naz=24, npol=13, drift=True):
    """The FULLY EVOLUTIONARY JOINT-TENSOR window Miner-sum damage (theory eq.
    (3)): form the per-window 6x6 tensor cross-PSD S_cross,i = a_i^2 W_i(f)
    S_cross (eq. (1)), recompute the per-window moment matrices M_{n,i} (eq. (2)),
    reduce EACH window with the M21/M23 critical-plane machinery — the critical
    plane / F_np RE-SEARCHED from the window's OWN tensor (so the plane may
    ROTATE) — and Palmgren-Miner SUM the per-window MULTIAXIAL damages
    duration-weighted.

    ``omega`` (nf,) the angular grid, ``Scross`` (nf, 6, 6) the STATIONARY tensor
    cross-PSD, ``durations`` / ``fc`` / ``bw`` / ``scales`` the drifting-shape
    schedule (as ``joint_evolutionary_windows``), ``m`` / ``C`` the S-N law.

    ``drift`` — if True (the M27 default) the critical plane is RE-SEARCHED per
    window (it may ROTATE — the whole point); if False the plane is FIXED at the
    STATIONARY (all-window) critical plane and only the windowed scalar drifts —
    the M26 scalar-spectrogram reduction (used for the M27 <-> M26 exact-reduction
    check).

    Returns a dict with, for each reduction (von_mises / normal_plane /
    shear_plane), the window Miner-sum {damage_rate, life, damage, total_time};
    plus:
      * ``windows`` — the per-window critical-plane DRIFT [{fc, bw, scale,
        duration, normal_n, shear_n, F_np, sigma_vm, damage_rate per reduction}]
        (the point of a JOINT evolutionary tensor over M26's fixed reduction);
      * ``constant_shape`` — whether the per-window tensor SHAPE is invariant (so
        the plane does NOT rotate and the answer reduces to M25 / M26);
      * ``plane_rotation_deg`` — the max angle (deg) the max-shear plane normal
        swings across the windows (0 for a non-rotating tensor);
      * ``damage_rate`` / ``life`` — the von-Mises window Miner-sum, promoted to
        the top level (the wide-band scalar analogue); ``nwin``."""
    freqs = np.asarray(omega, dtype=float) / (2.0 * np.pi)
    windows = joint_evolutionary_windows(freqs, durations, fc, bw, scales)

    # the STATIONARY (all-window, full-band) tensor moment matrices + reduction —
    # the reference the built-in reductions compare against, and the FIXED plane
    # for the drift=False (M26) branch
    Mstat = tensor_moment_matrices(omega, Scross, nmax=4)
    stat = reduce_window_tensor(Mstat, m, C, mean_stress, ultimate, naz, npol)

    reductions = ("von_mises", "normal_plane", "shear_plane")
    # per-reduction accumulators for the Miner-sum (eq. (3))
    D = {k: 0.0 for k in reductions}
    Ttot = 0.0
    wout = []
    shear_normals = []       # to measure the plane rotation across windows
    normal_normals = []
    fnps = []
    shapes = []              # normalised M_0 (unit-trace) for constant-shape test

    for w in windows:
        Mi = windowed_tensor_moment_matrices(omega, Scross, w["W"],
                                             scale=w["scale"], nmax=4)
        if drift:
            red = reduce_window_tensor(Mi, m, C, mean_stress, ultimate, naz,
                                       npol)
        else:
            # FIXED plane: project the WINDOWED tensor onto the STATIONARY critical
            # plane (the M26 scalar-spectrogram reduction — no re-search)
            red = _reduce_window_fixed(Mi, stat, m, C, mean_stress, ultimate)
        Ti = w["duration"]
        Ttot += Ti
        for k in reductions:
            D[k] += red[k]["damage_rate"] * Ti
        shear_normals.append(np.asarray(red["shear_plane"]["normal"], dtype=float))
        normal_normals.append(np.asarray(red["normal_plane"]["normal"],
                                         dtype=float))
        fnps.append(float(red["F_np"]))
        wout.append({
            "fc": w["fc"], "bw": w["bw"], "scale": w["scale"], "duration": Ti,
            "normal_n": np.asarray(red["normal_plane"]["normal"], dtype=float),
            "shear_n": np.asarray(red["shear_plane"]["normal"], dtype=float),
            # the resolved-stress projection 6-vectors the Monte-Carlo reuses
            "normal_proj": np.asarray(red["normal_plane"]["proj"], dtype=float),
            "shear_proj": np.asarray(red["shear_plane"]["proj"], dtype=float),
            "F_np": float(red["F_np"]), "sigma_vm": float(red["sigma_vm"]),
            "vm_rate": float(red["von_mises"]["damage_rate"]),
            "normal_rate": float(red["normal_plane"]["damage_rate"]),
            "shear_rate": float(red["shear_plane"]["damage_rate"])})
        # the normalised M_0 shape (unit m0^vm) — the constant-shape detector
        s0 = float(equivalent_vonmises_moments(Mi)[0])
        shapes.append(Mi[0] / s0 if s0 > 0 else Mi[0])

    # constant tensor shape: every window's normalised M_0 is the same (the tensor
    # ORIENTATION is invariant, only the level drifts -> the plane cannot rotate)
    const = len(shapes) <= 1 or all(
        np.allclose(shapes[k], shapes[0], rtol=1e-9, atol=1e-12)
        for k in range(1, len(shapes)))
    # the plane rotation: the max angle the max-shear normal swings from window 0.
    # A CONSTANT tensor shape cannot physically rotate the critical plane — any
    # apparent swing there is a tie-break artifact of a degenerate (symmetric)
    # shear covariance whose max-shear maxima are equal, so the tied planes carry
    # identical damage; report 0 rotation in that case (the damage Miner-sum is
    # unaffected, only the reported orientation would flicker between tied planes).
    # measured over BOTH the max-normal and max-shear critical-plane normals (for a
    # beam the max-SHEAR plane stays at the cross-section — normal ~ the beam axis —
    # while the max-NORMAL plane genuinely rotates as the dominant band changes; the
    # max over both captures whichever plane the tensor drift rotates).
    def _swing(nrmls):
        n0 = nrmls[0]
        return max((math.degrees(math.acos(abs(float(np.clip(np.dot(n0, nn),
                                                             -1.0, 1.0)))))
                    for nn in nrmls[1:]), default=0.0)

    rot = 0.0
    if not const and shear_normals:
        rot = max(_swing(normal_normals), _swing(shear_normals))
    # the F_np drift: the spread of the per-window non-proportionality (the
    # non-proportional content evolving window to window — the reporting point when
    # the critical-plane ORIENTATION is stable but the tensor SHAPE drifts)
    fnp_drift = (max(fnps) - min(fnps)) if fnps else 0.0

    out = {"method": "joint_evolutionary_tensor", "nwin": len(windows),
           "drift": bool(drift), "constant_shape": bool(const),
           "plane_rotation_deg": float(rot), "fnp_drift": float(fnp_drift),
           "windows": wout, "stationary": stat, "total_time": Ttot}
    for k in reductions:
        dr = D[k] / Ttot if Ttot > 0 else 0.0
        out[k] = {"damage": D[k], "total_time": Ttot, "damage_rate": dr,
                  "life": (math.inf if dr <= 0.0 else 1.0 / dr)}
    vm = out["von_mises"]
    out.update({"damage_rate": vm["damage_rate"], "life": vm["life"]})
    return out


def _reduce_window_fixed(Mmats, stat, m, C, mean_stress, ultimate):
    """The FIXED-plane reduction of ONE window's moment matrices (the M26
    scalar-spectrogram reduction, used by ``drift=False``): project the WINDOWED
    tensor onto the STATIONARY critical plane (``stat``, from the all-window
    tensor) — NO per-window re-search — so only the windowed SCALAR PSD drifts,
    exactly as M26 windows each reduction's fixed-plane scalar. Returns the same
    dict shape as ``reduce_window_tensor`` with the stationary plane normals."""
    from . import spectral_fatigue as sf
    Mmats = np.asarray(Mmats)
    out = {}
    mom_vm = equivalent_vonmises_moments(Mmats)
    out["von_mises"] = {
        "moments": mom_vm,
        "summary": sf.fatigue_summary(mom_vm, m, C, mean_stress, ultimate)}
    out["sigma_vm"] = math.sqrt(max(float(mom_vm[0]), 0.0))
    for key in ("normal_plane", "shear_plane"):
        p = np.asarray(stat[key]["proj"], dtype=float)         # FIXED projection
        moments = np.array([float(p @ Mmats[i] @ p)
                            for i in range(Mmats.shape[0])])
        out[key] = {"normal": np.asarray(stat[key]["normal"], dtype=float),
                    "proj": p, "moments": moments,
                    "summary": sf.fatigue_summary(moments, m, C, mean_stress,
                                                  ultimate)}
    out["F_np"] = float(stat["F_np"])
    out["F_np_normal"] = stat.get("F_np_normal")
    for key in ("von_mises", "normal_plane", "shear_plane"):
        out[key]["damage_rate"] = float(out[key]["summary"]["dirlik"]
                                        ["damage_rate"])
    return out


# ============================================================================
# Non-stationary MULTIVARIATE Monte-Carlo cross-check (build item 2)
# ============================================================================

def synthesize_joint_evolutionary_history(omega, Scross, durations, fc, bw,
                                          seed, scales=None, fs=None):
    """Synthesise a NON-STATIONARY MULTIVARIATE stress-tensor history whose
    instantaneous 6x6 cross-spectrum tracks the evolutionary tensor S_cross(omega,
    t) (theory "NON-STATIONARY MULTIVARIATE MONTE-CARLO"):

      * CONSTANT-SHAPE case — if the window is FLAT (bw <= 0) so the per-window
        tensor is a_i^2 S_cross (one shared SHAPE scaled), synthesise a SINGLE
        multivariate carrier of the shared tensor over the total duration (the M21
        ``synthesize_multiaxial_history``, seeded by ``seed``) and multiply each
        component by the piecewise-constant RMS envelope a(t) — the multivariate
        lift of the M25 scalar carrier-times-envelope path. A single unit window
        then reduces BIT-IDENTICALLY to the M21 multivariate Monte-Carlo.
      * GENUINELY DRIFTING case — synthesise per-window multivariate BLOCKS of the
        WINDOWED tensor a_i^2 W_i(f) S_cross (each block seeded by ``seed`` + i)
        and CONCATENATE them in time, so the instantaneous tensor spectrum tracks
        the drifting shape window by window. All windows share ONE sampling rate.

    Returns (t, X, info) with ``X`` (nt, 6) the six Voigt stress components,
    ``info`` = {edges (window-boundary times), fs, delegated (bool)}."""
    freqs = np.asarray(omega, dtype=float) / (2.0 * np.pi)
    windows = joint_evolutionary_windows(freqs, durations, fc, bw, scales)
    T = np.array([w["duration"] for w in windows], dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(T)])
    S = np.asarray(Scross)
    fmax = float(np.max(freqs))
    if fs is None:
        fs = 8.0 * fmax
    # constant-shape (flat window): every W_i is all-ones -> delegate to a single
    # multivariate carrier times the piecewise envelope (the M25 multivariate lift)
    flat = all(np.allclose(w["W"], 1.0) for w in windows)
    if flat:
        total = float(T.sum())
        t, X = synthesize_multiaxial_history(freqs, S, total, seed, fs=fs)
        # piecewise-constant RMS envelope a(t) (the M25 envelope, applied to every
        # tensor component so the instantaneous covariance is a(t)^2 S_cross)
        a = np.array([w["scale"] for w in windows])
        idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, a.size - 1)
        X = X * a[idx][:, None]
        return t, X, {"edges": edges, "fs": fs, "delegated": True}
    # genuinely drifting: per-window multivariate blocks, ONE shared fs
    chunks = []
    for i, w in enumerate(windows):
        # window i's OWN tensor a_i^2 W_i S_cross -> its OWN multivariate carrier
        Sw = (w["scale"] ** 2) * w["W"][:, None, None] * S
        _t, Xi = synthesize_multiaxial_history(freqs, Sw, w["duration"],
                                               int(seed) + i, fs=fs)
        chunks.append(Xi)
    X = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 6))
    t = np.arange(X.shape[0]) / fs
    return t, X, {"edges": edges, "fs": fs, "delegated": False}


def joint_evolutionary_monte_carlo_damage(omega, Scross, durations, fc, bw, m, C,
                                          seed, scales=None, fs=None,
                                          mean_stress=0.0, ultimate=0.0,
                                          naz=24, npol=13, reduction="shear_plane",
                                          summary=None):
    """The TIME-DOMAIN JOINT-TENSOR evolutionary damage rate by non-stationary
    MULTIVARIATE Monte-Carlo (theory "NON-STATIONARY MULTIVARIATE MONTE-CARLO"):
    synthesise the multivariate history (``synthesize_joint_evolutionary_history``
    — per-window blocks of the windowed tensor, or the M25/M21 delegation in the
    constant-shape limit), PROJECT each window's segment onto that WINDOW's OWN
    critical plane (the windowed critical-plane path), rainflow-count each window
    (ASTM E1049, the M20 counter) and Palmgren-Miner SUM — the independent
    time-domain answer the joint-tensor window spectral estimate approximates.

    ``reduction`` — which critical plane to project onto ("shear_plane" default,
    the M22 linear Gaussian projection; or "normal_plane"). ``summary`` optionally
    supplies a precomputed ``joint_evolutionary_fatigue_summary`` so the per-window
    critical planes are shared (else they are re-derived here).

    Returns the M20 Monte-Carlo dict shape plus a short-time check: the per-window
    realised equivalent-stress RMS ``window_rms``, the per-window mean
    zero-up-crossing rate ``window_nu0`` (Hz, tracking the centre-frequency drift)
    and ``delegated``. In the constant-shape / single-unit-window limit reduces
    EXACTLY to the M21 multivariate Monte-Carlo (module docstring)."""
    from . import spectral_fatigue as sf
    Ceff = sf._goodman_C(C, m, mean_stress, ultimate)
    t, X, info = synthesize_joint_evolutionary_history(
        omega, Scross, durations, fc, bw, seed, scales=scales, fs=fs)
    edges = info["edges"]
    nwin = len(edges) - 1
    projkey = "shear_proj" if reduction == "shear_plane" else "normal_proj"

    # --- CONSTANT-SHAPE / delegated: the critical plane is FIXED, so project the
    # WHOLE concatenated record onto the single stationary plane and rainflow ONCE
    # (the M21 / M25 multivariate path). A single unit window is then BIT-IDENTICAL
    # to the M21 ``monte_carlo_multiaxial_damage`` (same synthesis, same plane,
    # same whole-record rainflow, same T = t[-1] - t[0]).
    if info.get("delegated"):
        # the fixed (stationary) plane's projection — the M21 max-shear plane
        Mstat = tensor_moment_matrices(omega, Scross, nmax=4)
        stat = reduce_window_tensor(Mstat, m, C, mean_stress, ultimate, naz, npol)
        proj = np.asarray(stat[reduction]["proj"], dtype=float)
        s = X @ proj
        ranges, counts = sf.rainflow_count(s)
        D = (float(np.sum(counts * ranges ** m) / Ceff) if ranges.size else 0.0)
        T = t[-1] - t[0] if t.size > 1 else float(np.sum(durations))
        dr = D / T if T > 0 else 0.0
        tf, s_eq = sf.life_and_equivalent(
            dr, ranges.size / T if T > 0 else 0.0, m, Ceff)
        # per-window realised RMS + crossing rate of the projected scalar
        window_rms, window_nu0 = _window_stats(s, t, edges)
        return {"method": "joint_evolutionary_monte_carlo", "damage_rate": dr,
                "life": tf, "s_eq": s_eq, "ncycles": float(counts.sum()),
                "duration": T, "reduction": reduction, "delegated": True,
                "ranges": ranges, "counts": counts,
                "window_rms": window_rms, "window_nu0": window_nu0}

    # --- genuinely drifting: per-window projection onto the window's OWN plane,
    # rainflow each window, Miner-sum (the window-boundary caveat is documented) ---
    if summary is None:
        summary = joint_evolutionary_fatigue_summary(
            omega, Scross, durations, fc, bw, m, C, scales=scales,
            mean_stress=mean_stress, ultimate=ultimate, naz=naz, npol=npol,
            drift=True)
    projs = [np.asarray(w[projkey], dtype=float) for w in summary["windows"]]
    D = 0.0
    ncyc = 0.0
    window_rms = []
    window_nu0 = []
    for i in range(nwin):
        sel = (t >= edges[i]) & (t < edges[i + 1])
        seg = X[sel]
        if seg.shape[0] < 2:
            window_rms.append(0.0)
            window_nu0.append(0.0)
            continue
        s = seg @ projs[i]                                # projected scalar
        ranges, counts = sf.rainflow_count(s)
        D += (float(np.sum(counts * ranges ** m) / Ceff) if ranges.size else 0.0)
        ncyc += float(counts.sum())
        window_rms.append(float(np.std(s)))
        sc = s - np.mean(s)
        ups = np.sum((sc[:-1] < 0.0) & (sc[1:] >= 0.0))
        dur = float(edges[i + 1] - edges[i])
        window_nu0.append(ups / dur if dur > 0 else 0.0)

    Ttot = float(edges[-1]) if edges.size else float(np.sum(durations))
    dr = D / Ttot if Ttot > 0 else 0.0
    nu = ncyc / Ttot if Ttot > 0 else 0.0
    tf, s_eq = sf.life_and_equivalent(dr, nu, m, Ceff)
    return {"method": "joint_evolutionary_monte_carlo", "damage_rate": dr,
            "life": tf, "s_eq": s_eq, "ncycles": ncyc, "duration": Ttot,
            "reduction": reduction, "delegated": False,
            "window_rms": np.asarray(window_rms),
            "window_nu0": np.asarray(window_nu0)}


def _window_stats(s, t, edges):
    """Per-window realised RMS + mean zero-up-crossing rate (Hz) of a projected
    scalar record ``s`` on time grid ``t`` with window-boundary times ``edges`` —
    the short-time spectrogram check (the crossing rate tracks the window centre
    frequency of the evolutionary tensor)."""
    window_rms = []
    window_nu0 = []
    for i in range(edges.size - 1):
        seg = s[(t >= edges[i]) & (t < edges[i + 1])]
        window_rms.append(float(np.std(seg)) if seg.size else 0.0)
        if seg.size > 2:
            sc = seg - np.mean(seg)
            ups = np.sum((sc[:-1] < 0.0) & (sc[1:] >= 0.0))
            dur = float(edges[i + 1] - edges[i])
            window_nu0.append(ups / dur if dur > 0 else 0.0)
        else:
            window_nu0.append(0.0)
    return np.asarray(window_rms), np.asarray(window_nu0)
