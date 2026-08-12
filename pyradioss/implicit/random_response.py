"""
Random / spectral (PSD) response — M19: the stationary response of a linear
structure to a random (stochastic) input described by its Power Spectral
Density, through the modal transfer function.

Fortran origin
--------------
There is NO frequency-domain / random-vibration / PSD path anywhere in the
open-source OpenRadioss engine. ``engine/source/input/freimpl.F`` (the /IMPL
reader, read line by line for M16/M17/M18 AND re-read for M19) parses only
/IMPL/DYNA (the DIRECT Newmark/HHT integrator, imp_dyna.F), /IMPL/BUCKL,
/IMPL/DT, /IMPL/NONLIN and /IMPL/ARCL, plus the linear-solver / QSTAT / SPRB /
MONVOL housekeeping — no /PSD, no random-response, no spectral-moment or
RMS driver anywhere in the open tree (the sole ``PSD`` token in freimpl.F is
``IMUMPSD``, a MUMPS-solver flag — not a power spectral density). OpenRadioss
is a time-domain crash/impact code; the stationary random-vibration analysis
is simply not part of the open-source solver — exactly the finding M16 made
for the real eigensolver, M17 for mode superposition and M18 for the complex
modes.

So M19 does exactly what M16/M17/M18 did: it ports random / spectral (PSD)
response as a clean LIBRARY capability and drives it with a minimal PORT engine
card (/IMPL/PSD — the random-response analogue of M16's /IMPL/EIGV). Nothing in
the M10 direct integrator, the M16 REAL eigensolver, the M17 REAL-mode
superposition or the M18 COMPLEX-mode path is touched: the random-response path
is a NEW, parallel path that CONSUMES their frequency-response functions (FRFs)
read-only (asserted, like the M14-M18 parity contracts).

Theory — random vibration & spectral moments
---------------------------------------------
(Newland, "An Introduction to Random Vibrations, Spectral & Wavelet Analysis",
3rd ed., ch. 5-7; Wirsching, Paez & Ortiz, "Random Vibrations: Theory and
Practice", ch. 3-5; Vanmarcke, "Random Fields", ch. 4 for the spectral-moment
/ crossing-rate relations; the Wiener-Khinchin theorem.)

A stationary random EXCITATION is characterised by its Power Spectral Density
(PSD) S_ff(Omega): the distribution of its mean-square over frequency, with the
Wiener-Khinchin relation tying it to the autocorrelation,
R_ff(tau) = (1/2pi) integral S_ff(Omega) e^{i Omega tau} dOmega, so the input
variance is sigma_f^2 = R_ff(0) = (1/2pi) integral S_ff(Omega) dOmega. Passed
through a LINEAR time-invariant system with frequency-response function
H(Omega) (the FRF — response per unit harmonic input at frequency Omega), the
stationary RESPONSE PSD is

    S_uu(Omega) = H(Omega) S_ff(Omega) H(Omega)*                          (1)

— the fundamental input-output relation of linear random vibration (Newland
eq. 6.31; the |H|^2 law for a single scalar input process). For a scalar random
process f(t) with a fixed SPATIAL pattern (the force vector is F * f(t), or a
rigid-base acceleration a(t)), H_j(Omega) is exactly the FRF to the unit-
amplitude pattern at DOF j — the M17 real-mode FRF q_i(Omega)=(phi_i^T F)/
(omega_i^2-Omega^2+2i zeta_i omega_i Omega), u=sum phi_i q_i, for CLASSICAL
damping, or the M18 complex FRF U(Omega)=sum phi_k p_k^F/(iOmega-lambda_k) for
NON-classical damping — so (1) becomes, per DOF,

    S_uu,j(Omega) = |U_j(Omega)|^2 S_ff(Omega) .                          (2)

M19 consumes whichever FRF the analyst selected (real-mode or complex-mode),
which is why this module is FRF-source-agnostic: it takes an FRF dict (``U``,
``omega``/``freqs``) and the input PSD, and produces the response statistics.

SPECTRAL MOMENTS. The n-th spectral moment of the (one-sided-in-angular-
frequency) response PSD is

    m_n = (1/pi) integral_0^inf Omega^n S_uu(Omega) dOmega                (3)

(the factor 1/pi folds the 1/2pi of the Wiener-Khinchin variance with the
factor 2 from S_uu being an EVEN function of Omega for a real system, so the
one-sided integral equals the two-sided (1/2pi) integral). With this convention

    m_0 = sigma_u^2                    (the response VARIANCE)
    RMS = sigma_u = sqrt(m_0)          (the root-mean-square response)          .

The convention is EXACTLY the task's sigma_u = sqrt( integral S_uu dOmega /
2pi ) written for a two-sided PSD; equivalently, if the input /FUNCT is read as
a one-sided PSD in frequency f (Hz, the engineering MIL-STD form, e.g. g^2/Hz),
sigma_u^2 = integral_0^inf G_uu(f) df — the AREA under the response PSD, since
df = dOmega/2pi. Both statements are the same number; this module documents and
uses the angular-frequency form (3), integrating trapezoidally over the swept
grid (the FRF sweep supplies Omega). See ``spectral_moments``.

CROSSING / PEAK RATES (Rice's formula; Newland ch. 7; Vanmarcke). For a
stationary Gaussian process the mean rate of UP-crossings of the mean level is

    nu_0 = (1/2pi) sqrt(m_2/m_0)       [Hz]                               (4)

(the mean zero-crossing frequency — the "apparent frequency" of the response),
and the mean rate of PEAKS (local maxima) is

    nu_p = (1/2pi) sqrt(m_4/m_2)       [Hz]                               (5)

(needs the 4th moment; reported when the grid supports it). Their ratio gives
the irregularity factor alpha = nu_0/nu_p = sqrt(m_2^2/(m_0 m_4)) and the
spectral-bandwidth measure; the m_1-based Vanmarcke bandwidth
q = sqrt(1 - m_1^2/(m_0 m_2)) is reported too (0 = narrow-band, ->1 = wide-
band). The RATE ratios are convention-independent (the 1/pi of (3) cancels).

WHITE-NOISE SDOF closed form (the validation anchor). For an SDOF
m u_ddot + c u_dot + k u = f(t) with f a white-noise force of constant two-
sided PSD S_0, the receptance H(Omega)=1/(k - m Omega^2 + i c Omega) gives, by
the standard Lorentzian integral integral_{-inf}^{inf}|H|^2 dOmega = pi/(c k),

    sigma_u^2 = (1/2pi) integral |H|^2 S_0 dOmega = S_0 / (2 c k)         (6)

= S_0 / (4 zeta omega_n^3 m^2) with c = 2 zeta omega_n m, k = m omega_n^2 — the
classic result (Newland eq. 7.10 form; sigma^2 ~ S_0/(zeta omega^3 m^2)). The
M19 validation asserts (6) from the numerical integral (3).

BASE (support-motion) excitation. For a rigid-base acceleration process
a_g(t) in a support direction d, the modal force is r_i = -Gamma_i a_g with
Gamma_i = phi_i^T M iota_d the M16 participation factor (iota_d the rigid
influence vector) — so the FRF is taken with ``base_excitation=True`` (the M17
participation feed), and (2) with S_ff = the base-acceleration PSD S_aa gives
the response PSD directly. The relative-displacement response variance follows;
this is the shaker-table / seismic random-input path.

QUASI-STATIC limit. If S_ff is concentrated well below the fundamental
frequency, H(Omega) -> H(0) = 1/k (the static compliance), so S_uu ->
S_ff/k^2 and sigma_u -> sigma_f/k — the response RMS reduces to the static
answer (RMS force / stiffness). The M19 validation asserts this reduction.

Deliberate deviations / deferrals (documented, not hidden)
----------------------------------------------------------
* LIBRARY-FIRST card (/IMPL/PSD) — no upstream equivalent, exactly as
  established for M16's /IMPL/EIGV, M17's /IMPL/MODAL, /IMPL/FREQ and M18's
  /IMPL/CEIGV.
* SINGLE scalar input process (a force pattern OR one base-acceleration
  direction), so the response PSD is the |H|^2 S law (2). MULTI-INPUT
  cross-PSD with coherence (a full S_ff MATRIX with off-diagonal cross-spectra)
  is DEFERRED (PORTING_GUIDE M19) — it needs the H S H* matrix triple product
  and a coherence model.
* STATIONARY response only. Non-stationary / evolutionary PSD (a
  time-varying spectrum, e.g. an earthquake's build-up/decay envelope) is
  DEFERRED.
* SPECTRAL MOMENTS and the basic crossing/peak rates are reported here; the
  full peak-factor / extreme-value distribution beyond the mean rate remains
  DEFERRED. FATIGUE damage from the moments (narrow-band Miner / Dirlik /
  rainflow) was deferred out of M19 and is delivered in M20 — see the
  STRESS-PSD recovery section below and ``implicit/spectral_fatigue.py``.

M20 addition — STRESS-PSD recovery
----------------------------------
M20 extends this response path so the transfer function can target a STRESS
(or stress-resultant) component, not just a displacement DOF. Because the
element stress is a LINEAR operator on the nodal displacement for a
small-strain element evaluated on the reference configuration (sigma =
C : B : u), the per-mode STRESS MODES sigma_i = (element stress operator) .
phi_i are recovered by running the SAME force kernels the M8 static solve uses
on each mode shape (read-only — the buffers are restored, the M14-M19 parity
contract). The stress FRF is then the modal combination of the stress modes
with the SAME modal coordinates the displacement FRF uses,

    H_sigma(Omega) = sum_i sigma_i q_i(Omega)                             (7)

(stress commutes with the superposition), and the stress response PSD is the
|H|^2 S law applied to H_sigma,

    S_sigmasigma(Omega) = |H_sigma(Omega)|^2 S_ff(Omega)                  (8)

with its own spectral moments m_0..m_4 — the input the M20 fatigue estimators
(``spectral_fatigue.py``) consume. The recovered STATIC stress (H_sigma at
Omega -> 0 under a static force) equals the M8 implicit-static stress, and the
stress-PSD moments match a direct |H_sigma|^2 S integral — both asserted in
the M20 validation. The stress FRF is built on the REAL-mode FRF (classical
damping — the standard random-vibration-fatigue assumption); the complex-FRF
stress recovery is DEFERRED (PORTING_GUIDE M20).
"""

from __future__ import annotations

import math

import numpy as np

from ..common.npcompat import trapezoid
from . import require_scipy


# ============================================================================
# The response PSD (build-order item 1)
# ============================================================================

def response_psd(frf, input_psd):
    """The stationary response PSD S_uu,j(Omega) = |U_j(Omega)|^2 S_ff(Omega)
    per equation DOF (theory eq. (2)), given an FRF dict and the input PSD
    sampled on the SAME frequency grid.

    Parameters
    ----------
    frf : dict
        an FRF as returned by ``modal_response.modal_frequency_response`` (the
        M17 real-mode FRF) or ``complex_modal.complex_frf`` (the M18 complex
        FRF): must carry ``U`` (nf, ndof) complex, ``omega`` (nf,) angular
        frequencies (rad/s) and ``freqs`` (nf,) frequencies (Hz). This module
        is deliberately FRF-source-agnostic — it works off either transfer
        function (classical or non-classical damping).
    input_psd : (nf,)
        the input PSD S_ff(Omega) evaluated on ``frf['freqs']`` (the driving
        force PSD, or the base-acceleration PSD for a support-motion input).

    Returns a dict with ``omega``, ``freqs``, ``Suu`` (nf, ndof) the response
    PSD, and ``Sff`` (nf,) the input PSD echoed back.
    """
    U = np.asarray(frf["U"])
    Sff = np.asarray(input_psd, dtype=float)
    if Sff.shape[0] != U.shape[0]:
        raise ValueError(
            f"input PSD has {Sff.shape[0]} samples but the FRF sweep has "
            f"{U.shape[0]} — evaluate the PSD on frf['freqs'] first.")
    # |U|^2 S_ff, broadcast over the (nf, ndof) grid (eq. (2))
    Suu = (np.abs(U) ** 2) * Sff[:, None]
    return {"omega": np.asarray(frf["omega"], dtype=float),
            "freqs": np.asarray(frf["freqs"], dtype=float),
            "Suu": Suu, "Sff": Sff}


# ============================================================================
# Spectral moments + RMS + crossing rates (build-order item 2)
# ============================================================================

def spectral_moments(omega, Suu, nmax=4):
    """The spectral moments m_n = (1/pi) integral_0^inf Omega^n S_uu dOmega,
    n = 0..nmax, by the trapezoidal rule over the swept angular-frequency grid
    (theory eq. (3)). ``Suu`` is (nf, ndof) (or (nf,) for a single DOF);
    returns an (nmax+1, ndof) array (or (nmax+1,) for 1-D input).

    m_0 is the response VARIANCE sigma_u^2; m_2 the velocity variance (times
    the same 1/pi convention); the ratios m_2/m_0 and m_4/m_2 feed the
    crossing / peak rates (their common 1/pi cancels, so the rates are
    convention-independent)."""
    omega = np.asarray(omega, dtype=float)
    Suu = np.asarray(Suu, dtype=float)
    one_d = Suu.ndim == 1
    if one_d:
        Suu = Suu[:, None]
    # sort by frequency defensively (a sweep is usually already ascending)
    order = np.argsort(omega)
    w = omega[order]
    S = Suu[order, :]
    out = np.zeros((nmax + 1, S.shape[1]))
    for n in range(nmax + 1):
        integrand = (w ** n)[:, None] * S          # Omega^n S_uu, (nf, ndof)
        out[n] = trapezoid(integrand, w, axis=0) / np.pi
    return out[:, 0] if one_d else out


def rms_response(moments):
    """The RMS response sigma_u = sqrt(m_0) per DOF, from the moment array of
    ``spectral_moments`` (its first row is m_0 = the variance). Accepts the
    full (nmax+1, ndof) array or just the m_0 row."""
    moments = np.asarray(moments, dtype=float)
    m0 = moments[0] if moments.ndim == 2 else moments
    return np.sqrt(np.clip(m0, 0.0, None))


def crossing_rates(moments):
    """Mean up-crossing rate nu_0 = (1/2pi) sqrt(m_2/m_0), peak rate nu_p =
    (1/2pi) sqrt(m_4/m_2) (Hz), the Vanmarcke bandwidth q = sqrt(1 -
    m_1^2/(m_0 m_2)) and the irregularity factor alpha = nu_0/nu_p, per DOF
    (Rice's formula, theory eqs. (4)-(5)). ``moments`` is the (>=5, ndof) array
    of ``spectral_moments`` (needs m_0..m_4 for the peak rate; nu_p / alpha are
    NaN where m_4 is absent). Returns a dict of per-DOF arrays."""
    moments = np.asarray(moments, dtype=float)
    two_d = moments.ndim == 2
    if not two_d:
        moments = moments[:, None]
    m = moments
    with np.errstate(divide="ignore", invalid="ignore"):
        nu0 = np.where(m[0] > 0.0,
                       np.sqrt(np.clip(m[2] / m[0], 0.0, None)) / (2.0 * np.pi),
                       0.0)
        if m.shape[0] > 4:
            nup = np.where(m[2] > 0.0,
                           np.sqrt(np.clip(m[4] / m[2], 0.0, None))
                           / (2.0 * np.pi), np.nan)
            alpha = np.where(nup > 0.0, nu0 / nup, np.nan)
        else:
            nup = np.full(m.shape[1], np.nan)
            alpha = np.full(m.shape[1], np.nan)
        q = np.where((m[0] > 0.0) & (m[2] > 0.0),
                     np.sqrt(np.clip(1.0 - m[1] ** 2 / (m[0] * m[2]), 0.0,
                                     None)), 0.0)
    out = {"nu0": nu0, "nup": nup, "bandwidth": q, "irregularity": alpha}
    if not two_d:
        out = {k: (v[0] if np.ndim(v) else v) for k, v in out.items()}
    return out


# ============================================================================
# Engine-card driver (/IMPL/PSD)
# ============================================================================

def _lookup_psd(model, funct_id):
    """Resolve the input-PSD /FUNCT table by user id (the same
    ``model.functions`` map every load references), with a clear error."""
    if funct_id is None or int(funct_id) <= 0:
        raise ValueError(
            "/IMPL/PSD needs an input-PSD function id (the /FUNCT giving "
            "S_ff(f)); none was supplied on the card.")
    fid = int(funct_id)
    if fid not in model.functions:
        raise ValueError(
            f"/IMPL/PSD input PSD /FUNCT/{fid} is not defined in the deck.")
    return model.functions[fid]


def run_random_response(model, ip, log, result, constr=None, contacts=(),
                        loads=None):
    """/IMPL/PSD (M19): stationary random / spectral (PSD) response.

    Extract the modes (real or complex), build the FRF over the requested band,
    read the input PSD /FUNCT, form the response PSD S_uu = |H|^2 S_ff, and
    report the RMS (sqrt m_0), the spectral moments m_0/m_1/m_2 and the
    zero-crossing / peak rate — storing them on ``result`` (mirroring how
    ``run_freq_response`` stores ``freq_response``). A PORT card: the
    open-source engine has no random-response path (module docstring).

    Card variants (set on ``ip`` by the reader):
    * force-PSD (default): the deck's /CLOAD pattern is the spatial force
      pattern, driven by the input force PSD; FRF from the M17 real modes;
    * ``impl_psd_base``: rigid-base ACCELERATION PSD in direction
      ``impl_psd_dir`` (the M17 participation feed);
    * ``impl_psd_cplx``: use the M18 COMPLEX FRF (non-classical damping) —
      the assembled C = Rayleigh (/IMPL/DYNA/DAMP) + discrete dashpots.
    """
    from ..engine.kinematics import LoadsAndConstraints
    if loads is None:
        loads = LoadsAndConstraints(model, log)
    require_scipy()

    nev = max(1, int(getattr(ip, "impl_psd_nmode", 6)))
    prestress = bool(getattr(ip, "impl_psd_prestress", False))
    base = bool(getattr(ip, "impl_psd_base", False))
    cplx = bool(getattr(ip, "impl_psd_cplx", False))
    base_dir = int(getattr(ip, "impl_psd_dir", 0))
    fmin = float(getattr(ip, "impl_psd_fmin", 0.0))
    fmax = float(getattr(ip, "impl_psd_fmax", 0.0))
    nf = max(2, int(getattr(ip, "impl_psd_nf", 400)))
    zeta_u = float(getattr(ip, "impl_psd_zeta", 0.02))
    funct_id = getattr(ip, "impl_psd_funct", 0)

    log.info("\n     ** RANDOM / SPECTRAL (PSD) RESPONSE **      (/IMPL/PSD)")
    if prestress:
        log.info("        (prestressed modes: K = K_mat + K_geo of the "
                 "committed state)")

    psd_tab = _lookup_psd(model, funct_id)

    # a pure random-response analysis linearizes about the REST state at x0
    # (unless prestressed), exactly like _run_modal / run_freq_response
    x_saved = model.x
    if not prestress:
        model.x = model.x0.copy()
    try:
        frf = _build_frf(model, ip, log, constr, contacts, loads, nev,
                         prestress, cplx, base, base_dir, zeta_u, fmin, fmax,
                         nf)
    finally:
        if not prestress:
            model.x = x_saved

    # sample the input PSD on the sweep grid (the /FUNCT abscissa is frequency
    # in Hz — the same units the FRF sweep uses)
    Sff = psd_tab.eval(frf["freqs"])
    Sff = np.clip(np.asarray(Sff, dtype=float), 0.0, None)   # a PSD is >= 0
    rp = response_psd(frf, Sff)
    mom = spectral_moments(rp["omega"], rp["Suu"], nmax=4)
    rms = rms_response(mom)
    rates = crossing_rates(mom)

    # scatter the per-eq-DOF statistics back to nodes for reporting
    du_rms, dur_rms = frf["dof"].scatter_solution(rms)
    result.random_response = {
        "omega": rp["omega"], "freqs": rp["freqs"], "Suu": rp["Suu"],
        "Sff": rp["Sff"], "moments": mom, "rms": rms, "rms_nodal": du_rms,
        "rms_rot_nodal": dur_rms, "rates": rates,
        "frf": frf, "base": base, "complex": cplx,
    }

    band = f"[{frf['freqs'].min():.5E}, {frf['freqs'].max():.5E}]"
    log.info(f"      FRF SOURCE . . . . . . . . . . . : "
             f"{'COMPLEX (non-classical)' if cplx else 'REAL modal'}")
    log.info(f"      INPUT . . . . . . . . . . . . . : "
             f"{'BASE ACCELERATION PSD (dir %d)' % base_dir if base else 'FORCE PSD'}"
             f"  /FUNCT/{int(funct_id)}")
    log.info(f"      SWEEP BAND (HZ) / POINTS . . . . : {band} / "
             f"{len(frf['freqs'])}")
    # peak RMS DOF (a compact scalar summary; the full field is on the result)
    jmax = int(np.argmax(rms))
    log.info(f"      PEAK RMS RESPONSE (any DOF)  . . : {rms[jmax]:.5E}")
    log.info(f"      SPECTRAL MOMENTS AT PEAK DOF m0/m1/m2:")
    log.info(f"          {mom[0, jmax]:.5E}  {mom[1, jmax]:.5E}  "
             f"{mom[2, jmax]:.5E}")
    log.info(f"      MEAN ZERO-CROSSING RATE (HZ) . . : "
             f"{rates['nu0'][jmax]:.5E}")
    if np.isfinite(rates["nup"][jmax]):
        log.info(f"      MEAN PEAK RATE (HZ)  . . . . . . : "
                 f"{rates['nup'][jmax]:.5E}")

    # M28: the MULTI-INPUT response runs ALONGSIDE the single-input answer above
    # (a NEW parallel path — result.random_response is fully formed and left
    # byte-identical; the multi-input diagnostics attach as a sub-entry)
    if bool(getattr(ip, "impl_psd_multi", False)) and not cplx:
        _run_multi_input_response(model, ip, log, result, frf, loads)


def _run_multi_input_response(model, ip, log, result, frf, loads):
    """/IMPL/PSD/MULTI (M28): MULTI-INPUT / partially-coherent random RESPONSE.
    Assemble the input cross-PSD S_ff, build the per-input displacement FRF
    columns, recover the response cross-PSD diagonal S_uu[j,j] = (H S_ff H^H)[j,j]
    (the per-DOF response PSD), its RMS field and the response coherence
    diagnostics, and store them on ``result.random_response['multi_input']``
    ALONGSIDE the single-input RMS. Never mutates the single-input result.

    The single-input funct (card line 0) stays the reference; the multi-input
    RMS is the correlated-input answer (the SUM of per-input variances for
    incoherent inputs, the coherent combination for coherent inputs)."""
    from . import multi_input_response as mir
    basis = frf.get("_basis")
    zeta = frf.get("_zeta")
    if basis is None:
        log.warning("        /IMPL/PSD/MULTI needs the real-mode FRF basis; "
                    "the complex-FRF multi-input feed is deferred (M28).",
                    "IMPL/PSD/MULTI")
        return
    freqs_hz = np.asarray(frf["freqs"], dtype=float)
    data = _assemble_multi_input(model, ip, log, basis, freqs_hz, zeta, loads)
    Sff = data["Sff"]
    # per-input physical (equation-space) FRF columns U_a(f) (nf, ndof, ninput)
    Ucols = mir.displacement_frf_columns(data["q_cols"], basis.Phi)
    Suu_diag = mir.response_cross_psd_diagonal(Ucols, Sff)     # (nf, ndof) real
    mom = spectral_moments(data["omega"], Suu_diag, nmax=4)
    rms = rms_response(mom)
    du_rms, dur_rms = basis.dof.scatter_solution(rms)
    # response coherence between the few highest-RMS DOFs (a compact diagnostic)
    jmax = int(np.argmax(rms))
    ndof = rms.size
    sel = np.argsort(rms)[-min(4, ndof):][::-1]
    Suu_sel = mir.response_cross_psd(Ucols, Sff, sel)
    coh = mir.response_coherence(Suu_sel)
    result.random_response["multi_input"] = {
        "inputs": data["inputs"], "ninput": data["ninput"],
        "coh_label": data["coh_label"], "projected": data["projected"],
        "min_eig": data["min_eig"], "nclipped": data["nclipped"],
        "Sff": Sff, "Suu_diag": Suu_diag, "moments": mom, "rms": rms,
        "rms_nodal": du_rms, "rms_rot_nodal": dur_rms, "peak_dof": jmax,
        "coherence_dofs": sel, "response_coherence": coh,
        "freqs": freqs_hz, "omega": data["omega"],
    }
    log.info("\n     ** MULTI-INPUT / PARTIALLY-COHERENT RESPONSE **   "
             "(/IMPL/PSD/MULTI)")
    log.info(f"      NUMBER OF INPUTS . . . . . . . . : {data['ninput']}")
    log.info(f"      COHERENCE MODEL  . . . . . . . . : {data['coh_label']}")
    log.info(f"      INPUT CROSS-PSD PSD-PROJECTED  . : "
             f"{'YES' if data['projected'] else 'NO (already valid)'}"
             f"  (min eig {data['min_eig']:.3E})")
    log.info(f"      PEAK MULTI-INPUT RMS (any DOF) . : {rms[jmax]:.5E}")
    log.info(f"      (single-input peak RMS)  . . . . : "
             f"{np.max(result.random_response['rms']):.5E}")


def _build_frf(model, ip, log, constr, contacts, loads, nev, prestress, cplx,
               base, base_dir, zeta_u, fmin, fmax, nf):
    """Build the FRF the PSD response consumes — the M17 real-mode FRF (default)
    or the M18 complex FRF (``cplx``). Returns the FRF dict augmented with a
    ``dof`` handle for the nodal scatter of the RMS field."""
    n = model.numnod
    if cplx:
        # M18 complex FRF (non-classical damping): assemble (K, C, M) with the
        # Rayleigh a,b of /IMPL/DYNA/DAMP folded into C
        from .complex_modal import build_complex_basis, complex_frf
        alpha = beta = 0.0
        if getattr(ip, "impl_dyna_damp", False):
            alpha = float(getattr(ip, "impl_dyna_dampa", 0.0))
            beta = float(getattr(ip, "impl_dyna_dampb", 0.0))
        basis = build_complex_basis(model, nev=nev, log=None,
                                    constraints=constr, contacts=contacts,
                                    prestress=prestress, alpha=alpha, beta=beta)
        if base:
            raise NotImplementedError(
                "/IMPL/PSD/CPLX with base excitation is not combined in M19: "
                "the complex-FRF base-input feed needs the participation "
                "vector projected through the state-space biorthogonality "
                "(PORTING_GUIDE M19 deferral). Use the real-mode FRF for base "
                "PSD, or a force PSD for the complex FRF.")
        if fmax <= fmin:
            fmin, fmax = 0.0, 1.2 * float(basis.natural_freqs_hz.max())
        F = np.zeros((n, 3))
        loads.external_forces(1.0, F, model.x0)      # deck /CLOAD pattern
        freqs_hz = np.linspace(fmin, fmax, nf)
        frf = complex_frf(basis, F, np.zeros((n, 3)), freqs_hz)
        frf["dof"] = basis.dof
        return frf

    # M17 real-mode FRF (classical damping)
    from .modal_response import (build_modal_basis, modal_damping,
                                 modal_frequency_response)
    basis = build_modal_basis(model, nev=nev, log=None, constraints=constr,
                              contacts=contacts, prestress=prestress)
    if fmax <= fmin:
        fmin, fmax = 0.0, 1.2 * float(basis.freqs.max())
    zeta = modal_damping(basis.omega, uniform=(zeta_u or None), log=None)
    freqs_hz = np.linspace(fmin, fmax, nf)
    F = np.zeros((n, 3))
    if not base:
        loads.external_forces(1.0, F, model.x0)      # deck /CLOAD pattern
    frf = modal_frequency_response(basis, F, np.zeros((n, 3)), freqs_hz, zeta,
                                   base_excitation=base, base_dir=base_dir)
    frf["dof"] = basis.dof
    # M28 multi-input reuses the real-mode basis + damping read-only (stashed
    # for /IMPL/PSD/MULTI; harmless for the single-input path)
    frf["_basis"] = basis
    frf["_zeta"] = zeta
    return frf


# ============================================================================
# M20 — STRESS-PSD recovery (the fatigue path's transfer function targets a
# STRESS component; see the module docstring's M20 section)
# ============================================================================

_VOIGT_LABELS = ("xx", "yy", "zz", "xy", "yz", "zx")


def _stress_state_field(group):
    """The scalar stress channel a group's element buffer exposes:
    trusses/solids/shells carry ``state['sig']`` (axial for the truss, the
    6-component Voigt Cauchy stress for solids/shells), springs carry
    ``state['force']`` (the axial FORCE resultant — a spring has no stress
    tensor, so its stress-RESULTANT is the natural fatigue channel). Returns
    ``(field_name, array)`` or ``(None, None)`` for a group with neither."""
    st = group.state
    if "sig" in st:
        return "sig", np.asarray(st["sig"])
    if "force" in st:
        return "force", np.asarray(st["force"])
    return None, None


def recover_element_stresses(model, du, dur):
    """Read-only recovery of every element's scalar stress channel from a nodal
    displacement field ``(du, dur)``, applied through the SAME force kernels
    the M8 static solve uses (``statics._internal_forces``), on the ZERO-stress
    reference at x0. Restores the element buffers afterward (the M14-M19
    read-only parity contract — this NEVER perturbs the element state the modal
    solvers built on). Returns ``{group_name: stress_array}`` (a truss/spring
    array is (n,), a solid/shell array is (n, 6) Voigt).

    The STRESS MODES sigma_i = (element stress operator) . phi_i are exactly
    this recovery applied to each mode shape phi_i: for a small-strain linear
    element evaluated on the zero-stress reference, the kernel integrates
    sigma = C : B : u in one pseudo-velocity step at dt = 1 (the linear stress
    operator — see the statics module docstring), so the recovered stress IS
    the stress mode. Because the base state is zero-stress, the answer is the
    pure linear operator (no committed pre-stress bias)."""
    from .statics import _snapshot, _restore, _internal_forces
    # snapshot the live buffers so we can restore them (read-only contract)
    saved = {name: _snapshot(g) for name, g in model.element_groups()}
    # a ZERO-stress committed base (the linear operator's origin): copy the
    # snapshot but zero the stress / force / plastic / energy accumulators so
    # the kernel returns exactly C : B : u (no pre-stress offset)
    zero = {}
    for name, g in model.element_groups():
        snap = _snapshot(g)
        for key in ("sig", "force", "epsp", "ehour", "eint"):
            v = snap.get(key)
            if isinstance(v, np.ndarray):
                snap[key] = np.zeros_like(v)
        zero[name] = snap
    try:
        _internal_forces(model, model.x0, du, dur, zero, nlgeom=False)
        out = {}
        for name, g in model.element_groups():
            fld, arr = _stress_state_field(g)
            if fld is not None:
                out[name] = arr.copy()
    finally:
        for name, g in model.element_groups():
            _restore(g, saved[name])
    return out


def stress_channels(model):
    """Enumerate the model's scalar stress channels — one per (element,
    stress-component): a truss / spring contributes a single axial channel,
    a solid / shell its 6 Voigt components. Returns a list of tuples
    ``(group_name, elem_local_index, comp_index, label)`` the stress-mode /
    stress-FRF machinery indexes."""
    channels = []
    for name, g in model.element_groups():
        fld, arr = _stress_state_field(g)
        if fld is None:
            continue
        ids = getattr(g, "ids", None)
        ncomp = arr.shape[1] if arr.ndim == 2 else 1
        for e in range(g.n):
            eid = int(ids[e]) if ids is not None else e + 1
            if ncomp == 1:
                tag = "N" if fld == "force" else "sig"
                channels.append((name, e, 0, f"{name}#{eid}:{tag}"))
            else:
                for c in range(ncomp):
                    channels.append(
                        (name, e, c, f"{name}#{eid}:s{_VOIGT_LABELS[c]}"))
    return channels


def stress_modes(model, basis, channels=None):
    """The per-mode STRESS MODES as a ``(nmode, nchan)`` matrix Sigma: recover
    the element stresses of each mode shape phi_i (``recover_element_stresses``)
    and pull out every scalar stress channel (``stress_channels``). Column j is
    the modal history of channel j; row i is the stress field of mode i.
    Read-only in the element state. Returns ``(Sigma, channels)``."""
    if channels is None:
        channels = stress_channels(model)
    nmode = basis.nmode
    Sigma = np.zeros((nmode, len(channels)))
    for i in range(nmode):
        du, dur = basis.dof.scatter_solution(basis.Phi[:, i])
        rec = recover_element_stresses(model, du, dur)
        for j, (name, e, c, _lab) in enumerate(channels):
            arr = rec[name]
            Sigma[i, j] = arr[e, c] if arr.ndim == 2 else arr[e]
    return Sigma, channels


def element_voigt_blocks(channels):
    """Group the scalar stress channels (``stress_channels``) into per-element
    6-component VOIGT blocks — the M21 multiaxial path needs the FULL stress
    TENSOR of an element, not the individual scalar components the M20 path
    treats independently. Returns a list of tuples ``(group_name, elem_local,
    label_base, cols)`` where ``cols`` is the length-6 array of channel-column
    indices (in Voigt order xx,yy,zz,xy,yz,zx) for that element; only elements
    that expose ALL 6 Voigt components (solids / shells) qualify — trusses /
    springs (a single axial channel) have no stress TENSOR and are skipped
    (their scalar fatigue is the M20 path). ``label_base`` is the element tag
    without the component suffix (e.g. ``solids#7``)."""
    # index channels by (group, elem) -> {comp: column}
    per_elem = {}
    for j, (name, e, c, lab) in enumerate(channels):
        per_elem.setdefault((name, e), {})[c] = (j, lab)
    blocks = []
    for (name, e), comps in per_elem.items():
        if len(comps) < 6 or any(c not in comps for c in range(6)):
            continue                              # not a full 6-Voigt element
        cols = np.array([comps[c][0] for c in range(6)], dtype=int)
        # the label base is the tag stripped of its ":s.." component suffix
        lab0 = comps[0][1]
        base = lab0.rsplit(":", 1)[0]
        blocks.append((name, e, base, cols))
    return blocks


def element_voigt_frf(frf, Sigma, cols):
    """The 6-component Voigt stress FRF H_sigma(Omega) (nf, 6) of ONE element,
    pulling the element's 6 channel columns (``cols`` from
    ``element_voigt_blocks``) out of the full stress FRF q @ Sigma. Read-only in
    the modal data. Reuses ``stress_frf`` (the H_sigma = q Sigma product) and
    slices — the same modal coordinates the displacement FRF uses (M20 eq. (7))."""
    Hs = stress_frf(frf, Sigma)["U"]                # (nf, nchan)
    return Hs[:, np.asarray(cols, dtype=int)]       # (nf, 6)


def stress_frf(frf, Sigma):
    """The stress FRF H_sigma(Omega) = sum_i sigma_i q_i(Omega) (theory eq.
    (7)), returned as a dict ``{omega, freqs, U}`` shaped like a displacement
    FRF so ``response_psd`` / ``spectral_moments`` consume it unchanged.

    Stress is a LINEAR operator on displacement, so it commutes with the modal
    superposition: with the physical FRF U = sum_i phi_i q_i and the stress
    modes sigma_i = StressOp . phi_i, H_sigma = StressOp . U = sum_i sigma_i
    q_i = q @ Sigma — one small (nf, nmode) x (nmode, nchan) product, no extra
    kernel calls per frequency. Requires a REAL-mode FRF carrying ``q`` (the
    classical-damping fatigue path; the complex-FRF stress recovery is
    deferred, module docstring)."""
    if "q" not in frf:
        raise NotImplementedError(
            "stress FRF needs the modal coordinates q(Omega) of a REAL-mode "
            "FRF (modal_frequency_response). The complex-FRF stress recovery "
            "is deferred (PORTING_GUIDE M20) — use the real-mode FRF for "
            "spectral fatigue.")
    q = np.asarray(frf["q"])                       # (nf, nmode) complex
    Hs = q @ np.asarray(Sigma)                     # (nf, nchan) complex
    return {"omega": np.asarray(frf["omega"], dtype=float),
            "freqs": np.asarray(frf["freqs"], dtype=float), "U": Hs}


def stress_response_psd(frf, Sigma, input_psd):
    """The stress response PSD S_sigmasigma(Omega) = |H_sigma|^2 S_ff (theory
    eq. (8)) and its spectral moments per channel. Returns a dict with the
    stress FRF, the response PSD ``Ssigma`` (nf, nchan), and the moment array
    ``moments`` (5, nchan) (m_0..m_4). Convenience wrapper over ``stress_frf``
    + ``response_psd`` + ``spectral_moments`` for the fatigue driver."""
    sfrf = stress_frf(frf, Sigma)
    rp = response_psd(sfrf, input_psd)
    mom = spectral_moments(rp["omega"], rp["Suu"], nmax=4)
    return {"omega": rp["omega"], "freqs": rp["freqs"], "Ssigma": rp["Suu"],
            "Sff": rp["Sff"], "moments": mom, "frf": sfrf}


# ============================================================================
# Engine-card driver (/IMPL/FATIG) — M20
# ============================================================================

def run_fatigue(model, ip, log, result, constr=None, contacts=(), loads=None):
    """/IMPL/FATIG (M20): stationary random-vibration (spectral) FATIGUE.

    Build the REAL-mode FRF over the requested band (classical damping), read
    the input PSD /FUNCT, recover the per-mode STRESS modes, form the stress
    response PSD S_sigmasigma = |H_sigma|^2 S_ff and its spectral moments per
    stress channel, pick the CRITICAL channel (highest Dirlik damage), and
    evaluate the narrow-band (Bendat), Dirlik, Wirsching-Light and
    Tovo-Benasciutti damage estimators for an S-N curve N = C S^-m under a
    Miner sum — reporting the damage rate, equivalent stress and life on
    ``result.fatigue`` (mirroring how ``run_random_response`` stores
    ``random_response``). An optional seeded Monte-Carlo rainflow cross-check
    runs when a duration is given. A PORT card: the open-source engine has no
    spectral-fatigue path (module docstring).

    Card controls (set on ``ip`` by the reader):
    * force-PSD (default) or ``impl_fatig_base`` rigid-base ACCELERATION PSD in
      direction ``impl_fatig_dir``;
    * S-N: ``impl_fatig_snm`` (slope m), ``impl_fatig_snc`` (coefficient C);
    * optional ``impl_fatig_mean`` / ``impl_fatig_ult`` (basic Goodman
      mean-stress correction), ``impl_fatig_mcdur`` / ``impl_fatig_seed``
      (Monte-Carlo cross-check duration + seed)."""
    from ..engine.kinematics import LoadsAndConstraints
    from . import spectral_fatigue as sfmod
    from .modal_response import (build_modal_basis, modal_damping,
                                 modal_frequency_response)
    if loads is None:
        loads = LoadsAndConstraints(model, log)
    require_scipy()

    nev = max(1, int(getattr(ip, "impl_fatig_nmode", 6)))
    prestress = bool(getattr(ip, "impl_fatig_prestress", False))
    base = bool(getattr(ip, "impl_fatig_base", False))
    base_dir = int(getattr(ip, "impl_fatig_dir", 0))
    fmin = float(getattr(ip, "impl_fatig_fmin", 0.0))
    fmax = float(getattr(ip, "impl_fatig_fmax", 0.0))
    nf = max(2, int(getattr(ip, "impl_fatig_nf", 800)))
    zeta_u = float(getattr(ip, "impl_fatig_zeta", 0.02))
    funct_id = getattr(ip, "impl_fatig_funct", 0)
    m_sn = float(getattr(ip, "impl_fatig_snm", 0.0))
    C_sn = float(getattr(ip, "impl_fatig_snc", 0.0))
    mean_stress = float(getattr(ip, "impl_fatig_mean", 0.0))
    ultimate = float(getattr(ip, "impl_fatig_ult", 0.0))
    mc_dur = float(getattr(ip, "impl_fatig_mcdur", 0.0))
    mc_seed = int(getattr(ip, "impl_fatig_seed", 1))
    mult = bool(getattr(ip, "impl_fatig_mult", False))

    if mult:
        log.info("\n     ** MULTIAXIAL / CRITICAL-PLANE SPECTRAL FATIGUE **"
                 "  (/IMPL/FATIG/MULT)")
    else:
        log.info("\n     ** RANDOM-VIBRATION (SPECTRAL) FATIGUE **    "
                 "(/IMPL/FATIG)")
    if prestress:
        log.info("        (prestressed modes: K = K_mat + K_geo of the "
                 "committed state)")
    if m_sn <= 0.0 or C_sn <= 0.0:
        raise ValueError(
            "/IMPL/FATIG needs a valid S-N curve N = C*S^-m: a positive slope "
            "m and coefficient C (card line 2: m C). Got "
            f"m={m_sn}, C={C_sn}.")

    psd_tab = _lookup_psd_fatig(model, funct_id)

    # linearize about the REST state at x0 (unless prestressed), exactly like
    # run_random_response / _run_modal / run_freq_response
    x_saved = model.x
    if not prestress:
        model.x = model.x0.copy()
    try:
        basis = build_modal_basis(model, nev=nev, log=None, constraints=constr,
                                  contacts=contacts, prestress=prestress)
        if fmax <= fmin:
            fmin, fmax = 0.0, 1.2 * float(basis.freqs.max())
        zeta = modal_damping(basis.omega, uniform=(zeta_u or None), log=None)
        freqs_hz = np.linspace(fmin, fmax, nf)
        n = model.numnod
        F = np.zeros((n, 3))
        if not base:
            loads.external_forces(1.0, F, model.x0)   # deck /CLOAD pattern
        frf = modal_frequency_response(basis, F, np.zeros((n, 3)), freqs_hz,
                                       zeta, base_excitation=base,
                                       base_dir=base_dir)
        # the STRESS modes are recovered on the same x0 rest state
        Sigma, channels = stress_modes(model, basis)
    finally:
        if not prestress:
            model.x = x_saved

    if not channels:
        raise ValueError(
            "/IMPL/FATIG found no stress channel to evaluate — the model has "
            "no stress-carrying elements (truss / spring / solid / shell).")

    # M21: the MULTIAXIAL / critical-plane path branches here — it keeps the
    # FULL 6-component stress tensor of every element (not the scalar channels),
    # forms the stress-tensor cross-PSD, reduces it to an equivalent-stress PSD
    # and runs the M20 estimators on THAT. Shares the FRF + stress-mode recovery
    # above (read-only). The scalar M20 path continues below unchanged.
    minput = bool(getattr(ip, "impl_fatig_minput", False))
    if mult:
        nplane = int(getattr(ip, "impl_fatig_nplane", 24))
        _run_multiaxial(model, ip, log, result, frf, Sigma, channels,
                        psd_tab, m_sn, C_sn, mean_stress, ultimate, mc_dur,
                        mc_seed, base, base_dir, funct_id, nev, nplane)
        if minput:
            # M28: attach the multi-input answer ALONGSIDE the single-input
            # /MULT result (a NEW parallel path — the single-input result above
            # is fully formed and left byte-identical)
            _run_multi_input_fatigue(
                model, ip, log, result, basis, Sigma, channels,
                frf["freqs"], zeta, loads, m_sn, C_sn, mean_stress, ultimate,
                mc_dur, mc_seed, nplane)
        return

    # stress response PSD + moments per channel (theory eqs. (8))
    Sff = np.clip(np.asarray(psd_tab.eval(frf["freqs"]), dtype=float), 0.0,
                  None)
    sp = stress_response_psd(frf, Sigma, Sff)
    mom = sp["moments"]                                # (5, nchan)

    # per-channel closed-form damage, then pick the CRITICAL channel (highest
    # Dirlik damage rate — the wide-band industry standard)
    nchan = mom.shape[1]
    dirlik_rate = np.zeros(nchan)
    rms_stress = np.sqrt(np.clip(mom[0], 0.0, None))
    for j in range(nchan):
        if rms_stress[j] <= 0.0:
            continue
        dirlik_rate[j] = sfmod.dirlik_damage(
            mom[:, j], m_sn, C_sn, mean_stress, ultimate)["damage_rate"]
    jcrit = int(np.argmax(dirlik_rate)) if nchan else 0
    crit_mom = mom[:, jcrit]

    summary = sfmod.fatigue_summary(crit_mom, m_sn, C_sn, mean_stress,
                                    ultimate)
    mc = None
    if mc_dur > 0.0:
        mc = sfmod.monte_carlo_damage(
            sp["freqs"], sp["Ssigma"][:, jcrit], m_sn, C_sn, mc_dur, mc_seed,
            mean_stress=mean_stress, ultimate=ultimate)

    # M24: the NON-GAUSSIAN / KURTOSIS correction runs ALONGSIDE the M20 Gaussian
    # summary (a NEW parallel path — the Gaussian ``summary`` above is fully
    # formed and left byte-identical). It scales the Gaussian spectral damage of
    # the critical channel by the closed-form lambda_ng (the Winterstein Hermite
    # model) and, if a Monte-Carlo duration is given, runs a non-Gaussian
    # time-domain cross-check (the Gaussian history pushed through the memoryless
    # Hermite transform). See implicit/nongaussian_fatigue.py.
    nongaussian = _run_nongaussian(
        ip, summary, crit_mom, sp["freqs"], sp["Ssigma"][:, jcrit],
        m_sn, C_sn, mean_stress, ultimate, mc_dur, mc_seed)

    # M25: the NON-STATIONARY / EVOLUTIONARY-PSD correction runs ALONGSIDE the M20
    # stationary summary (a NEW parallel path — the stationary ``summary`` above
    # is fully formed and left byte-identical). It evaluates the M20 estimators
    # per stationary block of the RMS mission profile and Miner-sums them (+ the
    # amplitude-modulated E[a^m] closed form + the M25<->M24 kurtosis bridge), and
    # if a Monte-Carlo duration is given runs a non-stationary time-domain
    # cross-check (the Gaussian carrier times a time-varying RMS envelope). See
    # implicit/nonstationary_fatigue.py.
    nonstationary = _run_nonstationary(
        ip, summary, crit_mom, sp["freqs"], sp["Ssigma"][:, jcrit],
        m_sn, C_sn, mean_stress, ultimate, mc_dur, mc_seed, model)

    # M26: the FULLY EVOLUTIONARY / NON-SEPARABLE-PSD correction runs ALONGSIDE the
    # M20 stationary AND the M25 non-stationary numbers (a NEW parallel path — the
    # summaries above are fully formed and left byte-identical). It windows the
    # recovered stress PSD with a swept-centre / broadening Gaussian per time-window
    # (a spectrogram whose SHAPE drifts, not just its RMS level — each window its
    # OWN full moment set) and Miner-sums the window damages, with a non-separable
    # time-varying-filter Monte-Carlo cross-check. See implicit/evolutionary_fatigue.py.
    evolutionary = _run_evolutionary(
        ip, summary, sp["freqs"], sp["Ssigma"][:, jcrit], m_sn, C_sn,
        mean_stress, ultimate, mc_dur, mc_seed, model)

    # M31: the CONTINUOUS WIGNER-VILLE INSTANTANEOUS time-frequency spectrum runs
    # ALONGSIDE the M20 stationary, M25 non-stationary AND M26 windowed-evolutionary
    # numbers (a NEW parallel path — everything above is left byte-identical). It
    # evaluates the drifting shape CONTINUOUSLY at a fine instant grid and Miner-
    # INTEGRATES the per-instant damage (of which the M26 window Miner-SUM is the
    # coarse-grid limit, delegated byte-identically). See implicit/
    # wigner_ville_fatigue.py.
    wigner_ville = _run_wigner_ville(
        ip, evolutionary, sp["freqs"], sp["Ssigma"][:, jcrit], m_sn, C_sn,
        mean_stress, ultimate, mc_dur, mc_seed, model)

    # M32: the TIME-VARYING NON-GAUSSIAN correction of the M31 continuous scalar
    # spectrum (the leptokurtic amplification lambda_ng(t) drifting ALONG the
    # continuous spectrum) runs when BOTH /NGAUSS and /WVILLE are set — the
    # convergence of the M24 stationary kurtosis correction and the M31 continuous
    # instantaneous spectrum, reported as a sub-entry of the wigner_ville entry
    # ALONGSIDE the M31 Gaussian-continuous and M24 stationary-non-Gaussian numbers.
    # See implicit/nongaussian_wigner_ville_fatigue.py.
    if wigner_ville is not None:
        wigner_ville["nongaussian"] = _run_nongaussian_wigner_ville(
            ip, model, wigner_ville, sp["freqs"], sp["Ssigma"][:, jcrit], m_sn,
            C_sn, mean_stress, ultimate, mc_dur, mc_seed)

    result.fatigue = {
        "channels": channels, "critical_channel": jcrit,
        "critical_label": channels[jcrit][3], "moments": mom,
        "rms_stress": rms_stress, "dirlik_rate": dirlik_rate,
        "freqs": sp["freqs"], "omega": sp["omega"], "Ssigma": sp["Ssigma"],
        "Sff": sp["Sff"], "summary": summary, "monte_carlo": mc,
        "nongaussian": nongaussian, "nonstationary": nonstationary,
        "evolutionary": evolutionary, "joint_evolutionary": None,
        "wigner_ville": wigner_ville,
        "sn_m": m_sn, "sn_C": C_sn, "mean_stress": mean_stress,
        "ultimate": ultimate, "base": base, "stress_modes": Sigma,
    }

    _report_fatigue(log, result.fatigue, funct_id, base, base_dir, frf, nev)


def _run_nongaussian(ip, gaussian_summary, moments, freqs, psd, m, C,
                     mean_stress, ultimate, mc_dur, mc_seed):
    """M24: the NON-GAUSSIAN / KURTOSIS correction of a Gaussian estimator
    ``gaussian_summary`` (the M20 ``fatigue_summary`` shape) on one channel's
    ``moments`` [m0..m4]. Returns ``None`` unless /IMPL/FATIG/NGAUSS is set.

    Reads the target kurtosis ``impl_fatig_kurt`` and skewness ``impl_fatig_skew``
    (card line 3) and computes the closed-form correction lambda_ng (Winterstein
    Hermite model + Benasciutti-Braccesi / Rizzi-Kihm) that scales the Gaussian
    spectral damage of every estimator (``nongaussian_summary``); if a Monte-Carlo
    duration is given it also runs the non-Gaussian time-domain cross-check on the
    channel PSD (``freqs`` / ``psd``). The result is stored ALONGSIDE the Gaussian
    numbers so the listing shows both. A PORT sub-flag: the open-source engine has
    no non-Gaussian fatigue path (module docstring)."""
    if not bool(getattr(ip, "impl_fatig_ngauss", False)):
        return None
    from . import nongaussian_fatigue as ngf
    gamma4 = float(getattr(ip, "impl_fatig_kurt", 3.0))
    gamma3 = float(getattr(ip, "impl_fatig_skew", 0.0))
    bwcorr = bool(getattr(ip, "impl_fatig_bwcorr", True))
    ng = ngf.nongaussian_summary(
        moments, m, C, gamma3, gamma4, mean_stress=mean_stress,
        ultimate=ultimate, bandwidth_correction=bwcorr,
        gaussian=gaussian_summary)
    ng_mc = None
    if mc_dur > 0.0 and freqs is not None and psd is not None:
        ng_mc = ngf.nongaussian_monte_carlo_damage(
            freqs, psd, m, C, mc_dur, mc_seed, gamma3, gamma4,
            mean_stress=mean_stress, ultimate=ultimate)
    ng["monte_carlo"] = ng_mc
    return ng


def _run_nongaussian_multiaxial(ip, summ, freqs, m, C, mean_stress, ultimate,
                                mc_dur, mc_seed):
    """M24 (multiaxial): the NON-GAUSSIAN / KURTOSIS correction of the M21
    equivalent-stress reductions (von Mises / max-normal / max-shear critical
    plane). Returns ``None`` unless /IMPL/FATIG/NGAUSS is set.

    Scales each reduction's Gaussian spectral damage by its own lambda_ng (each
    reduction has its own bandwidth alpha_2, so the correction is per-reduction),
    and runs a non-Gaussian Monte-Carlo on the max-shear-plane LINEAR projection
    (the M21 multivariate synthesiser + the Hermite transform). Stored ALONGSIDE
    the Gaussian reductions. Composes with /MULT / /NPROP / /SPEC — the kurtosis
    correction on the equivalent scalar the multiaxial reductions produce."""
    if not bool(getattr(ip, "impl_fatig_ngauss", False)):
        return None
    from . import nongaussian_fatigue as ngf
    gamma4 = float(getattr(ip, "impl_fatig_kurt", 3.0))
    gamma3 = float(getattr(ip, "impl_fatig_skew", 0.0))
    bwcorr = bool(getattr(ip, "impl_fatig_bwcorr", True))
    out = {"gamma4": gamma4, "gamma3": gamma3, "bandwidth_correction": bwcorr}
    for key in ("von_mises", "normal_plane", "shear_plane"):
        red = summ[key]
        out[key] = ngf.nongaussian_summary(
            red["moments"], m, C, gamma3, gamma4, mean_stress=mean_stress,
            ultimate=ultimate, bandwidth_correction=bwcorr,
            gaussian=red["summary"])
    # a non-Gaussian Monte-Carlo cross-check on the max-shear-plane projection
    ng_mc = None
    if mc_dur > 0.0:
        sp = summ["shear_plane"]
        ng_mc = ngf.nongaussian_monte_carlo_projected(
            freqs, summ["Scross"], sp["proj"], m, C, mc_dur, mc_seed, gamma3,
            gamma4, mean_stress=mean_stress, ultimate=ultimate)
    out["monte_carlo"] = ng_mc
    return out


def _build_modulation(model, modfunct, nseg):
    """Turn an RMS scale-vs-time modulation /FUNCT into the (scales, durations)
    block schedule the M25 non-stationary estimators + Monte-Carlo consume.

    ``modfunct`` — the /FUNCT id whose (time, scale) piecewise-linear curve gives
    the RMS scaling a(t) (a mission profile: run-up / dwell / run-down). If
    ``nseg`` > 0 the curve is sampled into ``nseg`` equal-duration blocks (scale =
    the curve at each block midpoint — a smooth amplitude-modulated envelope);
    otherwise the curve's OWN breakpoint intervals become the blocks (scale = the
    midpoint value, duration = the interval). Returns (scales, durations, func)."""
    from . import nonstationary_fatigue as nsf
    if modfunct is None or int(modfunct) <= 0:
        raise ValueError(
            "/IMPL/FATIG/NSTAT needs an RMS scale-vs-time modulation /FUNCT id "
            "(the mission profile) on the card line after the sweep / S-N "
            "(and kurtosis, if NGAUSS) lines: modfunct [nseg].")
    fid = int(modfunct)
    if fid not in model.functions:
        raise ValueError(
            f"/IMPL/FATIG/NSTAT modulation /FUNCT/{fid} is not defined in the "
            "deck.")
    func = model.functions[fid]
    if nseg and int(nseg) > 0:
        scales, durations = nsf.sample_function_modulation(
            func, int(nseg), t0=float(func.x[0]), t1=float(func.x[-1]))
    else:
        # the curve's own breakpoint intervals are the blocks (scale = the
        # midpoint value, duration = the interval length)
        x = np.asarray(func.x, dtype=float)
        mids = 0.5 * (x[:-1] + x[1:])
        scales = np.abs(np.atleast_1d(func.eval(mids)).astype(float))
        durations = np.diff(x)
    return scales, durations, func


def _run_nonstationary(ip, stationary_summary, base_moments, freqs, psd, m, C,
                       mean_stress, ultimate, mc_dur, mc_seed, model):
    """M25: the NON-STATIONARY / EVOLUTIONARY-PSD correction of a stationary
    estimator ``stationary_summary`` (the M20 ``fatigue_summary`` shape) on one
    channel's shared-shape ``base_moments`` [m0..m4]. Returns ``None`` unless
    /IMPL/FATIG/NSTAT is set.

    Reads the RMS scale-vs-time modulation /FUNCT (``impl_fatig_modfunct``, an
    optional block count ``impl_fatig_nstat_nseg``), builds the block schedule,
    and:
      * runs the M20 estimators PER BLOCK and Palmgren-Miner SUMs the block
        damages duration-weighted (``block_fatigue_summary`` — the piecewise-
        stationary "mission profile" model);
      * scales every stationary estimator by E[a^m], the AMPLITUDE-MODULATED /
        evolutionary damage (``amplitude_modulated_summary``), and reports the
        induced kurtosis + the M25<->M24 bridge (kappa_ns vs lambda_ng);
      * if a Monte-Carlo duration is given, runs the non-stationary time-domain
        cross-check on the channel PSD (``freqs`` / ``psd``).
    The result is stored ALONGSIDE the stationary numbers so the listing shows the
    stationary and the non-stationary answers side by side. A PORT sub-flag."""
    if not bool(getattr(ip, "impl_fatig_nstat", False)):
        return None
    from . import nonstationary_fatigue as nsf
    modfunct = getattr(ip, "impl_fatig_modfunct", 0)
    nseg = int(getattr(ip, "impl_fatig_nstat_nseg", 0))
    scales, durations, func = _build_modulation(model, modfunct, nseg)
    scales_w, weights = nsf.modulation_from_schedule(scales, durations)
    # the block "mission profile" Miner sum (Dirlik — the wide-band standard) with
    # the per-block breakdown, and the amplitude-modulated four-estimator summary
    blocks = [{"scale": float(s), "duration": float(d)}
              for s, d in zip(scales, durations)]
    block_summary = nsf.block_fatigue_summary(
        blocks, m, C, mean_stress=mean_stress, ultimate=ultimate,
        estimator="dirlik", base_moments=base_moments)
    am = nsf.amplitude_modulated_summary(
        base_moments, scales_w, weights, m, C, mean_stress=mean_stress,
        ultimate=ultimate, stationary=stationary_summary)
    alpha2 = stationary_summary["params"]["alpha2"]
    bridge = nsf.bridge_to_nongaussian(scales_w, weights, m, alpha2=alpha2,
                                       bandwidth_correction=False)
    ns_mc = None
    if mc_dur > 0.0 and freqs is not None and psd is not None:
        # scale the block durations so the MC record totals mc_dur (the fractions
        # — hence E[a^m] and the induced kurtosis — are preserved)
        tot = float(np.sum(durations))
        mc_durs = durations * (mc_dur / tot) if tot > 0 else durations
        ns_mc = nsf.nonstationary_monte_carlo_damage(
            freqs, psd, m, C, scales, mc_durs, mc_seed, mean_stress=mean_stress,
            ultimate=ultimate)
    return {"modfunct": int(modfunct), "nseg": nseg, "nblocks": len(blocks),
            "scales": scales, "durations": durations, "weights": weights,
            "kurtosis": am["kurtosis"], "kappa_ns": am["kappa_ns"],
            "e_am": am["e_am"], "rms_scale": am["rms_scale"],
            "block": block_summary, "amplitude_modulated": am,
            "bridge": bridge, "monte_carlo": ns_mc}


def _evol_schedule(ip, model, nwin):
    """Build the per-window RMS LEVEL schedule + window DURATIONS for the M26
    evolutionary spectrogram. The levels + mission time span come from the shared
    RMS modulation /FUNCT (``impl_fatig_modfunct`` — the SAME mission profile
    /NSTAT uses, so /EVOL composes with /NSTAT): sample it into ``nwin`` equal
    windows. Without a modulation /FUNCT the levels are unity and the windows are
    unit-duration (the damage RATE is then the physical quantity; the absolute life
    scales with the — arbitrary — total window time). Returns (scales, durations)."""
    from . import nonstationary_fatigue as nsf
    modfunct = int(getattr(ip, "impl_fatig_modfunct", 0) or 0)
    if modfunct > 0 and modfunct in model.functions:
        func = model.functions[modfunct]
        scales, durations = nsf.sample_function_modulation(
            func, nwin, t0=float(func.x[0]), t1=float(func.x[-1]))
    else:
        scales = np.ones(nwin)
        durations = np.ones(nwin)
    return np.asarray(scales, dtype=float), np.asarray(durations, dtype=float)


def _run_evolutionary(ip, stationary_summary, freqs, psd, m, C, mean_stress,
                      ultimate, mc_dur, mc_seed, model):
    """M26: the FULLY EVOLUTIONARY / NON-SEPARABLE-PSD damage of a stationary
    estimator ``stationary_summary`` (the M20 ``fatigue_summary`` shape) on one
    channel's stress PSD (``freqs`` / ``psd``). Returns ``None`` unless
    /IMPL/FATIG/EVOL is set.

    Reads the drifting-shape schedule (``impl_fatig_evol_fc0`` .. ``_bw1`` /
    ``_nwin``), samples the RMS level schedule from the shared modulation /FUNCT
    (composing with /NSTAT), applies the swept-centre / broadening Gaussian window
    per time-window to the recovered stress PSD (each window carries its OWN full
    moment set — a genuinely NON-SEPARABLE spectrogram), runs the M20 estimators
    PER WINDOW and Palmgren-Miner SUMs them (``evolutionary_fatigue_summary`` — the
    general non-separable extension of the M25 block model), and if a Monte-Carlo
    duration is given runs the non-separable time-domain cross-check (per-window
    spectral-representation blocks concatenated — a time-varying filter). Stored
    ALONGSIDE the M20 stationary and M25 non-stationary numbers so the listing
    shows the stationary, RMS-non-stationary and shape-evolutionary answers side by
    side. A PORT sub-flag."""
    if not bool(getattr(ip, "impl_fatig_evol", False)):
        return None
    from . import evolutionary_fatigue as ef
    from . import nonstationary_fatigue as nsf
    fc0 = float(getattr(ip, "impl_fatig_evol_fc0", 0.0))
    fc1 = float(getattr(ip, "impl_fatig_evol_fc1", fc0))
    bw0 = float(getattr(ip, "impl_fatig_evol_bw0", 0.0))
    bw1 = float(getattr(ip, "impl_fatig_evol_bw1", bw0))
    nwin = max(1, int(getattr(ip, "impl_fatig_evol_nwin", 12)))
    scales, durations = _evol_schedule(ip, model, nwin)
    # the NON-SEPARABLE spectrogram: the recovered stress PSD windowed by a
    # swept-centre / broadening Gaussian per window (theory eq. (2))
    windows = ef.drifting_shape_spectrogram(
        freqs, psd, durations, fc=(fc0, fc1), bw=(bw0, bw1), scales=scales)
    summary = ef.evolutionary_fatigue_summary(
        windows, m, C, mean_stress=mean_stress, ultimate=ultimate)
    # the induced kurtosis of the LEVEL modulation (the RMS part; the shape drift
    # itself does not enter the marginal kurtosis of a Gaussian carrier)
    sc_w, wt = nsf.modulation_from_schedule(scales, durations)
    kurt = nsf.rms_modulation_kurtosis(sc_w, wt)
    ns_mc = None
    if mc_dur > 0.0 and freqs is not None and psd is not None:
        tot = float(np.sum(durations))
        mc_durs = durations * (mc_dur / tot) if tot > 0 else durations
        mc_windows = ef.drifting_shape_spectrogram(
            freqs, psd, mc_durs, fc=(fc0, fc1), bw=(bw0, bw1), scales=scales)
        ns_mc = ef.evolutionary_monte_carlo_damage(
            mc_windows, m, C, mc_seed, mean_stress=mean_stress, ultimate=ultimate)
    return {"fc": (fc0, fc1), "bw": (bw0, bw1), "nwin": nwin,
            "modfunct": int(getattr(ip, "impl_fatig_modfunct", 0) or 0),
            "scales": scales, "durations": durations, "kurtosis": kurt,
            "constant_shape": summary["constant_shape"], "summary": summary,
            "damage_rate": summary["damage_rate"], "life": summary["life"],
            "monte_carlo": ns_mc}


def _run_wigner_ville(ip, evolutionary, freqs, psd, m, C, mean_stress, ultimate,
                      mc_dur, mc_seed, model):
    """M31 (scalar): the CONTINUOUS WIGNER-VILLE / LOEVE INSTANTANEOUS time-frequency
    spectrum S_WV(f, t) of one channel's stress PSD (``freqs`` / ``psd``). Returns
    ``None`` unless /IMPL/FATIG/WVILLE is set.

    Where M26 (``_run_evolutionary``) sampled the mission into ``nwin`` short WINDOWS
    and Miner-SUMMED the per-window damages, M31 evaluates the drifting shape
    CONTINUOUSLY at a fine instant grid (``impl_fatig_wv_refine`` instants per window,
    a Cohen-class smoothing width ``impl_fatig_wv_smooth``), reduces the M20
    estimators AT EACH INSTANT and Miner-INTEGRATES over time (an integral, not a
    per-window sum), with the continuous non-separable Monte-Carlo cross-check. The
    windowed spectrogram (refine = 1, smooth = 0) is EXACTLY the long-window limit,
    delegated byte-identically — so the M26 answer (passed as ``evolutionary``) is
    reported ALONGSIDE the continuous one. Stored on
    ``result.fatigue['wigner_ville']``. A PORT sub-flag."""
    if not bool(getattr(ip, "impl_fatig_wville", False)):
        return None
    from . import wigner_ville_fatigue as wv
    fc0 = float(getattr(ip, "impl_fatig_evol_fc0", 0.0))
    fc1 = float(getattr(ip, "impl_fatig_evol_fc1", fc0))
    bw0 = float(getattr(ip, "impl_fatig_evol_bw0", 0.0))
    bw1 = float(getattr(ip, "impl_fatig_evol_bw1", bw0))
    nwin = max(1, int(getattr(ip, "impl_fatig_evol_nwin", 12)))
    refine = max(1, int(getattr(ip, "impl_fatig_wv_refine", 8)))
    smooth = float(getattr(ip, "impl_fatig_wv_smooth", 0.0))
    scales, durations = _evol_schedule(ip, model, nwin)
    summary = wv.wigner_ville_fatigue_summary(
        freqs, psd, durations, (fc0, fc1), (bw0, bw1), m, C, scales=scales,
        refine=refine, smooth=smooth, mean_stress=mean_stress, ultimate=ultimate)
    mc = None
    if mc_dur > 0.0 and freqs is not None and psd is not None:
        tot = float(np.sum(durations))
        mc_durs = durations * (mc_dur / tot) if tot > 0 else durations
        mc = wv.wigner_ville_monte_carlo_damage(
            freqs, psd, mc_durs, (fc0, fc1), (bw0, bw1), m, C, mc_seed,
            scales=scales, refine=refine, smooth=smooth, mean_stress=mean_stress,
            ultimate=ultimate)
    # the M26 WINDOWED reference (the long-window limit) for the side-by-side listing
    windowed = None
    if evolutionary is not None:
        windowed = float(evolutionary["summary"]["dirlik"]["damage_rate"])
    return {"fc": (fc0, fc1), "bw": (bw0, bw1), "nwin": nwin, "refine": refine,
            "smooth": smooth, "nt": summary["nt"],
            "peak_drift": summary["peak_drift"],
            "boundary_jump": summary["boundary_jump"],
            "boundary_jump_windowed": summary["boundary_jump_windowed"],
            "constant_shape": summary["constant_shape"], "summary": summary,
            "damage_rate": summary["damage_rate"], "life": summary["life"],
            "windowed_damage_rate": windowed, "monte_carlo": mc}


def _kurtosis_grid(ip, model, s):
    """M32: build the TIME-VARYING target kurtosis gamma_4(t) / skewness gamma_3(t)
    for the continuous Wigner-Ville spectrum, sampled onto the fine-grid
    mission-fraction axis ``s`` (from ``wigner_ville_fatigue.instantaneous_schedule``).

    A kurtosis-vs-time /FUNCT (``impl_fatig_kfunct``) is evaluated CONTINUOUSLY across
    its own x-range mapped onto the mission fraction (col 3 of the M24 kurtosis line);
    otherwise a linear sweep ``impl_fatig_kurt`` -> ``impl_fatig_kurt1`` (col 4,
    defaulting to the constant kurtosis). Returns (kurt, skew, kurt_grid, label) where
    ``kurt_grid`` is the per-instant gamma_4 array (or None for the scalar/sweep form)
    and ``label`` is the reporting descriptor of the schedule."""
    kfunct = int(getattr(ip, "impl_fatig_kfunct", 0) or 0)
    kurt0 = float(getattr(ip, "impl_fatig_kurt", 3.0))
    k1 = float(getattr(ip, "impl_fatig_kurt1", 0.0))
    kurt1 = k1 if k1 > 0.0 else kurt0
    skew = float(getattr(ip, "impl_fatig_skew", 0.0))
    if kfunct > 0 and kfunct in model.functions:
        fn = model.functions[kfunct]
        x0, x1 = float(fn.x[0]), float(fn.x[-1])
        kg = np.asarray(fn.eval(x0 + np.asarray(s, dtype=float) * (x1 - x0)),
                        dtype=float)
        return kurt0, skew, kg, (f"/FUNCT/{kfunct} g4(t) "
                                 f"{kg.min():.3g}..{kg.max():.3g}")
    if kurt1 != kurt0:
        return (kurt0, kurt1), skew, None, f"SWEEP g4 {kurt0:.4g}->{kurt1:.4g}"
    return kurt0, skew, None, f"CONSTANT g4 {kurt0:.4g}"


def _run_nongaussian_wigner_ville(ip, model, wigner_ville, freqs, psd, m, C,
                                  mean_stress, ultimate, mc_dur, mc_seed):
    """M32 (scalar): the TIME-VARYING NON-GAUSSIAN correction of the CONTINUOUS
    Wigner-Ville instantaneous SCALAR spectrum — the leptokurtic amplification
    lambda_ng(t) drifting ALONG the M31 continuous spectrum, reduced per instant and
    Miner-INTEGRATED. Returns ``None`` unless BOTH /IMPL/FATIG/NGAUSS and
    /IMPL/FATIG/WVILLE are set (i.e. /IMPL/FATIG/NGAUSS/WVILLE).

    Where M24 (``_run_nongaussian``) corrected ONE stationary equivalent scalar for a
    FIXED kurtosis and M31 (``_run_wigner_ville``) gave the Gaussian scalar a
    continuous instantaneous spectrum, M32 re-computes the M24 lambda_ng AT EACH
    INSTANT from that instant's bandwidth alpha_2(t) and a TIME-VARYING target
    gamma_4(t) (a kurtosis-vs-time /FUNCT, or a linear sweep) sampled continuously
    onto the fine grid, and Miner-INTEGRATES the non-Gaussian per-instant damage. The
    M31 Gaussian-continuous and the M24 stationary-non-Gaussian answers are reported
    ALONGSIDE (both left byte-identical). Stored on
    ``result.fatigue['wigner_ville']['nongaussian']``. A PORT sub-flag."""
    if not (bool(getattr(ip, "impl_fatig_ngauss", False))
            and bool(getattr(ip, "impl_fatig_wville", False))):
        return None
    if wigner_ville is None:
        return None
    from . import nongaussian_wigner_ville_fatigue as ngwv
    from . import wigner_ville_fatigue as wv
    fc = wigner_ville["fc"]
    bw = wigner_ville["bw"]
    nwin = wigner_ville["nwin"]
    refine = wigner_ville["refine"]
    smooth = wigner_ville["smooth"]
    scales, durations = _evol_schedule(ip, model, nwin)
    bwcorr = bool(getattr(ip, "impl_fatig_bwcorr", True))
    sch = wv.instantaneous_schedule(durations, fc, bw, scales=scales, refine=refine)
    kurt, skew, kgrid, label = _kurtosis_grid(ip, model, sch["s"])
    summary = ngwv.nongaussian_wigner_ville_summary(
        freqs, psd, durations, fc, bw, m, C, kurt, skew=skew, scales=scales,
        refine=refine, smooth=smooth, kurt_grid=kgrid,
        bandwidth_correction=bwcorr, mean_stress=mean_stress, ultimate=ultimate)
    mc = None
    if mc_dur > 0.0 and freqs is not None and psd is not None:
        tot = float(np.sum(durations))
        mc_durs = durations * (mc_dur / tot) if tot > 0 else durations
        kgrid_mc = kgrid  # same per-instant grid (refine unchanged; durations scaled)
        mc = ngwv.nongaussian_wigner_ville_monte_carlo_damage(
            freqs, psd, mc_durs, fc, bw, m, C, mc_seed, kurt, skew=skew,
            scales=scales, refine=refine, smooth=smooth, kurt_grid=kgrid_mc,
            mean_stress=mean_stress, ultimate=ultimate)
    st = summary.get("stationary_ng")
    return {"schedule": label, "kfunct": int(getattr(ip, "impl_fatig_kfunct", 0)
                                             or 0),
            "gamma4_range": summary["gamma4_range"],
            "lambda_min": summary["lambda_min"],
            "lambda_max": summary["lambda_max"],
            "lambda_mean": summary["lambda_mean"],
            "bandwidth_correction": bwcorr, "summary": summary,
            "damage_rate": summary["damage_rate"], "life": summary["life"],
            "gaussian_damage_rate": float(wigner_ville["damage_rate"]),
            "stationary_ng_damage_rate":
                (float(st["dirlik"]["damage_rate"]) if st is not None else None),
            "monte_carlo": mc, "tensor": False, "multi_input": False}


def _run_evolutionary_multiaxial(ip, summ, freqs, m, C, mean_stress, ultimate,
                                 mc_dur, mc_seed, model):
    """M26 (multiaxial): the FULLY EVOLUTIONARY / NON-SEPARABLE-PSD damage of the
    M21 equivalent-stress reductions (von Mises / max-normal / max-shear critical
    plane). Returns ``None`` unless /IMPL/FATIG/EVOL is set.

    Recovers each reduction's frequency-resolved SCALAR PSD (the von Mises PSD is
    stored; the critical-plane scalar PSDs are the projected cross-PSD
    p^T S_cross(f) p), applies the SAME swept/broadening drifting-shape window per
    window (each reduction its own per-window moments), and Miner-sums. Runs a
    non-separable Monte-Carlo on the von-Mises drifting spectrogram. Stored
    ALONGSIDE the M21 reductions. Composes with /MULT / /NPROP / /SPEC / /NGAUSS /
    /NSTAT — the drifting-shape window on the equivalent scalar the multiaxial
    reductions produce (a jointly evolutionary tensor is deferred)."""
    if not bool(getattr(ip, "impl_fatig_evol", False)):
        return None
    from . import evolutionary_fatigue as ef
    from . import nonstationary_fatigue as nsf
    fc0 = float(getattr(ip, "impl_fatig_evol_fc0", 0.0))
    fc1 = float(getattr(ip, "impl_fatig_evol_fc1", fc0))
    bw0 = float(getattr(ip, "impl_fatig_evol_bw0", 0.0))
    bw1 = float(getattr(ip, "impl_fatig_evol_bw1", bw0))
    nwin = max(1, int(getattr(ip, "impl_fatig_evol_nwin", 12)))
    scales, durations = _evol_schedule(ip, model, nwin)
    Scross = np.asarray(summ["Scross"])            # (nf, 6, 6) tensor cross-PSD
    # each reduction's frequency-resolved SCALAR PSD (von Mises is stored; the
    # critical planes are the projected cross-PSD p^T S_cross p)
    scalar_psd = {"von_mises": np.clip(np.real(np.asarray(summ["von_mises"]["psd"],
                                                          dtype=float)), 0.0, None)}
    for key in ("normal_plane", "shear_plane"):
        proj = np.asarray(summ[key]["proj"], dtype=float)
        sp = np.einsum("i,fij,j->f", proj, Scross, proj)
        scalar_psd[key] = np.clip(np.real(sp), 0.0, None)
    out = {"fc": (fc0, fc1), "bw": (bw0, bw1), "nwin": nwin,
           "modfunct": int(getattr(ip, "impl_fatig_modfunct", 0) or 0),
           "scales": scales, "durations": durations}
    for key in ("von_mises", "normal_plane", "shear_plane"):
        windows = ef.drifting_shape_spectrogram(
            freqs, scalar_psd[key], durations, fc=(fc0, fc1), bw=(bw0, bw1),
            scales=scales)
        out[key] = {"summary": ef.evolutionary_fatigue_summary(
            windows, m, C, mean_stress=mean_stress, ultimate=ultimate)}
        out[key]["damage_rate"] = out[key]["summary"]["damage_rate"]
    sc_w, wt = nsf.modulation_from_schedule(scales, durations)
    out["kurtosis"] = nsf.rms_modulation_kurtosis(sc_w, wt)
    out["constant_shape"] = out["von_mises"]["summary"]["constant_shape"]
    # a non-separable Monte-Carlo on the von-Mises drifting spectrogram
    ns_mc = None
    if mc_dur > 0.0:
        tot = float(np.sum(durations))
        mc_durs = durations * (mc_dur / tot) if tot > 0 else durations
        mc_windows = ef.drifting_shape_spectrogram(
            freqs, scalar_psd["von_mises"], mc_durs, fc=(fc0, fc1),
            bw=(bw0, bw1), scales=scales)
        ns_mc = ef.evolutionary_monte_carlo_damage(
            mc_windows, m, C, mc_seed, mean_stress=mean_stress, ultimate=ultimate)
    out["monte_carlo"] = ns_mc
    return out


def _run_joint_evolutionary(ip, summ, frf, m, C, mean_stress, ultimate,
                            mc_dur, mc_seed, model, naz, npol):
    """M27: the FULLY EVOLUTIONARY MULTIAXIAL (JOINT-TENSOR) damage of the critical
    element. Returns ``None`` unless /IMPL/FATIG/MULT/EVOL/JOINT is set.

    Where M26 (``_run_evolutionary_multiaxial``) windowed the equivalent SCALAR PSD
    of a FIXED reduction, M27 windows the FULL 6x6 stress-TENSOR cross-PSD per
    window and RE-SEARCHES the critical plane / F_np from the window's OWN tensor —
    so the critical plane may ROTATE and F_np may DRIFT window to window. Reads the
    SAME drifting-shape schedule (``impl_fatig_evol_fc0`` .. ``_nwin``) and RMS
    level schedule (the shared modulation /FUNCT, composing with /NSTAT) the M26
    scalar path uses; forms the per-window tensor cross-PSD, reduces each window
    (von Mises / max-normal / max-shear critical plane, plane RE-SEARCHED), and
    Miner-sums; then runs the non-stationary MULTIVARIATE Monte-Carlo cross-check
    (per-window multivariate blocks of the windowed tensor, per-window critical-
    plane projection, rainflow, Miner-sum). Stored ALONGSIDE the M21 stationary,
    M25 non-stationary and M26 scalar-evolutionary numbers so the listing shows all
    four side by side. A PORT sub-flag (a jointly evolutionary tensor — the item
    M25/M26 deferred). See implicit/joint_evolutionary_fatigue.py."""
    if not bool(getattr(ip, "impl_fatig_joint", False)):
        return None
    from . import joint_evolutionary_fatigue as jf
    from . import nonstationary_fatigue as nsf
    fc0 = float(getattr(ip, "impl_fatig_evol_fc0", 0.0))
    fc1 = float(getattr(ip, "impl_fatig_evol_fc1", fc0))
    bw0 = float(getattr(ip, "impl_fatig_evol_bw0", 0.0))
    bw1 = float(getattr(ip, "impl_fatig_evol_bw1", bw0))
    nwin = max(1, int(getattr(ip, "impl_fatig_evol_nwin", 12)))
    scales, durations = _evol_schedule(ip, model, nwin)
    omega = np.asarray(frf["omega"], dtype=float)
    Scross = np.asarray(summ["Scross"])            # (nf, 6, 6) tensor cross-PSD
    # the JOINT-TENSOR window Miner-sum, the critical plane RE-SEARCHED per window
    summary = jf.joint_evolutionary_fatigue_summary(
        omega, Scross, durations, fc=(fc0, fc1), bw=(bw0, bw1), m=m, C=C,
        scales=scales, mean_stress=mean_stress, ultimate=ultimate, naz=naz,
        npol=npol, drift=True)
    sc_w, wt = nsf.modulation_from_schedule(scales, durations)
    kurt = nsf.rms_modulation_kurtosis(sc_w, wt)
    # the non-stationary MULTIVARIATE Monte-Carlo cross-check on the max-shear plane
    ns_mc = None
    if mc_dur > 0.0:
        tot = float(np.sum(durations))
        mc_durs = durations * (mc_dur / tot) if tot > 0 else durations
        # reuse the spectral summary's per-window critical planes (their
        # orientation depends only on the per-window tensor shape — fc / bw /
        # scales — NOT on the durations, so the mc_durs-scaled MC shares them)
        ns_mc = jf.joint_evolutionary_monte_carlo_damage(
            omega, Scross, mc_durs, fc=(fc0, fc1), bw=(bw0, bw1), m=m, C=C,
            seed=mc_seed, scales=scales, mean_stress=mean_stress,
            ultimate=ultimate, naz=naz, npol=npol, reduction="shear_plane",
            summary=summary)
    return {"fc": (fc0, fc1), "bw": (bw0, bw1), "nwin": nwin,
            "modfunct": int(getattr(ip, "impl_fatig_modfunct", 0) or 0),
            "scales": scales, "durations": durations, "kurtosis": kurt,
            "constant_shape": summary["constant_shape"],
            "plane_rotation_deg": summary["plane_rotation_deg"],
            "fnp_drift": summary.get("fnp_drift", 0.0),
            "summary": summary,
            "von_mises": {"damage_rate": summary["von_mises"]["damage_rate"],
                          "life": summary["von_mises"]["life"]},
            "normal_plane": {"damage_rate": summary["normal_plane"]["damage_rate"],
                             "life": summary["normal_plane"]["life"]},
            "shear_plane": {"damage_rate": summary["shear_plane"]["damage_rate"],
                            "life": summary["shear_plane"]["life"]},
            "damage_rate": summary["damage_rate"], "life": summary["life"],
            "monte_carlo": ns_mc}


def _run_wigner_ville_multiaxial(ip, summ, frf, joint_evolutionary, m, C,
                                 mean_stress, ultimate, mc_dur, mc_seed, model,
                                 naz, npol):
    """M31 (multiaxial / JOINT-TENSOR): the CONTINUOUS WIGNER-VILLE INSTANTANEOUS
    6x6 stress-TENSOR spectrum S_sigmasigma,WV(omega, t) of the critical element.
    Returns ``None`` unless /IMPL/FATIG/WVILLE is set.

    Where M27 (``_run_joint_evolutionary``) windowed the full tensor cross-PSD per
    WINDOW and RE-SEARCHED the critical plane / F_np window to window, M31 evaluates
    the tensor spectrum CONTINUOUSLY at a fine instant grid and re-searches the plane
    / F_np AT EACH INSTANT (the plane drifting CONTINUOUSLY, finer than the windows
    resolve), Miner-INTEGRATING the per-instant multiaxial damages, with the
    continuous multivariate Monte-Carlo cross-check. The windowed spectrogram
    (refine = 1, smooth = 0) is EXACTLY the long-window limit, delegated
    byte-identically — so the M27 answer (passed as ``joint_evolutionary``) is
    reported ALONGSIDE. Reuses the /EVOL drifting-shape schedule + the WVILLE
    refine / smooth parameters. Stored on ``result.fatigue['wigner_ville']``. A PORT
    sub-flag."""
    if not bool(getattr(ip, "impl_fatig_wville", False)):
        return None
    from . import wigner_ville_fatigue as wv
    fc0 = float(getattr(ip, "impl_fatig_evol_fc0", 0.0))
    fc1 = float(getattr(ip, "impl_fatig_evol_fc1", fc0))
    bw0 = float(getattr(ip, "impl_fatig_evol_bw0", 0.0))
    bw1 = float(getattr(ip, "impl_fatig_evol_bw1", bw0))
    nwin = max(1, int(getattr(ip, "impl_fatig_evol_nwin", 12)))
    refine = max(1, int(getattr(ip, "impl_fatig_wv_refine", 8)))
    smooth = float(getattr(ip, "impl_fatig_wv_smooth", 0.0))
    scales, durations = _evol_schedule(ip, model, nwin)
    omega = np.asarray(frf["omega"], dtype=float)
    Scross = np.asarray(summ["Scross"])
    summary = wv.wigner_ville_tensor_summary(
        omega, Scross, durations, (fc0, fc1), (bw0, bw1), m, C, scales=scales,
        refine=refine, smooth=smooth, mean_stress=mean_stress, ultimate=ultimate,
        naz=naz, npol=npol, drift=True)
    mc = None
    if mc_dur > 0.0:
        tot = float(np.sum(durations))
        mc_durs = durations * (mc_dur / tot) if tot > 0 else durations
        mc = wv.wigner_ville_tensor_monte_carlo_damage(
            omega, Scross, mc_durs, (fc0, fc1), (bw0, bw1), m, C, seed=mc_seed,
            scales=scales, refine=refine, smooth=smooth, mean_stress=mean_stress,
            ultimate=ultimate, naz=naz, npol=npol, reduction="shear_plane")
    windowed = None
    if joint_evolutionary is not None:
        windowed = {k: float(joint_evolutionary[k]["damage_rate"])
                    for k in ("von_mises", "normal_plane", "shear_plane")}
    return {"fc": (fc0, fc1), "bw": (bw0, bw1), "nwin": nwin, "refine": refine,
            "smooth": smooth, "nt": summary["nt"],
            "multiaxial": True, "tensor": True,
            "peak_drift": summary.get("peak_drift", 0.0),
            "plane_rotation_deg": summary["plane_rotation_deg"],
            "fnp_drift": summary.get("fnp_drift", 0.0),
            "constant_shape": summary["constant_shape"],
            "delegated": summary.get("delegated"), "summary": summary,
            "von_mises": {"damage_rate": summary["von_mises"]["damage_rate"],
                          "life": summary["von_mises"]["life"]},
            "normal_plane": {"damage_rate": summary["normal_plane"]["damage_rate"],
                             "life": summary["normal_plane"]["life"]},
            "shear_plane": {"damage_rate": summary["shear_plane"]["damage_rate"],
                            "life": summary["shear_plane"]["life"]},
            "damage_rate": summary["damage_rate"], "life": summary["life"],
            "windowed": windowed, "monte_carlo": mc}


def _run_nongaussian_wigner_ville_multiaxial(ip, summ, frf, wigner_ville, m, C,
                                             mean_stress, ultimate, mc_dur, mc_seed,
                                             model, naz, npol):
    """M32 (multiaxial / JOINT-TENSOR): the TIME-VARYING NON-GAUSSIAN correction of
    the CONTINUOUS Wigner-Ville instantaneous 6x6 stress-TENSOR spectrum — the
    per-instant re-searched critical-plane / von-Mises equivalent scalar corrected by
    its OWN per-instant lambda_ng(t) (from its per-instant bandwidth and the
    time-varying target gamma_4(t)), Miner-INTEGRATED. Returns ``None`` unless BOTH
    /IMPL/FATIG/NGAUSS and /IMPL/FATIG/WVILLE are set.

    Where M31 (``_run_wigner_ville_multiaxial``) gave the Gaussian joint-tensor a
    continuous instantaneous spectrum with a per-instant plane re-search, M32 scales
    each per-instant reduction by its per-instant lambda_ng (the M24 correction on the
    equivalent scalar, re-computed at each instant). The M31 Gaussian-continuous
    tensor is reported ALONGSIDE (left byte-identical). Stored on
    ``result.fatigue['wigner_ville']['nongaussian']``. A PORT sub-flag (the
    equivalent-scalar Hermite correction on the continuous tensor — a full non-Gaussian
    JOINT-tensor distribution is deferred, module docstring)."""
    if not (bool(getattr(ip, "impl_fatig_ngauss", False))
            and bool(getattr(ip, "impl_fatig_wville", False))):
        return None
    if wigner_ville is None:
        return None
    from . import nongaussian_wigner_ville_fatigue as ngwv
    from . import wigner_ville_fatigue as wv
    fc = wigner_ville["fc"]
    bw = wigner_ville["bw"]
    nwin = wigner_ville["nwin"]
    refine = wigner_ville["refine"]
    smooth = wigner_ville["smooth"]
    scales, durations = _evol_schedule(ip, model, nwin)
    bwcorr = bool(getattr(ip, "impl_fatig_bwcorr", True))
    omega = np.asarray(frf["omega"], dtype=float)
    Scross = np.asarray(summ["Scross"])
    sch = wv.instantaneous_schedule(durations, fc, bw, scales=scales, refine=refine)
    kurt, skew, kgrid, label = _kurtosis_grid(ip, model, sch["s"])
    summary = ngwv.nongaussian_wigner_ville_tensor_summary(
        omega, Scross, durations, fc, bw, m, C, kurt, skew=skew, scales=scales,
        refine=refine, smooth=smooth, kurt_grid=kgrid, bandwidth_correction=bwcorr,
        mean_stress=mean_stress, ultimate=ultimate, naz=naz, npol=npol, drift=True)
    # the non-Gaussian NON-STATIONARY Monte-Carlo on the von-Mises equivalent scalar
    # continuous spectrum (the fatigue driver; a full non-Gaussian tensor MC is
    # deferred — the equivalent-scalar cross-check tracks the induced kurtosis)
    mc = None
    if mc_dur > 0.0:
        tot = float(np.sum(durations))
        mc_durs = durations * (mc_dur / tot) if tot > 0 else durations
        Svm = wv._vonmises_scalar_psd(Scross)
        freqs_hz = omega / (2.0 * np.pi)
        mc = ngwv.nongaussian_wigner_ville_monte_carlo_damage(
            freqs_hz, Svm, mc_durs, fc, bw, m, C, mc_seed, kurt, skew=skew,
            scales=scales, refine=refine, smooth=smooth, kurt_grid=kgrid,
            mean_stress=mean_stress, ultimate=ultimate)
    gwin = {k: float(wigner_ville[k]["damage_rate"])
            for k in ("von_mises", "normal_plane", "shear_plane")}
    return {"schedule": label, "kfunct": int(getattr(ip, "impl_fatig_kfunct", 0)
                                             or 0),
            "gamma4_range": summary["gamma4_range"],
            "lambda_min": summary["lambda_min"], "lambda_max": summary["lambda_max"],
            "lambda_mean": summary["lambda_mean"],
            "plane_rotation_deg": summary["plane_rotation_deg"],
            "fnp_drift": summary.get("fnp_drift", 0.0),
            "bandwidth_correction": bwcorr, "summary": summary, "tensor": True,
            "multi_input": False,
            "von_mises": {"damage_rate": summary["von_mises"]["damage_rate"],
                          "life": summary["von_mises"]["life"]},
            "normal_plane": {"damage_rate": summary["normal_plane"]["damage_rate"],
                             "life": summary["normal_plane"]["life"]},
            "shear_plane": {"damage_rate": summary["shear_plane"]["damage_rate"],
                            "life": summary["shear_plane"]["life"]},
            "damage_rate": summary["damage_rate"], "life": summary["life"],
            "gaussian": gwin, "monte_carlo": mc}


def _run_joint_nongaussian_multiaxial(ip, summ, frf, wigner_ville, m, C,
                                      mean_stress, ultimate, mc_dur, mc_seed,
                                      model, naz, npol):
    """M33 (JOINT-TENSOR NON-GAUSSIAN): a VECTOR (component-wise) Winterstein-Hermite /
    translation-process transform of the CORRELATED 6x6 stress tensor imposing a
    PER-COMPONENT target kurtosis (the Voigt components xx yy zz xy yz zx) JOINTLY on the
    tensor while PRESERVING the marginal variances / covariance, so the resolved critical
    plane INHERITS the INDUCED kurtosis of the joint tensor statistics (NOT the M24/M32
    kurtosis imposed on the already-resolved scalar), applied along the M31/M32
    CONTINUOUS Wigner-Ville instantaneous spectrum, reduced per instant and
    Miner-INTEGRATED, cross-validated by a MULTIVARIATE non-Gaussian non-stationary
    Monte-Carlo. Returns ``None`` unless /IMPL/FATIG/JOINT + /NGAUSS + /WVILLE are all
    set AND a per-component kurtosis line (cols 4..9 of the M24 kurtosis line) is given.

    Where M32 (``_run_nongaussian_wigner_ville_multiaxial``) imposed the target kurtosis
    on the RESOLVED equivalent scalar, M33 imposes it on the TENSOR components and lets
    the reduction inherit the induced non-Gaussianity — the FIRST item M32 deferred. The
    M32 equivalent-scalar answer (kurtosis on the scalar) is reported ALONGSIDE as the
    ``scalar_equivalent`` reference; the M31 Gaussian-continuous tensor is reported
    alongside too (both left byte-identical). Stored on
    ``result.fatigue['wigner_ville']['joint_nongaussian']``. A PORT sub-flag (freimpl.F
    has no non-Gaussian / joint-tensor / Hermite path). See implicit/
    joint_nongaussian_fatigue.py."""
    joint_kurt = tuple(getattr(ip, "impl_fatig_joint_kurt", ()) or ())
    if not (bool(getattr(ip, "impl_fatig_joint", False))
            and bool(getattr(ip, "impl_fatig_ngauss", False))
            and bool(getattr(ip, "impl_fatig_wville", False))
            and len(joint_kurt) > 0):
        return None
    if wigner_ville is None:
        return None
    from . import joint_nongaussian_fatigue as jng
    fc = wigner_ville["fc"]
    bw = wigner_ville["bw"]
    nwin = wigner_ville["nwin"]
    refine = wigner_ville["refine"]
    smooth = wigner_ville["smooth"]
    scales, durations = _evol_schedule(ip, model, nwin)
    bwcorr = bool(getattr(ip, "impl_fatig_bwcorr", True))
    omega = np.asarray(frf["omega"], dtype=float)
    Scross = np.asarray(summ["Scross"])
    kurt = np.asarray(joint_kurt, dtype=float)
    skew = float(getattr(ip, "impl_fatig_skew", 0.0))
    summary = jng.joint_nongaussian_tensor_summary(
        omega, Scross, durations, fc, bw, m, C, kurt, skew=skew, scales=scales,
        refine=refine, smooth=smooth, bandwidth_correction=bwcorr,
        mean_stress=mean_stress, ultimate=ultimate, naz=naz, npol=npol, drift=True,
        scalar_equivalent=True)
    # the MULTIVARIATE non-Gaussian non-stationary Monte-Carlo (the M21 synthesiser
    # pushed through the VECTOR Hermite transform per instant, projected onto the
    # per-instant critical plane, rainflow over the WHOLE record)
    mc = None
    if mc_dur > 0.0:
        tot = float(np.sum(durations))
        mc_durs = durations * (mc_dur / tot) if tot > 0 else durations
        mc = jng.joint_nongaussian_monte_carlo_damage(
            omega, Scross, mc_durs, fc, bw, m, C, mc_seed, kurt, skew=skew,
            scales=scales, refine=refine, smooth=smooth, mean_stress=mean_stress,
            ultimate=ultimate, naz=naz, npol=npol, reduction="shear_plane",
            summary=summary)
    gwin = {k: float(wigner_ville[k]["damage_rate"])
            for k in ("von_mises", "normal_plane", "shear_plane")}
    se = summary.get("scalar_equivalent")
    se_rates = ({k: float(se[k]["damage_rate"])
                 for k in ("von_mises", "normal_plane", "shear_plane")}
                if se is not None else None)
    # M34: the EXACT translation-process CORRELATION-DISTORTION INVERSION. When /EXACT
    # (or /NORTA) composes with the M33 /JOINT + /NGAUSS + /WVILLE path, additionally
    # build the COVARIANCE-EXACT joint tensor (the underlying-Gaussian correlation solved
    # by the Grigoriu / NORTA matching so the transformed covariance reproduces the target
    # EXACTLY, the preservation error driven to ~0) and the corrected multivariate MC, and
    # attach them as an ``exact_covariance`` sub-entry ALONGSIDE the M33 leading-order
    # numbers (which stay byte-identical — the EXACT path is a NEW path). See implicit/
    # joint_nongaussian_fatigue.py.
    exact_entry = None
    if bool(getattr(ip, "impl_fatig_exact", False)):
        ex_summary = jng.joint_nongaussian_tensor_summary(
            omega, Scross, durations, fc, bw, m, C, kurt, skew=skew, scales=scales,
            refine=refine, smooth=smooth, bandwidth_correction=bwcorr,
            mean_stress=mean_stress, ultimate=ultimate, naz=naz, npol=npol, drift=True,
            scalar_equivalent=False, exact=True)
        ex_mc = None
        if mc_dur > 0.0:
            tot = float(np.sum(durations))
            mc_durs = durations * (mc_dur / tot) if tot > 0 else durations
            ex_mc = jng.joint_nongaussian_monte_carlo_damage(
                omega, Scross, mc_durs, fc, bw, m, C, mc_seed, kurt, skew=skew,
                scales=scales, refine=refine, smooth=smooth, mean_stress=mean_stress,
                ultimate=ultimate, naz=naz, npol=npol, reduction="shear_plane",
                summary=ex_summary, exact=True)
        exact_entry = {
            "preservation_error": ex_summary["preservation_error"],
            "preservation_error_leading": ex_summary["preservation_error_leading"],
            "induced_kurt_min": ex_summary["induced_kurt_min"],
            "induced_kurt_max": ex_summary["induced_kurt_max"],
            "induced_kurt_mean": ex_summary["induced_kurt_mean"],
            "lambda_min": ex_summary["lambda_min"],
            "lambda_max": ex_summary["lambda_max"],
            "lambda_mean": ex_summary["lambda_mean"],
            "von_mises": {"damage_rate": ex_summary["von_mises"]["damage_rate"],
                          "life": ex_summary["von_mises"]["life"],
                          "induced_kurt": ex_summary["von_mises"]["induced_kurt"]},
            "normal_plane": {"damage_rate": ex_summary["normal_plane"]["damage_rate"],
                             "life": ex_summary["normal_plane"]["life"],
                             "induced_kurt": ex_summary["normal_plane"]["induced_kurt"]},
            "shear_plane": {"damage_rate": ex_summary["shear_plane"]["damage_rate"],
                            "life": ex_summary["shear_plane"]["life"],
                            "induced_kurt": ex_summary["shear_plane"]["induced_kurt"]},
            "damage_rate": ex_summary["damage_rate"], "life": ex_summary["life"],
            "summary": ex_summary, "monte_carlo": ex_mc}
    return {"joint_kurt": tuple(float(x) for x in kurt), "skew": skew,
            "induced_kurt_min": summary["induced_kurt_min"],
            "induced_kurt_max": summary["induced_kurt_max"],
            "induced_kurt_mean": summary["induced_kurt_mean"],
            "lambda_min": summary["lambda_min"], "lambda_max": summary["lambda_max"],
            "lambda_mean": summary["lambda_mean"],
            "plane_rotation_deg": summary["plane_rotation_deg"],
            "preservation_error": summary["preservation_error"],
            "bandwidth_correction": bwcorr, "summary": summary, "tensor": True,
            "von_mises": {"damage_rate": summary["von_mises"]["damage_rate"],
                          "life": summary["von_mises"]["life"],
                          "induced_kurt": summary["von_mises"]["induced_kurt"]},
            "normal_plane": {"damage_rate": summary["normal_plane"]["damage_rate"],
                             "life": summary["normal_plane"]["life"],
                             "induced_kurt": summary["normal_plane"]["induced_kurt"]},
            "shear_plane": {"damage_rate": summary["shear_plane"]["damage_rate"],
                            "life": summary["shear_plane"]["life"],
                            "induced_kurt": summary["shear_plane"]["induced_kurt"]},
            "damage_rate": summary["damage_rate"], "life": summary["life"],
            "gaussian": gwin, "scalar_equivalent": se_rates, "monte_carlo": mc,
            "exact_covariance": exact_entry}


def _run_wigner_ville_multi_input(ip, model, Hcols, G, positions, omega, freqs_hz,
                                  m, C, mean_stress, ultimate, mc_dur, mc_seed,
                                  naz, npol, m29_ev, m30_ev):
    """M31 (multi-input): the CONTINUOUS WIGNER-VILLE INSTANTANEOUS multi-input
    stress-tensor spectrum of the critical element — the input coherence matrix
    S_ff(omega, t) drifting CONTINUOUSLY. Returns ``None`` unless /IMPL/FATIG/WVILLE
    AND /IMPL/FATIG/MINPUT AND /IMPL/FATIG/EVOL are all set.

    Where M29 (``_run_evolutionary_multi_input``) / M30 (``_run_freq_evolutionary_
    multi_input``) windowed the coherence-driven tensor per WINDOW, M31 evaluates it
    CONTINUOUSLY at a fine instant grid with a per-instant plane re-search, Miner-
    INTEGRATING, with the continuous multi-input Monte-Carlo. Uses the M30
    frequency-dependent stack path when /FCOH is set (``freq_dependent``), else the
    M29 scalar-coherence path. The windowed answer (refine = 1, smooth = 0) is
    delegated byte-identically and reported ALONGSIDE. Stored on
    ``result.fatigue['multi_input']['wigner_ville']``. A PORT sub-flag."""
    if not (bool(getattr(ip, "impl_fatig_wville", False))
            and bool(getattr(ip, "impl_fatig_minput", False))
            and bool(getattr(ip, "impl_fatig_evol", False))):
        return None
    from . import wigner_ville_fatigue as wv
    from . import freq_evolutionary_multi_input as fem
    fc0 = float(getattr(ip, "impl_fatig_evol_fc0", 0.0))
    fc1 = float(getattr(ip, "impl_fatig_evol_fc1", fc0))
    bw0 = float(getattr(ip, "impl_fatig_evol_bw0", 0.0))
    bw1 = float(getattr(ip, "impl_fatig_evol_bw1", bw0))
    nwin = max(1, int(getattr(ip, "impl_fatig_evol_nwin", 12)))
    refine = max(1, int(getattr(ip, "impl_fatig_wv_refine", 8)))
    smooth = float(getattr(ip, "impl_fatig_wv_smooth", 0.0))
    scales, durations = _evol_schedule(ip, model, nwin)
    ninput = int(np.asarray(Hcols).shape[2])
    phase0 = math.radians(float(getattr(ip, "impl_mi_phase", 0.0)))
    phase1 = math.radians(float(getattr(ip, "impl_mi_phase1",
                                        getattr(ip, "impl_mi_phase", 0.0))))
    fcoh = bool(getattr(ip, "impl_mi_fcoh", False))
    coh_label = ""
    if fcoh:
        # the M30 FREQUENCY-DEPENDENT coherence stacks (same schedules as
        # _run_freq_evolutionary_multi_input) drifting CONTINUOUSLY
        cohmodel = int(getattr(ip, "impl_mi_cohmodel", 0))
        gfunct0 = int(getattr(ip, "impl_mi_gfunct0", 0) or 0)
        gfunct1 = int(getattr(ip, "impl_mi_gfunct1", gfunct0) or gfunct0)
        if cohmodel == 1:
            decay0 = float(getattr(ip, "impl_mi_decay", 0.0))
            d1 = float(getattr(ip, "impl_mi_decay1", -1.0))
            decay1 = decay0 if d1 < 0.0 else d1
            speed0 = float(getattr(ip, "impl_mi_speed", 1.0)) or 1.0
            s1 = float(getattr(ip, "impl_mi_speed1", -1.0))
            speed1 = speed0 if s1 < 0.0 else (s1 or 1.0)
            g0 = fem.exponential_coherence_stack(freqs_hz, positions, decay0, speed0)
            g1 = fem.exponential_coherence_stack(freqs_hz, positions, decay1, speed1)
            coh_label = (f"EXPONENTIAL decay {decay0:.4g}->{decay1:.4g}, "
                         f"speed {speed0:.4g}->{speed1:.4g}")
        elif gfunct0 > 0 and gfunct0 in model.functions \
                and gfunct1 in model.functions:
            shape0 = np.asarray(model.functions[gfunct0].eval(freqs_hz), dtype=float)
            shape1 = np.asarray(model.functions[gfunct1].eval(freqs_hz), dtype=float)
            g0 = fem.measured_coherence_stack(freqs_hz, shape0, ninput)
            g1 = fem.measured_coherence_stack(freqs_hz, shape1, ninput)
            coh_label = f"MEASURED gamma(f) /FUNCT {gfunct0}->{gfunct1}"
        else:
            fcoh = False
    if not fcoh:
        # the M29 SCALAR coherence schedule (frequency-flat, drifting CONTINUOUSLY)
        gamma0 = float(getattr(ip, "impl_mi_gamma", 0.0))
        gg1 = float(getattr(ip, "impl_mi_gamma1", -1.0))
        g0 = gamma0
        g1 = gamma0 if gg1 < 0.0 else gg1
        coh_label = f"SCALAR gamma {g0:.4g}->{g1:.4g}"
    summary = wv.wigner_ville_multi_input_summary(
        omega, Hcols, G, durations, m, C, gamma0=g0, gamma1=g1, phase0=phase0,
        phase1=phase1, fc=(fc0, fc1), bw=(bw0, bw1), scales=scales, refine=refine,
        smooth=smooth, mean_stress=mean_stress, ultimate=ultimate, naz=naz,
        npol=npol, drift=True, freq_dependent=fcoh)
    mc = None
    if mc_dur > 0.0:
        tot = float(np.sum(durations))
        mc_durs = durations * (mc_dur / tot) if tot > 0 else durations
        mc = wv.wigner_ville_multi_input_monte_carlo_damage(
            omega, Hcols, G, mc_durs, m, C, seed=mc_seed, gamma0=g0, gamma1=g1,
            phase0=phase0, phase1=phase1, fc=(fc0, fc1), bw=(bw0, bw1),
            scales=scales, refine=refine, smooth=smooth, mean_stress=mean_stress,
            ultimate=ultimate, naz=naz, npol=npol, reduction="shear_plane",
            summary=None, measure=True, freq_dependent=fcoh)
    windowed = None
    ref = m30_ev if fcoh else m29_ev
    if ref is not None:
        windowed = {k: float(ref[k]["damage_rate"])
                    for k in ("von_mises", "normal_plane", "shear_plane")}
    return {"fc": (fc0, fc1), "bw": (bw0, bw1), "nwin": nwin, "refine": refine,
            "smooth": smooth, "nt": summary["nt"], "multi_input": True,
            "freq_dependent": fcoh, "coh_label": coh_label,
            "peak_drift": summary.get("peak_drift", 0.0),
            "plane_rotation_deg": summary["plane_rotation_deg"],
            "fnp_drift": summary.get("fnp_drift", 0.0),
            "constant_shape": summary["constant_shape"],
            "delegated": summary.get("wv_delegated"), "summary": summary,
            "von_mises": {"damage_rate": summary["von_mises"]["damage_rate"],
                          "life": summary["von_mises"]["life"]},
            "normal_plane": {"damage_rate": summary["normal_plane"]["damage_rate"],
                             "life": summary["normal_plane"]["life"]},
            "shear_plane": {"damage_rate": summary["shear_plane"]["damage_rate"],
                            "life": summary["shear_plane"]["life"]},
            "damage_rate": summary["damage_rate"], "life": summary["life"],
            "windowed": windowed, "monte_carlo": mc}


def _run_evolutionary_multi_input(ip, model, Hcols, G, omega, freqs_hz, m, C,
                                  mean_stress, ultimate, mc_dur, mc_seed, naz,
                                  npol, m28_summ):
    """M29: the FULLY EVOLUTIONARY MULTI-INPUT damage of the critical element — the
    input COHERENCE matrix S_ff(w, t) itself DRIFTING with time. Returns ``None``
    unless BOTH /IMPL/FATIG/MINPUT and /IMPL/FATIG/EVOL are set (i.e.
    /IMPL/FATIG/MULT/MINPUT/EVOL).

    Where M28 (``_run_multi_input_fatigue``) formed ONE stationary multi-input
    stress-tensor cross-PSD S_sigmasigma = H_sigma S_ff H_sigma^H and (composing with
    /EVOL / /JOINT) windowed its SCALAR shape with a FIXED coherence, M29 lets the
    INPUT COHERENCE drift window to window: it interpolates gamma_ab / theta_ab across
    the M26/M27 windows from the start pair (impl_mi_gamma / impl_mi_phase) to the end
    pair (impl_mi_gamma1 / impl_mi_phase1), assembles the per-window S_ff(t_i), forms
    the per-window multi-input S_sigmasigma,i and RE-SEARCHES the critical plane / F_np
    from the window's OWN tensor (so the plane may ROTATE as the coherence evolves),
    Miner-sums, and runs the non-stationary MULTI-INPUT multivariate Monte-Carlo
    cross-check (per-window blocks of the M28 correlated-input synthesiser, per-window
    plane projection, rainflow, Miner). Reuses the /EVOL drifting-shape schedule
    (impl_fatig_evol_fc0 .. _nwin) and the shared modulation /FUNCT the M26/M27 scalar
    paths use. Stored ALONGSIDE the M28 stationary multi-input and the M27
    single-input evolutionary numbers so the listing shows the coherence DRIFT / the
    per-window RMS / the critical-plane drift / the damage-life side by side. A PORT
    sub-flag (freimpl.F has no time-varying-coherence path). See
    implicit/evolutionary_multi_input.py."""
    if not (bool(getattr(ip, "impl_fatig_evol", False))
            and bool(getattr(ip, "impl_fatig_minput", False))):
        return None
    from . import evolutionary_multi_input as emi
    from . import nonstationary_fatigue as nsf
    fc0 = float(getattr(ip, "impl_fatig_evol_fc0", 0.0))
    fc1 = float(getattr(ip, "impl_fatig_evol_fc1", fc0))
    bw0 = float(getattr(ip, "impl_fatig_evol_bw0", 0.0))
    bw1 = float(getattr(ip, "impl_fatig_evol_bw1", bw0))
    nwin = max(1, int(getattr(ip, "impl_fatig_evol_nwin", 12)))
    scales, durations = _evol_schedule(ip, model, nwin)
    # the coherence schedule: the start pair (M28 gamma/phase) -> the end pair
    # (impl_mi_gamma1 / impl_mi_phase1). A negative gamma1 means NO coherence drift
    # (the coherence held at gamma0 — the M28 stationary special case, so /EVOL then
    # windows the multi-input SHAPE only, exactly like the M27 composition).
    gamma0 = float(getattr(ip, "impl_mi_gamma", 0.0))
    g1 = float(getattr(ip, "impl_mi_gamma1", -1.0))
    gamma1 = gamma0 if g1 < 0.0 else g1
    phase0 = math.radians(float(getattr(ip, "impl_mi_phase", 0.0)))
    phase1 = math.radians(float(getattr(ip, "impl_mi_phase1",
                                        getattr(ip, "impl_mi_phase", 0.0))))
    if int(getattr(ip, "impl_mi_cohmodel", 0)) == 1:
        # the EXPONENTIAL/decay coherence model DRIFT (a time-varying decay
        # coefficient) is DEFERRED for M29 — the constant-coherence start/end
        # schedule drives the evolutionary path; the exponential model is held
        # stationary (its stationary answer is the M28 exponential result).
        gamma1 = gamma0
    summary = emi.evolutionary_multi_input_summary(
        omega, Hcols, G, durations, m, C, gamma0=gamma0, gamma1=gamma1,
        phase0=phase0, phase1=phase1, fc=(fc0, fc1), bw=(bw0, bw1),
        scales=scales, mean_stress=mean_stress, ultimate=ultimate, naz=naz,
        npol=npol, drift=True)
    sc_w, wt = nsf.modulation_from_schedule(scales, durations)
    kurt = nsf.rms_modulation_kurtosis(sc_w, wt)
    mc = None
    if mc_dur > 0.0:
        tot = float(np.sum(durations))
        mc_durs = durations * (mc_dur / tot) if tot > 0 else durations
        mc = emi.evolutionary_multi_input_monte_carlo_damage(
            omega, Hcols, G, mc_durs, m, C, seed=mc_seed, gamma0=gamma0,
            gamma1=gamma1, phase0=phase0, phase1=phase1, fc=(fc0, fc1),
            bw=(bw0, bw1), scales=scales, mean_stress=mean_stress,
            ultimate=ultimate, naz=naz, npol=npol, reduction="shear_plane",
            summary=summary, measure=True)
    # the M28 STATIONARY multi-input reference (the coherence held at the mean) for
    # the side-by-side listing — the stationary Dirlik rate the drift departs from
    m28_ref = {k: float(m28_summ[k]["summary"]["dirlik"]["damage_rate"])
               for k in ("von_mises", "normal_plane", "shear_plane")}
    return {"fc": (fc0, fc1), "bw": (bw0, bw1), "nwin": nwin,
            "modfunct": int(getattr(ip, "impl_fatig_modfunct", 0) or 0),
            "scales": scales, "durations": durations, "kurtosis": kurt,
            "gamma0": gamma0, "gamma1": gamma1,
            "phase0_deg": math.degrees(phase0), "phase1_deg": math.degrees(phase1),
            "coherence_drift": summary["coherence_drift"],
            "constant_coherence": summary.get("constant_coherence", True),
            "constant_shape": summary["constant_shape"],
            "plane_rotation_deg": summary["plane_rotation_deg"],
            "fnp_drift": summary.get("fnp_drift", 0.0),
            "delegated": summary.get("delegated"), "summary": summary,
            "von_mises": {"damage_rate": summary["von_mises"]["damage_rate"],
                          "life": summary["von_mises"]["life"]},
            "normal_plane": {"damage_rate": summary["normal_plane"]["damage_rate"],
                             "life": summary["normal_plane"]["life"]},
            "shear_plane": {"damage_rate": summary["shear_plane"]["damage_rate"],
                            "life": summary["shear_plane"]["life"]},
            "damage_rate": summary["damage_rate"], "life": summary["life"],
            "stationary_multi_input": m28_ref, "monte_carlo": mc}


def _run_freq_evolutionary_multi_input(ip, model, Hcols, G, positions, omega,
                                       freqs_hz, m, C, mean_stress, ultimate,
                                       mc_dur, mc_seed, naz, npol, m28_summ,
                                       m29_ev):
    """M30: the FREQUENCY-DEPENDENT + TIME-VARYING (EVOLUTIONARY) INPUT COHERENCE
    damage of the critical element — the coherence matrix gamma_ab(f, t) varying
    with BOTH frequency AND time. Returns ``None`` unless /FCOH is set on top of
    /IMPL/FATIG/MINPUT and /EVOL (i.e. /IMPL/FATIG/MULT/MINPUT/EVOL/FCOH).

    Where M29 (``_run_evolutionary_multi_input``) drifted a per-window SCALAR
    gamma_ab(t_i) (frequency-FLAT) and held the M28 exponential coherence STATIONARY,
    M30 builds a FULL (nf, ninput, ninput) frequency-dependent coherence stack
    gamma_ab(f) that ALSO drifts window to window: either
      * the M28 EXPONENTIAL / convection field (cohmodel = 1) with a TIME-VARYING
        decay coefficient (impl_mi_decay -> impl_mi_decay1) and/or reference speed
        (impl_mi_speed -> impl_mi_speed1) — a turbulence field whose DECORRELATION
        FREQUENCY drifts through the mission; or
      * a per-pair MEASURED coherence SHAPE gamma(f) as a /FUNCT (impl_mi_gfunct0
        START -> impl_mi_gfunct1 END).
    It assembles the per-window frequency-dependent S_ff(w, t_i), forms the per-window
    multi-input S_sigmasigma,i and RE-SEARCHES the critical plane / F_np from the
    window's OWN tensor (so the plane may ROTATE as the coherence FREQUENCY-SHAPE
    evolves), Miner-sums, and runs the non-stationary MULTI-INPUT multivariate
    Monte-Carlo cross-check whose measured coherence SPECTRUM (per frequency band)
    tracks the target. Reuses the /EVOL drifting-shape schedule (impl_fatig_evol_fc0
    .. _nwin) and the shared modulation /FUNCT. Stored ALONGSIDE the M29
    scalar-coherence and the M28 frequency-dependent-stationary numbers so the listing
    shows the coherence FREQUENCY-SHAPE drift / the decorrelation-frequency drift / the
    critical-plane drift / the damage-life side by side. A PORT sub-flag (freimpl.F has
    no frequency-dependent-time-varying-coherence path). See
    implicit/freq_evolutionary_multi_input.py."""
    if not (bool(getattr(ip, "impl_mi_fcoh", False))
            and bool(getattr(ip, "impl_fatig_evol", False))
            and bool(getattr(ip, "impl_fatig_minput", False))):
        return None
    from . import freq_evolutionary_multi_input as fem
    from . import nonstationary_fatigue as nsf
    fc0 = float(getattr(ip, "impl_fatig_evol_fc0", 0.0))
    fc1 = float(getattr(ip, "impl_fatig_evol_fc1", fc0))
    bw0 = float(getattr(ip, "impl_fatig_evol_bw0", 0.0))
    bw1 = float(getattr(ip, "impl_fatig_evol_bw1", bw0))
    nwin = max(1, int(getattr(ip, "impl_fatig_evol_nwin", 12)))
    scales, durations = _evol_schedule(ip, model, nwin)
    ninput = int(np.asarray(Hcols).shape[2])
    phase0 = math.radians(float(getattr(ip, "impl_mi_phase", 0.0)))
    phase1 = math.radians(float(getattr(ip, "impl_mi_phase1",
                                        getattr(ip, "impl_mi_phase", 0.0))))

    # ---- build the START / END FREQUENCY-DEPENDENT coherence stacks -------------
    cohmodel = int(getattr(ip, "impl_mi_cohmodel", 0))
    gfunct0 = int(getattr(ip, "impl_mi_gfunct0", 0) or 0)
    gfunct1 = int(getattr(ip, "impl_mi_gfunct1", gfunct0) or gfunct0)
    if cohmodel == 1:
        # the EXPONENTIAL / convection field with a TIME-VARYING decay / speed
        decay0 = float(getattr(ip, "impl_mi_decay", 0.0))
        d1 = float(getattr(ip, "impl_mi_decay1", -1.0))
        decay1 = decay0 if d1 < 0.0 else d1
        speed0 = float(getattr(ip, "impl_mi_speed", 1.0)) or 1.0
        s1 = float(getattr(ip, "impl_mi_speed1", -1.0))
        speed1 = speed0 if s1 < 0.0 else (s1 or 1.0)
        g0 = fem.exponential_coherence_stack(freqs_hz, positions, decay0, speed0)
        g1 = fem.exponential_coherence_stack(freqs_hz, positions, decay1, speed1)
        coh_label = (f"EXPONENTIAL decay {decay0:.4g} -> {decay1:.4g}, "
                     f"speed {speed0:.4g} -> {speed1:.4g}")
    elif gfunct0 > 0:
        # a per-pair MEASURED coherence SHAPE gamma(f) /FUNCT, START -> END
        if gfunct0 not in model.functions or gfunct1 not in model.functions:
            return None
        shape0 = np.asarray(model.functions[gfunct0].eval(freqs_hz), dtype=float)
        shape1 = np.asarray(model.functions[gfunct1].eval(freqs_hz), dtype=float)
        g0 = fem.measured_coherence_stack(freqs_hz, shape0, ninput)
        g1 = fem.measured_coherence_stack(freqs_hz, shape1, ninput)
        coh_label = (f"MEASURED gamma(f) /FUNCT {gfunct0} -> {gfunct1}")
    else:
        # /FCOH requested but no frequency-dependent schedule supplied — nothing to
        # do beyond the M29 scalar path already run; skip cleanly.
        return None

    summary = fem.freq_evolutionary_multi_input_summary(
        omega, Hcols, G, durations, m, C, gamma0=g0, gamma1=g1, phase0=phase0,
        phase1=phase1, fc=(fc0, fc1), bw=(bw0, bw1), scales=scales,
        mean_stress=mean_stress, ultimate=ultimate, naz=naz, npol=npol,
        drift=True)
    sc_w, wt = nsf.modulation_from_schedule(scales, durations)
    kurt = nsf.rms_modulation_kurtosis(sc_w, wt)
    mc = None
    if mc_dur > 0.0:
        tot = float(np.sum(durations))
        mc_durs = durations * (mc_dur / tot) if tot > 0 else durations
        mc = fem.freq_evolutionary_multi_input_monte_carlo_damage(
            omega, Hcols, G, mc_durs, m, C, seed=mc_seed, gamma0=g0, gamma1=g1,
            phase0=phase0, phase1=phase1, fc=(fc0, fc1), bw=(bw0, bw1),
            scales=scales, mean_stress=mean_stress, ultimate=ultimate, naz=naz,
            npol=npol, reduction="shear_plane", summary=summary, measure=True)
    # the M28 STATIONARY frequency-dependent reference (the coherence held at the
    # mission-mean frequency shape) for the side-by-side listing
    m28_ref = {k: float(m28_summ[k]["summary"]["dirlik"]["damage_rate"])
               for k in ("von_mises", "normal_plane", "shear_plane")}
    # the M29 SCALAR-coherence reference (the frequency-flat drift) if it ran
    m29_ref = None
    if m29_ev is not None:
        m29_ref = {k: float(m29_ev[k]["damage_rate"])
                   for k in ("von_mises", "normal_plane", "shear_plane")}
    # a coarse band-coherence table of the START / END shapes for the listing
    _bc_c, band0 = fem.band_coherence(freqs_hz, g0)
    _bc_c2, band1 = fem.band_coherence(freqs_hz, g1)
    return {"fc": (fc0, fc1), "bw": (bw0, bw1), "nwin": nwin,
            "cohmodel": cohmodel, "coh_label": coh_label,
            "modfunct": int(getattr(ip, "impl_fatig_modfunct", 0) or 0),
            "scales": scales, "durations": durations, "kurtosis": kurt,
            "phase0_deg": math.degrees(phase0), "phase1_deg": math.degrees(phase1),
            "band_centres": _bc_c, "band_gamma0": band0, "band_gamma1": band1,
            "decorr_drift": summary.get("decorr_drift", 0.0),
            "freq_dependent": summary.get("freq_dependent", True),
            "constant_shape": summary["constant_shape"],
            "plane_rotation_deg": summary["plane_rotation_deg"],
            "fnp_drift": summary.get("fnp_drift", 0.0),
            "delegated": summary.get("delegated"), "summary": summary,
            "von_mises": {"damage_rate": summary["von_mises"]["damage_rate"],
                          "life": summary["von_mises"]["life"]},
            "normal_plane": {"damage_rate": summary["normal_plane"]["damage_rate"],
                             "life": summary["normal_plane"]["life"]},
            "shear_plane": {"damage_rate": summary["shear_plane"]["damage_rate"],
                            "life": summary["shear_plane"]["life"]},
            "damage_rate": summary["damage_rate"], "life": summary["life"],
            "stationary_multi_input": m28_ref,
            "scalar_evolutionary_multi_input": m29_ref, "monte_carlo": mc}


def _run_nonstationary_multiaxial(ip, summ, freqs, m, C, mean_stress, ultimate,
                                  mc_dur, mc_seed, model):
    """M25 (multiaxial): the NON-STATIONARY / EVOLUTIONARY-PSD correction of the
    M21 equivalent-stress reductions (von Mises / max-normal / max-shear critical
    plane). Returns ``None`` unless /IMPL/FATIG/NSTAT is set.

    Scales each reduction's shared-shape moment array by the SAME RMS modulation
    (the block Miner-sum + amplitude-modulated E[a^m], per reduction), and runs a
    non-stationary Monte-Carlo on the max-shear-plane LINEAR projection PSD.
    Stored ALONGSIDE the Gaussian reductions. Composes with /MULT / /NPROP /
    /SPEC / /NGAUSS — the block/modulation scaling on the equivalent scalar the
    multiaxial reductions produce (a full evolutionary tensor is deferred)."""
    if not bool(getattr(ip, "impl_fatig_nstat", False)):
        return None
    from . import nonstationary_fatigue as nsf
    modfunct = getattr(ip, "impl_fatig_modfunct", 0)
    nseg = int(getattr(ip, "impl_fatig_nstat_nseg", 0))
    scales, durations, func = _build_modulation(model, modfunct, nseg)
    scales_w, weights = nsf.modulation_from_schedule(scales, durations)
    blocks = [{"scale": float(s), "duration": float(d)}
              for s, d in zip(scales, durations)]
    out = {"modfunct": int(modfunct), "nseg": nseg, "nblocks": len(blocks),
           "scales": scales, "durations": durations, "weights": weights}
    for key in ("von_mises", "normal_plane", "shear_plane"):
        red = summ[key]
        base_moments = red["moments"]
        am = nsf.amplitude_modulated_summary(
            base_moments, scales_w, weights, m, C, mean_stress=mean_stress,
            ultimate=ultimate, stationary=red["summary"])
        block_summary = nsf.block_fatigue_summary(
            blocks, m, C, mean_stress=mean_stress, ultimate=ultimate,
            estimator="dirlik", base_moments=base_moments)
        out[key] = {"amplitude_modulated": am, "block": block_summary,
                    "kurtosis": am["kurtosis"], "kappa_ns": am["kappa_ns"],
                    "e_am": am["e_am"]}
    alpha2 = summ["von_mises"]["summary"]["params"]["alpha2"]
    out["kurtosis"] = out["von_mises"]["kurtosis"]
    out["bridge"] = nsf.bridge_to_nongaussian(
        scales_w, weights, m, alpha2=alpha2, bandwidth_correction=False)
    # a non-stationary Monte-Carlo on the equivalent-von-Mises scalar PSD (the
    # natural scalar analogue the M21 reduction stores as ``psd`` on the sweep
    # grid) — the non-stationary time-domain answer the block/modulated von-Mises
    # spectral estimate approximates
    ns_mc = None
    vm = summ["von_mises"]
    if mc_dur > 0.0 and freqs is not None and vm.get("psd") is not None:
        tot = float(np.sum(durations))
        mc_durs = durations * (mc_dur / tot) if tot > 0 else durations
        ns_mc = nsf.nonstationary_monte_carlo_damage(
            freqs, vm["psd"], m, C, scales, mc_durs, mc_seed,
            mean_stress=mean_stress, ultimate=ultimate)
    out["monte_carlo"] = ns_mc
    return out


def _lookup_psd_fatig(model, funct_id):
    """Resolve the input-PSD /FUNCT table by user id (the same map every load
    references), with a clear error — mirrors ``_lookup_psd``."""
    if funct_id is None or int(funct_id) <= 0:
        raise ValueError(
            "/IMPL/FATIG needs an input-PSD function id (the /FUNCT giving "
            "S_ff(f)); none was supplied on the card.")
    fid = int(funct_id)
    if fid not in model.functions:
        raise ValueError(
            f"/IMPL/FATIG input PSD /FUNCT/{fid} is not defined in the deck.")
    return model.functions[fid]


def _report_fatigue(log, fat, funct_id, base, base_dir, frf, nev):
    """Print the RANDOM-VIBRATION (SPECTRAL) FATIGUE listing block: the sweep,
    the critical channel, the spectral descriptors, and the damage / equivalent
    stress / life of every estimator (mirrors ``run_random_response``'s
    reporting)."""
    p = fat["summary"]["params"]
    band = f"[{frf['freqs'].min():.5E}, {frf['freqs'].max():.5E}]"
    log.info(f"      MODES / FRF SOURCE . . . . . . . : {nev} / REAL modal")
    log.info(f"      INPUT . . . . . . . . . . . . . : "
             f"{'BASE ACCELERATION PSD (dir %d)' % base_dir if base else 'FORCE PSD'}"
             f"  /FUNCT/{int(funct_id)}")
    log.info(f"      SWEEP BAND (HZ) / POINTS . . . . : {band} / "
             f"{len(frf['freqs'])}")
    log.info(f"      S-N CURVE  N = C S^-m  . . . . . : "
             f"m = {fat['sn_m']:.4G} , C = {fat['sn_C']:.5E}")
    if fat["mean_stress"]:
        log.info(f"      MEAN-STRESS (GOODMAN) / ULT  . . : "
                 f"{fat['mean_stress']:.5E} / {fat['ultimate']:.5E}")
    log.info(f"      CRITICAL STRESS CHANNEL  . . . . : {fat['critical_label']}")
    log.info(f"      RMS STRESS (sigma = sqrt m0) . . : {p['sigma']:.5E}")
    log.info(f"      RATES  nu0 / nup (HZ)  . . . . . : "
             f"{p['nu0']:.5E} / {p['nup']:.5E}")
    log.info(f"      IRREGULARITY alpha2 / BANDWIDTH  : "
             f"{p['alpha2']:.5F} / {p['epsilon']:.5F}")
    log.info("      METHOD              DAMAGE RATE      LIFE (T_f)       "
             "S_eq (RANGE)")
    for key, name in (("narrow_band", "NARROW-BAND (Bendat)"),
                      ("dirlik", "DIRLIK 1985"),
                      ("wirsching_light", "WIRSCHING-LIGHT"),
                      ("tovo_benasciutti", "TOVO-BENASCIUTTI")):
        r = fat["summary"][key]
        life = r["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      {name:20s}{r['damage_rate']:14.5E}   "
                 f"{life_s:>14s}   {r['s_eq']:14.5E}")
    if fat["monte_carlo"] is not None:
        mc = fat["monte_carlo"]
        life = mc["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      {'MONTE-CARLO rainflow':20s}{mc['damage_rate']:14.5E}"
                 f"   {life_s:>14s}   (n_cyc = {mc['ncycles']:.0f}, "
                 f"seed check)")
    # M24: the NON-GAUSSIAN / KURTOSIS block, printed ALONGSIDE the M20 Gaussian
    # numbers so the Gaussian and the kurtosis-corrected lives show side by side.
    if fat.get("nongaussian") is not None:
        _report_nongaussian(log, fat["nongaussian"])
    # M25: the NON-STATIONARY / EVOLUTIONARY-PSD block, printed ALONGSIDE the M20
    # stationary numbers so the stationary and non-stationary lives show side by
    # side.
    if fat.get("nonstationary") is not None:
        _report_nonstationary(log, fat["nonstationary"])
    # M26: the FULLY EVOLUTIONARY / NON-SEPARABLE-PSD block, printed ALONGSIDE the
    # M20 stationary and M25 non-stationary numbers so all three (stationary,
    # RMS-non-stationary, shape-evolutionary) lives show side by side.
    if fat.get("evolutionary") is not None:
        _report_evolutionary(log, fat["evolutionary"], fat["summary"],
                             fat.get("nonstationary"))
    # M31: the CONTINUOUS WIGNER-VILLE INSTANTANEOUS block, printed ALONGSIDE the
    # M26 windowed-evolutionary numbers (the continuous integral next to the
    # windowed sum).
    if fat.get("wigner_ville") is not None:
        _report_wigner_ville(log, fat["wigner_ville"])


def _report_wigner_ville(log, wv):
    """Print the CONTINUOUS WIGNER-VILLE / LOEVE INSTANTANEOUS TIME-FREQUENCY (M31)
    listing block: the continuous instantaneous spectrum settings (the fine instant
    grid / the Cohen-class smoothing), the instantaneous spectral-PEAK drift, the
    window-boundary-caveat shrink (fine grid vs coarse windows) and — for the scalar
    path — the continuous-integral Dirlik damage / life ALONGSIDE the M26 windowed
    Miner-SUM, plus the continuous non-separable Monte-Carlo. Handles the scalar,
    the JOINT-TENSOR (multiaxial) and the MULTI-INPUT sub-entry shapes."""
    log.info("\n     ** CONTINUOUS WIGNER-VILLE INSTANTANEOUS TIME-FREQUENCY "
             "FATIGUE **  (/IMPL/FATIG/WVILLE)")
    fc0, fc1 = wv["fc"]
    bw0, bw1 = wv["bw"]
    log.info(f"      INSTANTANEOUS SPECTRUM S_WV(w,t) . : fine grid nt = {wv['nt']}"
             f"  ({wv['nwin']} windows x refine {wv['refine']})")
    log.info(f"      COHEN-CLASS SMOOTHING (x-terms)  . : {wv['smooth']:.4g}  "
             f"(0 = raw instantaneous WVD)")
    log.info(f"      DRIFTING SHAPE fc / bw (HZ)  . . . : "
             f"{fc0:.4g}->{fc1:.4g} / {bw0:.4g}->{bw1:.4g}")
    log.info(f"      INSTANTANEOUS PEAK-FREQ DRIFT (HZ) : {wv['peak_drift']:.4g}")
    if wv.get("boundary_jump_windowed") is not None:
        log.info(f"      WINDOW-BOUNDARY CAVEAT (fine/coarse): "
                 f"{wv['boundary_jump']:.4g} / {wv['boundary_jump_windowed']:.4g}"
                 f"  (continuous shrinks it)")
    if wv.get("tensor") or wv.get("multi_input"):
        if wv.get("coh_label"):
            log.info(f"      COHERENCE MODEL gamma(.,t) . . . . : {wv['coh_label']}")
        log.info(f"      CRITICAL-PLANE ROTATION (deg)  . . : "
                 f"{wv['plane_rotation_deg']:.4g}  (F_np drift "
                 f"{wv['fnp_drift']:.4g})")
        win = wv.get("windowed") or {}
        for key, name in (("von_mises", "VON MISES"),
                          ("normal_plane", "MAX-NORMAL"),
                          ("shear_plane", "MAX-SHEAR")):
            r = wv[key]
            life = r["life"]
            life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
            log.info(f"      {name:11s} CONTINUOUS / WINDOWED  : "
                     f"{r['damage_rate']:.5E} / {win.get(key, 0.0):.5E}  "
                     f"life {life_s}")
    else:
        dk = wv["summary"]["dirlik"]
        life = dk["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        wd = wv.get("windowed_damage_rate")
        wd_s = "n/a" if wd is None else f"{wd:.5E}"
        log.info(f"      DIRLIK CONTINUOUS / WINDOWED RATE  : "
                 f"{dk['damage_rate']:.5E} / {wd_s}   life {life_s}")
    if wv.get("monte_carlo") is not None:
        mc = wv["monte_carlo"]
        life = mc["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      CONTINUOUS MONTE-CARLO DAMAGE/LIFE : "
                 f"{mc['damage_rate']:.5E} / {life_s}")
    # M32: the TIME-VARYING NON-GAUSSIAN correction of the continuous spectrum,
    # printed ALONGSIDE the M31 Gaussian-continuous numbers when /NGAUSS composes
    # with /WVILLE (the leptokurtic amplification lambda_ng(t) drifting with time).
    if wv.get("nongaussian") is not None:
        _report_nongaussian_wigner_ville(log, wv["nongaussian"])
    # M33: the JOINT-TENSOR NON-GAUSSIAN distribution (the vector Hermite transform of
    # the correlated tensor, the resolved plane inheriting the induced kurtosis),
    # printed ALONGSIDE the M32 equivalent-scalar and the M31 Gaussian-continuous
    # numbers when /JOINT composes with /NGAUSS + /WVILLE and a per-component kurtosis
    # line is given.
    if wv.get("joint_nongaussian") is not None:
        _report_joint_nongaussian(log, wv["joint_nongaussian"])


def _report_joint_nongaussian(log, ng):
    """Print the JOINT-TENSOR NON-GAUSSIAN DISTRIBUTION (M33) listing block: the
    per-component target kurtoses imposed on the 6 Voigt tensor components (a vector
    Winterstein-Hermite / translation transform), the INDUCED critical-plane kurtosis
    the resolved plane inherits from the joint tensor statistics, the covariance
    preservation error, and the JOINT-tensor-non-Gaussian damage / life ALONGSIDE the
    M32 EQUIVALENT-SCALAR (kurtosis on the resolved scalar) and the M31/M27 Gaussian
    numbers, plus the multivariate non-Gaussian Monte-Carlo (its resolved-projection
    sample kurtosis tracking the induced value) — the M33 <-> M32 boundary made
    explicit."""
    log.info("\n     ** JOINT-TENSOR NON-GAUSSIAN DISTRIBUTION FATIGUE **  "
             "(/IMPL/FATIG/NGAUSS+JOINT+WVILLE)")
    jk = ng.get("joint_kurt", ())
    jk_s = " ".join(f"{v:.3g}" for v in jk)
    log.info(f"      PER-COMPONENT KURTOSIS g4_c (Voigt): {jk_s}")
    log.info(f"      INDUCED CRITICAL-PLANE g4^s  . . . : "
             f"{ng['induced_kurt_min']:.4F} .. {ng['induced_kurt_max']:.4F}  "
             f"(mean {ng['induced_kurt_mean']:.4F})")
    bw = "ON (Benasciutti-Tovo alpha2(t))" if ng["bandwidth_correction"] else "OFF"
    log.info(f"      INDUCED lambda_ng(t) DRIFT . . . . : "
             f"{ng['lambda_min']:.4F} .. {ng['lambda_max']:.4F}  "
             f"(mean {ng['lambda_mean']:.4F})  [BW-atten {bw}]")
    log.info(f"      COVARIANCE PRESERVATION ERROR  . . : "
             f"{ng['preservation_error']:.4E}  (max off-diagonal drift; marginals "
             f"exact)")
    log.info(f"      CRITICAL-PLANE ROTATION (deg)  . . : "
             f"{ng['plane_rotation_deg']:.4g}")
    gwin = ng.get("gaussian") or {}
    se = ng.get("scalar_equivalent") or {}
    for key, name in (("von_mises", "VON MISES"), ("normal_plane", "MAX-NORMAL"),
                      ("shear_plane", "MAX-SHEAR")):
        r = ng[key]
        life = r["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      {name:11s} JOINT / SCALAR-EQ / GAUSS : "
                 f"{r['damage_rate']:.5E} / {se.get(key, 0.0):.5E} / "
                 f"{gwin.get(key, 0.0):.5E}  life {life_s}")
    if ng.get("monte_carlo") is not None:
        mc = ng["monte_carlo"]
        life = mc["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      MULTIVARIATE NON-GAUSS MC DMG/LIFE : "
                 f"{mc['damage_rate']:.5E} / {life_s}  (proj sample g4 = "
                 f"{mc.get('kurtosis', 3.0):.3F})")
    # M34: the EXACT translation-process CORRELATION-DISTORTION INVERSION block, printed
    # when /EXACT (or /NORTA) composes with the M33 joint path — the NORTA correlation
    # matching driving the covariance preservation error to ~0.
    if ng.get("exact_covariance") is not None:
        _report_exact_covariance(log, ng, ng["exact_covariance"])


def _report_exact_covariance(log, ng, ex):
    """Print the EXACT TRANSLATION-PROCESS CORRELATION-DISTORTION INVERSION (M34) block:
    the Grigoriu / Nataf / NORTA correlation matching that solves the underlying-Gaussian
    correlation rho^U so the component-wise Winterstein-Hermite transform of the joint 6x6
    tensor reproduces the TARGET covariance EXACTLY — the (now ~0) covariance preservation
    error NEXT TO the M33 leading-order value, and the EXACT-covariance induced kurtosis /
    damage / life ALONGSIDE the M33 leading-order joint numbers (the same reductions on a
    covariance-EXACT joint tensor)."""
    log.info("\n     ** EXACT TRANSLATION-PROCESS CORRELATION-DISTORTION INVERSION **  "
             "(/IMPL/FATIG/NGAUSS+JOINT+WVILLE+EXACT)")
    log.info(f"      NORTA / NATAF / GRIGORIU CORRELATION MATCHING + HIGHAM PD REPAIR")
    log.info(f"      COVARIANCE PRESERVATION EXACT / M33 : "
             f"{ex['preservation_error']:.4E} / "
             f"{ex['preservation_error_leading']:.4E}  (target reproduced to machine "
             f"precision)")
    log.info(f"      INDUCED CRITICAL-PLANE g4^s (EXACT). : "
             f"{ex['induced_kurt_min']:.4F} .. {ex['induced_kurt_max']:.4F}  "
             f"(mean {ex['induced_kurt_mean']:.4F})")
    for key, name in (("von_mises", "VON MISES"), ("normal_plane", "MAX-NORMAL"),
                      ("shear_plane", "MAX-SHEAR")):
        r = ex[key]
        rl = ng[key]
        life = r["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      {name:11s} EXACT / M33-LEADING RATE  : "
                 f"{r['damage_rate']:.5E} / {rl['damage_rate']:.5E}  life {life_s}")
    if ex.get("monte_carlo") is not None:
        mc = ex["monte_carlo"]
        life = mc["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      COVARIANCE-EXACT MC DMG/LIFE . . . : "
                 f"{mc['damage_rate']:.5E} / {life_s}  (sample-cov err "
                 f"{mc.get('sample_cov_error', 0.0):.4E}, proj g4 = "
                 f"{mc.get('kurtosis', 3.0):.3F})")


def _report_nongaussian_wigner_ville(log, ng):
    """Print the TIME-VARYING NON-GAUSSIAN INSTANTANEOUS TIME-FREQUENCY (M32) listing
    block: the kurtosis-vs-time schedule gamma_4(t), the DRIFT of the per-instant
    amplification lambda_ng(t) (min .. max), and the non-Gaussian continuous-integral
    damage / life ALONGSIDE the M31 Gaussian-continuous and the M24 stationary-non-
    Gaussian numbers, plus the non-Gaussian non-stationary Monte-Carlo (its induced
    sample kurtosis tracking gamma_4(t)). Handles the scalar and JOINT-TENSOR shapes."""
    log.info("\n     ** TIME-VARYING NON-GAUSSIAN INSTANTANEOUS TIME-FREQUENCY "
             "FATIGUE **  (/IMPL/FATIG/NGAUSS+WVILLE)")
    log.info(f"      KURTOSIS-VS-TIME g4(t) SCHEDULE . . : {ng['schedule']}  "
             f"(range {ng['gamma4_range']:.4g})")
    bw = "ON (Benasciutti-Tovo alpha2(t))" if ng["bandwidth_correction"] else "OFF"
    log.info(f"      INSTANTANEOUS lambda_ng(t) DRIFT . : "
             f"{ng['lambda_min']:.4F} .. {ng['lambda_max']:.4F}  "
             f"(mean {ng['lambda_mean']:.4F})  [BW-atten {bw}]")
    if ng.get("tensor"):
        log.info(f"      CRITICAL-PLANE ROTATION (deg)  . . : "
                 f"{ng['plane_rotation_deg']:.4g}  (F_np drift "
                 f"{ng['fnp_drift']:.4g})")
        gwin = ng.get("gaussian") or {}
        for key, name in (("von_mises", "VON MISES"),
                          ("normal_plane", "MAX-NORMAL"),
                          ("shear_plane", "MAX-SHEAR")):
            r = ng[key]
            life = r["life"]
            life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
            log.info(f"      {name:11s} NON-GAUSS / GAUSS RATE : "
                     f"{r['damage_rate']:.5E} / {gwin.get(key, 0.0):.5E}  "
                     f"life {life_s}")
    else:
        life = ng["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        st = ng.get("stationary_ng_damage_rate")
        st_s = "n/a" if st is None else f"{st:.5E}"
        log.info(f"      DIRLIK NON-GAUSS(t) / GAUSS RATE  : "
                 f"{ng['damage_rate']:.5E} / {ng['gaussian_damage_rate']:.5E}  "
                 f"life {life_s}")
        log.info(f"      M24 STATIONARY-NON-GAUSS RATE  . . : {st_s}  "
                 f"(constant-kurtosis reference)")
    if ng.get("monte_carlo") is not None:
        mc = ng["monte_carlo"]
        life = mc["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      NON-GAUSS MONTE-CARLO DAMAGE/LIFE  : "
                 f"{mc['damage_rate']:.5E} / {life_s}  (sample g4 = "
                 f"{mc.get('kurtosis', 3.0):.3F})")


def _report_evolutionary(log, ev, stationary_summary, nonstationary):
    """Print the FULLY EVOLUTIONARY / NON-SEPARABLE-PSD (M26) listing block: the
    drifting-shape schedule (swept centre frequency / broadening bandwidth,
    windows), the per-window SHAPE breakdown (RMS / rate / alpha2 drift — the point
    of a NON-separable spectrum), the window Miner-sum per estimator ALONGSIDE the
    M20 stationary and (if present) the M25 non-stationary damage, and the
    non-separable Monte-Carlo cross-check."""
    log.info("\n     ** FULLY EVOLUTIONARY / NON-SEPARABLE-PSD FATIGUE **  "
             "(/IMPL/FATIG/EVOL)")
    fc0, fc1 = ev["fc"]
    bw0, bw1 = ev["bw"]
    log.info(f"      DRIFTING SHAPE  fc0->fc1 (HZ) . . : "
             f"{fc0:.4G} -> {fc1:.4G}   (bw {bw0:.4G} -> {bw1:.4G})")
    log.info(f"      SPECTROGRAM WINDOWS  . . . . . . : {ev['nwin']}  "
             f"(RMS level /FUNCT/{ev['modfunct']})")
    log.info(f"      NON-SEPARABLE ? / INDUCED g4 . . : "
             f"{'NO (constant shape -> M25)' if ev['constant_shape'] else 'YES'}"
             f"  /  g4 = {ev['kurtosis']:.4F}")
    # the per-window SHAPE drift (RMS / nu0 / alpha2) — a compact spectrogram view
    wins = ev["summary"]["windows"]
    a2 = [w["alpha2"] for w in wins]
    nu0 = [w["nu0"] for w in wins]
    sig = [w["sigma"] for w in wins]
    log.info(f"      WINDOW alpha2  (min..max)  . . . : "
             f"{min(a2):.4F} .. {max(a2):.4F}  (narrow=1, wide->0)")
    log.info(f"      WINDOW nu0 (HZ)(min..max)  . . . : "
             f"{min(nu0):.4G} .. {max(nu0):.4G}  (centre-freq drift)")
    log.info(f"      WINDOW RMS     (min..max)  . . . : "
             f"{min(sig):.5E} .. {max(sig):.5E}")
    # per-estimator: stationary rate | evolutionary (window Miner-sum) rate | life
    log.info("      METHOD              STATIONARY RATE  EVOLUTION RATE   "
             "EVOLUTION LIFE")
    for key, name in (("narrow_band", "NARROW-BAND (Bendat)"),
                      ("dirlik", "DIRLIK 1985"),
                      ("wirsching_light", "WIRSCHING-LIGHT"),
                      ("tovo_benasciutti", "TOVO-BENASCIUTTI")):
        drS = stationary_summary[key]["damage_rate"]
        r = ev["summary"][key]
        life = r["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      {name:20s}{drS:14.5E}   "
                 f"{r['damage_rate']:14.5E}   {life_s:>14s}")
    if nonstationary is not None:
        # side-by-side: the M25 (separable RMS) block Miner-sum vs the M26
        # (non-separable shape) window Miner-sum, both Dirlik
        m25 = nonstationary["block"]["damage_rate"]
        m26 = ev["summary"]["dirlik"]["damage_rate"]
        log.info(f"      M25 (sep RMS) vs M26 (shape) . . : "
                 f"{m25:.5E} / {m26:.5E}  (Dirlik window Miner-sum)")
    if ev.get("monte_carlo") is not None:
        mc = ev["monte_carlo"]
        life = mc["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        tag = "M25-delegated" if mc.get("delegated") else "non-separable"
        log.info(f"      EVOLUTION MONTE-CARLO  . . . . . : "
                 f"{mc['damage_rate']:.5E}  ({tag}, sample g4 = "
                 f"{mc['kurtosis']:.3F}, life {life_s})")


def _report_nonstationary(log, ns):
    """Print the NON-STATIONARY / EVOLUTIONARY-PSD (M25) listing block: the RMS
    mission profile (blocks / modulation), the induced kurtosis, the block
    Miner-sum + amplitude-modulated damage ALONGSIDE the stationary numbers, the
    M25<->M24 kurtosis bridge and the non-stationary Monte-Carlo cross-check."""
    log.info("\n     ** NON-STATIONARY / EVOLUTIONARY-PSD FATIGUE **  "
             "(/IMPL/FATIG/NSTAT)")
    log.info(f"      RMS MODULATION /FUNCT / BLOCKS  . : "
             f"/FUNCT/{ns['modfunct']} / {ns['nblocks']} block(s)")
    sc = np.asarray(ns["scales"], dtype=float)
    log.info(f"      RMS SCALE RANGE (min..max) . . . : "
             f"{sc.min():.4F} .. {sc.max():.4F}  "
             f"(overall RMS x {ns['rms_scale']:.4F})")
    log.info(f"      INDUCED KURTOSIS g4  . . . . . . : "
             f"{ns['kurtosis']:.4F}  (3 = stationary Gaussian)")
    br = ns["bridge"]
    log.info(f"      M25<->M24 BRIDGE  kappa_ns / l_ng: "
             f"{br['kappa_ns']:.5F} / {br['lambda_ng']:.5F}  "
             f"(ratio {br['ratio']:.4F})")
    # per-estimator: stationary rate | non-stationary (E[a^m]-scaled) rate | life
    am = ns["amplitude_modulated"]
    log.info("      METHOD              STATIONARY RATE  NON-STAT RATE    "
             "NON-STAT LIFE")
    for key, name in (("narrow_band", "NARROW-BAND (Bendat)"),
                      ("dirlik", "DIRLIK 1985"),
                      ("wirsching_light", "WIRSCHING-LIGHT"),
                      ("tovo_benasciutti", "TOVO-BENASCIUTTI")):
        r = am[key]
        life = r["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      {name:20s}{r['stationary_damage_rate']:14.5E}   "
                 f"{r['damage_rate']:14.5E}   {life_s:>14s}")
    bl = ns["block"]
    bl_life = "INF" if not np.isfinite(bl["life"]) else f"{bl['life']:.5E}"
    log.info(f"      BLOCK MINER-SUM (Dirlik) . . . . : "
             f"{bl['damage_rate']:.5E} / {bl_life}  "
             f"(E[a^m] = {ns['e_am']:.5F}, {ns['nblocks']} blocks)")
    if ns.get("monte_carlo") is not None:
        mc = ns["monte_carlo"]
        life = mc["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      NON-STAT MONTE-CARLO . . . . . . : "
                 f"{mc['damage_rate']:.5E}  (sample g4 = {mc['kurtosis']:.3F}, "
                 f"life {life_s})")


def _report_nongaussian(log, ng):
    """Print the NON-GAUSSIAN / KURTOSIS (M24) listing block: the target kurtosis
    / skewness, the Hermite coefficients, the closed-form correction factor
    lambda_ng and, for each estimator, the CORRECTED damage rate / life ALONGSIDE
    the Gaussian one, plus the non-Gaussian Monte-Carlo cross-check."""
    log.info("\n     ** NON-GAUSSIAN / KURTOSIS FATIGUE **      "
             "(/IMPL/FATIG/NGAUSS)")
    log.info(f"      TARGET KURTOSIS g4 / SKEWNESS g3 . : "
             f"{ng['gamma4']:.4F} / {ng['gamma3']:.4F}  (g4 = 3 is Gaussian)")
    log.info(f"      HERMITE h3 / h4 / kappa  . . . . . : "
             f"{ng['h3']:+.5F} / {ng['h4']:+.5F} / {ng['kappa']:.5F}")
    bw = "ON (Benasciutti-Tovo alpha2)" if ng["bandwidth_correction"] else "OFF"
    log.info(f"      CORRECTION lambda_ng / BW-ATTEN  . : "
             f"{ng['lambda_ng']:.5F}  [bandwidth atten = {bw}]")
    log.info("      METHOD              GAUSSIAN RATE    NON-GAUSS RATE   "
             "NON-GAUSS LIFE")
    for key, name in (("narrow_band", "NARROW-BAND (Bendat)"),
                      ("dirlik", "DIRLIK 1985"),
                      ("wirsching_light", "WIRSCHING-LIGHT"),
                      ("tovo_benasciutti", "TOVO-BENASCIUTTI")):
        r = ng[key]
        life = r["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      {name:20s}{r['gaussian_damage_rate']:14.5E}   "
                 f"{r['damage_rate']:14.5E}   {life_s:>14s}")
    if ng.get("monte_carlo") is not None:
        mc = ng["monte_carlo"]
        life = mc["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      {'NON-GAUSS MONTE-CARLO':20s}{mc['damage_rate']:14.5E}"
                 f"   (sample g4 = {mc['kurtosis']:.3F}, life {life_s})")


# ============================================================================
# Engine-card driver (/IMPL/FATIG/MULT) — M21 multiaxial / critical-plane
# ============================================================================

def _run_multiaxial(model, ip, log, result, frf, Sigma, channels, psd_tab,
                    m_sn, C_sn, mean_stress, ultimate, mc_dur, mc_seed, base,
                    base_dir, funct_id, nev, nplane):
    """/IMPL/FATIG/MULT (M21): multiaxial / critical-plane spectral fatigue.

    Consumes the M20 vector stress modes (``Sigma``, ``channels``) read-only,
    groups them into per-element 6-Voigt blocks, and for each solid/shell
    element forms the stress-tensor cross-PSD S_sigmasigma = H_sigma S_ff
    H_sigma^H, reduces it to a scalar EQUIVALENT-stress PSD three ways —
    equivalent VON MISES (Preumont & Piefort 1994 / Pitoiset & Preumont 2000,
    the trace(Q S) projection), MAX-NORMAL-stress and MAX-SHEAR-stress CRITICAL
    PLANE (Carpinteri-Spagnoli / Cristofori-Susmel-Tovo) — and runs the M20
    estimators on each. Picks the CRITICAL ELEMENT (highest von Mises Dirlik
    damage), stores the full multiaxial result on ``result.fatigue`` and prints
    the listing. A PORT sub-card: the open-source engine has no multiaxial
    spectral-fatigue path (module docstring). See
    ``implicit/multiaxial_fatigue.py`` for the theory."""
    from . import multiaxial_fatigue as mf
    from . import spectral_fatigue as sf

    Sff = np.clip(np.asarray(psd_tab.eval(frf["freqs"]), dtype=float), 0.0,
                  None)
    omega = np.asarray(frf["omega"], dtype=float)

    # per-element 6-Voigt blocks (solids / shells only — a full stress tensor)
    blocks = element_voigt_blocks(channels)
    if not blocks:
        raise ValueError(
            "/IMPL/FATIG/MULT found no multiaxial (6-component) stress "
            "element — the model has only scalar-channel elements "
            "(truss / spring). Use /IMPL/FATIG (the M20 scalar path) for a "
            "uniaxial model, or add solid / shell elements.")

    # every element's Voigt stress FRF, then the CHEAP equivalent-von-Mises
    # Dirlik damage per element -> the critical element (Pitoiset & Preumont's
    # comparison uses the von Mises damage as the ranking metric)
    naz, npol = int(nplane), max(7, int(nplane) // 2 + 1)
    elem_vm_rate = np.zeros(len(blocks))
    for k, (_name, _e, _base, cols) in enumerate(blocks):
        Hv = element_voigt_frf(frf, Sigma, cols)          # (nf, 6)
        Svm = mf.equivalent_vonmises_psd(Hv, Sff)
        mom_vm = spectral_moments(omega, Svm, nmax=4)
        if mom_vm[0] <= 0.0:
            continue
        elem_vm_rate[k] = sf.dirlik_damage(
            mom_vm, m_sn, C_sn, mean_stress, ultimate)["damage_rate"]
    kcrit = int(np.argmax(elem_vm_rate)) if len(blocks) else 0
    cname, ce, cbase, ccols = blocks[kcrit]

    # the full multiaxial summary on the critical element (all three reductions)
    Hcrit = element_voigt_frf(frf, Sigma, ccols)
    summ = mf.multiaxial_fatigue_summary(
        Hcrit, Sff, omega, m_sn, C_sn, mean_stress=mean_stress,
        ultimate=ultimate, naz=naz, npol=npol)

    # Monte-Carlo cross-check on the MAX-SHEAR critical plane's linear scalar
    # (the projected history is exactly Gaussian; the quadratic von Mises MC is
    # deferred — module docstring)
    mc = None
    if mc_dur > 0.0:
        sp = summ["shear_plane"]
        mc = mf.monte_carlo_multiaxial_damage(
            frf["freqs"], summ["Scross"], sp["proj"], m_sn, C_sn, mc_dur,
            mc_seed, mean_stress=mean_stress, ultimate=ultimate)

    # M22: the NON-PROPORTIONAL critical-plane TIME-DOMAIN path count runs
    # ALONGSIDE the M21 spectral reductions (a NEW parallel path — the M21
    # answer above is fully formed and left byte-identical). It reuses the
    # critical element's stress-tensor cross-PSD (summ["Scross"]) to synthesise
    # the correlated stress-component histories (seeded via mcdur/seed) and
    # searches the Findley / Fatemi-Socie / shear-path critical plane. See
    # implicit/nonproportional_fatigue.py.
    nprop = bool(getattr(ip, "impl_fatig_nprop", False))
    spec_np = bool(getattr(ip, "impl_fatig_spec", False))
    npres = None
    if nprop:
        from . import nonproportional_fatigue as npf
        if mc_dur <= 0.0:
            raise ValueError(
                "/IMPL/FATIG/MULT/NPROP needs a Monte-Carlo record length "
                "(mcdur on card line 2: m C zeta mean ult mcdur seed) — the "
                "non-proportional path count runs on the synthesised history.")
        k_np = float(getattr(ip, "impl_fatig_k", 0.3))
        sigy = float(getattr(ip, "impl_fatig_sigy", 1.0))
        amp = str(getattr(ip, "impl_fatig_amp", "mrh"))
        npres = npf.nonproportional_summary(
            frf["freqs"], summ["Scross"], m_sn, C_sn, mc_dur, mc_seed,
            k=k_np, sigma_y=sigy, amp_method=amp, naz=naz, npol=npol)
        # M23: the SPECTRAL non-proportional estimate runs ALONGSIDE the M22
        # time-domain path count (a NEW parallel path — the M22 answer above is
        # fully formed and left byte-identical). It reuses the SAME critical
        # element's 6x6 spectral-MOMENT matrices (summ["Mmats"]) — NO synthesised
        # history — and estimates the frequency-domain F_np + critical-plane
        # damage directly, so the listing shows the proportional-spectral (M21),
        # non-proportional time-domain (M22) and non-proportional spectral (M23)
        # answers side by side. See implicit/spectral_nonproportional_fatigue.py.
        if spec_np:
            from . import spectral_nonproportional_fatigue as snp
            npres["spectral"] = snp.spectral_nonproportional_summary(
                summ["Mmats"], m_sn, C_sn, k=k_np, sigma_y=sigy,
                mean_stress=mean_stress, ultimate=ultimate, naz=naz, npol=npol)

    # M24: the NON-GAUSSIAN / KURTOSIS correction of the MULTIAXIAL equivalent
    # scalars, run ALONGSIDE the M21 spectral reductions (a NEW parallel path —
    # the reductions above are fully formed and left byte-identical). It scales
    # each reduction's (von Mises / max-normal / max-shear) Gaussian spectral
    # damage by the closed-form lambda_ng of the target kurtosis, and runs a
    # non-Gaussian Monte-Carlo on the max-shear-plane LINEAR projection (the M21
    # multivariate synthesiser + the Hermite transform). See
    # implicit/nongaussian_fatigue.py.
    nongaussian = _run_nongaussian_multiaxial(
        ip, summ, frf["freqs"], m_sn, C_sn, mean_stress, ultimate, mc_dur,
        mc_seed)

    # M25: the NON-STATIONARY / EVOLUTIONARY-PSD correction of the MULTIAXIAL
    # equivalent scalars, run ALONGSIDE the M21 spectral reductions (a NEW
    # parallel path — the reductions above are left byte-identical). It scales
    # each reduction's shared-shape moments by the RMS mission profile (block
    # Miner-sum + amplitude-modulated E[a^m]) and runs a non-stationary
    # Monte-Carlo on the von-Mises scalar PSD. See implicit/nonstationary_fatigue.py.
    nonstationary = _run_nonstationary_multiaxial(
        ip, summ, frf["freqs"], m_sn, C_sn, mean_stress, ultimate, mc_dur,
        mc_seed, model)

    # M26: the FULLY EVOLUTIONARY / NON-SEPARABLE-PSD correction of the MULTIAXIAL
    # equivalent scalars, run ALONGSIDE the M21 reductions (a NEW parallel path —
    # the reductions above are left byte-identical). It applies the drifting-shape
    # window per time-window to each reduction's frequency-resolved scalar PSD
    # (each window its OWN full moments) and Miner-sums, with a non-separable
    # Monte-Carlo on the von-Mises drifting spectrogram. See
    # implicit/evolutionary_fatigue.py.
    evolutionary = _run_evolutionary_multiaxial(
        ip, summ, frf["freqs"], m_sn, C_sn, mean_stress, ultimate, mc_dur,
        mc_seed, model)

    # M27: the FULLY EVOLUTIONARY MULTIAXIAL JOINT-TENSOR correction, run ALONGSIDE
    # the M21 reductions AND the M26 scalar-evolutionary correction (a NEW parallel
    # path — everything above is left byte-identical). Where M26 windowed the
    # equivalent SCALAR of a FIXED reduction, M27 windows the FULL 6x6 stress-TENSOR
    # cross-PSD per window and RE-SEARCHES the critical plane / F_np from the
    # window's OWN tensor (so the plane may ROTATE, F_np may DRIFT), Miner-summing
    # the per-window MULTIAXIAL damages, with a non-stationary MULTIVARIATE
    # Monte-Carlo. See implicit/joint_evolutionary_fatigue.py.
    joint_evolutionary = _run_joint_evolutionary(
        ip, summ, frf, m_sn, C_sn, mean_stress, ultimate, mc_dur, mc_seed,
        model, naz, npol)

    # M31: the CONTINUOUS WIGNER-VILLE INSTANTANEOUS 6x6 stress-TENSOR spectrum runs
    # ALONGSIDE the M21 reductions AND the M26/M27 windowed corrections (a NEW
    # parallel path — everything above is left byte-identical). It evaluates the
    # tensor spectrum CONTINUOUSLY at a fine instant grid with a per-instant plane
    # re-search (of which the M27 window re-search is the coarse-grid limit,
    # delegated byte-identically). See implicit/wigner_ville_fatigue.py.
    wigner_ville = _run_wigner_ville_multiaxial(
        ip, summ, frf, joint_evolutionary, m_sn, C_sn, mean_stress, ultimate,
        mc_dur, mc_seed, model, naz, npol)

    # M32: the TIME-VARYING NON-GAUSSIAN correction of the M31 continuous TENSOR
    # spectrum (per-instant lambda_ng(t) on the re-searched critical-plane /
    # von-Mises equivalent scalar) runs when BOTH /NGAUSS and /WVILLE are set — the
    # convergence of the M24 kurtosis correction and the M31 continuous tensor,
    # reported as a sub-entry of the wigner_ville entry. See implicit/
    # nongaussian_wigner_ville_fatigue.py.
    if wigner_ville is not None:
        wigner_ville["nongaussian"] = _run_nongaussian_wigner_ville_multiaxial(
            ip, summ, frf, wigner_ville, m_sn, C_sn, mean_stress, ultimate,
            mc_dur, mc_seed, model, naz, npol)

    # M33: the JOINT-TENSOR NON-GAUSSIAN distribution (a VECTOR component-wise
    # Winterstein-Hermite / translation transform of the correlated 6x6 tensor imposing
    # a PER-COMPONENT kurtosis, the resolved critical plane INHERITING the induced
    # kurtosis) runs when /JOINT + /NGAUSS + /WVILLE are all set AND a per-component
    # kurtosis line is given — the FIRST item M32 deferred, reported as a sub-entry of
    # the wigner_ville entry ALONGSIDE the M32 equivalent-scalar (nongaussian) and the
    # M31 Gaussian-continuous numbers (both left byte-identical). See implicit/
    # joint_nongaussian_fatigue.py.
    if wigner_ville is not None:
        wigner_ville["joint_nongaussian"] = _run_joint_nongaussian_multiaxial(
            ip, summ, frf, wigner_ville, m_sn, C_sn, mean_stress, ultimate,
            mc_dur, mc_seed, model, naz, npol)

    result.fatigue = {
        "multiaxial": True, "nonproportional": nprop,
        "spectral_nonproportional": spec_np, "nongaussian": nongaussian,
        "nonstationary": nonstationary, "evolutionary": evolutionary,
        "joint_evolutionary": joint_evolutionary,
        "wigner_ville": wigner_ville,
        "channels": channels, "voigt_blocks": blocks,
        "critical_element": (cname, ce, cbase),
        "critical_label": cbase,
        "elem_vm_dirlik_rate": elem_vm_rate,
        "von_mises": summ["von_mises"], "normal_plane": summ["normal_plane"],
        "shear_plane": summ["shear_plane"], "Mmats": summ["Mmats"],
        "Scross": summ["Scross"], "monte_carlo": mc, "nprop_result": npres,
        "freqs": np.asarray(frf["freqs"], dtype=float), "omega": omega,
        "Sff": Sff, "sn_m": m_sn, "sn_C": C_sn, "mean_stress": mean_stress,
        "ultimate": ultimate, "base": base, "stress_modes": Sigma,
        # keep an M20-compatible top-level "summary" (the von Mises equivalent,
        # the natural scalar analogue) so downstream code that reads the M20
        # dict shape still finds a summary
        "summary": summ["von_mises"]["summary"],
    }
    _report_multiaxial(log, result.fatigue, funct_id, base, base_dir, frf, nev)


# ============================================================================
# M28 — MULTI-INPUT / partially-coherent random-vibration RESPONSE & FATIGUE
# (a NEW parallel path — attaches a "multi_input" sub-entry ALONGSIDE the
#  single-input M19/M20/M21 answers, never mutating them)
# ============================================================================

def _multi_input_force_pattern(model, loads, cload_funct):
    """The unit-amplitude spatial FORCE pattern F_a (numnod, 3) of input pattern
    ``a`` — the sum of the /CLOAD contributions whose time-/FUNCT id is
    ``cload_funct`` (each contributes ``scale`` along its unit ``direction`` on
    its node group). The /FUNCT carries only the SPATIAL pattern here (the
    frequency content is the input's auto-PSD G_a(f)); a unit /FUNCT/1 (the
    M19-M27 example convention) gives the bare pattern, matching how the
    single-input path calls ``loads.external_forces(1.0, F, x0)``."""
    n = model.numnod
    F = np.zeros((n, 3))
    target = model.functions.get(int(cload_funct))
    if target is None:
        raise ValueError(
            f"/IMPL/FATIG/MINPUT input references /CLOAD /FUNCT/{cload_funct}, "
            "which is not defined in the deck.")
    hit = False
    for idx, direction, fct, scale, _sens, _t_scale in loads.cloads:
        if id(fct) == id(target):
            F[idx] += float(scale) * np.asarray(direction, dtype=float)
            hit = True
    if not hit:
        raise ValueError(
            f"/IMPL/FATIG/MINPUT input pattern /FUNCT/{cload_funct} matches no "
            "/CLOAD in the deck — each input names the /CLOAD /FUNCT id that "
            "identifies its spatial load pattern.")
    return F


def _assemble_multi_input(model, ip, log, basis, freqs_hz, zeta, loads):
    """Build the M28 multi-input driving data from the card's input table:
    the per-input modal-coordinate FRF columns q_cols (nf, nmode, ninput), the
    input auto-PSDs G_a(f), the coherence/phase model and the assembled
    Hermitian input cross-PSD S_ff(f) (nf, ninput, ninput). Returns a dict.

    A PORT sub-flag: force-pattern multi-input (the common case). Base-
    acceleration multi-input is DEFERRED (the per-direction participation feed
    would need a base-input column stack — PORTING_GUIDE M28 deferral)."""
    from . import multi_input_response as mir
    from .modal_response import modal_frequency_response

    inputs = tuple(getattr(ip, "impl_mi_inputs", ()))
    if not inputs:
        raise ValueError(
            "/IMPL/FATIG/MINPUT (or /IMPL/PSD/MULTI) needs an input-pattern "
            "table (ninput cohmodel gamma phase, then a 'cload_funct psd_funct "
            "[x y z]' row per input) after the sweep / S-N lines.")
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    nf = freqs_hz.size
    ninput = len(inputs)
    n = model.numnod

    # per-input auto-PSDs G_a(f) and modal FRF columns q_a(f)
    G = np.zeros((nf, ninput))
    q_cols = np.zeros((nf, basis.nmode, ninput), dtype=complex)
    positions = np.zeros((ninput, 3))
    for a, (cf, pf, x, y, z) in enumerate(inputs):
        positions[a] = (x, y, z)
        psd_tab = _lookup_psd_fatig(model, pf)
        G[:, a] = np.clip(np.asarray(psd_tab.eval(freqs_hz), dtype=float), 0.0,
                          None)
        Fa = _multi_input_force_pattern(model, loads, cf)
        frf_a = modal_frequency_response(basis, Fa, np.zeros((n, 3)), freqs_hz,
                                         zeta)
        q_cols[:, :, a] = frf_a["q"]

    # coherence model -> gamma (constant matrix OR frequency-dependent stack)
    cohmodel = int(getattr(ip, "impl_mi_cohmodel", 0))
    gamma_c = float(getattr(ip, "impl_mi_gamma", 0.0))
    phase_deg = float(getattr(ip, "impl_mi_phase", 0.0))
    phase = math.radians(phase_deg)
    if cohmodel == 1:
        decay = float(getattr(ip, "impl_mi_decay", 0.0))
        speed = float(getattr(ip, "impl_mi_speed", 1.0)) or 1.0
        gstack, _th = mir.exponential_coherence(freqs_hz, positions, decay,
                                                ref_speed=speed, phase=phase)
        gamma_arg = gstack
        coh_label = f"EXPONENTIAL (decay={decay:.4g}, speed={speed:.4g})"
    else:
        gamma_arg = gamma_c
        coh_label = f"CONSTANT (gamma={gamma_c:.4g}, phase={phase_deg:.4g} deg)"
    mi = mir.input_cross_psd_matrix(G, gamma=gamma_arg, phase=phase)
    return {"inputs": inputs, "ninput": ninput, "G": G, "q_cols": q_cols,
            "Sff": mi["Sff"], "positions": positions, "coh_label": coh_label,
            "projected": mi["projected"], "min_eig": mi["min_eig"],
            "nclipped": mi["nclipped"], "freqs": freqs_hz,
            "omega": 2.0 * np.pi * freqs_hz}


def _run_multi_input_fatigue(model, ip, log, result, basis, Sigma, channels,
                             freqs_hz, zeta, loads, m_sn, C_sn, mean_stress,
                             ultimate, mc_dur, mc_seed, nplane):
    """/IMPL/FATIG/MINPUT (M28): MULTI-INPUT / partially-coherent spectral
    fatigue. Assembles the input cross-PSD S_ff, builds the per-input stress FRF
    columns, forms the multi-input stress-tensor cross-PSD S_sigmasigma = H_sigma
    S_ff H_sigma^H on the critical element, reduces it by the SAME M21 machinery
    (equivalent von Mises / max-normal / max-shear critical plane) and — because
    the multi-input S_sigmasigma flows into them UNCHANGED — composes with the
    M22-M27 corrections exactly as the single-input /MULT path does. Runs the
    multi-input Monte-Carlo (input-level synthesis + the M21-on-S_sigmasigma
    stress-level check) and stores everything on ``result.fatigue['multi_input']``
    ALONGSIDE the single-input answer. Never mutates the single-input result."""
    from . import multi_input_response as mir
    from . import multi_input_fatigue as mif
    from . import multiaxial_fatigue as mf
    from . import spectral_fatigue as sf

    if bool(getattr(ip, "impl_fatig_base", False)):
        log.warning("        /IMPL/FATIG/MINPUT with base excitation is "
                    "DEFERRED (M28) — using the force-pattern feed; base "
                    "multi-input needs the per-direction participation stack.",
                    "IMPL/FATIG/MINPUT")
    data = _assemble_multi_input(model, ip, log, basis, freqs_hz, zeta, loads)
    Sff = data["Sff"]
    omega = data["omega"]
    naz, npol = int(nplane), max(7, int(nplane) // 2 + 1)

    # per-input stress FRF columns (nf, nchan, ninput)
    Hs_cols = mir.stress_frf_columns(data["q_cols"], Sigma)
    blocks = element_voigt_blocks(channels)

    multi = {"inputs": data["inputs"], "ninput": data["ninput"],
             "coh_label": data["coh_label"], "projected": data["projected"],
             "min_eig": data["min_eig"], "nclipped": data["nclipped"],
             "Sff": Sff, "freqs": freqs_hz, "omega": omega, "G": data["G"]}

    if blocks:
        # critical element by equivalent-von-Mises Dirlik on the MULTI-INPUT
        # stress-tensor cross-PSD (the same ranking metric the single-input
        # /MULT path uses, but formed from H S_ff H^H)
        elem_rate = np.zeros(len(blocks))
        for k, (_nm, _e, _bs, cols) in enumerate(blocks):
            Hcols = mir.element_voigt_frf_columns(Hs_cols, cols)
            Scr = mir.stress_tensor_cross_psd_multi(Hcols, Sff)
            Svm = mir.equivalent_vonmises_psd_multi(Scr)
            mom = spectral_moments(omega, Svm, nmax=4)
            if mom[0] <= 0.0:
                continue
            elem_rate[k] = sf.dirlik_damage(mom, m_sn, C_sn, mean_stress,
                                            ultimate)["damage_rate"]
        kcrit = int(np.argmax(elem_rate)) if len(blocks) else 0
        cname, ce, cbase, ccols = blocks[kcrit]
        Hcrit = mir.element_voigt_frf_columns(Hs_cols, ccols)
        Scross = mir.stress_tensor_cross_psd_multi(Hcrit, Sff)
        summ = mir.multi_input_multiaxial_summary(
            Scross, omega, m_sn, C_sn, mean_stress=mean_stress,
            ultimate=ultimate, naz=naz, npol=npol)

        # a minimal FRF-like handle (only omega/freqs are read by the M22-M27
        # correction helpers — they take S_sigmasigma from ``summ`` read-only)
        frf_like = {"omega": omega, "freqs": freqs_hz}

        mc = None
        mc_input = None
        if mc_dur > 0.0:
            sp = summ["shear_plane"]
            # stress-level MC (the M21 synthesiser on the multi-input
            # S_sigmasigma — bit-identical to single-input when rank-1)
            mc = mf.monte_carlo_multiaxial_damage(
                freqs_hz, Scross, sp["proj"], m_sn, C_sn, mc_dur, mc_seed,
                mean_stress=mean_stress, ultimate=ultimate)
            # input-level MC (the INDEPENDENT cross-check: synthesise the
            # correlated inputs, drive through the stress columns, sum)
            mc_input = mif.monte_carlo_multi_input_damage(
                freqs_hz, Sff, Hcrit, sp["proj"], m_sn, C_sn, mc_dur, mc_seed,
                mean_stress=mean_stress, ultimate=ultimate)

        # the M22-M27 corrections compose UNCHANGED (they read summ read-only)
        nprop = bool(getattr(ip, "impl_fatig_nprop", False))
        spec_np = bool(getattr(ip, "impl_fatig_spec", False))
        npres = None
        if nprop and mc_dur > 0.0:
            from . import nonproportional_fatigue as npf
            k_np = float(getattr(ip, "impl_fatig_k", 0.3))
            sigy = float(getattr(ip, "impl_fatig_sigy", 1.0))
            amp = str(getattr(ip, "impl_fatig_amp", "mrh"))
            npres = npf.nonproportional_summary(
                freqs_hz, summ["Scross"], m_sn, C_sn, mc_dur, mc_seed, k=k_np,
                sigma_y=sigy, amp_method=amp, naz=naz, npol=npol)
            if spec_np:
                from . import spectral_nonproportional_fatigue as snp
                npres["spectral"] = snp.spectral_nonproportional_summary(
                    summ["Mmats"], m_sn, C_sn, k=k_np, sigma_y=sigy,
                    mean_stress=mean_stress, ultimate=ultimate, naz=naz,
                    npol=npol)
        nongaussian = _run_nongaussian_multiaxial(
            ip, summ, freqs_hz, m_sn, C_sn, mean_stress, ultimate, mc_dur,
            mc_seed)
        nonstationary = _run_nonstationary_multiaxial(
            ip, summ, freqs_hz, m_sn, C_sn, mean_stress, ultimate, mc_dur,
            mc_seed, model)
        evolutionary = _run_evolutionary_multiaxial(
            ip, summ, freqs_hz, m_sn, C_sn, mean_stress, ultimate, mc_dur,
            mc_seed, model)
        joint_evolutionary = _run_joint_evolutionary(
            ip, summ, frf_like, m_sn, C_sn, mean_stress, ultimate, mc_dur,
            mc_seed, model, naz, npol)
        # M29: the FULLY EVOLUTIONARY MULTI-INPUT path — the input COHERENCE matrix
        # itself DRIFTING with time (as opposed to the M27 joint_evolutionary above,
        # which windows the SHAPE of a stationary-coherence multi-input tensor).
        # Runs only when BOTH /MINPUT and /EVOL are set; reported ALONGSIDE the M28
        # stationary multi-input and the M27 single-input evolutionary numbers.
        evolutionary_multi_input = _run_evolutionary_multi_input(
            ip, model, Hcrit, data["G"], omega, freqs_hz, m_sn, C_sn,
            mean_stress, ultimate, mc_dur, mc_seed, naz, npol, summ)
        # M30: the FREQUENCY-DEPENDENT + TIME-VARYING (evolutionary) coherence path —
        # the coherence gamma_ab(f, t) drifting in BOTH frequency AND time (as opposed
        # to M29 above, which drifts a frequency-FLAT scalar coherence). Runs only when
        # /FCOH is also set; reported ALONGSIDE the M29 scalar-coherence and the M28
        # frequency-dependent-stationary numbers.
        freq_evolutionary_multi_input = _run_freq_evolutionary_multi_input(
            ip, model, Hcrit, data["G"], data["positions"], omega, freqs_hz,
            m_sn, C_sn, mean_stress, ultimate, mc_dur, mc_seed, naz, npol, summ,
            evolutionary_multi_input)
        # M31: the CONTINUOUS WIGNER-VILLE INSTANTANEOUS multi-input tensor spectrum
        # — the coherence-driven tensor evaluated CONTINUOUSLY (the M29 scalar or M30
        # frequency-dependent coherence drifting continuously, per-instant plane
        # re-search, Miner-INTEGRATED). Runs only when /WVILLE + /MINPUT + /EVOL are
        # set; the M29/M30 windowed answers are delegated to (byte-identical) in the
        # coarse-grid limit and reported ALONGSIDE.
        wigner_ville_multi = _run_wigner_ville_multi_input(
            ip, model, Hcrit, data["G"], data["positions"], omega, freqs_hz,
            m_sn, C_sn, mean_stress, ultimate, mc_dur, mc_seed, naz, npol,
            evolutionary_multi_input, freq_evolutionary_multi_input)

        multi.update({
            "multiaxial": True, "critical_element": (cname, ce, cbase),
            "critical_label": cbase, "elem_vm_dirlik_rate": elem_rate,
            "von_mises": summ["von_mises"], "normal_plane": summ["normal_plane"],
            "shear_plane": summ["shear_plane"], "Mmats": summ["Mmats"],
            "Scross": Scross, "monte_carlo": mc, "monte_carlo_input": mc_input,
            "nprop_result": npres, "nonproportional": nprop,
            "spectral_nonproportional": spec_np, "nongaussian": nongaussian,
            "nonstationary": nonstationary, "evolutionary": evolutionary,
            "joint_evolutionary": joint_evolutionary,
            "evolutionary_multi_input": evolutionary_multi_input,
            "freq_evolutionary_multi_input": freq_evolutionary_multi_input,
            "wigner_ville": wigner_ville_multi,
            "summary": summ["von_mises"]["summary"],
        })
    else:
        # scalar-channel fallback (truss / spring model): per-channel MULTI-INPUT
        # PSD S_jj = sum_ab H_ja S_ff,ab conj(H_jb); the M20 estimators on that
        nchan = Hs_cols.shape[1]
        Sjj = np.einsum("fja,fab,fjb->fj", Hs_cols, Sff,
                        np.conj(Hs_cols)).real
        dirlik = np.zeros(nchan)
        for j in range(nchan):
            mom = spectral_moments(omega, Sjj[:, j], nmax=4)
            if mom[0] > 0.0:
                dirlik[j] = sf.dirlik_damage(mom, m_sn, C_sn, mean_stress,
                                             ultimate)["damage_rate"]
        jcrit = int(np.argmax(dirlik)) if nchan else 0
        crit_mom = spectral_moments(omega, Sjj[:, jcrit], nmax=4)
        summary = sf.fatigue_summary(crit_mom, m_sn, C_sn, mean_stress,
                                     ultimate)
        multi.update({
            "multiaxial": False, "critical_channel": jcrit,
            "critical_label": channels[jcrit][3], "moments": crit_mom,
            "Sjj": Sjj, "dirlik_rate": dirlik, "summary": summary,
            "monte_carlo": None, "monte_carlo_input": None,
        })

    result.fatigue["multi_input"] = multi
    _report_multi_input(log, multi, result.fatigue)


def _report_multi_input(log, mi, single):
    """Print the M28 MULTI-INPUT listing block ALONGSIDE the single-input numbers
    (the side-by-side view: the single-input critical answer and the multi-input
    critical answer with the coherence diagnostics)."""
    log.info("\n     ** MULTI-INPUT / PARTIALLY-COHERENT SPECTRAL FATIGUE **"
             "  (/IMPL/FATIG/MINPUT)")
    log.info(f"      NUMBER OF INPUTS . . . . . . . . : {mi['ninput']}")
    log.info(f"      COHERENCE MODEL  . . . . . . . . : {mi['coh_label']}")
    log.info(f"      INPUT CROSS-PSD PSD-PROJECTED  . : "
             f"{'YES (%d bins)' % mi['nclipped'] if mi['projected'] else 'NO (already valid)'}"
             f"  (min eig {mi['min_eig']:.3E})")
    for a, (cf, pf, x, y, z) in enumerate(mi["inputs"]):
        log.info(f"        input {a+1}: /CLOAD /FUNCT/{cf}  auto-PSD "
                 f"/FUNCT/{pf}  pos ({x:.4g},{y:.4g},{z:.4g})")
    if mi.get("multiaxial"):
        cn, ce, cb = mi["critical_element"]
        log.info(f"      CRITICAL ELEMENT . . . . . . . . : {cb}")
        red = mi["von_mises"]["summary"]
        log.info(f"      VON MISES DIRLIK DAMAGE RATE . . : "
                 f"{red['dirlik']['damage_rate']:.5E}  life "
                 f"{red['dirlik']['life']:.5E}")
        np_ = mi["normal_plane"]["summary"]["dirlik"]
        sh = mi["shear_plane"]["summary"]["dirlik"]
        log.info(f"      MAX-NORMAL / MAX-SHEAR DAMAGE  . : "
                 f"{np_['damage_rate']:.5E} / {sh['damage_rate']:.5E}")
        if mi.get("monte_carlo") is not None:
            log.info(f"      MC (stress-level / input-level) . : "
                     f"{mi['monte_carlo']['damage_rate']:.5E} / "
                     f"{mi['monte_carlo_input']['damage_rate']:.5E}")
        # the single-input critical answer, for the side-by-side view
        if single is not None and single.get("summary"):
            s = single["summary"].get("dirlik", {})
            if s:
                log.info(f"      (single-input critical DIRLIK) . . : "
                         f"{s.get('damage_rate', 0.0):.5E}  life "
                         f"{s.get('life', 0.0):.5E}")
        # M29: the FULLY EVOLUTIONARY MULTI-INPUT block (the coherence DRIFT), printed
        # ALONGSIDE the M28 stationary multi-input numbers above
        if mi.get("evolutionary_multi_input") is not None:
            _report_evolutionary_multi_input(log, mi["evolutionary_multi_input"])
        # M30: the FREQUENCY-DEPENDENT + TIME-VARYING coherence block, printed
        # ALONGSIDE the M29 scalar-coherence + M28 stationary numbers above
        if mi.get("freq_evolutionary_multi_input") is not None:
            _report_freq_evolutionary_multi_input(
                log, mi["freq_evolutionary_multi_input"])
        # M31: the CONTINUOUS WIGNER-VILLE INSTANTANEOUS multi-input block, printed
        # ALONGSIDE the M29 scalar-coherence / M30 frequency-dependent windowed
        # numbers (the continuous coherence-driven tensor next to the windowed one)
        if mi.get("wigner_ville") is not None:
            _report_wigner_ville(log, mi["wigner_ville"])
    else:
        red = mi["summary"]["dirlik"]
        log.info(f"      CRITICAL CHANNEL . . . . . . . . : {mi['critical_label']}")
        log.info(f"      DIRLIK DAMAGE RATE / LIFE . . . . : "
                 f"{red['damage_rate']:.5E} / {red['life']:.5E}")


def _report_evolutionary_multi_input(log, ev):
    """Print the FULLY EVOLUTIONARY MULTI-INPUT (M29) listing block: the coherence
    DRIFT schedule (start -> end coherence), the critical-plane ROTATION driven by the
    evolving coherence, the per-window response RMS / coherence drift, and, for each
    reduction (von Mises / max-normal / max-shear), the evolutionary multi-input
    window Miner-sum Dirlik damage rate / life ALONGSIDE the M28 STATIONARY
    multi-input one, plus the non-stationary MULTI-INPUT multivariate Monte-Carlo."""
    log.info("\n     ** FULLY EVOLUTIONARY MULTI-INPUT FATIGUE **  "
             "(/IMPL/FATIG/MULT/MINPUT/EVOL)")
    log.info(f"      COHERENCE SCHEDULE gamma . . . . . : "
             f"{ev['gamma0']:.4g} -> {ev['gamma1']:.4g}  (drift "
             f"{ev['coherence_drift']:.4g})")
    if abs(ev['phase1_deg'] - ev['phase0_deg']) > 1e-9:
        log.info(f"      PHASE SCHEDULE theta (deg) . . . : "
                 f"{ev['phase0_deg']:.4g} -> {ev['phase1_deg']:.4g}")
    fc0, fc1 = ev["fc"]
    bw0, bw1 = ev["bw"]
    log.info(f"      DRIFTING-SHAPE fc / bw (HZ)  . . . : "
             f"{fc0:.4g}->{fc1:.4g} / {bw0:.4g}->{bw1:.4g}  ({ev['nwin']} windows)")
    log.info(f"      CRITICAL-PLANE ROTATION (deg)  . . : "
             f"{ev['plane_rotation_deg']:.4g}  (F_np drift {ev['fnp_drift']:.4g})")
    if ev.get("delegated"):
        log.info(f"      (reduced via delegation) . . . . : {ev['delegated']}")
    st = ev.get("stationary_multi_input", {})
    for key, name in (("von_mises", "VON MISES"),
                      ("normal_plane", "MAX-NORMAL"),
                      ("shear_plane", "MAX-SHEAR")):
        r = ev[key]
        life = r["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      {name:11s} EVOL / STATIONARY RATE : "
                 f"{r['damage_rate']:.5E} / {st.get(key, 0.0):.5E}  "
                 f"life {life_s}")
    if ev.get("monte_carlo") is not None:
        mc = ev["monte_carlo"]
        log.info(f"      NON-STAT MULTI-INPUT MC (shear) DAMAGE / LIFE : "
                 f"{mc['damage_rate']:.5E} / "
                 f"{'INF' if not np.isfinite(mc['life']) else '%.5E' % mc['life']}")
        wg = mc.get("window_gamma")
        if wg is not None and np.size(wg):
            with np.errstate(invalid="ignore"):
                log.info("      MEASURED per-window COHERENCE  . : ["
                         + " ".join(f"{g:.3f}" for g in np.atleast_1d(wg)) + "]")


def _report_freq_evolutionary_multi_input(log, ev):
    """Print the FREQUENCY-DEPENDENT + TIME-VARYING (evolutionary) COHERENCE (M30)
    listing block: the coherence FREQUENCY-SHAPE schedule (the start / end
    band-resolved coherence and the model), the DECORRELATION-FREQUENCY drift, the
    critical-plane ROTATION driven by the evolving frequency-shape, and, for each
    reduction, the M30 frequency-dependent evolutionary damage rate ALONGSIDE the M29
    scalar-coherence and the M28 frequency-dependent-STATIONARY ones, plus the
    non-stationary MULTI-INPUT Monte-Carlo and the MEASURED per-window band-resolved
    coherence SPECTRUM tracking the target gamma_ab(f, t_i)."""
    log.info("\n     ** FREQUENCY-DEPENDENT EVOLUTIONARY MULTI-INPUT FATIGUE **  "
             "(/IMPL/FATIG/MULT/MINPUT/EVOL/FCOH)")
    log.info(f"      COHERENCE MODEL (gamma_ab(f,t))  . : {ev['coh_label']}")
    bc = ev.get("band_centres")
    if bc is not None and np.size(bc):
        b0 = "/".join(f"{g:.2f}" for g in ev["band_gamma0"])
        b1 = "/".join(f"{g:.2f}" for g in ev["band_gamma1"])
        log.info(f"      BAND-COHERENCE gamma(f) START/END : [{b0}] -> [{b1}]")
    log.info(f"      DECORRELATION-FREQ DRIFT (HZ)  . . : {ev['decorr_drift']:.4g}")
    fc0, fc1 = ev["fc"]
    bw0, bw1 = ev["bw"]
    log.info(f"      DRIFTING-SHAPE fc / bw (HZ)  . . . : "
             f"{fc0:.4g}->{fc1:.4g} / {bw0:.4g}->{bw1:.4g}  ({ev['nwin']} windows)")
    log.info(f"      CRITICAL-PLANE ROTATION (deg)  . . : "
             f"{ev['plane_rotation_deg']:.4g}  (F_np drift {ev['fnp_drift']:.4g})")
    if ev.get("delegated"):
        log.info(f"      (reduced via delegation) . . . . : {ev['delegated']}")
    st = ev.get("stationary_multi_input", {})
    sc = ev.get("scalar_evolutionary_multi_input") or {}
    for key, name in (("von_mises", "VON MISES"),
                      ("normal_plane", "MAX-NORMAL"),
                      ("shear_plane", "MAX-SHEAR")):
        r = ev[key]
        life = r["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      {name:11s} FREQ-EVOL RATE . . : "
                 f"{r['damage_rate']:.5E}  life {life_s}")
        log.info(f"                  (M29 scalar / M28 stat) : "
                 f"{sc.get(key, 0.0):.5E} / {st.get(key, 0.0):.5E}")
    if ev.get("monte_carlo") is not None:
        mc = ev["monte_carlo"]
        log.info(f"      NON-STAT MULTI-INPUT MC (shear) DAMAGE / LIFE : "
                 f"{mc['damage_rate']:.5E} / "
                 f"{'INF' if not np.isfinite(mc['life']) else '%.5E' % mc['life']}")
        wg = mc.get("window_gamma")
        if wg is not None and np.size(wg):
            with np.errstate(invalid="ignore"):
                log.info("      MEASURED per-window band-mean COH: ["
                         + " ".join(f"{g:.3f}" for g in np.atleast_1d(wg)) + "]")
        wd = mc.get("window_decorr")
        if wd is not None and np.size(wd) and np.any(np.isfinite(wd)):
            with np.errstate(invalid="ignore"):
                log.info("      MEASURED per-window DECORR-FREQ  : ["
                         + " ".join(("INF" if not np.isfinite(g) else f"{g:.1f}")
                                    for g in np.atleast_1d(wd)) + "]")


def _report_multiaxial(log, fat, funct_id, base, base_dir, frf, nev):
    """Print the MULTIAXIAL / CRITICAL-PLANE SPECTRAL FATIGUE listing block: the
    sweep, the critical element, the equivalent-von-Mises + max-normal +
    max-shear critical-plane reductions (each with the four M20 estimators), the
    critical-plane orientations and the Monte-Carlo cross-check."""
    band = f"[{frf['freqs'].min():.5E}, {frf['freqs'].max():.5E}]"
    log.info(f"      MODES / FRF SOURCE . . . . . . . : {nev} / REAL modal")
    log.info(f"      INPUT . . . . . . . . . . . . . : "
             f"{'BASE ACCELERATION PSD (dir %d)' % base_dir if base else 'FORCE PSD'}"
             f"  /FUNCT/{int(funct_id)}")
    log.info(f"      SWEEP BAND (HZ) / POINTS . . . . : {band} / "
             f"{len(frf['freqs'])}")
    log.info(f"      S-N CURVE  N = C S^-m  . . . . . : "
             f"m = {fat['sn_m']:.4G} , C = {fat['sn_C']:.5E}")
    if fat["mean_stress"]:
        log.info(f"      MEAN-STRESS (GOODMAN) / ULT  . . : "
                 f"{fat['mean_stress']:.5E} / {fat['ultimate']:.5E}")
    log.info(f"      CRITICAL ELEMENT . . . . . . . . : {fat['critical_label']}")

    def _block(title, red):
        p = red["summary"]["params"]
        log.info(f"      --- {title} ---")
        log.info(f"        RMS EQUIV STRESS (sqrt m0) . . : {p['sigma']:.5E}")
        log.info(f"        RATES  nu0 / nup (HZ)  . . . . : "
                 f"{p['nu0']:.5E} / {p['nup']:.5E}")
        log.info(f"        IRREGULARITY alpha2 / BANDWIDTH: "
                 f"{p['alpha2']:.5F} / {p['epsilon']:.5F}")
        if "normal" in red:
            n = red["normal"]
            log.info(f"        CRITICAL PLANE NORMAL  . . . . : "
                     f"[{n[0]:+.4F} {n[1]:+.4F} {n[2]:+.4F}]")
        log.info("        METHOD              DAMAGE RATE      LIFE (T_f)")
        for key, name in (("narrow_band", "NARROW-BAND"),
                          ("dirlik", "DIRLIK 1985"),
                          ("wirsching_light", "WIRSCHING-LIGHT"),
                          ("tovo_benasciutti", "TOVO-BENASCIUTTI")):
            r = red["summary"][key]
            life = r["life"]
            life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
            log.info(f"        {name:18s}{r['damage_rate']:14.5E}   "
                     f"{life_s:>14s}")

    _block("EQUIVALENT VON MISES (Preumont-Piefort)", fat["von_mises"])
    _block("MAX-NORMAL-STRESS CRITICAL PLANE", fat["normal_plane"])
    _block("MAX-SHEAR-STRESS CRITICAL PLANE", fat["shear_plane"])
    if fat["monte_carlo"] is not None:
        mc = fat["monte_carlo"]
        life = mc["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      MONTE-CARLO (shear-plane rainflow) DAMAGE / LIFE : "
                 f"{mc['damage_rate']:.5E} / {life_s}  "
                 f"(n_cyc = {mc['ncycles']:.0f})")
    # M22: the NON-PROPORTIONAL critical-plane path-counting block, printed
    # ALONGSIDE the M21 spectral reductions so the listing shows the
    # non-proportional correction explicitly (the M21 numbers above are the
    # proportional-spectral answer; these are the rotating-shear-path answer).
    if fat.get("nprop_result") is not None:
        _report_nonproportional(log, fat["nprop_result"])
    # M24: the NON-GAUSSIAN / KURTOSIS correction of the multiaxial reductions,
    # printed ALONGSIDE the Gaussian numbers.
    if fat.get("nongaussian") is not None:
        _report_nongaussian_multiaxial(log, fat["nongaussian"])
    # M25: the NON-STATIONARY / EVOLUTIONARY-PSD correction of the multiaxial
    # reductions, printed ALONGSIDE the Gaussian numbers.
    if fat.get("nonstationary") is not None:
        _report_nonstationary_multiaxial(log, fat["nonstationary"])
    # M26: the FULLY EVOLUTIONARY / NON-SEPARABLE-PSD correction of the multiaxial
    # reductions, printed ALONGSIDE the Gaussian numbers.
    if fat.get("evolutionary") is not None:
        _report_evolutionary_multiaxial(log, fat["evolutionary"], fat)
    # M27: the FULLY EVOLUTIONARY MULTIAXIAL JOINT-TENSOR correction, printed
    # ALONGSIDE the M21 stationary, M25 non-stationary and M26 scalar-evolutionary
    # numbers so the listing shows the stationary, the RMS-non-stationary, the
    # scalar-shape-evolutionary and the joint-tensor-evolutionary answers side by
    # side (the point of a JOINT evolutionary tensor over the fixed reduction).
    if fat.get("joint_evolutionary") is not None:
        _report_joint_evolutionary(log, fat["joint_evolutionary"], fat)
    # M31: the CONTINUOUS WIGNER-VILLE INSTANTANEOUS TENSOR block, printed ALONGSIDE
    # the M27 windowed joint-tensor numbers (the continuous per-instant plane
    # re-search next to the windowed one).
    if fat.get("wigner_ville") is not None:
        _report_wigner_ville(log, fat["wigner_ville"])


def _report_joint_evolutionary(log, jv, fat):
    """Print the FULLY EVOLUTIONARY MULTIAXIAL JOINT-TENSOR (M27) listing block:
    the drifting-shape schedule, the critical-plane ROTATION across the windows,
    and, for each reduction (von Mises / max-normal / max-shear), the joint-tensor
    window Miner-sum Dirlik damage rate / life ALONGSIDE the stationary (M21) one,
    plus the per-window critical-plane / F_np / RMS drift and the non-stationary
    MULTIVARIATE Monte-Carlo."""
    log.info("\n     ** FULLY EVOLUTIONARY MULTIAXIAL JOINT-TENSOR FATIGUE **  "
             "(/IMPL/FATIG/MULT/EVOL/JOINT)")
    fc0, fc1 = jv["fc"]
    bw0, bw1 = jv["bw"]
    log.info(f"      DRIFTING TENSOR fc0->fc1 (HZ) . . : "
             f"{fc0:.4G} -> {fc1:.4G}   (bw {bw0:.4G} -> {bw1:.4G})")
    log.info(f"      WINDOWS / g4 / PLANE ROTATION . . : {jv['nwin']}  /  "
             f"g4 = {jv['kurtosis']:.4F}  /  {jv['plane_rotation_deg']:.1F} deg  "
             f"({'constant tensor -> M25/M26' if jv['constant_shape'] else 'joint tensor drifts'})")
    log.info(f"      F_np DRIFT (max - min over windows): "
             f"{jv.get('fnp_drift', 0.0):.4F}  (the non-proportionality "
             f"evolving window to window)")
    log.info("      REDUCTION            STAT DIRLIK RATE   JOINT-EVOL RATE  "
             "JOINT-EVOL LIFE")
    for key, name in (("von_mises", "VON MISES"),
                      ("normal_plane", "MAX-NORMAL PLANE"),
                      ("shear_plane", "MAX-SHEAR PLANE")):
        drS = fat[key]["summary"]["dirlik"]["damage_rate"]
        r = jv["summary"][key]
        life = r["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      {name:18s}{drS:16.5E}   "
                 f"{r['damage_rate']:14.5E}   {life_s:>14s}")
    # the per-window critical-plane DRIFT (the point of a JOINT evolutionary tensor)
    log.info("      PER-WINDOW DRIFT  (fc | max-shear plane normal | F_np | "
             "sigma_vm):")
    wins = jv["summary"]["windows"]
    step = max(1, len(wins) // 6)                 # at most ~6 rows in the listing
    for w in wins[::step]:
        n = w["shear_n"]
        log.info(f"        fc={w['fc']:7.2F}  n=[{n[0]:+.3F} {n[1]:+.3F} "
                 f"{n[2]:+.3F}]  F_np={w['F_np']:.3F}  s_vm={w['sigma_vm']:.4E}")
    if jv.get("monte_carlo") is not None:
        mc = jv["monte_carlo"]
        life = mc["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        tag = "M21-delegated" if mc.get("delegated") else "non-stationary"
        log.info(f"      JOINT MONTE-CARLO (max-shear) . . : "
                 f"{mc['damage_rate']:.5E}  ({tag} multivariate, life {life_s})")


def _report_evolutionary_multiaxial(log, ev, fat):
    """Print the MULTIAXIAL FULLY EVOLUTIONARY / NON-SEPARABLE-PSD (M26) listing
    block: the drifting-shape schedule and, for each reduction (von Mises /
    max-normal / max-shear), the window Miner-sum Dirlik damage rate / life
    ALONGSIDE the stationary one, plus the non-separable Monte-Carlo."""
    log.info("\n     ** FULLY EVOLUTIONARY / NON-SEPARABLE-PSD FATIGUE **  "
             "(/IMPL/FATIG/EVOL)")
    fc0, fc1 = ev["fc"]
    bw0, bw1 = ev["bw"]
    log.info(f"      DRIFTING SHAPE  fc0->fc1 (HZ) . . : "
             f"{fc0:.4G} -> {fc1:.4G}   (bw {bw0:.4G} -> {bw1:.4G})")
    log.info(f"      SPECTROGRAM WINDOWS / g4  . . . . : {ev['nwin']}  /  "
             f"g4 = {ev['kurtosis']:.4F}  "
             f"({'constant shape -> M25' if ev['constant_shape'] else 'non-separable'})")
    log.info("      REDUCTION            STAT DIRLIK RATE   EVOLUTION RATE   "
             "EVOLUTION LIFE")
    for key, name in (("von_mises", "VON MISES"),
                      ("normal_plane", "MAX-NORMAL PLANE"),
                      ("shear_plane", "MAX-SHEAR PLANE")):
        r = ev.get(key)
        if r is None:
            continue
        drS = fat[key]["summary"]["dirlik"]["damage_rate"]
        dk = r["summary"]["dirlik"]
        life = dk["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      {name:18s}{drS:16.5E}   "
                 f"{dk['damage_rate']:14.5E}   {life_s:>14s}")
    if ev.get("monte_carlo") is not None:
        mc = ev["monte_carlo"]
        life = mc["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        tag = "M25-delegated" if mc.get("delegated") else "non-separable"
        log.info(f"      EVOLUTION MONTE-CARLO (von Mises): "
                 f"{mc['damage_rate']:.5E}  ({tag}, sample g4 = "
                 f"{mc['kurtosis']:.3F}, life {life_s})")


def _report_nonstationary_multiaxial(log, ns):
    """Print the MULTIAXIAL NON-STATIONARY / EVOLUTIONARY-PSD (M25) listing block:
    the RMS mission profile, the induced kurtosis, and, for each reduction (von
    Mises / max-normal / max-shear), the amplitude-modulated Dirlik damage rate /
    life ALONGSIDE the stationary one, plus the non-stationary Monte-Carlo."""
    log.info("\n     ** NON-STATIONARY / EVOLUTIONARY-PSD FATIGUE **  "
             "(/IMPL/FATIG/NSTAT)")
    log.info(f"      RMS MODULATION /FUNCT / BLOCKS  . : "
             f"/FUNCT/{ns['modfunct']} / {ns['nblocks']} block(s)")
    log.info(f"      INDUCED KURTOSIS g4  . . . . . . : "
             f"{ns['kurtosis']:.4F}  (3 = stationary Gaussian)")
    br = ns["bridge"]
    log.info(f"      M25<->M24 BRIDGE  kappa_ns / l_ng: "
             f"{br['kappa_ns']:.5F} / {br['lambda_ng']:.5F}  "
             f"(ratio {br['ratio']:.4F})")
    log.info("      REDUCTION            E[a^m]     STAT DIRLIK RATE   "
             "NON-STAT RATE   NON-STAT LIFE")
    for key, name in (("von_mises", "VON MISES"),
                      ("normal_plane", "MAX-NORMAL PLANE"),
                      ("shear_plane", "MAX-SHEAR PLANE")):
        r = ns.get(key)
        if r is None:
            continue
        dk = r["amplitude_modulated"]["dirlik"]
        life = dk["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      {name:18s}{r['e_am']:11.5F}  "
                 f"{dk['stationary_damage_rate']:14.5E}  "
                 f"{dk['damage_rate']:14.5E}   {life_s:>14s}")
    if ns.get("monte_carlo") is not None:
        mc = ns["monte_carlo"]
        life = mc["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      NON-STAT MONTE-CARLO (von Mises) . : "
                 f"{mc['damage_rate']:.5E}  (sample g4 = {mc['kurtosis']:.3F}, "
                 f"life {life_s})")


def _report_nongaussian_multiaxial(log, ng):
    """Print the MULTIAXIAL NON-GAUSSIAN / KURTOSIS (M24) listing block: the
    target kurtosis / skewness and, for each reduction (von Mises / max-normal /
    max-shear), the correction lambda_ng and the CORRECTED Dirlik damage rate /
    life ALONGSIDE the Gaussian one, plus the non-Gaussian Monte-Carlo."""
    log.info("\n     ** NON-GAUSSIAN / KURTOSIS FATIGUE **      "
             "(/IMPL/FATIG/NGAUSS)")
    bw = "ON (Benasciutti-Tovo alpha2)" if ng["bandwidth_correction"] else "OFF"
    log.info(f"      TARGET KURTOSIS g4 / SKEWNESS g3 . : "
             f"{ng['gamma4']:.4F} / {ng['gamma3']:.4F}  "
             f"[bandwidth atten = {bw}]")
    log.info("      REDUCTION            lambda_ng   GAUSS DIRLIK RATE  "
             "NON-GAUSS RATE   NON-GAUSS LIFE")
    for key, name in (("von_mises", "VON MISES"),
                      ("normal_plane", "MAX-NORMAL PLANE"),
                      ("shear_plane", "MAX-SHEAR PLANE")):
        r = ng.get(key)
        if r is None:
            continue
        dk = r["dirlik"]
        life = dk["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      {name:18s}{r['lambda_ng']:11.5F}  "
                 f"{dk['gaussian_damage_rate']:14.5E}  "
                 f"{dk['damage_rate']:14.5E}   {life_s:>14s}")
    if ng.get("monte_carlo") is not None:
        mc = ng["monte_carlo"]
        life = mc["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      NON-GAUSS MONTE-CARLO (shear plane) : "
                 f"{mc['damage_rate']:.5E}  (sample g4 = {mc['kurtosis']:.3F}, "
                 f"life {life_s})")


def _report_nonproportional(log, npres):
    """Print the NON-PROPORTIONAL / CRITICAL-PLANE PATH-COUNTING (M22) listing
    block: the shear-path amplitude comparison (MCC / longest chord / MRH), the
    non-proportionality factor F_np, and, for each critical-plane model (Findley,
    Fatemi-Socie, shear-path), the critical plane normal, sigma_n,max, the shear
    amplitude tau_a and the damage rate / life."""
    amp = npres["amplitudes"]
    log.info("\n     ** NON-PROPORTIONAL MULTIAXIAL FATIGUE **  "
             "(/IMPL/FATIG/MULT/NPROP)")
    log.info(f"      TIME-DOMAIN PATH COUNT (dur / seed)  : "
             f"{npres['duration']:.5E} / {npres['seed']}")
    log.info(f"      SHEAR-PATH AMPLITUDE  MCC / CHORD/2 / MRH (Findley plane): "
             f"{amp['mcc']:.5E} / {amp['chord']:.5E} / {amp['mrh']:.5E}")
    log.info(f"      NON-PROPORTIONALITY  F_np (0 line .. 1 circle) : "
             f"{amp['F_np']:.5F}   [amp op = {npres['amp_method'].upper()}]")
    log.info("      MODEL            CRIT-PLANE NORMAL         sig_n,max   "
             "tau_a       F_np   DAMAGE RATE     LIFE")
    for key, name in (("findley", "FINDLEY 1959"),
                      ("fatemi_socie", "FATEMI-SOCIE 88"),
                      ("shear_path", "SHEAR-PATH")):
        r = npres.get(key)
        if r is None:
            continue
        n = r["normal"]
        life = r["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      {name:15s} [{n[0]:+.3F} {n[1]:+.3F} {n[2]:+.3F}]  "
                 f"{r['sn_max']:+.4E} {r['tau_a']:.4E} {r['F_np']:.3F}  "
                 f"{r['damage_rate']:.5E}  {life_s}")
    # M23: the SPECTRAL non-proportional block, printed ALONGSIDE the M22
    # time-domain numbers so all THREE answers (M21 proportional-spectral, M22
    # non-proportional time-domain, M23 non-proportional spectral) show side by
    # side. The spectral estimate uses the cross-PSD moment matrices directly —
    # NO synthesised history.
    if npres.get("spectral") is not None:
        _report_spectral_nonproportional(log, npres["spectral"])


def _report_spectral_nonproportional(log, sp):
    """Print the SPECTRAL NON-PROPORTIONAL / CRITICAL-PLANE (M23) listing block:
    the frequency-domain non-proportionality factor F_np and the Susmel-Tovo
    stress ratio rho on the critical plane (from the cross-PSD moment matrices,
    no synthesised history), and for each critical-plane model (Findley,
    Fatemi-Socie, shear-path) the critical plane normal, F_np, rho, the effective
    shear amplitude tau_a and the Dirlik damage rate / life."""
    amp = sp["amplitudes"]
    log.info("\n     ** SPECTRAL NON-PROPORTIONAL FATIGUE **   "
             "(/IMPL/FATIG/MULT/NPROP/SPEC)")
    log.info(f"      FREQUENCY-DOMAIN ESTIMATE (no synthesised history; "
             f"psf = {sp['peak_factor']:.4F})")
    log.info(f"      SHEAR RMS  DOMINANT / EFFECTIVE (Findley plane) : "
             f"{amp['shear_rms_dominant']:.5E} / "
             f"{amp['shear_rms_effective']:.5E}")
    log.info(f"      NON-PROPORTIONALITY  F_np / PATH FACTOR g / rho : "
             f"{amp['F_np']:.5F} / {amp['g']:.5F} / {amp['rho']:.5F}")
    log.info("      MODEL            CRIT-PLANE NORMAL         F_np    rho    "
             "tau_a       DAMAGE RATE     LIFE")
    for key, name in (("findley", "FINDLEY 1959"),
                      ("fatemi_socie", "FATEMI-SOCIE 88"),
                      ("shear_path", "SHEAR-PATH")):
        r = sp.get(key)
        if r is None:
            continue
        n = r["normal"]
        life = r["life"]
        life_s = "INF" if not np.isfinite(life) else f"{life:.5E}"
        log.info(f"      {name:15s} [{n[0]:+.3F} {n[1]:+.3F} {n[2]:+.3F}]  "
                 f"{r['F_np']:.3F}  {r['rho']:.3F}  {r['tau_a']:.4E}  "
                 f"{r['damage_rate']:.5E}  {life_s}")
