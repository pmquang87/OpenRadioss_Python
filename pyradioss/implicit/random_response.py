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

import numpy as np

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
        out[n] = np.trapezoid(integrand, w, axis=0) / np.pi
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
    if mult:
        nplane = int(getattr(ip, "impl_fatig_nplane", 24))
        _run_multiaxial(model, ip, log, result, frf, Sigma, channels,
                        psd_tab, m_sn, C_sn, mean_stress, ultimate, mc_dur,
                        mc_seed, base, base_dir, funct_id, nev, nplane)
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

    result.fatigue = {
        "channels": channels, "critical_channel": jcrit,
        "critical_label": channels[jcrit][3], "moments": mom,
        "rms_stress": rms_stress, "dirlik_rate": dirlik_rate,
        "freqs": sp["freqs"], "omega": sp["omega"], "Ssigma": sp["Ssigma"],
        "Sff": sp["Sff"], "summary": summary, "monte_carlo": mc,
        "nongaussian": nongaussian, "nonstationary": nonstationary,
        "evolutionary": evolutionary,
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

    result.fatigue = {
        "multiaxial": True, "nonproportional": nprop,
        "spectral_nonproportional": spec_np, "nongaussian": nongaussian,
        "nonstationary": nonstationary, "evolutionary": evolutionary,
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
