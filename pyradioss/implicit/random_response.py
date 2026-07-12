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

    log.info("\n     ** RANDOM-VIBRATION (SPECTRAL) FATIGUE **    (/IMPL/FATIG)")
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

    result.fatigue = {
        "channels": channels, "critical_channel": jcrit,
        "critical_label": channels[jcrit][3], "moments": mom,
        "rms_stress": rms_stress, "dirlik_rate": dirlik_rate,
        "freqs": sp["freqs"], "omega": sp["omega"], "Ssigma": sp["Ssigma"],
        "Sff": sp["Sff"], "summary": summary, "monte_carlo": mc,
        "sn_m": m_sn, "sn_C": C_sn, "mean_stress": mean_stress,
        "ultimate": ultimate, "base": base, "stress_modes": Sigma,
    }

    _report_fatigue(log, result.fatigue, funct_id, base, base_dir, frf, nev)


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
