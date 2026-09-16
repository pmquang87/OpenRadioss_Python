"""
Response-spectrum analysis — M19: the peak response of a structure to a design
RESPONSE SPECTRum, by modal combination (SRSS and CQC).

Fortran origin
--------------
There is NO response-spectrum path anywhere in the open-source OpenRadioss
engine. ``engine/source/input/freimpl.F`` (the /IMPL reader, re-read line by
line for M19) has no /RSPEC, no spectral-ordinate combination, no SRSS/CQC
driver — OpenRadioss is a time-domain crash/impact code and the seismic /
shock design-spectrum envelope analysis is not part of the open-source solver
(the same finding M16-M18 made). So M19 ports response-spectrum analysis as a
clean LIBRARY capability driven by a minimal PORT card (/IMPL/RSPEC — the
response-spectrum analogue of M16's /IMPL/EIGV). The M10 integrator, the M16
REAL eigensolver, the M17 REAL-mode superposition and the M18 COMPLEX-mode
path all stay bit-identical: this is a NEW, parallel path consuming the M16/M17
REAL modes read-only.

Theory — response-spectrum modal combination
---------------------------------------------
(Chopra, "Dynamics of Structures", ch. 13; Der Kiureghian, "A response
spectrum method for random vibration analysis of MDF systems", Earthquake Eng.
& Struct. Dyn. 9, 1981 — the CQC rule and its closed-form correlation
coefficient; Wilson, Der Kiureghian & Bayo 1981.)

A DESIGN RESPONSE SPECTRUM Sa(omega, zeta) gives the PEAK pseudo-acceleration
of a single-DOF oscillator of frequency omega and damping zeta subjected to a
prescribed ground motion — a smoothed envelope over an ensemble of records, the
standard seismic / shock design input. For a multi-DOF structure the response
is expanded in the undamped REAL modes (the M16/M17 basis): mode i has natural
frequency omega_i, mass-normalized shape phi_i and, for a support acceleration
in direction d, the PARTICIPATION factor

    Gamma_i = phi_i^T M iota_d                                            (1)

(iota_d the rigid-body influence vector of direction d — the same
``_rigid_body_vectors`` the M16 effective mass uses; note the effective mass
Gamma_i^2 loses the SIGN, so Gamma_i is recomputed here with its sign). The
PEAK modal displacement of mode i is the spectral DISPLACEMENT
Sd(omega_i) = Sa(omega_i)/omega_i^2, so the peak physical-response
contribution of mode i (the "modal peak") is

    r_i = Gamma_i * Sa(omega_i, zeta_i) / omega_i^2 * phi_i              (2)

the participation-scaled spectral ordinate distributed over the DOFs by the
mode shape (Chopra eq. 13.7.1-3). Because the modal peaks do NOT occur at the
same instant, they cannot simply be added; they are combined statistically.

SRSS (Square-Root-of-Sum-of-Squares; Rosenblueth):

    u_j^peak = sqrt( sum_i r_{i,j}^2 )                                    (3)

— correct when the modes are WELL SEPARATED (their peak responses are nearly
uncorrelated). SRSS ERRS for CLOSELY-SPACED modes (a symmetric structure's
degenerate pair, a tuned system): the modal peaks are correlated and the
cross-terms matter.

CQC (Complete Quadratic Combination; Der Kiureghian 1981):

    u_j^peak = sqrt( sum_i sum_k rho_{ik} r_{i,j} r_{k,j} )              (4)

with the closed-form modal CORRELATION coefficient (for the white-noise input
model underlying a design spectrum), for modes of damping zeta_i, zeta_k and
frequency ratio r = omega_k/omega_i,

              8 sqrt(zeta_i zeta_k) (zeta_i + r zeta_k) r^{3/2}
    rho_ik = --------------------------------------------------------     (5)
             (1-r^2)^2 + 4 zeta_i zeta_k r (1+r^2) + 4(zeta_i^2+zeta_k^2) r^2

(Der Kiureghian 1981 eq. 24; the general unequal-damping form. For equal
damping zeta it reduces to the familiar
rho = 8 zeta^2 (1+r) r^{3/2} / [(1-r^2)^2 + 4 zeta^2 r (1+r)^2].) rho_ii = 1,
rho_ik -> 0 for well-separated modes (so CQC -> SRSS there) and rho_ik -> 1 for
r -> 1 (fully-correlated degenerate modes, where CQC's cross-terms are the
whole story and SRSS is badly wrong — the CQC analogue of M18's classical-vs-
non-classical gap). CQC is the default; SRSS is reported alongside so the gap
is visible.

Deliberate deviations / deferrals (documented, not hidden)
----------------------------------------------------------
* LIBRARY-FIRST card (/IMPL/RSPEC) — no upstream equivalent, exactly as
  M16-M18 established.
* REAL modes + classical modal damping (the M16/M17 basis). A design spectrum
  is defined per damping ratio; per-mode zeta is accepted (uniform in the
  card). Complex-mode response-spectrum combination is not standard practice
  and is not ported.
* SINGLE excitation direction per analysis (one iota_d). Multi-directional
  combination (the 100-30-30 / SRSS-of-directions rules) is a straightforward
  post-combination of per-direction runs and is DEFERRED (PORTING_GUIDE M19).
* Sign of the CQC/SRSS result: the combination gives a POSITIVE magnitude
  envelope per DOF (the peak is sign-indeterminate — the standard response-
  spectrum output). A dominant-mode sign convention is not applied.
"""

from __future__ import annotations

import numpy as np

from . import require_scipy
from .modal import _rigid_body_vectors


# ============================================================================
# The CQC modal-correlation coefficient (theory eq. (5))
# ============================================================================

def cqc_correlation(omega, zeta):
    """The Der Kiureghian (1981) CQC correlation matrix rho_ik for modes of
    angular frequency ``omega`` (nmode,) and damping ``zeta`` (nmode, or a
    scalar broadcast). Symmetric, unit diagonal, entries in [0, 1]; -> 0 for
    well-separated modes and -> 1 for r = omega_k/omega_i -> 1 (theory eq.
    (5), the general unequal-damping form)."""
    omega = np.asarray(omega, dtype=float)
    z = np.broadcast_to(np.asarray(zeta, dtype=float), omega.shape)
    wi = np.maximum(omega[:, None], 1e-12)
    wk = np.maximum(omega[None, :], 1e-12)
    zi = z[:, None]
    zk = z[None, :]
    r = wk / wi                                       # omega_k / omega_i
    num = 8.0 * np.sqrt(zi * zk) * (zi + r * zk) * r ** 1.5
    den = ((1.0 - r ** 2) ** 2 + 4.0 * zi * zk * r * (1.0 + r ** 2)
           + 4.0 * (zi ** 2 + zk ** 2) * r ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(den > 1e-14, num / den, np.where(np.abs(r - 1.0) < 1e-6, 1.0, 0.0))
    # numerical guard: the diagonal is exactly 1 (r = 1, zi = zk analytically),
    # clip round-off outside [0, 1]
    rho = np.clip(rho, 0.0, 1.0)
    np.fill_diagonal(rho, 1.0)
    return rho


# ============================================================================
# The participation-scaled modal peaks + SRSS / CQC combination
# ============================================================================

def modal_peaks(basis, design_spectrum, direction, zeta):
    """The per-mode peak physical-response contributions r_{i,j} = Gamma_i *
    Sa(omega_i)/omega_i^2 * phi_{i,j} (theory eq. (2)), returned as
    ``(peaks, gamma, Sa)``:

    * ``peaks`` (ndof, nmode) — column i is the signed modal peak field r_i;
    * ``gamma`` (nmode,)      — the signed participation factors Gamma_i (1);
    * ``Sa``    (nmode,)      — the spectral ordinate at each modal frequency.

    ``basis`` is an M17 ``ModalBasis`` (real modes, consistent mass);
    ``design_spectrum`` a /FUNCT giving Sa as a function of frequency (Hz);
    ``direction`` a 0..5 rigid direction index; ``zeta`` the per-mode damping
    (scalar or (nmode,)) at which the spectrum is defined."""
    omega = basis.omega
    Phi = basis.Phi
    d = int(direction)
    R = _rigid_body_vectors(basis.dof.model, basis.dof)
    # signed participation Gamma_i = phi_i^T M iota_d (eq. (1)); Phi is
    # M-orthonormal in eq space, M the consistent mass
    gamma = Phi.T @ (basis.M @ R[:, d])
    # spectral ordinate at each modal frequency (Hz); spectral displacement
    # Sd = Sa / omega^2
    Sa = np.asarray(design_spectrum.eval(basis.freqs), dtype=float)
    Sd = np.where(omega > 1e-12, Sa / (omega * omega), 0.0)
    # modal peak field: column i = Gamma_i * Sd_i * phi_i (eq. (2))
    peaks = Phi * (gamma * Sd)[None, :]
    return peaks, gamma, Sa


def srss_combination(peaks):
    """SRSS peak u_j = sqrt(sum_i r_{i,j}^2) (theory eq. (3)), per equation DOF.
    ``peaks`` is the (ndof, nmode) modal-peak array of ``modal_peaks``."""
    return np.sqrt(np.sum(np.asarray(peaks) ** 2, axis=1))


def cqc_combination(peaks, omega, zeta):
    """CQC peak u_j = sqrt(sum_i sum_k rho_ik r_{i,j} r_{k,j}) (theory eq. (4)),
    per equation DOF, with the Der Kiureghian correlation ``cqc_correlation``.
    ``peaks`` is (ndof, nmode)."""
    peaks = np.asarray(peaks)
    rho = cqc_correlation(omega, zeta)
    # quadratic form per DOF: sum_ik rho_ik r_i r_k = einsum over the modes
    val = np.einsum("ji,ik,jk->j", peaks, rho, peaks)
    return np.sqrt(np.clip(val, 0.0, None))


def response_spectrum_analysis(basis, design_spectrum, direction, zeta):
    """Full response-spectrum modal combination: the per-mode peaks, then BOTH
    SRSS and CQC envelopes (theory). Returns a dict:

        peaks   (ndof, nmode)  the signed modal-peak fields r_i
        gamma   (nmode,)       signed participation factors
        Sa      (nmode,)       spectral ordinates at the modal frequencies
        rho     (nmode, nmode) CQC correlation matrix
        srss    (ndof,)        SRSS peak (eq. (3))
        cqc     (ndof,)        CQC peak (eq. (4))
    """
    peaks, gamma, Sa = modal_peaks(basis, design_spectrum, direction, zeta)
    rho = cqc_correlation(basis.omega, zeta)
    srss = srss_combination(peaks)
    cqc = cqc_combination(peaks, basis.omega, zeta)
    return {"peaks": peaks, "gamma": gamma, "Sa": Sa, "rho": rho,
            "srss": srss, "cqc": cqc}


# ============================================================================
# Engine-card driver (/IMPL/RSPEC)
# ============================================================================

def _lookup_spectrum(model, funct_id):
    """Resolve the design-spectrum /FUNCT table by user id, with a clear
    error (the same ``model.functions`` map every load references)."""
    if funct_id is None or int(funct_id) <= 0:
        raise ValueError(
            "/IMPL/RSPEC needs a design-spectrum function id (the /FUNCT "
            "giving Sa(f)); none was supplied on the card.")
    fid = int(funct_id)
    if fid not in model.functions:
        raise ValueError(
            f"/IMPL/RSPEC design spectrum /FUNCT/{fid} is not defined in the "
            f"deck.")
    return model.functions[fid]


def run_response_spectrum(model, ip, log, result, constr=None, contacts=(),
                          loads=None):
    """/IMPL/RSPEC (M19): design-response-spectrum modal combination.

    Extract the M17 real modes, read the design spectrum /FUNCT, form the
    participation-scaled modal peaks and combine them by SRSS and CQC —
    storing the peak fields on ``result`` (mirroring how ``_run_modal`` stores
    the eigenpairs). A PORT card: the open-source engine has no
    response-spectrum path (module docstring)."""
    require_scipy()
    from .modal_response import build_modal_basis

    nev = max(1, int(getattr(ip, "impl_rspec_nmode", 6)))
    prestress = bool(getattr(ip, "impl_rspec_prestress", False))
    direction = int(getattr(ip, "impl_rspec_dir", 0))
    zeta_u = float(getattr(ip, "impl_rspec_zeta", 0.05))
    funct_id = getattr(ip, "impl_rspec_funct", 0)

    log.info("\n     ** RESPONSE SPECTRUM (modal combination) **  (/IMPL/RSPEC)")
    if prestress:
        log.info("        (prestressed modes: K = K_mat + K_geo of the "
                 "committed state)")

    spectrum = _lookup_spectrum(model, funct_id)

    x_saved = model.x
    if not prestress:
        model.x = model.x0.copy()
    try:
        basis = build_modal_basis(model, nev=nev, log=None, constraints=constr,
                                  contacts=contacts, prestress=prestress)
    finally:
        if not prestress:
            model.x = x_saved

    zeta = float(zeta_u)
    res = response_spectrum_analysis(basis, spectrum, direction, zeta)

    # scatter the combined peak envelopes to nodes
    du_srss, dur_srss = basis.dof.scatter_solution(res["srss"])
    du_cqc, dur_cqc = basis.dof.scatter_solution(res["cqc"])
    result.response_spectrum = {
        "freqs": basis.freqs, "gamma": res["gamma"], "Sa": res["Sa"],
        "rho": res["rho"], "srss": res["srss"], "cqc": res["cqc"],
        "srss_nodal": du_srss, "srss_rot_nodal": dur_srss,
        "cqc_nodal": du_cqc, "cqc_rot_nodal": dur_cqc,
        "direction": direction, "zeta": zeta,
    }

    log.info(f"      MODES RETAINED . . . . . . . . . : {basis.nmode}")
    log.info(f"      EXCITATION DIRECTION . . . . . . : {direction}")
    log.info(f"      DESIGN SPECTRUM /FUNCT / DAMPING : "
             f"/FUNCT/{int(funct_id)} , zeta = {zeta:.4E}")
    log.info("      MODE   FREQ(HZ)     PARTIC.       Sa            "
             "MODAL PEAK(|Gamma Sa/w^2|)")
    for i in range(basis.nmode):
        wi = basis.omega[i]
        peak = abs(res['gamma'][i] * res['Sa'][i] / (wi * wi)) if wi > 1e-12 else 0.0
        log.info(f"      {i + 1:4d}  {basis.freqs[i]:12.5E} "
                 f"{res['gamma'][i]:12.4E} {res['Sa'][i]:12.5E}  "
                 f"{peak:12.5E}")
    js = int(np.argmax(res["srss"]))
    jc = int(np.argmax(res["cqc"]))
    log.info(f"      PEAK RESPONSE  SRSS / CQC  . . . : "
             f"{res['srss'][js]:.5E} / {res['cqc'][jc]:.5E}")
    # the SRSS/CQC gap is the closely-spaced-mode signature
    ref = max(float(np.max(res["srss"])), 1e-30)
    gap = float(np.max(np.abs(res["cqc"] - res["srss"]))) / ref * 100.0
    log.info(f"      MAX |CQC - SRSS| / max SRSS  . . : {gap:8.2f} %  "
             f"(closely-spaced-mode correlation)")
