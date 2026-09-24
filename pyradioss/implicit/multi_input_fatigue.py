"""
Multi-input random fatigue Monte-Carlo cross-check — M28: the MULTI-INPUT
multivariate spectral-representation Monte-Carlo that INDEPENDENTLY validates
the multi-input stress-tensor cross-PSD (M28 ``multi_input_response``) against
a time-domain rainflow + Miner count driven by SYNTHESISED correlated input
histories.

Where the M21 multiaxial Monte-Carlo (``multiaxial_fatigue.
monte_carlo_multiaxial_damage``) synthesises the 6 correlated STRESS components
directly from the stress-tensor cross-PSD S_sigmasigma, the M28 cross-check
synthesises the ninput correlated INPUT processes from the INPUT cross-PSD
S_ff (the SAME per-bin eigen/Cholesky factor, applied to the input matrix
instead of the stress matrix — Shinozuka & Deodatis 1996, the multivariate
spectral-representation method; Wirsching, Paez & Ortiz 1995 for the
multi-input random-fatigue application), drives each synthesised input through
its OWN Voigt stress FRF column, SUMS the stress contributions and
rainflow-counts (ASTM E1049, the M20 counter) + Miner-sums the projected
critical-plane scalar. The two formulations must agree within scatter — the
independent confirmation that H S_ff H^H captures the multi-input coherent
response.

Fortran origin
--------------
A PORT library, no upstream equivalent — the open-source OpenRadioss engine has
no random-vibration / spectral / multi-input machinery of any kind (module
docstring of ``multi_input_response``; re-confirmed for M28). Built ALONGSIDE
the M21 multivariate synthesiser (which stays byte-identical); the single-input
and fully-coherent cases DELEGATE to the M21 path so the reduction is
bit-identical.

Theory — multivariate spectral representation of correlated inputs
------------------------------------------------------------------
For the Hermitian input cross-PSD S_ff(Omega) (ninput x ninput), each
positive-frequency bin's matrix is factored S = V diag(lam) V^H (a batched
Hermitian eigendecomposition — a general Cholesky-type factor L = V sqrt(lam)
with L L^H = S that handles the RANK-1 fully-coherent case gracefully, where a
plain Cholesky would fail on the semi-definite matrix). Each nonzero
eigen-direction gets an INDEPENDENT uniform random phase; one inverse real FFT
per input synthesises the correlated input histories f_a(t) whose measured
cross-PSD reproduces S_ff. Driving each through its stress FRF column and
summing gives the stress history whose statistics the M28 spectral answer
predicts (Newland ch. 6-8; Bendat & Piersol ch. 5-7).

Two limiting delegations (for the bit-identical reductions):
* ninput = 1 (a single scalar input) -> delegate to the M21 single-input
  Monte-Carlo on the rank-1 stress-tensor cross-PSD (bit-identical);
* a RANK-1 fully-coherent S_ff (one common source) -> the effective single
  input; delegate to the M21 Monte-Carlo on the effective stress-tensor
  cross-PSD H S_ff H^H (bit-identical to the M21 single-input answer for the
  combined pattern).
Otherwise (partially coherent / incoherent inputs) the input-level synthesis
runs and is cross-checked against the stress-level (M21-on-S_sigmasigma) answer
and the per-input root-sum-of-squares within the documented seeded scatter.

See PORTING_GUIDE.md roadmap M28.
"""

from __future__ import annotations

import numpy as np

from .multi_input_response import (stress_tensor_cross_psd_multi,
                                    matrix_rank_psd)


# ============================================================================
# Multivariate spectral-representation synthesis of the INPUT histories
# ============================================================================

def _one_sided_grid(freqs_hz, Smat, duration, fs):
    """Interpolate a Hermitian cross-PSD stack ``Smat`` (nf, n, n) onto the FFT
    grid and form the ONE-SIDED stack G(f) = 2 S(2 pi f) (zero outside the
    band). Shared helper — identical convention to the M20/M21 synthesisers.
    Returns ``(fft_f, Gstack, nt, df)``."""
    f = np.asarray(freqs_hz, dtype=float)
    S = np.asarray(Smat)
    n = S.shape[1]
    fmax = float(f.max()) if len(f) > 0 else 0.0
    if fs is None:
        fs = max(8.0 * fmax, 1.0)
    nt = int(max(round(duration * fs), 4))
    if nt % 2:
        nt += 1
    df = fs / nt
    fft_f = np.fft.rfftfreq(nt, d=1.0 / fs)
    nbin = fft_f.size
    G = np.zeros((nbin, n, n), dtype=complex)
    for a in range(n):
        for b in range(n):
            re = np.interp(fft_f, f, S[:, a, b].real, left=0.0, right=0.0)
            im = np.interp(fft_f, f, S[:, a, b].imag, left=0.0, right=0.0)
            G[:, a, b] = 2.0 * (re + 1j * im)
    return fft_f, G, nt, df


def synthesize_input_spectra(freqs_hz, Sff, duration, seed, fs=None):
    """Synthesise the frequency-domain spectra of ninput CORRELATED Gaussian
    input processes whose cross-PSD is ``Sff`` (nf, ninput, ninput), by the
    MULTIVARIATE spectral-representation method (the M21 per-bin
    eigendecomposition applied to the INPUT matrix). Returns
    ``(fft_f, Finput, nt, fs)`` with ``Finput`` of shape (nbin, ninput) the
    complex input spectra (DC and Nyquist handled by the caller's irfft)."""
    if fs is None:
        fmax = float(np.asarray(freqs_hz).max()) if len(freqs_hz) > 0 else 0.0
        fs = max(8.0 * fmax, 1.0)
    fft_f, G, nt, df = _one_sided_grid(freqs_hz, Sff, duration, fs)
    nbin, n = fft_f.size, G.shape[1]
    lam, V = np.linalg.eigh(G)                    # lam (nbin,n), V (nbin,n,n)
    lam = np.clip(lam.real, 0.0, None)
    amp = nt * np.sqrt(lam * df / 2.0)            # (nbin, n) per eigen-direction
    rng = np.random.default_rng(int(seed))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(nbin, n))
    coeff = amp * np.exp(1j * phase)
    Finput = np.einsum("kab,kb->ka", V, coeff)   # (nbin, n) input spectra
    Finput[0] = 0.0                              # no DC
    return fft_f, Finput, nt, fs


def synthesize_multi_input_forces(freqs_hz, Sff, duration, seed, fs=None):
    """Synthesise the TIME histories of ninput correlated Gaussian input
    processes from ``Sff`` (nf, ninput, ninput). Returns ``(t, F)`` with
    ``F`` (nt, ninput). Used for the measured-coherence diagnostic (the
    synthesised inputs' coherence must match the target gamma_ab)."""
    fft_f, Finput, nt, fs = synthesize_input_spectra(freqs_hz, Sff, duration,
                                                     seed, fs=fs)
    if nt % 2 == 0:
        Finput[-1] = Finput[-1].real             # Nyquist bin real
    F = np.fft.irfft(Finput, n=nt, axis=0)        # (nt, ninput)
    t = np.arange(nt) / fs
    return t, F


def synthesize_multi_input_stress(freqs_hz, Sff, Hcols, duration, seed,
                                  fs=None):
    """Synthesise ONE element's 6 correlated stress-component histories by the
    MULTI-INPUT path: synthesise the ninput correlated INPUT spectra from
    ``Sff`` (nf, ninput, ninput), drive each through the element's Voigt stress
    FRF column ``Hcols`` (nf, 6, ninput) in the frequency domain, SUM the
    contributions and inverse-FFT. Returns ``(t, X)`` with ``X`` (nt, 6).

    The stress spectrum of component c is
    Xspec[c] = sum_a H[c,a](f) Finput[a](f), so the synthesised stress carries
    exactly the multi-input coherent cross-spectrum H S_ff H^H (whose
    time-domain statistics this cross-checks)."""
    fft_f, Finput, nt, fs = synthesize_input_spectra(freqs_hz, Sff, duration,
                                                     seed, fs=fs)
    H = np.asarray(Hcols)
    f = np.asarray(freqs_hz, dtype=float)
    # interpolate the 6xN stress FRF columns onto the FFT grid (zero outside)
    nbin = fft_f.size
    Hgrid = np.zeros((nbin, 6, H.shape[2]), dtype=complex)
    for c in range(6):
        for a in range(H.shape[2]):
            re = np.interp(fft_f, f, H[:, c, a].real, left=0.0, right=0.0)
            im = np.interp(fft_f, f, H[:, c, a].imag, left=0.0, right=0.0)
            Hgrid[:, c, a] = re + 1j * im
    Xspec = np.einsum("kca,ka->kc", Hgrid, Finput)   # (nbin, 6) stress spectra
    Xspec[0] = 0.0
    if nt % 2 == 0:
        Xspec[-1] = Xspec[-1].real
    X = np.fft.irfft(Xspec, n=nt, axis=0)             # (nt, 6)
    t = np.arange(nt) / fs
    return t, X


# ============================================================================
# The multi-input damage Monte-Carlo (input-level cross-check + delegation)
# ============================================================================

def monte_carlo_multi_input_damage(freqs_hz, Sff, Hcols, proj, m, C, duration,
                                   seed, fs=None, mean_stress=0.0, ultimate=0.0,
                                   force_input_level=False):
    """The TIME-DOMAIN multi-input damage rate by Monte-Carlo on a critical
    plane's LINEAR projection ``proj`` (a normal_/shear_projection 6-vector).

    Default behaviour DELEGATES to the M21 single-input Monte-Carlo
    (``multiaxial_fatigue.monte_carlo_multiaxial_damage``) on the multi-input
    stress-tensor cross-PSD S_sigmasigma = H S_ff H^H whenever the input matrix
    is effectively RANK-1 (ninput = 1 or a fully-coherent common source) — so
    the single-input and fully-coherent reductions are BIT-IDENTICAL to the M21
    answer. Otherwise (partially coherent / incoherent inputs) it runs the
    INPUT-LEVEL synthesis: synthesise the ninput correlated inputs, drive them
    through the stress FRF columns, sum, rainflow (ASTM E1049) and Miner-sum.

    ``force_input_level`` forces the input-level synthesis even for a rank-1
    matrix (used to VERIFY the input-level and stress-level answers agree).
    Returns the M21 Monte-Carlo dict shape plus ``rms`` of the projected
    scalar and a ``path`` tag ('delegated' or 'input_level')."""
    from . import spectral_fatigue as sf
    from . import multiaxial_fatigue as mf

    Sff = np.asarray(Sff)
    Hcols = np.asarray(Hcols)
    proj = np.asarray(proj, dtype=float)
    ninput = Sff.shape[1]

    # the multi-input stress-tensor cross-PSD (the object the M21 estimators and
    # the M21 stress-level Monte-Carlo consume)
    Scross = stress_tensor_cross_psd_multi(Hcols, Sff)

    rank1 = (ninput == 1) or bool(np.all(matrix_rank_psd(Sff) <= 1))
    if rank1 and not force_input_level:
        # delegate to the M21 single-input Monte-Carlo on the effective
        # stress-tensor cross-PSD (bit-identical to the M21 answer)
        out = mf.monte_carlo_multiaxial_damage(
            freqs_hz, Scross, proj, m, C, duration, seed, fs=fs,
            mean_stress=mean_stress, ultimate=ultimate)
        out["path"] = "delegated"
        return out

    # input-level synthesis: correlated inputs -> stress -> project -> rainflow
    t, X = synthesize_multi_input_stress(freqs_hz, Sff, Hcols, duration, seed,
                                         fs=fs)
    s = X @ proj
    Ceff = sf._goodman_C(C, m, mean_stress, ultimate)
    ranges, counts = sf.rainflow_count(s)
    D = float(np.sum(counts * ranges ** m) / Ceff) if ranges.size else 0.0
    T = t[-1] - t[0] if t.size > 1 else duration
    dr = D / T if T > 0 else 0.0
    tf, s_eq = sf.life_and_equivalent(
        dr, ranges.size / T if T > 0 else 0.0, m, Ceff)
    return {"method": "monte_carlo_multi_input", "path": "input_level",
            "damage_rate": dr, "life": tf, "s_eq": s_eq,
            "ncycles": float(counts.sum()), "ranges": ranges, "counts": counts,
            "duration": T, "rms": float(np.std(s))}


# ============================================================================
# Measured coherence diagnostic (validation: synthesised inputs match target)
# ============================================================================

def measure_coherence(F, fs, nperseg=None, noverlap=None):
    """The measured ordinary-coherence matrix gamma_ab^2(f) of the synthesised
    input histories ``F`` (nt, ninput) at sample rate ``fs``, by Welch-style
    segment averaging (a periodogram averaged over ``nperseg``-length,
    50%-overlapped Hann-windowed segments — the standard coherence estimator,
    Bendat & Piersol ch. 9). Returns ``(f, coh)`` with ``coh`` (nseg_f, n, n)
    the coherence in [0, 1] (diagonal 1). The AVERAGING is essential: a single
    periodogram gives coherence identically 1 (the classic estimator bias),
    so several segments are needed to recover the true gamma_ab."""
    F = np.asarray(F, dtype=float)
    nt, n = F.shape
    if nperseg is None:
        nperseg = max(64, nt // 16)
    nperseg = int(min(nperseg, nt))
    if noverlap is None:
        noverlap = nperseg // 2
    step = max(1, nperseg - int(noverlap))
    win = np.hanning(nperseg)
    starts = list(range(0, nt - nperseg + 1, step))
    if not starts:
        starts = [0]
    nf = nperseg // 2 + 1
    Pxy = np.zeros((nf, n, n), dtype=complex)
    for s0 in starts:
        seg = F[s0:s0 + nperseg] * win[:, None]
        Sp = np.fft.rfft(seg, axis=0)                 # (nf, n)
        Pxy += np.einsum("fa,fb->fab", Sp, np.conj(Sp))
    Pxy /= len(starts)
    diag = np.einsum("faa->fa", Pxy).real
    denom = np.sqrt(np.clip(diag[:, :, None] * diag[:, None, :], 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        coh = np.where(denom > 0.0, (np.abs(Pxy) ** 2) / (denom ** 2), 0.0)
    f = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return f, np.clip(coh, 0.0, 1.0)
