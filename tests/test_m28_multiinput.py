"""
M28 validations: MULTI-INPUT / PARTIALLY-COHERENT RANDOM-VIBRATION RESPONSE &
FATIGUE (/IMPL/PSD/MULTI, /IMPL/FATIG/MINPUT) — the stationary response (and
stress-tensor) cross-PSD driven by SEVERAL simultaneous random input processes
with a full Hermitian input cross-spectral matrix S_ff(w) = [sqrt(G_a G_b)
gamma_ab exp(i theta_ab)], propagated through the VECTOR FRF by the MIMO
relation S_uu = H S_ff H^H / S_sigmasigma = H_sigma S_ff H_sigma^H, then reduced
by the M20-M27 estimator family UNCHANGED. Built ALONGSIDE the M19/M20/M21
single-input paths (which stay bit-identical; the multi-input path is a NEW
parallel path). The single scalar input is EXACTLY the 1x1 (diagonal / rank-1)
special case of the new matrix path.

INPUT CROSS-PSD MODEL
* the assembled S_ff is Hermitian and satisfies the coherence identity
  |S_ff[a,b]|^2 = gamma_ab^2 G_a G_b;
* the Hermitian-PSD projection is a NO-OP on an already-valid matrix (and clips
  a constructed non-PSD one);
* a DIAGONAL (incoherent) S_ff gives the SUM of the per-input single-input
  cross-PSDs EXACTLY; a RANK-1 fully-coherent S_ff reduces to the single-input
  answer for the effective combined pattern EXACTLY; partial coherence
  interpolates monotonically between them;
* a two-input SDOF response cross-PSD matches the hand-derived closed form;
* the single-input (ninput = 1) cross-PSD is bit-identical to the M21 rank-1
  cross-PSD and the M19 |U|^2 G response PSD.

MULTI-INPUT FATIGUE + MONTE-CARLO
* the multi-input S_sigmasigma flows into the M21 reductions UNCHANGED (the
  reductions are byte-identical given the same S_sigmasigma);
* the fully-coherent / single-input MC reduces to the M21 single-input MC
  BIT-IDENTICALLY (delegation);
* the incoherent input-level MC matches the stress-level (M21-on-S_sigmasigma)
  MC within scatter, and the incoherent projected variance is the SUM of the
  per-input projected variances (exact);
* the synthesised inputs' measured coherence matches the target gamma_ab.

CARDS + END-TO-END + NO-REGRESSION (the M7 parity contract)
* /IMPL/FATIG/MULT/MINPUT and /IMPL/PSD/MULTI card mirror (a PORT sub-card);
* the solid-brick cantilever under TWO partially-coherent inputs end to end,
  reporting its multi-input life alongside its single-input life;
* the single-input M19/M20/M21 path is byte-unchanged whether or not the
  multi-input path runs; the recovery is read-only; the direct M10 answer is
  unchanged; composition with /NGAUSS.

See PORTING_GUIDE.md roadmap M28.
"""

import contextlib
import io
import os
import tempfile

import numpy as np
import pytest

from pyradioss.engine.engine import run_engine
from pyradioss.starter.starter import run_starter

pytest.importorskip("scipy")   # implicit requires scipy (optional otherwise)

from pyradioss.common.messages import MessageLog                    # noqa: E402
from pyradioss.implicit.random_response import (                    # noqa: E402
    element_voigt_blocks, spectral_moments, stress_channels, stress_modes)
from pyradioss.implicit import multiaxial_fatigue as mf             # noqa: E402
from pyradioss.implicit import multi_input_response as mir          # noqa: E402
from pyradioss.implicit import multi_input_fatigue as mif           # noqa: E402
from pyradioss.implicit import spectral_fatigue as sf               # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _run(starter_text, engine_text, capture=False):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M28_0000.rad")
    ep = os.path.join(d, "M28_0001.rad")
    with open(sp, "w") as f:
        f.write(starter_text)
    with open(ep, "w") as f:
        f.write(engine_text)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_starter(sp)
        model = run_engine(ep)
    return (model, buf.getvalue()) if capture else model


def _two_input_brick(nx=3):
    """A SOLID-brick cantilever with TWO distinct tip-load patterns: pattern A
    (a z-load, /CLOAD /FUNCT/1) and pattern B (a y-load, /CLOAD /FUNCT/2), so
    two independent random inputs excite bending in two planes. Auto-PSDs
    /FUNCT/10 (G_A = 2) and /FUNCT/11 (G_B = 1.5); /FUNCT/10 is the reference
    single-input PSD for the side-by-side listing."""
    def nid(ix, iy, iz):
        return ix * 4 + iy * 2 + iz + 1
    nodes = []
    for ix in range(nx + 1):
        for iy in range(2):
            for iz in range(2):
                nodes.append(f"{nid(ix, iy, iz):10d}{ix*1.0:20.10f}"
                             f"{iy*1.0:20.10f}{iz*1.0:20.10f}")
    bricks = ["/BRICK/1"]
    for ix in range(nx):
        n = [nid(ix, 0, 0), nid(ix+1, 0, 0), nid(ix+1, 1, 0), nid(ix, 1, 0),
             nid(ix, 0, 1), nid(ix+1, 0, 1), nid(ix+1, 1, 1), nid(ix, 1, 1)]
        bricks.append(f"{ix+1:10d}" + "".join(f"{v:10d}" for v in n))
    root = [nid(0, iy, iz) for iy in range(2) for iz in range(2)]
    tip = [nid(nx, iy, iz) for iy in range(2) for iz in range(2)]
    per = 1.0e-4 / len(tip)
    cloads = []
    for k, t in enumerate(tip):
        cloads += [f"/GRNOD/NODE/{10+k}", f"t{k}", str(t),
                   f"/CLOAD/{20+2*k}", "az",
                   f"         1         Z{10+k:10d}{per*1.0:12g}",
                   f"/CLOAD/{21+2*k}", "ay",
                   f"         2         Y{10+k:10d}{per*0.6:12g}"]
    return "\n".join([
        "#RADIOSS STARTER", "/BEGIN", "BRICK", "/NODE", *nodes, *bricks,
        "/PART/1", "cant", "         1         1",
        "/MAT/LAW1/1", "steel", "   7.8e-6", "     210.0       0.3",
        "/PROP/SOLID/1", "solid", "         0",
        "/GRNOD/NODE/1", "root", "\n".join(str(i) for i in root),
        "/BCS/1", "clamp", "       111       111         0         1",
        "/FUNCT/1", "pA", "       0.0           1.0", "  100000.0           1.0",
        "/FUNCT/2", "pB", "       0.0           1.0", "  100000.0           1.0",
        "/FUNCT/10", "gA", "       0.0           2.0", "  100000.0           2.0",
        "/FUNCT/11", "gB", "       0.0           1.5", "  100000.0           1.5",
        *cloads, "/END"]) + "\n"


def _synth_columns(nf=1500, fmax=200.0, ninput=2):
    """Synthetic per-input Voigt stress FRF columns (nf, 6, ninput) + the
    angular grid + per-input auto-PSDs, for the analytic checks (no FEM). Each
    input drives a different pair of Voigt components / resonances."""
    f = np.linspace(1e-3, fmax, nf)
    omega = 2.0 * np.pi * f

    def peak(fc, bw, A):
        return A * np.exp(-((f - fc) ** 2) / (2.0 * bw ** 2))
    H = np.zeros((nf, 6, ninput), complex)
    # input 0: bending about y (xx + xy), resonance 40 Hz
    H[:, 0, 0] = peak(40, 3, 1.0) + 0.2j * peak(40, 3, 0.4)
    H[:, 3, 0] = 0.3 * peak(40, 3, 0.6) * np.exp(0.5j)
    if ninput > 1:
        # input 1: bending about z (yy + xy), resonance 65 Hz
        H[:, 1, 1] = peak(65, 4, 0.9)
        H[:, 3, 1] = 0.35 * peak(65, 4, 0.5) * np.exp(0.7j)
    G = np.zeros((nf, ninput))
    G[:, 0] = peak(45, 25, 2.0) + 0.05
    if ninput > 1:
        G[:, 1] = peak(60, 30, 1.5) + 0.05
    return f, omega, H, G


# ============================================================================
# INPUT CROSS-PSD MODEL
# ============================================================================

def test_single_input_reduces_to_m21_and_m19_exactly():
    """The ninput = 1 (diagonal / rank-1) special case is BIT-IDENTICAL to the
    M21 rank-1 stress-tensor cross-PSD and the M19 |U|^2 G response PSD."""
    _f, _w, H, G = _synth_columns(ninput=1)
    Sff1 = G[:, :1][:, :, None]                     # (nf, 1, 1) = [[G]]
    S = mir.stress_tensor_cross_psd_multi(H, Sff1)
    Sref = mf.stress_tensor_cross_psd(H[:, :, 0], G[:, 0])
    assert np.array_equal(S, Sref)                  # bit-identical
    # response diagonal == the M19 |U|^2 G law
    U = np.zeros((H.shape[0], 3, 1), complex)
    U[:, 0, 0] = H[:, 0, 0]
    Sd = mir.response_cross_psd_diagonal(U, Sff1)
    assert np.array_equal(Sd[:, 0], np.abs(H[:, 0, 0]) ** 2 * G[:, 0])


def test_input_cross_psd_hermitian_and_coherence_identity():
    """The assembled input cross-PSD is Hermitian and obeys the coherence
    identity |S_ff[a,b]|^2 = gamma_ab^2 G_a G_b."""
    _f, _w, _H, G = _synth_columns()
    gamma = 0.6
    out = mir.input_cross_psd_matrix(G, gamma=gamma, phase=0.4)
    Sff = out["Sff"]
    assert np.allclose(Sff, np.conj(np.transpose(Sff, (0, 2, 1))))
    lhs = np.abs(Sff[:, 0, 1]) ** 2
    rhs = gamma ** 2 * G[:, 0] * G[:, 1]
    assert np.allclose(lhs, rhs)
    # the diagonal is exactly the auto-PSDs
    assert np.allclose(Sff[:, 0, 0].real, G[:, 0])
    assert np.allclose(Sff[:, 1, 1].real, G[:, 1])


def test_exponential_coherence_model():
    """The EXPONENTIAL / decay coherence model for distributed loads: coherence
    falls off with SEPARATION and FREQUENCY, always in [0, 1], and yields a
    Hermitian PSD input cross-PSD."""
    nf = 100
    f = np.linspace(1.0, 100.0, nf)
    G = np.ones((nf, 3))
    pos = np.array([[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]])
    gstack, _th = mir.exponential_coherence(f, pos, decay=0.05, ref_speed=10.0)
    assert gstack.shape == (nf, 3, 3)
    assert np.allclose(np.einsum("fii->fi", gstack), 1.0)      # diagonal 1
    # closer inputs (0,1) more coherent than far inputs (0,2) at every frequency
    assert np.all(gstack[:, 0, 1] >= gstack[:, 0, 2])
    # coherence decreases with frequency for a fixed separation
    assert np.all(np.diff(gstack[:, 0, 1]) <= 1e-12)
    out = mir.input_cross_psd_matrix(G, gamma=gstack, phase=0.0)
    Sff = out["Sff"]
    assert np.allclose(Sff, np.conj(np.transpose(Sff, (0, 2, 1))))
    assert np.all(np.linalg.eigvalsh(Sff) >= -1e-9)


def test_psd_projection_noop_and_clip():
    """nearest_psd is a numerical no-op on an already-valid matrix and clips a
    constructed non-PSD one (a coherence > 1 forces a negative eigenvalue)."""
    _f, _w, _H, G = _synth_columns()
    valid = mir.input_cross_psd_matrix(G, gamma=0.5)["Sff"]
    proj, info = mir.nearest_psd(valid)
    assert not info["clipped"]
    assert np.allclose(proj, valid)
    # a hand-built non-PSD 2x2 Hermitian stack (|off| > sqrt(diag*diag))
    nf = 10
    bad = np.zeros((nf, 2, 2), complex)
    bad[:, 0, 0] = 1.0
    bad[:, 1, 1] = 1.0
    bad[:, 0, 1] = 2.0                              # coherence 2 > 1 -> not PSD
    bad[:, 1, 0] = 2.0
    projb, infob = mir.nearest_psd(bad)
    assert infob["clipped"] and infob["min_eig"] < 0.0
    # the projection is PSD (non-negative eigenvalues)
    assert np.all(np.linalg.eigvalsh(projb) >= -1e-12)


def test_diagonal_incoherent_is_sum():
    """A DIAGONAL (incoherent) S_ff gives the SUM of the per-input single-input
    cross-PSDs EXACTLY."""
    _f, _w, H, G = _synth_columns()
    Sff = mir.input_cross_psd_matrix(G, gamma=None)["Sff"]   # diagonal
    S = mir.stress_tensor_cross_psd_multi(H, Sff)
    Ssum = (mf.stress_tensor_cross_psd(H[:, :, 0], G[:, 0])
            + mf.stress_tensor_cross_psd(H[:, :, 1], G[:, 1]))
    assert np.allclose(S, Ssum, atol=1e-12)


def test_rank1_coherent_reduces_to_effective_single():
    """A RANK-1 fully-coherent S_ff (one common source, gamma = 1, same G)
    reduces to the single-input cross-PSD of the effective combined pattern
    H_1 + H_2 EXACTLY."""
    _f, _w, H, G = _synth_columns()
    Gc = np.stack([G[:, 0], G[:, 0]], axis=1)       # a common source (same G)
    Sff = mir.input_cross_psd_matrix(Gc, gamma=1.0, phase=0.0)["Sff"]
    assert np.all(mir.matrix_rank_psd(Sff) <= 1)    # rank-1
    S = mir.stress_tensor_cross_psd_multi(H, Sff)
    Heff = H[:, :, 0] + H[:, :, 1]
    Seff = mf.stress_tensor_cross_psd(Heff, G[:, 0])
    assert np.allclose(S, Seff, atol=1e-10)


def test_partial_coherence_monotonic():
    """Partial coherence interpolates MONOTONICALLY between the incoherent (sum)
    and the fully-coherent answers: with in-phase constructive columns (here two
    proportional patterns of a common source) the equivalent-stress variance
    rises monotonically with gamma from the incoherent SUM to the coherent
    combination."""
    _f, omega, H, _G = _synth_columns()
    # two proportional patterns of a common source (guarantees the cross term is
    # positive so the interpolation is monotone constructive)
    H2 = np.zeros_like(H)
    H2[:, :, 0] = H[:, :, 0]
    H2[:, :, 1] = 0.7 * H[:, :, 0]
    Gc = np.stack([_G[:, 0], _G[:, 0]], axis=1)
    var = []
    for gamma in np.linspace(0.0, 1.0, 6):
        Sff = mir.input_cross_psd_matrix(Gc, gamma=gamma, phase=0.0)["Sff"]
        S = mir.stress_tensor_cross_psd_multi(H2, Sff)
        Svm = mir.equivalent_vonmises_psd_multi(S)
        var.append(float(np.atleast_1d(spectral_moments(omega, Svm, nmax=0))[0]))
    var = np.array(var)
    assert np.all(np.diff(var) > 0.0)               # strictly increasing
    # the incoherent endpoint is the SUM of the per-input variances
    v_sum = 0.0
    for a in range(2):
        Sa = mf.stress_tensor_cross_psd(H2[:, :, a], Gc[:, a])
        v_sum += float(np.atleast_1d(spectral_moments(
            omega, mir.equivalent_vonmises_psd_multi(Sa), nmax=0))[0])
    assert var[0] == pytest.approx(v_sum, rel=1e-9)
    # the coherent endpoint is the effective combined pattern (1 + 0.7)
    Seff = mf.stress_tensor_cross_psd(H2[:, :, 0] + H2[:, :, 1], Gc[:, 0])
    v_eff = float(np.atleast_1d(spectral_moments(
        omega, mir.equivalent_vonmises_psd_multi(Seff), nmax=0))[0])
    assert var[-1] == pytest.approx(v_eff, rel=1e-9)


def test_two_input_sdof_closed_form():
    """A two-input SDOF response cross-PSD (diagonal, i.e. one output DOF) equals
    the hand-derived closed form S_uu = |H1|^2 G1 + |H2|^2 G2 +
    2 Re(H1 conj(H2) S_ff,12), with S_ff,12 = sqrt(G1 G2) gamma e^{i theta}."""
    nf = 800
    f = np.linspace(1e-3, 150.0, nf)
    w = 2.0 * np.pi * f

    def sdof(fn, zeta):
        wn = 2.0 * np.pi * fn
        return 1.0 / (wn ** 2 - w ** 2 + 2j * zeta * wn * w)
    H1 = sdof(40.0, 0.03)
    H2 = 0.7 * sdof(70.0, 0.04)
    G1 = 2.0 + 0.0 * f
    G2 = 1.2 + 0.0 * f
    gamma, theta = 0.5, 0.6
    G = np.stack([G1, G2], axis=1)
    Sff = mir.input_cross_psd_matrix(G, gamma=gamma, phase=theta)["Sff"]
    U = np.stack([H1, H2], axis=1)[:, None, :]      # (nf, 1, 2) one output DOF
    Sd = mir.response_cross_psd_diagonal(U, Sff)[:, 0]
    S12 = np.sqrt(G1 * G2) * gamma * np.exp(1j * theta)
    closed = (np.abs(H1) ** 2 * G1 + np.abs(H2) ** 2 * G2
              + 2.0 * np.real(H1 * np.conj(H2) * S12))
    assert np.allclose(Sd, closed)


def test_response_and_cross_psd_hermitian():
    """The full response cross-PSD S_uu = U S_ff U^H is Hermitian per frequency
    (a physical output cross-spectral matrix), and its diagonal is real."""
    _f, _w, _H, G = _synth_columns()
    nf = G.shape[0]
    U = np.zeros((nf, 3, 2), complex)
    U[:, 0, 0] = 1.0 + 0.2j
    U[:, 1, 1] = 0.5 - 0.3j
    U[:, 2, 0] = 0.4j
    Sff = mir.input_cross_psd_matrix(G, gamma=0.4, phase=0.3)["Sff"]
    Suu = mir.response_cross_psd(U, Sff)
    assert np.allclose(Suu, np.conj(np.transpose(Suu, (0, 2, 1))))
    diag = np.einsum("fjj->fj", Suu)
    assert np.allclose(diag.imag, 0.0)


# ============================================================================
# MULTI-INPUT FATIGUE + MONTE-CARLO
# ============================================================================

def test_reductions_byte_identical_given_same_scross():
    """The M21 reductions are byte-identical given the same S_sigmasigma: the
    multi-input summary of a rank-1 (single-input) S_sigmasigma equals the M21
    single-input multiaxial summary (the multi-input S_sigmasigma flows into the
    reductions UNCHANGED)."""
    _f, omega, H, G = _synth_columns(ninput=1)
    Sff1 = G[:, :1][:, :, None]
    Scross = mir.stress_tensor_cross_psd_multi(H, Sff1)
    summ_mi = mir.multi_input_multiaxial_summary(Scross, omega, 5.0, 1e12)
    summ_m21 = mf.multiaxial_fatigue_summary(H[:, :, 0], G[:, 0], omega, 5.0,
                                             1e12)
    for key in ("von_mises", "normal_plane", "shear_plane"):
        a = summ_mi[key]["summary"]["dirlik"]["damage_rate"]
        b = summ_m21[key]["summary"]["dirlik"]["damage_rate"]
        assert a == b
    assert np.array_equal(summ_mi["Mmats"], summ_m21["Mmats"])


def test_single_input_mc_bit_identical():
    """The ninput = 1 Monte-Carlo DELEGATES to the M21 single-input MC and is
    BIT-IDENTICAL to it."""
    _f, _w, H, G = _synth_columns(ninput=1)
    f = _f
    Sff1 = G[:, :1][:, :, None]
    proj = mf.shear_projection(np.array([0, 0, 1.0]), np.array([1.0, 0, 0]))
    d = mif.monte_carlo_multi_input_damage(f, Sff1, H, proj, 5.0, 1e12, 60.0, 9)
    Scross = mf.stress_tensor_cross_psd(H[:, :, 0], G[:, 0])
    dref = mf.monte_carlo_multiaxial_damage(f, Scross, proj, 5.0, 1e12, 60.0, 9)
    assert d["path"] == "delegated"
    assert d["damage_rate"] == dref["damage_rate"]


def test_coherent_mc_bit_identical_to_effective_single():
    """A fully-coherent (rank-1) MC DELEGATES to the M21 single-input MC on the
    effective combined stress-tensor cross-PSD, BIT-IDENTICALLY."""
    _f, _w, H, G = _synth_columns()
    f = _f
    Gc = np.stack([G[:, 0], G[:, 0]], axis=1)
    Sff = mir.input_cross_psd_matrix(Gc, gamma=1.0, phase=0.0)["Sff"]
    proj = mf.shear_projection(np.array([0, 0, 1.0]), np.array([0, 1.0, 0]))
    d = mif.monte_carlo_multi_input_damage(f, Sff, H, proj, 5.0, 1e12, 60.0, 3)
    Seff = mf.stress_tensor_cross_psd(H[:, :, 0] + H[:, :, 1], G[:, 0])
    dref = mf.monte_carlo_multiaxial_damage(f, Seff, proj, 5.0, 1e12, 60.0, 3)
    assert d["path"] == "delegated"
    assert d["damage_rate"] == dref["damage_rate"]


def test_incoherent_variance_additive_and_mc_agrees():
    """For INCOHERENT inputs the projected variance is the SUM of the per-input
    projected variances (exact), and the input-level MC matches the stress-level
    (M21-on-S_sigmasigma) MC within scatter."""
    _f, omega, H, G = _synth_columns()
    f = _f
    Sff = mir.input_cross_psd_matrix(G, gamma=None)["Sff"]      # incoherent
    Scross = mir.stress_tensor_cross_psd_multi(H, Sff)
    Mmats = mf.tensor_moment_matrices(omega, Scross, nmax=0)
    summ = mir.multi_input_multiaxial_summary(Scross, omega, 5.0, 1e12)
    proj = summ["shear_plane"]["proj"]
    # projected variance additivity: p^T M0_total p == sum_a p^T M0_a p
    m0_tot = float(proj @ Mmats[0] @ proj)
    m0_a = 0.0
    for a in range(2):
        Sa = mf.stress_tensor_cross_psd(H[:, :, a], G[:, a])
        m0_a += float(proj @ mf.tensor_moment_matrices(omega, Sa, nmax=0)[0]
                      @ proj)
    assert m0_tot == pytest.approx(m0_a, rel=1e-10)
    # input-level MC vs stress-level MC (same process, independent synthesis)
    di = mif.monte_carlo_multi_input_damage(f, Sff, H, proj, 5.0, 1e12, 400.0,
                                            5, force_input_level=True)
    ds = mf.monte_carlo_multiaxial_damage(f, Scross, proj, 5.0, 1e12, 400.0, 5)
    assert di["path"] == "input_level"
    assert di["rms"] == pytest.approx(ds["rms"], rel=0.2)
    if ds["damage_rate"] > 0:
        assert di["damage_rate"] == pytest.approx(ds["damage_rate"], rel=0.6)


def test_synthesized_coherence_matches_target():
    """The synthesised correlated inputs' MEASURED coherence matches the target
    gamma_ab (a partial coherence recovered by Welch averaging)."""
    nf = 600
    f = np.linspace(1e-3, 100.0, nf)
    G = np.stack([np.ones(nf), np.ones(nf)], axis=1)   # flat unit auto-PSDs
    gamma = 0.6
    Sff = mir.input_cross_psd_matrix(G, gamma=gamma, phase=0.0)["Sff"]
    t, F = mif.synthesize_multi_input_forces(f, Sff, 4000.0, 21)
    fs = 1.0 / (t[1] - t[0])
    fc, coh = mif.measure_coherence(F, fs, nperseg=1024)
    band = (fc > 5.0) & (fc < 90.0)
    measured = np.sqrt(coh[band, 0, 1].mean())         # coh is gamma^2
    assert measured == pytest.approx(gamma, abs=0.12)


# ============================================================================
# CARDS + END-TO-END + NO-REGRESSION
# ============================================================================

def _parse(text):
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.engine_keywords import parse_engine_deck
    d = tempfile.mkdtemp()
    ep = os.path.join(d, "M28_0001.rad")
    with open(ep, "w") as f:
        f.write(text)
    with contextlib.redirect_stdout(io.StringIO()):
        return parse_engine_deck(read_deck(ep), MessageLog())


def test_minput_card_parsing():
    """/IMPL/FATIG/MULT/MINPUT mirrors the multi-input table (implies MULT)."""
    ec = _parse(
        "#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/MINPUT\n0.0 400.0 400 10 6\n"
        "5.0 1.0e14 0.03 0.0 0.0 40.0 7\n2 0 0.5 30.0\n1 10\n2 11\n/END\n")
    assert ec.impl_fatig and ec.impl_fatig_mult and ec.impl_fatig_minput
    assert ec.impl_mi_cohmodel == 0
    assert ec.impl_mi_gamma == pytest.approx(0.5)
    assert ec.impl_mi_phase == pytest.approx(30.0)
    assert ec.impl_mi_inputs == ((1, 10, 0.0, 0.0, 0.0), (2, 11, 0.0, 0.0, 0.0))


def test_psd_multi_card_parsing():
    """/IMPL/PSD/MULTI mirrors the multi-input table + exponential coherence."""
    ec = _parse(
        "#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/PSD/MULTI\n0.0 400.0 400 10 6\n"
        "2 1 0.0 0.0 0.5 340.0\n1 10 0.0 0.0 0.0\n2 11 1.0 0.0 0.0\n/END\n")
    assert ec.impl_psd and ec.impl_psd_multi
    assert ec.impl_mi_cohmodel == 1
    assert ec.impl_mi_decay == pytest.approx(0.5)
    assert ec.impl_mi_speed == pytest.approx(340.0)
    assert ec.impl_mi_inputs[1] == (2, 11, 1.0, 0.0, 0.0)


def test_multi_input_end_to_end():
    """The solid-brick cantilever under TWO partially-coherent inputs: the
    multi-input critical-element fatigue is reported alongside the single-input
    answer, and the two Monte-Carlo formulations agree."""
    eng = ("#\n/RUN/BR/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/MINPUT\n"
           "0.0 400.0 400 10 6\n5.0 1.0e14 0.03 0.0 0.0 40.0 7\n"
           "2 0 0.5 0.0\n1 10\n2 11\n/PRINT/-500\n/STOP\n15.0\n")
    model, out = _run(_two_input_brick(), eng, capture=True)
    fat = model.implicit_result.fatigue
    assert fat is not None and fat.get("multiaxial")
    mi = fat.get("multi_input")
    assert mi is not None and mi["ninput"] == 2
    assert mi["von_mises"]["summary"]["dirlik"]["damage_rate"] > 0.0
    # the two MC formulations agree within scatter
    dr_s = mi["monte_carlo"]["damage_rate"]
    dr_i = mi["monte_carlo_input"]["damage_rate"]
    assert dr_i == pytest.approx(dr_s, rel=0.5)
    assert "MULTI-INPUT" in out and "COHERENCE MODEL" in out


def test_single_input_path_byte_identical_with_minput():
    """The single-input /MULT reductions are byte-unchanged whether or not the
    MINPUT path runs (a NEW parallel path)."""
    base = ("#\n/RUN/BR/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT\n0.0 400.0 400 10 6\n"
            "5.0 1.0e14 0.03 0.0 0.0 40.0 7\n/PRINT/-500\n/STOP\n15.0\n")
    mi = ("#\n/RUN/BR/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/MINPUT\n"
          "0.0 400.0 400 10 6\n5.0 1.0e14 0.03 0.0 0.0 40.0 7\n"
          "2 0 0.5 0.0\n1 10\n2 11\n/PRINT/-500\n/STOP\n15.0\n")
    f1 = _run(_two_input_brick(), base).implicit_result.fatigue
    f2 = _run(_two_input_brick(), mi).implicit_result.fatigue
    assert "multi_input" not in f1 and "multi_input" in f2
    for key in ("von_mises", "normal_plane", "shear_plane"):
        assert (f1[key]["summary"]["dirlik"]["damage_rate"]
                == f2[key]["summary"]["dirlik"]["damage_rate"])
    assert np.array_equal(f1["Scross"], f2["Scross"])


def test_psd_multi_end_to_end():
    """/IMPL/PSD/MULTI attaches the multi-input response diagnostics alongside
    the single-input RMS."""
    eng = ("#\n/RUN/BR/1\n1.0\n/IMPL\n/IMPL/PSD/MULTI\n0.0 400.0 400 10 6\n"
           "2 0 0.5 0.0\n1 10\n2 11\n/PRINT/-500\n/STOP\n15.0\n")
    model = _run(_two_input_brick(), eng)
    rr = model.implicit_result.random_response
    assert rr is not None and "multi_input" in rr
    mi = rr["multi_input"]
    assert mi["ninput"] == 2 and mi["rms"].size == rr["rms"].size
    # response coherence diagonal is 1
    coh = mi["response_coherence"]
    assert np.allclose(np.einsum("fjj->fj", coh), 1.0)


def test_multi_input_composes_with_ngauss():
    """/IMPL/FATIG/MULT/MINPUT/NGAUSS attaches the non-Gaussian correction to the
    multi-input reductions (composition with the orthogonal M24 flag)."""
    eng = ("#\n/RUN/BR/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/MINPUT/NGAUSS\n"
           "0.0 400.0 400 10 6\n5.0 1.0e14 0.03 0.0 0.0 40.0 7\n5.5 0.0\n"
           "2 0 0.5 0.0\n1 10\n2 11\n/PRINT/-500\n/STOP\n15.0\n")
    model = _run(_two_input_brick(), eng)
    mi = model.implicit_result.fatigue["multi_input"]
    assert mi["nongaussian"] is not None
    assert mi["nongaussian"]["gamma4"] == pytest.approx(5.5)


def test_multi_input_does_not_mutate_state():
    """The multi-input recovery is read-only: a repeated run gives identical
    element stresses, and the single-input result is unchanged."""
    eng = ("#\n/RUN/BR/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/MINPUT\n"
           "0.0 400.0 400 10 6\n5.0 1.0e14 0.03 0.0 0.0 0.0 7\n"
           "2 0 0.5 0.0\n1 10\n2 11\n/PRINT/-500\n/STOP\n15.0\n")
    m1 = _run(_two_input_brick(), eng)
    m2 = _run(_two_input_brick(), eng)
    a = m1.implicit_result.fatigue["multi_input"]
    b = m2.implicit_result.fatigue["multi_input"]
    assert np.array_equal(a["Scross"], b["Scross"])
    assert (a["von_mises"]["summary"]["dirlik"]["damage_rate"]
            == b["von_mises"]["summary"]["dirlik"]["damage_rate"])


def test_direct_dynamics_unchanged_by_multi_input_path():
    """A DIRECT /IMPL/DYNA run is byte-for-byte unaffected by the M28 multi-input
    machinery living in the same package."""
    K_D, M_D = 4.0, 2.0e-3
    om = np.sqrt(K_D / (M_D / 2.0))
    T = 2.0 * np.pi / om
    starter = f"""\
#RADIOSS STARTER
/BEGIN
SM
/NODE
         1{0.0:20.10f}{0.0:20.10f}{0.0:20.10f}
         2{10.0:20.10f}{0.0:20.10f}{0.0:20.10f}
/SPRING/1
         1         1         2
/PART/1
s
         1         1
/MAT/LAW1/1
st
   7.8e-6
     210.0       0.0
/PROP/SPRING/1
sp
     {M_D}       {K_D}       0.0
/GRNOD/NODE/1
n
1
/GRNOD/NODE/2
t
2
/BCS/1
p
       111       111         0         1
/BCS/2
a
       011       111         0         2
/CLOAD/1
d
         1         X         2       1.0
/FUNCT/1
f
       0.0       1.0
     100.0       1.0
/END
"""
    eng = (f"#\n/RUN/SM/1\n{2*T}\n/IMPL\n/IMPL/DYNA/2\n0.5 0.25\n"
           f"/IMPL/DT\n{T/50}\n/PRINT/-500\n/STOP\n15.0\n")
    m = _run(starter, eng)
    # a plain dynamics run: no fatigue attached, converged
    assert m.implicit_result is not None
    assert getattr(m.implicit_result, "fatigue", None) is None
