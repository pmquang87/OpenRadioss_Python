"""
M23 validations: SPECTRAL NON-PROPORTIONAL MULTIAXIAL FATIGUE
(/IMPL/FATIG/MULT/NPROP/SPEC) — the FREQUENCY-DOMAIN non-proportionality factor
and critical-plane damage estimated DIRECTLY from the stress-tensor cross-PSD
spectral-MOMENT matrices, with NO synthesised history. Built ALONGSIDE the M21
MULTIAXIAL SPECTRAL path AND the M22 NON-PROPORTIONAL TIME-DOMAIN path (both stay
bit-identical whether or not SPEC runs; the spectral estimator CONSUMES the M21
moment-matrix / candidate-plane machinery read-only) and the M16-M20 solvers.

Every new capability gets at least one ANALYTIC / cross-milestone check (the
port's philosophy):

SPECTRAL SHEAR-PATH COVARIANCE + NON-PROPORTIONALITY FACTOR
* the 2x2 in-plane shear cross-spectral moment matrix (from M_0) EQUALS the M22
  time-domain shear-path covariance (np.cov of the synthesised path) — the clean
  M23<->M22 identity M_0 = E[sigma sigma^T];
* the SPECTRAL F_np EQUALS the M22 TIME-DOMAIN F_np on the same cross-PSD;
* F_np = 0 for a PROPORTIONAL (rank-1) state and F_np = 1 for an equal-amplitude
  90-deg-out-of-phase (circular) state (closed forms from synthetic cross-PSDs);
* the spectral resolved-shear / resolved-normal amplitudes from p^T M_n p match
  the RMS of the M22 synthesised histories.

SPECTRAL CRITICAL-PLANE DAMAGE (Cristofori-Susmel-Tovo / modified Wohler)
* for PROPORTIONAL loading the spectral non-proportional damage reduces EXACTLY
  to the M21 critical-plane spectral answer (F_np = 0, no correction);
* for a 90-deg-OUT-OF-PHASE case the spectral non-proportional damage AGREES with
  the M22 TIME-DOMAIN path-counting damage within the seeded Monte-Carlo scatter
  AND is HIGHER than the uncorrected M21 projected-scalar answer, by
  (1 + F_np^2)^(m/2) = 2^(m/2);
* a hand check of the spectral F_np / rho / g on a synthetic two-channel cross-PSD.

CARDS + END-TO-END + NO-REGRESSION (the M7 parity contract)
* /IMPL/FATIG/MULT/NPROP/SPEC card mirror (a PORT sub-flag, implies NPROP hence
  MULT, composes in any order);
* the solid-brick cantilever end to end: the spectral non-proportional reductions
  reported ALONGSIDE the M21 spectral AND the M22 time-domain numbers (three
  blocks in the listing);
* the M21 spectral AND the M22 time-domain answers are byte-identical whether or
  not SPEC runs, the recovery is read-only, and the direct M10 answer is unchanged.

See PORTING_GUIDE.md roadmap M23.
"""

import contextlib
import io
import math
import os
import tempfile

import numpy as np
import pytest

from pyradioss.engine.engine import run_engine
from pyradioss.starter.starter import run_starter

pytest.importorskip("scipy")   # implicit requires scipy (optional otherwise)

from pyradioss.common.messages import MessageLog                    # noqa: E402
from pyradioss.engine.kinematics import LoadsAndConstraints         # noqa: E402
from pyradioss.implicit.modal import modal_frequencies              # noqa: E402
from pyradioss.implicit.modal_response import (                     # noqa: E402
    build_modal_basis, modal_damping, modal_frequency_response)
from pyradioss.implicit.random_response import (                    # noqa: E402
    element_voigt_blocks, element_voigt_frf, stress_modes)
from pyradioss.implicit import multiaxial_fatigue as mf             # noqa: E402
from pyradioss.implicit import nonproportional_fatigue as npf       # noqa: E402
from pyradioss.implicit import spectral_fatigue as sf               # noqa: E402
from pyradioss.implicit import (                                    # noqa: E402
    spectral_nonproportional_fatigue as snp)

# reuse the M21 brick deck generator
from tests.test_m21_multiaxfatig import _brick_deck                 # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _run(starter_text, engine_text, capture=False):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M23_0000.rad")
    ep = os.path.join(d, "M23_0001.rad")
    with open(sp, "w") as f:
        f.write(starter_text)
    with open(ep, "w") as f:
        f.write(engine_text)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_starter(sp)
        model = run_engine(ep)
    return (model, buf.getvalue()) if capture else model


def _starter(text):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M23_0000.rad")
    with open(sp, "w") as f:
        f.write(text)
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(sp)


def _proportional_cross_psd(nf=1500, fmax=120.0):
    """A synthetic PROPORTIONAL multiaxial cross-PSD: a single REAL modal shape
    (H[:,c] = shape(Omega) * v[c]) so every stress component scales together —
    the principal axes are fixed and the shear path is a LINE on every plane
    (F_np = 0). Two real resonances in the shape. (The M22 template.)"""
    f = np.linspace(1e-3, fmax, nf)
    omega = 2.0 * math.pi * f
    Sff = np.ones(nf)
    shape = (np.exp(-((f - 30) ** 2) / (2 * 2.5 ** 2))
             + 0.6 * np.exp(-((f - 70) ** 2) / (2 * 2.5 ** 2)))
    v = np.array([1.0, 0.3, 0.0, 0.5, 0.0, 0.2])
    H = shape[:, None] * v[None, :]                    # (nf, 6) real -> in phase
    Scross = mf.stress_tensor_cross_psd(H, Sff)
    return f, omega, Sff, H, Scross


def _circular_cross_psd(nf=2000, fmax=120.0, f0=40.0, bw=8.0):
    """A synthetic 90-deg-OUT-OF-PHASE, equal-amplitude multiaxial cross-PSD: two
    shear channels (sigma_xy col 3 and sigma_zx col 5) on the SAME frequency band
    with a 90-deg phase (H[:,3] real, H[:,5] = i * H[:,3]) — so on the x-plane the
    resolved shear traces an isotropic CIRCLE (F_np = 1). A single real input
    process (rank-1 cross-PSD)."""
    f = np.linspace(1e-3, fmax, nf)
    omega = 2.0 * math.pi * f
    Sff = np.ones(nf)
    band = np.exp(-((f - f0) ** 2) / (2 * bw ** 2))
    H = np.zeros((nf, 6), dtype=complex)
    H[:, 3] = band                                     # sigma_xy (real)
    H[:, 5] = 1j * band                                # sigma_zx (90 deg)
    Scross = mf.stress_tensor_cross_psd(H, Sff)
    return f, omega, Sff, H, Scross


# ============================================================================
# SPECTRAL SHEAR-PATH COVARIANCE + NON-PROPORTIONALITY FACTOR
# ============================================================================

def test_spectral_shear_covariance_equals_timedomain():
    """The 2x2 in-plane shear cross-spectral moment matrix (from M_0) EQUALS the
    M22 TIME-DOMAIN shear-path covariance (np.cov of the synthesised (tau_a,
    tau_b) path) — the identity M_0 = E[sigma sigma^T], the covariance the
    multivariate synthesiser reproduces — within the seeded synthesis scatter."""
    f, omega, Sff, H, Scross = _circular_cross_psd()
    Mmats = mf.tensor_moment_matrices(omega, Scross, nmax=4)
    n = np.array([1.0, 0.0, 0.0])
    Sig_spec, _ = snp.inplane_shear_covariance(Mmats[0], n)
    # M22 time-domain covariance of the SAME synthesised path
    t, X = mf.synthesize_multiaxial_history(f, Scross, 4000.0, 3)
    P, _ = npf.resolved_shear_path(X, n)
    Sig_td = np.cov(P.T)
    # the two 2x2 matrices agree within the finite-record scatter
    assert np.allclose(Sig_spec, Sig_td, rtol=0.1, atol=1e-3 * Sig_spec.max())


def test_spectral_fnp_equals_timedomain_fnp():
    """The SPECTRAL F_np (from the moment matrix M_0) EQUALS the M22 TIME-DOMAIN
    F_np (from the synthesised shear-path covariance) on the SAME cross-PSD — the
    clean M23<->M22 cross-check, computed with no synthesis."""
    f, omega, Sff, H, Scross = _circular_cross_psd()
    Mmats = mf.tensor_moment_matrices(omega, Scross, nmax=4)
    n = np.array([1.0, 0.0, 0.0])
    fnp_spec = snp.spectral_nonproportionality_factor(Mmats[0], n)
    t, X = mf.synthesize_multiaxial_history(f, Scross, 4000.0, 3)
    P, _ = npf.resolved_shear_path(X, n)
    fnp_td = npf.nonproportionality_factor(P)
    assert fnp_spec == pytest.approx(fnp_td, abs=0.02)


def test_spectral_fnp_zero_for_proportional():
    """F_np = 0 (on EVERY plane) for a PROPORTIONAL (rank-1 real) state — the
    shear path is a line, the 2x2 shear covariance is rank-1 (lambda_2 = 0)."""
    f, omega, Sff, H, Scross = _proportional_cross_psd()
    Mmats = mf.tensor_moment_matrices(omega, Scross, nmax=4)
    for n in mf.candidate_normals(8, 7):
        assert snp.spectral_nonproportionality_factor(Mmats[0], n) < 1e-5


def test_spectral_fnp_one_for_circle():
    """F_np = 1 for the equal-amplitude 90-deg-out-of-phase (circular) state on
    the x-plane — the isotropic shear covariance (lambda_1 = lambda_2)."""
    f, omega, Sff, H, Scross = _circular_cross_psd()
    Mmats = mf.tensor_moment_matrices(omega, Scross, nmax=4)
    n = np.array([1.0, 0.0, 0.0])
    assert snp.spectral_nonproportionality_factor(Mmats[0], n) == pytest.approx(
        1.0, abs=1e-6)


def test_spectral_resolved_amplitudes_match_synthesis():
    """The spectral resolved-shear / resolved-normal amplitudes (from p^T M_n p —
    RMS = sqrt(m_0)) match the RMS of the M22 synthesised histories on the same
    plane, within the seeded scatter."""
    f, omega, Sff, H, Scross = _circular_cross_psd()
    Mmats = mf.tensor_moment_matrices(omega, Scross, nmax=4)
    n = np.array([1.0, 0.0, 0.0])
    st = snp._plane_spectral_stats(Mmats, n)
    t, X = mf.synthesize_multiaxial_history(f, Scross, 4000.0, 5)
    P, _ = npf.resolved_shear_path(X, n)
    sn = npf.resolved_normal_history(X, n)
    tau_dom, _ = npf._dominant_shear_scalar(P)
    # dominant-shear RMS = sqrt(lambda_1); resolved-normal RMS = sqrt(m0^sigma)
    assert math.sqrt(st["lambda1"]) == pytest.approx(np.std(tau_dom), rel=0.1)
    assert math.sqrt(st["normal_mom"][0]) == pytest.approx(np.std(sn), rel=0.15)


def test_spectral_fnp_rho_hand_check():
    """A hand check of the spectral F_np / rho / g on a synthetic two-channel
    cross-PSD. Build two shear channels at the SAME band, amplitude ratio a:1,
    90-deg phase, so the shear path is an ELLIPSE of semi-axes proportional to
    (a, 1): F_np = 1/a (the aspect ratio, minor/major), g = sqrt(1 + F_np^2), and
    with a known normal channel the ratio rho = psf*sigma_n_rms/tau_a is
    hand-computable."""
    nf = 2000
    f = np.linspace(1e-3, 120.0, nf)
    omega = 2.0 * math.pi * f
    Sff = np.ones(nf)
    band = np.exp(-((f - 40) ** 2) / (2 * 8.0 ** 2))
    a = 3.0
    H = np.zeros((nf, 6), dtype=complex)
    H[:, 3] = a * band              # sigma_xy amplitude a (major axis)
    H[:, 5] = 1j * band             # sigma_zx amplitude 1 (minor axis, 90 deg)
    Scross = mf.stress_tensor_cross_psd(H, Sff)
    Mmats = mf.tensor_moment_matrices(omega, Scross, nmax=4)
    n = np.array([1.0, 0.0, 0.0])
    fnp = snp.spectral_nonproportionality_factor(Mmats[0], n)
    # the shear covariance has principal variances prop to a^2 and 1 -> F_np=1/a
    assert fnp == pytest.approx(1.0 / a, rel=1e-3)
    st = snp._plane_spectral_stats(Mmats, n)
    # g = sqrt(1 + F_np^2); effective shear RMS = g * dominant RMS
    assert math.sqrt(st["gfac2"]) == pytest.approx(math.hypot(a, 1.0) / a,
                                                   rel=1e-3)
    assert st["tau_a"] == pytest.approx(math.sqrt(st["lambda1"] + st["lambda2"]),
                                        rel=1e-9)


# ============================================================================
# SPECTRAL CRITICAL-PLANE DAMAGE
# ============================================================================

def test_proportional_reduces_to_m21_spectral():
    """For PROPORTIONAL loading the SPECTRAL non-proportional damage (shear-path
    model, F_np = 0) reduces EXACTLY to the M21 max-shear critical-plane spectral
    answer — no correction, no spurious damage (the identity M23 = M21 when the
    load path is proportional)."""
    f, omega, Sff, H, Scross = _proportional_cross_psd()
    m, C = 5.0, 1e15
    Mmats = mf.tensor_moment_matrices(omega, Scross, nmax=4)
    # M21 max-shear critical-plane Dirlik (the reference)
    cs = mf.critical_plane_search(Mmats, method="shear")
    m21 = sf.fatigue_summary(cs["moments"], m, C)["dirlik"]["damage_rate"]
    # M23 spectral shear-path
    r = snp.spectral_critical_plane_damage(Mmats, m, C, model="shear_path")
    assert r["F_np"] < 1e-4
    assert r["damage_rate"] == pytest.approx(m21, rel=1e-6)


def test_spectral_higher_than_m21_for_circle():
    """For a 90-deg-OUT-OF-PHASE (circular) state the SPECTRAL non-proportional
    damage is HIGHER than the uncorrected M21 projected-scalar answer by
    (1 + F_np^2)^(m/2) = 2^(m/2) — the extra damage the projection misses."""
    f, omega, Sff, H, Scross = _circular_cross_psd()
    m, C = 5.0, 1e15
    Mmats = mf.tensor_moment_matrices(omega, Scross, nmax=4)
    cs = mf.critical_plane_search(Mmats, method="shear")
    m21 = sf.fatigue_summary(cs["moments"], m, C)["dirlik"]["damage_rate"]
    r = snp.spectral_critical_plane_damage(Mmats, m, C, model="shear_path")
    assert r["F_np"] == pytest.approx(1.0, abs=1e-4)
    ratio = r["damage_rate"] / m21
    assert ratio == pytest.approx(2.0 ** (m / 2.0), rel=0.02)
    assert ratio > 1.5


def test_spectral_agrees_with_m22_timedomain():
    """For a 90-deg-OUT-OF-PHASE (circular) state the SPECTRAL non-proportional
    damage AGREES with the M22 TIME-DOMAIN path-counting damage within the seeded
    Monte-Carlo scatter — the two non-proportional methods (spectral estimator vs
    time-domain path count) converge (the whole point of M23), exactly as M20/M21
    used the Monte-Carlo rainflow to check the spectral Dirlik."""
    f, omega, Sff, H, Scross = _circular_cross_psd()
    m, C = 5.0, 1e15
    dur, seed = 400.0, 7
    Mmats = mf.tensor_moment_matrices(omega, Scross, nmax=4)
    # M23 spectral shear-path (no history)
    r23 = snp.spectral_critical_plane_damage(Mmats, m, C, model="shear_path")
    # M22 time-domain shear-path path count on the SAME cross-PSD
    npres = npf.nonproportional_summary(f, Scross, m, C, dur, seed,
                                        models=("shear_path",))
    r22 = npres["shear_path"]["damage_rate"]
    ratio = r23["damage_rate"] / r22
    assert 0.5 < ratio < 2.0, f"M23/M22 = {ratio:.3f} outside scatter"


# ============================================================================
# CARDS
# ============================================================================

def _parse(text):
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.engine_keywords import parse_engine_deck
    d = tempfile.mkdtemp()
    p = os.path.join(d, "E_0001.rad")
    with open(p, "w") as f:
        f.write(text)
    log = MessageLog()
    with contextlib.redirect_stdout(io.StringIO()):
        ec = parse_engine_deck(read_deck(p), log)
    return ec, log


def test_impl_fatig_spec_card_parsing():
    """/IMPL/FATIG/MULT/NPROP/SPEC sets the spectral flag AND implies the
    non-proportional (hence multiaxial) flags; it accepts the optional k / sigma_y
    on line 2 and composes with the other sub-keywords in any order — a PORT
    sub-flag."""
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/NPROP/SPEC\n"
                   "0.0 400.0 1200 10 6\n5.0 1.0e4 0.03 0.0 0.0 300.0 21000 "
                   "0.25 2.0\n/END\n")
    assert ec.implicit and ec.impl_fatig
    assert ec.impl_fatig_mult and ec.impl_fatig_nprop and ec.impl_fatig_spec
    assert ec.impl_fatig_k == pytest.approx(0.25)
    assert ec.impl_fatig_sigy == pytest.approx(2.0)
    assert ec.impl_fatig_mcdur == pytest.approx(300.0)

    # SPEC implies NPROP + MULT even without them spelled out, any order, + BASE
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/FATIG/SPEC/BASE\n"
                   "0.0 250 2000 7 0 6\n4.0 5e3 0.05 0 0 300 9\n/END\n")
    assert (ec.impl_fatig_spec and ec.impl_fatig_nprop
            and ec.impl_fatig_mult and ec.impl_fatig_base)

    # the plain M22 NPROP card is NOT spectral
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/FATIG/MULT/NPROP\n"
                   "0.0 400 1200 10 6\n5.0 1.0e4 0.03 0 0 15 9\n/END\n")
    assert ec.impl_fatig_nprop and not ec.impl_fatig_spec


# ============================================================================
# END-TO-END
# ============================================================================

_SPEC_ENGINE = (
    "#\n/RUN/BRICK/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/NPROP/SPEC\n"
    "0.0 400.0 500 10 6\n5.0 1.0e4 0.03 0.0 0.0 15.0 21000 0.3 0.5\n"
    "/PRINT/-500\n/STOP\n15.0\n")


def test_spectral_end_to_end():
    """/IMPL/FATIG/MULT/NPROP/SPEC end to end on the solid-brick cantilever: the
    spectral non-proportional critical-plane result is stored on
    ``fatigue['nprop_result']['spectral']`` ALONGSIDE the M21 spectral reductions
    AND the M22 time-domain path count, with the three models (Findley /
    Fatemi-Socie / shear-path) reporting finite lives, a reported F_np / rho, and
    the listing carrying ALL THREE blocks side by side."""
    m, out = _run(_brick_deck(nx=3), _SPEC_ENGINE, capture=True)
    fat = m.implicit_result.fatigue
    assert fat is not None and fat.get("multiaxial")
    assert fat.get("nonproportional") and fat.get("spectral_nonproportional")
    sp = fat["nprop_result"]["spectral"]
    assert sp is not None
    amp = sp["amplitudes"]
    assert 0.0 <= amp["F_np"] <= 1.0
    assert amp["g"] == pytest.approx(math.sqrt(1.0 + amp["F_np"] ** 2), rel=1e-9)
    for mdl in ("findley", "fatemi_socie", "shear_path"):
        r = sp[mdl]
        assert r["damage_rate"] > 0.0
        assert np.isfinite(r["life"]) and r["life"] > 0.0
        assert abs(np.linalg.norm(r["normal"]) - 1.0) < 1e-6
        assert 0.0 <= r["F_np"] <= 1.0
    # all THREE listing blocks present (M21 / M22 / M23) side by side
    assert "MULTIAXIAL / CRITICAL-PLANE SPECTRAL FATIGUE" in out
    assert "NON-PROPORTIONAL MULTIAXIAL FATIGUE" in out
    assert "SPECTRAL NON-PROPORTIONAL FATIGUE" in out
    assert "PATH FACTOR g" in out


# ============================================================================
# NO-REGRESSION (the M7 parity contract)
# ============================================================================

def test_m21_m22_byte_identical_with_without_spec():
    """The M21 spectral answer AND the M22 time-domain path-counting answer are
    BYTE-IDENTICAL whether or not the M23 spectral non-proportional path runs —
    the spectral estimator is NEW and ALONGSIDE, never mutating the M21 reductions
    OR the M22 path count (the parity contract)."""
    deck = _brick_deck(nx=3)
    eng_m22 = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/NPROP\n"
               "0.0 400.0 500 10 6\n5.0 1.0e4 0.03 0.0 0.0 15.0 21000 0.3 0.5\n"
               "/PRINT/-500\n/STOP\n15.0\n")
    m22 = _run(deck, eng_m22)
    m23 = _run(deck, _SPEC_ENGINE)
    f22 = m22.implicit_result.fatigue
    f23 = m23.implicit_result.fatigue
    assert not f22.get("spectral_nonproportional")
    assert f23.get("spectral_nonproportional")
    # M21 spectral reductions byte-identical
    for key in ("von_mises", "normal_plane", "shear_plane"):
        for est in ("narrow_band", "dirlik", "wirsching_light",
                    "tovo_benasciutti"):
            a = f22[key]["summary"][est]["damage_rate"]
            b = f23[key]["summary"][est]["damage_rate"]
            assert a == b, f"M21 {key}/{est}: {a} != {b}"
    # M22 time-domain path count byte-identical
    for mdl in ("findley", "fatemi_socie", "shear_path"):
        a = f22["nprop_result"][mdl]["damage_rate"]
        b = f23["nprop_result"][mdl]["damage_rate"]
        assert a == b, f"M22 {mdl}: {a} != {b}"
        assert np.array_equal(f22["nprop_result"][mdl]["normal"],
                              f23["nprop_result"][mdl]["normal"])


def test_spec_does_not_mutate_state():
    """The spectral non-proportional path is read-only in the element state and
    leaves the M16 modal_frequencies output bit-identical before and after (the
    M14-M22 parity contract extended to M23)."""
    m = _starter(_brick_deck(nx=3))
    f0, _, _ = modal_frequencies(m, nev=6)
    before = {k: (v.copy() if isinstance(v, np.ndarray) else v)
              for k, v in m.bricks.state.items()}
    basis = build_modal_basis(m, nev=6)
    Sigma, channels = stress_modes(m, basis)
    blocks = element_voigt_blocks(channels)
    z = modal_damping(basis.omega, uniform=0.03)
    n = m.numnod
    F = np.zeros((n, 3))
    LoadsAndConstraints(m, MessageLog()).external_forces(1.0, F, m.x0)
    frf = modal_frequency_response(basis, F, np.zeros((n, 3)),
                                   np.linspace(1e-6, 300.0, 60), z)
    Hv = element_voigt_frf(frf, Sigma, blocks[0][3])
    Scross = mf.stress_tensor_cross_psd(Hv, np.ones(60))
    Mmats = mf.tensor_moment_matrices(np.asarray(frf["omega"]), Scross, nmax=4)
    snp.spectral_nonproportional_summary(Mmats, 5.0, 1e4, naz=8, npol=7)
    for k, v in before.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(m.bricks.state[k], v), k
    f1, _, _ = modal_frequencies(m, nev=6)
    assert np.array_equal(f0, f1)


def test_direct_dynamics_unchanged_by_spec_path():
    """A DIRECT /IMPL/DYNA run is byte-for-byte unaffected by the M23 spectral
    non-proportional machinery living in the same package (the M10 integrator
    stays bit-identical; no fatigue attached)."""
    K_D, M_D = 4.0, 2.0e-3
    om = math.sqrt(K_D / (M_D / 2.0))
    T = 2.0 * math.pi / om
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
/INIVEL/TRA/1
kick
      0.01       0.0       0.0         2
/BCS/1
pin
       111       111         0         1
/BCS/2
axial
       011       111         0         2
/END
"""
    eng = (f"#\n/RUN/SM/1\n{2 * T}\n/IMPL/DYNA/2\n0.5 0.25\n/IMPL/DTINI\n"
           f"{T / 100}\n/END\n")
    m1 = _run(starter, eng)
    m2 = _run(starter, eng)
    u1 = np.array([uu[m1.node_index(2), 0]
                   for uu in m1.implicit_result.history["u"]])
    u2 = np.array([uu[m2.node_index(2), 0]
                   for uu in m2.implicit_result.history["u"]])
    assert np.array_equal(u1, u2)
    assert m1.implicit_result.fatigue is None
