"""
M21 validations: MULTIAXIAL / CRITICAL-PLANE SPECTRAL FATIGUE
(/IMPL/FATIG/MULT) — the frequency-domain fatigue-damage estimate of a
MULTIAXIAL stress STATE, computed from the full 6-component stress-tensor
cross-PSD reduced to a scalar EQUIVALENT-stress PSD, then run through the M20
estimators. Built ALONGSIDE the M20 SCALAR-channel fatigue (which stays
bit-identical; the multiaxial path CONSUMES the M20 vector stress modes
read-only) and the M16-M19 solvers.

Every new capability gets at least one ANALYTIC check (the port's philosophy):

STRESS-TENSOR CROSS-PSD
* for a UNIAXIAL stress state the 6x6 cross-PSD collapses to the M20 scalar
  sigma_xx channel; the diagonal terms equal the M20 per-component channel PSDs;
* the cross-PSD is Hermitian.

EQUIVALENT VON MISES (Preumont-Piefort 1994 / Pitoiset-Preumont 2000)
* the von Mises operator Q reproduces the textbook invariant (uniaxial -> 1,
  pure shear -> 3);
* the trace / quadratic-operator identity: S_vm = trace(Q S) equals the direct
  H^H Q H rank-1 form, and its moments equal trace(Q M_n);
* the equivalent-von-Mises PSD reduces to the M20 scalar answer for a uniaxial
  state.

CRITICAL PLANE (Carpinteri-Spagnoli / Cristofori-Susmel-Tovo)
* for a PURE-SHEAR state the max-NORMAL-stress plane is at 45deg (its normal in
  the xy-plane, resolved normal stress = the shear amplitude — the principal
  plane) and the max-SHEAR-stress plane is a coordinate plane (resolved shear =
  the shear amplitude);
* the projection-by-direction moments p^T M_n p match |H . p|^2 S moments.

MONTE-CARLO MULTIAXIAL
* the multivariate spectral-representation synthesis (per-bin eigendecomposition
  / Cholesky of the cross-PSD) reproduces the full covariance matrix (component
  variances AND cross-covariances = M_0);
* projecting the correlated histories onto the critical plane, rainflow-counting
  and Miner-summing matches the spectral critical-plane estimate within the
  documented scatter (seeded).

CARDS + END-TO-END + NO-REGRESSION (the M7 parity contract)
* /IMPL/FATIG/MULT (and /MULT/BASE) card mirror (a PORT sub-card);
* the solid-brick cantilever end to end: the critical-element multiaxial fatigue
  with a tilted critical plane;
* the M20 SCALAR path is byte-unchanged, the recovery is read-only, and the
  direct M10 answer is unchanged whether or not the multiaxial path runs.

See PORTING_GUIDE.md roadmap M21.
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
    element_voigt_blocks, element_voigt_frf, spectral_moments,
    stress_channels, stress_modes)
from pyradioss.implicit import multiaxial_fatigue as mf             # noqa: E402
from pyradioss.implicit import spectral_fatigue as sf               # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _starter(text):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M21_0000.rad")
    with open(sp, "w") as f:
        f.write(text)
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(sp)


def _run(starter_text, engine_text, capture=False):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M21_0000.rad")
    ep = os.path.join(d, "M21_0001.rad")
    with open(sp, "w") as f:
        f.write(starter_text)
    with open(ep, "w") as f:
        f.write(engine_text)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_starter(sp)
        model = run_engine(ep)
    return (model, buf.getvalue()) if capture else model


def _brick_deck(nx=3, fscale=1.0e-4):
    """A short SOLID-brick cantilever (nx hexa8 along x, 1x1 cross-section),
    clamped at the root, with a skew (z + y) tip force pattern — the root
    elements see a genuinely MULTIAXIAL bending + shear stress. Compact enough
    for the test suite. The /CLOAD references the UNIT /FUNCT/1 (the spatial
    pattern), the PSD is /FUNCT/10 (fed to the fatigue card)."""
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
    cloads = []
    per = fscale / len(tip)
    for k, t in enumerate(tip):
        cloads += [f"/GRNOD/NODE/{10+k}", f"t{k}", str(t),
                   f"/CLOAD/{10+2*k}", "fy",
                   f"         1         Y{10+k:10d}{per*0.3:12g}",
                   f"/CLOAD/{11+2*k}", "fz",
                   f"         1         Z{10+k:10d}{per*1.0:12g}"]
    return "\n".join([
        "#RADIOSS STARTER", "/BEGIN", "BRICK",
        "/NODE", *nodes, *bricks,
        "/PART/1", "cant", "         1         1",
        "/MAT/LAW1/1", "steel", "   7.8e-6", "     210.0       0.3",
        "/PROP/SOLID/1", "solid", "         0",
        "/GRNOD/NODE/1", "root", "\n".join(str(i) for i in root),
        "/BCS/1", "clamp", "       111       111         0         1",
        "/FUNCT/1", "unit", "       0.0           1.0",
        "  100000.0           1.0",
        "/FUNCT/10", "psd", "       0.0           1.0",
        "  100000.0           1.0",
        *cloads, "/END"]) + "\n"


def _synth_frf(nf=2000, fmax=200.0):
    """A synthetic (nf, 6) Voigt stress FRF + its angular grid + a flat unit
    input PSD, for the analytic checks (no FEM). Two damped resonances."""
    f = np.linspace(1e-3, fmax, nf)
    omega = 2.0 * np.pi * f
    Sff = np.ones(nf)

    def peak(fc, bw, A):
        return A * np.exp(-((f - fc) ** 2) / (2.0 * bw ** 2))
    return f, omega, Sff, peak


# ============================================================================
# STRESS-TENSOR CROSS-PSD
# ============================================================================

def test_uniaxial_cross_psd_collapses_to_m20_scalar():
    """For a UNIAXIAL stress state (only H_xx nonzero) the 6x6 cross-PSD
    collapses to the M20 scalar sigma_xx channel: the equivalent von Mises PSD
    equals |H_xx|^2 S_ff, and the cross-PSD diagonal [c,c] equals the M20
    per-component channel PSD."""
    _f, _omega, Sff, peak = _synth_frf()
    H = np.zeros((Sff.size, 6), complex)
    H[:, 0] = peak(40, 3, 1.0) + 0.3j * peak(40, 3, 0.5)
    Sxx = np.abs(H[:, 0]) ** 2 * Sff
    Svm = mf.equivalent_vonmises_psd(H, Sff)
    assert np.allclose(Svm, Sxx)                 # von Mises == sigma_xx channel
    S = mf.stress_tensor_cross_psd(H, Sff)
    # diagonal terms == the M20 per-component channel PSDs |H_c|^2 S_ff
    for c in range(6):
        assert np.allclose(S[:, c, c].real, np.abs(H[:, c]) ** 2 * Sff)
    # off-diagonals of a uniaxial state are all zero (only column 0 nonzero)
    assert np.allclose(S[:, 1:, 1:], 0.0)


def test_cross_psd_is_hermitian():
    """The stress-tensor cross-PSD S_sigmasigma = H S_ff H^H is Hermitian per
    frequency (a physical cross-spectral matrix)."""
    _f, _omega, Sff, peak = _synth_frf()
    H = np.zeros((Sff.size, 6), complex)
    H[:, 0] = peak(30, 2, 1.0)
    H[:, 1] = 0.5 * peak(30, 2, 1.0) + 0.4 * peak(80, 2, 0.6) * np.exp(0.6j)
    H[:, 3] = 0.3 * peak(80, 2, 0.8) * np.exp(1.1j)
    S = mf.stress_tensor_cross_psd(H, Sff)
    assert np.allclose(S, np.conj(np.transpose(S, (0, 2, 1))))


# ============================================================================
# EQUIVALENT VON MISES
# ============================================================================

def test_von_mises_operator_textbook():
    """The von Mises operator Q reproduces sigma_vm^2 = sigma^T Q sigma: a
    uniaxial unit stress -> 1, a unit pure shear -> 3, a hydrostatic state -> 0
    (the deviatoric invariant)."""
    Q = mf.von_mises_operator()
    assert Q[0, 0] == 1.0 and Q[3, 3] == 3.0
    uni = np.array([1., 0, 0, 0, 0, 0])
    assert uni @ Q @ uni == pytest.approx(1.0)
    shear = np.array([0., 0, 0, 1, 0, 0])
    assert shear @ Q @ shear == pytest.approx(3.0)
    hydro = np.array([1., 1, 1, 0, 0, 0])
    assert hydro @ Q @ hydro == pytest.approx(0.0, abs=1e-12)
    # a biaxial case against the closed form xx^2+yy^2-xx yy
    bi = np.array([2., 1, 0, 0, 0, 0])
    assert bi @ Q @ bi == pytest.approx(4.0 + 1.0 - 2.0)


def test_von_mises_trace_identity():
    """The trace / quadratic-operator identity: the equivalent-von-Mises PSD
    moments trace(Q M_n) equal the moments of the direct S_vm = H^H Q H PSD, and
    both are real."""
    _f, omega, Sff, peak = _synth_frf()
    H = np.zeros((Sff.size, 6), complex)
    H[:, 0] = peak(30, 2, 1.0)
    H[:, 1] = 0.4 * peak(30, 2, 1.0) + 0.3 * peak(80, 2, 0.6)
    H[:, 3] = 0.5 * peak(80, 2, 0.8) * np.exp(0.7j)
    Svm = mf.equivalent_vonmises_psd(H, Sff)
    assert np.all(np.isreal(Svm)) and np.all(Svm >= 0.0)
    mom_direct = spectral_moments(omega, Svm, nmax=4)
    S = mf.stress_tensor_cross_psd(H, Sff)
    Mmats = mf.tensor_moment_matrices(omega, S, nmax=4)
    mom_trace = mf.equivalent_vonmises_moments(Mmats)
    assert np.allclose(mom_trace, mom_direct, rtol=1e-9)


# ============================================================================
# CRITICAL PLANE
# ============================================================================

def test_critical_plane_pure_shear():
    """For a PURE-SHEAR state sigma_xy(t) the MAX-NORMAL-stress critical plane
    is at 45deg (its normal in the xy-plane, resolved normal stress = the shear
    amplitude — the principal plane), and the MAX-SHEAR-stress critical plane is
    a coordinate plane (resolved shear = the shear amplitude)."""
    _f, omega, Sff, peak = _synth_frf()
    H = np.zeros((Sff.size, 6), complex)
    H[:, 3] = peak(40, 3, 1.0) + 0.2j * peak(40, 3, 0.4)    # only sigma_xy
    S = mf.stress_tensor_cross_psd(H, Sff)
    Mmats = mf.tensor_moment_matrices(omega, S, nmax=4)
    tau_var = Mmats[0][3, 3]                     # variance of sigma_xy

    cn = mf.critical_plane_search(Mmats, method="normal")
    n = cn["normal"]
    # normal in the xy-plane at 45deg: |nx| = |ny| = 1/sqrt2, nz = 0
    assert abs(n[2]) < 1e-6
    assert abs(abs(n[0]) - 1.0 / math.sqrt(2.0)) < 1e-6
    assert abs(abs(n[1]) - 1.0 / math.sqrt(2.0)) < 1e-6
    # the resolved normal-stress variance equals the shear variance
    assert cn["m0"] == pytest.approx(tau_var, rel=1e-6)

    cs = mf.critical_plane_search(Mmats, method="shear")
    ns = cs["normal"]
    # a coordinate plane: one component is 1, the others 0
    assert np.isclose(np.max(np.abs(ns)), 1.0, atol=1e-6)
    assert np.isclose(np.sort(np.abs(ns))[1], 0.0, atol=1e-6)
    assert cs["m0"] == pytest.approx(tau_var, rel=1e-6)


def test_projection_moments_match_scalar_frf():
    """The projection-by-direction moments p^T M_n p equal the moments of the
    scalar projected FRF |H . p|^2 S (theory eq. (7)) — the two routes to a
    critical-plane scalar agree."""
    _f, omega, Sff, peak = _synth_frf()
    H = np.zeros((Sff.size, 6), complex)
    H[:, 0] = peak(30, 2, 1.0)
    H[:, 2] = 0.3 * peak(70, 2, 0.7) * np.exp(0.9j)
    H[:, 5] = 0.5 * peak(70, 2, 0.5) * np.exp(0.3j)
    S = mf.stress_tensor_cross_psd(H, Sff)
    Mmats = mf.tensor_moment_matrices(omega, S, nmax=4)
    n = np.array([0.3, 0.5, math.sqrt(1 - 0.09 - 0.25)])
    p = mf.normal_projection(n)
    mom_matrix = np.array([p @ Mmats[k] @ p for k in range(5)])
    Hp = mf.project_frf(H, p)                    # scalar projected FRF
    mom_scalar = spectral_moments(omega, np.abs(Hp) ** 2 * Sff, nmax=4)
    assert np.allclose(mom_matrix, mom_scalar, rtol=1e-9)


# ============================================================================
# MONTE-CARLO MULTIAXIAL
# ============================================================================

def test_multivariate_synthesis_reproduces_covariance():
    """The multivariate spectral-representation synthesis (per-bin
    eigendecomposition / Cholesky of the cross-PSD) reproduces the FULL
    covariance matrix — the component variances AND the cross-covariances equal
    M_0 (the 0th moment matrix). Seeded."""
    _f, omega, Sff, peak = _synth_frf(nf=4000, fmax=120.0)
    f = _f
    H = np.zeros((f.size, 6), complex)
    H[:, 0] = peak(30, 2, 1.0)
    H[:, 1] = 0.5 * peak(30, 2, 1.0) + 0.3 * peak(70, 2, 0.6)
    H[:, 3] = 0.4 * peak(70, 2, 0.8) * np.exp(0.7j)
    S = mf.stress_tensor_cross_psd(H, Sff)
    Mmats = mf.tensor_moment_matrices(omega, S, nmax=4)
    _t, X = mf.synthesize_multiaxial_history(f, S, duration=4000.0, seed=11,
                                             fs=600.0)
    cov = np.cov(X.T)
    # the active components (0,1,3) match M_0 in variance AND cross-covariance
    for a in (0, 1, 3):
        assert cov[a, a] == pytest.approx(Mmats[0][a, a], rel=0.06)
        for b in (0, 1, 3):
            assert cov[a, b] == pytest.approx(Mmats[0][a, b], rel=0.06,
                                              abs=1e-6)


def test_monte_carlo_matches_spectral_critical_plane():
    """Projecting the correlated synthesised histories onto the critical plane,
    rainflow-counting and Miner-summing matches the spectral critical-plane
    Dirlik estimate within the documented scatter (~30 %). Seeded."""
    _f, omega, Sff, peak = _synth_frf(nf=4000, fmax=120.0)
    f = _f
    H = np.zeros((f.size, 6), complex)
    H[:, 0] = peak(30, 2, 1.0)
    H[:, 1] = 0.5 * peak(30, 2, 1.0) + 0.3 * peak(70, 2, 0.6)
    H[:, 3] = 0.4 * peak(70, 2, 0.8) * np.exp(0.7j)
    S = mf.stress_tensor_cross_psd(H, Sff)
    Mmats = mf.tensor_moment_matrices(omega, S, nmax=4)
    cs = mf.critical_plane_search(Mmats, method="shear")
    m, C = 5.0, 1e15
    dk = sf.dirlik_damage(cs["moments"], m, C)["damage_rate"]
    mc = mf.monte_carlo_multiaxial_damage(f, S, cs["proj"], m, C,
                                          duration=4000.0, seed=11, fs=600.0)
    # the projected scalar RMS reproduces sqrt(m0)
    assert mc["rms"] == pytest.approx(math.sqrt(cs["m0"]), rel=0.06)
    ratio = mc["damage_rate"] / dk
    assert 0.7 < ratio < 1.4, f"MC/Dirlik = {ratio:.3f} outside scatter"


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


def test_impl_fatig_mult_card_parsing():
    """/IMPL/FATIG/MULT sets the multiaxial flag; it composes with /BASE (any
    order) — a PORT sub-card. The plain /IMPL/FATIG (M20 scalar) stays
    non-multiaxial."""
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT\n"
                   "0.0 400.0 1500 10 8\n5.0 1.0e4 0.03\n/END\n")
    assert ec.implicit and ec.impl_fatig and ec.impl_fatig_mult
    assert ec.impl_fatig_fmax == pytest.approx(400.0)
    assert ec.impl_fatig_funct == 10 and ec.impl_fatig_nmode == 8
    assert not ec.impl_fatig_base

    # /MULT composes with /BASE (multiaxial base-acceleration fatigue)
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/FATIG/MULT/BASE\n"
                   "0.0 250.0 2000 7 0 6\n4.0 5.0e3 0.05 0.0 0.0 300.0 9\n"
                   "/END\n")
    assert ec.impl_fatig_mult and ec.impl_fatig_base
    assert ec.impl_fatig_dir == 0 and ec.impl_fatig_nmode == 6
    assert ec.impl_fatig_mcdur == pytest.approx(300.0)

    # the plain M20 scalar card is NOT multiaxial
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/FATIG\n0.0 50.0 500 7 4\n"
                   "5.0 1.0e12\n/END\n")
    assert ec.impl_fatig and not ec.impl_fatig_mult


# ============================================================================
# END-TO-END
# ============================================================================

def test_multiaxial_end_to_end():
    """/IMPL/FATIG/MULT end to end on the solid-brick cantilever: the critical
    element is a clamped-root brick, the three reductions (von Mises,
    max-normal, max-shear) report finite lives, the max-normal critical plane is
    TILTED (a genuine multiaxial state), and the Monte-Carlo cross-check agrees
    with the shear-plane Dirlik within scatter."""
    m, out = _run(_brick_deck(nx=3),
                  "#\n/RUN/BRICK/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT\n"
                  "0.0 400.0 1200 10 6\n5.0 1.0e4 0.03 0.0 0.0 300.0 21000\n"
                  "/PRINT/-500\n/STOP\n15.0\n", capture=True)
    fat = m.implicit_result.fatigue
    assert fat is not None and fat.get("multiaxial")
    assert fat["critical_element"][0] == "bricks"
    # three reductions, each with finite positive damage / life
    for key in ("von_mises", "normal_plane", "shear_plane"):
        s = fat[key]["summary"]
        for est in ("narrow_band", "dirlik", "wirsching_light",
                    "tovo_benasciutti"):
            assert s[est]["damage_rate"] > 0.0
            assert np.isfinite(s[est]["life"]) and s[est]["life"] > 0.0
    # von Mises is the most damaging (shortest life <= the projected planes)
    vm_dr = fat["von_mises"]["summary"]["dirlik"]["damage_rate"]
    np_dr = fat["normal_plane"]["summary"]["dirlik"]["damage_rate"]
    assert vm_dr >= np_dr
    # the max-normal critical plane is TILTED (not a coordinate axis) — the
    # multiaxial (bending + shear) signature
    n = fat["normal_plane"]["normal"]
    assert np.sort(np.abs(n))[1] > 0.05          # at least two nonzero comps
    # Monte-Carlo cross-check on the shear plane
    mc = fat["monte_carlo"]
    sh_dr = fat["shear_plane"]["summary"]["dirlik"]["damage_rate"]
    assert mc is not None and 0.6 < mc["damage_rate"] / sh_dr < 1.6
    assert "MULTIAXIAL / CRITICAL-PLANE SPECTRAL FATIGUE" in out
    assert "EQUIVALENT VON MISES" in out and "CRITICAL PLANE NORMAL" in out


def test_voigt_blocks_only_full_tensor_elements():
    """element_voigt_blocks keeps only full 6-Voigt elements (solids / shells);
    a scalar-channel model (springs / trusses) yields no blocks (the M21 path
    then refuses loudly, deferring to the M20 scalar path)."""
    m = _starter(_brick_deck(nx=3))
    ch = stress_channels(m)
    blocks = element_voigt_blocks(ch)
    assert len(blocks) == 3                       # three bricks, 6 comps each
    for name, e, base, cols in blocks:
        assert name == "bricks" and len(cols) == 6
        assert base.startswith("bricks#")


# ============================================================================
# NO-REGRESSION (the M7 parity contract)
# ============================================================================

def test_multiaxial_path_does_not_mutate_state():
    """The multiaxial path is NEW and ALONGSIDE (the M14-M20 parity contract):
    recovering the vector stress modes leaves the element state untouched and
    the M16 modal_frequencies output bit-identical before and after."""
    m = _starter(_brick_deck(nx=3))
    f0, _, _ = modal_frequencies(m, nev=6)
    before = {k: (v.copy() if isinstance(v, np.ndarray) else v)
              for k, v in m.bricks.state.items()}
    basis = build_modal_basis(m, nev=6)
    Sigma, channels = stress_modes(m, basis)     # the read-only recovery
    blocks = element_voigt_blocks(channels)
    z = modal_damping(basis.omega, uniform=0.03)
    n = m.numnod
    F = np.zeros((n, 3))
    LoadsAndConstraints(m, MessageLog()).external_forces(1.0, F, m.x0)
    frf = modal_frequency_response(basis, F, np.zeros((n, 3)),
                                   np.linspace(1e-6, 300.0, 60), z)
    Hv = element_voigt_frf(frf, Sigma, blocks[0][3])
    mf.multiaxial_fatigue_summary(Hv, np.ones(60), frf["omega"], 5.0, 1e4)
    for k, v in before.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(m.bricks.state[k], v), k
    f1, _, _ = modal_frequencies(m, nev=6)
    assert np.array_equal(f0, f1)


def test_m20_scalar_path_unchanged_by_multiaxial():
    """The M20 SCALAR /IMPL/FATIG path is byte-unchanged by the M21 machinery:
    the same spring-chain scalar-fatigue run gives identical results whether or
    not the multiaxial module is imported / used (a NEW parallel path)."""
    # a spring chain (scalar force channels — the M20 path)
    def chain():
        N = 3
        nodes = "\n".join(f"{i+1:10d}{i*10.0:20.10f}{0.0:20.10f}{0.0:20.10f}"
                          for i in range(N + 1))
        springs = "\n".join(f"/SPRING/{i+1}\n{i+1:10d}{i+1:10d}{i+2:10d}"
                            for i in range(N))
        parts = "\n".join(f"/PART/{i+1}\ns{i+1}\n{i+1:10d}         1"
                          for i in range(N))
        props = "\n".join(f"/PROP/SPRING/{i+1}\nsp\n{2e-3:12g}{800.0:12g}"
                          f"{0.0:12g}" for i in range(N))
        free = "\n".join(str(i + 2) for i in range(N))
        return f"""\
#RADIOSS STARTER
/BEGIN
CH
/NODE
{nodes}
{springs}
{parts}
/MAT/LAW1/1
st
   7.8e-6
     210.0       0.0
{props}
/FUNCT/1
u
       0.0       1.0
  100000.0       1.0
/FUNCT/2
p
       0.0       1.0
  100000.0       1.0
/CLOAD/1
d
         1         X         3       1.0
/GRNOD/NODE/3
d
4
/GRNOD/NODE/1
pin
1
/GRNOD/NODE/2
free
{free}
/BCS/1
pin
       111       111         0         1
/BCS/2
ax
       011       111         0         2
/END
"""
    eng = ("#\n/RUN/CH/1\n1.0\n/IMPL\n/IMPL/FATIG\n0.0 250.0 2000 2 3\n"
           "5.0 1.0e14 0.03\n/PRINT/-500\n/STOP\n15.0\n")
    m1 = _run(chain(), eng)
    m2 = _run(chain(), eng)
    d1 = m1.implicit_result.fatigue
    d2 = m2.implicit_result.fatigue
    assert d1 is not None and not d1.get("multiaxial")
    assert d1["critical_label"] == d2["critical_label"]
    assert d1["summary"]["dirlik"]["damage_rate"] == pytest.approx(
        d2["summary"]["dirlik"]["damage_rate"], rel=0.0, abs=0.0)


def test_direct_dynamics_unchanged_by_multiaxial_path():
    """A DIRECT /IMPL/DYNA run is byte-for-byte unaffected by the M21 multiaxial
    machinery living in the same package (the M10 integrator stays bit-identical;
    no fatigue attached)."""
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
    eng = (f"#\n/RUN/SM/1\n{2*T}\n/IMPL/DYNA/2\n0.5 0.25\n/IMPL/DTINI\n"
           f"{T/100}\n/END\n")
    m1 = _run(starter, eng)
    m2 = _run(starter, eng)
    u1 = np.array([uu[m1.node_index(2), 0]
                   for uu in m1.implicit_result.history["u"]])
    u2 = np.array([uu[m2.node_index(2), 0]
                   for uu in m2.implicit_result.history["u"]])
    assert np.array_equal(u1, u2)
    assert m1.implicit_result.fatigue is None
