"""
M22 validations: NON-PROPORTIONAL MULTIAXIAL FATIGUE (/IMPL/FATIG/MULT/NPROP) —
the CRITICAL-PLANE, TIME-DOMAIN, PATH-COUNTING fatigue-damage estimate of a
multiaxial stress state whose principal axes ROTATE (a non-proportional load
path). Built ALONGSIDE the M21 MULTIAXIAL SPECTRAL path (which stays
bit-identical whether or not NPROP runs; the non-proportional path CONSUMES the
M21 multivariate synthesiser + candidate-plane machinery read-only) and the
M16-M20 solvers.

Every new capability gets at least one ANALYTIC check (the port's philosophy):

SHEAR-PATH AMPLITUDE OPERATORS (Papadopoulos MCC / longest chord / Mamiya MRH)
* for a PROPORTIONAL (in-phase) line path all three collapse to the scalar M21
  amplitude and F_np = 0;
* for a CIRCULAR (90-deg-out-of-phase, equal-amplitude) path the MCC radius = the
  component amplitude, the longest chord = the diameter, the MRH = r*sqrt(2), and
  F_np = 1 (closed forms);
* an ELLIPSE closed form: MRH = sqrt(p^2+q^2), MCC = the major semi-axis, F_np =
  q/p.

CRITICAL-PLANE TIME-DOMAIN DAMAGE (Findley / Fatemi-Socie)
* for PROPORTIONAL loading the path-counting (shear-path) damage on the critical
  plane reduces to the M21 max-shear critical-plane rainflow within the seeded
  Monte-Carlo scatter;
* for a 90-deg-OUT-OF-PHASE case the non-proportional path-counting damage is
  HIGHER than the scalar projection (the M21 spectral method) at the same channel
  amplitudes — the extra damage the projection misses — by ~2^(m/2), and
  F_np ~ 1;
* a hand check of the MCC radius / MRH / Findley damage on a synthetic two-channel
  sinusoid.

CARDS + END-TO-END + NO-REGRESSION (the M7 parity contract)
* /IMPL/FATIG/MULT/NPROP card mirror (a PORT sub-flag, composes in any order);
* the solid-brick cantilever end to end: the non-proportional critical-plane
  reductions reported ALONGSIDE the M21 spectral numbers;
* the M21 spectral answer is byte-identical whether or not NPROP runs, the
  recovery is read-only, and the direct M10 answer is unchanged.

See PORTING_GUIDE.md roadmap M22.
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
    element_voigt_blocks, element_voigt_frf, stress_channels, stress_modes)
from pyradioss.implicit import multiaxial_fatigue as mf             # noqa: E402
from pyradioss.implicit import nonproportional_fatigue as npf       # noqa: E402
from pyradioss.implicit import spectral_fatigue as sf               # noqa: E402

# reuse the M21 brick deck generator
from tests.test_m21_multiaxfatig import _brick_deck                 # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _run(starter_text, engine_text, capture=False):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M22_0000.rad")
    ep = os.path.join(d, "M22_0001.rad")
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
    sp = os.path.join(d, "M22_0000.rad")
    with open(sp, "w") as f:
        f.write(text)
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(sp)


def _circle(r, nt=4000):
    t = np.linspace(0.0, 2.0 * math.pi, nt, endpoint=False)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def _line(A, nt=4000, direction=(1.0, 1.0)):
    t = np.linspace(0.0, 2.0 * math.pi, nt, endpoint=False)
    u = np.asarray(direction, float)
    u = u / np.linalg.norm(u)
    return np.outer(A * np.cos(t), u)


def _ellipse(p, q, nt=4000):
    t = np.linspace(0.0, 2.0 * math.pi, nt, endpoint=False)
    return np.column_stack([p * np.cos(t), q * np.sin(t)])


# ============================================================================
# SHEAR-PATH AMPLITUDE OPERATORS
# ============================================================================

def test_proportional_line_collapses_all_operators():
    """For a PROPORTIONAL (in-phase) LINE path of half-length A the three
    shear-amplitude operators (MCC / longest chord / MRH) all collapse to the
    scalar amplitude A, and the non-proportionality factor F_np = 0."""
    A = 3.0
    P = _line(A)
    assert npf.shear_amplitude(P, "mcc") == pytest.approx(A, rel=1e-6)
    assert 0.5 * npf.longest_chord(P) == pytest.approx(A, rel=1e-6)
    assert npf.shear_amplitude(P, "mrh") == pytest.approx(A, rel=1e-3)
    assert npf.nonproportionality_factor(P) == pytest.approx(0.0, abs=1e-6)


def test_circular_path_closed_forms():
    """For a CIRCULAR (90-deg-out-of-phase, equal-amplitude) path of radius r:
    the MCC radius = r (the component amplitude), the longest chord = 2 r (the
    diameter), the MRH = r*sqrt(2), and F_np = 1 (the closed forms)."""
    r = 2.0
    P = _circle(r)
    assert npf.shear_amplitude(P, "mcc") == pytest.approx(r, rel=1e-4)
    assert npf.longest_chord(P) == pytest.approx(2.0 * r, rel=1e-4)   # diameter
    assert 0.5 * npf.longest_chord(P) == pytest.approx(r, rel=1e-4)   # amplitude
    assert npf.shear_amplitude(P, "mrh") == pytest.approx(
        r * math.sqrt(2.0), rel=1e-3)
    assert npf.nonproportionality_factor(P) == pytest.approx(1.0, rel=1e-6)


def test_ellipse_amplitude_closed_forms():
    """An ELLIPSE of semi-axes p > q: the MRH = sqrt(p^2+q^2) (the prismatic-hull
    closed form, constant over rotation), the MCC radius = the major semi-axis p,
    and F_np = q/p (the aspect ratio)."""
    p, q = 4.0, 1.5
    P = _ellipse(p, q)
    assert npf.shear_amplitude(P, "mrh") == pytest.approx(math.hypot(p, q),
                                                          rel=2e-3)
    assert npf.shear_amplitude(P, "mcc") == pytest.approx(p, rel=1e-3)
    assert npf.nonproportionality_factor(P) == pytest.approx(q / p, rel=1e-3)


def test_min_circumscribed_circle_offcenter():
    """Papadopoulos MCC on an OFF-CENTRE circle (a mean shear offset): the radius
    is still r and the centre is the offset — the amplitude is mean-independent
    (Papadopoulos separates the shear amplitude from its mean value)."""
    r = 1.7
    c0 = np.array([5.0, -3.0])
    P = _circle(r) + c0
    c, rad = npf.min_circumscribed_circle(P)
    assert rad == pytest.approx(r, rel=1e-3)
    assert np.allclose(c, c0, atol=2e-2)


# ============================================================================
# CRITICAL-PLANE TIME-DOMAIN DAMAGE
# ============================================================================

def _proportional_cross_psd(nf=1500, fmax=120.0):
    """A synthetic PROPORTIONAL multiaxial cross-PSD: a single real modal shape
    (H[:,c] = shape(Omega) * v[c]) so every stress component scales together —
    the principal axes are fixed (F_np = 0). Two real resonances in the shape."""
    f = np.linspace(1e-3, fmax, nf)
    omega = 2.0 * math.pi * f
    Sff = np.ones(nf)
    shape = (np.exp(-((f - 30) ** 2) / (2 * 2.5 ** 2))
             + 0.6 * np.exp(-((f - 70) ** 2) / (2 * 2.5 ** 2)))
    v = np.array([1.0, 0.3, 0.0, 0.5, 0.0, 0.2])       # a multiaxial voigt shape
    H = shape[:, None] * v[None, :]                     # (nf, 6) real -> in phase
    Scross = mf.stress_tensor_cross_psd(H, Sff)
    return f, omega, Sff, H, Scross


def test_proportional_reduces_to_m21_maxshear():
    """For PROPORTIONAL loading the M22 path-counting (shear-path) damage on the
    critical plane reduces to the M21 max-shear critical-plane rainflow (the same
    seeded synthesised history projected onto the max-shear plane), within the
    Monte-Carlo scatter — the non-proportional machinery adds no spurious damage
    to a proportional path (g = 1, F_np = 0)."""
    f, omega, Sff, H, Scross = _proportional_cross_psd()
    m, C = 5.0, 1e15
    dur, seed = 2000.0, 7

    # M21 max-shear critical plane + its Monte-Carlo rainflow (the reference)
    Mmats = mf.tensor_moment_matrices(omega, Scross, nmax=4)
    cs = mf.critical_plane_search(Mmats, method="shear")
    mc21 = mf.monte_carlo_multiaxial_damage(f, Scross, cs["proj"], m, C, dur,
                                            seed)

    # M22 shear-path critical-plane damage on the SAME seeded synthesised history
    t, X = mf.synthesize_multiaxial_history(f, Scross, dur, seed)
    d22 = npf.critical_plane_damage(X, m, C, model="shear_path",
                                    duration=t[-1] - t[0])
    # a proportional path: F_np ~ 0 and g ~ 1 on the critical plane
    assert d22["F_np"] < 0.15
    assert d22["g"] == pytest.approx(1.0, abs=0.1)
    ratio = d22["damage_rate"] / mc21["damage_rate"]
    assert 0.6 < ratio < 1.6, f"M22/M21 = {ratio:.3f} outside scatter"


def test_nonproportional_higher_than_scalar_projection():
    """For a 90-deg-OUT-OF-PHASE (rotating) shear path the NON-PROPORTIONAL
    path-counting damage is HIGHER than the SCALAR-projection damage (the M21
    spectral method's linear projection) at the SAME channel amplitudes — the
    extra damage the projection misses (the whole point of a non-proportional
    criterion) — by ~2^(m/2), with the path factor g ~ sqrt(2) and F_np ~ 1."""
    nt = 40000
    tt = np.linspace(0.0, 400.0, nt, endpoint=False)
    w = 2.0 * math.pi * 1.0
    tau0 = 1.0
    # two equal-amplitude shear channels on the x-plane, 90 deg out of phase:
    # sigma_xy (col 3) and sigma_zx (col 5) -> a rotating (circular) shear path
    X = np.zeros((nt, 6))
    X[:, 3] = tau0 * np.sin(w * tt)
    X[:, 5] = tau0 * np.cos(w * tt)
    m, C = 5.0, 1e12

    d = npf.critical_plane_damage(X, m, C, model="shear_path", duration=400.0)
    # the scalar projection (M21 max-shear): rainflow the dominant scalar, g = 1
    P, _ = npf.resolved_shear_path(X, d["normal"])
    tau_dom, _ = npf._dominant_shear_scalar(P)
    r, cnt = sf.rainflow_count(tau_dom)
    proj_dr = float(np.sum(cnt * r ** m) / C) / 400.0

    assert d["F_np"] == pytest.approx(1.0, rel=0.05)
    assert d["g"] == pytest.approx(math.sqrt(2.0), rel=0.05)
    ratio = d["damage_rate"] / proj_dr
    # the MRH-scaled path counts 2^(m/2) more damage than the scalar projection
    assert ratio == pytest.approx(2.0 ** (m / 2.0), rel=0.1)
    assert ratio > 1.5


def test_nonproportional_line_vs_circle_at_equal_channel_amplitude():
    """The non-proportional (90-deg circle) path is more damaging than the
    proportional (single-channel line) path at the SAME per-channel amplitude:
    the rotating path counts g = sqrt(2) larger shear ranges (2^(m/2) more
    damage), the closed-form signature of non-proportional damage."""
    nt = 40000
    tt = np.linspace(0.0, 400.0, nt, endpoint=False)
    w = 2.0 * math.pi * 1.0
    tau0, m, C = 1.0, 5.0, 1e12
    line = np.zeros((nt, 6))
    line[:, 3] = tau0 * np.sin(w * tt)                  # one channel -> line
    circ = np.zeros((nt, 6))
    circ[:, 3] = tau0 * np.sin(w * tt)
    circ[:, 5] = tau0 * np.cos(w * tt)                  # 90 deg -> circle
    dl = npf.critical_plane_damage(line, m, C, model="shear_path",
                                   duration=400.0)
    dc = npf.critical_plane_damage(circ, m, C, model="shear_path",
                                   duration=400.0)
    assert dl["F_np"] < 0.05 and dc["F_np"] > 0.95
    ratio = dc["damage_rate"] / dl["damage_rate"]
    assert ratio == pytest.approx(2.0 ** (m / 2.0), rel=0.1)


def test_findley_fatemi_socie_hand_check():
    """A hand check of the MCC radius / MRH / Findley / Fatemi-Socie damage on a
    synthetic two-channel sinusoid with a constant normal stress. A proportional
    LINE shear (amplitude A) with a constant normal sigma_n = sn: sigma_n,max =
    sn, tau_a = A (MCC = MRH = A), g = 1, and rainflow of the N-cycle sinusoid
    gives N cycles of range 2A. Findley damage = N (2A + 2 k sn)^m / C; FS damage
    = N (2A (1 + k sn/sigma_y))^m / C — both hand-computable."""
    A, sn, k, sigy = 5.0, 2.0, 0.3, 4.0
    m, C = 4.0, 1e10
    ncyc = 50
    nt = ncyc * 200
    tt = np.linspace(0.0, ncyc * 2.0 * math.pi, nt, endpoint=False)
    P = np.column_stack([A * np.cos(tt), np.zeros(nt)])   # a line, half-length A
    sigma_n = np.full(nt, sn)

    dF = npf.plane_damage(P, sigma_n, m, C, model="findley", k=k, duration=1.0)
    dS = npf.plane_damage(P, sigma_n, m, C, model="fatemi_socie", k=k,
                          sigma_y=sigy, duration=1.0)
    # geometry: tau_a = A (MCC = MRH), sigma_n,max = sn, g = 1
    assert npf.shear_amplitude(P, "mcc") == pytest.approx(A, rel=1e-4)
    assert dF["sn_max"] == pytest.approx(sn)
    assert dF["tau_a"] == pytest.approx(A, rel=1e-3)
    assert dF["g"] == pytest.approx(1.0, rel=1e-3)
    # ncyc full cycles of range 2A (a clean sinusoid; allow +/-1 for the residue)
    assert dF["ncycles"] == pytest.approx(ncyc, abs=1.0)
    # Findley: N (2A + 2 k sn)^m / C
    D_find = dF["ncycles"] * (2.0 * A + 2.0 * k * sn) ** m / C
    assert dF["damage"] == pytest.approx(D_find, rel=0.02)
    # Fatemi-Socie: N (2A (1 + k sn/sigma_y))^m / C
    D_fs = dF["ncycles"] * (2.0 * A * (1.0 + k * sn / sigy)) ** m / C
    assert dS["damage"] == pytest.approx(D_fs, rel=0.02)


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


def test_impl_fatig_nprop_card_parsing():
    """/IMPL/FATIG/MULT/NPROP sets the non-proportional flag AND implies the
    multiaxial flag; it accepts the optional k / sigma_y on line 2 and composes
    with the other sub-keywords in any order — a PORT sub-flag."""
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/NPROP\n"
                   "0.0 400.0 1200 10 6\n5.0 1.0e4 0.03 0.0 0.0 300.0 21000 "
                   "0.25 2.0\n/END\n")
    assert ec.implicit and ec.impl_fatig
    assert ec.impl_fatig_mult and ec.impl_fatig_nprop
    assert ec.impl_fatig_k == pytest.approx(0.25)
    assert ec.impl_fatig_sigy == pytest.approx(2.0)
    assert ec.impl_fatig_mcdur == pytest.approx(300.0)
    assert ec.impl_fatig_seed == 21000

    # NPROP implies MULT even without MULT spelled out, any order, + BASE
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/FATIG/NPROP/BASE\n"
                   "0.0 250 2000 7 0 6\n4.0 5e3 0.05 0 0 300 9\n/END\n")
    assert ec.impl_fatig_nprop and ec.impl_fatig_mult and ec.impl_fatig_base
    assert ec.impl_fatig_k == pytest.approx(0.3)     # default

    # the plain M21 MULT card is NOT non-proportional
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/FATIG/MULT\n0.0 400 1200 10 6\n"
                   "5.0 1.0e4 0.03\n/END\n")
    assert ec.impl_fatig_mult and not ec.impl_fatig_nprop


# ============================================================================
# END-TO-END
# ============================================================================

_NPROP_ENGINE = (
    "#\n/RUN/BRICK/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/NPROP\n"
    "0.0 400.0 500 10 6\n5.0 1.0e4 0.03 0.0 0.0 15.0 21000 0.3 0.5\n"
    "/PRINT/-500\n/STOP\n15.0\n")


def test_nonproportional_end_to_end():
    """/IMPL/FATIG/MULT/NPROP end to end on the solid-brick cantilever: the
    non-proportional critical-plane path-counting result is stored on
    ``fatigue['nprop_result']`` ALONGSIDE the M21 spectral reductions, with the
    three models (Findley / Fatemi-Socie / shear-path) reporting finite lives, a
    reported F_np, and the shear-path amplitude comparison (MCC / chord / MRH),
    and the listing carries both the M21 and the M22 blocks."""
    m, out = _run(_brick_deck(nx=3), _NPROP_ENGINE, capture=True)
    fat = m.implicit_result.fatigue
    assert fat is not None and fat.get("multiaxial") and fat.get(
        "nonproportional")
    npr = fat["nprop_result"]
    assert npr is not None
    # the amplitude comparison: MCC <= chord/2 <= MRH always (a rotating path
    # only grows through the hull)
    amp = npr["amplitudes"]
    assert amp["mcc"] <= amp["mrh"] + 1e-12
    assert 0.0 <= amp["F_np"] <= 1.0
    for mdl in ("findley", "fatemi_socie", "shear_path"):
        r = npr[mdl]
        assert r["damage_rate"] > 0.0
        assert np.isfinite(r["life"]) and r["life"] > 0.0
        assert abs(np.linalg.norm(r["normal"]) - 1.0) < 1e-6
    # the M21 spectral reductions are STILL present (M22 runs alongside)
    for key in ("von_mises", "normal_plane", "shear_plane"):
        assert fat[key]["summary"]["dirlik"]["damage_rate"] > 0.0
    # the listing carries both blocks
    assert "MULTIAXIAL / CRITICAL-PLANE SPECTRAL FATIGUE" in out
    assert "NON-PROPORTIONAL MULTIAXIAL FATIGUE" in out
    assert "FINDLEY 1959" in out and "SHEAR-PATH AMPLITUDE" in out


# ============================================================================
# NO-REGRESSION (the M7 parity contract)
# ============================================================================

def test_m21_spectral_byte_identical_with_without_nprop():
    """The M21 spectral answer (von Mises / max-normal / max-shear critical
    plane) is BYTE-IDENTICAL whether or not the M22 non-proportional path runs —
    the non-proportional path is NEW and ALONGSIDE, never mutating the M21
    reductions (the parity contract)."""
    deck = _brick_deck(nx=3)
    eng_m21 = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT\n"
               "0.0 400.0 500 10 6\n5.0 1.0e4 0.03\n/PRINT/-500\n/STOP\n15.0\n")
    m21 = _run(deck, eng_m21)
    m22 = _run(deck, _NPROP_ENGINE)
    f21 = m21.implicit_result.fatigue
    f22 = m22.implicit_result.fatigue
    assert not f21.get("nonproportional") and f22.get("nonproportional")
    for key in ("von_mises", "normal_plane", "shear_plane"):
        for est in ("narrow_band", "dirlik", "wirsching_light",
                    "tovo_benasciutti"):
            a = f21[key]["summary"][est]["damage_rate"]
            b = f22[key]["summary"][est]["damage_rate"]
            assert a == b, f"{key}/{est}: {a} != {b}"
        # the critical-plane orientations are identical too
        assert np.array_equal(f21[key].get("normal", np.zeros(3)),
                              f22[key].get("normal", np.zeros(3)))


def test_nprop_does_not_mutate_state():
    """The non-proportional path is read-only in the element state and leaves the
    M16 modal_frequencies output bit-identical before and after (the M14-M21
    parity contract extended to M22)."""
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
    npf.nonproportional_summary(frf["freqs"], Scross, 5.0, 1e4,
                                duration=8.0, seed=3, naz=8, npol=7)
    for k, v in before.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(m.bricks.state[k], v), k
    f1, _, _ = modal_frequencies(m, nev=6)
    assert np.array_equal(f0, f1)


def test_direct_dynamics_unchanged_by_nprop_path():
    """A DIRECT /IMPL/DYNA run is byte-for-byte unaffected by the M22
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
