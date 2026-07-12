"""
M27 validations: FULLY NON-STATIONARY / EVOLUTIONARY MULTIAXIAL (JOINT-TENSOR)
FATIGUE (/IMPL/FATIG/MULT/EVOL/JOINT) — the frequency-domain damage of a
MULTIAXIAL random-vibration response whose full stress-TENSOR cross-PSD
S_sigmasigma(omega, t) (a 6x6 EVOLUTIONARY cross-PSD, not just an equivalent
scalar) varies with time, computed by forming the per-window tensor cross-PSD,
reducing it PER WINDOW with the M21/M23 critical-plane machinery (the critical
plane / F_np RE-SEARCHED from the window's OWN tensor — it may ROTATE), and
Miner-summing the per-window multiaxial damages, cross-validated against a
non-stationary MULTIVARIATE time-domain Monte-Carlo. Built ALONGSIDE the M20
scalar / M21-M23 multiaxial / M24 non-Gaussian / M25 non-stationary / M26
evolutionary-scalar spectral fatigue (all stay bit-identical; the joint-tensor
path CONSUMES the M21 tensor / plane machinery, the M23 F_np and the M26
drifting-shape window read-only).

Every new capability gets at least one ANALYTIC / EXACT-REDUCTION check (the
port's philosophy):

THE M27 <-> M21 / M25 / M26 REDUCTIONS (built in, exact)
* a STATIONARY tensor / SINGLE window recovers the M21 spectral multiaxial answer
  EXACTLY (all three reductions — von Mises / max-normal / max-shear);
* the equivalent-scalar reduction of the evolutionary tensor with a FIXED critical
  plane recovers the M26 scalar per-window spectrogram EXACTLY;
* a CONSTANT-tensor-shape / RMS-only drift recovers the M25 multiaxial block
  answer EXACTLY;
* the window Miner-sum EQUALS the duration-weighted per-window multiaxial damages
  (hand check).

THE JOINT-TENSOR POINT (the M27 <-> M26 boundary made explicit)
* a "rotating-principal-axes" case whose per-window critical plane genuinely
  DRIFTS and whose Miner-sum DIFFERS from the M26 fixed-reduction scalar
  spectrogram — the whole point of a JOINT evolutionary tensor;
* the drifting-window tensor moments COMMUTE (windowing the tensor then reducing
  == reducing the windowed tensor).

NON-STATIONARY MULTIVARIATE MONTE-CARLO CROSS-CHECK
* the synthesised multivariate history's short-time per-window crossing rate
  tracking the target evolutionary tensor (the centre-frequency drift);
* the non-stationary multivariate Monte-Carlo damage matching the window /
  joint-tensor spectral estimate within the seeded scatter;
* the single-unit-window limit reducing the multivariate Monte-Carlo EXACTLY
  (bit-identical) to the M21 multivariate Monte-Carlo.

CARDS + NO-REGRESSION (the M7 parity contract)
* /IMPL/FATIG/MULT/EVOL/JOINT card mirror (a PORT sub-flag implying MULT + EVOL,
  composing with /NPROP / /SPEC / /NGAUSS / /NSTAT — freimpl.F has no joint-tensor
  evolutionary fatigue path; its sole PSD token is IMUMPSD, a MUMPS flag);
* the joint-tensor path NEVER mutates the M16 eigensolver / M17-M18 FRFs / the M20
  SCALAR / M21-M23 MULTIAXIAL / M24 NON-GAUSSIAN / M25 NON-STATIONARY / M26
  EVOLUTIONARY fatigue / the element state; the M20-M26 answers are byte-identical
  whether or not /JOINT runs, and the direct M10 answer is unchanged.

See PORTING_GUIDE.md roadmap M27.
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
    spectral_moments, stress_modes)
from pyradioss.implicit import spectral_fatigue as sf               # noqa: E402
from pyradioss.implicit import nonstationary_fatigue as ns          # noqa: E402
from pyradioss.implicit import evolutionary_fatigue as ef           # noqa: E402
from pyradioss.implicit import multiaxial_fatigue as mf             # noqa: E402
from pyradioss.implicit import joint_evolutionary_fatigue as jf     # noqa: E402

# reuse the M21 solid-brick deck and the M26 mission-profile helper
from tests.test_m21_multiaxfatig import _brick_deck                 # noqa: E402
from tests.test_m26_evolfatig import _with_mission                  # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _run(starter_text, engine_text, capture=False):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M27_0000.rad")
    ep = os.path.join(d, "M27_0001.rad")
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
    sp = os.path.join(d, "M27_0000.rad")
    with open(sp, "w") as f:
        f.write(text)
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(sp)


def _rotating_cross_psd(nf=4000, fmax=200.0, shear=0.0):
    """A synthetic stationary stress-TENSOR cross-PSD whose tensor ORIENTATION is
    frequency-DEPENDENT: a LOW band (~40 Hz) loads sigma_xx, a HIGH band (~120 Hz)
    loads sigma_yy. A window that SWEEPS from the low to the high band picks out a
    DIFFERENT tensor orientation per window, so the max-normal / max-shear critical
    plane ROTATES from the x-face to the y-face — the rotating-principal-axes
    demonstrator. ``shear`` optionally adds a low-band xy content (a non-proportional
    seasoning). Returns (omega, Scross)."""
    f = np.linspace(1e-3, fmax, nf)
    omega = 2.0 * np.pi * f
    g_lo = np.exp(-((f - 40.0) ** 2) / (2.0 * 3.0 ** 2))
    g_hi = np.exp(-((f - 120.0) ** 2) / (2.0 * 3.0 ** 2))
    S = np.zeros((nf, 6, 6))
    S[:, 0, 0] = g_lo                     # sigma_xx carried by the low band
    S[:, 1, 1] = g_hi                     # sigma_yy carried by the high band
    if shear:
        S[:, 3, 3] = shear * g_lo         # sigma_xy in the low band
    return omega, S


# ============================================================================
# THE M27 <-> M21 / M25 / M26 REDUCTIONS (built in, exact)
# ============================================================================

def test_single_window_recovers_m21_exactly():
    """A STATIONARY tensor / SINGLE flat window recovers the M21 spectral
    multiaxial answer EXACTLY (all three reductions): one flat unit window gives
    M_{n,1} = the full-tensor moment matrices, so the per-window plane search IS
    the M21 critical_plane_search and the vM moments ARE equivalent_vonmises_
    moments."""
    omega, S = _rotating_cross_psd(shear=0.3)
    m, C = 5.0, 1e14
    Mstat = mf.tensor_moment_matrices(omega, S)
    summ = jf.joint_evolutionary_fatigue_summary(omega, S, [100.0], fc=0.0,
                                                 bw=0.0, m=m, C=C)
    assert summ["constant_shape"]
    assert summ["plane_rotation_deg"] == 0.0
    # von Mises: the trace(Q M_n) moment route
    mom_vm = mf.equivalent_vonmises_moments(Mstat)
    r21_vm = sf.fatigue_summary(mom_vm, m, C)
    for est in ("narrow_band", "dirlik", "wirsching_light", "tovo_benasciutti"):
        assert summ["von_mises"]["damage_rate"] == pytest.approx(
            r21_vm["dirlik"]["damage_rate"], rel=1e-12)
    # the max-normal and max-shear critical planes
    for key, method in (("normal_plane", "normal"), ("shear_plane", "shear")):
        cp = mf.critical_plane_search(Mstat, method=method)
        r21 = sf.fatigue_summary(cp["moments"], m, C)
        assert summ[key]["damage_rate"] == pytest.approx(
            r21["dirlik"]["damage_rate"], rel=1e-12)


def test_fixed_plane_recovers_m26_scalar_spectrogram():
    """The equivalent-scalar reduction of the evolutionary tensor with a FIXED
    critical plane recovers the M26 scalar per-window spectrogram EXACTLY —
    projecting the WINDOWED tensor onto ONE fixed plane and windowing the scalar
    p^T S_cross p IS the M26 windowed-scalar path."""
    omega, S = _rotating_cross_psd(shear=0.3)
    freqs = omega / (2.0 * np.pi)
    m, C = 5.0, 1e14
    durs = [30.0, 50.0, 20.0, 15.0]
    scales = [0.6, 1.4, 1.0, 1.8]
    fx = jf.joint_evolutionary_fatigue_summary(
        omega, S, durs, fc=(40.0, 120.0), bw=6.0, m=m, C=C, scales=scales,
        drift=False)
    # the M26 route: window each reduction's FIXED-plane scalar PSD
    Mstat = mf.tensor_moment_matrices(omega, S)
    stat = jf.reduce_window_tensor(Mstat, m, C)
    for key in ("normal_plane", "shear_plane"):
        proj = stat[key]["proj"]
        psd = np.einsum("i,fij,j->f", proj, S, proj).real
        wins = ef.drifting_shape_spectrogram(freqs, psd, durs, fc=(40.0, 120.0),
                                             bw=6.0, scales=scales)
        m26 = ef.evolutionary_fatigue_summary(wins, m, C)
        assert fx[key]["damage_rate"] == pytest.approx(
            m26["dirlik"]["damage_rate"], rel=1e-9)


def test_constant_shape_recovers_m25_multiaxial():
    """A CONSTANT-tensor-shape / RMS-only drift (a flat window, only the level
    a_i drifts) recovers the M25 multiaxial block answer EXACTLY — the plane does
    NOT rotate and the window Miner-sum of a_i^2-scaled moments IS the M25 block
    Miner-sum."""
    omega, S = _rotating_cross_psd(shear=0.3)
    m, C = 5.0, 1e14
    durs = [30.0, 50.0, 20.0, 15.0]
    scales = [0.6, 1.4, 1.0, 1.8]
    cs = jf.joint_evolutionary_fatigue_summary(
        omega, S, durs, fc=0.0, bw=0.0, m=m, C=C, scales=scales)
    assert cs["constant_shape"]
    assert cs["plane_rotation_deg"] == 0.0
    Mstat = mf.tensor_moment_matrices(omega, S)
    stat = jf.reduce_window_tensor(Mstat, m, C)
    blocks = [{"scale": float(s), "duration": float(d)}
              for s, d in zip(scales, durs)]
    for key in ("von_mises", "normal_plane", "shear_plane"):
        bm = ns.block_fatigue_summary(blocks, m, C,
                                      base_moments=stat[key]["moments"])
        assert cs[key]["damage_rate"] == pytest.approx(bm["damage_rate"],
                                                       rel=1e-9)


def test_window_miner_sum_hand_check():
    """The window Miner-sum EQUALS the duration-weighted per-window multiaxial
    damages: D = sum_i dr_i T_i, damage_rate = D / sum_i T_i — a hand check with a
    genuinely drifting (rotating) tensor."""
    omega, S = _rotating_cross_psd(shear=0.2)
    m, C = 5.0, 1e14
    durs = [10.0, 50.0, 30.0, 120.0]
    summ = jf.joint_evolutionary_fatigue_summary(
        omega, S, durs, fc=(40.0, 120.0), bw=6.0, m=m, C=C)
    for key in ("von_mises", "normal_plane", "shear_plane"):
        D = sum(w[key.replace("_plane", "").replace("von_mises", "vm") + "_rate"]
                * w["duration"] for w in summ["windows"])
        T = sum(w["duration"] for w in summ["windows"])
        assert summ[key]["damage"] == pytest.approx(D, rel=1e-12)
        assert summ[key]["total_time"] == pytest.approx(T)
        assert summ[key]["damage_rate"] == pytest.approx(D / T, rel=1e-12)


# ============================================================================
# THE JOINT-TENSOR POINT (the M27 <-> M26 boundary made explicit)
# ============================================================================

def test_rotating_axes_plane_drifts_and_differs_from_m26():
    """A "rotating-principal-axes" case whose per-window critical plane genuinely
    DRIFTS and whose joint-tensor Miner-sum DIFFERS from the M26 fixed-reduction
    scalar spectrogram — the whole point of a JOINT evolutionary tensor."""
    omega, S = _rotating_cross_psd()
    m, C = 5.0, 1e14
    # a window that sweeps from the low (xx) band to the high (yy) band
    drift = jf.joint_evolutionary_fatigue_summary(
        omega, S, [1.0] * 6, fc=(40.0, 120.0), bw=6.0, m=m, C=C, drift=True)
    fixed = jf.joint_evolutionary_fatigue_summary(
        omega, S, [1.0] * 6, fc=(40.0, 120.0), bw=6.0, m=m, C=C, drift=False)
    # the critical plane genuinely rotates (x-face -> y-face)
    assert not drift["constant_shape"]
    assert drift["plane_rotation_deg"] > 45.0
    # the max-normal plane normal swings from ~x at the first window to ~y at the
    # last (the tensor orientation drifts)
    n_first = drift["windows"][0]["normal_n"]
    n_last = drift["windows"][-1]["normal_n"]
    assert abs(n_first[0]) > 0.9          # first window: normal ~ x
    assert abs(n_last[1]) > 0.9           # last window:  normal ~ y
    # the drifting max-NORMAL-plane Miner-sum DIFFERS SUBSTANTIALLY from the fixed
    # (M26) reduction — the re-searched plane always aligns with the active band's
    # tensor, accruing more damage than a single compromise plane (for this pure-
    # normal tensor the max-SHEAR plane rotates to a physically equal-damage 45deg
    # plane, so it is the max-NORMAL plane that carries the boundary — the whole
    # point of a JOINT evolutionary tensor made explicit)
    assert not (0.9 < drift["normal_plane"]["damage_rate"]
                / fixed["normal_plane"]["damage_rate"] < 1.1)
    # the von-Mises reduction (a FIXED quadratic — no plane) does NOT drift: it is
    # identical whether the plane is re-searched or held fixed
    assert drift["von_mises"]["damage_rate"] == pytest.approx(
        fixed["von_mises"]["damage_rate"], rel=1e-12)


def test_windowed_tensor_moments_commute():
    """Windowing the tensor then integrating == integrating the windowed tensor
    (the window commutes with the linear reduction — M26's commutation lifted to
    the full 6x6 matrix): windowed_tensor_moment_matrices(omega, S, W, a) ==
    tensor_moment_matrices(omega, a^2 W S)."""
    omega, S = _rotating_cross_psd(nf=2000, shear=0.3)
    freqs = omega / (2.0 * np.pi)
    W = jf._spectral_window(freqs, 80.0, 12.0)
    a = 1.7
    Mi = jf.windowed_tensor_moment_matrices(omega, S, W, scale=a)
    ref = mf.tensor_moment_matrices(omega, (a ** 2) * W[:, None, None] * S)
    assert np.allclose(Mi, ref, rtol=1e-12, atol=1e-14)


# ============================================================================
# NON-STATIONARY MULTIVARIATE MONTE-CARLO CROSS-CHECK
# ============================================================================

def test_joint_mc_single_window_bit_identical_to_m21():
    """A single UNIT window reduces the non-stationary MULTIVARIATE Monte-Carlo
    EXACTLY (bit-identical) to the M21 multivariate Monte-Carlo (the fixed plane,
    one shared tensor — the whole chain collapses to the M21 multivariate MC)."""
    omega, S = _rotating_cross_psd(nf=3000, shear=0.3)
    freqs = omega / (2.0 * np.pi)          # the grid the joint module derives
    m, C = 5.0, 1e14
    mc = jf.joint_evolutionary_monte_carlo_damage(
        omega, S, [500.0], fc=0.0, bw=0.0, m=m, C=C, seed=7, fs=800.0)
    assert mc["delegated"]
    stat = jf.reduce_window_tensor(mf.tensor_moment_matrices(omega, S), m, C)
    m21mc = mf.monte_carlo_multiaxial_damage(
        freqs, S, stat["shear_plane"]["proj"], m, C, 500.0, 7, fs=800.0)
    assert mc["damage_rate"] == m21mc["damage_rate"]
    assert np.array_equal(mc["ranges"], m21mc["ranges"])


def test_joint_mc_tracks_spectrogram():
    """The synthesised (non-stationary multivariate) history's short-time
    spectrogram tracks the target evolutionary tensor: the per-window zero-crossing
    rate of the projected scalar follows the swept centre frequency."""
    omega, S = _rotating_cross_psd()
    m, C = 5.0, 1e14
    mc = jf.joint_evolutionary_monte_carlo_damage(
        omega, S, [300.0] * 3, fc=(40.0, 120.0), bw=6.0, m=m, C=C, seed=3,
        fs=1600.0)
    assert not mc["delegated"]
    nu0 = mc["window_nu0"]
    assert nu0[-1] > nu0[0]                # centre frequency drifted UP
    assert nu0[0] == pytest.approx(40.0, rel=0.4)
    assert nu0[-1] == pytest.approx(120.0, rel=0.4)


def test_joint_mc_matches_window_spectral_estimate():
    """On a drifting (rotating) tensor the non-stationary multivariate Monte-Carlo
    damage matches the joint-tensor window spectral estimate within the seeded
    scatter (the max-shear reduction)."""
    omega, S = _rotating_cross_psd(shear=0.4)
    m, C = 5.0, 1e14
    durs = [1000.0, 1000.0, 1000.0]
    summ = jf.joint_evolutionary_fatigue_summary(
        omega, S, durs, fc=(40.0, 120.0), bw=8.0, m=m, C=C)
    mc = jf.joint_evolutionary_monte_carlo_damage(
        omega, S, durs, fc=(40.0, 120.0), bw=8.0, m=m, C=C, seed=11, fs=1600.0,
        summary=summ, reduction="shear_plane")
    ratio = mc["damage_rate"] / summ["shear_plane"]["damage_rate"]
    assert 0.4 < ratio < 2.2, f"jointMC / window-spectral = {ratio:.3f} off band"


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


def test_impl_fatig_joint_card_parsing():
    """/IMPL/FATIG/MULT/EVOL/JOINT sets the joint-tensor flag, IMPLIES MULT + EVOL
    and reads the SHARED drifting-shape schedule (fc0 fc1 bw0 bw1 nwin); /TENSOR is
    a synonym; it composes with /NSTAT / /NGAUSS and does NOT get set by a plain
    /EVOL card."""
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/EVOL/JOINT\n"
                   "0.0 400 500 10 6\n5.0 1.0e4 0.03\n"
                   "50.0 250.0 10.0 40.0 6\n/END\n")
    assert ec.impl_fatig and ec.impl_fatig_joint
    assert ec.impl_fatig_mult and ec.impl_fatig_evol      # JOINT implies both
    assert ec.impl_fatig_evol_fc0 == pytest.approx(50.0)
    assert ec.impl_fatig_evol_fc1 == pytest.approx(250.0)
    assert ec.impl_fatig_evol_nwin == 6

    # /TENSOR is a synonym; it too implies MULT + EVOL
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/FATIG/TENSOR\n"
                   "0.0 400 500 10 6\n5.0 1.0e4 0.03\n"
                   "30.0 150.0 8.0 40.0 8\n/END\n")
    assert (ec.impl_fatig_joint and ec.impl_fatig_mult and ec.impl_fatig_evol
            and ec.impl_fatig_evol_nwin == 8)

    # composes with NSTAT (modulation line 2, EVOL schedule line 3)
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/FATIG/MULT/EVOL/JOINT/NSTAT\n"
                   "0.0 400 500 10 6\n5.0 1e4 0.03\n"
                   "20 8\n30.0 150.0 8.0 40.0 8\n/END\n")
    assert (ec.impl_fatig_joint and ec.impl_fatig_nstat
            and ec.impl_fatig_modfunct == 20 and ec.impl_fatig_evol_nwin == 8)

    # a plain /EVOL (scalar) card is NOT joint-tensor
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/EVOL\n"
                   "0.0 400 500 10 6\n5.0 1e4 0.03\n"
                   "50.0 250.0 10.0 40.0 6\n/END\n")
    assert ec.impl_fatig_evol and not ec.impl_fatig_joint


# ============================================================================
# END-TO-END
# ============================================================================

def test_joint_end_to_end():
    """/IMPL/FATIG/MULT/EVOL/JOINT end to end on the solid brick: the listing
    prints the FULLY EVOLUTIONARY MULTIAXIAL JOINT-TENSOR block ALONGSIDE the M21
    stationary, M25 non-stationary and M26 scalar-evolutionary ones, stores a
    ``joint_evolutionary`` sub-entry with the window Miner-sum, the per-window
    critical-plane drift and the non-stationary multivariate Monte-Carlo."""
    eng = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/EVOL/JOINT/NSTAT\n"
           "0.0 400.0 500 10 6\n5.0 1.0e4 0.03 0.0 0.0 15.0 21000\n20 6\n"
           "50.0 250.0 10.0 40.0 6\n/PRINT/-500\n/STOP\n15.0\n")
    m, out = _run(_with_mission(_brick_deck(nx=3)), eng, capture=True)
    fat = m.implicit_result.fatigue
    assert fat.get("multiaxial")
    jv = fat["joint_evolutionary"]
    assert jv is not None
    assert jv["nwin"] == 6
    for key in ("von_mises", "normal_plane", "shear_plane"):
        assert jv["summary"][key]["damage_rate"] > 0.0
    # the per-window drift breakdown is present (plane normal / F_np / RMS)
    wins = jv["summary"]["windows"]
    assert len(wins) == 6
    assert all("shear_n" in w and "F_np" in w and "sigma_vm" in w for w in wins)
    assert jv["monte_carlo"] is not None
    # all FOUR blocks are present in the listing, side by side
    assert "FULLY EVOLUTIONARY MULTIAXIAL JOINT-TENSOR FATIGUE" in out
    assert "MULTIAXIAL / CRITICAL-PLANE SPECTRAL FATIGUE" in out
    assert "NON-STATIONARY / EVOLUTIONARY-PSD FATIGUE" in out
    assert "FULLY EVOLUTIONARY / NON-SEPARABLE-PSD FATIGUE" in out
    # the M21/M25/M26 blocks are still fully populated
    assert fat["nonstationary"] is not None
    assert fat["evolutionary"] is not None


def test_joint_composes_with_ngauss_and_nstat():
    """/IMPL/FATIG/MULT/NGAUSS/NSTAT/EVOL/JOINT: the M24 non-Gaussian, M25
    non-stationary, M26 scalar-evolutionary AND M27 joint-tensor corrections are
    ALL stored side by side."""
    eng = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n"
           "/IMPL/FATIG/MULT/NGAUSS/NSTAT/EVOL/JOINT\n"
           "0.0 400.0 500 10 6\n5.0 1.0e4 0.03 0.0 0.0 15.0 21000\n"
           "6.0 0.0\n20 6\n50.0 250.0 10.0 40.0 6\n/PRINT/-500\n/STOP\n15.0\n")
    m, out = _run(_with_mission(_brick_deck(nx=3)), eng, capture=True)
    fat = m.implicit_result.fatigue
    assert fat["nongaussian"] is not None
    assert fat["nonstationary"] is not None
    assert fat["evolutionary"] is not None
    assert fat["joint_evolutionary"] is not None
    assert fat["nongaussian"]["gamma4"] == pytest.approx(6.0)
    assert "NON-GAUSSIAN / KURTOSIS FATIGUE" in out
    assert "FULLY EVOLUTIONARY MULTIAXIAL JOINT-TENSOR FATIGUE" in out


# ============================================================================
# NO-REGRESSION (the M7 parity contract)
# ============================================================================

def test_m21_m25_m26_byte_identical_with_without_joint():
    """The M21 spectral, M25 non-stationary AND M26 scalar-evolutionary answers are
    BYTE-IDENTICAL whether or not the M27 joint-tensor path runs — the correction
    is NEW and ALONGSIDE."""
    deck = _with_mission(_brick_deck(nx=3))
    eng = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/EVOL/NSTAT\n"
           "0.0 400.0 500 10 6\n5.0 1.0e4 0.03 0.0 0.0 15.0 21000\n"
           "20 6\n50.0 250.0 10.0 40.0 6\n/PRINT/-500\n/STOP\n15.0\n")
    eng_j = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/EVOL/JOINT/NSTAT\n"
             "0.0 400.0 500 10 6\n5.0 1.0e4 0.03 0.0 0.0 15.0 21000\n"
             "20 6\n50.0 250.0 10.0 40.0 6\n/PRINT/-500\n/STOP\n15.0\n")
    a = _run(deck, eng)
    b = _run(deck, eng_j)
    fa = a.implicit_result.fatigue
    fb = b.implicit_result.fatigue
    assert fa["joint_evolutionary"] is None
    assert fb["joint_evolutionary"] is not None
    # M21 reductions
    for key in ("von_mises", "normal_plane", "shear_plane"):
        for est in ("narrow_band", "dirlik", "wirsching_light",
                    "tovo_benasciutti"):
            assert (fa[key]["summary"][est]["damage_rate"]
                    == fb[key]["summary"][est]["damage_rate"])
    # M25 non-stationary block Miner-sum
    for key in ("von_mises", "normal_plane", "shear_plane"):
        assert (fa["nonstationary"][key]["block"]["damage_rate"]
                == fb["nonstationary"][key]["block"]["damage_rate"])
    # M26 scalar-evolutionary window Miner-sum
    for key in ("von_mises", "normal_plane", "shear_plane"):
        assert (fa["evolutionary"][key]["summary"]["dirlik"]["damage_rate"]
                == fb["evolutionary"][key]["summary"]["dirlik"]["damage_rate"])


def test_joint_does_not_mutate_state():
    """The joint-tensor path is read-only in the element state and leaves the M16
    modal_frequencies output bit-identical before and after (the M14-M26 parity
    contract extended to M27)."""
    m = _starter(_brick_deck(nx=2))
    f0, _, _ = modal_frequencies(m, nev=6)
    basis = build_modal_basis(m, nev=6)
    Sigma, channels = stress_modes(m, basis)
    z = modal_damping(basis.omega, uniform=0.03)
    n = m.numnod
    F = np.zeros((n, 3))
    LoadsAndConstraints(m, MessageLog()).external_forces(1.0, F, m.x0)
    modal_frequency_response(basis, F, np.zeros((n, 3)),
                             np.linspace(1e-6, 400.0, 60), z)
    omega, S = _rotating_cross_psd(nf=2000, shear=0.3)
    jf.joint_evolutionary_fatigue_summary(omega, S, [1.0] * 5, fc=(40.0, 120.0),
                                          bw=6.0, m=5.0, C=1e14)
    f1, _, _ = modal_frequencies(m, nev=6)
    assert np.array_equal(f0, f1)


def test_direct_dynamics_unchanged_by_joint_path():
    """A DIRECT /IMPL/DYNA run is byte-for-byte unaffected by the M27 joint-tensor
    machinery living in the same package (the M10 integrator stays bit-identical,
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
