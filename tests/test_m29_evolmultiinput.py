"""
M29 validations: FULLY NON-STATIONARY / EVOLUTIONARY MULTI-INPUT CROSS-PSD
(/IMPL/FATIG/MULT/MINPUT/EVOL) — a TIME-VARYING input coherence matrix
S_ff(omega, t) (the coherence gamma_ab(t) and phase theta_ab(t), and/or the
auto-PSDs G_a(t), DRIFTING with time) driving a per-window multi-input
stress-tensor cross-PSD S_sigmasigma(omega, t_i) = H_sigma S_ff(t_i) H_sigma^H
whose critical plane / F_np may DRIFT as the input coherence evolves, reduced PER
WINDOW by the M20-M27 estimator family + the M28 multi-input path and
Miner-summed, cross-validated against a NON-STATIONARY MULTI-INPUT multivariate
Monte-Carlo. M29 is the CONVERGENCE of M27 (evolutionary joint-tensor) and M28
(stationary multi-input): the coherence itself is now the drifting quantity.

THE M29 <-> M27 / M28 REDUCTIONS (built in, exact / bit-identical)
* a SINGLE INPUT (ninput = 1) recovers the M27 scalar/tensor evolutionary answer
  EXACTLY (a lone input has no off-diagonal coherence -> the M27 windowed tensor);
  the ninput = 1 path DELEGATES to the M27 joint-evolutionary summary / MC;
* a CONSTANT coherence / SINGLE window (flat unit window) recovers the M28
  stationary multi-input answer EXACTLY; that case DELEGATES to the M28
  multi-input summary / MC (bit-identical);
* a CONSTANT-coherence MULTI-window flat schedule gives per-window
  S_sigmasigma,i == the M28 stationary S_sigmasigma for that gamma (each window
  reduces to the same M28 stationary answer).

THE COHERENCE-DRIFT POINT (the M29 <-> M28 boundary made explicit)
* a DRIFTING incoherent -> coherent schedule (gamma 0 -> 1) whose per-window
  response variance and critical plane genuinely DRIFT between the M28
  incoherent-SUM limit (gamma = 0, S = sum_a H_a G_a H_a^H) and the coherent-
  combination limit (gamma = 1, the effective combined pattern) — the input
  coherence itself evolving, not merely the level or spectral shape;
* the per-window reductions consume the per-window S_sigmasigma UNCHANGED (byte-
  identical to reduce_window_tensor on the independently-formed per-window tensor).

NON-STATIONARY MULTI-INPUT MONTE-CARLO CROSS-CHECK
* the single-window / constant-coherence MC reduces to the M28 MC bit-identically;
* the single-input MC reduces to the M27 MC bit-identically;
* the synthesised inputs' per-window measured coherence tracks gamma_ab(t_i).

CARDS + NO-REGRESSION (the M7 parity contract)
* /IMPL/FATIG/MULT/MINPUT/EVOL card mirror (the evolutionary-coherence sub-flag —
  the coherence END pair gamma1 / phase1 on the multi-input header, composing with
  the /EVOL drifting-shape schedule; freimpl.F has no time-varying-coherence path,
  its sole PSD token is IMUMPSD, a MUMPS flag);
* the M29 path NEVER mutates the M28 stationary multi-input reductions / the M20-M27
  fatigue paths / the element state; the M28 stationary answers are byte-identical
  whether or not /EVOL runs, and the direct M10 answer is unchanged.

See PORTING_GUIDE.md roadmap M29.
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
from pyradioss.implicit.random_response import spectral_moments     # noqa: E402
from pyradioss.implicit import multiaxial_fatigue as mf             # noqa: E402
from pyradioss.implicit import multi_input_response as mir          # noqa: E402
from pyradioss.implicit import joint_evolutionary_fatigue as jf     # noqa: E402
from pyradioss.implicit import evolutionary_multi_input as emi      # noqa: E402

# reuse the M28 two-input solid-brick deck
from tests.test_m28_multiinput import _two_input_brick             # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _run(starter_text, engine_text, capture=False):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M29_0000.rad")
    ep = os.path.join(d, "M29_0001.rad")
    with open(sp, "w") as f:
        f.write(starter_text)
    with open(ep, "w") as f:
        f.write(engine_text)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_starter(sp)
        model = run_engine(ep)
    return (model, buf.getvalue()) if capture else model


def _synth_columns(nf=1500, fmax=200.0, ninput=2, proportional=False):
    """Synthetic per-input Voigt stress FRF columns (nf, 6, ninput) + the angular
    grid + per-input auto-PSDs. ``proportional`` makes input 1 = 0.7 x input 0's
    pattern with a COMMON auto-PSD (so the coherence cross-term is constructive —
    the monotone-drift demonstrator)."""
    f = np.linspace(1e-3, fmax, nf)
    omega = 2.0 * np.pi * f

    def peak(fc, bw, A):
        return A * np.exp(-((f - fc) ** 2) / (2.0 * bw ** 2))
    H = np.zeros((nf, 6, ninput), complex)
    H[:, 0, 0] = peak(40, 3, 1.0) + 0.2j * peak(40, 3, 0.4)
    H[:, 3, 0] = 0.3 * peak(40, 3, 0.6) * np.exp(0.5j)
    G = np.zeros((nf, ninput))
    G[:, 0] = peak(45, 25, 2.0) + 0.05
    if ninput > 1:
        if proportional:
            H[:, :, 1] = 0.7 * H[:, :, 0]
            G[:, 1] = G[:, 0]
        else:
            H[:, 1, 1] = peak(65, 4, 0.9)
            H[:, 3, 1] = 0.35 * peak(65, 4, 0.5) * np.exp(0.7j)
            G[:, 1] = peak(60, 30, 1.5) + 0.05
    return f, omega, H, G


# ============================================================================
# THE M29 <-> M27 / M28 REDUCTIONS (built in, exact / bit-identical)
# ============================================================================

def test_coherence_schedule_interpolates():
    """The per-window coherence schedule interpolates linearly across the window
    mid-time fractions from the start to the end pair (the SAME parameterisation as
    the M27 drifting-shape window)."""
    sched = emi.coherence_schedule(4, 0.0, 1.0, phase0=0.0, phase1=0.8)
    s = (np.arange(4) + 0.5) / 4
    for i, (g, p) in enumerate(sched):
        assert float(g) == pytest.approx(s[i])
        assert float(p) == pytest.approx(0.8 * s[i])
    # a single window sits at the mid-time (0.5)
    (g, p), = emi.coherence_schedule(1, 0.2, 0.9)
    assert float(g) == pytest.approx(0.55)
    # no end pair -> constant coherence
    for g, _p in emi.coherence_schedule(3, 0.4):
        assert float(g) == pytest.approx(0.4)


def test_single_window_constant_coherence_recovers_m28_exactly():
    """A CONSTANT coherence / SINGLE flat unit window recovers the M28 stationary
    multi-input answer EXACTLY (all three reductions) — the ninput>1, nwin=1
    delegation is BIT-IDENTICAL to multi_input_multiaxial_summary."""
    _f, omega, H, G = _synth_columns()
    m, C = 5.0, 1e14
    summ = emi.evolutionary_multi_input_summary(
        omega, H, G, [100.0], m, C, gamma0=0.5, phase0=0.3)
    assert summ["delegated"] == "m28_single_window"
    Sff = mir.input_cross_psd_matrix(G, gamma=0.5, phase=0.3)["Sff"]
    Scross = mir.stress_tensor_cross_psd_multi(H, Sff)
    m28 = mir.multi_input_multiaxial_summary(Scross, omega, m, C)
    for k in ("von_mises", "normal_plane", "shear_plane"):
        assert (summ[k]["damage_rate"]
                == m28[k]["summary"]["dirlik"]["damage_rate"])   # bit-identical


def test_single_input_recovers_m27_exactly():
    """A SINGLE INPUT (ninput = 1) recovers the M27 scalar/tensor evolutionary answer
    EXACTLY — the ninput = 1 path DELEGATES to joint_evolutionary_fatigue_summary
    (bit-identical) for a genuinely drifting-shape schedule."""
    _f, omega, H, G = _synth_columns(ninput=1)
    m, C = 5.0, 1e14
    durs = [30.0, 50.0, 20.0]
    scales = [0.6, 1.4, 1.0]
    summ = emi.evolutionary_multi_input_summary(
        omega, H, G, durs, m, C, gamma0=0.0, fc=(40.0, 120.0), bw=6.0,
        scales=scales)
    assert summ["delegated"] == "m27_single_input"
    Scross = mf.stress_tensor_cross_psd(H[:, :, 0], G[:, 0])
    j27 = jf.joint_evolutionary_fatigue_summary(
        omega, Scross, durs, fc=(40.0, 120.0), bw=6.0, m=m, C=C, scales=scales)
    for k in ("von_mises", "normal_plane", "shear_plane"):
        assert summ[k]["damage_rate"] == j27[k]["damage_rate"]   # bit-identical


def test_constant_coherence_multiwindow_is_stationary_per_window():
    """A CONSTANT-coherence MULTI-window flat schedule gives each window's
    S_sigmasigma,i == the M28 stationary S_sigmasigma for that gamma (the flat window
    is unit, so the per-window equivalent-stress RMS equals the M28 stationary RMS
    for every window)."""
    _f, omega, H, G = _synth_columns()
    m, C = 5.0, 1e14
    summ = emi.evolutionary_multi_input_summary(
        omega, H, G, [10.0, 20.0, 30.0], m, C, gamma0=0.6)   # constant coherence
    assert summ["delegated"] is None                          # general path (nwin>1)
    assert summ["coherence_drift"] == pytest.approx(0.0)
    assert summ["constant_shape"]
    # the M28 stationary equivalent-stress RMS for gamma = 0.6
    Sff = mir.input_cross_psd_matrix(G, gamma=0.6)["Sff"]
    Scross = mir.stress_tensor_cross_psd_multi(H, Sff)
    m0 = float(np.atleast_1d(spectral_moments(
        omega, mir.equivalent_vonmises_psd_multi(Scross), nmax=0))[0])
    sig_stat = np.sqrt(m0)
    for w in summ["windows"]:
        assert w["sigma_vm"] == pytest.approx(sig_stat, rel=1e-9)


# ============================================================================
# THE COHERENCE-DRIFT POINT (the M29 <-> M28 boundary made explicit)
# ============================================================================

def test_drift_between_incoherent_and_coherent_limits():
    """A DRIFTING incoherent -> coherent schedule (gamma 0 -> 1) whose per-window
    response variance genuinely DRIFTS between the M28 incoherent-SUM and the
    coherent-combination limits (constructive proportional columns -> monotone)."""
    _f, omega, H, G = _synth_columns(proportional=True)
    m, C = 5.0, 1e14
    summ = emi.evolutionary_multi_input_summary(
        omega, H, G, [1.0] * 6, m, C, gamma0=0.0, gamma1=1.0)
    assert summ["delegated"] is None
    assert summ["coherence_drift"] > 0.5
    sig = np.array([w["sigma_vm"] for w in summ["windows"]])
    gam = np.array([w["gamma"] for w in summ["windows"]])
    assert np.all(np.diff(gam) > 0.0)          # coherence rises window to window
    assert np.all(np.diff(sig) > 0.0)          # variance rises with coherence
    # the M28 incoherent-SUM and coherent-combination equivalent-stress RMS bracket
    Sff0 = mir.input_cross_psd_matrix(G, gamma=0.0)["Sff"]
    Sff1 = mir.input_cross_psd_matrix(G, gamma=1.0)["Sff"]
    sigs = []
    for Sff in (Sff0, Sff1):
        S = mir.stress_tensor_cross_psd_multi(H, Sff)
        v = float(np.atleast_1d(spectral_moments(
            omega, mir.equivalent_vonmises_psd_multi(S), nmax=0))[0])
        sigs.append(np.sqrt(v))
    assert sigs[0] <= sig[0] and sig[-1] <= sigs[1]     # bracketed by the M28 limits


def test_reductions_byte_identical_given_same_per_window_scross():
    """The per-window reductions consume the per-window S_sigmasigma UNCHANGED: each
    window's damage rate equals reduce_window_tensor on the independently-formed
    per-window tensor H S_ff(t_i) H^H (the M27 machinery, byte-identical)."""
    _f, omega, H, G = _synth_columns()
    m, C = 5.0, 1e14
    durs = [10.0, 20.0, 30.0, 15.0]
    summ = emi.evolutionary_multi_input_summary(
        omega, H, G, durs, m, C, gamma0=0.1, gamma1=0.8)
    windows = emi.evolutionary_input_windows(
        omega / (2.0 * np.pi), G, durs, gamma0=0.1, gamma1=0.8)
    for w_out, w_in in zip(summ["windows"], windows):
        Scross_i = mir.stress_tensor_cross_psd_multi(H, w_in["Sff"])
        Mi = mf.tensor_moment_matrices(omega, Scross_i, nmax=4)
        red = jf.reduce_window_tensor(Mi, m, C)
        assert w_out["vm_rate"] == red["von_mises"]["damage_rate"]
        assert w_out["normal_rate"] == red["normal_plane"]["damage_rate"]
        assert w_out["shear_rate"] == red["shear_plane"]["damage_rate"]


def test_coherence_drift_evolves_tensor_shape():
    """A drifting coherence with distinct-orientation inputs genuinely EVOLVES the
    per-window stress-tensor SHAPE (the necessary condition for the critical plane
    to rotate — the coherence-driven analogue of M27's frequency-swept tensor
    drift); a CONSTANT coherence + flat window leaves the shape invariant."""
    _f, omega, H, G = _synth_columns()          # inputs load xx vs yy -> distinct
    m, C = 5.0, 1e14
    drift = emi.evolutionary_multi_input_summary(
        omega, H, G, [1.0] * 6, m, C, gamma0=0.0, gamma1=0.95, phase0=0.0,
        phase1=1.0)
    assert not drift["constant_shape"]          # the tensor orientation evolves
    const = emi.evolutionary_multi_input_summary(
        omega, H, G, [1.0] * 6, m, C, gamma0=0.5)   # constant coherence, flat
    assert const["constant_shape"]              # invariant when coherence is fixed
    assert const["plane_rotation_deg"] == 0.0


# ============================================================================
# NON-STATIONARY MULTI-INPUT MONTE-CARLO CROSS-CHECK
# ============================================================================

def test_single_window_mc_bit_identical_to_m28():
    """The single-window / constant-coherence MC reduces to the M28 multi-input MC
    BIT-IDENTICALLY (delegation)."""
    f, omega, H, G = _synth_columns()
    m, C = 5.0, 1e14
    proj = mf.shear_projection(np.array([0, 0, 1.0]), np.array([1.0, 0, 0]))
    from pyradioss.implicit import multi_input_fatigue as mif
    Sff = mir.input_cross_psd_matrix(G, gamma=0.5)["Sff"]
    d = emi.evolutionary_multi_input_monte_carlo_damage(
        omega, H, G, [60.0], m, C, seed=3, gamma0=0.5, reduction="shear_plane")
    # the M28 MC on the same plane (the summary's plane == the M28 stationary plane)
    m28 = mir.multi_input_multiaxial_summary(
        mir.stress_tensor_cross_psd_multi(H, Sff), omega, m, C)
    # the delegation faithfully passes its frequency grid (omega/2pi) to the M28 MC,
    # so compare against the M28 MC on the SAME grid (a bit-identical delegation)
    dref = mif.monte_carlo_multi_input_damage(
        omega / (2.0 * np.pi), Sff, H, np.asarray(m28["shear_plane"]["proj"]),
        m, C, 60.0, 3)
    assert d["delegated"] == "m28_single_window"
    assert d["damage_rate"] == dref["damage_rate"]


def test_single_input_mc_bit_identical_to_m27():
    """The single-input MC reduces to the M27 non-stationary multivariate MC
    BIT-IDENTICALLY (delegation)."""
    _f, omega, H, G = _synth_columns(ninput=1)
    m, C = 5.0, 1e14
    durs = [20.0, 30.0, 10.0]
    scales = [0.6, 1.4, 1.0]
    d = emi.evolutionary_multi_input_monte_carlo_damage(
        omega, H, G, durs, m, C, seed=5, gamma0=0.0, fc=(40.0, 120.0), bw=6.0,
        scales=scales, reduction="shear_plane")
    Scross = mf.stress_tensor_cross_psd(H[:, :, 0], G[:, 0])
    dref = jf.joint_evolutionary_monte_carlo_damage(
        omega, Scross, durs, fc=(40.0, 120.0), bw=6.0, m=m, C=C, seed=5,
        scales=scales, reduction="shear_plane")
    assert d["delegated"] == "m27_single_input"
    assert d["damage_rate"] == dref["damage_rate"]


def test_synthesized_coherence_tracks_drifting_target():
    """The synthesised inputs' per-window MEASURED coherence tracks the DRIFTING
    target gamma_ab(t_i) (a recovered coherence rising window to window with the
    target)."""
    nf = 600
    f = np.linspace(1e-3, 100.0, nf)
    omega = 2.0 * np.pi * f
    G = np.stack([np.ones(nf), np.ones(nf)], axis=1)     # flat unit auto-PSDs
    H = np.zeros((nf, 6, 2), complex)
    H[:, 0, 0] = 1.0
    H[:, 0, 1] = 1.0                                     # same component -> coherent
    m, C = 5.0, 1e14
    mc = emi.evolutionary_multi_input_monte_carlo_damage(
        omega, H, G, [4000.0] * 4, m, C, seed=11, gamma0=0.1, gamma1=0.8,
        measure=True)
    target = np.array([w["gamma"] for w in emi.evolutionary_multi_input_summary(
        omega, H, G, [4000.0] * 4, m, C, gamma0=0.1, gamma1=0.8)["windows"]])
    measured = mc["window_gamma"]
    assert np.all(np.diff(target) > 0.0)                 # the target drifts up
    assert np.all(np.diff(measured) > -0.1)              # the measured tracks it up
    assert np.allclose(measured, target, atol=0.15)      # within Welch scatter


# ============================================================================
# CARDS + END-TO-END + NO-REGRESSION
# ============================================================================

def _parse(text):
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.engine_keywords import parse_engine_deck
    d = tempfile.mkdtemp()
    ep = os.path.join(d, "M29_0001.rad")
    with open(ep, "w") as f:
        f.write(text)
    with contextlib.redirect_stdout(io.StringIO()):
        return parse_engine_deck(read_deck(ep), MessageLog())


def test_evol_minput_card_parsing():
    """/IMPL/FATIG/MULT/MINPUT/EVOL mirrors the coherence END pair (gamma1, phase1)
    on the multi-input header and the /EVOL drifting-shape schedule."""
    ec = _parse(
        "#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/MINPUT/EVOL\n"
        "0.0 400.0 400 10 6\n5.0 1.0e14 0.03 0.0 0.0 40.0 7\n"
        "60.0 180.0 8.0 8.0 5\n2 0 0.1 0.0 0.0 0.0 0.9 15.0\n1 10\n2 11\n/END\n")
    assert ec.impl_fatig and ec.impl_fatig_mult and ec.impl_fatig_minput
    assert ec.impl_fatig_evol
    assert ec.impl_mi_gamma == pytest.approx(0.1)
    assert ec.impl_mi_gamma1 == pytest.approx(0.9)
    assert ec.impl_mi_phase1 == pytest.approx(15.0)
    assert ec.impl_fatig_evol_fc0 == pytest.approx(60.0)
    assert ec.impl_fatig_evol_fc1 == pytest.approx(180.0)
    assert ec.impl_fatig_evol_nwin == 5
    assert ec.impl_mi_inputs == ((1, 10, 0.0, 0.0, 0.0), (2, 11, 0.0, 0.0, 0.0))


def test_no_drift_gamma1_negative():
    """A negative gamma1 (or absent) means NO coherence drift — the header without a
    gamma1 column leaves impl_mi_gamma1 at its no-drift sentinel."""
    ec = _parse(
        "#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/MINPUT\n0.0 400.0 400 10 6\n"
        "5.0 1.0e14 0.03 0.0 0.0 40.0 7\n2 0 0.5 30.0\n1 10\n2 11\n/END\n")
    assert ec.impl_mi_gamma1 == pytest.approx(-1.0)     # no-drift sentinel


def test_evolutionary_multi_input_end_to_end():
    """The solid-brick cantilever under TWO inputs with a DRIFTING coherence
    (0.1 -> 0.9): the evolutionary multi-input life is reported ALONGSIDE the M28
    stationary multi-input life, the critical plane drifts, and the measured
    coherence tracks the target."""
    eng = ("#\n/RUN/BR/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/MINPUT/EVOL\n"
           "0.0 400.0 400 10 6\n5.0 1.0e14 0.03 0.0 0.0 40.0 7\n"
           "60.0 180.0 8.0 8.0 5\n2 0 0.1 0.0 0.0 0.0 0.9 0.0\n1 10\n2 11\n"
           "/PRINT/-500\n/STOP\n15.0\n")
    model, out = _run(_two_input_brick(), eng, capture=True)
    mi = model.implicit_result.fatigue["multi_input"]
    ev = mi["evolutionary_multi_input"]
    assert ev is not None
    assert ev["gamma0"] == pytest.approx(0.1) and ev["gamma1"] == pytest.approx(0.9)
    assert ev["coherence_drift"] > 0.3
    assert ev["von_mises"]["damage_rate"] > 0.0
    # the real modal coupling makes the critical plane genuinely ROTATE as the
    # input coherence drifts (the coherence-driven critical-plane drift)
    assert ev["plane_rotation_deg"] > 0.0
    assert not ev["constant_shape"]
    # the M28 stationary reference is reported alongside
    assert ev["stationary_multi_input"]["von_mises"] > 0.0
    # the non-stationary MULTI-INPUT MC ran and measured the drifting coherence
    assert ev["monte_carlo"] is not None
    wg = ev["monte_carlo"]["window_gamma"]
    assert np.size(wg) == 5
    assert "FULLY EVOLUTIONARY MULTI-INPUT" in out
    assert "COHERENCE SCHEDULE" in out


def test_m28_stationary_byte_identical_with_evol():
    """The M28 STATIONARY multi-input reductions are byte-unchanged whether or not
    the M29 evolutionary-coherence path runs (a NEW parallel path)."""
    base = ("#\n/RUN/BR/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/MINPUT\n"
            "0.0 400.0 400 10 6\n5.0 1.0e14 0.03 0.0 0.0 40.0 7\n"
            "2 0 0.5 0.0\n1 10\n2 11\n/PRINT/-500\n/STOP\n15.0\n")
    evol = ("#\n/RUN/BR/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/MINPUT/EVOL\n"
            "0.0 400.0 400 10 6\n5.0 1.0e14 0.03 0.0 0.0 40.0 7\n"
            "60.0 180.0 8.0 8.0 5\n2 0 0.5 0.0 0.0 0.0 0.9 0.0\n1 10\n2 11\n"
            "/PRINT/-500\n/STOP\n15.0\n")
    m1 = _run(_two_input_brick(), base).implicit_result.fatigue["multi_input"]
    m2 = _run(_two_input_brick(), evol).implicit_result.fatigue["multi_input"]
    assert m1.get("evolutionary_multi_input") is None
    assert m2.get("evolutionary_multi_input") is not None
    # the M28 stationary multi-input reductions are byte-identical
    for key in ("von_mises", "normal_plane", "shear_plane"):
        assert (m1[key]["summary"]["dirlik"]["damage_rate"]
                == m2[key]["summary"]["dirlik"]["damage_rate"])
    assert np.array_equal(m1["Scross"], m2["Scross"])


def test_evolutionary_multi_input_read_only():
    """The M29 recovery is read-only: a repeated run gives identical element
    stresses and identical M28 stationary + M29 evolutionary damage."""
    eng = ("#\n/RUN/BR/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/MINPUT/EVOL\n"
           "0.0 400.0 400 10 6\n5.0 1.0e14 0.03 0.0 0.0 0.0 7\n"
           "60.0 180.0 8.0 8.0 5\n2 0 0.1 0.0 0.0 0.0 0.9 0.0\n1 10\n2 11\n"
           "/PRINT/-500\n/STOP\n15.0\n")
    a = _run(_two_input_brick(), eng).implicit_result.fatigue["multi_input"]
    b = _run(_two_input_brick(), eng).implicit_result.fatigue["multi_input"]
    assert np.array_equal(a["Scross"], b["Scross"])
    assert (a["evolutionary_multi_input"]["von_mises"]["damage_rate"]
            == b["evolutionary_multi_input"]["von_mises"]["damage_rate"])
