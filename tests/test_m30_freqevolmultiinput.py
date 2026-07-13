"""
M30 validations: FREQUENCY-DEPENDENT + TIME-VARYING (EVOLUTIONARY) INPUT COHERENCE
(/IMPL/FATIG/MULT/MINPUT/EVOL/FCOH) — a coherence matrix gamma_ab(f, t) varying
with BOTH FREQUENCY AND TIME (a measured / modelled gamma_ab(f) SHAPE that DRIFTS
window to window, and/or the M28 exponential/convection field with a TIME-VARYING
decay coefficient / reference speed), driving a per-window multi-input
stress-tensor cross-PSD S_sigmasigma(omega, t_i) = H_sigma S_ff(t_i) H_sigma^H
whose critical plane / F_np may DRIFT as the coherence FREQUENCY-SHAPE evolves,
reduced PER WINDOW by the M20-M27 estimator family + the M28/M29 multi-input paths
and Miner-summed, cross-validated against a NON-STATIONARY MULTI-INPUT multivariate
Monte-Carlo. M30 is the CONVERGENCE of M28 (frequency-dependent coherence) and M29
(time-varying coherence): the coherence now varies in BOTH f and t.

THE M30 <-> M29 / M28 REDUCTIONS (built in, exact / bit-identical)
* a FREQUENCY-FLAT coherence (gamma_ab(f) constant across frequency — a scalar /
  (n, n) drifting-in-time coherence) recovers the M29 scalar-coherence answer
  EXACTLY; that case DELEGATES to evolutionary_multi_input_summary (bit-identical),
  which in turn recovers M28 / M27 in ITS special cases — M29 is EXACTLY the
  frequency-flat special case of M30;
* a STATIONARY (single-window / constant-in-time) frequency-dependent coherence
  recovers the M28 frequency-dependent answer EXACTLY; that case DELEGATES to the
  M28 multi_input_multiaxial_summary — M28 is EXACTLY the single-window special
  case of M30.

THE FREQUENCY-SHAPE-DRIFT POINT (the M30 <-> M28/M29 boundary made explicit)
* a DRIFTING-frequency-shape schedule (an exponential/convection field whose
  DECORRELATION FREQUENCY moves UP through the mission — a convection speed that
  ramps or a decay that falls) whose per-window response variance AND critical
  plane genuinely DRIFT as the frequency-shape of the coherence evolves, not merely
  its scalar level;
* the per-window reductions consume the per-window S_sigmasigma UNCHANGED (byte-
  identical to reduce_window_tensor on the independently-formed per-window tensor).

NON-STATIONARY MULTI-INPUT MONTE-CARLO CROSS-CHECK
* the frequency-flat MC reduces to the M29 MC bit-identically;
* the single-window MC reduces to the M28 MC bit-identically;
* the synthesised inputs' per-window measured coherence SPECTRUM (Welch, per
  frequency band) tracks the target gamma_ab(f, t_i) — NOT just a band-mean scalar.

CARDS + NO-REGRESSION (the M7 parity contract)
* /IMPL/FATIG/MULT/MINPUT/EVOL/FCOH card mirror (the frequency-dependent-coherence
  sub-flag — the time-varying exponential decay/speed pair and the per-pair measured
  gamma(f) /FUNCT on the multi-input header; freimpl.F has no
  frequency-dependent-time-varying-coherence path, its sole PSD token is IMUMPSD);
* the M30 path NEVER mutates the M28 stationary / M29 scalar-coherence reductions /
  the M20-M27 fatigue paths / the element state; the M28 + M29 answers are
  byte-identical whether or not /FCOH runs.

See PORTING_GUIDE.md roadmap M30.
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
from pyradioss.implicit import evolutionary_multi_input as emi      # noqa: E402
from pyradioss.implicit import freq_evolutionary_multi_input as fem  # noqa: E402

# reuse the M28 two-input solid-brick deck
from tests.test_m28_multiinput import _two_input_brick             # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _run(starter_text, engine_text, capture=False):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M30_0000.rad")
    ep = os.path.join(d, "M30_0001.rad")
    with open(sp, "w") as f:
        f.write(starter_text)
    with open(ep, "w") as f:
        f.write(engine_text)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_starter(sp)
        model = run_engine(ep)
    return (model, buf.getvalue()) if capture else model


def _synth_columns(nf=800, fmax=200.0, ninput=2):
    """Synthetic per-input Voigt stress FRF columns (nf, 6, ninput) + the angular
    grid + per-input auto-PSDs — input 0 loads sigma_xx around 40 Hz, input 1 loads
    sigma_yy around 65 Hz (distinct tensor orientations, so a frequency-dependent
    coherence that weights them differently across the band evolves the tensor)."""
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
        H[:, 1, 1] = peak(65, 4, 0.9)
        H[:, 3, 1] = 0.35 * peak(65, 4, 0.5) * np.exp(0.7j)
        G[:, 1] = peak(60, 30, 1.5) + 0.05
    return f, omega, H, G


def _exp_stacks(f, decay0, decay1, speed0=100.0, speed1=None, sep=3.0):
    """Two-input START / END exponential coherence stacks with separation ``sep``."""
    pos = np.array([[0.0, 0, 0], [sep, 0, 0]])
    g0 = fem.exponential_coherence_stack(f, pos, decay0, ref_speed=speed0)
    g1 = fem.exponential_coherence_stack(f, pos, decay1,
                                         ref_speed=(speed0 if speed1 is None
                                                    else speed1))
    return g0, g1


# ============================================================================
# THE M30 <-> M29 / M28 REDUCTIONS (built in, exact / bit-identical)
# ============================================================================

def test_frequency_flat_recovers_m29_exactly():
    """A FREQUENCY-FLAT coherence (scalar gamma0 / gamma1) recovers the M29
    scalar-coherence answer EXACTLY — the call DELEGATES to M29 bit-identically
    (M29 is exactly the frequency-flat special case of M30)."""
    _f, omega, H, G = _synth_columns()
    m, C = 5.0, 1e14
    durs = [10.0, 20.0, 30.0, 15.0]
    fe = fem.freq_evolutionary_multi_input_summary(
        omega, H, G, durs, m, C, gamma0=0.1, gamma1=0.8)
    m29 = emi.evolutionary_multi_input_summary(
        omega, H, G, durs, m, C, gamma0=0.1, gamma1=0.8)
    assert fe["delegated_m29"] is True
    assert fe["freq_dependent"] is False
    for k in ("von_mises", "normal_plane", "shear_plane"):
        assert fe[k]["damage_rate"] == m29[k]["damage_rate"]     # bit-identical


def test_stationary_freq_dependent_recovers_m28_exactly():
    """A STATIONARY (single-window / constant-in-time) FREQUENCY-DEPENDENT coherence
    recovers the M28 frequency-dependent answer EXACTLY — the ninput>1, nwin=1
    delegation is BIT-IDENTICAL to multi_input_multiaxial_summary on the
    frequency-dependent S_ff."""
    f, omega, H, G = _synth_columns()
    m, C = 5.0, 1e14
    gstack, _g1 = _exp_stacks(f, 0.3, 0.3)          # constant-in-time freq shape
    fe = fem.freq_evolutionary_multi_input_summary(
        omega, H, G, [100.0], m, C, gamma0=gstack)
    assert fe["delegated"] == "m28_single_window"
    Sff = mir.input_cross_psd_matrix(G, gamma=gstack)["Sff"]
    Scross = mir.stress_tensor_cross_psd_multi(H, Sff)
    m28 = mir.multi_input_multiaxial_summary(Scross, omega, m, C)
    for k in ("von_mises", "normal_plane", "shear_plane"):
        assert (fe[k]["damage_rate"]
                == m28[k]["summary"]["dirlik"]["damage_rate"])   # bit-identical


def test_single_input_recovers_m29_delegation():
    """A SINGLE INPUT (ninput = 1) has no off-diagonal coherence, so a
    frequency-dependent coherence is irrelevant — the call DELEGATES to M29 (which
    delegates to the M27 single-input path), bit-identical."""
    _f, omega, H, G = _synth_columns(ninput=1)
    m, C = 5.0, 1e14
    durs = [30.0, 50.0, 20.0]
    scales = [0.6, 1.4, 1.0]
    fe = fem.freq_evolutionary_multi_input_summary(
        omega, H, G, durs, m, C, gamma0=0.0, fc=(40.0, 120.0), bw=6.0,
        scales=scales)
    m29 = emi.evolutionary_multi_input_summary(
        omega, H, G, durs, m, C, gamma0=0.0, fc=(40.0, 120.0), bw=6.0,
        scales=scales)
    assert fe["delegated_m29"] is True
    for k in ("von_mises", "normal_plane", "shear_plane"):
        assert fe[k]["damage_rate"] == m29[k]["damage_rate"]


def test_freq_coherence_stacks_interpolate():
    """The per-window frequency-dependent coherence stacks interpolate LINEARLY from
    the START stack to the END stack across the window mid-time fractions, with a
    unit diagonal and clipped to [0, 1] per window."""
    f, _omega, _H, _G = _synth_columns(nf=200)
    g0, g1 = _exp_stacks(f, 0.6, 0.1)
    stacks = fem.freq_coherence_stacks(f, 5, g0, g1)
    s = (np.arange(5) + 0.5) / 5
    for i, gs in enumerate(stacks):
        expect = np.clip(g0 + s[i] * (g1 - g0), 0.0, 1.0)
        for a in range(2):
            expect[:, a, a] = 1.0
        assert np.allclose(gs, expect)
        assert np.allclose(np.diagonal(gs, axis1=1, axis2=2), 1.0)
    # a stationary (gamma1 = gamma0) schedule -> every window == g0
    for gs in fem.freq_coherence_stacks(f, 3, g0):
        assert np.allclose(gs, np.clip(g0, 0.0, 1.0))


def test_measured_coherence_stack_shape():
    """A per-frequency MEASURED shape gamma(f) is applied to the off-diagonal pairs
    (diagonal held at 1), the frequency-dependent generalisation of the scalar
    constant coherence."""
    f = np.linspace(1e-3, 100.0, 300)
    shape = np.clip(1.0 - f / 120.0, 0.0, 1.0)       # rolls off with frequency
    g = fem.measured_coherence_stack(f, shape, 3)
    assert g.shape == (300, 3, 3)
    assert np.allclose(np.diagonal(g, axis1=1, axis2=2), 1.0)
    assert np.allclose(g[:, 0, 1], shape)            # off-diagonal = the shape
    assert np.allclose(g[:, 1, 0], shape)            # Hermitian (real symmetric)


# ============================================================================
# THE FREQUENCY-SHAPE-DRIFT POINT (the M30 <-> M28/M29 boundary made explicit)
# ============================================================================

def test_exponential_drift_moves_decorrelation_up():
    """An exponential/convection field whose decay FALLS (or convection speed ramps)
    moves the DECORRELATION FREQUENCY monotonically UP through the mission — the
    'decorrelation moves up in frequency' case NEITHER M28 nor M29 covers."""
    f, omega, H, G = _synth_columns(nf=600, fmax=400.0)
    m, C = 5.0, 1e14
    g0, g1 = _exp_stacks(f, 0.9, 0.2, speed0=100.0)
    fe = fem.freq_evolutionary_multi_input_summary(
        omega, H, G, [1.0] * 6, m, C, gamma0=g0, gamma1=g1)
    assert fe["freq_dependent"] is True
    assert fe["delegated"] is None and not fe["delegated_m29"]
    fdec = np.array([w["decorr_freq"] for w in fe["windows"]])
    assert np.all(np.isfinite(fdec))
    assert np.all(np.diff(fdec) > 0.0)               # decorrelation moves UP
    assert fe["decorr_drift"] > 20.0                 # a genuine frequency drift


def test_convection_speed_ramp_moves_decorrelation_up():
    """A convection SPEED that ramps (decay fixed) also moves the decorrelation
    frequency up — the same physics via the ref-speed schedule."""
    f, omega, H, G = _synth_columns(nf=600, fmax=400.0)
    m, C = 5.0, 1e14
    g0, g1 = _exp_stacks(f, 0.6, 0.6, speed0=40.0, speed1=200.0)
    fe = fem.freq_evolutionary_multi_input_summary(
        omega, H, G, [1.0] * 6, m, C, gamma0=g0, gamma1=g1)
    fdec = np.array([w["decorr_freq"] for w in fe["windows"]])
    assert np.all(np.isfinite(fdec)) and np.all(np.diff(fdec) > 0.0)
    assert fe["decorr_drift"] > 20.0


def test_freq_shape_drift_evolves_variance_and_plane():
    """A drifting frequency-shape coherence genuinely EVOLVES the per-window
    stress-tensor SHAPE (the necessary condition for the critical plane to rotate)
    and its per-window response variance; a STATIONARY frequency-dependent coherence
    leaves the shape invariant."""
    f, omega, H, G = _synth_columns(nf=600, fmax=400.0)
    m, C = 5.0, 1e14
    g0, g1 = _exp_stacks(f, 1.2, 0.05, speed0=100.0)
    drift = fem.freq_evolutionary_multi_input_summary(
        omega, H, G, [1.0] * 6, m, C, gamma0=g0, gamma1=g1)
    assert not drift["constant_shape"]               # the tensor orientation evolves
    sig = np.array([w["sigma_vm"] for w in drift["windows"]])
    assert np.ptp(sig) > 0.0                          # per-window variance drifts
    # a STATIONARY frequency-dependent coherence: the shape is invariant
    const = fem.freq_evolutionary_multi_input_summary(
        omega, H, G, [1.0] * 6, m, C, gamma0=g0)      # gamma1 = gamma0
    assert const["constant_shape"]
    assert const["plane_rotation_deg"] == 0.0


def test_reductions_byte_identical_given_same_per_window_scross():
    """The per-window reductions consume the per-window S_sigmasigma UNCHANGED: each
    window's damage rate equals reduce_window_tensor on the independently-formed
    per-window tensor H S_ff(f, t_i) H^H (the M27 machinery, byte-identical)."""
    from pyradioss.implicit import joint_evolutionary_fatigue as jf
    f, omega, H, G = _synth_columns(nf=500, fmax=300.0)
    m, C = 5.0, 1e14
    durs = [10.0, 20.0, 30.0, 15.0]
    g0, g1 = _exp_stacks(f, 0.9, 0.1)
    summ = fem.freq_evolutionary_multi_input_summary(
        omega, H, G, durs, m, C, gamma0=g0, gamma1=g1)
    # rebuild the per-window windows with the SAME public builders the summary used
    stacks = fem.freq_coherence_stacks(f, len(durs),
                                       np.clip(g0, 0.0, 1.0),
                                       np.clip(g1, 0.0, 1.0))
    windows = fem.freq_evolutionary_input_windows(f, G, durs, stacks)
    for w_out, w_in in zip(summ["windows"], windows):
        Scross_i = mir.stress_tensor_cross_psd_multi(H, w_in["Sff"])
        Mi = mf.tensor_moment_matrices(omega, Scross_i, nmax=4)
        red = jf.reduce_window_tensor(Mi, m, C)
        assert w_out["vm_rate"] == red["von_mises"]["damage_rate"]
        assert w_out["normal_rate"] == red["normal_plane"]["damage_rate"]
        assert w_out["shear_rate"] == red["shear_plane"]["damage_rate"]


# ============================================================================
# NON-STATIONARY MULTI-INPUT MONTE-CARLO CROSS-CHECK
# ============================================================================

def test_freq_flat_mc_bit_identical_to_m29():
    """The frequency-flat MC reduces to the M29 evolutionary multi-input MC
    BIT-IDENTICALLY (delegation)."""
    _f, omega, H, G = _synth_columns()
    m, C = 5.0, 1e14
    d = fem.freq_evolutionary_multi_input_monte_carlo_damage(
        omega, H, G, [20.0, 30.0, 10.0], m, C, seed=5, gamma0=0.1, gamma1=0.8,
        reduction="shear_plane")
    dref = emi.evolutionary_multi_input_monte_carlo_damage(
        omega, H, G, [20.0, 30.0, 10.0], m, C, seed=5, gamma0=0.1, gamma1=0.8,
        reduction="shear_plane")
    assert d["delegated_m29"] is True
    assert d["damage_rate"] == dref["damage_rate"]


def test_single_window_mc_bit_identical_to_m28():
    """The single-window / constant frequency-dependent coherence MC reduces to the
    M28 multi-input MC BIT-IDENTICALLY (delegation)."""
    from pyradioss.implicit import multi_input_fatigue as mif
    f, omega, H, G = _synth_columns()
    m, C = 5.0, 1e14
    gstack, _g1 = _exp_stacks(f, 0.3, 0.3)
    d = fem.freq_evolutionary_multi_input_monte_carlo_damage(
        omega, H, G, [60.0], m, C, seed=3, gamma0=gstack, reduction="shear_plane")
    Sff = mir.input_cross_psd_matrix(G, gamma=gstack)["Sff"]
    m28 = mir.multi_input_multiaxial_summary(
        mir.stress_tensor_cross_psd_multi(H, Sff), omega, m, C)
    # the delegation passes its frequency grid as omega/(2pi) to the M28 MC, so
    # compare against the M28 MC on the SAME grid (a bit-identical delegation)
    dref = mif.monte_carlo_multi_input_damage(
        omega / (2.0 * np.pi), Sff, H, np.asarray(m28["shear_plane"]["proj"]),
        m, C, 60.0, 3)
    assert d["delegated"] == "m28_single_window"
    assert d["damage_rate"] == dref["damage_rate"]


def test_measured_coherence_spectrum_tracks_target():
    """The synthesised inputs' per-window measured coherence SPECTRUM (per frequency
    band) tracks the DRIFTING target gamma_ab(f, t_i) — NOT just a band-mean scalar.
    A frequency-dependent coherence that is COHERENT at low frequency and
    DECORRELATED at high frequency (and whose shape drifts up) is recovered
    band-by-band."""
    nf = 600
    f = np.linspace(1e-3, 100.0, nf)
    omega = 2.0 * np.pi * f
    G = np.stack([np.ones(nf), np.ones(nf)], axis=1)     # flat unit auto-PSDs
    H = np.zeros((nf, 6, 2), complex)
    H[:, 0, 0] = 1.0
    H[:, 0, 1] = 1.0                                     # same component -> coherent
    m, C = 5.0, 1e14
    g0, g1 = _exp_stacks(f, 0.30, 0.05, speed0=100.0)
    mc = fem.freq_evolutionary_multi_input_monte_carlo_damage(
        omega, H, G, [4000.0] * 4, m, C, seed=11, gamma0=g0, gamma1=g1,
        measure=True)
    summ = fem.freq_evolutionary_multi_input_summary(
        omega, H, G, [4000.0] * 4, m, C, gamma0=g0, gamma1=g1)
    # compare the measured band spectrum to the target band spectrum, window by
    # window, band by band (the SHAPE, not just the band-mean)
    for i, w in enumerate(summ["windows"]):
        measured = np.asarray(mc["window_gamma_spectrum"][i], dtype=float)
        target = np.asarray(w["band_gamma"], dtype=float)
        assert np.allclose(measured, target, atol=0.12)   # within Welch scatter
        # the shape is genuinely NON-flat (low band coherent, high band decorrelated)
        assert target[0] > target[-1] + 0.1
    # the low band stays highly coherent across the mission; the high band recovers
    band0 = np.array([np.asarray(s)[0] for s in mc["window_gamma_spectrum"]])
    bandN = np.array([np.asarray(s)[-1] for s in mc["window_gamma_spectrum"]])
    assert np.all(band0 > 0.8)                            # low band coherent
    assert bandN[-1] > bandN[0] - 0.1                     # high band recovers up


# ============================================================================
# CARDS + END-TO-END + NO-REGRESSION
# ============================================================================

def _parse(text):
    from pyradioss.input.deck_reader import read_deck
    from pyradioss.input.engine_keywords import parse_engine_deck
    d = tempfile.mkdtemp()
    ep = os.path.join(d, "M30_0001.rad")
    with open(ep, "w") as f:
        f.write(text)
    with contextlib.redirect_stdout(io.StringIO()):
        return parse_engine_deck(read_deck(ep), MessageLog())


def test_fcoh_card_parsing_exponential():
    """/IMPL/FATIG/MULT/MINPUT/EVOL/FCOH mirrors the TIME-VARYING exponential
    decay/speed pair (decay1, speed1) on the multi-input header and implies
    MINPUT + EVOL + MULT."""
    ec = _parse(
        "#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/MINPUT/EVOL/FCOH\n"
        "0.0 400.0 400 10 6\n5.0 1.0e14 0.03 0.0 0.0 40.0 7\n"
        "60.0 180.0 8.0 8.0 5\n"
        "2 1 0.0 0.0 0.9 100.0 -1 0.0 0.2 200.0\n"
        "1 10 0.0 0.0 0.0\n2 11 3.0 0.0 0.0\n/END\n")
    assert ec.impl_mi_fcoh and ec.impl_fatig_minput and ec.impl_fatig_evol
    assert ec.impl_fatig_mult
    assert ec.impl_mi_cohmodel == 1
    assert ec.impl_mi_decay == pytest.approx(0.9)
    assert ec.impl_mi_speed == pytest.approx(100.0)
    assert ec.impl_mi_decay1 == pytest.approx(0.2)
    assert ec.impl_mi_speed1 == pytest.approx(200.0)
    assert ec.impl_mi_inputs == ((1, 10, 0.0, 0.0, 0.0), (2, 11, 3.0, 0.0, 0.0))


def test_fcoh_card_parsing_measured():
    """The MEASURED coherence-shape /FUNCT pair (gfunct0, gfunct1) is read from the
    trailing header columns."""
    ec = _parse(
        "#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/MINPUT/EVOL/FCOH\n"
        "0.0 400.0 400 10 6\n5.0 1.0e14 0.03 0.0 0.0 40.0 7\n"
        "60.0 180.0 8.0 8.0 5\n"
        "2 0 0.0 0.0 0.0 0.0 -1 0.0 -1 -1 20 21\n1 10\n2 11\n/END\n")
    assert ec.impl_mi_fcoh
    assert ec.impl_mi_gfunct0 == 20
    assert ec.impl_mi_gfunct1 == 21


def test_no_fcoh_leaves_flag_off():
    """Without /FCOH the frequency-dependent-coherence flag stays off (the M28/M29
    paths are unaffected)."""
    ec = _parse(
        "#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/MINPUT/EVOL\n"
        "0.0 400.0 400 10 6\n5.0 1.0e14 0.03 0.0 0.0 40.0 7\n"
        "60.0 180.0 8.0 8.0 5\n2 0 0.1 0.0 0.0 0.0 0.9 0.0\n1 10\n2 11\n/END\n")
    assert ec.impl_mi_fcoh is False


def test_freq_evolutionary_multi_input_end_to_end():
    """The solid-brick cantilever under TWO inputs with a CONVECTION FIELD whose
    decorrelation moves UP in frequency (decay 0.9 -> 0.2): the frequency-dependent
    evolutionary multi-input life is reported ALONGSIDE the M28 stationary and M29
    scalar-coherence lives, and the measured coherence spectrum tracks the target."""
    eng = ("#\n/RUN/BR/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/MINPUT/EVOL/FCOH\n"
           "0.0 400.0 400 10 6\n5.0 1.0e14 0.03 0.0 0.0 40.0 7\n"
           "60.0 180.0 80.0 80.0 5\n"
           "2 1 0.0 0.0 0.9 100.0 -1 0.0 0.2 100.0\n"
           "1 10 0.0 0.0 0.0\n2 11 3.0 0.0 0.0\n"
           "/PRINT/-500\n/STOP\n15.0\n")
    model, out = _run(_two_input_brick(), eng, capture=True)
    mi = model.implicit_result.fatigue["multi_input"]
    fe = mi["freq_evolutionary_multi_input"]
    assert fe is not None
    assert fe["freq_dependent"] is True
    assert fe["cohmodel"] == 1
    assert fe["von_mises"]["damage_rate"] > 0.0
    assert fe["decorr_drift"] > 0.0                  # decorrelation moves up
    # the M28 stationary + M29 scalar references are reported alongside
    assert fe["stationary_multi_input"]["von_mises"] > 0.0
    assert fe["scalar_evolutionary_multi_input"] is not None
    # the non-stationary MULTI-INPUT MC ran and measured the coherence spectrum
    assert fe["monte_carlo"] is not None
    spec = fe["monte_carlo"]["window_gamma_spectrum"]
    assert len(spec) == 5
    assert np.asarray(spec[0]).size == 4             # per-band coherence spectrum
    assert "FREQUENCY-DEPENDENT EVOLUTIONARY MULTI-INPUT" in out
    assert "BAND-COHERENCE" in out


def test_m28_m29_byte_identical_with_fcoh():
    """The M28 STATIONARY + M29 scalar-coherence reductions are byte-unchanged whether
    or not the M30 frequency-dependent-coherence path runs (a NEW parallel path)."""
    # IDENTICAL headers except the /FCOH flag, so the M28 stationary coherence model
    # (exponential decay 0.9) and the M29 scalar-coherence drift are the SAME in both;
    # /FCOH only ADDS the M30 frequency-dependent path on top.
    hdr = ("0.0 400.0 400 10 6\n5.0 1.0e14 0.03 0.0 0.0 40.0 7\n"
           "60.0 180.0 8.0 8.0 5\n"
           "2 1 0.5 0.0 0.9 100.0 0.9 0.0 0.2 100.0\n"
           "1 10 0.0 0.0 0.0\n2 11 3.0 0.0 0.0\n"
           "/PRINT/-500\n/STOP\n15.0\n")
    base = "#\n/RUN/BR/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/MINPUT/EVOL\n" + hdr
    fcoh = "#\n/RUN/BR/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/MINPUT/EVOL/FCOH\n" + hdr
    m1 = _run(_two_input_brick(), base).implicit_result.fatigue["multi_input"]
    m2 = _run(_two_input_brick(), fcoh).implicit_result.fatigue["multi_input"]
    assert m1.get("freq_evolutionary_multi_input") is None
    assert m2.get("freq_evolutionary_multi_input") is not None
    # the M28 stationary multi-input reductions are byte-identical
    for key in ("von_mises", "normal_plane", "shear_plane"):
        assert (m1[key]["summary"]["dirlik"]["damage_rate"]
                == m2[key]["summary"]["dirlik"]["damage_rate"])
    assert np.array_equal(m1["Scross"], m2["Scross"])


def test_freq_evolutionary_multi_input_read_only():
    """The M30 recovery is read-only: a repeated run gives identical element stresses
    and identical M28 + M30 damage."""
    eng = ("#\n/RUN/BR/1\n1.0\n/IMPL\n/IMPL/FATIG/MULT/MINPUT/EVOL/FCOH\n"
           "0.0 400.0 400 10 6\n5.0 1.0e14 0.03 0.0 0.0 0.0 7\n"
           "60.0 180.0 80.0 80.0 5\n"
           "2 1 0.0 0.0 0.9 100.0 -1 0.0 0.2 100.0\n"
           "1 10 0.0 0.0 0.0\n2 11 3.0 0.0 0.0\n"
           "/PRINT/-500\n/STOP\n15.0\n")
    a = _run(_two_input_brick(), eng).implicit_result.fatigue["multi_input"]
    b = _run(_two_input_brick(), eng).implicit_result.fatigue["multi_input"]
    assert np.array_equal(a["Scross"], b["Scross"])
    assert (a["freq_evolutionary_multi_input"]["von_mises"]["damage_rate"]
            == b["freq_evolutionary_multi_input"]["von_mises"]["damage_rate"])
