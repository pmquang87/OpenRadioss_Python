"""
M32 validations: NON-GAUSSIAN INSTANTANEOUS-TENSOR TIME-FREQUENCY DISTRIBUTION
(/IMPL/FATIG/NGAUSS + /WVILLE) — a per-instant, time-VARYING NON-GAUSSIAN (kurtosis /
skewness) correction of the CONTINUOUS Wigner-Ville instantaneous stress spectrum
(M31), so the leptokurtic damage amplification lambda_ng(t) itself DRIFTS with time
along the continuous spectrum, reduced per instant and Palmgren-Miner INTEGRATED,
cross-validated by a non-Gaussian NON-STATIONARY Monte-Carlo. M32 is the CONVERGENCE
of M24 (stationary Winterstein-Hermite kurtosis correction of the equivalent scalar)
and M31 (the continuous instantaneous tensor spectrum). Built ALONGSIDE the M20-M31
spectral fatigue (all stay bit-identical; the M24 stationary-non-Gaussian and the M31
continuous-Gaussian paths stay byte-identical, the time-varying non-Gaussian path is a
NEW path ALONGSIDE them).

Every new capability gets at least one ANALYTIC / EXACT-REDUCTION check (the port's
philosophy):

THE M32 <-> M24 / M31 REDUCTIONS (built in, exact)
* gamma_4(t) == 3 (Gaussian at every instant) -> lambda_ng(t) == 1 EXACTLY, the M32
  answer DELEGATES to the M31 Gaussian continuous answer BYTE-IDENTICALLY (scalar +
  tensor, summary + Monte-Carlo);
* a CONSTANT kurtosis with the bandwidth attenuation OFF -> a CONSTANT lambda_ng that
  FACTORS out: the M32 answer is EXACTLY lambda_ng * (the M31 continuous Gaussian
  answer) — the M24 correction applied to the M31 continuous spectrum;
* a STATIONARY process with a CONSTANT kurtosis -> EXACTLY the M24 stationary answer
  (M31 recovers the stationary reduction at every instant);
* the WINDOWED limit (refine = 1, smooth = 0) -> the per-window non-Gaussian
  Miner-sum, the M24 correction applied to the M27 windowed joint-tensor EXACTLY (the
  Gaussian per-window reduction byte-identical to M27 before lambda_ng scales it).

THE TIME-VARYING POINT
* a genuinely time-varying kurtosis whose per-instant amplification lambda_ng(t)
  DRIFTS continuously (min .. max), amplifying the continuous-integral damage above
  the Gaussian one;
* the non-Gaussian NON-STATIONARY Monte-Carlo's induced sample kurtosis tracking
  gamma_4(t); the Gaussian-limit MC bit-identical to the M31 MC.

CARDS + NO-REGRESSION (the M7 parity contract)
* /IMPL/FATIG/NGAUSS composing with /WVILLE + a kurtosis-vs-time /FUNCT (col 3 of the
  M24 kurtosis line) or a linear sweep (col 4);
* the M24 stationary-non-Gaussian AND the M31 continuous-Gaussian answers byte-
  identical whether or not the M32 time-varying path runs.

See PORTING_GUIDE.md roadmap M32.
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

from pyradioss.common.messages import MessageLog                          # noqa: E402
from pyradioss.implicit import nongaussian_wigner_ville_fatigue as ngwv    # noqa: E402
from pyradioss.implicit import wigner_ville_fatigue as wv                  # noqa: E402
from pyradioss.implicit import nongaussian_fatigue as ngf                  # noqa: E402
from pyradioss.implicit import joint_evolutionary_fatigue as jf           # noqa: E402
from pyradioss.implicit import evolutionary_fatigue as ef                 # noqa: E402

from tests.test_m20_fatigue import _chain_deck                            # noqa: E402
from tests.test_m21_multiaxfatig import _brick_deck                       # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _run(starter_text, engine_text, capture=False):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M32_0000.rad")
    ep = os.path.join(d, "M32_0001.rad")
    with open(sp, "w") as f:
        f.write(starter_text)
    with open(ep, "w") as f:
        f.write(engine_text)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_starter(sp)
        model = run_engine(ep)
    return (model, buf.getvalue()) if capture else model


_MISSION = ("/FUNCT/20\nmission\n       0.0       0.5\n"
            "     100.0       1.5\n     200.0       1.5\n"
            "     300.0       0.3\n")

# a kurtosis-vs-time /FUNCT: Gaussian -> leptokurtic burst -> relax
_KFUN = ("/FUNCT/30\nkurt\n       0.0       3.0\n"
         "     150.0       8.0\n     400.0       3.5\n")


def _with(deck, *extra):
    return deck.replace("/END", "".join(extra) + "/END", 1)


def _bimodal_psd(nf=20000, fmax=200.0):
    f = np.linspace(1e-3, fmax, nf)
    S = (np.exp(-((f - 40.0) ** 2) / (2.0 * 3.0 ** 2))
         + 0.5 * np.exp(-((f - 120.0) ** 2) / (2.0 * 4.0 ** 2))) / 2.0
    return f, S


def _tensor_cross_psd(nf=400, fmax=600.0):
    f = np.linspace(1e-3, fmax, nf)
    omega = 2.0 * np.pi * f
    S = np.zeros((nf, 6, 6), dtype=complex)
    low = np.exp(-((f - 100.0) ** 2) / (2.0 * 20.0 ** 2))
    high = np.exp(-((f - 450.0) ** 2) / (2.0 * 30.0 ** 2))
    S[:, 0, 0] = low + 0.05
    S[:, 3, 3] = high + 0.02
    S[:, 0, 3] = 0.3 * np.sqrt(low * high)
    S[:, 3, 0] = np.conj(S[:, 0, 3])
    return omega, S


# ============================================================================
# kurtosis_schedule primitive
# ============================================================================

def test_kurtosis_schedule_forms():
    s = np.linspace(0.0, 1.0, 5)
    # constant
    g4, g3 = ngwv.kurtosis_schedule(s, 5.0)
    assert np.allclose(g4, 5.0) and np.allclose(g3, 0.0)
    # linear sweep
    g4, g3 = ngwv.kurtosis_schedule(s, (3.0, 8.0), skew=(0.0, 0.5))
    assert g4[0] == 3.0 and g4[-1] == 8.0
    assert g3[0] == 0.0 and g3[-1] == 0.5
    # explicit grid
    grid = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    g4, _ = ngwv.kurtosis_schedule(s, 3.0, kurt_grid=grid)
    assert np.array_equal(g4, grid)
    # Gaussian detection
    assert ngwv._is_gaussian_schedule(np.full(5, 3.0), np.zeros(5))
    assert not ngwv._is_gaussian_schedule(grid, np.zeros(5))


# ============================================================================
# THE M32 <-> M31 REDUCTION (gamma_4 == 3 -> M31 Gaussian, byte-identical)
# ============================================================================

def test_scalar_gaussian_limit_byte_identical_to_m31():
    """gamma_4(t) == 3: the M32 scalar summary DELEGATES to the M31 continuous
    Gaussian summary BYTE-IDENTICALLY (every estimator rate identical)."""
    f, S = _bimodal_psd()
    m, C = 5.0, 1e14
    durs = [30.0, 50.0, 20.0, 15.0]
    scales = [0.6, 1.4, 1.0, 1.8]
    g = ngwv.nongaussian_wigner_ville_summary(
        f, S, durs, (30.0, 120.0), (5.0, 20.0), m, C, kurt=3.0, scales=scales,
        refine=6, smooth=0.0)
    gm31 = wv.wigner_ville_fatigue_summary(
        f, S, durs, (30.0, 120.0), (5.0, 20.0), m, C, scales=scales, refine=6,
        smooth=0.0)
    assert g["delegated"] == "m31_gaussian"
    for est in ("narrow_band", "dirlik", "wirsching_light", "tovo_benasciutti"):
        assert g[est]["damage_rate"] == gm31[est]["damage_rate"]


def test_tensor_gaussian_limit_byte_identical_to_m31():
    """gamma_4(t) == 3: the M32 tensor summary DELEGATES to the M31 continuous
    Gaussian tensor summary BYTE-IDENTICALLY (per reduction)."""
    omega, Scross = _tensor_cross_psd()
    m, C = 5.0, 1e4
    durs = [10.0] * 6
    g = ngwv.nongaussian_wigner_ville_tensor_summary(
        omega, Scross, durs, (120.0, 480.0), (20.0, 40.0), m, C, kurt=3.0,
        refine=6, smooth=0.0)
    gm31 = wv.wigner_ville_tensor_summary(
        omega, Scross, durs, (120.0, 480.0), (20.0, 40.0), m, C, refine=6,
        smooth=0.0)
    assert g["delegated"] == "m31_gaussian"
    for k in ("von_mises", "normal_plane", "shear_plane"):
        assert g[k]["damage_rate"] == gm31[k]["damage_rate"]


# ============================================================================
# THE M32 <-> M24 REDUCTIONS (constant kurtosis -> M24 factoring, exact)
# ============================================================================

def test_scalar_constant_kurtosis_factors_exactly():
    """A CONSTANT kurtosis with the bandwidth attenuation OFF gives a CONSTANT
    lambda_ng that FACTORS: the M32 scalar rate is EXACTLY lambda_ng * (the M31
    continuous Gaussian rate) — the M24 correction on the M31 continuous answer."""
    f, S = _bimodal_psd()
    m, C = 5.0, 1e14
    durs = [30.0, 50.0, 20.0, 15.0]
    scales = [0.6, 1.4, 1.0, 1.8]
    g = ngwv.nongaussian_wigner_ville_summary(
        f, S, durs, (30.0, 120.0), (5.0, 20.0), m, C, kurt=5.0, scales=scales,
        refine=6, smooth=0.0, bandwidth_correction=False)
    gm31 = wv.wigner_ville_fatigue_summary(
        f, S, durs, (30.0, 120.0), (5.0, 20.0), m, C, scales=scales, refine=6,
        smooth=0.0)
    lam = ngf.nongaussian_correction_factor(0.0, 5.0, m, alpha2=1.0,
                                            bandwidth_correction=False)
    for est in ("narrow_band", "dirlik", "wirsching_light", "tovo_benasciutti"):
        assert g[est]["damage_rate"] == lam * gm31[est]["damage_rate"]


def test_scalar_stationary_constant_kurtosis_equals_m24():
    """A STATIONARY process with a CONSTANT kurtosis recovers the M24 stationary
    answer EXACTLY (M31 recovers the stationary reduction at every instant, so the
    Miner-integral is lambda_ng * the stationary damage)."""
    f, S = _bimodal_psd()
    m, C = 5.0, 1e14
    g = ngwv.nongaussian_wigner_ville_summary(
        f, S, [100.0] * 3, 0.0, 0.0, m, C, kurt=6.0, refine=8, smooth=0.0,
        bandwidth_correction=True)
    m24 = ngf.nongaussian_summary(ef._moments_of_psd(f, S), m, C, 0.0, 6.0,
                                  bandwidth_correction=True)
    for est in ("narrow_band", "dirlik", "wirsching_light", "tovo_benasciutti"):
        assert g[est]["damage_rate"] == pytest.approx(
            m24[est]["damage_rate"], rel=1e-9)


def test_tensor_windowed_limit_equals_m24_on_m27():
    """refine = 1, smooth = 0 with a CONSTANT kurtosis (bandwidth OFF): the Gaussian
    per-window tensor reduction is BYTE-IDENTICAL to M27, and the M32 rate is EXACTLY
    lambda_ng * M27 — the M24 correction applied to the M27 windowed joint-tensor."""
    omega, Scross = _tensor_cross_psd()
    m, C = 5.0, 1e4
    durs = [10.0] * 6
    gw = ngwv.nongaussian_wigner_ville_tensor_summary(
        omega, Scross, durs, (120.0, 480.0), (20.0, 40.0), m, C, kurt=5.0,
        refine=1, smooth=0.0, bandwidth_correction=False)
    m27 = jf.joint_evolutionary_fatigue_summary(
        omega, Scross, durs, fc=(120.0, 480.0), bw=(20.0, 40.0), m=m, C=C,
        drift=True)
    lam = ngf.nongaussian_correction_factor(0.0, 5.0, m, alpha2=1.0,
                                            bandwidth_correction=False)
    for k in ("von_mises", "normal_plane", "shear_plane"):
        assert gw[k]["gaussian_damage_rate"] == m27[k]["damage_rate"]
        assert gw[k]["damage_rate"] == lam * m27[k]["damage_rate"]


# ============================================================================
# THE TIME-VARYING POINT
# ============================================================================

def test_scalar_time_varying_kurtosis_drifts():
    """A genuinely time-varying kurtosis (a chirp with gamma_4 sweeping 3 -> 8): the
    per-instant amplification lambda_ng(t) DRIFTS continuously (min < max) and the
    continuous-integral damage exceeds the Gaussian one."""
    f, S = _bimodal_psd()
    m, C = 5.0, 1e14
    g = ngwv.nongaussian_wigner_ville_summary(
        f, S, [1.0] * 6, (35.0, 125.0), 8.0, m, C, kurt=(3.0, 8.0), refine=8,
        smooth=0.0)
    assert g["lambda_max"] > g["lambda_min"] > 0.9
    assert g["gamma4_range"] > 3.0
    assert g["damage_rate"] > g["gaussian"]["damage_rate"]
    # the per-instant lambda array follows the kurtosis ramp (monotone-ish up)
    assert np.asarray(g["lambda_ng"])[-1] > np.asarray(g["lambda_ng"])[0]


def test_tensor_time_varying_kurtosis_drifts():
    """The 6x6 tensor continuous spectrum re-searches the plane AND applies a
    per-instant, per-reduction lambda_ng(t): a swept kurtosis amplifies every
    reduction above its Gaussian rate and drifts lambda_ng."""
    omega, Scross = _tensor_cross_psd()
    m, C = 5.0, 1e4
    durs = [10.0] * 6
    g = ngwv.nongaussian_wigner_ville_tensor_summary(
        omega, Scross, durs, (120.0, 480.0), (20.0, 40.0), m, C, kurt=(3.0, 8.0),
        refine=6, smooth=0.0)
    assert g["lambda_max"] > g["lambda_min"]
    assert g["plane_rotation_deg"] > 0.0          # M31 plane drift preserved
    for k in ("von_mises", "normal_plane", "shear_plane"):
        assert g[k]["damage_rate"] > g[k]["gaussian_damage_rate"]


def test_mc_gaussian_limit_bit_identical_to_m31():
    """The Gaussian-schedule non-Gaussian Monte-Carlo (gamma_4 == 3) is the M31
    continuous Monte-Carlo BIT-IDENTICALLY (delegation)."""
    f, S = _bimodal_psd(nf=4000)
    m, C = 5.0, 1e14
    mc = ngwv.nongaussian_wigner_ville_monte_carlo_damage(
        f, S, [1000.0] * 3, (40.0, 120.0), 6.0, m, C, seed=11, kurt=3.0, refine=1,
        smooth=0.0, fs=1600.0)
    mc31 = wv.wigner_ville_monte_carlo_damage(
        f, S, [1000.0] * 3, (40.0, 120.0), 6.0, m, C, 11, refine=1, smooth=0.0,
        fs=1600.0)
    assert mc["damage_rate"] == mc31["damage_rate"]


def test_mc_induced_kurtosis_tracks_target():
    """The non-Gaussian non-stationary Monte-Carlo's induced sample kurtosis tracks
    gamma_4(t): a strongly leptokurtic target lifts the record's sample kurtosis well
    above the Gaussian 3, and the non-Gaussian damage exceeds the Gaussian MC."""
    f, S = _bimodal_psd(nf=4000)
    m, C = 5.0, 1e14
    mc = ngwv.nongaussian_wigner_ville_monte_carlo_damage(
        f, S, [1000.0] * 3, (40.0, 120.0), 6.0, m, C, seed=11, kurt=8.0, refine=8,
        smooth=0.0, fs=1600.0)
    mc31 = wv.wigner_ville_monte_carlo_damage(
        f, S, [1000.0] * 3, (40.0, 120.0), 6.0, m, C, 11, refine=8, smooth=0.0,
        fs=1600.0)
    assert mc["kurtosis"] > 4.0                  # leptokurtic record
    assert mc["damage_rate"] > mc31["damage_rate"]


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


def test_impl_fatig_ngauss_wville_card_parsing():
    """/IMPL/FATIG/NGAUSS/WVILLE reads the kurtosis-vs-time /FUNCT (col 3) and/or the
    end kurtosis for a linear sweep (col 4) from the M24 kurtosis line
    (kurt [skew [kfunct [kurt1]]]); it composes /NGAUSS with /WVILLE (implying
    /EVOL)."""
    # kurtosis /FUNCT id on col 3
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/FATIG/WVILLE/NGAUSS/BASE\n"
                   "0.0 250 2000 7 0 4\n5.0 1.0e12 0.03\n3.0 0.0 30\n"
                   "20.0 120.0 5.0 30.0 8 12 0.25\n/END\n")
    assert ec.impl_fatig_ngauss and ec.impl_fatig_wville and ec.impl_fatig_evol
    assert ec.impl_fatig_kfunct == 30
    assert ec.impl_fatig_wv_refine == 12

    # end kurtosis (linear sweep) on col 4
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/FATIG/MULT/EVOL/JOINT/WVILLE/NGAUSS\n"
                   "0.0 400 500 10\n5.0 1e4 0.03\n3.0 0.0 0 8.0\n"
                   "50.0 250.0 10.0 40.0 6 8\n/END\n")
    assert ec.impl_fatig_joint and ec.impl_fatig_wville and ec.impl_fatig_ngauss
    assert ec.impl_fatig_kurt == 3.0 and ec.impl_fatig_kurt1 == 8.0

    # a plain /NGAUSS (no WVILLE) keeps the M32 fields at defaults
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/FATIG/NGAUSS\n"
                   "0.0 250 2000 7\n5.0 1e12\n5.0 0.0\n/END\n")
    assert ec.impl_fatig_ngauss and not ec.impl_fatig_wville
    assert ec.impl_fatig_kfunct == 0 and ec.impl_fatig_kurt1 == 0.0


# ============================================================================
# END-TO-END
# ============================================================================

def test_ngwville_scalar_end_to_end():
    """/IMPL/FATIG/WVILLE/NGAUSS end to end with a kurtosis-vs-time /FUNCT: the
    listing prints the TIME-VARYING NON-GAUSSIAN INSTANTANEOUS block ALONGSIDE the
    M31 Gaussian-continuous and the M24 stationary-non-Gaussian ones; the
    ``wigner_ville`` entry carries a ``nongaussian`` sub-entry whose lambda_ng(t)
    drifts and whose damage exceeds the Gaussian one."""
    starter = _with(_chain_deck([2e-3] * 5, [800.] * 5, [0.] * 5), _MISSION, _KFUN)
    eng = ("#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/FATIG/WVILLE/NGAUSS/NSTAT/BASE\n"
           "0.0 250.0 3000 2 0 5\n5.0 1.0e14 0.03 0.0 0.0 400.0 123\n3.0 0.0 30\n"
           "20 8\n20.0 120.0 5.0 30.0 8 8 0.0\n/PRINT/-500\n/STOP\n15.0\n")
    m, out = _run(starter, eng, capture=True)
    wvr = m.implicit_result.fatigue["wigner_ville"]
    ng = wvr["nongaussian"]
    assert ng is not None and not ng["tensor"]
    assert ng["lambda_max"] > ng["lambda_min"]
    assert ng["damage_rate"] > ng["gaussian_damage_rate"]
    assert ng["monte_carlo"]["kurtosis"] > 4.0
    assert "TIME-VARYING NON-GAUSSIAN INSTANTANEOUS" in out
    assert "CONTINUOUS WIGNER-VILLE INSTANTANEOUS" in out
    assert "NON-GAUSSIAN / KURTOSIS FATIGUE" in out       # M24 stationary alongside


def test_ngwville_tensor_end_to_end():
    """/IMPL/FATIG/MULT/EVOL/JOINT/WVILLE/NGAUSS: the continuous TENSOR spectrum with
    a per-instant plane re-search AND a per-instant lambda_ng(t) (a kurtosis sweep),
    reported ALONGSIDE the M31 Gaussian-continuous tensor."""
    eng = ("#\n/RUN/BRICK/1\n1.0\n/IMPL\n"
           "/IMPL/FATIG/MULT/EVOL/JOINT/WVILLE/NGAUSS/NSTAT\n"
           "0.0 400.0 500 10 6\n5.0 1.0e4 0.03 0.0 0.0 15.0 21000\n3.0 0.0 0 8.0\n"
           "20 6\n50.0 250.0 10.0 40.0 6 6 0.0\n/PRINT/-500\n/STOP\n15.0\n")
    m, out = _run(_with(_brick_deck(nx=3), _MISSION), eng, capture=True)
    wvr = m.implicit_result.fatigue["wigner_ville"]
    ng = wvr["nongaussian"]
    assert ng is not None and ng["tensor"]
    assert ng["lambda_max"] > ng["lambda_min"]
    for key in ("von_mises", "normal_plane", "shear_plane"):
        assert ng[key]["damage_rate"] >= ng["gaussian"][key]
    assert "TIME-VARYING NON-GAUSSIAN INSTANTANEOUS" in out


# ============================================================================
# NO-REGRESSION (the M7 parity contract)
# ============================================================================

def test_m24_m31_byte_identical_with_without_ngwville():
    """The M24 stationary-non-Gaussian AND the M31 continuous-Gaussian scalar answers
    are BYTE-IDENTICAL whether or not the M32 time-varying path runs — the M32 path is
    NEW and ALONGSIDE."""
    starter = _with(_chain_deck([2e-3] * 5, [800.] * 5, [0.] * 5), _MISSION, _KFUN)
    base = ("#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/FATIG/WVILLE/NSTAT/BASE\n"
            "0.0 250.0 3000 2 0 5\n5.0 1.0e14 0.03 0.0 0.0 400.0 123\n"
            "20 8\n20.0 120.0 5.0 30.0 8 8 0.0\n/PRINT/-500\n/STOP\n15.0\n")
    m32 = ("#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/FATIG/WVILLE/NGAUSS/NSTAT/BASE\n"
           "0.0 250.0 3000 2 0 5\n5.0 1.0e14 0.03 0.0 0.0 400.0 123\n3.0 0.0 30\n"
           "20 8\n20.0 120.0 5.0 30.0 8 8 0.0\n/PRINT/-500\n/STOP\n15.0\n")
    a = _run(starter, base)
    b = _run(starter, m32)
    fa = a.implicit_result.fatigue
    fb = b.implicit_result.fatigue
    # M31 continuous-Gaussian byte-identical
    assert fa["wigner_ville"]["damage_rate"] == fb["wigner_ville"]["damage_rate"]
    assert fa["wigner_ville"].get("nongaussian") is None
    assert fb["wigner_ville"]["nongaussian"] is not None
    for est in ("narrow_band", "dirlik", "wirsching_light", "tovo_benasciutti"):
        assert (fa["summary"][est]["damage_rate"]
                == fb["summary"][est]["damage_rate"])
    # the M31 continuous Monte-Carlo untouched
    assert (fa["wigner_ville"]["monte_carlo"]["damage_rate"]
            == fb["wigner_ville"]["monte_carlo"]["damage_rate"])


def test_ngwville_does_not_mutate_state():
    """The M32 time-varying non-Gaussian path is read-only in the element state (the
    M14-M31 parity contract extended to M32)."""
    starter = _chain_deck([2e-3] * 3, [800.] * 3, [0.] * 3)
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M32_0000.rad")
    with open(sp, "w") as f:
        f.write(starter)
    with contextlib.redirect_stdout(io.StringIO()):
        m = run_starter(sp)
    before = {k: (v.copy() if isinstance(v, np.ndarray) else v)
              for k, v in m.springs.state.items()}
    f, S = _bimodal_psd()
    ngwv.nongaussian_wigner_ville_summary(
        f, S, [1.0] * 5, (30.0, 130.0), (5.0, 25.0), 5.0, 1e14, kurt=(3.0, 7.0),
        scales=[0.5, 1., 1.5, 1., .5], refine=6, smooth=0.2)
    for k, v in before.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(m.springs.state[k], v), k
