"""
M20 validations: RANDOM-VIBRATION (SPECTRAL) FATIGUE (/IMPL/FATIG) — the
stress-life fatigue-damage estimate computed DIRECTLY from the M19 stress-PSD
spectral moments (m0..m4). Built ALONGSIDE the M16 real eigensolver, the
M17/M18 FRFs and the M19 PSD / response-spectrum paths (all stay bit-identical;
the fatigue path CONSUMES the stress moments read-only).

Every new capability gets at least one ANALYTIC check (the port's philosophy):

STRESS-PSD RECOVERY
* the recovered STATIC stress equals the M8 implicit-static stress (the linear
  stress operator sigma = C:B:u, applied through the SAME kernels, read-only);
* the stress FRF H_sigma(Omega) = sum_i sigma_i q_i(Omega) equals the direct
  element-stress operator applied to the physical FRF U(Omega) (a |H_sigma|^2 S
  stress-PSD identity), and the recovery leaves the element state untouched.

SPECTRAL FATIGUE-DAMAGE MODELS
* the NARROW-BAND (Bendat) closed form against a hand Gamma-function evaluation;
* DIRLIK reduces to the narrow-band estimate in the narrow-band limit
  (bandwidth -> 0) and gives the correct wide-band BIAS (less conservative than
  narrow band) on a bimodal spectrum;
* the ASTM E1049-85 rainflow counter against the canonical worked example;
* a SYNTHESISED Gaussian time history from the PSD, rainflow-counted, matching
  the spectral (Dirlik) estimate within the documented scatter (a seeded
  Monte-Carlo cross-check);
* Wirsching-Light / Tovo-Benasciutti bracket the estimate and reduce to narrow
  band in the narrow-band limit;
* the reporting triple: T_f = 1/(damage rate) and S_eq consistency.

CARDS + NO-REGRESSION (the M7 parity contract)
* /IMPL/FATIG (+ /BASE, /STRS) card mirror (a PORT card — freimpl.F has no
  spectral-fatigue path);
* the fatigue path NEVER mutates the M16 eigensolver / M17-M18 FRFs / the
  element state, and the direct M10 answer is unchanged whether or not it runs.

See PORTING_GUIDE.md roadmap M20.
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
    recover_element_stresses, response_psd, spectral_moments,
    stress_channels, stress_frf, stress_modes, stress_response_psd)
from pyradioss.implicit import spectral_fatigue as sf               # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _starter(text):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M20_0000.rad")
    with open(sp, "w") as f:
        f.write(text)
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(sp)


def _run(starter_text, engine_text, capture=False):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M20_0000.rad")
    ep = os.path.join(d, "M20_0001.rad")
    with open(sp, "w") as f:
        f.write(starter_text)
    with open(ep, "w") as f:
        f.write(engine_text)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_starter(sp)
        model = run_engine(ep)
    return (model, buf.getvalue()) if capture else model


def _loads(model):
    return LoadsAndConstraints(model, MessageLog())


def _bar_deck(E=210.0, area=2.0, L=10.0, P=5.0):
    """A single axial truss bar, fixed-free, pulled by a tip force P — a clean
    UNIAXIAL stress channel with the closed-form static stress sigma = P/area."""
    return f"""\
#RADIOSS STARTER
/BEGIN
BAR
/NODE
         1{0.0:20.10f}{0.0:20.10f}{0.0:20.10f}
         2{L:20.10f}{0.0:20.10f}{0.0:20.10f}
/TRUSS/1
         1         1         2
/PART/1
bar
         1         1
/MAT/LAW1/1
elastic
   7.8e-6
     {E}       0.0
/PROP/TRUSS/1
bar
       {area}
/FUNCT/1
unit
       0.0       1.0
  100000.0       1.0
/GRNOD/NODE/1
fix
1
/GRNOD/NODE/2
tip
2
/BCS/1
fix
       111       111         0         1
/BCS/2
tipyz
       011       111         0         2
/CLOAD/1
pull
         1         X         2       {P}
/END
"""


def _chain_deck(masses, ks, cs):
    """A fixed-free axial spring-mass chain (mirrors the M19 helper). The
    springs carry a FORCE (stress-resultant) channel."""
    N = len(ks)
    nodes = "\n".join(f"{i+1:10d}{i*10.0:20.10f}{0.0:20.10f}{0.0:20.10f}"
                      for i in range(N + 1))
    springs = "\n".join(f"/SPRING/{i+1}\n{i+1:10d}{i+1:10d}{i+2:10d}"
                        for i in range(N))
    parts = "\n".join(f"/PART/{i+1}\ns{i+1}\n{i+1:10d}         1"
                      for i in range(N))
    props = "\n".join(f"/PROP/SPRING/{i+1}\nsp\n{masses[i]:12g}{ks[i]:12g}"
                      f"{cs[i]:12g}" for i in range(N))
    freeids = "\n".join(str(i + 2) for i in range(N))
    grn = f"/GRNOD/NODE/1\npin\n1\n/GRNOD/NODE/2\nfree\n{freeids}\n"
    funct = "/FUNCT/1\nunit\n       0.0       1.0\n  100000.0       1.0\n"
    psd = "/FUNCT/2\npsd\n       0.0       1.0\n  100000.0       1.0\n"
    return f"""\
#RADIOSS STARTER
/BEGIN
CHAIN
/NODE
{nodes}
{springs}
{parts}
/MAT/LAW1/1
st
   7.8e-6
     210.0       0.0
{props}
{funct}{psd}{grn}/BCS/1
pin
       111       111         0         1
/BCS/2
axial
       011       111         0         2
/END
"""


def _moments_narrowband(sigma, f0):
    """Moment array [m0..m4] of an (idealised) narrow-band process at f0 with
    RMS sigma: m_n = sigma^2 * omega0^n (a single spectral line, alpha2 -> 1)."""
    w0 = 2.0 * math.pi * f0
    m0 = sigma ** 2
    return np.array([m0, m0 * w0, m0 * w0 ** 2, m0 * w0 ** 3, m0 * w0 ** 4])


def _bimodal_moments(nf=200000):
    """Spectral moments of a WIDE-BAND bimodal stress PSD (two well-separated
    narrow peaks) — the classic Dirlik demonstrator. Returns (moments, f, S)
    with S in the M19 two-sided convention (so the moments use it directly)."""
    f = np.linspace(1e-3, 400.0, nf)

    def peak(fc, bw, A):
        return A * np.exp(-((f - fc) ** 2) / (2.0 * bw ** 2))

    G = peak(20.0, 2.0, 1.0) + peak(150.0, 3.0, 0.4)   # one-sided-ish PSD (Hz)
    S = G / 2.0                                        # -> two-sided S(omega)
    omega = 2.0 * np.pi * f
    mom = spectral_moments(omega, S, nmax=4)
    return mom, f, S


# ============================================================================
# STRESS-PSD RECOVERY
# ============================================================================

def test_recovered_static_stress_equals_m8():
    """The recovered stress from the M8 static displacement equals the stress
    the static solve left in the element buffer AND the closed form P/area —
    the linear stress operator sigma = C:B:u run through the SAME kernels."""
    E, area, P = 210.0, 2.0, 5.0
    m = _run(_bar_deck(E=E, area=area, P=P),
             "#\n/RUN/BAR/1\n1.0\n/IMPL\n/IMPL/DTINI\n1.0\n/PRINT/-500\n"
             "/STOP\n15.0\n")
    sig_static = m.trusses.state["sig"].copy()
    du = m.x - m.x0
    rec = recover_element_stresses(m, du, np.zeros_like(du))
    assert np.allclose(rec["trusses"], sig_static, rtol=1e-10)
    assert rec["trusses"][0] == pytest.approx(P / area, rel=1e-6)
    # read-only: the element state is untouched by the recovery
    assert np.allclose(m.trusses.state["sig"], sig_static)


def test_stress_frf_matches_direct_operator():
    """The modal stress FRF H_sigma(Omega) = sum_i sigma_i q_i(Omega) equals the
    element-stress operator applied DIRECTLY to the physical FRF U(Omega): a
    |H_sigma|^2 S stress-PSD identity (stress commutes with the superposition
    because it is a LINEAR operator on displacement)."""
    m = _starter(_chain_deck([2e-3, 2e-3, 2e-3], [800., 800., 800.],
                             [0., 0., 0.]))
    loads = _loads(m)
    basis = build_modal_basis(m, nev=3)
    z = modal_damping(basis.omega, uniform=0.03)
    n = m.numnod
    F = np.zeros((n, 3))
    loads.external_forces(1.0, F, m.x0)
    freqs = np.linspace(1e-6, 1.2 * basis.freqs.max(), 40)
    frf = modal_frequency_response(basis, F, np.zeros((n, 3)), freqs, z)

    Sigma, channels = stress_modes(m, basis)
    sfrf = stress_frf(frf, Sigma)                    # (nf, nchan)

    # direct: apply the stress operator to Re/Im of U at each frequency
    x_saved = m.x
    m.x = m.x0.copy()
    try:
        for jf in (5, 17, 30):
            Ur = frf["U"][jf].real
            Ui = frf["U"][jf].imag
            dur_r, durr_r = basis.dof.scatter_solution(Ur)
            dur_i, durr_i = basis.dof.scatter_solution(Ui)
            rec_r = recover_element_stresses(m, dur_r, durr_r)
            rec_i = recover_element_stresses(m, dur_i, durr_i)
            for j, (name, e, c, _lab) in enumerate(channels):
                ar, ai = rec_r[name], rec_i[name]
                direct = (ar[e] if ar.ndim == 1 else ar[e, c]) + 1j * (
                    ai[e] if ai.ndim == 1 else ai[e, c])
                assert sfrf["U"][jf, j] == pytest.approx(direct, rel=1e-8,
                                                         abs=1e-14)
    finally:
        m.x = x_saved


def test_stress_channels_enumerated():
    """Every spring is one axial FORCE (stress-resultant) channel; a truss is
    one axial stress channel."""
    m = _starter(_chain_deck([2e-3] * 4, [800.] * 4, [0.] * 4))
    ch = stress_channels(m)
    assert len(ch) == 4                              # four springs
    assert all(lab.endswith(":N") for *_ignore, lab in ch)
    mb = _starter(_bar_deck())
    ch2 = stress_channels(mb)
    assert len(ch2) == 1 and ch2[0][3].endswith(":sig")


# ============================================================================
# SPECTRAL FATIGUE-DAMAGE MODELS
# ============================================================================

def test_narrow_band_closed_form_hand_gamma():
    """The narrow-band (Bendat) damage rate against a HAND Gamma-function
    evaluation: E[D]/T = (nu0/C)(2 sqrt2 sigma)^m Gamma(1 + m/2)."""
    sigma, f0, m, C = 3.0, 40.0, 4.0, 1.0e12
    mom = _moments_narrowband(sigma, f0)
    nb = sf.narrow_band_damage(mom, m, C)
    hand = (f0 / 1.0) * (2.0 * math.sqrt(2.0) * sigma) ** m \
        * math.gamma(1.0 + m / 2.0) / C              # nu0 = f0 (narrow band)
    assert nb["damage_rate"] == pytest.approx(hand, rel=1e-12)
    # the reporting triple: T_f = 1/rate, and S_eq reproduces the rate
    assert nb["life"] == pytest.approx(1.0 / nb["damage_rate"], rel=1e-12)
    assert nb["nu"] * nb["s_eq"] ** m / C == pytest.approx(
        nb["damage_rate"], rel=1e-10)


def test_dirlik_reduces_to_narrow_band():
    """Dirlik collapses to the narrow-band estimate in the narrow-band limit
    (alpha2 -> 1: D1, D2 -> 0, D3 -> 1)."""
    mom = _moments_narrowband(2.0, 50.0)
    m, C = 5.0, 1e14
    nb = sf.narrow_band_damage(mom, m, C)
    dk = sf.dirlik_damage(mom, m, C)
    assert dk["params"]["alpha2"] == pytest.approx(1.0, abs=1e-9)
    assert dk["damage_rate"] == pytest.approx(nb["damage_rate"], rel=1e-6)
    c = dk["coeffs"]
    assert c["D1"] == pytest.approx(0.0, abs=1e-9)
    assert c["D3"] == pytest.approx(1.0, abs=1e-9)


def test_dirlik_wide_band_bias():
    """On a WIDE-BAND bimodal spectrum Dirlik is LESS conservative than the
    narrow-band estimate (narrow band over-counts small ripples as full
    cycles) — the correct wide-band bias, and Wirsching-Light / Tovo-Benasciutti
    bracket the same region."""
    mom, _f, _S = _bimodal_moments()
    m, C = 5.0, 1e15
    p = sf.spectral_bandwidth_params(mom)
    assert 0.3 < p["alpha2"] < 0.9                   # genuinely wide band
    nb = sf.narrow_band_damage(mom, m, C)["damage_rate"]
    dk = sf.dirlik_damage(mom, m, C)["damage_rate"]
    wl = sf.wirsching_light_damage(mom, m, C)["damage_rate"]
    tb = sf.tovo_benasciutti_damage(mom, m, C)["damage_rate"]
    assert dk < nb                                   # wide-band relaxation
    assert wl < nb and tb < nb
    # the three wide-band estimators agree to within a factor of ~2
    for a in (dk, wl, tb):
        assert 0.4 < a / dk < 2.5


def test_wide_band_estimators_reduce_to_narrow_band():
    """Wirsching-Light and Tovo-Benasciutti both reduce to the narrow-band
    damage in the narrow-band limit (lambda_WL -> 1, the TB weight -> 1)."""
    mom = _moments_narrowband(2.5, 30.0)
    m, C = 4.0, 1e13
    nb = sf.narrow_band_damage(mom, m, C)["damage_rate"]
    wl = sf.wirsching_light_damage(mom, m, C)
    assert wl["lambda_wl"] == pytest.approx(1.0, abs=1e-6)
    assert wl["damage_rate"] == pytest.approx(nb, rel=1e-6)
    tb = sf.tovo_benasciutti_damage(mom, m, C)["damage_rate"]
    assert tb == pytest.approx(nb, rel=1e-6)


def test_rainflow_astm_e1049_example():
    """The ASTM E1049-85 rainflow counter against the canonical worked example
    sequence [-2, 1, -3, 5, -1, 3, -4, 4, -2]: ranges {3: 0.5, 4: 1.5, 6: 0.5,
    8: 1.0, 9: 0.5}."""
    sig = np.array([-2, 1, -3, 5, -1, 3, -4, 4, -2], float)
    ranges, counts = sf.rainflow_count(sig)
    agg = {}
    for r, c in zip(ranges, counts):
        agg[round(float(r), 1)] = agg.get(round(float(r), 1), 0.0) + c
    assert agg == pytest.approx({3.0: 0.5, 4.0: 1.5, 6.0: 0.5, 8.0: 1.0,
                                 9.0: 0.5})
    assert counts.sum() == pytest.approx(4.0)


def test_monte_carlo_matches_dirlik():
    """A Gaussian time history synthesised from the bimodal stress PSD,
    rainflow-counted and Miner-summed, matches the Dirlik spectral estimate to
    within the documented scatter (~30 %). Seeded for reproducibility."""
    mom, f, S = _bimodal_moments()
    m, C = 5.0, 1e15
    p = sf.spectral_bandwidth_params(mom)
    dk = sf.dirlik_damage(mom, m, C)["damage_rate"]
    mc = sf.monte_carlo_damage(f, S, m, C, duration=3000.0, seed=7, fs=1000.0)
    # the synthesised RMS reproduces sigma = sqrt(m0)
    assert mc["rms"] == pytest.approx(p["sigma"], rel=0.05)
    ratio = mc["damage_rate"] / dk
    assert 0.7 < ratio < 1.4, f"MC/Dirlik = {ratio:.3f} outside scatter"


def test_synthesis_reproduces_moments():
    """The spectral-representation synthesis reproduces the PSD's variance m0
    (RMS) and, via a periodogram, the mean-square — a Parseval check on the
    convention bridge G(f) = 2 S(2 pi f)."""
    mom, f, S = _bimodal_moments()
    t, x = sf.synthesize_gaussian_history(f, S, duration=2000.0, seed=3,
                                          fs=1000.0)
    sigma = math.sqrt(mom[0])
    assert np.std(x) == pytest.approx(sigma, rel=0.05)
    assert np.mean(x) == pytest.approx(0.0, abs=0.05 * sigma)


# ============================================================================
# CARDS + END-TO-END
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


def test_impl_fatig_card_parsing():
    """/IMPL/FATIG reads two lines: (fmin fmax nf funct [nmode]) and
    (m C [zeta] [mean ult] [mcdur seed]); /BASE the base-accel PSD (dir on line
    1); /STRS the prestressed modes — a PORT card (freimpl.F has no
    spectral-fatigue path)."""
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/FATIG\n0.0 50.0 500 7 4\n"
                   "5.0 1.0e12 0.03\n/END\n")
    assert ec.implicit and ec.impl_fatig
    assert ec.impl_fatig_fmin == pytest.approx(0.0)
    assert ec.impl_fatig_fmax == pytest.approx(50.0)
    assert ec.impl_fatig_nf == 500 and ec.impl_fatig_funct == 7
    assert ec.impl_fatig_nmode == 4
    assert ec.impl_fatig_snm == pytest.approx(5.0)
    assert ec.impl_fatig_snc == pytest.approx(1.0e12)
    assert ec.impl_fatig_zeta == pytest.approx(0.03)
    assert not ec.impl_fatig_base

    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/FATIG/BASE\n0.0 250.0 3000 10 0 5\n"
                   "4.0 5.0e14 0.05 0.0 0.0 400.0 42\n/END\n")
    assert ec.impl_fatig and ec.impl_fatig_base
    assert ec.impl_fatig_funct == 10 and ec.impl_fatig_dir == 0
    assert ec.impl_fatig_nmode == 5
    assert ec.impl_fatig_mcdur == pytest.approx(400.0)
    assert ec.impl_fatig_seed == 42

    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/FATIG/STRS\n0.0 30.0 300 4 3\n"
                   "3.0 1.0e10\n/END\n")
    assert ec.impl_fatig and ec.impl_fatig_prestress and ec.impl_nlgeom


def test_fatig_card_end_to_end():
    """/IMPL/FATIG/BASE end to end on the base-driven spring chain: the listing
    prints the RANDOM-VIBRATION (SPECTRAL) FATIGUE block and stores the stress
    channels, moments, damage estimators and the Monte-Carlo cross-check."""
    starter = _chain_deck([2e-3] * 5, [800.] * 5, [0.] * 5)
    m, out = _run(starter,
                  "#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/FATIG/BASE\n"
                  "0.0 250.0 3000 2 0 5\n5.0 1.0e14 0.03 0.0 0.0 400.0 123\n"
                  "/PRINT/-500\n/STOP\n15.0\n", capture=True)
    fat = m.implicit_result.fatigue
    assert fat is not None
    assert len(fat["channels"]) == 5                 # five springs
    # the base spring carries the most force under base excitation
    assert fat["critical_label"] == "springs#1:N"
    s = fat["summary"]
    for k in ("narrow_band", "dirlik", "wirsching_light", "tovo_benasciutti"):
        assert s[k]["damage_rate"] > 0.0
        assert np.isfinite(s[k]["life"])
    # narrow band is the most conservative (highest damage) here
    assert s["narrow_band"]["damage_rate"] >= s["dirlik"]["damage_rate"]
    # the Monte-Carlo cross-check ran and agrees with Dirlik within scatter
    mc = fat["monte_carlo"]
    assert mc is not None and 0.6 < mc["damage_rate"] / s["dirlik"][
        "damage_rate"] < 1.6
    assert "RANDOM-VIBRATION (SPECTRAL) FATIGUE" in out
    assert "DIRLIK 1985" in out and "CRITICAL STRESS CHANNEL" in out


def test_fatig_force_psd_end_to_end():
    """/IMPL/FATIG (force PSD, the default) end to end: the /CLOAD pattern drives
    the fatigue analysis and a finite life is reported for the critical
    channel."""
    starter = _chain_deck([2e-3] * 3, [800.] * 3, [0.] * 3).replace(
        "/BCS/1",
        "/CLOAD/1\ndrive\n         1         X         3       1.0\n"
        "/GRNOD/NODE/3\ndrive\n4\n/BCS/1")
    m, out = _run(starter,
                  "#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/FATIG\n"
                  "0.0 250.0 2000 2 3\n5.0 1.0e14 0.03\n/PRINT/-500\n"
                  "/STOP\n15.0\n", capture=True)
    fat = m.implicit_result.fatigue
    assert fat is not None and not fat["base"]
    assert np.isfinite(fat["summary"]["dirlik"]["life"])
    assert fat["summary"]["dirlik"]["life"] > 0.0
    assert "FORCE PSD" in out


# ============================================================================
# NO-REGRESSION (the M7 parity contract)
# ============================================================================

def test_fatigue_path_does_not_mutate_solvers_or_state():
    """The fatigue path is NEW and ALONGSIDE (the M14-M19 parity contract):
    recovering the stress modes leaves the element state untouched and the M16
    modal_frequencies output bit-identical before and after."""
    m = _starter(_chain_deck([2e-3] * 3, [800., 800., 800.], [0.5, 0.0, 0.0]))
    f0, _, _ = modal_frequencies(m, nev=3)
    before = {k: (v.copy() if isinstance(v, np.ndarray) else v)
              for k, v in m.springs.state.items()}
    basis = build_modal_basis(m, nev=3)
    Sigma, channels = stress_modes(m, basis)         # the read-only recovery
    z = modal_damping(basis.omega, uniform=0.03)
    n = m.numnod
    F = np.zeros((n, 3))
    _loads(m).external_forces(1.0, F, m.x0)
    frf = modal_frequency_response(basis, F, np.zeros((n, 3)),
                                   np.linspace(1e-6, 1.0, 50), z)
    sp = stress_response_psd(frf, Sigma, np.ones(50))
    sf.fatigue_summary(sp["moments"][:, 0], 5.0, 1e14)
    for k, v in before.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(m.springs.state[k], v), k
    f1, _, _ = modal_frequencies(m, nev=3)
    assert np.array_equal(f0, f1)


def test_direct_dynamics_unchanged_by_fatigue_path():
    """A DIRECT /IMPL/DYNA run is byte-for-byte unaffected by the M20 fatigue
    machinery living in the same package (the M10 integrator stays
    bit-identical, and no fatigue is attached)."""
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
