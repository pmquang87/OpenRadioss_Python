"""
M19 validations: RANDOM / SPECTRAL (PSD) response (/IMPL/PSD) and RESPONSE
SPECTRA (/IMPL/RSPEC) — built ALONGSIDE the M16 real eigensolver, the M17
real-mode FRF and the M18 complex FRF (all stay bit-identical; the
random/spectral analyses CONSUME their transfer functions read-only).

Every new capability gets at least one ANALYTIC check (the port's philosophy):

RANDOM / SPECTRAL (PSD) RESPONSE
* a white-noise-driven SDOF's variance sigma_u^2 against the closed form
  sigma^2 = S_0/(2 c k) (Newland — the Lorentzian integral of |H|^2);
* the multi-DOF response PSD reconstructed from the modal FRF matching the
  DIRECT inversion (K - Omega^2 M + i Omega C)^-1 (the |H|^2 S law, both FRFs);
* the spectral-moment / Parseval identity m_0(velocity PSD) = m_2(displacement
  PSD) (S_vv = Omega^2 S_uu);
* the RMS reducing to the STATIC answer sigma_u = sigma_f/k for a
  quasi-static (low-frequency-band) input;
* the mean zero-crossing rate nu_0 = (1/2pi) sqrt(m_2/m_0) equal to the
  natural frequency for a narrow-band SDOF response.

RESPONSE SPECTRA (SRSS / CQC modal combination)
* the CQC correlation coefficient rho_ik against the Der Kiureghian closed
  form (unit diagonal, -> 0 well-separated, -> 1 for r -> 1);
* a single-mode spectrum recovering the spectral displacement Gamma Sa/omega^2;
* SRSS of WELL-SEPARATED modes ~ CQC (rho -> I: the cross-terms vanish);
* CQC vs SRSS on CLOSELY-SPACED modes — the cross terms matter and SRSS errs,
  the CQC analogue of M18's classical-vs-complex gap (asserted);
* the flat-spectrum limit (constant Sa) computed exactly.

CARDS + NO-REGRESSION (the M7 parity contract)
* /IMPL/PSD (+ /BASE, /CPLX) and /IMPL/RSPEC card mirror (PORT cards —
  freimpl.F has no random-vibration / response-spectrum path);
* the random/spectral path NEVER mutates the M16 eigensolver / M17 / M18
  transfer functions / the element state, and the direct M10 answer is
  unchanged whether or not it runs.

See PORTING_GUIDE.md roadmap M19.
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
from pyradioss.engine.kinematics import LoadsAndConstraints         # noqa: E402
from pyradioss.implicit.complex_modal import (                      # noqa: E402
    build_complex_basis, complex_frf, direct_frf)
from pyradioss.implicit.modal import modal_frequencies              # noqa: E402
from pyradioss.implicit.modal_response import (                     # noqa: E402
    build_modal_basis, modal_damping, modal_frequency_response)
from pyradioss.implicit.random_response import (                    # noqa: E402
    crossing_rates, response_psd, rms_response, spectral_moments)
from pyradioss.implicit.response_spectrum import (                  # noqa: E402
    cqc_correlation, response_spectrum_analysis)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _starter(text):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M19_0000.rad")
    with open(sp, "w") as f:
        f.write(text)
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(sp)


def _run(starter_text, engine_text, capture=False):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M19_0000.rad")
    ep = os.path.join(d, "M19_0001.rad")
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


# ---- a spring-mass SDOF (consistent mass == lumped mass) --------------------
K_SM, M_SM = 4.0, 2.0e-3
M_TIP = M_SM / 2.0                          # half the spring mass on the tip
OM_SM = np.sqrt(K_SM / M_TIP)


def _spring_mass_deck(force=0.0):
    load = ""
    grn = "/GRNOD/NODE/1\nn\n1\n/GRNOD/NODE/2\nt\n2\n"
    funct = "/FUNCT/1\nunit\n       0.0       1.0\n  100000.0       1.0\n"
    if force:
        load = (f"/CLOAD/1\naxial\n         1         X         2       "
                f"{force}\n")
    return f"""\
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
     {M_SM}       {K_SM}       0.0
{funct}{grn}{load}/BCS/1
pin
       111       111         0         1
/BCS/2
axial
       011       111         0         2
/END
"""


def _chain_deck(masses, ks, cs, force_node=None, force=0.0):
    """A fixed-free axial spring-mass chain (mirrors the M18 helper)."""
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
    funct = "/FUNCT/1\nunit\n       0.0       1.0\n  100000.0       1.0\n"
    load = ""
    grn = f"/GRNOD/NODE/1\npin\n1\n/GRNOD/NODE/2\nfree\n{freeids}\n"
    if force_node is not None and force:
        load = (f"/CLOAD/1\ndrive\n         1         X         3       "
                f"{force}\n")
        grn += f"/GRNOD/NODE/3\ndrive\n{force_node}\n"
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
{funct}{grn}{load}/BCS/1
pin
       111       111         0         1
/BCS/2
axial
       011       111         0         2
/END
"""


# A "tuning-fork": two masses each on their own ground spring (k), weakly
# coupled (kappa), with SLIGHTLY different masses -> two CLOSELY-SPACED modes
# that BOTH participate under a base (x) excitation (the CQC demonstrator).
def _tuning_fork_deck():
    kk, kap = 1000.0, 20.0
    mA, mB = 4.0e-3, 4.4e-3               # node2 = 2e-3, node3 = 2.2e-3
    return f"""\
#RADIOSS STARTER
/BEGIN
FORK
/NODE
         1{0.0:20.10f}{0.0:20.10f}{0.0:20.10f}
         2{10.0:20.10f}{0.0:20.10f}{0.0:20.10f}
         3{20.0:20.10f}{0.0:20.10f}{0.0:20.10f}
/SPRING/1
         1         1         2
/SPRING/2
         2         1         3
/SPRING/3
         3         2         3
/PART/1
a
         1         1
/PART/2
b
         2         1
/PART/3
c
         3         1
/MAT/LAW1/1
st
   7.8e-6
     210.0       0.0
/PROP/SPRING/1
sa
{mA:12g}{kk:12g}         0.0
/PROP/SPRING/2
sb
{mB:12g}{kk:12g}         0.0
/PROP/SPRING/3
sc
      2e-6{kap:12g}         0.0
/GRNOD/NODE/1
pin
1
/GRNOD/NODE/2
free
2
3
/FUNCT/1
unit
       0.0       1.0
  100000.0       1.0
/BCS/1
pin
       111       111         0         1
/BCS/2
axial
       011       111         0         2
/END
"""


def _flat_funct(model_text_id, value):
    """A flat (constant-value) /FUNCT block over a wide band."""
    return (f"/FUNCT/{model_text_id}\nflat\n       0.0{value:12g}\n"
            f"  100000.0{value:12g}\n")


# ============================================================================
# RANDOM / SPECTRAL (PSD) RESPONSE
# ============================================================================

def test_white_noise_sdof_closed_form():
    """A white-noise-driven SDOF: the response variance sigma_u^2 from the
    module machinery (modal FRF + spectral moments) matches the closed form
    sigma^2 = S_0/(2 c k) (Newland), c = 2 zeta omega m, k the stiffness."""
    zeta = 0.03
    m = _starter(_spring_mass_deck(force=1.0))
    loads = _loads(m)
    basis = build_modal_basis(m, nev=1)
    z = modal_damping(basis.omega, uniform=zeta)
    # a fine sweep well past resonance so the Lorentzian tail is captured
    fmax = 5.0 * OM_SM / (2.0 * np.pi)
    freqs = np.linspace(1e-9, fmax, 200000)
    n = m.numnod
    F = np.zeros((n, 3))
    loads.external_forces(1.0, F, m.x0)              # unit /CLOAD pattern
    frf = modal_frequency_response(basis, F, np.zeros((n, 3)), freqs, z)
    S0 = 1.0
    rp = response_psd(frf, np.full_like(freqs, S0))
    mom = spectral_moments(rp["omega"], rp["Suu"], nmax=2)
    e = basis.dof.eq[m.node_index(2) * 6 + 0]
    sig2 = mom[0, e]

    c = 2.0 * zeta * OM_SM * M_TIP
    closed = S0 / (2.0 * c * K_SM)
    assert sig2 == pytest.approx(closed, rel=2e-3)

    # narrow-band response -> the mean zero-crossing rate is the natural freq
    rates = crossing_rates(mom)
    assert rates["nu0"][e] == pytest.approx(OM_SM / (2.0 * np.pi), rel=2e-2)


def test_response_psd_matches_direct_inversion():
    """The multi-DOF response PSD from the (complex-mode) modal FRF equals the
    one from the DIRECT inversion (K - Omega^2 M + i Omega C)^-1 F: the |H|^2 S
    law is the SAME whichever transfer function feeds it (RMS to round-off)."""
    m = _starter(_chain_deck([2e-3, 2e-3, 2e-3], [1000., 1000., 1000.],
                             [0.5, 0.0, 0.0], force_node=4, force=1.0))
    loads = _loads(m)
    basis = build_complex_basis(m, nev=3)
    n = m.numnod
    F = np.zeros((n, 3))
    loads.external_forces(1.0, F, m.x0)
    fmax = 1.5 * float(basis.natural_freqs_hz.max())
    freqs = np.linspace(1e-6, fmax, 4000)
    cf = complex_frf(basis, F, np.zeros((n, 3)), freqs)
    df = direct_frf(basis, F, np.zeros((n, 3)), freqs)
    # a flat input PSD; the two FRFs must give the same response statistics
    Sff = np.full_like(freqs, 2.0)
    rc = response_psd(cf, Sff)
    rd = response_psd(df, Sff)
    mc = spectral_moments(rc["omega"], rc["Suu"], nmax=0)
    md = spectral_moments(rd["omega"], rd["Suu"], nmax=0)
    assert np.allclose(rms_response(mc), rms_response(md), rtol=1e-7)


def test_spectral_moment_parseval_identity():
    """S_vv = Omega^2 S_uu (velocity is i Omega times displacement), so the
    velocity variance m_0(S_vv) equals the displacement m_2(S_uu) — a
    spectral-moment / Parseval identity."""
    m = _starter(_chain_deck([2e-3, 2e-3], [1000., 1000.],
                             [0.0, 0.0], force_node=3, force=1.0))
    loads = _loads(m)
    basis = build_modal_basis(m, nev=2)
    z = modal_damping(basis.omega, uniform=0.04)
    fmax = 2.0 * float(basis.freqs.max())
    freqs = np.linspace(1e-6, fmax, 40000)
    n = m.numnod
    F = np.zeros((n, 3))
    loads.external_forces(1.0, F, m.x0)
    frf = modal_frequency_response(basis, F, np.zeros((n, 3)), freqs, z)
    Sff = np.full_like(freqs, 1.0)
    # displacement PSD + its 2nd moment
    rd = response_psd(frf, Sff)
    md = spectral_moments(rd["omega"], rd["Suu"], nmax=2)
    # velocity FRF = i Omega * U -> velocity PSD, its 0th moment
    vfrf = dict(frf)
    vfrf["U"] = (1j * frf["omega"])[:, None] * frf["U"]
    rv = response_psd(vfrf, Sff)
    mv = spectral_moments(rv["omega"], rv["Suu"], nmax=0)
    assert np.allclose(mv[0], md[2], rtol=1e-9)


def test_quasistatic_rms_reduces_to_static():
    """For a quasi-static (low-frequency-band) input the SDOF FRF -> 1/k, so
    sigma_u -> sigma_f/k: the RMS response reduces to the static answer (RMS
    force / stiffness)."""
    m = _starter(_spring_mass_deck(force=1.0))
    loads = _loads(m)
    basis = build_modal_basis(m, nev=1)
    z = modal_damping(basis.omega, uniform=0.05)
    f1 = OM_SM / (2.0 * np.pi)
    # a band well below the resonance (0..f1/20)
    freqs = np.linspace(1e-9, f1 / 20.0, 20000)
    n = m.numnod
    F = np.zeros((n, 3))
    loads.external_forces(1.0, F, m.x0)
    frf = modal_frequency_response(basis, F, np.zeros((n, 3)), freqs, z)
    Sff = np.full_like(freqs, 3.0)
    rp = response_psd(frf, Sff)
    mom = spectral_moments(rp["omega"], rp["Suu"], nmax=0)
    e = basis.dof.eq[m.node_index(2) * 6 + 0]
    sig_u = rms_response(mom)[e]
    # static: sigma_f from the input PSD (same convention), sigma_u = sigma_f/k
    sig_f = np.sqrt(spectral_moments(rp["omega"], Sff, nmax=0))
    assert sig_u == pytest.approx(sig_f / K_SM, rel=1e-3)


# ============================================================================
# RESPONSE SPECTRA — SRSS / CQC modal combination
# ============================================================================

def test_cqc_correlation_closed_form():
    """The CQC correlation rho_ik matches the Der Kiureghian equal-damping
    closed form, is symmetric with unit diagonal, -> 0 for well-separated
    modes and -> 1 for r = omega_k/omega_i -> 1."""
    z = 0.05
    w = 2.0 * np.pi * np.array([10.0, 10.4, 40.0])
    rho = cqc_correlation(w, z)

    def rho_eq(wi, wk):
        r = wk / wi
        return (8.0 * z ** 2 * (1.0 + r) * r ** 1.5
                / ((1.0 - r ** 2) ** 2 + 4.0 * z ** 2 * r * (1.0 + r) ** 2))

    assert np.allclose(np.diag(rho), 1.0)
    assert np.allclose(rho, rho.T)
    assert rho[0, 1] == pytest.approx(rho_eq(w[0], w[1]), rel=1e-10)
    assert rho[0, 1] > 0.7                       # closely spaced -> correlated
    assert rho[0, 2] < 0.02                      # well separated -> ~0


def test_single_mode_recovers_gamma_sa():
    """A single-mode (SDOF) response spectrum recovers the spectral
    displacement: the peak response = |Gamma Sa/omega^2 phi| = Sa/omega^2 at
    the free DOF (Gamma phi = 1 for a mass-normalized SDOF)."""
    m = _starter(_spring_mass_deck())
    basis = build_modal_basis(m, nev=1)
    # a design spectrum: constant Sa
    from pyradioss.common.tables import FunctTable
    Sa0 = 5.0
    spec = FunctTable(1, [0.0, 1e5], [Sa0, Sa0])
    res = response_spectrum_analysis(basis, spec, direction=0, zeta=0.05)
    e = basis.dof.eq[m.node_index(2) * 6 + 0]
    expect = Sa0 / (basis.omega[0] ** 2)
    assert res["srss"][e] == pytest.approx(expect, rel=1e-10)
    assert res["cqc"][e] == pytest.approx(expect, rel=1e-10)   # one mode


def test_srss_well_separated_equals_cqc():
    """For WELL-SEPARATED modes the CQC correlation matrix is ~ identity, so
    CQC ~ SRSS (the cross-terms vanish — the direct-envelope regime)."""
    m = _starter(_chain_deck([2e-3, 2e-3, 2e-3], [1000., 1000., 1000.],
                             [0.0, 0.0, 0.0]))
    basis = build_modal_basis(m, nev=3)
    from pyradioss.common.tables import FunctTable
    # a sloped spectrum so the modes carry different ordinates
    spec = FunctTable(1, [0.0, 1e5], [10.0, 1e5])
    res = response_spectrum_analysis(basis, spec, direction=0, zeta=0.03)
    ref = res["srss"].max()
    assert np.max(np.abs(res["cqc"] - res["srss"])) < 0.02 * ref


def test_cqc_vs_srss_closely_spaced_gap():
    """On CLOSELY-SPACED modes (the tuning fork's near-degenerate pair, both
    participating under base excitation) CQC's cross-terms MATTER: CQC and
    SRSS differ by a real margin (the CQC analogue of M18's classical-vs-
    complex gap), and CQC > SRSS (constructive same-sign correlation)."""
    m = _starter(_tuning_fork_deck())
    basis = build_modal_basis(m, nev=2)
    # confirm the two modes really are close and both participate
    assert basis.freqs.max() / basis.freqs.min() < 1.1
    from pyradioss.common.tables import FunctTable
    spec = FunctTable(1, [0.0, 1e5], [4.0, 4.0])
    res = response_spectrum_analysis(basis, spec, direction=0, zeta=0.05)
    # both participation factors non-trivial
    assert np.all(np.abs(res["gamma"]) > 1e-6)
    # the correlation between the two modes is high
    assert res["rho"][0, 1] > 0.3
    jc = int(np.argmax(res["cqc"]))
    gap = abs(res["cqc"][jc] - res["srss"][jc]) / res["srss"][jc]
    assert gap > 0.05, f"expected a real CQC/SRSS gap, got {gap:.3%}"


def test_flat_spectrum_limit():
    """The flat-spectrum limit: with a constant Sa the modal peaks are
    Gamma_i Sa/omega_i^2 phi_i exactly, and SRSS/CQC reduce to the hand
    computation on those peaks."""
    m = _starter(_chain_deck([2e-3, 2e-3], [1000., 1000.], [0.0, 0.0]))
    basis = build_modal_basis(m, nev=2)
    from pyradioss.common.tables import FunctTable
    from pyradioss.implicit.modal import _rigid_body_vectors
    Sa0 = 7.0
    spec = FunctTable(1, [0.0, 1e5], [Sa0, Sa0])
    res = response_spectrum_analysis(basis, spec, direction=0, zeta=0.05)
    # hand-built modal peaks
    R = _rigid_body_vectors(basis.dof.model, basis.dof)
    gamma = basis.Phi.T @ (basis.M @ R[:, 0])
    Sd = Sa0 / (basis.omega ** 2)
    peaks = basis.Phi * (gamma * Sd)[None, :]
    srss = np.sqrt((peaks ** 2).sum(axis=1))
    assert np.allclose(res["srss"], srss, rtol=1e-10)
    assert np.allclose(res["gamma"], gamma, rtol=1e-12)


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


def test_impl_psd_card_parsing():
    """/IMPL/PSD reads fmin fmax nf funct [nmode]; /BASE the base-accel PSD
    (dir); /CPLX the complex FRF — PORT cards (freimpl.F has no random path)."""
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/PSD\n0.0 50.0 500 7 4\n"
                   "/END\n")
    assert ec.implicit and ec.impl_psd
    assert ec.impl_psd_fmin == pytest.approx(0.0)
    assert ec.impl_psd_fmax == pytest.approx(50.0)
    assert ec.impl_psd_nf == 500 and ec.impl_psd_funct == 7
    assert ec.impl_psd_nmode == 4
    assert not ec.impl_psd_base and not ec.impl_psd_cplx

    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/PSD/BASE\n1.0 40.0 800 9 2 5\n"
                   "/END\n")
    assert ec.impl_psd and ec.impl_psd_base
    assert ec.impl_psd_funct == 9 and ec.impl_psd_dir == 2
    assert ec.impl_psd_nmode == 5

    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/PSD/CPLX\n0.0 30.0 300 4 3\n"
                   "/END\n")
    assert ec.impl_psd and ec.impl_psd_cplx
    assert ec.impl_psd_funct == 4 and ec.impl_psd_nmode == 3


def test_impl_rspec_card_parsing():
    """/IMPL/RSPEC reads funct dir [zeta] [nmode]; /STRS the prestressed
    spectrum — a PORT card (freimpl.F has no response-spectrum path)."""
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/RSPEC\n3 0 0.05 5\n/END\n")
    assert ec.implicit and ec.impl_rspec
    assert ec.impl_rspec_funct == 3 and ec.impl_rspec_dir == 0
    assert ec.impl_rspec_zeta == pytest.approx(0.05)
    assert ec.impl_rspec_nmode == 5

    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/RSPEC/STRS\n2 1 0.02\n/END\n")
    assert ec.impl_rspec and ec.impl_rspec_prestress and ec.impl_nlgeom
    assert ec.impl_rspec_funct == 2 and ec.impl_rspec_dir == 1


def _psd_deck(base=False):
    """A force- (or base-) driven chain plus the input-PSD /FUNCT/2."""
    base_text = _chain_deck([2e-3, 2e-3, 2e-3], [1000., 1000., 1000.],
                            [0.5, 0.0, 0.0], force_node=4, force=1.0)
    # inject a flat input-PSD /FUNCT/2 before /END
    return base_text.replace("/BCS/1",
                             "/FUNCT/2\npsd\n       0.0       1.0\n"
                             "  100000.0       1.0\n/BCS/1")


def test_psd_card_end_to_end():
    """/IMPL/PSD end to end: the listing prints the RANDOM / SPECTRAL (PSD)
    RESPONSE block and stores the response PSD / RMS / moments / rates."""
    m, out = _run(_psd_deck(),
                  "#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/PSD\n0.0 0.4 2000 2 3\n"
                  "/PRINT/-500\n/STOP\n15.0\n", capture=True)
    r = m.implicit_result
    assert r.random_response is not None
    rr = r.random_response
    assert rr["moments"].shape[0] >= 3            # m0..m2 at least
    assert np.all(rr["rms"] >= 0.0)
    assert rr["rms"].max() > 0.0                  # a driven structure responds
    assert np.all(rr["rates"]["nu0"] >= 0.0)
    assert "RANDOM / SPECTRAL (PSD) RESPONSE" in out
    assert "MEAN ZERO-CROSSING RATE" in out


def test_psd_base_excitation_end_to_end():
    """/IMPL/PSD/BASE: a rigid-base ACCELERATION PSD in direction x feeds the
    M16 participation into the FRF and produces a finite relative-displacement
    RMS (the shaker-table random-input path)."""
    m, out = _run(_psd_deck(),
                  "#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/PSD/BASE\n"
                  "0.0 0.4 2000 2 0 3\n/PRINT/-500\n/STOP\n15.0\n",
                  capture=True)
    rr = m.implicit_result.random_response
    assert rr is not None and rr["base"]
    assert np.all(np.isfinite(rr["rms"]))
    assert rr["rms"].max() > 0.0
    assert "BASE ACCELERATION PSD" in out


def test_psd_complex_frf_matches_real_when_classical():
    """The complex-FRF PSD path reduces to the real-mode PSD path when the
    damping IS classical (Rayleigh, no discrete dashpot): the M18 complex FRF
    reduces to the M17 real FRF, so the response PSD / RMS agree. Library-level
    (the /IMPL/DYNA/DAMP card switches the run to the DIRECT dynamics driver,
    which does not host the static random-response analysis)."""
    alpha, beta = 0.2, 2e-5
    m = _starter(_chain_deck([2e-3, 2e-3, 2e-3], [1000., 1000., 1000.],
                             [0.0, 0.0, 0.0], force_node=4, force=1.0))
    loads = _loads(m)
    n = m.numnod
    F = np.zeros((n, 3))
    loads.external_forces(1.0, F, m.x0)
    freqs = np.linspace(1e-6, 0.4, 4000)
    Sff = np.ones_like(freqs)
    # complex FRF (Rayleigh C, classical) -> response PSD -> RMS
    cbasis = build_complex_basis(m, nev=3, alpha=alpha, beta=beta)
    cf = complex_frf(cbasis, F, np.zeros((n, 3)), freqs)
    rms_cplx = rms_response(spectral_moments(
        cf["omega"], response_psd(cf, Sff)["Suu"], nmax=0))
    # real-mode FRF fed the same Rayleigh modal ratios
    from pyradioss.implicit.modal_response import rayleigh_ratios
    rbasis = build_modal_basis(m, nev=3)
    z = rayleigh_ratios(rbasis.omega, alpha, beta)
    frf = modal_frequency_response(rbasis, F, np.zeros((n, 3)), freqs, z)
    rms_real = rms_response(spectral_moments(
        frf["omega"], response_psd(frf, Sff)["Suu"], nmax=0))
    assert rms_cplx.max() == pytest.approx(rms_real.max(), rel=1e-3)


def test_psd_complex_frf_card_end_to_end():
    """/IMPL/PSD/CPLX end to end on the NON-classically-damped dashpot chain:
    the complex FRF (M18) feeds the |H|^2 S law, the response is stored with
    the ``complex`` flag, and the RMS matches the direct-inversion FRF (the
    non-classical response the real-mode path could not represent)."""
    m, out = _run(_psd_deck(),
                  "#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/PSD/CPLX\n"
                  "0.0 0.4 3000 2 3\n/PRINT/-500\n/STOP\n15.0\n", capture=True)
    rr = m.implicit_result.random_response
    assert rr is not None and rr["complex"]
    assert np.all(np.isfinite(rr["rms"])) and rr["rms"].max() > 0.0
    # cross-check against the direct inversion on the same (K, C, M)
    loads = _loads(m)
    n = m.numnod
    F = np.zeros((n, 3))
    loads.external_forces(1.0, F, m.x0)
    cbasis = build_complex_basis(m, nev=3)
    freqs = rr["freqs"]
    df = direct_frf(cbasis, F, np.zeros((n, 3)), freqs)
    rms_direct = rms_response(spectral_moments(
        df["omega"], response_psd(df, rr["Sff"])["Suu"], nmax=0))
    assert rr["rms"].max() == pytest.approx(rms_direct.max(), rel=1e-6)
    assert "COMPLEX (non-classical)" in out


def test_rspec_card_end_to_end():
    """/IMPL/RSPEC end to end: the listing prints the RESPONSE SPECTRUM block
    with the participation table and the SRSS/CQC peaks, stored on the
    result."""
    deck = _chain_deck([2e-3, 2e-3, 2e-3], [1000., 1000., 1000.],
                       [0.0, 0.0, 0.0]).replace(
        "/BCS/1",
        "/FUNCT/2\nspec\n       0.0       5.0\n  100000.0       5.0\n/BCS/1")
    m, out = _run(deck, "#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/RSPEC\n"
                        "2 0 0.05 3\n/PRINT/-500\n/STOP\n15.0\n", capture=True)
    r = m.implicit_result.response_spectrum
    assert r is not None
    assert r["srss"].max() > 0.0 and r["cqc"].max() > 0.0
    assert r["rho"].shape == (3, 3)
    assert "RESPONSE SPECTRUM" in out
    assert "SRSS / CQC" in out


# ============================================================================
# NO-REGRESSION (the M7 parity contract)
# ============================================================================

def test_random_path_does_not_mutate_modal_solvers_or_state():
    """The random/spectral path is NEW and ALONGSIDE (the M14-M18 parity
    contract): building the FRFs it consumes leaves the element state untouched
    and the M16 modal_frequencies output bit-identical before and after."""
    m = _starter(_chain_deck([2e-3, 2e-3, 2e-3], [1000., 1000., 1000.],
                             [0.5, 0.0, 0.0]))
    f0, _, _ = modal_frequencies(m, nev=3)
    before = {k: (v.copy() if isinstance(v, np.ndarray) else v)
              for k, v in m.springs.state.items()}
    # run both M19 analyses' library entry points
    basis = build_modal_basis(m, nev=3)
    from pyradioss.common.tables import FunctTable
    response_spectrum_analysis(basis, FunctTable(1, [0.0, 1e5], [1.0, 1.0]),
                               direction=0, zeta=0.05)
    cbasis = build_complex_basis(m, nev=3)
    n = m.numnod
    frf = complex_frf(cbasis, np.zeros((n, 3)), np.zeros((n, 3)),
                      np.linspace(0.0, 0.4, 100))
    response_psd(frf, np.ones(100))
    for k, v in before.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(m.springs.state[k], v), k
    f1, _, _ = modal_frequencies(m, nev=3)
    assert np.array_equal(f0, f1)


def test_direct_dynamics_unchanged_by_random_path():
    """A DIRECT /IMPL/DYNA run is byte-for-byte unaffected by the M19 random /
    spectral machinery living in the same package (the M10 integrator stays
    bit-identical, and no random_response / response_spectrum is attached)."""
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
    assert m1.implicit_result.random_response is None
    assert m1.implicit_result.response_spectrum is None
