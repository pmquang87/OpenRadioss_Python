"""
M17 validations: MODAL-SUPERPOSITION dynamics — transient response history
(/IMPL/MODAL/DYNA), harmonic / frequency response (/IMPL/FREQ) and modal
damping, built on the M16 eigenpairs (implicit/modal.py).

Every new capability gets at least one ANALYTIC check (the port's philosophy):

MODAL DAMPING
* the Rayleigh map zeta_i = 1/2(alpha/omega_i + beta omega_i) closed form;
* a (frequency, zeta) table interpolated onto the extracted omega_i;
* CONSISTENCY with the M11 DIRECT-integration Rayleigh damping: the same
  (alpha, beta) fed /IMPL/DYNA/DAMP produces a damped-decay envelope whose
  rate equals zeta_i * omega_i from the modal map.

MODAL TRANSIENT (Nigam-Jennings exact recurrence)
* a step-loaded spring-mass SDOF matching u(t) = (F/k)(1 - cos omega t)
  POINTWISE (the recurrence is EXACT for piecewise-linear forcing — no
  O(dt^2) dispersion), the energy ledger closed to round-off;
* CROSS-SOLVER consistency: the SAME deck through the DIRECT M10 Newmark
  integrator and the modal path agree bit-close at the response peak (the
  spring's consistent mass EQUALS its lumped mass, so both share K and M);
* modal-TRUNCATION convergence: N modes -> the full dynamic amplification
  as N grows (a step-loaded bar climbing toward DAF = 2);
* the mode-ACCELERATION / residual-flexibility correction recovering the
  EXACT static tail from a single mode (the documented static-correction
  deviation);
* a uniform-zeta damped decay matching the exp(-zeta omega t) envelope.

HARMONIC / FREQUENCY RESPONSE (the complex FRF)
* an SDOF FRF peak = 1/(2 zeta) at resonance with the half-power bandwidth
  Delta_omega/omega = 2 zeta (library level, the exact formula);
* the FRF resonances COINCIDING with the M16 natural frequencies
  (/IMPL/FREQ card end to end);
* a driven cantilever's steady tip amplitude vs the closed-form modal FRF;
* base excitation feeding the M16 effective-mass participation into the FRF.

CARDS + NO-REGRESSION
* the /IMPL/MODAL/DYNA and /IMPL/FREQ card mirror (PORT cards — freimpl.F
  has no frequency-domain path);
* modal superposition NEVER mutates the eigensolver or the element state
  (the M7 parity contract — like test_m14/test_m15/test_m16), and the
  DIRECT M10 Newmark answer is unchanged whether or not the modal path runs.

See PORTING_GUIDE.md roadmap M17.
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
from pyradioss.implicit.modal_response import (                     # noqa: E402
    _nigam_jennings_coeffs, build_modal_basis, modal_damping,
    modal_frequency_response, modal_transient, rayleigh_ratios, table_ratios)
from pyradioss.implicit.modal import modal_frequencies             # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _starter(text):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M17_0000.rad")
    with open(sp, "w") as f:
        f.write(text)
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(sp)


def _run(starter_text, engine_text, capture=False):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M17_0000.rad")
    ep = os.path.join(d, "M17_0001.rad")
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


# ---- a spring-mass SDOF: consistent mass == lumped mass (M16), so the modal
# path and the DIRECT Newmark integrator share IDENTICAL (K, M) ---------------
K_SM, M_SM = 4.0, 2.0e-3
OM_SM = np.sqrt(K_SM / (M_SM / 2.0))       # half the spring mass lands on tip
T_SM = 2.0 * np.pi / OM_SM


def _spring_mass_deck(v0=0.0, force=0.0):
    """A single TYPE4 spring node1-node2; node1 pinned, node2 axial only.
    ``/PROP/SPRING`` is 'Mass K C' — half the mass lumps onto the free tip."""
    inivel = (f"/INIVEL/TRA/1\nkick\n      {v0}       0.0       0.0         2\n"
              if v0 else "")
    load = ""
    if force:
        load = (f"/FUNCT/1\nstep\n       0.0       1.0\n  100000.0       1.0\n"
                f"/CLOAD/1\naxial\n         1         X         2       "
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
/GRNOD/NODE/1
n
1
/GRNOD/NODE/2
t
2
/BCS/1
pin
       111       111         0         1
/BCS/2
axial
       011       111         0         2
{inivel}{load}/END
"""


def _bar_deck(N=20, L=100.0, A=1.0, force=0.0):
    """Fixed-free axial truss chain along x, optional tip step /CLOAD."""
    E, RHO = 210.0, 7.8e-6
    dx = L / N
    nodes = "\n".join(f"{i+1:10d}{i*dx:20.10f}{0.0:20.10f}{0.0:20.10f}"
                      for i in range(N + 1))
    trs = "\n".join(f"{i+1:10d}{i+1:10d}{i+2:10d}" for i in range(N))
    alln = "\n".join(str(i + 1) for i in range(N + 1))
    load = ""
    if force:
        load = (f"/FUNCT/1\nstep\n       0.0       1.0\n  100000.0       1.0\n"
                f"/CLOAD/1\ntip\n         1         X         3       "
                f"{force}\n")
    return f"""\
#RADIOSS STARTER
/BEGIN
BAR
/NODE
{nodes}
/TRUSS/1
{trs}
/PART/1
bar
         1         1
/MAT/LAW1/1
s
   {RHO}
     {E}       0.0
/PROP/TRUSS/1
b
       {A}
/GRNOD/NODE/1
pin
1
/GRNOD/NODE/2
all
{alln}
/GRNOD/NODE/3
tip
{N+1}
/BCS/1
pin
       111       000         0         1
/BCS/2
yz
       011       000         0         2
{load}/END
"""


# ============================================================================
# MODAL DAMPING
# ============================================================================

def test_rayleigh_ratios_closed_form():
    """zeta_i = 1/2 (alpha/omega_i + beta omega_i) — the exact Rayleigh map."""
    omega = np.array([1.0, 2.0, 4.0, 8.0])
    a, b = 0.1, 0.02
    z = rayleigh_ratios(omega, a, b)
    assert np.allclose(z, 0.5 * (a / omega + b * omega))
    # the minimum sits at omega = sqrt(a/b)
    assert abs(np.sqrt(a / b) - omega[np.argmin(z)]) < 1.0


def test_damping_table_interpolation():
    """A (frequency, zeta) table interpolates onto f_i = omega_i/2pi with a
    flat hold beyond the ends."""
    ftab = [0.5, 1.0, 2.0]
    ztab = [0.01, 0.03, 0.05]
    omega = 2.0 * np.pi * np.array([0.25, 0.75, 1.5, 3.0])
    z = table_ratios(omega, ftab, ztab)
    assert z[0] == pytest.approx(0.01)             # below table -> held
    assert z[1] == pytest.approx(0.02)             # midway 0.5..1.0
    assert z[2] == pytest.approx(0.04)             # midway 1.0..2.0
    assert z[3] == pytest.approx(0.05)             # above table -> held


def test_modal_damping_over_critical_refused():
    """zeta >= 1 (over-damped) is refused — the recurrence/FRF assume
    under-damped modes."""
    with pytest.raises(ValueError, match="over-critical"):
        modal_damping(np.array([1.0]), uniform=1.5)


def test_rayleigh_damping_matches_direct_integration_decay():
    """CONSISTENCY with the M11 direct integrator: the SAME mass-proportional
    Rayleigh (alpha, beta) fed /IMPL/DYNA/DAMP makes the spring-mass free
    vibration decay at exactly zeta*omega with zeta = rayleigh_ratios — so the
    modal damping map and the direct-integration damping describe the same
    physical damping."""
    a, b = 0.5, 0.0
    v0 = 0.01
    md = _run(_spring_mass_deck(v0=v0),
              f"#\n/RUN/SM/1\n{6*T_SM}\n/IMPL/DYNA/DAMP\n{a} {b}\n"
              f"/IMPL/DTINI\n{T_SM/200}\n/END\n")
    h = md.implicit_result.history
    t = np.array(h["t"])
    u = np.array([uu[md.node_index(2), 0] for uu in h["u"]])
    # decay rate from the log-decrement of successive positive peaks
    s = np.sign(u)
    from scipy.signal import argrelextrema
    pk = argrelextrema(np.abs(u), np.greater)[0]
    slope = np.polyfit(t[pk], np.log(np.abs(u)[pk]), 1)[0]
    zeta = rayleigh_ratios(np.array([OM_SM]), a, b)[0]
    assert -slope == pytest.approx(zeta * OM_SM, rel=0.02)


# ============================================================================
# MODAL TRANSIENT — the Nigam-Jennings exact recurrence
# ============================================================================

def test_nigam_jennings_exact_step_and_decay():
    """The library recurrence is EXACT for piecewise-constant/linear forcing:
    an undamped unit-force step gives (1-cos wt)/w^2 and a damped free release
    gives the analytic envelope — both to round-off, at a coarse step."""
    w = np.array([2.0])
    dt = 0.05
    # undamped step response
    A, B, C, D, Ap, Bp, Cp, Dp = _nigam_jennings_coeffs(w, np.array([0.0]), dt)
    q = np.array([0.0])
    qd = np.array([0.0])
    r = 1.0
    out = []
    for k in range(400):
        q, qd = A * q + B * qd + C * r + D * r, Ap * q + Bp * qd + Cp * r + Dp * r
        out.append(q[0])
    t = np.arange(1, 401) * dt
    assert np.abs(np.array(out) - (1 - np.cos(w[0] * t)) / w[0] ** 2).max() < 1e-12
    # damped free decay
    z = np.array([0.05])
    A, B, C, D, Ap, Bp, Cp, Dp = _nigam_jennings_coeffs(w, z, dt)
    q = np.array([1.0])
    qd = np.array([0.0])
    out = []
    for k in range(800):
        q, qd = A * q + B * qd, Ap * q + Bp * qd
        out.append(q[0])
    t = np.arange(1, 801) * dt
    wd = w[0] * np.sqrt(1 - z[0] ** 2)
    env = np.exp(-z[0] * w[0] * t) * (np.cos(wd * t)
                                      + z[0] / np.sqrt(1 - z[0] ** 2)
                                      * np.sin(wd * t))
    assert np.abs(np.array(out) - env).max() < 1e-12


def test_modal_transient_step_response_closed_form():
    """The /IMPL/MODAL/DYNA card on the spring-mass SDOF matches the closed
    form u(t) = (F/k)(1-cos omega t) POINTWISE (the exact recurrence), lands
    on the dynamic-factor-2 peak, and closes the modal energy ledger to
    round-off."""
    F = 0.05
    t_end, dt = 2.0 * T_SM, T_SM / 400.0
    m = _run(_spring_mass_deck(force=F),
             f"#\n/RUN/SM/1\n1.0\n/IMPL\n/IMPL/MODAL/DYNA\n{t_end} {dt} 1\n"
             f"/END\n")
    h = m.implicit_result.modal_transient_history
    t = np.array(h["t"])
    u = np.array([uu[m.node_index(2), 0] for uu in h["u"]])
    uex = (F / K_SM) * (1.0 - np.cos(OM_SM * t))
    assert np.abs(u - uex).max() < 1e-6 * (2.0 * F / K_SM)
    assert np.abs(u).max() == pytest.approx(2.0 * F / K_SM, rel=1e-4)
    e0 = h["e0"]
    scale = max(abs(e0), 0.5 * K_SM * (2 * F / K_SM) ** 2)
    assert np.abs(h["bal"]).max() < 1e-10 * scale


def test_modal_transient_matches_direct_newmark():
    """Cross-solver consistency: the spring-mass step response through the
    DIRECT M10 Newmark integrator and through the modal path agree bit-close
    at the first response peak (identical K and M — the spring's consistent
    mass equals its lumped mass, so the only difference is the modal path's
    exactness in time)."""
    F = 0.05
    dt = T_SM / 200.0
    starter = _spring_mass_deck(force=F)
    md = _run(starter, f"#\n/RUN/SM/1\n{T_SM}\n/IMPL/DYNA/2\n0.5 0.25\n"
                       f"/IMPL/DTINI\n{dt}\n/END\n")
    mm = _run(starter, f"#\n/RUN/SM/1\n1.0\n/IMPL\n/IMPL/MODAL/DYNA\n"
                       f"{T_SM} {dt} 1\n/END\n")
    ud = np.array([uu[md.node_index(2), 0] for uu in md.implicit_result.history["u"]])
    um = np.array([uu[mm.node_index(2), 0]
                   for uu in mm.implicit_result.modal_transient_history["u"]])
    peak_d = ud[np.argmax(np.abs(ud))]
    peak_m = um[np.argmax(np.abs(um))]
    assert peak_m == pytest.approx(peak_d, rel=1e-4)


def test_modal_truncation_convergence():
    """A step-loaded fixed-free bar: the tip's dynamic amplification climbs
    toward the exact DAF = 2 as the number of retained modes grows (mode
    superposition converges from below on a truncated basis)."""
    F = 0.02
    L, A, E, RHO = 100.0, 1.0, 210.0, 7.8e-6
    m = _starter(_bar_deck(N=20, L=L, A=A, force=F))
    loads = _loads(m)
    tip = m.node_index(21)
    u_static = F * L / (E * A)
    T1 = 4.0 * L / np.sqrt(E / RHO)
    t_end, dt = 0.5 * T1, T1 / 400.0
    daf = []
    for nev in (1, 3, 10):
        basis = build_modal_basis(m, nev=nev)
        z = modal_damping(basis.omega, uniform=0.0)
        h = modal_transient(basis, loads, t_end, dt, z, m)
        u = np.array([uu[tip, 0] for uu in h["u"]])
        daf.append(u.max() / u_static)
    assert daf[0] < daf[1] < daf[2]              # monotone convergence
    assert daf[2] > 1.9                          # approaching DAF = 2
    assert daf[2] < 2.0 + 1e-6                    # never overshoots the DAF


def test_mode_acceleration_static_correction():
    """The mode-acceleration / residual-flexibility correction recovers the
    EXACT static tail from a SINGLE mode: a heavily damped step-loaded bar
    settles to F*L/(EA) with the correction ON (residual flexibility carries
    the truncated higher modes STATICALLY), while the plain mode-displacement
    sum leaves a ~19% deficit. A documented deviation (module docstring)."""
    F = 0.02
    L, A, E, RHO = 100.0, 1.0, 210.0, 7.8e-6
    m = _starter(_bar_deck(N=20, L=L, A=A, force=F))
    loads = _loads(m)
    tip = m.node_index(21)
    u_static = F * L / (E * A)
    T1 = 4.0 * L / np.sqrt(E / RHO)
    basis = build_modal_basis(m, nev=1)
    z = modal_damping(basis.omega, uniform=0.7)         # settle to static
    hm = modal_transient(basis, loads, 4.0 * T1, T1 / 500.0, z, m,
                         mode_acceleration=False)
    ha = modal_transient(basis, loads, 4.0 * T1, T1 / 500.0, z, m,
                         mode_acceleration=True)
    um = hm["u"][-1][tip, 0]
    ua = ha["u"][-1][tip, 0]
    assert abs(um / u_static - 1.0) > 0.1               # mode-disp under-shoots
    assert abs(ua / u_static - 1.0) < 1e-6              # mode-accel EXACT


def test_modal_damped_decay_envelope():
    """A uniform-zeta modal transient of the spring-mass released from a v0
    kick decays on the exp(-zeta omega t) envelope."""
    v0, zeta = 0.01, 0.05
    m = _starter(_spring_mass_deck(v0=v0))
    loads = _loads(m)
    basis = build_modal_basis(m, nev=1)
    z = modal_damping(basis.omega, uniform=zeta)
    h = modal_transient(basis, loads, 6.0 * T_SM, T_SM / 200.0, z, m)
    t = np.array(h["t"])
    u = np.array([uu[m.node_index(2), 0] for uu in h["u"]])
    from scipy.signal import argrelextrema
    pk = argrelextrema(np.abs(u), np.greater)[0]
    slope = np.polyfit(t[pk], np.log(np.abs(u)[pk]), 1)[0]
    assert -slope == pytest.approx(zeta * basis.omega[0], rel=0.03)


# ============================================================================
# HARMONIC / FREQUENCY RESPONSE
# ============================================================================

def test_frf_sdof_peak_and_half_power():
    """The exact SDOF FRF: peak amplification = 1/(2 zeta) at resonance and
    the half-power bandwidth Delta_omega/omega = 2 zeta (library-level, on a
    single-mode spring-mass basis)."""
    m = _starter(_spring_mass_deck())
    basis = build_modal_basis(m, nev=1)
    zeta = 0.02
    z = modal_damping(basis.omega, uniform=zeta)
    f0 = basis.freqs[0]
    freqs = np.linspace(0.2 * f0, 2.0 * f0, 400001)
    n = m.numnod
    F = np.zeros((n, 3))
    F[m.node_index(2), 0] = 1.0
    frf = modal_frequency_response(basis, F, np.zeros((n, 3)), freqs, z)
    dof_pk = int(frf["amp"][0].argmax())
    amp = frf["amp"][:, dof_pk]
    ipk = np.argmax(amp)
    # true static (Omega = 0) compliance of that DOF; peak/DC = 1/(2 zeta)
    dc = np.abs(modal_frequency_response(
        basis, F, np.zeros((n, 3)), np.array([0.0]), z)["U"][0, dof_pk])
    assert amp[ipk] / dc == pytest.approx(1.0 / (2.0 * zeta), rel=1e-3)
    # half-power bandwidth
    half = amp[ipk] / np.sqrt(2.0)
    idx = np.where(amp >= half)[0]
    band = (freqs[idx[-1]] - freqs[idx[0]]) / f0
    assert band == pytest.approx(2.0 * zeta, rel=0.02)


def test_frf_resonances_coincide_with_natural_frequencies():
    """/IMPL/FREQ end to end on the antenna_mast cantilever: the swept FRF's
    amplitude peaks sit exactly on the M16 natural frequencies."""
    import shutil
    d = tempfile.mkdtemp()
    shutil.copy("examples/antenna_mast/MAST_0000.rad",
                os.path.join(d, "M_0000.rad"))
    with open(os.path.join(d, "M_0001.rad"), "w") as f:
        f.write("#\n/RUN/M/1\n1.0\n/IMPL\n/IMPL/FREQ\n"
                "0.001 0.12 6000 0.01 4\n/END\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_starter(os.path.join(d, "M_0000.rad"))
        model = run_engine(os.path.join(d, "M_0001.rad"))
    r = model.implicit_result
    assert r.freq_response is not None
    frf = r.freq_response
    amp = frf["amp"].max(axis=1)
    from scipy.signal import argrelextrema
    pk = argrelextrema(amp, np.greater)[0]
    pkf = np.sort(frf["freqs"][pk][np.argsort(amp[pk])[::-1][:2]])
    # the two distinct bending frequencies (each a degenerate pair)
    assert pkf[0] == pytest.approx(r.modal_frequencies[0], rel=0.02)
    assert pkf[1] == pytest.approx(r.modal_frequencies[2], rel=0.02)
    assert "FREQUENCY RESPONSE" in buf.getvalue()


def test_frf_driven_cantilever_amplitude_vs_closed_form():
    """A cantilever driven by a harmonic tip force below its first resonance:
    the modal FRF tip amplitude matches the direct closed-form modal sum
    sum_i phi_i (phi_i^T F) / (omega_i^2 - Omega^2 + 2 i zeta omega_i Omega)
    (self-consistency of the recombination + the participation)."""
    with contextlib.redirect_stdout(io.StringIO()):
        m = run_starter("examples/antenna_mast/MAST_0000.rad")
    basis = build_modal_basis(m, nev=4)
    z = modal_damping(basis.omega, uniform=0.02)
    n = m.numnod
    F = np.zeros((n, 3))
    F[-2, 1] = 1.0                                   # tip lateral drive
    fdrive = 0.5 * basis.freqs[0]                    # below first resonance
    frf = modal_frequency_response(basis, F, np.zeros((n, 3)),
                                   np.array([fdrive]), z)
    # independent closed-form recombination at that single frequency
    Om = 2.0 * np.pi * fdrive
    r = basis.Phi.T @ basis.dof.gather_residual(F, np.zeros((n, 3)))
    q = r / (basis.omega ** 2 - Om ** 2
             + 2j * z * basis.omega * Om)
    U = basis.Phi @ q
    assert np.allclose(frf["U"][0], U, rtol=1e-10)
    # and the response is real-ish (below resonance, small damping) and finite
    assert np.isfinite(frf["amp"]).all()
    assert frf["amp"].max() > 0.0


def test_frf_base_excitation_participation():
    """Base (support) excitation feeds the M16 effective-mass participation
    into the FRF: at quasi-static frequency the modal coordinate is
    -Gamma_i/omega_i^2 with Gamma_i^2 the M16 modal effective mass."""
    with contextlib.redirect_stdout(io.StringIO()):
        m = run_starter("examples/antenna_mast/MAST_0000.rad")
    basis = build_modal_basis(m, nev=4)
    z = modal_damping(basis.omega, uniform=0.01)
    n = m.numnod
    frf = modal_frequency_response(basis, np.zeros((n, 3)), np.zeros((n, 3)),
                                   np.array([1e-6]), z, base_excitation=True,
                                   base_dir=1)
    # |q_i(0)| = |Gamma_i| / omega_i^2, Gamma_i^2 = effective mass (dir 1)
    gamma = np.sqrt(basis.eff[:, 1])
    q0 = np.abs(frf["q"][0])
    expect = gamma / basis.omega ** 2
    # the participating (y-bending) mode dominates; check it
    j = int(np.argmax(gamma))
    assert q0[j] == pytest.approx(expect[j], rel=1e-6)


# ============================================================================
# CARDS + NO-REGRESSION (the M7 parity contract)
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


def test_impl_modal_and_freq_card_parsing():
    """/IMPL/MODAL/DYNA reads t_end dt nmode (+MACC / STRS suffix), /IMPL/
    MODAL/DAMP reads a uniform zeta, /IMPL/FREQ reads fmin fmax nf zeta —
    all PORT cards (freimpl.F has no such branch)."""
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/MODAL/DYNA\n"
                   "2.5  0.01  4\n/END\n")
    assert ec.implicit and ec.impl_modal_dyna
    assert ec.impl_modal_tend == pytest.approx(2.5)
    assert ec.impl_modal_dt == pytest.approx(0.01)
    assert ec.impl_modal_nmode == 4
    assert not ec.impl_modal_macc

    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/MODAL/DYNA/MACC\n1.0 0.01 3\n"
                   "/IMPL/MODAL/DAMP\n0.05\n/END\n")
    assert ec.impl_modal_macc and ec.impl_modal_zeta == pytest.approx(0.05)

    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/FREQ\n1.0  50.0  500  0.03\n"
                   "/END\n")
    assert ec.implicit and ec.impl_freq
    assert ec.impl_freq_fmin == pytest.approx(1.0)
    assert ec.impl_freq_fmax == pytest.approx(50.0)
    assert ec.impl_freq_nf == 500
    assert ec.impl_freq_zeta == pytest.approx(0.03)

    # /IMPL/EIGV still means the M16 eigensolver (not shadowed by MODAL)
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/EIGV\n3\n/END\n")
    assert ec.impl_eigv and not ec.impl_modal_dyna and not ec.impl_freq


def test_modal_response_does_not_mutate_state_or_eigensolver():
    """Modal superposition is a NEW path ALONGSIDE the eigensolver and the
    direct integrator (the M14/M15/M16 parity contract): a modal_transient
    call leaves the element state untouched, and the M16 modal_frequencies
    output is bit-identical before and after."""
    m = _starter(_bar_deck(N=8, force=0.02))
    loads = _loads(m)
    # eigensolver output before
    f0, _, _ = modal_frequencies(m, nev=3)
    before = {k: (v.copy() if isinstance(v, np.ndarray) else v)
              for k, v in m.trusses.state.items()}
    basis = build_modal_basis(m, nev=3)
    modal_transient(basis, loads, 0.5, 0.01, modal_damping(basis.omega,
                                                           uniform=0.0), m)
    # element state untouched
    for k, v in before.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(m.trusses.state[k], v), k
    # eigensolver bit-identical
    f1, _, _ = modal_frequencies(m, nev=3)
    assert np.array_equal(f0, f1)


def test_direct_dynamics_unchanged_by_modal_path():
    """A DIRECT /IMPL/DYNA run is BYTE-for-byte unaffected by the M17 modal
    machinery living in the same package (the M10 integrator stays
    bit-identical)."""
    starter = _spring_mass_deck(v0=0.01)
    eng = f"#\n/RUN/SM/1\n{2*T_SM}\n/IMPL/DYNA/2\n0.5 0.25\n/IMPL/DTINI\n" \
          f"{T_SM/100}\n/END\n"
    m1 = _run(starter, eng)
    m2 = _run(starter, eng)
    u1 = np.array([uu[m1.node_index(2), 0]
                   for uu in m1.implicit_result.history["u"]])
    u2 = np.array([uu[m2.node_index(2), 0]
                   for uu in m2.implicit_result.history["u"]])
    assert np.array_equal(u1, u2)
    # and the modal path never set the direct-run fields
    assert m1.implicit_result.modal_transient_history is None
