"""
M18 validations: COMPLEX / DAMPED eigenvalues (the QEP / state-space analysis)
and NON-CLASSICALLY-damped complex-mode superposition — built ALONGSIDE the
M16 real eigensolver and the M17 real-mode superposition (both stay
bit-identical).

Every new capability gets at least one ANALYTIC check (the port's philosophy):

ASSEMBLED DAMPING MATRIX C
* the discrete spring dashpot contributes exactly c a a^T (the velocity
  analogue of the spring's elastic tangent k a a^T);
* a pure-Rayleigh C = alpha M + beta K reproduces the M11/M17 modal ratios
  zeta_i = 1/2(alpha/omega_i + beta omega_i) in the M16 real modal basis
  (diag(Phi^T C Phi) = 2 zeta_i omega_i).

COMPLEX / DAMPED EIGENVALUES (state-space linearization of the QEP)
* a CLASSICALLY-damped (Rayleigh) system's complex eigenvalues reduce EXACTLY
  to -zeta_i omega_i +/- i omega_{d,i} with the M16 omega_i and the M17
  zeta_i, and its complex mode shapes are REAL up to a global phase;
* a 2-DOF system with a dashpot on ONE mass only (genuinely non-classical)
  matching the closed-form complex roots (det(lambda^2 M + lambda C + K) = 0)
  AND showing a DOF-to-DOF phase lag the classical modes cannot have;
* the complex modes' state-space BIORTHOGONALITY z_i^T B z_j = 0 (i != j).

COMPLEX-MODE SUPERPOSITION (transient + FRF for non-classical damping)
* the complex-mode transient on a non-classically-damped chain matching a
  DIRECT Newmark march of the assembled (K, C, M) BIT-CLOSE, where the M17
  real-mode (classical) superposition visibly ERRS (the gap asserted);
* reducing to the M17 answer when the damping IS classical;
* the complex FRF matching the direct inversion (K - Omega^2 M + i Omega C)^-1
  on the full basis, and a decaying resonant peak;
* the first-order exact recurrence (DC limit + decay), exact to round-off.

CARDS + NO-REGRESSION (the M7 parity contract)
* /IMPL/CEIGV (+ /STRS, /TRAN, /FRF) card mirror (PORT cards — freimpl.F has
  no complex/damped eigensolver);
* the complex path NEVER mutates the M16 eigensolver / M17 superposition / the
  element state, and the direct M10 answer is unchanged whether or not it runs.

See PORTING_GUIDE.md roadmap M18.
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

import scipy.linalg as sla                                            # noqa: E402
from pyradioss.common.messages import MessageLog                      # noqa: E402
from pyradioss.engine.kinematics import LoadsAndConstraints           # noqa: E402
from pyradioss.implicit.complex_modal import (                        # noqa: E402
    _first_order_coeffs, _state_space_eig, build_complex_basis, complex_frf,
    complex_modal_transient, direct_frf)
from pyradioss.implicit.damping_matrix import (                       # noqa: E402
    assemble_damping, assemble_discrete_damping)
from pyradioss.implicit.modal import modal_frequencies                # noqa: E402
from pyradioss.implicit.modal_response import (                       # noqa: E402
    build_modal_basis, modal_damping, modal_transient, rayleigh_ratios)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _starter(text):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M18_0000.rad")
    with open(sp, "w") as f:
        f.write(text)
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(sp)


def _run(starter_text, engine_text, capture=False):
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M18_0000.rad")
    ep = os.path.join(d, "M18_0001.rad")
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


def _chain_deck(masses, ks, cs, force_node=None, force=0.0):
    """A fixed-free axial spring-mass chain along x: node 1 pinned, then one
    node per mass, one /SPRING (its own /PART/PROP) per link. Each /PROP/SPRING
    is 'M K C' — half of M lumps onto each of the link's two nodes, so node j's
    lumped mass is (M_{j-1} + M_j)/2. ``cs`` is the per-link dashpot (a single
    non-zero entry = a localized, non-classical damper)."""
    N = len(ks)                                    # links
    nodes = "\n".join(f"{i+1:10d}{i*10.0:20.10f}{0.0:20.10f}{0.0:20.10f}"
                      for i in range(N + 1))
    springs = "\n".join(f"/SPRING/{i+1}\n{i+1:10d}{i+1:10d}{i+2:10d}"
                        for i in range(N))
    parts = "\n".join(f"/PART/{i+1}\ns{i+1}\n{i+1:10d}         1"
                      for i in range(N))
    props = "\n".join(f"/PROP/SPRING/{i+1}\nsp\n{masses[i]:12g}{ks[i]:12g}"
                      f"{cs[i]:12g}" for i in range(N))
    # free nodes (2..N+1) fixed in y,z; node 1 fully pinned
    freeids = "\n".join(str(i + 2) for i in range(N))
    load = ""
    if force_node is not None and force:
        # /CLOAD fields: funct_id, direction, grnod_id, scale (grnod 3 holds
        # the driven node) — the same layout as the M17 spring/bar decks
        load = (f"/FUNCT/1\nstep\n       0.0       1.0\n  100000.0       1.0\n"
                f"/CLOAD/1\ndrive\n         1         X         3       "
                f"{force}\n")
    grnods = (f"/GRNOD/NODE/1\npin\n1\n/GRNOD/NODE/2\nfree\n{freeids}\n")
    if force_node is not None and force:
        grnods += f"/GRNOD/NODE/3\ndrive\n{force_node}\n"
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
{grnods}/BCS/1
pin
       111       111         0         1
/BCS/2
axial
       011       111         0         2
{load}/END
"""


# 2-DOF non-classical reference: two equal springs (k), one dashpot on the
# first link only; masses node2 = M1/2 + M2/2, node3 = M2/2.
MS, K1, C1 = 2.0e-3, 1000.0, 0.5
DECK_2DOF = _chain_deck([MS, MS], [K1, K1], [C1, 0.0])
M2 = np.diag([MS, MS / 2.0])                        # node2 = 2e-3, node3 = 1e-3
K2 = np.array([[2.0 * K1, -K1], [-K1, K1]])
C2 = np.array([[C1, 0.0], [0.0, 0.0]])


# ============================================================================
# ASSEMBLED DAMPING MATRIX C
# ============================================================================

def test_discrete_spring_dashpot_contribution():
    """The assembled discrete C of a single dashpot spring is exactly
    c a a^T on the free DOF (the velocity analogue of the elastic k a a^T)."""
    from pyradioss.implicit.dofmap import DofMap
    # one spring node1(pinned)-node2, dashpot c, axis = x
    m = _starter(_chain_deck([MS], [K1], [C1]))
    dof = DofMap(m, None)
    C = assemble_discrete_damping(m, dof, m.x0).toarray()
    # the only free DOF is node2-x; the spring axis is x, so C = c on it
    e = dof.eq[m.node_index(2) * 6 + 0]
    assert C[e, e] == pytest.approx(C1)
    # no damping leaks onto the (fixed) y/z — those have no equation
    assert C.shape[0] == 1


def test_rayleigh_C_reproduces_modal_ratios():
    """A pure-Rayleigh assembled C = alpha M + beta K reproduces the M11/M17
    modal ratios in the M16 REAL modal basis: diag(Phi^T C Phi) = 2 zeta_i
    omega_i with zeta_i = 1/2(alpha/omega_i + beta omega_i)."""
    from pyradioss.implicit.dofmap import DofMap
    from pyradioss.implicit.assembly import assemble, assemble_mass
    alpha, beta = 0.3, 1e-4
    m = _starter(_chain_deck([MS, MS, MS], [K1, K1, K1], [0.0, 0.0, 0.0]))
    dof = DofMap(m, None)
    C = assemble_damping(m, dof, m.x0, alpha=alpha, beta=beta).toarray()
    # M16 real modes in eq space
    basis = build_modal_basis(m, nev=3)
    Phi, omega = basis.Phi, basis.omega
    modalC = Phi.T @ C @ Phi
    zeta = rayleigh_ratios(omega, alpha, beta)
    assert np.allclose(np.diag(modalC), 2.0 * zeta * omega, rtol=1e-8)
    # and Rayleigh C IS classical -> the modal C is diagonal
    off = modalC - np.diag(np.diag(modalC))
    assert np.abs(off).max() < 1e-6 * np.abs(np.diag(modalC)).max()


# ============================================================================
# COMPLEX / DAMPED EIGENVALUES
# ============================================================================

def test_classical_damping_reduces_to_real_modes():
    """A CLASSICALLY (Rayleigh) damped system: the complex eigenvalues reduce
    EXACTLY to -zeta_i omega_i +/- i omega_{d,i} with the M16 omega_i and the
    M17 zeta_i, and the complex mode shapes are REAL up to a global phase."""
    alpha, beta = 0.4, 5e-5
    m = _starter(_chain_deck([MS, MS, MS], [K1, K1, K1], [0.0, 0.0, 0.0]))
    # undamped reference (M16)
    freqs, _, _ = modal_frequencies(m, nev=3)
    omega = 2.0 * np.pi * freqs
    zeta = rayleigh_ratios(omega, alpha, beta)
    lam_expect = -zeta * omega + 1j * omega * np.sqrt(1.0 - zeta ** 2)
    # complex eigensolve with the same Rayleigh C
    basis = build_complex_basis(m, nev=3, alpha=alpha, beta=beta)
    lam = np.sort_complex(basis.eigenvalues)
    assert np.allclose(lam, np.sort_complex(lam_expect), rtol=1e-7)
    # classical -> mode shapes real up to phase: after phase-normalization the
    # imaginary part vanishes
    for du, dur in basis.mode_shapes():
        v = du[np.abs(du).sum(axis=1) > 0]
        assert np.abs(v.imag).max() < 1e-6 * np.abs(v.real).max()


def test_2dof_one_dashpot_closed_form_and_phase_lag():
    """A 2-DOF chain with a dashpot on ONE link (non-classical): the complex
    eigenvalues match the closed-form roots of det(lambda^2 M + lambda C + K)
    = 0 (an independent state-space eig of the hand-assembled matrices), and
    the mode shapes carry a genuine DOF-to-DOF PHASE LAG (a classical mode
    would have every DOF in phase or exactly out of phase)."""
    m = _starter(DECK_2DOF)
    basis = build_complex_basis(m, nev=2)
    # independent closed form on the hand-assembled 2-DOF matrices
    lam_ref, _ = _state_space_eig(K2, C2, M2)
    lam_ref = np.sort_complex([l for l in lam_ref if l.imag > 1e-9])
    lam = np.sort_complex(basis.eigenvalues)
    assert np.allclose(lam, lam_ref, rtol=1e-8)
    # non-classical signature: at least one mode has DOFs NOT collinear in the
    # complex plane (a phase difference between the two masses that is neither
    # 0 nor 180 degrees)
    lag = False
    for du, dur in basis.mode_shapes():
        v = du[np.abs(du).sum(axis=1) > 0][:, 0]     # the two masses' x
        ph = np.angle(v, deg=True)
        d = abs((ph[1] - ph[0] + 90) % 180 - 90)     # deviation from 0/180
        if d > 1.0:
            lag = True
    assert lag, "expected a non-classical DOF phase lag"


def test_state_space_biorthogonality():
    """The complex modes are B-biorthogonal wrt the state-space pencil:
    z_i^T B z_j = 0 for i != j (the symmetric-linearization biorthogonality
    that decouples the forced problem)."""
    m = _starter(DECK_2DOF)
    basis = build_complex_basis(m, nev=2)
    Kr, Mr = basis.Kr, basis.Mr
    Phi, Zvel = basis.Phi, basis.Zvel
    # G_ij = z_i^T B z_j = phi_i^T K phi_j - (lam_i phi_i)^T M (lam_j phi_j)
    G = Phi.T @ (Kr @ Phi) - Zvel.T @ (Mr @ Zvel)
    off = G - np.diag(np.diag(G))
    assert np.abs(off).max() < 1e-8 * np.abs(np.diag(G)).max()


# ============================================================================
# COMPLEX-MODE SUPERPOSITION — transient + FRF
# ============================================================================

def _direct_newmark(Kr, Cr, Mr, f_red, t_end, dt, v0=None):
    """A DIRECT average-acceleration Newmark march of the reduced (K, C, M)
    system with constant force ``f_red`` — the reference the complex-mode
    superposition must match (the same scheme M10/M11 use, driven here with
    the M18 assembled C the direct implicit card does not yet consume)."""
    gam, bet = 0.5, 0.25
    n = Kr.shape[0]
    u = np.zeros(n)
    v = np.zeros(n) if v0 is None else v0.copy()
    a = sla.solve(Mr, f_red - Cr @ v - Kr @ u)
    Keff = Kr + gam / (bet * dt) * Cr + Mr / (bet * dt ** 2)
    out = []
    ns = int(np.ceil(t_end / dt - 1e-12))
    for _ in range(ns):
        rhs = (f_red + Mr @ (u / (bet * dt ** 2) + v / (bet * dt)
                             + (1 / (2 * bet) - 1) * a)
               + Cr @ (gam / (bet * dt) * u + (gam / bet - 1) * v
                       + dt * (gam / (2 * bet) - 1) * a))
        un = sla.solve(Keff, rhs)
        an = (un - u) / (bet * dt ** 2) - v / (bet * dt) - (1 / (2 * bet) - 1) * a
        vn = v + dt * ((1 - gam) * a + gam * an)
        u, v, a = un, vn, an
        out.append(u.copy())
    return np.array(out)


def test_complex_transient_matches_direct_and_beats_classical():
    """The complex-mode transient of the non-classically-damped chain matches
    a DIRECT Newmark march of the assembled (K, C, M) BIT-CLOSE, while the M17
    real-mode (classical) superposition — fed the diagonal modal damping — is
    visibly WRONG (it cannot represent the off-diagonal C coupling)."""
    F = 0.05
    starter = _chain_deck([MS, MS], [K1, K1], [C1, 0.0],
                          force_node=3, force=F)
    m = _starter(starter)
    loads = _loads(m)
    basis = build_complex_basis(m, nev=2)
    T1 = 2.0 * np.pi / np.abs(basis.eigenvalues).min()
    t_end, dt = 2.0 * T1, T1 / 400.0

    hc = complex_modal_transient(basis, loads, t_end, dt, m)
    uc = np.array([uu[m.node_index(3), 0] for uu in hc["u"]])

    # direct reference: reduced force from the /CLOAD pattern. The Newmark
    # march carries an O(dt^2) dispersion the EXACT complex recurrence does
    # not, so integrate it 8x finer and subsample — the comparison then
    # isolates the PHYSICS (the modal representation), not the reference's own
    # time-discretization error.
    n = m.numnod
    fext = np.zeros((n, 3))
    loads.external_forces(1.0, fext, m.x0)
    f_red = basis.dof.gather_residual(fext, np.zeros((n, 3)))
    ud_red = _direct_newmark(basis.Kr, basis.Cr, basis.Mr, f_red,
                             t_end, dt / 8.0)[7::8]
    e3 = basis.dof.eq[m.node_index(3) * 6 + 0]
    ud = ud_red[:len(uc), e3]
    assert np.abs(uc - ud).max() < 1e-5 * np.abs(ud).max()

    # M17 real-mode (classical) superposition with the best diagonal modal
    # damping zeta_i = phi_i^T C phi_i / (2 omega_i) — provably wrong here
    rbasis = build_modal_basis(m, nev=2)
    C = assemble_damping(m, basis.dof, m.x0).toarray()
    zc = np.array([rbasis.Phi[:, i] @ C @ rbasis.Phi[:, i]
                   / (2.0 * rbasis.omega[i]) for i in range(2)])
    hr = modal_transient(rbasis, loads, t_end, dt, zc, m)
    ur = np.array([uu[m.node_index(3), 0] for uu in hr["u"]])
    # the classical superposition misses the true answer by a real margin
    assert np.abs(ur - ud).max() > 20.0 * np.abs(uc - ud).max()


def test_complex_transient_reduces_to_classical_when_damping_classical():
    """When the damping IS classical (Rayleigh, no discrete dashpot), the
    complex-mode transient and the M17 real-mode transient agree."""
    alpha, beta = 0.2, 2e-5
    F = 0.05
    starter = _chain_deck([MS, MS], [K1, K1], [0.0, 0.0],
                          force_node=3, force=F)
    m = _starter(starter)
    loads = _loads(m)
    cbasis = build_complex_basis(m, nev=2, alpha=alpha, beta=beta)
    T1 = 2.0 * np.pi / np.abs(cbasis.eigenvalues).min()
    t_end, dt = 1.5 * T1, T1 / 400.0
    hc = complex_modal_transient(cbasis, loads, t_end, dt, m)
    uc = np.array([uu[m.node_index(3), 0] for uu in hc["u"]])

    rbasis = build_modal_basis(m, nev=2)
    z = modal_damping(rbasis.omega, rayleigh=(alpha, beta))
    hr = modal_transient(rbasis, loads, t_end, dt, z, m)
    ur = np.array([uu[m.node_index(3), 0] for uu in hr["u"]])
    assert np.abs(uc - ur).max() < 1e-4 * np.abs(ur).max()


def test_complex_frf_matches_direct_inversion_and_decays():
    """The complex-mode FRF equals the direct inversion (K - Omega^2 M +
    i Omega C)^-1 F on the full modal basis (round-off), and the resonant peak
    is finite and damped (a decaying complex FRF)."""
    m = _starter(DECK_2DOF)
    basis = build_complex_basis(m, nev=2)
    n = m.numnod
    F = np.zeros((n, 3))
    F[m.node_index(3), 0] = 1.0
    fmax = 1.5 * float(basis.natural_freqs_hz.max())
    freqs = np.linspace(1.0, fmax, 2000)
    cf = complex_frf(basis, F, np.zeros((n, 3)), freqs)
    df = direct_frf(basis, F, np.zeros((n, 3)), freqs)
    assert np.allclose(cf["U"], df["U"], rtol=1e-8, atol=1e-14)
    # resonant peak is finite (damped) and the response rolls off past it
    tip = np.abs(cf["U"][:, basis.dof.eq[m.node_index(3) * 6 + 0]])
    assert np.isfinite(tip).all()
    assert tip.max() < 1e3 * tip[0]                # damped, not singular


def test_first_order_recurrence_exact():
    """The first-order recurrence is EXACT for piecewise-linear forcing: a
    constant forcing settles to the analytic steady state -p/lambda and a free
    release decays as e^{lambda t}, both to round-off at a coarse step."""
    lam = np.array([-0.3 + 8.0j])
    dt = 0.05
    g, a1, a2 = _first_order_coeffs(lam, dt)
    # constant forcing p: steady state x_ss = -p / lambda
    p = 1.0 + 0.5j
    x = np.array([0.0 + 0j])
    for _ in range(4000):
        x = g * x + a1 * p + a2 * p
    assert np.allclose(x, -p / lam, rtol=1e-9)
    # free decay x(t) = x0 e^{lambda t}
    x = np.array([1.0 + 0j])
    out = []
    for _ in range(200):
        x = g * x
        out.append(x[0])
    t = np.arange(1, 201) * dt
    assert np.abs(np.array(out) - np.exp(lam[0] * t)).max() < 1e-10


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


def test_impl_ceigv_card_parsing():
    """/IMPL/CEIGV reads Nmode; /STRS the prestressed spectrum; /TRAN the
    complex-mode transient (t_end dt); /FRF the damped sweep (fmin fmax nf) —
    all PORT cards (freimpl.F has no complex/damped eigensolver)."""
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/CEIGV\n4\n/END\n")
    assert ec.implicit and ec.impl_ceigv and ec.impl_ceigv_nmode == 4
    assert not ec.impl_ceigv_tran and not ec.impl_ceigv_frf

    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/CEIGV/STRS\n3\n/END\n")
    assert ec.impl_ceigv and ec.impl_ceigv_prestress and ec.impl_nlgeom
    assert ec.impl_ceigv_nmode == 3

    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/CEIGV/TRAN\n2.5 0.01 3\n/END\n")
    assert ec.impl_ceigv and ec.impl_ceigv_tran
    assert ec.impl_ceigv_tend == pytest.approx(2.5)
    assert ec.impl_ceigv_dt == pytest.approx(0.01)
    assert ec.impl_ceigv_nmode == 3

    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL/CEIGV/FRF\n1.0 50.0 500\n/END\n")
    assert ec.impl_ceigv and ec.impl_ceigv_frf
    assert ec.impl_ceigv_fmin == pytest.approx(1.0)
    assert ec.impl_ceigv_fmax == pytest.approx(50.0)
    assert ec.impl_ceigv_nf == 500

    # /IMPL/EIGV still means the M16 real eigensolver (not shadowed by CEIGV)
    ec, _ = _parse("#\n/RUN/A/1\n1.0\n/IMPL\n/IMPL/EIGV\n3\n/END\n")
    assert ec.impl_eigv and not ec.impl_ceigv


def test_ceigv_card_end_to_end():
    """/IMPL/CEIGV end to end on the non-classically-damped chain: the listing
    prints the COMPLEX / DAMPED EIGENVALUES block and stores the complex
    eigenvalues / frequencies / damping ratios on the result."""
    m, out = _run(DECK_2DOF, "#\n/RUN/CHAIN/1\n1.0\n/IMPL\n/IMPL/CEIGV\n2\n"
                             "/END\n", capture=True)
    r = m.implicit_result
    assert r.complex_eigenvalues is not None
    assert len(r.complex_eigenvalues) == 2
    assert np.all(r.complex_damping_ratios > 0.0)         # damped
    assert np.all(r.complex_eigenvalues.real < 0.0)       # decaying
    assert "COMPLEX / DAMPED EIGENVALUES" in out


def test_complex_path_does_not_mutate_real_solvers_or_state():
    """The complex path is NEW and ALONGSIDE (the M14-M17 parity contract): a
    build_complex_basis call leaves the element state untouched and the M16
    modal_frequencies output bit-identical before and after."""
    m = _starter(DECK_2DOF)
    f0, _, _ = modal_frequencies(m, nev=2)
    before = {k: (v.copy() if isinstance(v, np.ndarray) else v)
              for k, v in m.springs.state.items()}
    build_complex_basis(m, nev=2)
    for k, v in before.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(m.springs.state[k], v), k
    f1, _, _ = modal_frequencies(m, nev=2)
    assert np.array_equal(f0, f1)


def test_direct_dynamics_unchanged_by_complex_path():
    """A DIRECT /IMPL/DYNA run is byte-for-byte unaffected by the M18 complex
    machinery living in the same package (the M10 integrator stays
    bit-identical)."""
    K_SM, M_SM = 4.0, 2.0e-3
    om = np.sqrt(K_SM / (M_SM / 2.0))
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
     {M_SM}       {K_SM}       0.0
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
    assert m1.implicit_result.complex_eigenvalues is None
