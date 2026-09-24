"""
Tests for Component 3 & 4 of Milestone M614:
- Component 3: Implicit Dynamics (Generalized-alpha, Step Fixpoints, Tangent Policies,
  Accurate Energy Dissipation Tracking).
- Component 4: Modal Analysis (Sparse Shift-and-Invert Lanczos Eigensolver, Sturm Sequence Check,
  Rigid Body Modes, Modal Participation & Mass Completeness).

References:
- Chung & Hulbert (1993), "A Time Integration Algorithm for Structural Dynamics with
  Improved Numerical Dissipation: The Generalized-alpha Method", ASME J. Appl. Mech. 60(2).
- Wood, Bossak, Zienkiewicz (1980), "An alpha modification of Newmark's method",
  Int. J. Numer. Meth. Engng. 15.
- OpenRadioss upstream Fortran references:
  * engine/source/implicit/imp_dyna.F (DYNA_INI, IMP_DYNAM, IMP_DYNAR, INTE_DYNA, DYNA_WEX)
  * engine/source/implicit/imp_dt.F (IMP_DTF, IMP_DTN)
  * engine/source/implicit/imp_lanz.F (EIGBUCKP, Lanczos shift-and-invert)
  * engine/source/input/freimpl.F (reader cards: /IMPL/DYNA, /IMPL/NONLIN/KTANG/KTFUL/KTCON, /IMPL/DT/FIXP)
"""

import contextlib
import io
import os
import tempfile

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("scipy")

from pyradioss.engine.engine import run_engine
from pyradioss.starter.starter import run_starter
from pyradioss.implicit.modal import (
    modal_frequencies,
    sparse_lanczos_eigen,
    sturm_sequence_check,
    compute_rigid_body_modes,
    modal_participation,
    verify_modal_mass_completeness,
    ModalAnalysisResult,
    ModalEffectiveMass,
)
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.engine_keywords import parse_engine_deck
from pyradioss.common.messages import MessageLog


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

L_SD, E_SD, A_SD, RHO_SD = 100.0, 210.0, 1.0, 7.8e-6
M_SD = RHO_SD * A_SD * L_SD / 2.0
K_SD = E_SD * A_SD / L_SD
OM_SD = np.sqrt(K_SD / M_SD)
T_SD = 2.0 * np.pi / OM_SD


def _sdof_deck(v0=0.0, force=0.0):
    load = ""
    if force:
        load = f"""/FUNCT/1
step
       0.0       1.0
    1000.0       1.0
/CLOAD/1
axial step
         1         X         2       {force}
"""
    inivel = ""
    if v0:
        inivel = f"""/INIVEL/TRA/1
kick
      {v0}       0.0       0.0         2
"""
    return f"""\
#RADIOSS STARTER
/BEGIN
SDOF
/NODE
         1{0.0:20.10f}{0.0:20.10f}{0.0:20.10f}
         2{L_SD:20.10f}{0.0:20.10f}{0.0:20.10f}
/TRUSS/1
         1         1         2
/PART/1
bar
         1         1
/MAT/LAW1/1
steel
   {RHO_SD}
     {E_SD}       0.0
/PROP/TRUSS/1
bar
        {A_SD}
/GRNOD/NODE/1
pivot
1
/GRNOD/NODE/2
tip
2
/BCS/1
pin
       111       000         0         1
/BCS/2
axial only
       011       000         0         2
{inivel}{load}/END
"""


def _bar_chain_deck(nel=8, v0=0.01):
    dx = L_SD / nel
    nodes = [f"{i+1:10d}{i*dx:20.10f}{0.0:20.10f}{0.0:20.10f}"
             for i in range(nel + 1)]
    bars = [f"{i+1:10d}{i+1:10d}{i+2:10d}" for i in range(nel)]
    moving = " ".join(str(i + 2) for i in range(nel))
    return f"""\
#RADIOSS STARTER
/BEGIN
BAR
/NODE
{chr(10).join(nodes)}
/TRUSS/1
{chr(10).join(bars)}
/PART/1
bar
         1         1
/MAT/LAW1/1
steel
   {RHO_SD}
     {E_SD}       0.0
/PROP/TRUSS/1
bar
        {A_SD}
/GRNOD/NODE/1
root
1
/GRNOD/NODE/2
moving
{moving}
/BCS/1
clamp root
       111       000         0         1
/BCS/2
axial only
       011       000         0         2
/INIVEL/TRA/1
uniform kick
      {v0}       0.0       0.0         2
/END
"""


def _cantilever_shell_deck(nx=12, L=100.0, b=10.0, t=1.0):
    def nid(i, j):
        return 1 + i + j * (nx + 1)
    nodes = "\n".join(
        f"{nid(i,j):10d}{i*L/nx:20.10f}{j*b:20.10f}{0.0:20.10f}"
        for j in range(2) for i in range(nx + 1))
    els = "\n".join(
        f"{i+1:10d}{nid(i,0):10d}{nid(i+1,0):10d}{nid(i+1,1):10d}{nid(i,1):10d}"
        for i in range(nx))
    fixed = "\n".join(str(nid(0, j)) for j in range(2))
    return f"""\
#RADIOSS STARTER
/BEGIN
STRIP
/NODE
{nodes}
/SHELL/1
{els}
/PART/1
p
         1         1
/MAT/LAW1/1
steel
   7.8e-6
     210.0       0.3
/PROP/SHELL/1
sh
       {t}         5      0.833
/GRNOD/NODE/1
clamp
{fixed}
/BCS/1
clamp
       111       111         0         1
/END
"""


def _free_free_truss_deck():
    return f"""#RADIOSS STARTER
/BEGIN
FREE_BAR
/NODE
         1        0.0000000000        0.0000000000        0.0000000000
         2       25.0000000000        0.0000000000        0.0000000000
         3       50.0000000000        0.0000000000        0.0000000000
         4       75.0000000000        0.0000000000        0.0000000000
/TRUSS/1
         1         1         2
         2         2         3
         3         3         4
/PART/1
bar
         1         1
/MAT/LAW1/1
steel
   7.8e-6
     210.0       0.0
/PROP/TRUSS/1
bar
       1.0
/END
"""


def _run_model(make_deck, name, starter, engine):
    s, e = make_deck(name, starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        model = run_engine(e)
    return model


# ===========================================================================
# Component 3 Tests: Implicit Dynamics
# ===========================================================================

def test_generalized_alpha_recovers_hht(make_deck):
    """Verify that Generalized-alpha with alpha_m = 0 recovers HHT-alpha bit-identically:
    Equilibrium at intermediate configuration t_{n+1-alpha_f}:
      gamma = 0.5 + alpha_f, beta = 0.25 * (1 + alpha_f)^2,
      matching HHT-alpha with alpha = -alpha_f.
    Cite: Chung & Hulbert (1993) Eq. (7)-(10), imp_dyna.F DYNA_INI.
    """
    alpha_f = 0.15
    alpha_hht = -0.15

    engine_hht = f"""#
/RUN/HHT/1
0.04
/IMPL/DYNA/1
{alpha_hht}
/IMPL/DTINI
0.002
/END
"""

    engine_gen = f"""#
/RUN/GEN/1
0.04
/IMPL/DYNA/3
0.0 {alpha_f}
/IMPL/DTINI
0.002
/END
"""

    m_hht = _run_model(make_deck, "HHT", _sdof_deck(v0=0.01), engine_hht)
    m_gen = _run_model(make_deck, "GEN", _sdof_deck(v0=0.01), engine_gen)

    res_hht = m_hht.implicit_result
    res_gen = m_gen.implicit_result

    # Check scheme coefficients
    assert abs(res_gen.alpha_m - 0.0) < 1e-14
    assert abs(res_gen.alpha_f - alpha_f) < 1e-14
    assert abs(res_gen.gamma - (0.5 + alpha_f)) < 1e-14
    assert abs(res_gen.beta - (0.25 * (1.0 + alpha_f) ** 2)) < 1e-14
    assert abs(res_hht.gamma - res_gen.gamma) < 1e-14
    assert abs(res_hht.beta - res_gen.beta) < 1e-14

    # Check displacement and numerical dissipation bit-identity
    u_hht = np.array([uu[1, 0] for uu in res_hht.history["u"]])
    u_gen = np.array([uu[1, 0] for uu in res_gen.history["u"]])
    np.testing.assert_allclose(u_gen, u_hht, rtol=1e-12, atol=1e-14)

    enum_hht = np.array(res_hht.history["enum"])
    enum_gen = np.array(res_gen.history["enum"])
    np.testing.assert_allclose(enum_gen, enum_hht, rtol=1e-12, atol=1e-14)


def test_wood_bossak_dynamics(make_deck):
    """Verify Wood-Bossak time integration (alpha_f = 0, alpha_m > 0):
      gamma = 0.5 - alpha_m, beta = 0.25 * (1 - alpha_m)^2.
    Cite: Wood, Bossak, Zienkiewicz (1980), imp_dyna.F.
    """
    alpha_m = 0.1
    engine_wb = f"""#
/RUN/WB/1
0.04
/IMPL/DYNA/3
{alpha_m} 0.0
/IMPL/DTINI
0.002
/END
"""
    m_wb = _run_model(make_deck, "WB", _sdof_deck(v0=0.01), engine_wb)
    res = m_wb.implicit_result

    expected_gamma = 0.5 - alpha_m  # 0.4
    expected_beta = 0.25 * (1.0 - alpha_m) ** 2  # 0.2025
    assert abs(res.gamma - expected_gamma) < 1e-14
    assert abs(res.beta - expected_beta) < 1e-14
    assert abs(res.alpha_m - alpha_m) < 1e-14
    assert abs(res.alpha_f - 0.0) < 1e-14
    assert res.converged


def test_generalized_alpha_high_mode_dissipation(make_deck):
    """Verify high-frequency numerical dissipation on an 8-element bar:
    A uniform velocity kick excites all modes. The trapezoidal rule
    conserves total energy (Enum ~= 0), while Generalized-alpha
    monotonically drains unresolved high-mode content with Enum > 0.
    Cite: Chung & Hulbert (1993), imp_dyna.F INTE_DYNA / DYNA_WEX.
    """
    deck_start = _bar_chain_deck(nel=8, v0=0.01)

    nel, v0 = 8, 0.01
    c = np.sqrt(E_SD / RHO_SD)
    T1 = 4.0 * L_SD / c
    dt = T1 / 16.0
    e0 = 0.5 * (RHO_SD * A_SD * L_SD - RHO_SD * A_SD * L_SD / nel / 2.0) * v0 ** 2

    # 1. Trapezoidal rule: no numerical dissipation
    engine_trap = f"""#
/RUN/TRAP/1
{4.0 * T1}
/IMPL/DYNA/2
0.5 0.25
/IMPL/DTINI
{dt}
/END
"""
    # 2. Generalized-alpha with alpha_m=0.1, alpha_f=0.3
    engine_ga = f"""#
/RUN/GA/1
{4.0 * T1}
/IMPL/DYNA/3
0.1 0.3
/IMPL/DTINI
{dt}
/END
"""
    m_trap = _run_model(make_deck, "TRAP", deck_start, engine_trap)
    m_ga = _run_model(make_deck, "GA", deck_start, engine_ga)

    enum_trap = m_trap.implicit_result.history["enum"]
    enum_ga = m_ga.implicit_result.history["enum"]

    # Trapezoidal rule numerical dissipation stays negligible (to within round-off)
    assert abs(enum_trap[-1]) < 1e-12 * e0

    # Generalized-alpha absorbs significant energy from the unresolved modes (> 10% of e0)
    assert enum_ga[-1] > 0.10 * e0
    # Dissipation grows monotonically over time
    assert enum_ga[-1] > enum_ga[len(enum_ga) // 2]


def test_energy_dissipation_tracking_exact_balance(make_deck):
    """Verify that discrete numerical dissipation Enum closes the energy balance
    to machine precision:
      KE + IE + Edamp + Enum = Wext + E0  (< 10^-12 absolute error).
    Tests with applied external load and Rayleigh damping (/IMPL/DYNA/DAMP).
    Cite: imp_dyna.F DYNA_WEX, dynamics.py _dyn_summary.
    """
    engine = """#
/RUN/BAL/1
0.03
/IMPL/DYNA/3
0.05 0.15
/IMPL/DYNA/DAMP
0.5  0.001
/IMPL/DTINI
0.002
/END
"""
    m = _run_model(make_deck, "BAL", _sdof_deck(force=10.0), engine)
    h = m.implicit_result.history

    ke = np.array(h["ke"])
    ie = np.array(h["ie"])
    wext = np.array(h["wext"])
    edamp = np.array(h["edamp"])
    enum = np.array(h["enum"])
    bal = np.array(h["bal"])

    # Initial energy at t = 0 before any motion or load
    e0 = 0.0

    # Verify balance at every step: KE + IE + Edamp + Enum = Wext + E0
    total_energy = ke + ie + edamp + enum
    expected_energy = wext + e0
    diff = np.abs(total_energy - expected_energy)

    np.testing.assert_allclose(diff, 0.0, atol=1e-12)
    # Check that bal + enum == 0 to machine precision
    np.testing.assert_allclose(bal + enum, 0.0, atol=1e-13)
    assert edamp[-1] > 0.0  # Rayleigh damping dissipated positive energy
    assert enum[-1] != 0.0  # Algorithmic dissipation recorded


def test_step_fixpoints_exact_landing_and_recovery(make_deck):
    """Verify /IMPL/DT/FIXP exact landing and step size restoration:
    The solver must hit every fixpoint exactly without overshooting, and
    restore the regular controller step size on the subsequent step.
    Cite: imp_dt.F lines 122-140 (IMP_DTF), dynamics.py lines 634-645.
    """
    fixpoints = [0.01, 0.025, 0.038]
    engine = f"""#
/RUN/FIXP/1
0.05
/IMPL/DYNA/2
0.5 0.25
/IMPL/DTINI
0.008
/IMPL/DT/FIXP
{' '.join(str(tf) for tf in fixpoints)}
/END
"""
    m = _run_model(make_deck, "FIXP", _sdof_deck(v0=0.01), engine)
    t_hist = np.array(m.implicit_result.history["t"])

    # Every fixpoint must be hit exactly
    for tf in fixpoints:
        min_dist = float(np.min(np.abs(t_hist - tf)))
        assert min_dist < 1e-12, f"Fixpoint {tf} was not hit (min dist = {min_dist})"

    # After hitting fixpoint 0.01 (step index 1), step 2 advances by full controller dt = 0.008 -> t = 0.018
    idx_fp1 = np.where(np.abs(t_hist - 0.01) < 1e-12)[0][0]
    next_dt = t_hist[idx_fp1 + 1] - t_hist[idx_fp1]
    assert abs(next_dt - 0.008) < 1e-12, f"Step after fixpoint did not restore controller dt: {next_dt}"


def test_tangent_update_policies_ikt(make_deck):
    """Verify tangent update policies IKT (1=KTANG, 2=KTFUL, 4=KTCON):
    - KTANG: rebuild tangent every Newton iteration (full Newton)
    - KTFUL: rebuild tangent at start of each time step (modified Newton)
    - KTCON: assemble tangent once at simulation start and reuse across all steps
    All policies must converge to consistent physical displacements, with KTCON
    taking more iterations per step due to the frozen tangent.
    Cite: freimpl.F IKT read, imp_dyna.F / imp_solv.F tangent policies.
    """
    def run_policy(ikt_card):
        engine = f"""#
/RUN/IKT/1
0.02
/IMPL/DYNA/2
0.5 0.25
{ikt_card}
/IMPL/DTINI
0.002
/END
"""
        m = _run_model(make_deck, "IKT", _sdof_deck(force=10.0), engine)
        u = np.array([uu[1, 0] for uu in m.implicit_result.history["u"]])
        total_iters = sum(inc.iterations for inc in m.implicit_result.increments)
        return u, total_iters

    u_ktang, it_ktang = run_policy("/IMPL/NONLIN/KTANG")
    u_ktful, it_ktful = run_policy("/IMPL/NONLIN/KTFUL")
    u_ktcon, it_ktcon = run_policy("/IMPL/NONLIN/KTCON")

    # Consistent physical displacements
    np.testing.assert_allclose(u_ktang, u_ktful, atol=1e-8)
    np.testing.assert_allclose(u_ktang, u_ktcon, rtol=1e-5, atol=1e-6)

    # Iteration efficiency: KTANG <= KTFUL <= KTCON
    assert it_ktang <= it_ktful
    assert it_ktful <= it_ktcon


# ===========================================================================
# Component 4 Tests: Modal Analysis
# ===========================================================================

def test_sparse_lanczos_vs_dense_eigh():
    """Verify sparse shift-and-invert Lanczos eigensolver (scipy.sparse.linalg.eigsh):
      (K - sigma M)^-1 M phi = mu phi, where lambda = sigma + 1/mu
    Matches dense scipy.linalg.eigh eigenvalues and produces M-orthonormal eigenvectors:
      phi_i^T M phi_j = delta_ij
    Cite: imp_lanz.F EIGBUCKP, modal.py sparse_lanczos_eigen.
    """
    deck = _cantilever_shell_deck(nx=10)
    with tempfile.TemporaryDirectory() as d:
        sp_path = os.path.join(d, "MODAL_0000.rad")
        with open(sp_path, "w") as f:
            f.write(deck)
        with contextlib.redirect_stdout(io.StringIO()):
            m = run_starter(sp_path)

    f_dense, m_dense, eff_dense = modal_frequencies(m, nev=6, sparse=False)
    f_sparse, m_sparse, eff_sparse = modal_frequencies(m, nev=6, sparse=True)

    # Verify natural frequencies match dense eigh to < 10^-5 relative error
    rel_diff = np.abs(f_sparse - f_dense) / f_dense
    assert np.all(rel_diff < 1e-5), f"Max relative difference: {np.max(rel_diff)}"

    # Verify M-orthonormality of sparse modes
    from pyradioss.implicit.dofmap import DofMap
    from pyradioss.implicit.assembly import assemble_mass
    dof = DofMap(m)
    M = assemble_mass(m, dof, m.x0)
    for i, (du_i, dur_i) in enumerate(m_sparse):
        phi_i = dof.gather_residual(du_i, dur_i)
        for j, (du_j, dur_j) in enumerate(m_sparse):
            phi_j = dof.gather_residual(du_j, dur_j)
            val = float(phi_i.T @ (M @ phi_j))
            expected = 1.0 if i == j else 0.0
            assert abs(val - expected) < 1e-8, f"Orthogonality failure at ({i}, {j}): {val}"


def test_sturm_sequence_check():
    """Verify automatic Sturm sequence check using Sylvester's law of inertia:
    The number of negative eigenvalues of (K - sigma M) equals the exact
    count of natural frequencies below f_cutoff = sqrt(sigma) / (2 pi).
    Detects missed modes if fewer eigenvalues were extracted.
    Cite: modal.py sturm_sequence_check, imp_lanz.F Sturm check.
    """
    deck = _cantilever_shell_deck(nx=10)
    with tempfile.TemporaryDirectory() as d:
        sp_path = os.path.join(d, "MODAL_0000.rad")
        with open(sp_path, "w") as f:
            f.write(deck)
        with contextlib.redirect_stdout(io.StringIO()):
            m = run_starter(sp_path)

    # Extract first 5 frequencies
    freqs, _, _ = modal_frequencies(m, nev=5, sparse=False)

    # Choose cutoff frequency between mode 3 and mode 4
    f_cut = 0.5 * (freqs[2] + freqs[3])

    from pyradioss.implicit.dofmap import DofMap
    from pyradioss.implicit.assembly import assemble, assemble_mass
    dof = DofMap(m)
    K = assemble(m, dof, m.x0, kgeo=False)
    M = assemble_mass(m, dof, m.x0)

    # 1. Exact count check
    sturm_full = sturm_sequence_check(K, M, cutoff_freq_hz=f_cut, nev_found=3)
    assert sturm_full["n_below"] == 3
    assert sturm_full["missed"] == 0
    assert sturm_full["passed"] is True

    # 2. Missed modes detection: suppose the solver only found 1 mode
    sturm_incomplete = sturm_sequence_check(K, M, cutoff_freq_hz=f_cut, nev_found=1)
    assert sturm_incomplete["n_below"] == 3
    assert sturm_incomplete["missed"] == 2
    assert sturm_incomplete["passed"] is False


def test_rigid_body_modes_and_mass_completeness():
    """Verify rigid body mode extraction (omega ~= 0) and 100% modal mass completeness:
      sum_{k} m_eff,kd = M_total,d  in active rigid-body directions.
    On an unconstrained 3D truss, the rigid body translational modes
    must capture 100% of the total physical mass in Tx, Ty, Tz.
    Cite: modal.py compute_rigid_body_modes, modal_participation, verify_modal_mass_completeness.
    """
    deck = _free_free_truss_deck()
    with tempfile.TemporaryDirectory() as d:
        sp_path = os.path.join(d, "FREE_0000.rad")
        with open(sp_path, "w") as f:
            f.write(deck)
        with contextlib.redirect_stdout(io.StringIO()):
            m = run_starter(sp_path)

    rfreq, rmodes, reff, rpart = compute_rigid_body_modes(m)

    # Zero frequencies for rigid modes
    np.testing.assert_allclose(rfreq, 0.0, atol=1e-12)

    # Total physical mass
    total_physical_mass = float(np.sum(m.mass[m.mass < 1e20]))

    # Sum of effective mass in Tx, Ty, Tz
    eff_sums = np.sum(reff, axis=0)
    for d in range(3):  # Tx, Ty, Tz
        assert abs(eff_sums[d] - total_physical_mass) / total_physical_mass < 1e-4

    # Verify modal mass completeness helper
    tot_vec = np.zeros(6)
    tot_vec[:3] = total_physical_mass
    passed, rel_err = verify_modal_mass_completeness(reff, tot_vec, rtol=1e-3)
    assert passed
    assert np.all(rel_err[:3] < 1e-4)


def test_modal_effective_mass_metadata_and_details():
    """Verify ModalEffectiveMass array subclass and return_details=True API:
    ModalAnalysisResult contains participation factors, cumulative fractions,
    rigid modes, and Sturm sequence info with full metadata access.
    """
    deck = _cantilever_shell_deck(nx=8)
    with tempfile.TemporaryDirectory() as d:
        sp_path = os.path.join(d, "STRIP_0000.rad")
        with open(sp_path, "w") as f:
            f.write(deck)
        with contextlib.redirect_stdout(io.StringIO()):
            m = run_starter(sp_path)

    # Call with return_details=True and check_sturm=True
    res = modal_frequencies(m, nev=4, check_sturm=True, cutoff_freq=1.0, return_details=True)
    assert isinstance(res, ModalAnalysisResult)
    assert len(res.freqs_hz) == 4
    assert res.effective_mass.shape == (4, 6)
    assert res.participation_factors.shape == (4, 6)
    assert res.mass_fractions.shape == (4, 6)
    assert res.cumulative_fractions.shape == (4, 6)
    assert res.sturm_info is not None
    assert "n_below" in res.sturm_info

    # Verify ModalEffectiveMass metadata retention
    assert isinstance(res.effective_mass, ModalEffectiveMass)
    assert res.effective_mass.participation_factors is not None
    assert res.effective_mass.mass_fractions is not None
    assert res.effective_mass.total_mass is not None
    assert res.effective_mass.cumulative_fractions is not None


def test_engine_card_parsing_m614(tmp_path):
    """Verify engine parser support for all M614 cards:
    - /IMPL/DYNA/3 (Generalized-alpha)
    - /IMPL/NONLIN/KTANG, KTFUL, KTCON (tangent policies)
    - /IMPL/DT/FIXP (fixpoints sequence)
    Cite: freimpl.F, engine_keywords.py.
    """
    deck = """/RUN/M614/1
100.0
/IMPL/DYNA/3
0.15 0.25
/IMPL/NONLIN/KTFUL
/IMPL/DT/FIXP
0.05 0.12 0.35 1.2
"""
    p = tmp_path / "test_engine.rad"
    p.write_text(deck, encoding="utf-8")

    log = MessageLog()
    blocks = read_deck(str(p))
    ec = parse_engine_deck(blocks, log)

    assert ec.impl_dyna == 3
    assert abs(ec.impl_dyna_alpha_m - 0.15) < 1e-14
    assert abs(ec.impl_dyna_alpha_f - 0.25) < 1e-14
    assert abs(ec.impl_dyna_gamma - (0.5 - 0.15 + 0.25)) < 1e-14
    assert abs(ec.impl_dyna_beta - (0.25 * (1.0 - 0.15 + 0.25) ** 2)) < 1e-14

    assert ec.impl_ikt == 2
    assert ec.impl_dt_fixp == [0.05, 0.12, 0.35, 1.2]
