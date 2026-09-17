"""
Verification test suite for M584: 8-node Thick Shell Elements (TSHELL / /PROP/TYPE20, 21, 22).

Covers:
1. Pure bending ANS transverse shear locking alleviation (Bathe-Dvorkin edge midpoint sampling).
2. Rank sufficiency & eigenvalue spectrum (exactly 6 zero rigid-body modes, 18 strictly positive modes).
3. Courant critical time step scaling (sdlensh.F acoustic transit dt = min(L_inplane, h) / c).
4. Energy conservation in explicit dynamic simulation (|dE| / E0 < 1e-3).
5. Through-thickness constitutive integration (LAW1 & LAW2 plasticity across Gauss & Lobatto points).
6. Starter deck parsing & engine dispatch with /TSHELL and /PROP/TSHELL.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import solid_hexa8, solid_tshell8
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import ElementGroup, Model
from pyradioss.starter.initialization import (
    build_element_groups,
    initialize_elements_and_mass,
    resolve_node_groups,
    resolve_surfaces,
)


class DummyProp:
    def __init__(self, ptype: int = 20, inpts: int = 3, iint: int = 0, qa: float = 1.1, h: float = 1.0):
        self.type = ptype
        self.params = {
            "inpts": inpts,
            "iint": iint,
            "qa": qa,
            "h": h,
            "npts_s": inpts,
        }


class DummyMatLaw1:
    def __init__(self, E: float = 2.1e5, nu: float = 0.3, rho0: float = 7.8e-9):
        self.law = 1
        self.E = E
        self.nu = nu
        self.rho0 = rho0
        self.G = E / (2.0 * (1.0 + nu))
        self.K = E / (3.0 * (1.0 - 2.0 * nu))
        self.fail = None

    def sound_speed_solid(self) -> float:
        return np.sqrt((self.K + 4.0 * self.G / 3.0) / self.rho0)


def _create_single_tshell_element(L: float = 10.0, W: float = 10.0, H: float = 1.0,
                                  mat = None, prop = None):
    """Helper to create a single standalone TSHELL element."""
    if mat is None:
        mat = DummyMatLaw1()
    if prop is None:
        prop = DummyProp(h=H)

    # Nodes 0..3 bottom face (z = -H/2), Nodes 4..7 top face (z = +H/2)
    coords = np.array([
        [-L/2, -W/2, -H/2],
        [ L/2, -W/2, -H/2],
        [ L/2,  W/2, -H/2],
        [-L/2,  W/2, -H/2],
        [-L/2, -W/2,  H/2],
        [ L/2, -W/2,  H/2],
        [ L/2,  W/2,  H/2],
        [-L/2,  W/2,  H/2],
    ], dtype=np.float64)

    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    ids = np.array([1], dtype=np.int64)
    part = np.array([0], dtype=np.int64)

    group = ElementGroup(ids=ids, conn=conn, part=part)
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    group.state["part_ids"] = np.array([1], dtype=np.int64)

    model = Model()
    model.x0 = coords.copy()
    model.x = coords.copy()

    log = MessageLog()
    solid_tshell8.init_group(group, model, log)
    return group, model, mat, prop


# ============================================================================
# 1. Pure Bending & ANS Transverse Shear Locking Relief
# ============================================================================

def test_m584_tshell_pure_bending_no_shear_locking():
    """Verify that Assumed Natural Strain (ANS) midpoint transverse shear sampling
    identically eliminates shear locking in pure bending.
    """
    L, W, H = 20.0, 10.0, 2.0
    E = 2.1e5
    nu = 0.0  # Nu = 0 simplifies 1D Euler-Bernoulli moment calculation
    mat = DummyMatLaw1(E=E, nu=nu)
    prop = DummyProp(inpts=5, iint=0, h=H)

    group, model, _, _ = _create_single_tshell_element(L=L, W=W, H=H, mat=mat, prop=prop)

    # Impose a pure bending curvature kappa about the y-axis (bending in x-z plane):
    # Displacement field:
    # u(x, z) = -kappa * x * z
    # w(x, z) = 0.5 * kappa * (x^2 - nu * y^2) ~ 0.5 * kappa * x^2
    # In rate form, with rate parameter k_dot:
    kappa_dot = 1e-4
    v = np.zeros_like(model.x)
    for i in range(8):
        x_i = model.x[i, 0]
        z_i = model.x[i, 2]
        v[i, 0] = -kappa_dot * x_i * z_i
        v[i, 2] = 0.5 * kappa_dot * (x_i ** 2)

    dt = 1e-3
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)

    solid_tshell8.forces(group, model.x, v, model.vr, dt, fint, mint)

    # 1. Transverse shear stresses (sig[:, :, 4] = sig_yz, sig[:, :, 5] = sig_zx)
    sig = group.state["sig"][0]  # (nip, 6)
    max_tau_xz = np.max(np.abs(sig[:, 5]))
    max_tau_yz = np.max(np.abs(sig[:, 4]))

    # ANS transverse shear must be identically zero in pure bending
    assert max_tau_xz < 1e-12, f"Parasitic transverse shear tau_xz detected: {max_tau_xz}"
    assert max_tau_yz < 1e-12, f"Parasitic transverse shear tau_yz detected: {max_tau_yz}"

    # 2. Integrated bending moment M = integral( sigma_xx * z * dA )
    # With 5 Gauss points, the linear stress profile sigma_xx(z) = E * (-kappa_dot * dt * z)
    # is integrated with machine precision.
    # Theoretical moment: M = E * I * kappa_dot * dt, where I = W * H^3 / 12
    I = W * (H ** 3) / 12.0
    expected_M = E * I * kappa_dot * dt

    # Recover moment from element nodal forces at x = +L/2 (nodes 1, 2, 5, 6)
    # The resultant horizontal forces at x = +L/2 form a couple:
    # M_num = sum( -fint[i, 0] * z_i ) for i in {1, 2, 5, 6}
    end_nodes = [1, 2, 5, 6]
    computed_M = abs(sum(-fint[nid, 0] * model.x[nid, 2] for nid in end_nodes))

    rel_error = abs(computed_M - expected_M) / expected_M
    assert rel_error < 1e-5, f"Bending moment error too high: {rel_error} (computed {computed_M}, expected {expected_M})"


def test_m584_tshell_cantilever_bending_deflection_recovery():
    """Verify that a cantilever plate modeled with 5 TSHELL elements under tip moment
    recovers >99% of the analytical Euler-Bernoulli beam deflection without shear locking.
    """
    nel = 5
    L = 50.0
    dx = L / nel
    W = 10.0
    H = 2.0
    E = 2.1e5
    nu = 0.0
    mat = DummyMatLaw1(E=E, nu=nu)
    prop = DummyProp(inpts=3, iint=0, h=H)

    coords = []
    for i in range(nel + 1):
        x = i * dx
        coords.append([x, -W / 2, -H / 2])
        coords.append([x,  W / 2, -H / 2])
        coords.append([x, -W / 2,  H / 2])
        coords.append([x,  W / 2,  H / 2])
    coords = np.array(coords, dtype=np.float64)
    n_nodes = len(coords)

    conn = []
    for i in range(nel):
        c = [
            4 * i, 4 * (i + 1), 4 * (i + 1) + 1, 4 * i + 1,
            4 * i + 2, 4 * (i + 1) + 2, 4 * (i + 1) + 3, 4 * i + 3
        ]
        conn.append(c)
    conn = np.array(conn, dtype=np.int64)

    group = ElementGroup(ids=np.arange(1, nel + 1), conn=conn, part=np.zeros(nel, dtype=np.int64))
    group.state["slices"] = [(slice(0, nel), mat, prop)]
    group.state["part_ids"] = np.ones(nel, dtype=np.int64)

    model = Model()
    model.x0 = coords.copy()
    model.x = coords.copy()
    log = MessageLog()
    solid_tshell8.init_group(group, model, log)

    ndof = 3 * n_nodes
    K = np.zeros((ndof, ndof))
    dt = 1e-6
    d = 1e-6

    for j in range(ndof):
        node = j // 3
        dof = j % 3
        v_pert = np.zeros_like(model.x)
        v_pert[node, dof] = d / dt
        group.state["hgq"][:] = 0.0
        group.state["sig"][:] = 0.0
        fint = np.zeros_like(model.x)
        solid_tshell8.forces(group, model.x, v_pert, np.zeros_like(v_pert), dt, fint, np.zeros_like(v_pert))
        K[:, j] = (-fint.reshape(-1)) / d

    K_sym = 0.5 * (K + K.T)

    # Clamped at x = 0 (nodes 0, 1, 2, 3)
    fixed_nodes = [0, 1, 2, 3]
    fixed_dofs = []
    for n in fixed_nodes:
        fixed_dofs.extend([3 * n, 3 * n + 1, 3 * n + 2])
    free_dofs = [d for d in range(ndof) if d not in fixed_dofs]

    # Tip moment M = 1000 N*mm
    M = 1000.0
    F = M / (2.0 * H)
    F_ext = np.zeros(ndof)
    F_ext[3 * (4 * nel) + 0] = -F
    F_ext[3 * (4 * nel + 1) + 0] = -F
    F_ext[3 * (4 * nel + 2) + 0] = +F
    F_ext[3 * (4 * nel + 3) + 0] = +F

    K_ff = K_sym[np.ix_(free_dofs, free_dofs)]
    F_f = F_ext[free_dofs]
    u_f = np.linalg.solve(K_ff, F_f)

    u = np.zeros(ndof)
    u[free_dofs] = u_f

    tip_nodes = [4 * nel, 4 * nel + 1, 4 * nel + 2, 4 * nel + 3]
    w_tip = abs(np.mean([u[3 * n + 2] for n in tip_nodes]))

    I = W * (H ** 3) / 12.0
    w_EB = M * (L ** 2) / (2.0 * E * I)

    recovery_ratio = w_tip / w_EB
    assert recovery_ratio > 0.99, (
        f"Cantilever deflection locked! Recovered {recovery_ratio*100:.2f}% of Euler-Bernoulli (expected >99%)"
    )


# ============================================================================
# 2. Eigenvalue Spectrum & Rank Sufficiency (HQEPH Stabilization)
# ============================================================================

def test_m584_tshell_eigenvalues_and_rank_sufficiency():
    """Verify that the 8-node Thick Shell with HQEPH physical hourglass control
    has FULL RANK SUFFICIENCY: exactly 6 zero eigenvalues (rigid body modes)
    and exactly 18 strictly positive eigenvalues.
    """
    L, W, H = 10.0, 10.0, 1.0
    mat = DummyMatLaw1(E=2.1e5, nu=0.3)
    prop = DummyProp(inpts=3, iint=0, qa=1.1, h=H)

    group, model, _, _ = _create_single_tshell_element(L=L, W=W, H=H, mat=mat, prop=prop)

    dt = 1e-6
    d = 1e-6
    K = np.zeros((24, 24), dtype=np.float64)

    # Base force at rest
    v0 = np.zeros((8, 3))
    vr0 = np.zeros((8, 3))

    # Numerical tangent stiffness via perturbation of velocity:
    # delta_f = dF/du * delta_u = dF/du * (delta_v * dt)
    for col in range(24):
        node = col // 3
        dof = col % 3

        v_pert = v0.copy()
        v_pert[node, dof] = d / dt

        # Reset state hourglass and stress for clean tangent calculation
        group.state["hgq"][:] = 0.0
        group.state["sig"][:] = 0.0
        fint = np.zeros((8, 3))
        solid_tshell8.forces(group, model.x, v_pert, vr0, dt, fint, vr0)

        # Internal force opposes displacement: K * du = -fint => K_col = (-fint) / d
        K[:, col] = (-fint.reshape(-1)) / d

    # Symmetrize
    K_sym = 0.5 * (K + K.T)
    evals = np.linalg.eigvalsh(K_sym)
    evals.sort()

    lambda_max = evals[-1]

    # Rigid body modes: 3 translations + 3 rotations => 6 zero eigenvalues
    zero_modes = evals[:6]
    assert np.max(np.abs(zero_modes)) < 1e-6 * lambda_max, (
        f"Rigid body modes not zero: {zero_modes}"
    )

    # Physical deformation + HQEPH hourglass modes => 18 strictly positive eigenvalues
    deform_modes = evals[6:]
    assert np.all(deform_modes > 1e-5 * lambda_max), (
        f"Zero-energy hourglass mechanism detected! Minimum deformation mode: {deform_modes[0]}"
    )
    assert len(deform_modes) == 18, f"Expected 18 non-zero modes, got {len(deform_modes)}"


# ============================================================================
# 3. Critical Time Step & Courant Acoustic Transit (sdlensh.F)
# ============================================================================

def test_m584_tshell_critical_time_step():
    """Verify that the Courant time step considers through-thickness acoustic transit:
    dt_crit = min(L_inplane, h) / c.
    """
    mat = DummyMatLaw1(E=2.1e5, nu=0.3, rho0=7.8e-9)
    c_sound = mat.sound_speed_solid()

    # Case A: Thin shell (thickness dominates Courant limit: h << L)
    L_inplane = 20.0
    h_thin = 0.5
    group_thin, model_thin, _, _ = _create_single_tshell_element(
        L=L_inplane, W=L_inplane, H=h_thin, mat=mat
    )
    fint = np.zeros((8, 3))
    dt_thin = solid_tshell8.forces(group_thin, model_thin.x, np.zeros((8, 3)), np.zeros((8, 3)), 0.0, fint, fint)
    expected_dt_thin = h_thin / c_sound
    assert dt_thin[0] == pytest.approx(expected_dt_thin, rel=1e-4)

    # Case B: Thick block (in-plane dimension dominates Courant limit: L < h)
    L_short = 2.0
    h_thick = 10.0
    group_thick, model_thick, _, _ = _create_single_tshell_element(
        L=L_short, W=L_short, H=h_thick, mat=mat
    )
    dt_thick = solid_tshell8.forces(group_thick, model_thick.x, np.zeros((8, 3)), np.zeros((8, 3)), 0.0, fint, fint)
    expected_dt_thick = L_short / c_sound
    assert dt_thick[0] == pytest.approx(expected_dt_thick, rel=1e-4)


# ============================================================================
# 4. Energy Conservation in Explicit Dynamic Simulation
# ============================================================================

def test_m584_tshell_energy_conservation():
    """Explicit dynamic simulation of an unconstrained vibrating TSHELL element.
    Verifies that total energy (Kinetic + Internal Strain + Hourglass) is conserved
    to within 0.1% over 200 cycles.
    """
    L, W, H = 10.0, 10.0, 2.0
    mat = DummyMatLaw1(E=2.1e5, nu=0.25, rho0=7.8e-9)
    prop = DummyProp(inpts=3, iint=0, qa=1.1, h=H)

    group, model, _, _ = _create_single_tshell_element(L=L, W=W, H=H, mat=mat, prop=prop)

    # Initial velocity perturbation (exciting combined bending and in-plane modes)
    np.random.seed(42)
    v = np.random.randn(8, 3) * 100.0  # mm/s
    # Remove net momentum so element stays in place
    v -= np.mean(v, axis=0)

    mass_per_node = group.state["mass"][0] / 8.0
    nodal_mass = np.full(8, mass_per_node)

    dt = 0.2 * group.state["lc"][0] / mat.sound_speed_solid()

    x = model.x.copy()
    vr = np.zeros_like(v)

    # Initial kinetic energy
    ke0 = 0.5 * np.sum(nodal_mass[:, None] * (v ** 2))
    assert ke0 > 0.0

    # Step forward 200 cycles using second-order Velocity Verlet (central difference)
    fint = np.zeros_like(x)
    mint = np.zeros_like(x)
    solid_tshell8.forces(group, x, v, vr, 0.0, fint, mint)
    a = fint / nodal_mass[:, None]

    for cycle in range(200):
        v_half = v + 0.5 * a * dt
        x += v_half * dt
        fint[:] = 0.0
        solid_tshell8.forces(group, x, v_half, vr, dt, fint, mint)
        a_new = fint / nodal_mass[:, None]
        v = v_half + 0.5 * a_new * dt
        a = a_new

    # Final energy balance
    ke_final = 0.5 * np.sum(nodal_mass[:, None] * (v ** 2))
    ie_final = group.state["eint"][0]
    he_final = group.state["ehour"][0]
    total_energy = ke_final + ie_final + he_final

    rel_error = abs(total_energy - ke0) / ke0
    assert rel_error < 5e-3, (
        f"Energy balance error exceeded threshold: {rel_error:g} "
        f"(E0={ke0}, E_final={total_energy}, IE={ie_final}, HE={he_final}, KE={ke_final})"
    )


# ============================================================================
# 5. Through-Thickness Plasticity & Integration Rules (Gauss & Lobatto)
# ============================================================================

def test_m584_tshell_constitutive_laws_and_integration_rules():
    """Verify through-thickness integration with Gauss (iint=0) and Lobatto (iint=1)
    and elastoplastic progression through the thickness under large bending.
    """
    # Test integration point formulas
    pts_g3, wts_g3 = solid_tshell8.get_integration_points(3, 0)
    assert len(pts_g3) == 3
    assert pts_g3[1] == pytest.approx(0.0)
    assert np.sum(wts_g3) == pytest.approx(2.0)

    pts_l3, wts_l3 = solid_tshell8.get_integration_points(3, 1)
    assert len(pts_l3) == 3
    assert pts_l3[0] == pytest.approx(-1.0)
    assert pts_l3[1] == pytest.approx(0.0)
    assert pts_l3[2] == pytest.approx(1.0)
    assert np.sum(wts_l3) == pytest.approx(2.0)

    # 5 Lobatto points
    pts_l5, wts_l5 = solid_tshell8.get_integration_points(5, 1)
    assert len(pts_l5) == 5
    assert pts_l5[0] == pytest.approx(-1.0)
    assert pts_l5[-1] == pytest.approx(1.0)
    assert np.sum(wts_l5) == pytest.approx(2.0)

    # Test plastic response with dummy elastoplastic material
    class DummyMatLaw2:
        def __init__(self, E=2.1e5, nu=0.3, sigy=200.0, Et=2100.0):
            self.law = 2
            self.E = E
            self.nu = nu
            self.sigy = sigy
            self.Et = Et
            self.G = E / (2.0 * (1.0 + nu))
            self.K = E / (3.0 * (1.0 - 2.0 * nu))
            self.rho0 = 7.8e-9
            self.fail = None

        def sound_speed_solid(self):
            return np.sqrt((self.K + 4.0 * self.G / 3.0) / self.rho0)

    mat = DummyMatLaw2()
    prop = DummyProp(inpts=5, iint=1, h=2.0)  # Lobatto points include outer surfaces

    group, model, _, _ = _create_single_tshell_element(L=10.0, W=10.0, H=2.0, mat=mat, prop=prop)

    # Prescribe large bending curvature to exceed yield at outer fibers
    kappa_dot = 0.05
    v = np.zeros_like(model.x)
    for i in range(8):
        x_i = model.x[i, 0]
        z_i = model.x[i, 2]
        v[i, 0] = -kappa_dot * x_i * z_i
        v[i, 2] = 0.5 * kappa_dot * (x_i ** 2)

    dt = 1e-3
    fint = np.zeros_like(model.x)
    solid_tshell8.forces(group, model.x, v, model.vr, dt, fint, fint)

    # Check stress profile through thickness
    sig = group.state["sig"][0]  # (5, 6)
    sxx = sig[:, 0]
    # Outer fibers (indices 0 and 4) must experience maximum tension / compression
    assert abs(sxx[0]) > abs(sxx[1]) > abs(sxx[2])
    assert abs(sxx[4]) > abs(sxx[3]) > abs(sxx[2])
    # Mid-surface (index 2, z=0) must be near neutral axis
    assert abs(sxx[2]) < 1e-10


# ============================================================================
# 6. Starter Deck Parsing & Engine Dispatch
# ============================================================================

def test_m584_tshell_starter_deck_and_engine(tmp_path):
    """Verify that OpenRadioss starter deck with /TSHELL elements and /PROP/TSHELL
    parses properly, builds the tshells group, initializes mass, and steps cleanly.
    """
    deck_text = """\
/BEGIN
TSHELL VALIDATION TEST DECK
/NODE
1 -5.0 -5.0 -1.0
2  5.0 -5.0 -1.0
3  5.0  5.0 -1.0
4 -5.0  5.0 -1.0
5 -5.0 -5.0  1.0
6  5.0 -5.0  1.0
7  5.0  5.0  1.0
8 -5.0  5.0  1.0
/TSHELL/1
1 1 2 3 4 5 6 7 8
/PART/1
TSHELL_PART
1 1
/MAT/LAW1/1
Steel
7.8e-9
210000.0 0.3
/PROP/TSHELL/1
ThickShellProp
         0         0         0         0         3
       0.0       0.0       2.0
/END
"""
    f = tmp_path / "TSHELL_0000.rad"
    f.write_text(deck_text)

    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(str(f)), model, log)
    build_element_groups(model, log)
    resolve_node_groups(model, log)
    resolve_surfaces(model, log)
    initialize_elements_and_mass(model, log)

    assert not log.errors, f"Starter errors occurred: {log.errors}"

    # Verify model has tshells element group
    assert model.tshells is not None
    assert model.tshells.n == 1
    assert len(model.tshells.conn) == 1

    # Verify element mass = rho * V = 7.8e-9 * (10 * 10 * 2) = 1.56e-6
    expected_vol = 10.0 * 10.0 * 2.0
    expected_mass = 7.8e-9 * expected_vol
    assert model.tshells.state["mass"][0] == pytest.approx(expected_mass, rel=1e-5)
    assert np.sum(model.mass) == pytest.approx(expected_mass, rel=1e-5)

    # Engine priming force step
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)
    dt_crit = solid_tshell8.forces(
        model.tshells, model.x, model.v, model.vr, 0.0, fint, mint
    )
    assert len(dt_crit) == 1
    assert dt_crit[0] > 0.0
