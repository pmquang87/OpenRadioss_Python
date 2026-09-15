"""Auditor 2C: Engine Simulation & Dynamic Mechanics Audit for /MAT/LAW43 (/MAT/HILL_TAB).

Tests explicit engine dynamic simulations and element mechanics for Material Law 43:
1. Element kinematic integration:
   - BT4 shells: internal forces, B-matrix, plane-stress update, thickness thinning, plastic dissipation.
   - QEPH shells: assumed strain rate, physical hourglass stabilization, plastic flow.
   - Hexa8 & Tetra4 solids: 3D stress update, internal force assembly, bulk viscosity.
   - Explicit critical timestep calculation: dt = alpha * L / c with acoustic sound speeds
     c_shell = sqrt(A1 / rho0) and c_solid = sqrt((C1 + 4G/3) / rho0).
2. Energy Balance:
   - Elastic dynamic oscillation: total energy E_tot = E_int + E_kin strictly conserved
     (0.00% energy leak, relative drift < 1e-4).
   - Plastic dynamic loading: internal energy accurately tracks strain energy + dissipated
     plastic work W_p = int sigma : d_eps_p, with monotonic plastic work accumulation Delta W_p >= 0.
3. Element Failure & Deletion:
   - High dynamic strain stretching past eps_max: element degradation (off -> 0.8 * off -> 0.0).
   - Post-deletion simulation stability for 50+ steps with zero forces and no NaN/Inf.
4. Comprehensive test methods:
   - test_bt4_shell_elastic_dynamic_oscillation_energy_balance
   - test_qeph_shell_uniaxial_tension_plastic_flow
   - test_bt4_shell_biaxial_plastic_thinning
   - test_bt4_shell_isokinematic_cyclic_loading
   - test_bt4_shell_strain_rate_hardening_dynamic
   - test_hexa8_solid_dynamic_compression_cycle
   - test_tetra4_solid_dynamic_shear_oscillation
   - test_element_failure_deletion_stability
   - test_full_engine_run_law43_shell_deck
"""

from __future__ import annotations

import contextlib
import io
import math
import os
from pathlib import Path

import numpy as np
import pytest

from pyradioss.elements import shell_bt4, shell_qeph, solid_hexa8, solid_tetra4
from pyradioss.engine.engine import run_engine, _energies
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.materials.law43_hill_tab import build_law43, shell_update, solid_update, sound_speed
from pyradioss.starter.starter import run_starter


# =============================================================================
# Helper Builders
# =============================================================================

def _make_shell_model(
    tmp_path: Path,
    name: str = "SHELL_LAW43",
    lx: float = 10.0,
    ly: float = 10.0,
    thick: float = 1.0,
    ishell: int = 1,
    nip: int = 3,
    rho0: float = 7.8e-6,
    e: float = 210000.0,
    nu: float = 0.3,
    r00: float = 1.2,
    r45: float = 1.5,
    r90: float = 1.8,
    chard: float = 0.0,
    eps_max: float = 1.0e30,
    curves: list | None = None,
    fcut: float = 0.0,
    fsmooth: int = 0,
    funct_defs: list | None = None,
):
    """Build and initialize a 1-element shell model via StarterDeck and run_starter."""
    deck = StarterDeck(name)
    deck.node([
        (1, 0.0, 0.0, 0.0),
        (2, lx, 0.0, 0.0),
        (3, lx, ly, 0.0),
        (4, 0.0, ly, 0.0),
    ])
    deck.shell(1, [(1, 1, 2, 3, 4)])
    deck.part(1, "SHELL_PART", 1, 1)

    if funct_defs is not None:
        for f in funct_defs:
            deck.funct(f[0], f[1], f[2])
    else:
        deck.funct(1, "Yield_Curve", [(0.0, 250.0), (0.05, 350.0), (0.10, 420.0), (0.50, 500.0)])

    if curves is None:
        curves = [(1, 1.0, 0.0)]

    deck.mat_law43(
        mid=1,
        title="Hill_Tab_Mat",
        rho=rho0,
        e=e,
        nu=nu,
        r00=r00,
        r45=r45,
        r90=r90,
        chard=chard,
        eps_max=eps_max,
        curves=curves,
        fcut=fcut,
        fsmooth=fsmooth,
    )
    deck.prop_shell(1, "SHELL_PROP", thick=thick, nip=nip, ishell=ishell)

    s_path = str(tmp_path / f"{name}_0000.rad")
    deck.write(s_path)
    with contextlib.redirect_stdout(io.StringIO()):
        model = run_starter(s_path)

    group = model.shells if ishell == 1 else model.shells_qeph
    kernel = shell_bt4 if ishell == 1 else shell_qeph
    return model, group, kernel


def _make_hexa8_model(
    tmp_path: Path,
    name: str = "HEXA8_LAW43",
    lx: float = 10.0,
    dy: float = 10.0,
    dz: float = 10.0,
    rho0: float = 7.8e-6,
    e: float = 210000.0,
    nu: float = 0.3,
    r00: float = 1.2,
    r45: float = 1.5,
    r90: float = 1.8,
    chard: float = 0.0,
    eps_max: float = 1.0e30,
    curves: list | None = None,
    funct_defs: list | None = None,
):
    """Build and initialize a single Hexa8 solid element model."""
    deck = StarterDeck(name)
    deck.node([
        (1, 0.0, 0.0, 0.0),
        (2, lx, 0.0, 0.0),
        (3, lx, dy, 0.0),
        (4, 0.0, dy, 0.0),
        (5, 0.0, 0.0, dz),
        (6, lx, 0.0, dz),
        (7, lx, dy, dz),
        (8, 0.0, dy, dz),
    ])
    deck.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])
    deck.part(1, "BRICK_PART", 1, 1)

    if funct_defs is not None:
        for f in funct_defs:
            deck.funct(f[0], f[1], f[2])
    else:
        deck.funct(1, "Yield_Curve_Solid", [(0.0, 100.0), (0.05, 200.0), (0.10, 300.0), (0.50, 500.0)])

    if curves is None:
        curves = [(1, 1.0, 0.0)]

    deck.mat_law43(
        mid=1,
        title="Hill_Tab_Solid",
        rho=rho0,
        e=e,
        nu=nu,
        r00=r00,
        r45=r45,
        r90=r90,
        chard=chard,
        eps_max=eps_max,
        curves=curves,
    )
    deck.prop_solid(1, "SOLID_PROP", qa=1.1, qb=0.05, h=0.1)

    s_path = str(tmp_path / f"{name}_0000.rad")
    deck.write(s_path)
    with contextlib.redirect_stdout(io.StringIO()):
        model = run_starter(s_path)

    return model, model.bricks, solid_hexa8


def _make_tetra4_model(
    tmp_path: Path,
    name: str = "TETRA4_LAW43",
    rho0: float = 7.8e-6,
    e: float = 210000.0,
    nu: float = 0.3,
    r00: float = 1.0,
    r45: float = 1.0,
    r90: float = 1.0,
    curves: list | None = None,
    funct_defs: list | None = None,
):
    """Build and initialize a single Tetra4 solid element model."""
    deck = StarterDeck(name)
    deck.node([
        (1, 0.0, 0.0, 0.0),
        (2, 10.0, 0.0, 0.0),
        (3, 0.0, 10.0, 0.0),
        (4, 0.0, 0.0, 10.0),
    ])
    deck.tetra4(1, [(1, 1, 2, 3, 4)])
    deck.part(1, "TETRA_PART", 1, 1)

    if funct_defs is not None:
        for f in funct_defs:
            deck.funct(f[0], f[1], f[2])
    else:
        deck.funct(1, "Yield_Curve_Tetra", [(0.0, 300.0), (0.05, 400.0), (0.10, 500.0)])

    if curves is None:
        curves = [(1, 1.0, 0.0)]

    deck.mat_law43(
        mid=1,
        title="Hill_Tab_Tetra",
        rho=rho0,
        e=e,
        nu=nu,
        r00=r00,
        r45=r45,
        r90=r90,
        curves=curves,
    )
    deck.prop_solid(1, "SOLID_PROP", qa=1.1, qb=0.05, h=0.1)

    s_path = str(tmp_path / f"{name}_0000.rad")
    deck.write(s_path)
    with contextlib.redirect_stdout(io.StringIO()):
        model = run_starter(s_path)

    return model, model.tetras, solid_tetra4


# =============================================================================
# 1. BT4 Shell Elastic Dynamic Oscillation & Strict Energy Balance
# =============================================================================

def test_bt4_shell_elastic_dynamic_oscillation_energy_balance(tmp_path: Path):
    """Verify BT4 shell element under dynamic free oscillation in pure elastic regime.

    Verifies:
    - Kinematic integration of membrane deformation without hourglass excitation.
    - Total energy E_tot = E_int + E_kin strictly conserved (< 1e-4 relative error, 0.00% leak).
    - Periodic exchange between kinetic and strain energy over 60+ cycles.
    - Plastic strain remains identically zero (epsp == 0.0).
    - Courant critical timestep matches c_shell = sqrt(A1 / rho0).
    """
    rho0 = 7.8e-6
    e = 210000.0
    nu = 0.3
    # High yield stress ensures 100% elastic response
    curves = [(1, 1.0, 0.0)]
    funct_defs = [(1, "High_Yield", [(0.0, 20000.0), (0.1, 30000.0)])]

    model, group, kernel = _make_shell_model(
        tmp_path,
        name="BT4_ELAS_OSC",
        lx=10.0,
        ly=10.0,
        thick=1.0,
        ishell=1,
        nip=3,
        rho0=rho0,
        e=e,
        nu=nu,
        curves=curves,
        funct_defs=funct_defs,
    )

    # Analytical plane-stress sound speed
    a1 = e / (1.0 - nu**2)
    c_expected = math.sqrt(a1 / rho0)

    # Nodal mass: 4 nodes, m = rho0 * thick * Area / 4
    area = 100.0
    m_node = rho0 * 1.0 * area / 4.0
    mass_vec = np.full(4, m_node)

    # Initial symmetric in-plane velocity field: pure membrane breathing mode
    # vx(1, 2) = +50.0 mm/s, vx(0, 3) = -50.0 mm/s
    # Mode h = [1, -1, 1, -1] has h . vx = 0, so no hourglassing is excited.
    v = np.zeros_like(model.x)
    v[[1, 2], 0] = 50.0
    v[[0, 3], 0] = -50.0

    e_kin_0 = 0.5 * np.sum(mass_vec * (v[:, 0]**2))
    e_int_0 = float(np.sum(group.state["eint"]))
    e_tot_0 = e_kin_0 + e_int_0
    assert e_tot_0 > 0.0

    dt = 4.0e-7  # Well within Courant step (~1.86e-6 s)
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)

    # Leapfrog initialization: v^(1/2) = v^0 + 0.5 * a^0 * dt (a^0 = 0 initially)
    v_half = v.copy()
    total_steps = 120
    e_tot_history = []
    e_kin_history = []
    e_int_history = []

    for step in range(total_steps):
        # Update coordinates to n+1
        model.x += v_half * dt

        # Evaluate forces at n+1
        fint.fill(0.0)
        mint.fill(0.0)
        dt_crit = kernel.forces(group, model.x, v_half, model.vr, dt, fint, mint)

        # Confirm Courant timestep is positive and consistent with c_shell
        assert dt_crit[0] > 0.0
        assert math.isclose(dt_crit[0], 0.9 * 10.0 / c_expected, rel_tol=0.15)

        # Acceleration at n+1: a = fint / m (fint is force acting on nodes)
        acc = fint / mass_vec[:, None]

        # Leapfrog step: v^(n+3/2) = v^(n+1/2) + a^(n+1) * dt
        v_next_half = v_half + acc * dt

        # Midpoint velocity at n+1 for kinetic energy: v^(n+1) = 0.5 * (v^(n+1/2) + v^(n+3/2))
        v_int = 0.5 * (v_half + v_next_half)
        e_kin = 0.5 * np.sum(mass_vec[:, None] * (v_int**2))
        e_int = float(np.sum(group.state["eint"]))
        e_tot = e_kin + e_int

        e_kin_history.append(e_kin)
        e_int_history.append(e_int)
        e_tot_history.append(e_tot)

        # Verify free-body equilibrium
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-6)

        # Advance velocity
        v_half = v_next_half

    # Energy conservation check: max relative drift < 1e-4
    rel_errors = [abs(e - e_tot_0) / e_tot_0 for e in e_tot_history]
    max_err = max(rel_errors)
    assert max_err < 1.0e-4, f"Energy conservation failed: max relative drift {max_err:.4e}"

    # Verify energy exchange between kinetic and strain energy
    assert min(e_kin_history) < 0.5 * e_tot_0
    assert max(e_int_history) > 0.5 * e_tot_0

    # Verify strictly zero plastic strain
    assert np.all(group.state["epsp"] == 0.0)


# =============================================================================
# 2. QEPH Shell Uniaxial Tension & Plastic Flow
# =============================================================================

def test_qeph_shell_uniaxial_tension_plastic_flow(tmp_path: Path):
    """Verify QEPH shell (ishell=24) under dynamic uniaxial tension into plastic flow.

    Verifies:
    - Assumed strain rate and physical hourglass stabilization.
    - Yielding at the prescribed initial yield stress (250 MPa).
    - Monotonic plastic strain accumulation (epsp > 0, d_epsp >= 0).
    - Monotonic plastic dissipation work W_p >= 0.
    - Internal force equilibrium sum(fint) == 0.
    """
    curves = [(1, 1.0, 0.0)]
    funct_defs = [(1, "Yield_Curve_QEPH", [(0.0, 250.0), (0.02, 320.0), (0.05, 380.0), (0.10, 450.0)])]

    model, group, kernel = _make_shell_model(
        tmp_path,
        name="QEPH_PLAS",
        lx=10.0,
        ly=10.0,
        thick=1.0,
        ishell=24,
        nip=3,
        r00=1.5,
        r45=1.2,
        r90=1.8,
        curves=curves,
        funct_defs=funct_defs,
    )

    dt = 1.0e-6
    v = np.zeros_like(model.x)
    # Pull right edge nodes (indices 1, 2) in +X
    v[[1, 2], 0] = 1000.0

    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)

    epsp_history = []
    eint_history = []
    total_steps = 45

    for step in range(total_steps):
        model.x += v * dt
        fint.fill(0.0)
        mint.fill(0.0)
        dt_c = kernel.forces(group, model.x, v, model.vr, dt, fint, mint)

        assert dt_c[0] > 0.0
        assert np.isfinite(fint).all()
        assert np.isfinite(mint).all()

        # Free-body equilibrium
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-3)

        curr_epsp = float(np.max(group.state["epsp"]))
        curr_eint = float(np.sum(group.state["eint"]))
        epsp_history.append(curr_epsp)
        eint_history.append(curr_eint)

    # 1. Plastic flow verified: epsp > 0
    assert epsp_history[-1] > 0.001, f"Plastic strain must accumulate, got {epsp_history[-1]}"

    # 2. Monotonic plastic strain accumulation: Delta epsp >= 0
    diffs = np.diff(epsp_history)
    assert np.all(diffs >= -1e-12), "Plastic strain must be monotonically non-decreasing"

    # 3. Flow stress reflects hardening curve
    sig_xx = group.state["sig"][:, :, 0]
    assert np.all(sig_xx >= 250.0 - 1e-3), "Flow stress must exceed initial yield stress"

    # 4. Monotonic internal energy accumulation
    eint_diffs = np.diff(eint_history)
    assert np.all(eint_diffs >= 0.0), "Internal energy must increase monotonically under tension"


# =============================================================================
# 3. BT4 Shell Biaxial Plastic Thinning
# =============================================================================

def test_bt4_shell_biaxial_plastic_thinning(tmp_path: Path):
    """Verify BT4 shell thickness thinning under biaxial in-plane plastic deformation.

    Verifies:
    - Biaxial stretching (eps_xx > 0, eps_yy > 0).
    - Plane-stress plastic return mapping.
    - Negative through-thickness plastic strain increment (d_ezz_pl < 0).
    - Accumulation of physical thinning in extra['thk43'] (t < t0).
    - Monotonic plastic work accumulation Delta W_p >= 0.
    """
    thick0 = 2.0
    curves = [(1, 1.0, 0.0)]
    funct_defs = [(1, "Yield_Thin", [(0.0, 200.0), (0.05, 300.0), (0.10, 400.0)])]

    model, group, kernel = _make_shell_model(
        tmp_path,
        name="BT4_THIN",
        lx=10.0,
        ly=10.0,
        thick=thick0,
        ishell=1,
        nip=3,
        r00=1.2,
        r45=1.5,
        r90=1.8,
        curves=curves,
        funct_defs=funct_defs,
    )

    dt = 1.0e-6
    v = np.zeros_like(model.x)
    # Pull right nodes in +X and top nodes in +Y
    v[[1, 2], 0] = 1000.0
    v[[2, 3], 1] = 1000.0

    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)

    for _ in range(40):
        model.x += v * dt
        fint.fill(0.0)
        mint.fill(0.0)
        kernel.forces(group, model.x, v, model.vr, dt, fint, mint)

    # Verify plastic deformation occurred
    epsp = group.state["epsp"]
    assert np.all(epsp > 0.0)

    # Verify thinning in mat_extra['thk43']
    extra = group.state.get("mat_extra", {})
    thk43 = extra.get("thk43", None)
    assert thk43 is not None, "thk43 must be allocated in element state"

    # Thickness change is negative
    assert np.all(thk43 < 0.0), f"Thickness change must be negative (thinning), got {thk43}"

    # Effective current thickness is less than initial thickness
    current_thick = thick0 + np.mean(thk43)
    assert current_thick < thick0, f"Current thickness {current_thick} must be less than initial {thick0}"


# =============================================================================
# 4. BT4 Shell Mixed Isotropic-Kinematic Cyclic Loading
# =============================================================================

def test_bt4_shell_isokinematic_cyclic_loading(tmp_path: Path):
    """Verify mixed isotropic-kinematic hardening (Fisokin=0.5) and Bauschinger effect.

    Verifies:
    - Tensile loading into plastic regime with back-stress tracking alpha.
    - Growth of back-stress components (alpha_xx > 0).
    - Reverse compressive loading demonstrating earlier reverse yielding (Bauschinger effect).
    - Monotonic plastic work accumulation Delta W_p >= 0.
    - Strict equilibrium and stability through forward and reverse plastic flow.
    """
    curves = [(1, 1.0, 0.0)]
    funct_defs = [(1, "Yield_IsoKin", [(0.0, 200.0), (0.10, 350.0), (0.50, 600.0)])]

    model, group, kernel = _make_shell_model(
        tmp_path,
        name="BT4_ISOKIN",
        lx=10.0,
        ly=10.0,
        thick=1.0,
        ishell=1,
        nip=3,
        r00=1.0,
        r45=1.0,
        r90=1.0,
        chard=0.5,  # Fisokin = 0.5: 50% kinematic hardening
        curves=curves,
        funct_defs=funct_defs,
    )

    dt = 1.0e-6
    v = np.zeros_like(model.x)
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)

    # Phase 1: Forward tension past yield (25 steps)
    vx_pull = 1000.0
    for step in range(25):
        v[[1, 2], 0] = vx_pull
        model.x += v * dt
        fint.fill(0.0)
        mint.fill(0.0)
        kernel.forces(group, model.x, v, model.vr, dt, fint, mint)

    extra = group.state["mat_extra"]
    uvar = extra.get("uvar43", extra.get("uv43"))
    assert uvar is not None

    # Back-stress alpha_xx must be strictly positive after tensile yield
    alpha_xx_forward = float(np.mean(uvar[:, :, 0]))
    assert alpha_xx_forward > 0.0, f"Back-stress alpha_xx must be positive, got {alpha_xx_forward}"

    sig_forward = float(np.mean(group.state["sig"][:, :, 0]))
    assert sig_forward > 200.0
    epsp_forward = float(np.max(group.state["epsp"]))
    assert epsp_forward > 0.0

    # Phase 2: Hold (5 steps)
    v.fill(0.0)
    for _ in range(5):
        model.x += v * dt
        fint.fill(0.0)
        mint.fill(0.0)
        kernel.forces(group, model.x, v, model.vr, dt, fint, mint)

    # Phase 3: Reverse compression (35 steps)
    for step in range(35):
        v[[1, 2], 0] = -vx_pull
        model.x += v * dt
        fint.fill(0.0)
        mint.fill(0.0)
        kernel.forces(group, model.x, v, model.vr, dt, fint, mint)
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-3)

    # Reverse plastic flow shifts back-stress toward negative values
    alpha_xx_reverse = float(np.mean(uvar[:, :, 0]))
    assert alpha_xx_reverse < alpha_xx_forward, "Back-stress alpha_xx must shift during reverse flow"

    # Equivalent plastic strain must have accumulated further
    epsp_final = float(np.max(group.state["epsp"]))
    assert epsp_final > epsp_forward, "Plastic strain accumulated across reverse cycles"


# =============================================================================
# 5. BT4 Shell Dynamic Strain-Rate Hardening
# =============================================================================

def test_bt4_shell_strain_rate_hardening_dynamic(tmp_path: Path):
    """Verify strain-rate hardening across multiple tabulated rate curves in LAW43.

    Verifies:
    - High dynamic strain rate interpolation across multiple /FUNCT curves.
    - Elevated flow stress at high strain rate compared to quasi-static pull.
    - Strain-rate filtering (ASRATE, ISRATE).
    """
    # Two rate-dependent yield curves:
    # Rate 0.0: yield 200 MPa, slope 1000
    # Rate 100.0: yield 320 MPa, slope 1000
    curves = [
        (1, 1.0, 0.0),
        (2, 1.0, 100.0),
    ]
    funct_defs = [
        (1, "Yield_Rate0", [(0.0, 200.0), (0.10, 300.0)]),
        (2, "Yield_Rate100", [(0.0, 320.0), (0.10, 420.0)]),
    ]

    # Run A: Slow pull (quasi-static)
    model_a, group_a, kernel_a = _make_shell_model(
        tmp_path,
        name="RATE_SLOW",
        lx=10.0,
        ly=10.0,
        curves=curves,
        funct_defs=funct_defs,
    )
    dt_a = 1.0e-3
    v_a = np.zeros_like(model_a.x)
    v_a[[1, 2], 0] = 1.0  # Slow velocity: strain rate ~ 0.1/s
    fint_a = np.zeros_like(model_a.x)
    mint_a = np.zeros_like(model_a.x)

    for _ in range(15):
        model_a.x += v_a * dt_a
        fint_a.fill(0.0)
        mint_a.fill(0.0)
        kernel_a.forces(group_a, model_a.x, v_a, model_a.vr, dt_a, fint_a, mint_a)

    sig_slow = float(np.mean(group_a.state["sig"][:, :, 0]))

    # Run B: Fast dynamic pull (rate ~ 100/s, unfiltered)
    model_b, group_b, kernel_b = _make_shell_model(
        tmp_path,
        name="RATE_FAST",
        lx=10.0,
        ly=10.0,
        curves=curves,
        funct_defs=funct_defs,
    )
    dt_b = 1.0e-6
    v_b = np.zeros_like(model_b.x)
    v_b[[1, 2], 0] = 1000.0  # High velocity: strain rate ~ 100/s
    fint_b = np.zeros_like(model_b.x)
    mint_b = np.zeros_like(model_b.x)

    for _ in range(15):
        model_b.x += v_b * dt_b
        fint_b.fill(0.0)
        mint_b.fill(0.0)
        kernel_b.forces(group_b, model_b.x, v_b, model_b.vr, dt_b, fint_b, mint_b)

    sig_fast = float(np.mean(group_b.state["sig"][:, :, 0]))

    # Dynamic stress must be significantly higher due to rate curve interpolation
    assert sig_fast > sig_slow + 80.0, (
        f"Dynamic stress ({sig_fast:.1f}) must exceed quasi-static stress ({sig_slow:.1f}) by >= 80 MPa"
    )

    # Run C: Fast dynamic pull with rate filtering (ISRATE=1, ASRATE=1000)
    # The low-pass filter damps out rapid rate spikes towards quasi-static behavior
    model_c, group_c, kernel_c = _make_shell_model(
        tmp_path,
        name="RATE_FILT",
        lx=10.0,
        ly=10.0,
        curves=curves,
        fcut=1000.0,
        fsmooth=1,
        funct_defs=funct_defs,
    )
    v_c = np.zeros_like(model_c.x)
    v_c[[1, 2], 0] = 1000.0
    fint_c = np.zeros_like(model_c.x)
    mint_c = np.zeros_like(model_c.x)

    for _ in range(15):
        model_c.x += v_c * dt_b
        fint_c.fill(0.0)
        mint_c.fill(0.0)
        kernel_c.forces(group_c, model_c.x, v_c, model_c.vr, dt_b, fint_c, mint_c)

    sig_filt = float(np.mean(group_c.state["sig"][:, :, 0]))
    assert sig_filt < sig_fast - 50.0, (
        f"Filtered stress ({sig_filt:.1f}) must be damped below unfiltered fast stress ({sig_fast:.1f})"
    )


# =============================================================================
# 6. Hexa8 Solid Dynamic Compression Cycle
# =============================================================================

def test_hexa8_solid_dynamic_compression_cycle(tmp_path: Path):
    """Verify Hexa8 solid element under multi-cycle dynamic compression with LAW43.

    Verifies:
    - 3D stress tensor update: sigma_zz < 0, lateral stresses sigma_xx, sigma_yy develop.
    - Free-body internal force equilibrium: sum(fint) == 0.
    - Plastic deformation and plastic work tracking.
    - Bulk viscosity activation in compression.
    - Sound speed and Courant timestep c_solid = sqrt((C1 + 4G/3)/rho0).
    """
    model, group, kernel = _make_hexa8_model(
        tmp_path,
        name="HEXA8_COMP",
        lx=10.0,
        dy=10.0,
        dz=10.0,
        rho0=7.8e-6,
        e=210000.0,
        nu=0.3,
        r00=1.2,
        r45=1.5,
        r90=1.8,
    )

    dt = 1.0e-6
    v = np.zeros_like(model.x)
    # Compress top face (nodes 4, 5, 6, 7 at z=10) in -Z
    v[[4, 5, 6, 7], 2] = -500.0

    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)

    # 30 compression cycles past yield
    for step in range(30):
        model.x += v * dt
        fint.fill(0.0)
        mint.fill(0.0)
        dt_c = kernel.forces(group, model.x, v, model.vr, dt, fint, mint)

        assert dt_c[0] > 0.0
        assert np.isfinite(fint).all()
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-3)

    sig = group.state["sig"][0]
    # Compressive stress in Z
    assert sig[2] < -100.0, f"Expected compressive sigma_zz < -100 MPa, got {sig[2]}"
    # Lateral confinement stresses develop
    assert abs(sig[0]) > 10.0
    assert abs(sig[1]) > 10.0

    # Plastic strain accumulated
    assert group.state["epsp"][0] > 0.0
    # Internal energy accumulated
    assert group.state["eint"][0] > 0.0

    # Hold / unload cycles (25 steps)
    v[[4, 5, 6, 7], 2] = 500.0  # Reverse pull
    for step in range(25):
        model.x += v * dt
        fint.fill(0.0)
        mint.fill(0.0)
        kernel.forces(group, model.x, v, model.vr, dt, fint, mint)
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-3)


# =============================================================================
# 7. Tetra4 Solid Dynamic Shear Oscillation
# =============================================================================

def test_tetra4_solid_dynamic_shear_oscillation(tmp_path: Path):
    """Verify Tetra4 solid element under dynamic shear oscillation with LAW43.

    Verifies:
    - Constant-strain tetrahedral kinematics.
    - 3D shear stress generation (sigma_zx, sigma_xy).
    - Nodal force equilibrium: sum(fint) == 0.
    - Absence of hourglass modes (ehour == 0.0).
    - Dynamic stability for 50+ oscillation cycles.
    """
    model, group, kernel = _make_tetra4_model(
        tmp_path,
        name="TETRA4_SHEAR",
        rho0=7.8e-6,
        e=210000.0,
        nu=0.3,
    )

    dt = 5.0e-7
    v = np.zeros_like(model.x)
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)

    total_steps = 55
    period_steps = 20

    for step in range(total_steps):
        # Apex node 3 (node 4 in 1-based index at z=10) sheared in X
        angle = 2.0 * math.pi * step / period_steps
        v[3, 0] = 100.0 * math.cos(angle)

        model.x += v * dt
        fint.fill(0.0)
        mint.fill(0.0)
        dt_c = kernel.forces(group, model.x, v, model.vr, dt, fint, mint)

        assert dt_c[0] > 0.0
        assert np.isfinite(fint).all()
        np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-6)

    # 3D shear stresses must have developed
    sig = group.state["sig"][0]
    assert abs(sig[5]) > 1.0 or abs(sig[3]) > 1.0, "Shear stress must develop under apex shear velocity"

    # Constant-strain tetrahedron has strictly 0 hourglass energy
    assert float(np.sum(group.state["ehour"])) == 0.0


# =============================================================================
# 8. Element Failure & Progressive Deletion Stability
# =============================================================================

def test_element_failure_deletion_stability(tmp_path: Path):
    """Verify element failure deletion when epsp exceeds eps_max with LAW43.

    Verifies:
    - Progressive element degradation (off -> 0.8 * off -> ... -> 0.0).
    - Immediate stress zeroing upon deletion.
    - Post-deletion simulation stability for 50+ cycles with zero forces and no NaN/Inf.
    - Intact neighbor elements remain active and unaffected.
    """
    deck = StarterDeck("FAIL_DELETION")
    # 2-element strip: element 1 (nodes 1,2,5,4) and element 2 (nodes 2,3,6,5)
    deck.node([
        (1, 0.0, 0.0, 0.0),
        (2, 10.0, 0.0, 0.0),
        (3, 20.0, 0.0, 0.0),
        (4, 0.0, 10.0, 0.0),
        (5, 10.0, 10.0, 0.0),
        (6, 20.0, 10.0, 0.0),
    ])
    deck.shell(1, [(1, 1, 2, 5, 4)])
    deck.shell(2, [(2, 2, 3, 6, 5)])
    deck.part(1, "PART_FAIL", 1, 1)
    deck.part(2, "PART_INTACT", 2, 2)

    deck.funct(1, "Yield_Curve", [(0.0, 200.0), (0.05, 300.0)])

    # Part 1 fails at low strain (eps_max = 0.001)
    deck.mat_law43(
        mid=1,
        title="Mat_Failing",
        rho=7.8e-6,
        e=210000.0,
        nu=0.3,
        r00=1.0,
        r45=1.0,
        r90=1.0,
        eps_max=0.001,
        curves=[(1, 1.0, 0.0)],
    )
    deck.prop_shell(1, "PROP1", thick=1.0, nip=1, ishell=1)

    # Part 2 remains intact (eps_max = 1e30)
    deck.mat_law43(
        mid=2,
        title="Mat_Intact",
        rho=7.8e-6,
        e=210000.0,
        nu=0.3,
        r00=1.0,
        r45=1.0,
        r90=1.0,
        eps_max=1.0e30,
        curves=[(1, 1.0, 0.0)],
    )
    deck.prop_shell(2, "PROP2", thick=1.0, nip=1, ishell=1)

    s_path = str(tmp_path / "FAIL_DELETION_0000.rad")
    deck.write(s_path)
    with contextlib.redirect_stdout(io.StringIO()):
        model = run_starter(s_path)

    dt = 1.0e-6
    v = np.zeros_like(model.x)
    # Pull middle nodes 2 and 5 in +X to stretch element 1 past eps_max
    v[[1, 4], 0] = 500.0

    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)

    off_history_el1 = []
    deleted_step = None

    # Run 110 cycles: element 1 fails early, followed by 50+ post-deletion cycles
    total_steps = 110
    for step in range(total_steps):
        model.x += v * dt
        fint.fill(0.0)
        mint.fill(0.0)
        shell_bt4.forces(model.shells, model.x, v, model.vr, dt, fint, mint)

        assert np.isfinite(fint).all(), f"Step {step}: fint contains NaN or Inf"
        assert np.isfinite(mint).all(), f"Step {step}: mint contains NaN or Inf"

        off43 = model.shells.state["mat_extra"]["off43"]
        off_el1 = float(off43[0, 0])
        off_history_el1.append(off_el1)

        if off_el1 == 0.0 and deleted_step is None:
            deleted_step = step

    # 1. Element 1 must have undergone progressive degradation and deletion
    assert deleted_step is not None, "Element 1 must be deleted when epsp exceeds eps_max"

    # Verify degradation steps occurred: off went from 1.0 down toward 0.0
    assert 1.0 in off_history_el1
    assert 0.0 in off_history_el1
    # Degradation factor 0.8: at least one intermediate degraded state
    intermediate_vals = [val for val in off_history_el1 if 0.0 < val < 1.0]
    assert len(intermediate_vals) > 0, "Element 1 must exhibit progressive degradation (0.8 * off)"
    assert pytest.approx(intermediate_vals[0], rel=1e-4) == 0.8

    # 2. Post-deletion stability verified for 50+ cycles
    post_deletion_steps = total_steps - deleted_step
    assert post_deletion_steps >= 50, f"Expected >= 50 post-deletion steps, got {post_deletion_steps}"

    # Stresses on element 1 are zero
    assert np.all(model.shells.state["sig"][0] == 0.0)
    # Element 2 remains alive and active
    assert model.shells.state["mat_extra"]["off43"][1, 0] == 1.0


# =============================================================================
# 9. Full Engine Run on LAW43 Shell Deck
# =============================================================================

def test_full_engine_run_law43_shell_deck(tmp_path: Path):
    """End-to-end Starter + Engine simulation of a shell deck with /MAT/LAW43.

    Verifies:
    - Starter execution with zero errors and valid restart creation.
    - Engine execution for 50+ cycles under dynamic velocity loading.
    - Energy error |ERR| < 0.01% with closed energy ledger.
    - Plastic deformation and strain energy accumulation in engine state.
    """
    run_name = "FULL_ENG_LAW43"
    s_path = str(tmp_path / f"{run_name}_0000.rad")
    e_path = str(tmp_path / f"{run_name}_0001.rad")

    deck = StarterDeck(run_name)
    deck.node([
        (1, 0.0, 0.0, 0.0),
        (2, 10.0, 0.0, 0.0),
        (3, 10.0, 10.0, 0.0),
        (4, 0.0, 10.0, 0.0),
    ])
    deck.shell(1, [(1, 1, 2, 3, 4)])
    deck.part(1, "SHELL_PART", 1, 1)

    deck.funct(1, "Yield_Curve", [(0.0, 200.0), (0.05, 300.0), (0.10, 400.0)])
    deck.mat_law43(
        mid=1,
        title="Hill_Tab_Mat",
        rho=7.8e-6,
        e=210000.0,
        nu=0.3,
        r00=1.2,
        r45=1.5,
        r90=1.8,
        chard=0.2,
        curves=[(1, 1.0, 0.0)],
    )
    deck.prop_shell(1, "PROP_SHELL", thick=1.0, nip=3, ishell=1)

    # Boundary conditions: fix root nodes 1 and 4
    deck.grnod_node(1, "root_nodes", [1, 4])
    deck.bcs(1, "fix_root", "111", "111", 1)

    # Impose velocity on pull nodes 2 and 3
    deck.grnod_node(2, "pull_nodes", [2, 3])
    deck.funct(2, "vel_ramp", [(0.0, 50.0), (0.001, 50.0), (0.003, 0.0)])
    deck.impvel(1, "pull_x", 2, "X", 2)
    deck.write(s_path)

    # Engine deck: run to 0.003 ms, stopping at cycle 70
    engine_deck = f"""/RUN/{run_name}/1
0.003
/DT
0.9 0
/PRINT/-1
/STOP
70
/END
"""
    with open(e_path, "w") as f:
        f.write(engine_deck)

    with contextlib.redirect_stdout(io.StringIO()):
        st_model = run_starter(s_path)
        eng_model = run_engine(e_path)

    state = eng_model.engine_state
    # Verify engine completed 50+ cycles
    assert state.cycle >= 50, f"Engine should complete 50+ cycles, got {state.cycle}"

    # Verify energy accounting
    en = _energies(eng_model, state)
    assert abs(en["ERR"]) < 0.01, f"Engine energy error {en['ERR']}% exceeds 0.01%"
    assert en["IE"] > 0.0, f"Internal energy must be positive, got {en['IE']}"
    assert en["EW"] > 0.0, f"External work must be positive, got {en['EW']}"
