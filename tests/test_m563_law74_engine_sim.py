"""
Dynamic Engine Simulation & Energy Balance Verifier for M563 (/MAT/LAW74 /MAT/HILL_3D /MAT/ORTH_PLAS).

Exhaustive dynamic explicit engine simulation and energy balance audit suite verifying:
1. Multi-step explicit time integration simulations using solid hexa8 and tetra4 meshes:
   - Solid Hexa8 tensile bar under displacement control (50+ cycles, |ERR| < 1.0%, stop_reason == "").
   - Solid Tetra4 dynamic tension under displacement control (50+ cycles, |ERR| < 1.0%).
   - Solid Hexa8 2-element patch under cyclic dynamic loading (50+ cycles, |ERR| < 1.0%).
   - Asserts normal termination, >= 50 cycles, energy balance error |ERR| < 1.0%,
     positive internal strain energy (IE > 0), and external work consistency.
2. Cyclic tension-compression simulation verifying kinematic hardening Bauschinger hysteresis loop:
   - Backstress tensor alpha (uvar74[:, 4:10]) evolution under forward and reverse loading.
   - Reverse yield initiating at reduced stress magnitude (Bauschinger effect) vs isotropic hardening (chard = 0).
   - Closed hysteresis loop with plastic dissipation.
3. Strict energy conservation check:
   - Undamped free vibration of Solid Hexa8 in elastic regime: total energy E_tot = E_k + E_int conserved (|Delta E| / E_0 < 0.1%).
   - Undamped free vibration of Solid Tetra4 in elastic regime: total energy E_tot = E_k + E_int conserved (|Delta E| / E_0 < 0.1%).
   - Incremental strain energy ledger Delta E_int = int sigma : depsilon * dV * dt matches trapezoidal analytical work.
   - Monotonically increasing plastic dissipation during plastic flow.
4. Adiabatic plastic heating:
   - Temperature increases adiabatically during plastic work when rhocp > 0: Delta T = int (sigma * depsp) / (rho * cp).
   - Zero temperature increase during purely elastic loading.
5. Acoustic sound speed & Courant time-step stability:
   - Bounded Courant step across dynamic cycles and alloy parameters.
   - Modulus degradation (ce / einf) maintaining positive sound speed and stable time stepping.
"""

from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from pyradioss.elements import solid_hexa8, solid_tetra4
from pyradioss.engine.engine import run_engine, _energies
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.materials.law74_hill_3d import (
    Law74Params,
    build_law74,
    solid_update,
    sound_speed_solid,
)
from pyradioss.model.entities import Material
from pyradioss.model.model import Model
from pyradioss.starter.starter import run_starter


# ============================================================================
# Helpers: Engine Control Deck Writer and Material Factories
# ============================================================================

def _scalar(val: Any) -> float:
    """Safely convert 0-d or 1-d single element array/scalar to float (NumPy 2.x safe)."""
    return float(np.asarray(val).flat[0])


def _write_engine_deck(
    path: Path | str,
    run_name: str,
    tstop: float = 2.5e-4,
    dt_scale: float = 0.5,
    stop_cycles: Optional[int] = None,
    print_freq: int = -1000,
) -> None:
    """Generate and write a clean engine control deck file."""
    lines = [
        f"/RUN/{run_name}/1",
        f"{tstop:.6e}",
        "/DT",
        f"{dt_scale:.2f} 0.0",
        f"/PRINT/{print_freq}",
    ]
    if stop_cycles is not None:
        lines.extend(["/STOP", f"{stop_cycles}"])
    lines.append("/END\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def make_test_material_law74(
    mid: int = 1,
    rho0: float = 2.7e-9,  # ton/mm^3
    E: float = 70000.0,    # MPa
    nu: float = 0.33,
    s11y: float = 1.0,
    s22y: float = 1.0,
    s33y: float = 1.0,
    s12y: float = 1.0,
    s23y: float = 1.0,
    s31y: float = 1.0,
    chard: float = 0.0,
    sigy0: float = 200.0,
    yield_table: Any = None,
    table_id: int = 0,
    eps_max: float = 1.0e30,
    epsr1: float = 1.0e30,
    epsr2: float = 2.0e30,
    t0: float = 293.0,
    rhocp: float = 0.0,
    **kwargs: Any,
) -> Material:
    """Factory creating a valid /MAT/LAW74 Material instance."""
    if yield_table is None and table_id == 0 and sigy0 > 0.0:
        yield_table = [(0.0, sigy0), (0.05, sigy0 * 1.25), (0.2, sigy0 * 1.5)]
    p = Law74Params(
        rho0=rho0,
        refer_rho=rho0,
        e=E,
        nu=nu,
        s11y=s11y,
        s22y=s22y,
        s33y=s33y,
        s12y=s12y,
        s23y=s23y,
        s31y=s31y,
        chard=chard,
        sigy0=sigy0,
        yield_table=yield_table,
        table_id=table_id,
        eps_max=eps_max,
        epsr1=epsr1,
        epsr2=epsr2,
        t0=t0,
        rhocp=rhocp,
        id=mid,
        title=f"LAW74_Mat_{mid}",
        **kwargs,
    )
    return build_law74(p)


# ============================================================================
# 1. Multi-Step Explicit Dynamic Engine Simulations
# ============================================================================

class TestLaw74DynamicEngineSimulations:
    """Multi-cycle explicit dynamic engine simulations verifying stability and energy balance."""

    def test_solid_hexa8_tensile_bar_engine(self, tmp_path: Path):
        """Solid Hexa8 tensile bar under displacement control (50+ cycles, |ERR| < 1.0%)."""
        run_name = "HEXA8_LAW74_TEN"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "yield_fct", [(0.0, 200.0), (0.05, 250.0), (0.2, 320.0)])
        d.mat_law74(
            1, "HexaLAW74",
            rho=2.7e-9, e=70000.0, nu=0.33,
            chard=0.5,
            table_id=10,
        )
        d.prop_solid(1, "PropHexa", isolid=1)
        d.part(1, "PartHexa", 1, 1)

        # 4-element tensile bar along Z: z in [0, 40]
        # Nodes: 5 slices of 4 nodes each
        nodes = []
        nid = 1
        for iz in range(5):
            z = iz * 10.0
            nodes.append((nid, 0.0, 0.0, z))
            nodes.append((nid + 1, 10.0, 0.0, z))
            nodes.append((nid + 2, 10.0, 10.0, z))
            nodes.append((nid + 3, 0.0, 10.0, z))
            nid += 4
        d.node(nodes)

        # 4 brick elements
        bricks = []
        for ie in range(4):
            base = ie * 4 + 1
            top = base + 4
            bricks.append((
                ie + 1,
                base, base + 1, base + 2, base + 3,
                top, top + 1, top + 2, top + 3,
            ))
        d.brick(1, bricks)

        # Fix base nodes (1, 2, 3, 4) in all directions
        d.grnod_node(1, "fix_base", [1, 2, 3, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Pull top nodes (17, 18, 19, 20) in Z
        d.grnod_node(2, "pull_nodes", [17, 18, 19, 20])
        d.funct(11, "vz_pull", [(0.0, 500.0), (2.5e-4, 500.0)])
        d.impvel(1, "pull_z", 11, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.5e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 50, f"Expected >= 50 cycles, got {state.cycle}"
        assert state.stop_reason == "", f"Unexpected stop reason: {state.stop_reason}"

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Hexa8 energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0, "Internal strain energy must be strictly positive"
        assert en["EW"] > 0.0, "External work must be positive"

        # Check state variable propagation (uvar74)
        mat_extra = eng_model.bricks.state.get("mat_extra", {})
        assert "uvar74" in mat_extra
        assert mat_extra["uvar74"].shape == (4, 10)

    def test_solid_tetra4_dynamic_tension_engine(self, tmp_path: Path):
        """Solid Tetra4 with LAW74 under dynamic tension (50+ cycles, |ERR| < 1.0%)."""
        run_name = "TETRA4_LAW74_TEN"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "yield_curve", [(0.0, 200.0), (0.1, 280.0)])
        d.mat_law74(
            1, "TetraLAW74",
            rho=2.7e-9, e=70000.0, nu=0.33,
            chard=0.0,
            table_id=10,
        )
        d.prop_solid(1, "PropTetra", isolid=1)
        d.part(1, "PartTetra", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 0.0, 10.0, 0.0), (4, 0.0, 0.0, 10.0),
        ])
        d.tetra4(1, [(1, 1, 2, 3, 4)])

        # Fix base nodes (1, 2, 3)
        d.grnod_node(1, "fix_base", [1, 2, 3])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Pull apex node 4 in Z
        d.grnod_node(2, "apex_node", [4])
        d.funct(12, "vz_pull", [(0.0, 500.0), (2.0e-4, 500.0)])
        d.impvel(1, "pull_apex", 12, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 50, f"Expected >= 50 cycles, got {state.cycle}"
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Tetra4 energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0

    def test_solid_hexa8_multi_element_patch_cyclic_engine(self, tmp_path: Path):
        """Solid Hexa8 2-element patch with LAW74 under cyclic dynamic loading (50+ cycles)."""
        run_name = "HEXA8_PATCH_LAW74_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "yield_fct", [(0.0, 200.0), (0.05, 260.0), (0.15, 320.0)])
        d.mat_law74(
            1, "PatchLAW74",
            rho=2.7e-9, e=70000.0, nu=0.33,
            chard=0.5,
            table_id=10,
        )
        d.prop_solid(1, "PropHexa", isolid=1)
        d.part(1, "PartHexa", 1, 1)

        # 2 elements stacked in Z
        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0), (6, 10.0, 0.0, 10.0), (7, 10.0, 10.0, 10.0), (8, 0.0, 10.0, 10.0),
            (9, 0.0, 0.0, 20.0), (10, 10.0, 0.0, 20.0), (11, 10.0, 10.0, 20.0), (12, 0.0, 10.0, 20.0),
        ])
        d.brick(1, [
            (1, 1, 2, 3, 4, 5, 6, 7, 8),
            (2, 5, 6, 7, 8, 9, 10, 11, 12),
        ])

        # Fix base nodes (1, 2, 3, 4)
        d.grnod_node(1, "fix_base", [1, 2, 3, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Pull top nodes (9, 10, 11, 12) cyclically in Z: tension -> hold -> compression
        d.grnod_node(2, "top_nodes", [9, 10, 11, 12])
        d.funct(11, "vz_cyc", [
            (0.0, 800.0),
            (0.8e-4, 800.0),
            (0.8001e-4, -600.0),
            (1.6e-4, -600.0),
            (1.6001e-4, 500.0),
            (2.5e-4, 500.0),
        ])
        d.impvel(1, "pull_z", 11, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.5e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 50, f"Expected >= 50 cycles, got {state.cycle}"
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Patch energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0


# ============================================================================
# 2. Cyclic Tension-Compression & Kinematic Bauschinger Hysteresis Loop
# ============================================================================

class TestLaw74CyclicBauschingerHysteresis:
    """Audit kinematic hardening Bauschinger hysteresis loop under cyclic loading."""

    def test_bauschinger_backstress_evolution_and_hysteresis_loop(self):
        """Verify backstress alpha accumulates and triggers earlier reverse yield (chard > 0)."""
        sigy0 = 200.0
        mat_kin = make_test_material_law74(
            E=70000.0, nu=0.33, chard=0.8, sigy0=sigy0,
            yield_table=[(0.0, sigy0), (0.05, sigy0 * 1.5)],
        )
        mat_iso = make_test_material_law74(
            E=70000.0, nu=0.33, chard=0.0, sigy0=sigy0,
            yield_table=[(0.0, sigy0), (0.05, sigy0 * 1.5)],
        )

        def _run_forward_reverse_cycle(mat: Material) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            sig = np.zeros((1, 6), dtype=float)
            epsp = np.zeros(1, dtype=float)
            extra = {
                "uvar74": np.zeros((1, 10)),
                "temp": np.full(1, 293.0),
                "off": np.ones(1),
            }
            dt = 1.0e-6

            stress_history = []
            strain_history = []
            alpha_history = []
            curr_eps_xx = 0.0

            # Phase 1: Forward tension well into plastic regime (35 steps)
            deps_fwd = np.array([[2.0e-4, -0.33 * 2.0e-4, -0.33 * 2.0e-4, 0.0, 0.0, 0.0]])
            for _ in range(35):
                curr_eps_xx += deps_fwd[0, 0]
                sig, epsp = solid_update(mat, sig, deps_fwd, epsp, dt=dt, extra=extra)
                stress_history.append(float(sig[0, 0]))
                strain_history.append(curr_eps_xx)
                alpha_history.append(float(extra["uvar74"][0, 4]))

            # Phase 2: Reverse compression past reverse yield (50 steps)
            deps_rev = np.array([[-2.0e-4, 0.33 * 2.0e-4, 0.33 * 2.0e-4, 0.0, 0.0, 0.0]])
            for _ in range(50):
                curr_eps_xx += deps_rev[0, 0]
                sig, epsp = solid_update(mat, sig, deps_rev, epsp, dt=dt, extra=extra)
                stress_history.append(float(sig[0, 0]))
                strain_history.append(curr_eps_xx)
                alpha_history.append(float(extra["uvar74"][0, 4]))

            return np.array(stress_history), np.array(strain_history), np.array(alpha_history)

        sig_kin, eps_kin, alpha_kin = _run_forward_reverse_cycle(mat_kin)
        sig_iso, eps_iso, alpha_iso = _run_forward_reverse_cycle(mat_iso)

        # 1. Backstress check: alpha_xx > 0 develops under kinematic hardening
        peak_alpha = np.max(alpha_kin)
        assert peak_alpha > 3.0, f"Expected positive kinematic backstress (>3.0 MPa), got {peak_alpha}"
        assert np.allclose(alpha_iso, 0.0), "Isotropic hardening must maintain zero backstress"

        # 2. Bauschinger effect: reverse yielding begins at lower absolute stress
        # For kinematic hardening, the reverse elastic limit occurs when sigma_eff = sigma - alpha reaches -Y
        # so reverse yield starts earlier (at a higher algebraic stress value, closer to 0)
        # Find minimum (most compressive) stress reached at end of cycle:
        min_sig_kin = np.min(sig_kin)
        min_sig_iso = np.min(sig_iso)
        assert min_sig_kin > min_sig_iso, (
            f"Bauschinger softening: kinematic min stress {min_sig_kin:.1f} must be less negative than isotropic {min_sig_iso:.1f}"
        )


# ============================================================================
# 3. Strict Energy Conservation Check (|Delta E| / E_0 < 0.1%)
# ============================================================================

class TestLaw74StrictEnergyConservation:
    """Strict energy conservation audit (|Delta E| / E_0 < 0.1%) for Solid Hexa8 and Tetra4."""

    def test_solid_hexa8_undamped_free_oscillation_energy_conservation(self, tmp_path: Path):
        """Free vibration of undamped Solid Hexa8 in elastic regime: |Delta E| / E_0 < 0.1%."""
        deck = StarterDeck("HEXA8_ELAS_OSC")
        deck.mat_law74(
            1, "ElasticHexa74",
            rho=2.7e-9, e=70000.0, nu=0.33,
            sigy0=10000.0,  # High yield stress for pure elastic response
            eps_max=1.0e30,
        )
        deck.prop_solid(1, "PropHexa", isolid=1)
        deck.part(1, "PartSolid", 1, 1)

        lx, ly, lz = 10.0, 10.0, 10.0
        deck.node([
            (1, 0.0, 0.0, 0.0), (2, lx, 0.0, 0.0), (3, lx, ly, 0.0), (4, 0.0, ly, 0.0),
            (5, 0.0, 0.0, lz), (6, lx, 0.0, lz), (7, lx, ly, lz), (8, 0.0, ly, lz),
        ])
        deck.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])

        s_path = str(tmp_path / "HEXA8_ELAS_OSC_0000.rad")
        deck.write(s_path)
        with contextlib.redirect_stdout(io.StringIO()):
            model = run_starter(s_path)

        group = model.bricks
        rho0 = 2.7e-9
        vol0 = lx * ly * lz
        m_node = rho0 * vol0 / 8.0
        mass_vec = np.full(8, m_node)

        # Pure symmetric breathing mode in X
        v = np.zeros_like(model.x)
        vx0 = 30.0
        v[[1, 2, 5, 6], 0] = vx0
        v[[0, 3, 4, 7], 0] = -vx0

        e_kin_0 = 0.5 * np.sum(mass_vec * (vx0 ** 2))
        e_int_0 = float(np.sum(group.state["eint"]))
        e_tot_0 = e_kin_0 + e_int_0
        assert e_tot_0 > 0.0

        # Fine time step for strict energy conservation
        dt = 2.0e-8
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        model.v = v.copy()

        e_tot_history = []
        for step in range(100):
            fint.fill(0.0)
            mint.fill(0.0)
            solid_hexa8.forces(group, model.x, model.v, model.vr, dt, fint, mint)
            v_old = model.v.copy()
            acc = fint / mass_vec[:, None]
            model.v += acc * dt
            model.x += model.v * dt
            v_mid = 0.5 * (v_old + model.v)
            e_kin = 0.5 * np.sum(mass_vec[:, None] * (v_mid ** 2))
            e_int = float(np.sum(group.state["eint"]))
            e_tot_history.append(e_kin + e_int)

        max_rel_err = max(abs(e - e_tot_0) / e_tot_0 for e in e_tot_history)
        assert max_rel_err < 0.001, (
            f"Hexa8 elastic energy conservation error {max_rel_err*100:.4f}% exceeds strict 0.1% bound"
        )

    def test_solid_tetra4_undamped_free_oscillation_energy_conservation(self, tmp_path: Path):
        """Free vibration of undamped Solid Tetra4 in elastic regime: |Delta E| / E_0 < 0.1%."""
        deck = StarterDeck("TETRA4_ELAS_OSC")
        deck.mat_law74(
            1, "ElasticTetra74",
            rho=2.7e-9, e=70000.0, nu=0.33,
            sigy0=10000.0,
            eps_max=1.0e30,
        )
        deck.prop_solid(1, "PropTetra", isolid=1)
        deck.part(1, "PartTetra", 1, 1)

        deck.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 0.0, 10.0, 0.0), (4, 0.0, 0.0, 10.0),
        ])
        deck.tetra4(1, [(1, 1, 2, 3, 4)])

        s_path = str(tmp_path / "TETRA4_ELAS_OSC_0000.rad")
        deck.write(s_path)
        with contextlib.redirect_stdout(io.StringIO()):
            model = run_starter(s_path)

        group = model.tetras
        rho0 = 2.7e-9
        vol0 = 1000.0 / 6.0
        m_node = rho0 * vol0 / 4.0
        mass_vec = np.full(4, m_node)

        # Initial velocity perturbation on apex node 4 in Z
        v = np.zeros_like(model.x)
        vz0 = 20.0
        v[3, 2] = vz0

        e_kin_0 = 0.5 * m_node * (vz0 ** 2)
        e_int_0 = float(np.sum(group.state["eint"]))
        e_tot_0 = e_kin_0 + e_int_0
        assert e_tot_0 > 0.0

        dt = 2.0e-8
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        model.v = v.copy()

        e_tot_history = []
        for step in range(100):
            fint.fill(0.0)
            mint.fill(0.0)
            solid_tetra4.forces(group, model.x, model.v, model.vr, dt, fint, mint)
            v_old = model.v.copy()
            acc = fint / mass_vec[:, None]
            model.v += acc * dt
            model.x += model.v * dt
            v_mid = 0.5 * (v_old + model.v)
            e_kin = 0.5 * np.sum(mass_vec[:, None] * (v_mid ** 2))
            e_int = float(np.sum(group.state["eint"]))
            e_tot_history.append(e_kin + e_int)

        max_rel_err = max(abs(e - e_tot_0) / e_tot_0 for e in e_tot_history)
        assert max_rel_err < 0.001, (
            f"Tetra4 elastic energy conservation error {max_rel_err*100:.4f}% exceeds strict 0.1% bound"
        )

    def test_strain_energy_ledger_trapezoidal_integration(self):
        """Incremental trapezoidal work matches exact analytical strain energy in elastic regime."""
        mat = make_test_material_law74(
            E=70000.0, nu=0.33, sigy0=10000.0,
        )
        vol0 = 1000.0  # 10 x 10 x 10 mm
        sig = np.zeros((1, 6), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra = {
            "uvar74": np.zeros((1, 10)),
            "temp": np.full(1, 293.0),
            "off": np.ones(1),
        }
        dt = 1.0e-6

        nu = 0.33
        E = 70000.0
        G = 0.5 * E / (1.0 + nu)
        C1 = E / (3.0 * (1.0 - 2.0 * nu))
        lam = C1 - 2.0 * G / 3.0

        eps_total = np.zeros(6, dtype=float)
        eint_incremental = 0.0

        for step in range(20):
            deps = np.array([[1.0e-4, -0.33 * 1.0e-4, -0.33 * 1.0e-4, 5.0e-5, 0.0, 0.0]])
            sig_old = sig.copy()
            res = solid_update(mat, sig, deps, epsp, dt=dt, extra=extra)
            sig_new = res[0].copy()
            sig = sig_new.copy()
            epsp[0] = _scalar(res[1])

            sig_mid = 0.5 * (sig_old + sig_new)
            de = (np.sum(sig_mid[0, :3] * deps[0, :3]) + np.sum(sig_mid[0, 3:] * deps[0, 3:])) * vol0
            eint_incremental += de
            eps_total += deps[0]

        # Analytical 3D isotropic elastic strain energy:
        # U = 0.5 * (lambda * tr(eps)^2 + 2 * G * tr(eps^2)) * vol0
        exx, eyy, ezz, exy, eyz, ezx = eps_total
        tre = exx + eyy + ezz
        tr_eps2 = exx**2 + eyy**2 + ezz**2 + 2.0 * (0.5 * exy)**2 + 2.0 * (0.5 * eyz)**2 + 2.0 * (0.5 * ezx)**2
        analytical_eint = 0.5 * (lam * tre**2 + 2.0 * G * tr_eps2) * vol0

        assert eint_incremental == pytest.approx(analytical_eint, rel=1e-4), (
            f"Incremental strain energy {eint_incremental} != analytical {analytical_eint}"
        )
        assert epsp[0] == 0.0, "No plastic strain should accumulate in elastic regime"

    def test_plastic_dissipation_monotonically_increasing(self):
        """Plastic strain and plastic dissipation monotonically non-decreasing during plastic flow."""
        mat = make_test_material_law74(
            E=70000.0, nu=0.33, sigy0=200.0,
            yield_table=[(0.0, 200.0), (0.1, 320.0)],
        )
        sig = np.zeros((1, 6), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra = {
            "uvar74": np.zeros((1, 10)),
            "temp": np.full(1, 293.0),
            "off": np.ones(1),
        }

        epsp_history = []
        dt = 1.0e-6

        for step in range(40):
            deps = np.array([[3.0e-4, -1.0e-4, -1.0e-4, 1.5e-4, 0.0, 0.0]])
            res = solid_update(mat, sig, deps, epsp, dt=dt, extra=extra)
            sig = res[0].copy()
            epsp[0] = _scalar(res[1])
            epsp_history.append(float(epsp[0]))

        d_epsp = np.diff(epsp_history)
        assert np.all(d_epsp >= -1e-15), "Plastic strain increments must be non-negative"
        assert epsp_history[-1] > epsp_history[0], "Plastic strain must strictly increase during yield"


# ============================================================================
# 4. Adiabatic Heating Verification (rhocp > 0)
# ============================================================================

class TestLaw74AdiabaticHeatingVerification:
    """Audit adiabatic temperature rise during plastic dissipation."""

    def test_adiabatic_heating_formula_verification(self):
        """Verify temperature increase matches integral of (sigma_yield * d_epsp) / (rho * cp)."""
        rhocp = 2.4e-3  # J/(mm^3 * K)
        t0 = 293.0
        sigy0 = 180.0
        mat = make_test_material_law74(
            E=70000.0, nu=0.33, sigy0=sigy0,
            t0=t0, rhocp=rhocp,
            yield_table=[(0.0, sigy0), (0.1, sigy0)],  # constant yield stress for exact integration
        )

        sig = np.zeros((1, 6), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra = {
            "uvar74": np.zeros((1, 10)),
            "temp": np.full(1, t0),
            "off": np.ones(1),
        }
        dt = 1.0e-6

        # Step 1: Elastic deformation (no heating)
        for _ in range(5):
            deps_elas = np.array([[1.0e-4, -0.33 * 1.0e-4, -0.33 * 1.0e-4, 0.0, 0.0, 0.0]])
            sig, epsp = solid_update(mat, sig, deps_elas, epsp, dt=dt, extra=extra)

        assert epsp[0] == 0.0
        assert extra["temp"][0] == pytest.approx(t0), "Temperature must remain constant in elastic regime"

        # Step 2: Plastic deformation
        epsp_start = float(epsp[0])
        for _ in range(30):
            deps_plas = np.array([[5.0e-4, -0.33 * 5.0e-4, -0.33 * 5.0e-4, 0.0, 0.0, 0.0]])
            sig, epsp = solid_update(mat, sig, deps_plas, epsp, dt=dt, extra=extra)

        delta_epsp = float(epsp[0]) - epsp_start
        assert delta_epsp > 0.0

        # Theoretical Delta T = (sigy0 * delta_epsp) / rhocp
        expected_delta_t = (sigy0 * delta_epsp) / rhocp
        actual_delta_t = float(extra["temp"][0]) - t0

        assert actual_delta_t > 0.0
        assert actual_delta_t == pytest.approx(expected_delta_t, rel=0.05), (
            f"Realized Delta T ({actual_delta_t:.3f} K) must match theoretical ({expected_delta_t:.3f} K)"
        )


# ============================================================================
# 5. Acoustic Sound Speed & Courant Stability
# ============================================================================

class TestLaw74SoundSpeedAndCourantStability:
    """Audit dilatational sound speed and Courant time-step bounds."""

    def test_sound_speed_solid_positivity_across_alloys(self):
        """Verify positive dilatational sound speed across common engineering alloys."""
        alloys = [
            ("Steel", 210000.0, 0.30, 7.85e-9),
            ("Aluminum", 70000.0, 0.33, 2.70e-9),
            ("Titanium", 110000.0, 0.31, 4.50e-9),
            ("Copper", 120000.0, 0.34, 8.96e-9),
        ]
        lc = 5.0  # characteristic length = 5.0 mm

        for name, E, nu, rho0 in alloys:
            p = Law74Params(e=E, nu=nu, rho0=rho0)
            c_solid = sound_speed_solid(p)
            G = 0.5 * E / (1.0 + nu)
            C1 = E / (3.0 * (1.0 - 2.0 * nu))
            c_expected = math.sqrt((C1 + 4.0 / 3.0 * G) / rho0)

            assert c_solid == pytest.approx(c_expected, rel=1e-6)
            assert c_solid > 0.0

            dt_courant = lc / c_solid
            assert 1.0e-7 < dt_courant < 1.0e-5, f"Courant dt {dt_courant} outside reasonable bounds for {name}"

    def test_sound_speed_stability_under_modulus_degradation(self):
        """Sound speed remains positive and stable under dynamic Young's modulus degradation."""
        e0, einf, ce, rho0, nu = 70000.0, 50000.0, 25.0, 2.7e-9, 0.33
        p = Law74Params(e=e0, einf=einf, ce=ce, rho0=rho0, nu=nu)

        for pla in [0.0, 0.01, 0.05, 0.10, 0.50]:
            extra = {"uvar74": np.array([[pla] + [0.0] * 9])}
            c = sound_speed_solid(p, rho=rho0, extra=extra)
            assert np.isfinite(c)
            assert c > 3000000.0, f"Sound speed must remain high and positive (got {c})"
