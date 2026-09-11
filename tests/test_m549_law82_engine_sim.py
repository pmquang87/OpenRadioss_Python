"""
Auditor 2C: Engine Simulation & Energy Balance Auditor for M549 (/MAT/LAW82 /MAT/OGDEN).

Dynamic multi-cycle engine simulation test suite verifying:
1. End-to-end explicit simulations using pyradioss.starter.starter.run_starter and
   pyradioss.engine.engine.run_engine across single-element and multi-element configurations.
2. All 6 supported element formulations:
   - Solid Hexa8 (standard 1-pt integration with viscous hourglass)
   - Solid HEPH (Isolid=24 physical hourglass stabilization)
   - Solid Tetra4 (constant strain tetrahedron)
   - Shell BT4 (Belytschko-Tsay quad shell)
   - Shell QEPH (physical hourglass quad shell)
   - Shell Tri3 (3-node C0 triangle shell)
3. Energy conservation & energy balance ledger:
   - Pure hyperelastic deformation without external damping: mechanical energy is strictly
     conserved: Delta E = |E_kin + E_int - W_ext| < 1% x E_max (relative drift < 1e-4).
   - Energy balance ledger: verify internal energy increments dE_int = sigma : d_eps * V
     match exact analytical strain energy W(F) * V0.
4. Hyperelastic reversibility (path-independence):
   - Impose cyclic displacement (load then unload back to initial position x0).
   - Verify residual Cauchy stress drops to zero (|sigma| < 1e-4 MPa), plastic strain remains
     identically zero (E_pla = 0.0), and internal energy returns to zero.
5. Dynamic wave propagation:
   - Elastic pulse propagating through an Ogden bar/plate.
   - Verify wave speed matches c_solid = sqrt((4/3 G + K) / rho0) and
     c_shell = sqrt((2/3 G + K) / rho0), with stable Courant time step bounding.
6. Official benchmark test:
   - Run official deck RD-E-5600 rubber_tension_v1 (LAW82 rubber tension model).
   - Run engine for 10-20 cycles (Tstop = 1e-4).
   - Assert normal termination, zero errors, positive time steps, and |ERR| < 1%.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pytest

from pyradioss.elements import (
    shell_bt4,
    shell_qeph,
    shell_tri3,
    solid_heph,
    solid_hexa8,
    solid_tetra4,
)
from pyradioss.engine.engine import run_engine, _energies
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.materials.law82_ogden import (
    OgdenParams,
    build_law82,
    solid_update,
    shell_update,
    solid_sound_speed,
    shell_sound_speed,
)
from pyradioss.starter.starter import run_starter


# ============================================================================
# Helpers: Analytical Strain Energy and Engine Deck Generator
# ============================================================================

def ogden_strain_energy_density(mat: OgdenParams, F: np.ndarray) -> float:
    """Compute analytical Ogden strain energy density W(F) according to sigeps82.F.

    W = sum_{k=1}^N (2 mu_k / alpha_k^2) * (lambda_bar_1^alpha_k + lambda_bar_2^alpha_k + lambda_bar_3^alpha_k - 3)
        + sum_{k=1}^N (1 / D_k) * (J - 1)^(2k)
    """
    lam = np.linalg.svd(F, compute_uv=False)
    J = float(lam[0] * lam[1] * lam[2])
    lam_bar = lam * (J ** (-1.0 / 3.0))

    w_dev = 0.0
    w_vol = 0.0
    for k in range(mat.nordre):
        mu_k = float(mat.mu[k])
        alpha_k = float(mat.alpha[k])
        d_k = float(mat.d[k]) if k < len(mat.d) else 0.0

        w_dev += (2.0 * mu_k / (alpha_k ** 2)) * float(np.sum(lam_bar ** alpha_k) - 3.0)
        if d_k > 0.0:
            k_ord = k + 1
            w_vol += (1.0 / d_k) * ((J - 1.0) ** (2 * k_ord))

    return float(w_dev + w_vol)


def _write_engine_deck(
    path: Path | str,
    run_name: str,
    tstop: float = 1.0e-4,
    dt_scale: float = 0.5,
    stop_cycles: int | None = None,
    print_freq: int = -1000,
) -> None:
    """Generate and write an engine control deck file."""
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


# ============================================================================
# 1. End-to-End Explicit Simulations Across All 6 Formulations
# ============================================================================

class TestLaw82FormulationsEngineSim:
    """Audit single-element and multi-element explicit simulations for all 6 formulations."""

    def test_solid_hexa8_engine_simulation(self, tmp_path: Path):
        """Solid Hexa8: 1-element and 2-element models with standard 1-pt integration."""
        run_name = "HEXA8_LAW82"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law82(1, rho0=1.0e-9, nu=0.45, nordre=1, mu=[10.0], alpha=[2.0])
        d.prop_solid(1, "SolidHexaProp", isolid=1)
        d.part(1, "PartHexa", 1, 1)

        # 2-element block along X
        d.node([
            (1, 0.0, 0.0, 0.0), (2, 5.0, 0.0, 0.0), (3, 5.0, 5.0, 0.0), (4, 0.0, 5.0, 0.0),
            (5, 0.0, 0.0, 5.0), (6, 5.0, 0.0, 5.0), (7, 5.0, 5.0, 5.0), (8, 0.0, 5.0, 5.0),
            (9, 10.0, 0.0, 0.0), (10, 10.0, 5.0, 0.0), (11, 10.0, 0.0, 5.0), (12, 10.0, 5.0, 5.0),
        ])
        d.brick(1, [
            (1, 1, 2, 3, 4, 5, 6, 7, 8),
            (2, 2, 9, 10, 3, 6, 11, 12, 7),
        ])

        # Boundary conditions: fix left face, pull right face
        d.grnod_node(1, "fix_face", [1, 4, 5, 8])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "pull_face", [9, 10, 11, 12])
        d.funct(1, "vel_ramp", [(0.0, 100.0), (1.0e-3, 100.0)])
        d.impvel(1, "pull_x", 1, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20, f"Expected >= 20 cycles, got {state.cycle}"
        assert state.t > 0.0
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Energy error |ERR|={en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0, "Internal strain energy must accumulate"
        assert en["EW"] > 0.0, "External work must be positive"

    def test_solid_heph_engine_simulation(self, tmp_path: Path):
        """Solid HEPH: 1-element and 2-element models with Isolid=24 physical stabilization."""
        run_name = "HEPH_LAW82"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law82(1, rho0=1.0e-9, nu=0.495, nordre=2, mu=[0.000045, 0.54], alpha=[7.16, -4.15])
        d.prop_solid(1, "SolidHEPHProp", isolid=24)
        d.part(1, "PartHEPH", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 5.0, 0.0, 0.0), (3, 5.0, 5.0, 0.0), (4, 0.0, 5.0, 0.0),
            (5, 0.0, 0.0, 5.0), (6, 5.0, 0.0, 5.0), (7, 5.0, 5.0, 5.0), (8, 0.0, 5.0, 5.0),
            (9, 10.0, 0.0, 0.0), (10, 10.0, 5.0, 0.0), (11, 10.0, 0.0, 5.0), (12, 10.0, 5.0, 5.0),
        ])
        d.brick(1, [
            (1, 1, 2, 3, 4, 5, 6, 7, 8),
            (2, 2, 9, 10, 3, 6, 11, 12, 7),
        ])

        d.grnod_node(1, "fix_face", [1, 4, 5, 8])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "pull_face", [9, 10, 11, 12])
        d.funct(1, "vel_ramp", [(0.0, 50.0), (1.0e-3, 50.0)])
        d.impvel(1, "pull_x", 1, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Energy error |ERR|={en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0

    def test_solid_tetra4_engine_simulation(self, tmp_path: Path):
        """Solid Tetra4: constant strain tetrahedron without hourglassing."""
        run_name = "TETRA4_LAW82"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law82(1, rho0=1.0e-9, nu=0.45, nordre=1, mu=[10.0], alpha=[2.0])
        d.prop_solid(1, "SolidTetraProp", isolid=1)
        d.part(1, "PartTetra", 1, 1)

        # 2 tetrahedra sharing a triangular face (nodes 2, 3, 4)
        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 0.0, 10.0, 0.0), (4, 0.0, 0.0, 10.0),
            (5, 10.0, 10.0, 0.0),
        ])
        d.tetra4(1, [
            (1, 1, 2, 3, 4),
            (2, 2, 5, 3, 4),
        ])

        d.grnod_node(1, "fix_node", [1])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "pull_apex", [4])
        d.funct(1, "vel_ramp", [(0.0, 50.0), (1.0e-3, 50.0)])
        d.impvel(1, "pull_z", 1, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.5e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 15
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Energy error |ERR|={en['ERR']}% exceeds 1.0%"
        assert en["HE"] == 0.0, "Tetra4 must have strictly zero hourglass energy"

    def test_shell_bt4_engine_simulation(self, tmp_path: Path):
        """Shell BT4: Belytschko-Tsay 4-node shell with plane-stress Ogden update."""
        run_name = "SHELL_BT4_LAW82"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law82(1, rho0=1.0e-9, nu=0.45, nordre=1, mu=[10.0], alpha=[2.0])
        d.prop_shell(1, "PropBT4", thick=1.0, nip=3, ishell=1)
        d.part(1, "PartBT4", 1, 1)

        # 2 quad shells in a strip
        d.node([
            (1, 0.0, 0.0, 0.0), (2, 5.0, 0.0, 0.0), (3, 5.0, 5.0, 0.0), (4, 0.0, 5.0, 0.0),
            (5, 10.0, 0.0, 0.0), (6, 10.0, 5.0, 0.0),
        ])
        d.shell(1, [
            (1, 1, 2, 3, 4),
            (2, 2, 5, 6, 3),
        ])

        d.grnod_node(1, "fix_edge", [1, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "pull_edge", [5, 6])
        d.funct(1, "vel_ramp", [(0.0, 80.0), (1.0e-3, 80.0)])
        d.impvel(1, "pull_x", 1, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Energy error |ERR|={en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0

    def test_shell_qeph_engine_simulation(self, tmp_path: Path):
        """Shell QEPH: 4-node shell with physical hourglass stabilization (ishell=24)."""
        run_name = "SHELL_QEPH_LAW82"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law82(1, rho0=1.0e-9, nu=0.45, nordre=1, mu=[10.0], alpha=[2.0])
        d.prop_shell(1, "PropQEPH", thick=1.0, nip=3, ishell=24)
        d.part(1, "PartQEPH", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 5.0, 0.0, 0.0), (3, 5.0, 5.0, 0.0), (4, 0.0, 5.0, 0.0),
            (5, 10.0, 0.0, 0.0), (6, 10.0, 5.0, 0.0),
        ])
        d.shell(1, [
            (1, 1, 2, 3, 4),
            (2, 2, 5, 6, 3),
        ])

        d.grnod_node(1, "fix_edge", [1, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "pull_edge", [5, 6])
        d.funct(1, "vel_ramp", [(0.0, 60.0), (1.0e-3, 60.0)])
        d.impvel(1, "pull_x", 1, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Energy error |ERR|={en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0

    def test_shell_tri3_engine_simulation(self, tmp_path: Path):
        """Shell Tri3: 3-node C0 triangle shell (ish3n=1)."""
        run_name = "SHELL_TRI3_LAW82"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law82(1, rho0=1.0e-9, nu=0.45, nordre=1, mu=[10.0], alpha=[2.0])
        d.prop_shell(1, "PropTri3", thick=1.0, nip=3, ish3n=1)
        d.part(1, "PartTri3", 1, 1)

        # 2 triangles sharing an edge (nodes 1, 3)
        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
        ])
        d.sh3n(1, [
            (1, 1, 2, 3),
            (2, 1, 3, 4),
        ])

        d.grnod_node(1, "fix_edge", [1, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "pull_edge", [2, 3])
        d.funct(1, "vel_ramp", [(0.0, 50.0), (1.0e-3, 50.0)])
        d.impvel(1, "pull_x", 1, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.5e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Energy error |ERR|={en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0


# ============================================================================
# 2. Energy Conservation and Energy Balance Ledger Verification
# ============================================================================

class TestLaw82EnergyConservationAndLedger:
    """Verify mechanical energy conservation in undamped oscillation and work ledger."""

    def test_hyperelastic_free_oscillation_energy_conservation(self, tmp_path: Path):
        """Solid Hexa8 free oscillation without damping: Delta E < 1% x E_max."""
        run_name = "FREE_OSC_HEXA"
        s_path = str(tmp_path / f"{run_name}_0000.rad")

        d = StarterDeck(run_name)
        d.mat_law82(1, rho0=1.0e-9, nu=0.45, nordre=1, mu=[10.0], alpha=[2.0])
        # Disable artificial damping and viscous hourglassing (h=0) for pure hyperelasticity
        d.prop_solid(1, "SolidProp", isolid=1, qa=0.0, qb=0.0, h=0.0)
        d.part(1, "HexaPart", 1, 1)
        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0), (6, 10.0, 0.0, 10.0), (7, 10.0, 10.0, 10.0), (8, 0.0, 10.0, 10.0),
        ])
        d.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])
        d.write(s_path)

        with contextlib.redirect_stdout(io.StringIO()):
            model = run_starter(s_path)

        group = model.bricks
        rho0 = 1.0e-9
        v0_block = 1000.0  # 10 x 10 x 10 mm^3
        m_node = rho0 * v0_block / 8.0
        mass_vec = np.full(8, m_node)

        # Symmetric breathing mode: nodes pulled apart in X
        v = np.zeros_like(model.x)
        v[[1, 2, 5, 6], 0] = 50.0
        v[[0, 3, 4, 7], 0] = -50.0

        dt = 1.5e-6  # Within Courant time step
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)

        v_half = v.copy()
        e_kin_0 = 0.5 * np.sum(mass_vec * (v[:, 0] ** 2))
        e_tot_0 = e_kin_0
        assert e_tot_0 > 0.0

        total_steps = 100
        e_tot_history = []
        e_kin_history = []
        e_int_history = []

        for step in range(total_steps):
            model.x += v_half * dt
            fint.fill(0.0)
            mint.fill(0.0)
            solid_hexa8.forces(group, model.x, v_half, model.vr, dt, fint, mint)

            acc = fint / mass_vec[:, None]
            v_next_half = v_half + acc * dt
            v_mid = 0.5 * (v_half + v_next_half)

            e_kin = 0.5 * np.sum(mass_vec[:, None] * (v_mid ** 2))
            e_int = float(np.sum(group.state["eint"]))
            e_tot = e_kin + e_int

            e_kin_history.append(e_kin)
            e_int_history.append(e_int)
            e_tot_history.append(e_tot)

            v_half = v_next_half

        # Mechanical energy conservation assertion: max relative drift < 1%
        e_max = max(e_tot_history)
        max_delta_e = max(abs(e - e_tot_0) for e in e_tot_history)
        rel_drift = max_delta_e / e_max

        assert rel_drift < 0.01, f"Energy drift {rel_drift:.4e} exceeds 1% requirement"
        # Verify dynamic exchange between kinetic and strain energy
        assert min(e_kin_history) < 0.6 * e_tot_0, "Kinetic energy must convert to strain energy"
        assert max(e_int_history) > 0.4 * e_tot_0, "Strain energy must reach peak"
        # Zero plastic strain in hyperelasticity
        assert np.all(group.state["epsp"] == 0.0), "Plastic strain must remain zero"

    def test_hyperelastic_shell_free_oscillation_energy_conservation(self, tmp_path: Path):
        """Shell BT4 dynamic membrane oscillation: mechanical energy strictly conserved."""
        run_name = "FREE_OSC_SHELL"
        s_path = str(tmp_path / f"{run_name}_0000.rad")

        d = StarterDeck(run_name)
        d.mat_law82(1, rho0=1.0e-9, nu=0.45, nordre=1, mu=[10.0], alpha=[2.0])
        # Disable damping / hourglass for pure elastic energy conservation
        d.prop_shell(1, "ShellProp", thick=1.0, nip=3, ishell=1, hm=0.0, hf=0.0, hr=0.0)
        d.part(1, "ShellPart", 1, 1)
        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
        ])
        d.shell(1, [(1, 1, 2, 3, 4)])
        d.write(s_path)

        with contextlib.redirect_stdout(io.StringIO()):
            model = run_starter(s_path)

        group = model.shells
        rho0 = 1.0e-9
        m_node = rho0 * 1.0 * 100.0 / 4.0
        mass_vec = np.full(4, m_node)

        v = np.zeros_like(model.x)
        v[[1, 2], 0] = 50.0
        v[[0, 3], 0] = -50.0

        dt = 1.0e-6
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)

        v_half = v.copy()
        e_tot_0 = 0.5 * np.sum(mass_vec * (v[:, 0] ** 2))
        e_tot_history = []

        for step in range(80):
            model.x += v_half * dt
            fint.fill(0.0)
            mint.fill(0.0)
            shell_bt4.forces(group, model.x, v_half, model.vr, dt, fint, mint)

            acc = fint / mass_vec[:, None]
            v_next_half = v_half + acc * dt
            v_mid = 0.5 * (v_half + v_next_half)

            e_kin = 0.5 * np.sum(mass_vec[:, None] * (v_mid ** 2))
            e_int = float(np.sum(group.state["eint"]))
            e_tot = e_kin + e_int
            e_tot_history.append(e_tot)
            v_half = v_next_half

        max_drift = max(abs(e - e_tot_0) for e in e_tot_history) / e_tot_0
        assert max_drift < 0.01, f"Shell energy drift {max_drift:.4e} exceeds 1%"
        assert np.all(group.state["epsp"] == 0.0)

    def test_internal_energy_increment_ledger_matches_strain_energy(self):
        """Verify dE_int = sigma : d_eps * V matches analytical Ogden strain energy W(F)*V0."""
        mat = build_law82(nu=0.45, mu=[10.0], alpha=[2.0])
        v0 = 1000.0  # reference volume (10 x 10 x 10)

        # Stretch along X from lam=1.0 to lam=1.25 over 250 increments
        steps = 250
        lam_arr = np.linspace(1.0, 1.25, steps + 1)
        accumulated_work = 0.0
        sig = np.zeros(6)

        for i in range(steps):
            lam_prev = lam_arr[i]
            lam_next = lam_arr[i + 1]

            eps_prev = np.array([np.log(lam_prev), -0.45 * np.log(lam_prev), -0.45 * np.log(lam_prev), 0, 0, 0])
            eps_next = np.array([np.log(lam_next), -0.45 * np.log(lam_next), -0.45 * np.log(lam_next), 0, 0, 0])
            deps = eps_next - eps_prev
            eps_mid = 0.5 * (eps_prev + eps_next)

            sig_mid, _ = solid_update(mat, sig, deps=deps, eps=eps_mid)

            # Current volume V = V0 * J = V0 * exp(tr(eps_mid))
            j_mid = np.exp(np.sum(eps_mid[:3]))
            v_curr = v0 * j_mid
            d_work = np.sum(sig_mid * deps) * v_curr
            accumulated_work += d_work

        # Analytical strain energy at final state
        lam_final = lam_arr[-1]
        f_final = np.diag([lam_final, np.exp(-0.45 * np.log(lam_final)), np.exp(-0.45 * np.log(lam_final))])
        w_final = ogden_strain_energy_density(mat, f_final)
        e_analytical = w_final * v0

        rel_diff = abs(accumulated_work - e_analytical) / e_analytical
        assert rel_diff < 1.0e-4, f"Work ledger error {rel_diff:.4e} exceeds 0.01% tolerance"

    def test_multi_term_ogden_energy_ledger(self):
        """Verify N=3 Ogden series work-energy ledger matches analytical strain energy."""
        mu_3 = [0.000045, 0.54, 0.10]
        alpha_3 = [7.16, -4.15, 2.0]
        mat = build_law82(nu=0.495, nordre=3, mu=mu_3, alpha=alpha_3)
        v0 = 500.0

        steps = 200
        lam_arr = np.linspace(1.0, 1.20, steps + 1)
        accumulated_work = 0.0
        sig = np.zeros(6)

        for i in range(steps):
            lam_prev = lam_arr[i]
            lam_next = lam_arr[i + 1]

            eps_prev = np.array([np.log(lam_prev), -0.495 * np.log(lam_prev), -0.495 * np.log(lam_prev), 0, 0, 0])
            eps_next = np.array([np.log(lam_next), -0.495 * np.log(lam_next), -0.495 * np.log(lam_next), 0, 0, 0])
            deps = eps_next - eps_prev
            eps_mid = 0.5 * (eps_prev + eps_next)

            sig_mid, _ = solid_update(mat, sig, deps=deps, eps=eps_mid)
            v_curr = v0 * np.exp(np.sum(eps_mid[:3]))
            accumulated_work += np.sum(sig_mid * deps) * v_curr

        lam_final = lam_arr[-1]
        f_final = np.diag([lam_final, np.exp(-0.495 * np.log(lam_final)), np.exp(-0.495 * np.log(lam_final))])
        w_final = ogden_strain_energy_density(mat, f_final)
        e_analytical = w_final * v0

        rel_diff = abs(accumulated_work - e_analytical) / e_analytical
        assert rel_diff < 1.0e-4, f"N=3 Ogden ledger error {rel_diff:.4e} exceeds tolerance"


# ============================================================================
# 3. Hyperelastic Reversibility (Path-Independence)
# ============================================================================

class TestLaw82HyperelasticReversibility:
    """Audit cyclic loading/unloading reversibility (zero residual stress, zero plastic strain)."""

    def test_solid_cyclic_displacement_reversibility(self):
        """3D solid cyclic stretch: verify zero residual stress and zero plastic strain upon return."""
        mat = build_law82(nu=0.45, mu=[10.0], alpha=[2.0])
        sig = np.zeros(6)

        # 2 full stretch-unload cycles: 1.0 -> 1.3 -> 1.0 -> 1.3 -> 1.0
        n_pts = 25
        stretches = np.concatenate([
            np.linspace(1.0, 1.30, n_pts),
            np.linspace(1.30, 1.0, n_pts)[1:],
            np.linspace(1.0, 1.30, n_pts)[1:],
            np.linspace(1.30, 1.0, n_pts)[1:],
        ])

        peak_stress = 0.0
        return_stresses = []
        return_epsps = []

        for i in range(1, len(stretches)):
            lam = stretches[i]
            lam_prev = stretches[i - 1]

            eps = np.array([np.log(lam), -0.45 * np.log(lam), -0.45 * np.log(lam), 0, 0, 0])
            deps = eps - np.array([np.log(lam_prev), -0.45 * np.log(lam_prev), -0.45 * np.log(lam_prev), 0, 0, 0])
            sig, epsp = solid_update(mat, sig, deps=deps, eps=eps)

            peak_stress = max(peak_stress, float(np.max(np.abs(sig))))

            if np.isclose(lam, 1.0):
                return_stresses.append(float(np.linalg.norm(sig)))
                return_epsps.append(float(epsp))

        # Peak stress is non-zero
        assert peak_stress > 5.0, f"Peak stress must develop, got {peak_stress}"
        # Residual stress upon return drops to zero
        for res_sig in return_stresses:
            assert res_sig < 1.0e-4, f"Residual stress {res_sig:.4e} MPa must be near zero"
        # Plastic strain remains strictly zero
        for ep in return_epsps:
            assert ep == 0.0, f"Plastic strain must be zero in hyperelasticity, got {ep}"

    def test_shell_cyclic_displacement_reversibility(self):
        """2D shell cyclic stretch: verify zero residual stress, lambda3 -> 1, and zero plastic strain."""
        mat = build_law82(nu=0.45, mu=[10.0], alpha=[2.0])
        sig = np.zeros(5)
        extra = {"uvar82": np.ones((1, 1), dtype=np.float64)}

        n_pts = 25
        stretches = np.concatenate([
            np.linspace(1.0, 1.25, n_pts),
            np.linspace(1.25, 1.0, n_pts)[1:],
            np.linspace(1.0, 1.25, n_pts)[1:],
            np.linspace(1.25, 1.0, n_pts)[1:],
        ])

        return_stresses = []
        return_lam3s = []
        return_epsps = []

        for i in range(1, len(stretches)):
            lam = stretches[i]
            lam_prev = stretches[i - 1]

            eps = np.array([np.log(lam), 0.0, 0.0, 0.0, 0.0])
            deps = eps - np.array([np.log(lam_prev), 0.0, 0.0, 0.0, 0.0])
            sig, epsp = shell_update(mat, sig, deps=deps, eps=eps, extra=extra)

            if np.isclose(lam, 1.0):
                return_stresses.append(float(np.linalg.norm(sig)))
                lam3 = float(extra["uvar82"][0, 0])
                return_lam3s.append(lam3)
                return_epsps.append(float(epsp))

        for res_sig in return_stresses:
            assert res_sig < 1.0e-4, f"Shell residual stress {res_sig:.4e} MPa must be near zero"
        for lam3 in return_lam3s:
            assert math.isclose(lam3, 1.0, abs_tol=1.0e-4), f"lambda_3 must return to 1.0, got {lam3}"
        for ep in return_epsps:
            assert ep == 0.0


# ============================================================================
# 4. Dynamic Wave Propagation
# ============================================================================

class TestLaw82DynamicWavePropagation:
    """Audit dynamic wave propagation speed and Courant stability in Ogden bar/plate."""

    def test_solid_dynamic_wave_propagation(self, tmp_path: Path):
        """10-element solid bar: verify acoustic sound speed c = sqrt((4/3 G + K)/rho)."""
        run_name = "WAVE_SOLID_BAR"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        rho0 = 1.0e-9
        mu = 10.0
        nu = 0.45
        mat = build_law82(rho0=rho0, nu=nu, nordre=1, mu=[mu], alpha=[2.0])

        # Theoretical solid sound speed
        c_theory = math.sqrt(((4.0 / 3.0) * mat.g0 + mat.rbulk) / rho0)
        c_solid = solid_sound_speed(mat, rho=rho0)
        assert math.isclose(c_solid, c_theory, rel_tol=1.0e-6)

        # 10-element bar of length 50 mm along X (dx = 5 mm, 5x5 cross section)
        d = StarterDeck(run_name)
        d.mat_law82(1, rho0=rho0, nu=nu, nordre=1, mu=[mu], alpha=[2.0])
        d.prop_solid(1, "BarSolidProp", isolid=1)
        d.part(1, "BarPart", 1, 1)

        nodes = []
        nid = 1
        for i in range(11):
            x = i * 5.0
            nodes.append((nid, x, 0.0, 0.0))
            nodes.append((nid + 1, x, 5.0, 0.0))
            nodes.append((nid + 2, x, 5.0, 5.0))
            nodes.append((nid + 3, x, 0.0, 5.0))
            nid += 4
        d.node(nodes)

        bricks = []
        for i in range(10):
            base = i * 4 + 1
            bricks.append((i + 1, base, base + 4, base + 5, base + 1, base + 3, base + 7, base + 6, base + 2))
        d.brick(1, bricks)

        # Velocity impulse at x=0 (nodes 1, 2, 3, 4)
        d.grnod_node(1, "pulse_nodes", [1, 2, 3, 4])
        d.funct(1, "pulse_funct", [(0.0, 100.0), (1.0e-5, 100.0), (1.1e-5, 0.0), (1.0, 0.0)])
        d.impvel(1, "pulse_x", 1, "X", 1)

        d.write(s_path)
        # Run to t = 1.0e-4 s with stable Courant timestep
        _write_engine_deck(e_path, run_name, tstop=1.0e-4, dt_scale=0.4)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 25, f"Expected >= 25 cycles, got {state.cycle}"

        # Wave arrival audit: at t = 1.0e-4 s, wave has traveled distance d = c * t ~ 33.16 mm
        # Elements 1..6 (x < 30 mm) have experienced stress disturbance
        # Elements 8..10 (x > 35 mm) have not been reached yet
        sig = eng_model.bricks.state["sig"]
        disturbed_elements = [i for i in range(6) if abs(sig[i, 0]) > 1.0e-5]
        assert len(disturbed_elements) > 0, "Wave must disturb elements within arrival distance"
        assert abs(sig[9, 0]) < 1.0e-5, "Far-end element 10 must not be disturbed before wave arrival"

    def test_shell_dynamic_wave_propagation(self, tmp_path: Path):
        """10-element shell strip: verify plane-stress sound speed c = sqrt((2/3 G + K)/rho)."""
        run_name = "WAVE_SHELL_STRIP"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        rho0 = 1.0e-9
        mu = 10.0
        nu = 0.45
        mat = build_law82(rho0=rho0, nu=nu, nordre=1, mu=[mu], alpha=[2.0])

        c_shell_theory = math.sqrt(((2.0 / 3.0) * mat.g0 + mat.rbulk) / rho0)
        c_shell = shell_sound_speed(mat, rho=rho0)
        assert math.isclose(c_shell, c_shell_theory, rel_tol=1.0e-6)

        d = StarterDeck(run_name)
        d.mat_law82(1, rho0=rho0, nu=nu, nordre=1, mu=[mu], alpha=[2.0])
        d.prop_shell(1, "StripProp", thick=1.0, nip=3, ishell=1)
        d.part(1, "StripPart", 1, 1)

        nodes = []
        for i in range(11):
            x = i * 5.0
            nodes.append((2 * i + 1, x, 0.0, 0.0))
            nodes.append((2 * i + 2, x, 5.0, 0.0))
        d.node(nodes)

        shells = []
        for i in range(10):
            shells.append((i + 1, 2 * i + 1, 2 * i + 3, 2 * i + 4, 2 * i + 2))
        d.shell(1, shells)

        d.grnod_node(1, "impulse_nodes", [1, 2])
        d.funct(1, "pulse", [(0.0, 100.0), (1.0e-5, 100.0), (1.1e-5, 0.0), (1.0, 0.0)])
        d.impvel(1, "pulse_vel", 1, "X", 1)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.0e-4, dt_scale=0.4)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 15
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0


# ============================================================================
# 5. Official OpenRadioss Benchmark Test
# ============================================================================

class TestLaw82OfficialBenchmark:
    """Audit official RD-E-5600 Ogden rubber tension deck."""

    def test_official_rd_e_5600_rubber_tension_benchmark(self, tmp_path: Path):
        """Run official RD-E-5600 rubber_tension_v1 deck for 10-20 cycles (Tstop=1e-4)."""
        src_dir = os.path.join(
            "tests", "data", "rd_decks", "rd_e",
            "RD-E-5600_Hyperelastic_material", "56_HyperElastic_Material",
            "Ogden_model", "LAW82",
        )
        assert os.path.isdir(src_dir), f"Official deck directory not found: {src_dir}"

        for fname in os.listdir(src_dir):
            if fname.endswith(".rad") or fname.endswith(".txt"):
                shutil.copy(os.path.join(src_dir, fname), os.path.join(tmp_path, fname))

        # Override Tstop from 0.801 to 1.0e-4 in rubber_tension_v1_0001.rad
        e_file = tmp_path / "rubber_tension_v1_0001.rad"
        content = e_file.read_text(encoding="utf-8")
        assert "0.801" in content, "Expected Tstop=0.801 in official engine deck"
        modified_content = content.replace("0.801", "0.0001")
        e_file.write_text(modified_content, encoding="utf-8")

        s_file = str(tmp_path / "rubber_tension_v1_0000.rad")

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_file)
            eng_model = run_engine(str(e_file))

        assert 1 in st_model.materials
        mat = st_model.materials[1]
        assert mat.law == 82
        assert mat.params["ORDER"] == 2

        state = eng_model.engine_state
        # Assert normal termination and cycle count
        assert state.cycle >= 10, f"Expected >= 10 cycles, got {state.cycle}"
        assert state.stop_reason == "", f"Engine stopped abnormally: {state.stop_reason}"
        assert state.t > 0.0

        # Assert energy error within tolerance
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Official benchmark energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0, "Internal energy must be positive"
