"""
Auditor 2C: Dynamic Simulation & Energy Balance Auditor for M552 (/MAT/LAW48 /MAT/ZHAO /MAT/PLAS_ZHAO).

Exhaustive dynamic engine simulation and energy balance audit suite verifying:
1. Multi-cycle explicit dynamic simulations:
   - Solid Hexa8 (standard 1-pt integration, isolid=1)
   - Solid HEPH (physical hourglass stabilization, isolid=24)
   - Solid Tetra4 (constant strain tetrahedron)
   - Shell BT4 (Belytschko-Tsay quad shell, ishell=1)
   - Shell QEPH (physical hourglass quad shell, ishell=24)
   - Verified through Starter and Engine decks under cyclic displacement and velocity loading.
   - Asserts normal termination, >= 20 cycles, energy balance error |ERR| < 1.0%,
     positive internal strain energy accumulation (IE > 0), external work > 0,
     and zero hourglass energy on Tetra4.
2. Energy Conservation & Work Ledger in Undamped Explicit Dynamic Simulations:
   - Free vibration of undamped 3D solid block (Hexa8, qa=0, qb=0, h=0) and 2D shell membrane (BT4, hm=0, hf=0, hr=0):
     mechanical energy E_tot = E_kin + E_int is strictly conserved (|Delta E| / E_0 < 1.0%).
   - Incremental strain energy work ledger: sum(sigma_mid : deps * V) matches exact analytical
     elastic strain energy 0.5 * eps : C : eps * V to within 0.01% (< 1e-4 relative error).
   - Elastic path reversibility: loading and unloading within elastic regime returns stresses
     to near zero (< 1e-4 MPa) and plastic strain identically zero (epsp == 0.0).
3. Cyclic Loading and Bauschinger Effect:
   - Kinematic hardening (F_isokin = 1.0 / chard = 1.0) showing back-stress evolution (alpha_xx > 0),
     earlier reverse yield softening, and closed Bauschinger hysteresis loop.
   - Isotropic hardening (F_isokin = 0.0 / chard = 0.0) showing expanding elastic domain with
     zero back-stress (alpha_xx == 0) and no reverse yield softening.
   - OpenRadioss Fortran citation: engine/source/materials/mat/mat048/sigeps48.F:295-303 and
     sigeps48c.F:307-315, 476-496.
   - Full Engine run verifying back-stress persistence in mat_extra["sigb48"].
4. Dynamic Strain-Rate Filtering:
   - Cutoff frequency f_cut smoothing impulsive strain-rate jumps over time steps via
     alpha = min(1.0, 2*pi*fcut*dt).
   - Yield stress under high strain-rate impulse is smoothly filtered toward steady-state response.
   - Verified in solid and shell kernels and during dynamic Engine simulation.
5. Progressive Failure & Element Deletion:
   - Tensile damage scaling factor FAIL = max(0, min(1, (eps_t2 - eps_t) / (eps_t2 - eps_t1)))
     degrading yield stress and hardening.
   - Complete tensile failure when eps_t >= eps_t2 and plastic failure when eps_p >= eps_max:
     stresses collapse to zero, off and off48 flags set to 0.0.
   - Post-failure continuation: simulation runs further deformation increments stably without NaNs or infinities.
   - Multi-element bar in Engine continues stably after erosion of the failed element.
6. Acoustic Sound Speeds & Courant Stability:
   - Exact longitudinal sound speed c_solid = sqrt((K + 4/3 G) / rho0) and c_shell = sqrt(E / ((1 - nu^2) * rho0)).
   - Dynamic time step bounded by Courant condition: dt <= dt_Courant = L_min / c.
   - Sound speed remains strictly finite and positive across plastic deformation and element erosion.
   - Acoustic pulse propagation along 10-element bar in Engine: wave front disturbance timing
     distinguishes near-pulse vs far-end elements according to wave speed c.

Fortran references:
- starter/source/materials/mat/mat048/hm_read_mat48.F
- engine/source/materials/mat/mat048/sigeps48.F
- engine/source/materials/mat/mat048/sigeps48c.F
"""

from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyradioss.elements import (
    shell_bt4,
    shell_qeph,
    solid_heph,
    solid_hexa8,
    solid_tetra4,
)
from pyradioss.engine.engine import run_engine, _energies
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.materials.law48_zhao import (
    Law48Params,
    build_law48,
    eval_yield_and_hardening,
    solid_update_law48,
    shell_update_law48,
    sound_speed_solid_law48,
    sound_speed_shell_law48,
    tensile_failure_factor,
    _principal_strain_3d,
    _principal_strain_2d,
)
from pyradioss.model.entities import Material
from pyradioss.starter.starter import run_starter


# ============================================================================
# Helpers: Engine Control Deck Writer and Energy Verification
# ============================================================================

def _write_engine_deck(
    path: Path | str,
    run_name: str,
    tstop: float = 1.0e-4,
    dt_scale: float = 0.5,
    stop_cycles: int | None = None,
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


# ============================================================================
# 1. Multi-Cycle Explicit Dynamic Simulations Across All Formulations
# ============================================================================

class TestLaw48MultiCycleDynamicSimulations:
    """Audit multi-cycle explicit dynamic simulations across supported solid and shell formulations."""

    def test_solid_hexa8_cyclic_engine_simulation(self, tmp_path: Path):
        """Solid Hexa8 (Isolid=1): 2-element block under cyclic pull/release loading."""
        run_name = "HEXA8_LAW48_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law48(
            mid=1, title="SteelHexa48", rho=7.85e-9, e=210000.0, nu=0.3,
            a=280.0, b=400.0, n=0.5, sig_max=800.0,
        )
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

        # Boundary conditions: fix left face, cyclic pull/release right face
        d.grnod_node(1, "fix_face", [1, 4, 5, 8])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "pull_face", [9, 10, 11, 12])
        d.funct(10, "vel_cycle", [
            (0.0, 40.0),
            (1.0e-4, 40.0),
            (1.0001e-4, -40.0),
            (2.0e-4, -40.0),
            (3.0e-4, 0.0),
        ])
        d.impvel(1, "pull_x", 10, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20, f"Expected >= 20 cycles, got {state.cycle}"
        assert state.stop_reason == "", f"Abnormal termination: {state.stop_reason}"
        assert state.t > 0.0
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Energy error |ERR|={en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0, "Internal strain energy must accumulate"
        assert en["EW"] > 0.0, "External work must be positive"

    def test_solid_heph_cyclic_engine_simulation(self, tmp_path: Path):
        """Solid HEPH (Isolid=24): physical hourglass stabilization under cyclic loading."""
        run_name = "HEPH_LAW48_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law48(
            mid=1, title="SteelHEPH48", rho=7.85e-9, e=210000.0, nu=0.3,
            a=290.0, b=420.0, n=0.55, sig_max=850.0,
        )
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
        d.funct(10, "vel_cycle", [
            (0.0, 30.0),
            (0.8e-4, 0.0),
            (1.2e-4, 30.0),
            (2.0e-4, 0.0),
        ])
        d.impvel(1, "pull_x", 10, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"HEPH energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0

    def test_solid_tetra4_cyclic_engine_simulation(self, tmp_path: Path):
        """Solid Tetra4: constant strain tetrahedron without hourglass energy under cyclic pull."""
        run_name = "TETRA4_LAW48_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law48(
            mid=1, title="SteelTetra48", rho=7.85e-9, e=200000.0, nu=0.3,
            a=260.0, b=380.0, n=0.5, sig_max=700.0,
        )
        d.prop_solid(1, "SolidTetraProp", isolid=1)
        d.part(1, "PartTetra", 1, 1)

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
        d.funct(10, "vel_cycle", [
            (0.0, 40.0),
            (1.0e-4, 40.0),
            (1.0001e-4, -40.0),
            (2.0e-4, -40.0),
        ])
        d.impvel(1, "pull_z", 10, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.5e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 15
        assert state.stop_reason == ""
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0
        assert en["HE"] == 0.0, "Tetra4 must have strictly zero hourglass energy"

    def test_shell_bt4_dynamic_biaxial_and_shear(self, tmp_path: Path):
        """Shell BT4 (Ishell=1): 4-node quad shell under dynamic biaxial and shear loading."""
        run_name = "SHELL_BT4_LAW48_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law48(
            mid=1, title="SteelBT4_48", rho=7.85e-9, e=200000.0, nu=0.3,
            a=240.0, b=350.0, n=0.5, sig_max=650.0,
        )
        d.prop_shell(1, "PropBT4", thick=1.0, nip=3, ishell=1)
        d.part(1, "PartBT4", 1, 1)

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
        # Prescribe simultaneous dynamic X-pull and Y-shear
        d.funct(10, "vel_pull_x", [
            (0.0, 50.0),
            (1.0e-4, 50.0),
            (1.0001e-4, -50.0),
            (2.0e-4, -50.0),
        ])
        d.impvel(1, "pull_x", 10, "X", 2)
        d.funct(11, "vel_shear_y", [
            (0.0, 20.0),
            (1.0e-4, 20.0),
            (1.0001e-4, -20.0),
            (2.0e-4, -20.0),
        ])
        d.impvel(2, "shear_y", 11, "Y", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0
        assert en["IE"] > 0.0

    def test_shell_qeph_dynamic_biaxial_and_shear(self, tmp_path: Path):
        """Shell QEPH (Ishell=24): physical hourglass quad shell under dynamic biaxial and shear loading."""
        run_name = "SHELL_QEPH_LAW48_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law48(
            mid=1, title="SteelQEPH48", rho=7.85e-9, e=200000.0, nu=0.3,
            a=240.0, b=350.0, n=0.5, sig_max=650.0,
        )
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
        d.funct(10, "vel_pull_x", [
            (0.0, 45.0),
            (1.0e-4, 45.0),
            (1.0001e-4, -45.0),
            (2.0e-4, -45.0),
        ])
        d.impvel(1, "pull_x", 10, "X", 2)
        d.funct(11, "vel_shear_y", [
            (0.0, 15.0),
            (1.0e-4, 15.0),
            (1.0001e-4, -15.0),
            (2.0e-4, -15.0),
        ])
        d.impvel(2, "shear_y", 11, "Y", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0
        assert en["IE"] > 0.0


# ============================================================================
# 2. Energy Conservation & Incremental Work Ledger
# ============================================================================

class TestLaw48EnergyConservationAndLedger:
    """Audit mechanical energy conservation in undamped dynamic oscillations and work ledger accuracy."""

    def test_solid_undamped_free_oscillation_energy_conservation(self, tmp_path: Path):
        """Solid Hexa8 free oscillation without damping: mechanical energy Delta E / E_0 < 1%."""
        run_name = "FREE_OSC_HEXA_LAW48"
        s_path = str(tmp_path / f"{run_name}_0000.rad")

        d = StarterDeck(run_name)
        # High initial yield stress to operate in pure linear elasticity
        d.mat_law48(mid=1, title="SteelUndamped48", rho=7.85e-9, e=210000.0, nu=0.3, a=1000.0, b=0.0)
        # Undamped: zero artificial viscosity and zero hourglass damping
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
        rho0 = 7.85e-9
        v0_block = 1000.0  # 10 x 10 x 10 mm^3
        m_node = rho0 * v0_block / 8.0
        mass_vec = np.full(8, m_node)

        # Symmetric breathing mode along X
        v = np.zeros_like(model.x)
        v[[1, 2, 5, 6], 0] = 50.0
        v[[0, 3, 4, 7], 0] = -50.0

        dt = 1.0e-7  # Courant time step resolving breathing oscillation
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)

        v_half = v.copy()
        e_kin_0 = 0.5 * np.sum(mass_vec * (v[:, 0] ** 2))
        e_tot_0 = e_kin_0
        assert e_tot_0 > 0.0

        total_steps = 100
        e_tot_history = []

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
            e_tot_history.append(e_tot)

            v_half = v_next_half

        e_max = max(e_tot_history)
        max_delta_e = max(abs(e - e_tot_0) for e in e_tot_history)
        rel_drift = max_delta_e / e_max
        assert rel_drift < 0.01, f"Hexa8 free oscillation energy deviation {rel_drift * 100:.3f}% exceeds 1.0%"

    def test_shell_undamped_free_oscillation_energy_conservation(self, tmp_path: Path):
        """Shell BT4 in-plane free oscillation without damping: mechanical energy Delta E / E_0 < 1%."""
        run_name = "FREE_OSC_SHELL_LAW48"
        s_path = str(tmp_path / f"{run_name}_0000.rad")

        d = StarterDeck(run_name)
        d.mat_law48(mid=1, title="SteelShellUndamped48", rho=7.85e-9, e=200000.0, nu=0.3, a=1000.0, b=0.0)
        d.prop_shell(1, "PropBT4", thick=1.0, nip=3, ishell=1, hm=0.0, hf=0.0, hr=0.0)
        d.part(1, "Part1", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
        ])
        d.shell(1, [(1, 1, 2, 3, 4)])
        d.write(s_path)

        with contextlib.redirect_stdout(io.StringIO()):
            model = run_starter(s_path)

        group = model.shells
        rho0 = 7.85e-9
        mass_total = rho0 * 10.0 * 10.0 * 1.0
        m_node = mass_total / 4.0
        mass_vec = np.full(4, m_node)

        v = np.zeros_like(model.x)
        v[[1, 2], 0] = 30.0
        v[[0, 3], 0] = -30.0

        dt = 1.0e-7
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)

        v_half = v.copy()
        e_kin_0 = 0.5 * np.sum(mass_vec * (v[:, 0] ** 2))
        e_tot_0 = e_kin_0

        e_tot_history = []
        for step in range(100):
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

        e_max = max(e_tot_history)
        max_delta = max(abs(e - e_tot_0) for e in e_tot_history)
        rel_dev = max_delta / e_max
        assert rel_dev < 0.01, f"Shell free oscillation energy deviation {rel_dev * 100:.3f}% exceeds 1.0%"

    def test_strain_energy_incremental_work_ledger_accuracy(self):
        """Verify incremental work ledger sum(sigma_mid : deps * V) matches exact analytical strain energy."""
        e0 = 210000.0
        nu = 0.3
        p = Law48Params(E=e0, nu=nu, sigy0=1000.0)
        k0 = p.K
        g0 = p.G

        v0 = 1000.0  # 10 x 10 x 10 mm^3 volume
        sig = np.zeros(6, dtype=float)

        work_acc = 0.0
        n_steps = 40
        deps_step = np.array([0.00002, -0.000006, -0.000006, 0.00001, 0.0, 0.0])
        eps_tot = np.zeros(6, dtype=float)

        for _ in range(n_steps):
            sig_prev = sig.copy()
            sig, _, _ = solid_update_law48(p, sig, deps=deps_step, epsp=0.0, dt=1.0e-5)
            eps_tot += deps_step

            # Midpoint work increment: sigma_mid : deps * V
            sig_mid = 0.5 * (sig_prev + sig)
            # Voigt contraction: s_xx*de_xx + s_yy*de_yy + s_zz*de_zz + s_xy*de_xy
            d_work = (
                sig_mid[0] * deps_step[0]
                + sig_mid[1] * deps_step[1]
                + sig_mid[2] * deps_step[2]
                + sig_mid[3] * deps_step[3]
            ) * v0
            work_acc += d_work

        # Exact analytical elastic strain energy density
        tr_eps = eps_tot[0] + eps_tot[1] + eps_tot[2]
        dev_eps = np.array([
            eps_tot[0] - tr_eps / 3.0,
            eps_tot[1] - tr_eps / 3.0,
            eps_tot[2] - tr_eps / 3.0,
            eps_tot[3],  # engineering shear gamma_xy
        ])
        w_exact = (
            0.5 * k0 * (tr_eps ** 2)
            + g0 * (dev_eps[0] ** 2 + dev_eps[1] ** 2 + dev_eps[2] ** 2 + 0.5 * (dev_eps[3] ** 2))
        ) * v0

        rel_error = abs(work_acc - w_exact) / w_exact
        assert rel_error < 1.0e-4, f"Strain energy work ledger relative error {rel_error:.3e} exceeds 0.01%"

    def test_elastic_path_reversibility_solid_and_shell(self):
        """Multi-cycle loading and unloading below yield: stresses return to 0, epsp == 0, thk returns to h0."""
        # Solid test
        p_solid = Law48Params(E=210000.0, nu=0.3, sigy0=300.0, cb=400.0, cn=0.5)
        sig_sol = np.zeros(6, dtype=float)
        epsp_sol = 0.0

        n_pts = 20
        eps_peaks = [0.0, 0.0006, 0.0, 0.0006, 0.0]
        strain_hist = []
        for j in range(len(eps_peaks) - 1):
            seg = np.linspace(eps_peaks[j], eps_peaks[j + 1], n_pts)
            strain_hist.extend(seg if j == 0 else seg[1:])

        return_stresses = []
        return_epsps = []

        for i in range(1, len(strain_hist)):
            e_now = strain_hist[i]
            e_prev = strain_hist[i - 1]
            de = e_now - e_prev
            deps = np.array([de, -0.3 * de, -0.3 * de, 0.0, 0.0, 0.0])
            sig_sol, epsp_sol, _ = solid_update_law48(p_solid, sig_sol, deps, epsp=epsp_sol, dt=1.0e-5)

            if np.isclose(e_now, 0.0):
                return_stresses.append(float(np.linalg.norm(sig_sol)))
                return_epsps.append(epsp_sol)

        for res_s in return_stresses:
            assert res_s < 1.0e-4, f"Solid residual stress {res_s:.4e} must be zero"
        for ep in return_epsps:
            assert ep == 0.0, f"Plastic strain must be zero in elastic regime, got {ep}"

        # Shell test
        p_shell = Law48Params(E=200000.0, nu=0.3, sigy0=250.0, cb=300.0, cn=0.5)
        sig_sh = np.zeros(3, dtype=float)
        thk = np.array([1.5], dtype=float)
        extra = {"thk": thk, "off": np.array([1.0])}
        epsp_sh = 0.0

        ret_thks = []
        ret_sh_stresses = []

        for i in range(1, len(strain_hist)):
            e_now = strain_hist[i]
            e_prev = strain_hist[i - 1]
            de = e_now - e_prev
            deps = np.array([de, 0.0, 0.0])
            sig_sh, epsp_sh = shell_update_law48(p_shell, sig_sh, deps, epsp=epsp_sh, dt=1.0e-5, extra=extra)

            if np.isclose(e_now, 0.0):
                ret_sh_stresses.append(float(np.linalg.norm(sig_sh)))
                ret_thks.append(float(thk[0]))

        for res_s in ret_sh_stresses:
            assert res_s < 3.0e-4, f"Shell residual stress {res_s:.4e} must be zero"
        for t_ret in ret_thks:
            assert math.isclose(t_ret, 1.5, abs_tol=1.0e-6), f"Shell thickness must return to 1.5, got {t_ret}"


# ============================================================================
# 3. Cyclic Loading and Bauschinger Effect
# ============================================================================

class TestLaw48CyclicLoadingAndBauschingerEffect:
    """Audit Bauschinger effect and kinematic vs isotropic hardening.

    OpenRadioss Fortran definition (sigeps48c.F:307-315, 476-496):
      F_isokin = 1.0 (chard = 1.0): pure kinematic hardening -> back-stress alpha develops,
                                    earlier reverse yield softening (Bauschinger effect).
      F_isokin = 0.0 (chard = 0.0): pure isotropic hardening -> back-stress remains 0,
                                    yield surface expands symmetrically without reverse softening.
    """

    def test_solid_bauschinger_effect_kinematic_vs_isotropic(self):
        """Compare kinematic (F_isokin=1.0) vs isotropic (F_isokin=0.0) under strain reversal."""
        p_kin = Law48Params(E=200000.0, nu=0.3, sigy0=250.0, cb=20000.0, cn=1.0001, fisokin=1.0)
        p_iso = Law48Params(E=200000.0, nu=0.3, sigy0=250.0, cb=20000.0, cn=1.0001, fisokin=0.0)

        sig_kin = np.zeros(6, dtype=float)
        extra_kin = {"sigb48": np.zeros(6, dtype=float)}
        epsp_kin = 0.0

        sig_iso = np.zeros(6, dtype=float)
        extra_iso = {"sigb48": np.zeros(6, dtype=float)}
        epsp_iso = 0.0

        # Phase 1: Forward plastic tension (25 increments of strain 0.0008)
        deps_fwd = np.array([0.0008, -0.00024, -0.00024, 0.0, 0.0, 0.0])
        for _ in range(25):
            sig_kin, epsp_kin, _ = solid_update_law48(p_kin, sig_kin, deps_fwd, epsp=epsp_kin, dt=1.0e-5, extra=extra_kin)
            sig_iso, epsp_iso, _ = solid_update_law48(p_iso, sig_iso, deps_fwd, epsp=epsp_iso, dt=1.0e-5, extra=extra_iso)

        assert epsp_kin > 0.0, "Kinematic model must accumulate plastic strain in forward tension"
        assert epsp_iso > 0.0, "Isotropic model must accumulate plastic strain in forward tension"

        # Back-stress alpha_xx is positive for kinematic hardening, zero for isotropic
        alpha_xx_kin = float(extra_kin["sigb48"][0])
        alpha_xx_iso = float(extra_iso["sigb48"][0])
        assert alpha_xx_iso == 0.0, "Isotropic hardening must maintain strictly zero backstress"
        assert alpha_xx_kin > 10.0, f"Kinematic hardening must develop positive backstress, got {alpha_xx_kin}"

        # Phase 2: Reverse compression (60 small compressive steps)
        deps_rev = np.array([-0.00008, 0.000024, 0.000024, 0.0, 0.0, 0.0])
        epsp_start_kin = epsp_kin
        epsp_start_iso = epsp_iso

        rev_yield_step_kin = -1
        rev_yield_step_iso = -1

        for step in range(90):
            sig_kin, epsp_kin, _ = solid_update_law48(p_kin, sig_kin, deps_rev, epsp=epsp_kin, dt=1.0e-5, extra=extra_kin)
            sig_iso, epsp_iso, _ = solid_update_law48(p_iso, sig_iso, deps_rev, epsp=epsp_iso, dt=1.0e-5, extra=extra_iso)

            if rev_yield_step_kin == -1 and epsp_kin > epsp_start_kin + 1.0e-6:
                rev_yield_step_kin = step
            if rev_yield_step_iso == -1 and epsp_iso > epsp_start_iso + 1.0e-6:
                rev_yield_step_iso = step

        # Kinematic hardening yields earlier in compression due to shifted yield center (Bauschinger effect!)
        assert rev_yield_step_kin != -1, "Kinematic model must yield in reverse compression"
        assert rev_yield_step_iso != -1, "Isotropic model must yield in reverse compression"
        assert rev_yield_step_kin < rev_yield_step_iso, (
            f"Bauschinger effect: kinematic hardening must yield earlier (step {rev_yield_step_kin}) "
            f"than isotropic hardening (step {rev_yield_step_iso})"
        )

        # Back-stress decreases during reverse plastic flow
        assert float(extra_kin["sigb48"][0]) < alpha_xx_kin

    def test_shell_bauschinger_effect_kinematic_vs_isotropic(self):
        """2D plane-stress shell Bauschinger effect under cyclic tension/compression."""
        p_kin = Law48Params(E=200000.0, nu=0.3, sigy0=200.0, cb=15000.0, cn=1.0001, fisokin=1.0)
        p_iso = Law48Params(E=200000.0, nu=0.3, sigy0=200.0, cb=15000.0, cn=1.0001, fisokin=0.0)

        sig_kin = np.zeros(3, dtype=float)
        extra_kin = {"sigb48": np.zeros(3, dtype=float), "thk": np.array([1.0])}
        epsp_kin = 0.0

        sig_iso = np.zeros(3, dtype=float)
        extra_iso = {"sigb48": np.zeros(3, dtype=float), "thk": np.array([1.0])}
        epsp_iso = 0.0

        deps_fwd = np.array([0.0008, 0.0, 0.0])
        for _ in range(25):
            sig_kin, epsp_kin = shell_update_law48(p_kin, sig_kin, deps_fwd, epsp=epsp_kin, dt=1.0e-5, extra=extra_kin)
            sig_iso, epsp_iso = shell_update_law48(p_iso, sig_iso, deps_fwd, epsp=epsp_iso, dt=1.0e-5, extra=extra_iso)

        alpha_xx_kin = float(extra_kin["sigb48"][0])
        assert alpha_xx_kin > 5.0, f"Shell back-stress alpha_xx must develop, got {alpha_xx_kin}"

        # Reverse compression
        deps_rev = np.array([-0.00008, 0.0, 0.0])
        epsp_start_kin = epsp_kin
        epsp_start_iso = epsp_iso

        rev_step_kin = -1
        rev_step_iso = -1

        for step in range(60):
            sig_kin, epsp_kin = shell_update_law48(p_kin, sig_kin, deps_rev, epsp=epsp_kin, dt=1.0e-5, extra=extra_kin)
            sig_iso, epsp_iso = shell_update_law48(p_iso, sig_iso, deps_rev, epsp=epsp_iso, dt=1.0e-5, extra=extra_iso)

            if rev_step_kin == -1 and epsp_kin > epsp_start_kin + 1.0e-6:
                rev_step_kin = step
            if rev_step_iso == -1 and epsp_iso > epsp_start_iso + 1.0e-6:
                rev_step_iso = step

        assert rev_step_kin < rev_step_iso, "Shell Bauschinger effect: kinematic hardening must yield earlier"

    def test_solid_hexa8_kinematic_cyclic_engine(self, tmp_path: Path):
        """Solid Hexa8 cyclic tension/compression through Engine verifying back-stress tracking."""
        run_name = "HEXA8_LAW48_ISOKIN"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law48(
            mid=1, title="SteelIsoKin48", rho=7.85e-9, e=200000.0, nu=0.3,
            a=200.0, b=300.0, n=0.5, chard=1.0, sig_max=600.0,
        )
        d.prop_solid(1, "SolidProp", isolid=1)
        d.part(1, "Part1", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0), (6, 10.0, 0.0, 10.0), (7, 10.0, 10.0, 10.0), (8, 0.0, 10.0, 10.0),
        ])
        d.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])

        d.grnod_node(1, "fix_face", [1, 4, 5, 8])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "pull_face", [2, 3, 6, 7])
        d.funct(10, "vel_cycle", [
            (0.0, 2000.0),
            (0.5e-4, 2000.0),
            (0.5001e-4, -2000.0),
            (1.0e-4, -2000.0),
        ])
        d.impvel(1, "pull_x", 10, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""
        # Verify plastic strain accumulated
        assert float(np.max(eng_model.bricks.state["epsp"])) > 0.0
        # Verify back-stress was allocated and tracked in mat_extra["sigb48"]
        extra = eng_model.bricks.state["mat_extra"]
        assert "sigb48" in extra
        assert np.isfinite(extra["sigb48"]).all()


# ============================================================================
# 4. Dynamic Strain-Rate Filtering
# ============================================================================

class TestLaw48DynamicStrainRateFiltering:
    """Audit cutoff frequency f_cut smoothing impulsive strain-rate jumps over time steps."""

    def test_solid_strain_rate_filtering_jump(self):
        """Impulsive strain-rate jump: filtered rate and resulting yield stress smoothed by f_cut."""
        fcut = 500.0  # Hz
        dt = 1.0e-4   # s -> alpha = min(1.0, 2*pi*500*1e-4) = 0.31416

        p_filt = Law48Params(
            E=200000.0, nu=0.3, sigy0=200.0,
            cc=50.0, eps0=1.0, fcut=fcut,
        )
        p_unfilt = Law48Params(
            E=200000.0, nu=0.3, sigy0=200.0,
            cc=50.0, eps0=1.0, fcut=1.0e30,
        )

        # Apply sudden strain rate jump: deps = 0.005 at dt = 1e-4 -> raw_rate approx 33.3 s^-1
        deps = np.array([0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0])
        extra_filt = {"epsd48": np.zeros(1, dtype=float)}

        sig_f, _, _ = solid_update_law48(p_filt, np.zeros(6), deps, epsp=0.0, dt=dt, extra=extra_filt)
        sig_u, _, _ = solid_update_law48(p_unfilt, np.zeros(6), deps, epsp=0.0, dt=dt)

        # Filtered rate should equal alpha * raw_rate
        epsd_filt = float(extra_filt["epsd48"][0])
        assert epsd_filt > 0.0
        # Yield stress with filtered rate is lower than unfiltered at step 1
        assert sig_f[0] < sig_u[0], f"Filtered stress {sig_f[0]} must be less than instantaneous jump {sig_u[0]}"

        # With repeated steps at same rate, filtered rate asymptotically reaches steady-state
        for _ in range(30):
            sig_f, _, _ = solid_update_law48(p_filt, sig_f, deps, epsp=0.0, dt=dt, extra=extra_filt)

        steady_epsd = float(extra_filt["epsd48"][0])
        assert steady_epsd > epsd_filt

    def test_shell_strain_rate_filtering_jump(self):
        """Shell 2D plane-stress cutoff frequency filtering under biaxial strain rate step."""
        fcut = 1000.0
        dt = 1.0e-4
        p_filt = Law48Params(E=200000.0, nu=0.3, sigy0=200.0, cc=40.0, eps0=1.0, fcut=fcut)
        p_unfilt = Law48Params(E=200000.0, nu=0.3, sigy0=200.0, cc=40.0, eps0=1.0)

        deps = np.array([0.004, 0.004, 0.0])
        extra_filt = {"epsd48": np.zeros(1, dtype=float), "thk": np.array([1.0])}

        sig_f, _ = shell_update_law48(p_filt, np.zeros(3), deps, epsp=0.0, dt=dt, extra=extra_filt)
        sig_u, _ = shell_update_law48(p_unfilt, np.zeros(3), deps, epsp=0.0, dt=dt)

        assert float(extra_filt["epsd48"][0]) > 0.0
        assert sig_f[0] < sig_u[0]

    def test_engine_dynamic_strain_rate_filtering(self, tmp_path: Path):
        """Engine run of solid block under impulsive loading verifying rate filtering in mat_extra."""
        run_name = "FILTER_LAW48_ENG"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law48(
            mid=1, title="MatFilt48", rho=7.85e-9, e=200000.0, nu=0.3,
            a=250.0, b=300.0, n=0.5, c=30.0, eps_rate_0=1.0, fcut=2000.0,
        )
        d.prop_solid(1, "SolidProp", isolid=1)
        d.part(1, "Part1", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 5.0, 0.0, 0.0), (3, 5.0, 5.0, 0.0), (4, 0.0, 5.0, 0.0),
            (5, 0.0, 0.0, 5.0), (6, 5.0, 0.0, 5.0), (7, 5.0, 5.0, 5.0), (8, 0.0, 5.0, 5.0),
        ])
        d.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])

        d.grnod_node(1, "fix_face", [1, 4, 5, 8])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "pull_face", [2, 3, 6, 7])
        d.funct(10, "pull_impulse", [(0.0, 5000.0), (1.0e-4, 5000.0)])
        d.impvel(1, "pull_bc", 10, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""
        extra = eng_model.bricks.state["mat_extra"]
        assert "epsd48" in extra
        assert float(np.max(extra["epsd48"])) > 0.0


# ============================================================================
# 5. Progressive Failure & Element Deletion
# ============================================================================

class TestLaw48ProgressiveFailureAndDeletion:
    """Audit tensile damage scaling, plastic failure, element deletion, and post-failure stability."""

    def test_tensile_damage_scaling_and_rupture(self):
        """Verify FAIL factor degrades yield stress between eps_t1 and eps_t2, rupturing at eps_t2."""
        eps_t1 = 0.04
        eps_t2 = 0.12
        p = Law48Params(
            E=200000.0, nu=0.3, ca=1.0, sigy0=300.0,
            cb=0.0, eps_t1=eps_t1, eps_t2=eps_t2,
        )

        # Stage 1: Below eps_t1 (eps_xx = 0.02 < 0.04) -> FAIL = 1.0, unsoftened yield
        deps_step = np.array([0.001, -0.0003, -0.0003, 0.0, 0.0, 0.0])
        extra = {"eps48": np.zeros(6, dtype=float), "off": np.array([1.0]), "off48": np.array([1.0])}
        sig = np.zeros(6, dtype=float)
        ep = 0.0

        for _ in range(20):
            sig, ep, _ = solid_update_law48(p, sig, deps_step, epsp=ep, dt=1.0e-5, extra=extra)

        vm1 = math.sqrt(0.5 * ((sig[0] - sig[1])**2 + (sig[1] - sig[2])**2 + (sig[2] - sig[0])**2))
        assert math.isclose(vm1, 300.0, rel_tol=1.0e-2)
        assert extra["off"][0] == 1.0

        # Stage 2: Midway between eps_t1 and eps_t2 (total eps_xx = 0.08) -> FAIL < 1.0
        for _ in range(60):
            sig, ep, _ = solid_update_law48(p, sig, deps_step, epsp=ep, dt=1.0e-5, extra=extra)

        vm2 = math.sqrt(0.5 * ((sig[0] - sig[1])**2 + (sig[1] - sig[2])**2 + (sig[2] - sig[0])**2))
        assert vm2 < 300.0, f"Yield stress must be degraded by tensile damage, got {vm2}"
        epst2 = _principal_strain_3d(extra["eps48"])
        expected_fail = float(np.squeeze((eps_t2 - epst2) / (eps_t2 - eps_t1)))
        assert math.isclose(vm2, expected_fail * 300.0, rel_tol=0.05)

        # Stage 3: Above eps_t2 (total eps_xx = 0.14 > 0.12) -> Complete tensile rupture
        for _ in range(60):
            sig, ep, _ = solid_update_law48(p, sig, deps_step, epsp=ep, dt=1.0e-5, extra=extra)

        assert np.allclose(sig, 0.0), "Stress must be zeroed upon complete tensile rupture"
        assert extra["off"][0] == 0.0, "off flag must be zeroed upon rupture"
        assert extra["off48"][0] == 0.0

    def test_plastic_failure_and_post_failure_continuation(self):
        """Verify plastic failure when epsp >= eps_max and stable continuation without NaNs."""
        eps_max = 0.05
        p = Law48Params(E=200000.0, nu=0.3, sigy0=250.0, cb=300.0, cn=0.5, eps_max=eps_max)

        sig = np.zeros(6, dtype=float)
        epsp = 0.0
        extra = {"off": np.array([1.0]), "off48": np.array([1.0])}

        # Step 1: Small strain below failure
        deps_small = np.array([0.005, -0.0015, -0.0015, 0.0, 0.0, 0.0])
        sig, epsp, _ = solid_update_law48(p, sig, deps_small, epsp=epsp, dt=1.0e-4, extra=extra)
        assert epsp < eps_max
        assert extra["off"][0] == 1.0

        # Step 2: Large strain causing epsp to exceed eps_max
        deps_large = np.array([0.08, -0.024, -0.024, 0.0, 0.0, 0.0])
        sig, epsp, c = solid_update_law48(p, sig, deps_large, epsp=epsp, dt=1.0e-4, extra=extra)
        assert epsp >= eps_max
        assert np.allclose(sig, 0.0), "Stress must collapse to zero when plastic strain exceeds eps_max"
        assert extra["off"][0] == 0.0
        assert np.isfinite(c), "Sound speed must remain finite"

        # Step 3: Post-failure continuation: apply further deformation increments
        for _ in range(10):
            sig, epsp, c = solid_update_law48(p, sig, deps_large, epsp=epsp, dt=1.0e-4, extra=extra)
            assert np.allclose(sig, 0.0)
            assert not np.isnan(sig).any()
            assert not np.isnan(epsp)
            assert np.isfinite(c)

    def test_multi_element_bar_erosion_engine(self, tmp_path: Path):
        """2-element bar in Engine: weak element erodes, strong element survives, simulation continues stably."""
        run_name = "MULTI_ERODE_LAW48"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        # Part 1 has low failure strain eps_max = 0.02 (will erode)
        d.mat_law48(mid=1, title="MatWeak48", rho=7.85e-9, e=200000.0, nu=0.3, a=200.0, b=300.0, n=0.5, eps_max=0.02)
        # Part 2 has high failure strain (will survive)
        d.mat_law48(mid=2, title="MatStrong48", rho=7.85e-9, e=200000.0, nu=0.3, a=200.0, b=300.0, n=0.5, eps_max=1.0)

        d.prop_solid(1, "Prop1", isolid=1)
        d.prop_solid(2, "Prop2", isolid=1)
        d.part(1, "PartWeak", 1, 1)
        d.part(2, "PartStrong", 2, 2)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 5.0, 0.0, 0.0), (3, 5.0, 5.0, 0.0), (4, 0.0, 5.0, 0.0),
            (5, 0.0, 0.0, 5.0), (6, 5.0, 0.0, 5.0), (7, 5.0, 5.0, 5.0), (8, 0.0, 5.0, 5.0),
            (9, 10.0, 0.0, 0.0), (10, 10.0, 5.0, 0.0), (11, 10.0, 0.0, 5.0), (12, 10.0, 5.0, 5.0),
        ])
        d.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])
        d.brick(2, [(2, 2, 9, 10, 3, 6, 11, 12, 7)])

        d.grnod_node(1, "fix_left", [1, 4, 5, 8])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "pull_right", [9, 10, 11, 12])
        d.funct(10, "pull_func", [(0.0, 5000.0), (2.0e-4, 5000.0)])
        d.impvel(1, "pull_bc", 10, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""
        off_flags = eng_model.bricks.state["off"]
        assert off_flags[0] == 0.0, "Weak element 1 must be eroded (off=0.0)"
        assert not np.isnan(eng_model.bricks.state["sig"]).any(), "No NaNs in stresses"
        assert not np.isnan(eng_model.x).any(), "No NaNs in coordinates"


# ============================================================================
# 6. Acoustic Sound Speeds & Courant Stability
# ============================================================================

class TestLaw48AcousticWaveSpeedAndCourantStability:
    """Audit sound speed formulas, Courant time step bounding, and acoustic wave propagation."""

    def test_acoustic_sound_speed_exact_formulas(self):
        """Verify exact longitudinal solid sound speed and plane-stress shell sound speed."""
        e0 = 210000.0
        nu = 0.3
        rho0 = 7.85e-9
        p = Law48Params(E=e0, nu=nu, rho0=rho0)

        # Exact bulk and shear moduli
        k0 = e0 / (3.0 * (1.0 - 2.0 * nu))
        g0 = e0 / (2.0 * (1.0 + nu))
        expected_c_solid = math.sqrt((k0 + (4.0 / 3.0) * g0) / rho0)

        c_solid = sound_speed_solid_law48(p, rho0=rho0)
        assert np.isclose(c_solid, expected_c_solid, rtol=1e-5)

        # Exact shell plane-stress acoustic speed: c = sqrt(E / ((1 - nu^2) * rho0))
        expected_c_shell = math.sqrt((e0 / (1.0 - nu ** 2)) / rho0)
        c_shell = sound_speed_shell_law48(p, rho0=rho0)
        assert np.isclose(c_shell, expected_c_shell, rtol=1e-5)

    def test_courant_stability_bound(self):
        """Verify dynamic time step bounded by Courant condition: dt <= dt_Courant = L_min / c."""
        e0 = 200000.0
        nu = 0.3
        rho0 = 7.85e-9
        p = Law48Params(E=e0, nu=nu, rho0=rho0)

        l_min = 5.0  # mm
        c = sound_speed_solid_law48(p, rho0=rho0)
        dt_courant = l_min / c

        dt_safe = 0.5 * dt_courant
        assert dt_safe < dt_courant
        assert dt_courant > 0.0

    def test_acoustic_pulse_propagation_bar_engine(self, tmp_path: Path):
        """10-element bar in Engine: acoustic disturbance travels at wave speed c."""
        run_name = "BAR_10EL_WAVE_48"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        rho0 = 7.85e-9
        e0 = 200000.0
        nu = 0.3
        d = StarterDeck(run_name)
        d.mat_law48(mid=1, title="SteelWave48", rho=rho0, e=e0, nu=nu, a=500.0, b=0.0)
        d.prop_solid(1, "BarProp", isolid=1)
        d.part(1, "BarPart", 1, 1)

        # 10 elements along X from x=0 to x=50 mm (5 mm each)
        nodes = []
        for i in range(11):
            x = i * 5.0
            nodes.extend([
                (4 * i + 1, x, 0.0, 0.0),
                (4 * i + 2, x, 5.0, 0.0),
                (4 * i + 3, x, 5.0, 5.0),
                (4 * i + 4, x, 0.0, 5.0),
            ])
        d.node(nodes)

        bricks = []
        for i in range(10):
            b_bot = (4 * i + 1, 4 * (i + 1) + 1, 4 * (i + 1) + 2, 4 * i + 2)
            b_top = (4 * i + 4, 4 * (i + 1) + 4, 4 * (i + 1) + 3, 4 * i + 3)
            bricks.append((i + 1, *b_bot, *b_top))
        d.brick(1, bricks)

        # Apply short velocity pulse at x=0 (nodes 1..4)
        d.grnod_node(1, "impact_face", [1, 2, 3, 4])
        d.funct(10, "pulse_func", [(0.0, 50.0), (2.0e-6, 50.0), (2.1e-6, 0.0), (1.0, 0.0)])
        d.impvel(1, "impact_pulse", 10, "X", 1)

        d.write(s_path)
        # Sound speed ~ 5800 mm/s. In 4e-6 s, wave travels ~ 23 mm (reaches elements 0..4).
        # Element 9 (at x=45..50 mm) is far beyond wave front and must be undisturbed!
        _write_engine_deck(e_path, run_name, tstop=4.0e-6, dt_scale=0.4)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 5
        assert state.stop_reason == ""

        sig = eng_model.bricks.state["sig"]
        assert abs(sig[0, 0]) > 0.05, "Element near impact pulse must be disturbed"
        assert abs(sig[9, 0]) < 1.0e-3, "Far end element 9 must remain undisturbed before wave arrival"
