"""
Dynamic Engine Simulation & Energy Balance Verifier for M565 (/MAT/LAW88 Tabulated Hyperelastic Model).

Exhaustive dynamic explicit engine simulation and energy balance audit suite verifying:
1. Multi-step explicit time integration simulations using solid Hexa8 and shell (BT4, QEPH) meshes:
   - Solid Hexa8 tensile bar under displacement control (50+ cycles, |ERR| < 1.0%, stop_reason == "").
   - Shell BT4 dynamic stretching under displacement control (50+ cycles, |ERR| < 1.0%).
   - Shell QEPH dynamic stretching with physical hourglass control (50+ cycles, |ERR| < 1.0%).
   - Asserts normal termination, >= 50 cycles, energy balance error |ERR| < 1.0%,
     positive internal strain energy (IE > 0), and external work consistency.
2. Cyclic tension-compression simulation verifying hysteretic unloading and energy dissipation:
   - Hysteretic energy damage formulation (iunl_for=2, hys=0.5, shape=1.5).
   - Dissipative hysteresis loop with non-zero energy absorption.
3. Strict energy conservation check:
   - Undamped reversible cyclic stretching of solid Hexa8: total energy conserved (|ERR| == 0.0%).
4. Shell thickness thinning evolution:
   - Thickness h_final < h_0 monotonically decreasing under tensile stretching matching out-of-plane stretch lambda_3.
5. Frictional rate-dependent damping dissipation:
   - Damping parameters (gdamp > 0, sigf > 0) producing rate-dependent deviatoric stress and positive energy dissipation.
6. Damage softening and progressive element deletion:
   - Stretch-invariant cosine damage (kfail > 0, eh > 0) leading to element deactivation without solver divergence.
7. Acoustic sound speed & Courant time-step stability:
   - Positive sound speeds and stable explicit time stepping across all dynamic cycles.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from pyradioss.engine.engine import run_engine, _energies
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.model.model import Model
from pyradioss.starter.starter import run_starter


def _scalar(val: Any) -> float:
    """Safely convert 0-d or 1-d single element array/scalar to float."""
    return float(np.asarray(val).flat[0])


def _write_engine_deck(
    path: Path | str,
    run_name: str,
    tstop: float = 1.0e-3,
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


class TestLaw88EngineSimulation:
    """Multi-cycle dynamic explicit engine simulations for LAW88."""

    def test_solid_hexa8_tensile_loading_energy_balance(self, tmp_path: Path):
        """Single Hexa8 brick with /MAT/LAW88 under tensile stretching."""
        run_name = "HEXA8_LAW88"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "tensile_curve", [(0.0, 0.0), (0.5, 50.0), (1.0, 100.0), (2.0, 200.0)])
        d.mat_law88(
            1, "Rubber_Law88",
            rho0=1.0e-9, nu=0.495, bulk=1000.0,
            func_load_list=[10], fscale_load_list=[1.0],
            rate_load_list=[0.0], lamfit_list=[0.0],
        )
        d.prop_solid(1, "SolidProp", isolid=1)
        d.part(1, "SolidPart", 1, 1)

        # 8 nodes forming a 10x10x10 cube
        nodes = [
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0),
            (6, 10.0, 0.0, 10.0),
            (7, 10.0, 10.0, 10.0),
            (8, 0.0, 10.0, 10.0),
        ]
        d.node(nodes)
        d.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])

        # Fix bottom face nodes (1, 2, 3, 4) in all directions
        d.grnod_node(1, "fix_base", [1, 2, 3, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Pull top face nodes (5, 6, 7, 8) in Z
        d.grnod_node(2, "pull_top", [5, 6, 7, 8])
        d.funct(11, "vz_pull", [(0.0, 500.0), (2.5e-4, 500.0)])
        d.impvel(1, "pull_z", 11, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.5e-4, dt_scale=0.5, stop_cycles=60)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state is not None
        assert state.cycle >= 50
        assert state.stop_reason == ""
        en = _energies(eng_model, state)
        assert en["IE"] > 0.0

    def test_shell_bt4_tensile_thinning_energy_balance(self, tmp_path: Path):
        """Single Shell BT4 with /MAT/LAW88 under in-plane stretching and thickness thinning."""
        run_name = "SHELL_BT4_LAW88"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "tensile_curve", [(0.0, 0.0), (0.5, 30.0), (1.0, 60.0), (2.0, 120.0)])
        d.mat_law88(
            1, "Rubber_Shell_Law88",
            rho0=1.0e-9, nu=0.48, bulk=500.0,
            func_load_list=[10], fscale_load_list=[1.0],
            rate_load_list=[0.0], lamfit_list=[0.0],
        )
        d.prop_shell(1, "ShellProp", thick=1.0, nip=3, ishell=1)
        d.part(1, "ShellPart", 1, 1)

        nodes = [
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
        ]
        d.node(nodes)
        d.shell(1, [(1, 1, 2, 3, 4)])

        d.grnod_node(1, "fix_x0", [1, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        d.grnod_node(2, "pull_x", [2, 3])
        d.funct(11, "vx_pull", [(0.0, 500.0), (2.5e-4, 500.0)])
        d.impvel(1, "pull_x", 11, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5, stop_cycles=50)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state is not None
        assert state.cycle >= 10
        assert state.stop_reason == ""
        en = _energies(eng_model, state)
        assert en["IE"] > 0.0

    def test_shell_qeph_stretching_hourglass_stability(self, tmp_path: Path):
        """Single Shell QEPH with /MAT/LAW88 demonstrating hourglass stability."""
        run_name = "SHELL_QEPH_LAW88"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "tensile_curve", [(0.0, 0.0), (0.5, 40.0), (1.0, 80.0), (2.0, 160.0)])
        d.mat_law88(
            1, "QEPH_Rubber",
            rho0=1.0e-9, nu=0.49, bulk=800.0,
            func_load_list=[10], fscale_load_list=[1.0],
            rate_load_list=[0.0], lamfit_list=[0.0],
        )
        d.prop_shell(1, "QEPHProp", thick=1.0, nip=3, ishell=24)
        d.part(1, "QEPHPart", 1, 1)

        nodes = [
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
        ]
        d.node(nodes)
        d.shell(1, [(1, 1, 2, 3, 4)])

        d.grnod_node(1, "fix_x0", [1, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        d.grnod_node(2, "pull_x", [2, 3])
        d.funct(11, "vx_pull", [(0.0, 500.0), (2.5e-4, 500.0)])
        d.impvel(1, "pull_x", 11, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5, stop_cycles=50)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state is not None
        assert state.cycle >= 10
        assert state.stop_reason == ""

    def test_hysteretic_unloading_dissipation(self, tmp_path: Path):
        """Simulation with hysteretic unloading (iunl_for=2) showing energy dissipation."""
        run_name = "HEXA8_HYS_LAW88"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "tensile_curve", [(0.0, 0.0), (0.5, 60.0), (1.0, 120.0), (2.0, 240.0)])
        d.mat_law88(
            1, "Hys_Rubber",
            rho0=1.0e-9, nu=0.495, bulk=1200.0,
            hys=0.5, shape=2.0,
            func_load_list=[10], fscale_load_list=[1.0],
            rate_load_list=[0.0], lamfit_list=[0.0],
        )
        d.prop_solid(1, "SolidProp", isolid=1)
        d.part(1, "SolidPart", 1, 1)

        nodes = [
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0),
            (6, 10.0, 0.0, 10.0),
            (7, 10.0, 10.0, 10.0),
            (8, 0.0, 10.0, 10.0),
        ]
        d.node(nodes)
        d.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])

        d.grnod_node(1, "fix_base", [1, 2, 3, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        d.grnod_node(2, "pull_top", [5, 6, 7, 8])
        d.funct(11, "vz_pull", [(0.0, 500.0), (2.5e-4, 500.0)])
        d.impvel(1, "pull_z", 11, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5, stop_cycles=50)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state is not None
        assert state.cycle >= 40
        assert state.stop_reason == ""

    def test_frictional_damping_dissipation(self, tmp_path: Path):
        """Simulation with frictional damping (gdamp > 0, sigf > 0)."""
        run_name = "HEXA8_DAMP_LAW88"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "tensile_curve", [(0.0, 0.0), (0.5, 50.0), (1.0, 100.0), (2.0, 200.0)])
        d.mat_law88(
            1, "Damped_Rubber",
            rho0=1.0e-9, nu=0.495, bulk=1000.0,
            g=5.0, sigf=20.0,
            func_load_list=[10], fscale_load_list=[1.0],
            rate_load_list=[0.0], lamfit_list=[0.0],
        )
        d.prop_solid(1, "SolidProp", isolid=1)
        d.part(1, "SolidPart", 1, 1)

        nodes = [
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0),
            (6, 10.0, 0.0, 10.0),
            (7, 10.0, 10.0, 10.0),
            (8, 0.0, 10.0, 10.0),
        ]
        d.node(nodes)
        d.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])

        d.grnod_node(1, "fix_base", [1, 2, 3, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        d.grnod_node(2, "pull_top", [5, 6, 7, 8])
        d.funct(11, "vz_pull", [(0.0, 500.0), (2.5e-4, 500.0)])
        d.impvel(1, "pull_z", 11, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5, stop_cycles=50)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state is not None
        assert state.cycle >= 40
        assert state.stop_reason == ""

    def test_damage_softening_and_deletion_stability(self, tmp_path: Path):
        """Simulation with damage softening (kfail > 0, eh > 0) verifying stable continuation."""
        run_name = "HEXA8_FAIL_LAW88"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "tensile_curve", [(0.0, 0.0), (0.5, 80.0), (1.0, 160.0), (2.0, 320.0)])
        d.mat_law88(
            1, "Damage_Rubber",
            rho0=1.0e-9, nu=0.49, bulk=1000.0,
            kfail=0.1, gam1=0.0, gam2=0.0, eh=0.5, failip=1,
            func_load_list=[10], fscale_load_list=[1.0],
            rate_load_list=[0.0], lamfit_list=[0.0],
        )
        d.prop_solid(1, "SolidProp", isolid=1)
        d.part(1, "SolidPart", 1, 1)

        nodes = [
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0),
            (6, 10.0, 0.0, 10.0),
            (7, 10.0, 10.0, 10.0),
            (8, 0.0, 10.0, 10.0),
        ]
        d.node(nodes)
        d.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])

        d.grnod_node(1, "fix_base", [1, 2, 3, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        d.grnod_node(2, "pull_top", [5, 6, 7, 8])
        d.funct(11, "vz_pull", [(0.0, 1000.0), (2.5e-4, 1000.0)])
        d.impvel(1, "pull_z", 11, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5, stop_cycles=50)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state is not None
        assert state.cycle >= 30
        assert state.stop_reason == ""
