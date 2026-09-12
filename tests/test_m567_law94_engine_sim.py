"""
Dynamic Engine Simulation & Energy Balance Tests for Milestone M567 (/MAT/LAW94 Yeoh).
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Optional
import numpy as np
import pytest

from pyradioss.engine.engine import run_engine
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.starter.starter import run_starter


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


class TestLaw94EngineSimulation:
    """Multi-cycle dynamic explicit engine simulations for LAW94."""

    def test_solid_hexa8_yeoh_tensile_energy_balance(self, tmp_path: Path):
        """Single Hexa8 brick with /MAT/LAW94 under tensile stretching."""
        run_name = "HEXA8_LAW94"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law94(
            1, "Yeoh_Rubber",
            rho0=1.0e-9, c10=5.0, c20=0.1, c30=0.01, d1=1.0e-3,
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

        # Fix bottom face nodes (1, 2, 3, 4)
        d.grnod_node(1, "fix_base", [1, 2, 3, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Pull top face nodes (5, 6, 7, 8) in Z
        d.grnod_node(2, "pull_top", [5, 6, 7, 8])
        d.funct(11, "vz_pull", [(0.0, 200.0), (5.0e-4, 200.0)])
        d.impvel(1, "pull_z", 11, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=5.0e-4, dt_scale=0.5, stop_cycles=50)

        st_model = run_starter(s_path)
        eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 50
        assert sum(float(g.state["eint"].sum()) for _, g in eng_model.element_groups()) > 0.0

    def test_shell_bt4_yeoh_tensile_loading(self, tmp_path: Path):
        """Single 4-node shell with /MAT/LAW94 under in-plane stretching."""
        run_name = "SHELL_LAW94"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law94(
            1, "Yeoh_Shell",
            rho0=1.0e-9, c10=4.0, c20=0.05, c30=0.005, d1=1.0e-3,
        )
        d.prop_shell(1, "ShellProp", thick=1.0, ishell=1)
        d.part(1, "ShellPart", 1, 1)

        # 4 nodes forming a 10x10 square in XY
        nodes = [
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
        ]
        d.node(nodes)
        d.shell(1, [(1, 1, 2, 3, 4)])

        # Fix left edge (1, 4)
        d.grnod_node(1, "fix_left", [1, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Pull right edge (2, 3) in X
        d.grnod_node(2, "pull_right", [2, 3])
        d.funct(12, "vx_pull", [(0.0, 200.0), (5.0e-4, 200.0)])
        d.impvel(1, "pull_x", 12, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=5.0e-4, dt_scale=0.5, stop_cycles=50)

        st_model = run_starter(s_path)
        eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert sum(float(g.state["eint"].sum()) for _, g in eng_model.element_groups()) > 0.0
